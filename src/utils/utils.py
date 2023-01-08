from typing import Optional, Dict
import logging
from os.path import join

import gin
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torch.utils.data import DataLoader
@gin.configurable
class Checkpoint:
    def __init__(self,
                 checkpoint_dir: str,
                 patience: Optional[int] = 7,
                 delta: Optional[float] = 0.):
        self.checkpoint_dir = checkpoint_dir
        self.model_path = join(checkpoint_dir, 'model.pth')

        # early stopping
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.delta = delta

        # logging
        self.summary_writer = SummaryWriter(log_dir=checkpoint_dir)

    def __call__(self,
                 epoch: int,
                 model: torch.nn.Module,
                 scalars: Optional[Dict[str, float]] = None):
        for name, value in scalars.items():
            # logging
            self.summary_writer.add_scalar(name, value, epoch)

            # early stopping
            if name == 'Loss/Val':
                val_loss = value
                if val_loss <= self.best_loss + self.delta:
                    logging.info(
                        f"Validation loss decreased ({self.best_loss:.3f} --> {val_loss:.3f}). Saving model ...")
                    torch.save(model.state_dict(), self.model_path)
                    self.best_loss = val_loss
                    self.counter = 0
                else:
                    self.counter += 1
                    logging.info(f"Validation loss increased ({self.best_loss:.3f} --> {val_loss:.3f}). "
                                 f"Early stopping counter: {self.counter} out of {self.patience}")
                    if self.counter >= self.patience >= 0:
                        self.early_stop = True

        self.summary_writer.flush()

    def close(self, scores: Optional[Dict[str, float]] = None):
        if scores is not None:
            for name, value in scores.items():
                self.summary_writer.add_scalar(name, value)
        self.summary_writer.close()


from typing import Optional, Callable
from functools import partial

import torch
import torch.nn.functional as F
from torch import Tensor


def get_loss_fn(loss_name: str,
                delta: Optional[float] = 1.0,
                beta: Optional[float] = 1.0) -> Callable:
    return {'mse': F.mse_loss,
            'mae': F.l1_loss,
            'huber': partial(F.huber_loss, delta=delta),
            'smooth_l1': partial(F.smooth_l1_loss, beta=beta)}[loss_name]

import numpy as np


def rse(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def corr(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def mae(pred, true):
    return np.mean(np.abs(pred - true))


def mse(pred, true):
    return np.mean((pred - true) ** 2)


def rmse(pred, true):
    return np.sqrt(mse(pred, true))


def mape(pred, true):
    return np.mean(np.abs((pred - true) / true))


def mspe(pred, true):
    return np.mean(np.square((pred - true) / true))


def calc_metrics(pred, true):
    return {'mae': mae(pred, true),
            'mse': mse(pred, true),
            'rmse': rmse(pred, true),
            'mape': mape(pred, true),
            'mspe': mspe(pred, true)}


from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from einops import reduce


def default_device() -> torch.device:
    """
    PyTorch default device is GPU when available, CPU otherwise.
    :return: Default device.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def to_tensor(array: np.ndarray, to_default_device: Optional[bool] = True) -> Tensor:
    """
    Convert numpy array to tensor on default device.
    :param array: Numpy array to convert.
    :param to_default_device Place tensor on default device or not.
    :return: PyTorch tensor, optionally on default device.
    """
    if to_default_device:
        return torch.as_tensor(array, dtype=torch.float32).to(default_device())


def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    mask = b == .0
    b[mask] = 1.
    result = a / b
    result[mask] = .0
    result[result != result] = .0
    result[result == np.inf] = .0
    return result


def scale(x: Tensor,
          scaling_factor: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
    if scaling_factor is not None:
        x = x / scaling_factor
        return x, scaling_factor

    scaling_factor = reduce(torch.abs(x).data, 'b t d -> b 1 d', 'mean')
    scaling_factor[scaling_factor == 0.0] = 1.0
    x = x / scaling_factor
    return x, scaling_factor


def descale(forecast: Tensor, scaling_factor: Tensor) -> Tensor:
    return forecast * scaling_factor


from abc import ABC, abstractmethod
from typing import Optional, List, Union

import numpy as np
import pandas as pd


class TimeFeature(ABC):
    """Abstract class for time features"""
    def __init__(self, normalise: bool, a: float, b: float):
        self.normalise = normalise
        self.a = a
        self.b = b

    @abstractmethod
    def __call__(self, idx: pd.DatetimeIndex) -> np.ndarray:
        ...

    @property
    @abstractmethod
    def _max_val(self) -> float:
        ...

    @property
    def max_val(self) -> float:
        return self._max_val if self.normalise else 1.0

    def scale(self, val: np.ndarray) -> np.ndarray:
        return val * (self.b - self.a) + self.a

    def process(self, val: np.ndarray) -> np.ndarray:
        features = self.scale(val / self.max_val)
        if self.normalise:
            return features
        return features.astype(int)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(normalise={self.normalise}, a={self.a}, b={self.b})"


class SecondOfMinute(TimeFeature):
    """Second of minute, unnormalised: [0, 59]"""
    def __call__(self, idx: pd.DatetimeIndex) -> np.ndarray:
        return self.process(idx.second)

    @property
    def _max_val(self):
        return 59.0


class MinuteOfHour(TimeFeature):
    """Minute of hour, unnormalised: [0, 59]"""
    def __call__(self, idx: pd.DatetimeIndex) -> np.ndarray:
        return self.process(idx.minute)

    @property
    def _max_val(self):
        return 59.0


class HourOfDay(TimeFeature):
    """Hour of day, unnormalised: [0, 23]"""
    def __call__(self, idx: pd.DatetimeIndex) -> np.ndarray:
        return self.process(idx.hour)

    @property
    def _max_val(self):
        return 23.0


class DayOfWeek(TimeFeature):
    """Hour of day, unnormalised: [0, 6]"""
    def __call__(self, idx: pd.DatetimeIndex) -> np.ndarray:
        return self.process(idx.dayofweek)

    @property
    def _max_val(self):
        return 6.0


class DayOfMonth(TimeFeature):
    """Day of month, unnormalised: [0, 30]"""
    def __call__(self, idx: pd.DatetimeIndex) -> np.ndarray:
        return self.process(idx.day - 1)

    @property
    def _max_val(self):
        return 30.0


class DayOfYear(TimeFeature):
    """Day of year, unnormalised: [0, 365]"""
    def __call__(self, idx: pd.DatetimeIndex) -> np.ndarray:
        return self.process(idx.dayofyear - 1)

    @property
    def _max_val(self):
        return 365.0


class WeekOfYear(TimeFeature):
    """Week of year, unnormalised: [0, 52]"""
    def __call__(self, idx: pd.DatetimeIndex) -> np.ndarray:
        return self.process(pd.Index(idx.isocalendar().week, dtype=int) - 1)

    @property
    def _max_val(self):
        return 52.0

class MonthOfYear(TimeFeature):
    """Month of year, unnormalised: [0, 11]"""
    def __call__(self, idx: pd.DatetimeIndex) -> np.ndarray:
        return self.process(idx.month - 1)

    @property
    def _max_val(self):
        return 11.0


class QuarterOfYear(TimeFeature):
    """Quarter of year, unnormalised: [0, 3]"""
    def __call__(self, idx: pd.DatetimeIndex) -> np.ndarray:
        return self.process(idx.quarter - 1)

    @property
    def _max_val(self):
        return 3.0


str_to_feat = {
    # dictionary mapping name to TimeFeature function
    'SecondOfMinute': SecondOfMinute,
    'MinuteOfHour': MinuteOfHour,
    'HourOfDay': HourOfDay,
    'DayOfWeek': DayOfWeek,
    'DayOfMonth': DayOfMonth,
    'DayOfYear': DayOfYear,
    'WeekOfYear': WeekOfYear,
    'MonthOfYear': MonthOfYear,
    'QuarterOfYear': QuarterOfYear,
}


freq_to_feats = {
    # dictionary mapping frequency to list of TimeFeature functions
    'q': [QuarterOfYear],
    'm': [QuarterOfYear, MonthOfYear],
    'w': [QuarterOfYear, MonthOfYear, WeekOfYear],
    'd': [QuarterOfYear, MonthOfYear, WeekOfYear, DayOfYear, DayOfMonth, DayOfWeek],
    'h': [QuarterOfYear, MonthOfYear, WeekOfYear, DayOfYear, DayOfMonth, DayOfWeek, HourOfDay],
    't': [QuarterOfYear, MonthOfYear, WeekOfYear, DayOfYear, DayOfMonth, DayOfWeek, HourOfDay, MinuteOfHour],
    's': [QuarterOfYear, MonthOfYear, WeekOfYear, DayOfYear, DayOfMonth, DayOfWeek, HourOfDay, MinuteOfHour, SecondOfMinute],
}


def get_time_features(dates: pd.DatetimeIndex, normalise: bool, a: Optional[float] = 0., b: Optional[float] = 1.,
                      features: Optional[Union[str, List[str]]] = None) -> np.ndarray:
    """
    Returns a numpy array of date/time features based on either frequency or directly specifying a list of features.
    :param dates: DatetimeIndex object of shape (time,)
    :param normalise: Whether to normalise feature between [a, b]. If not, return as an int in the original feature range.
    :param a: Lower bound of feature
    :param b: Upper bound of feature
    :param features: Frequency string used to obtain list of TimeFeatures, or directly a list of names of TimeFeatures
    :return: np array of date/time features of shape (time, n_feats)
    """
    if isinstance(features, list):
        assert all([feat in str_to_feat.keys() for feat in features]), \
            f"items in list should be one of {[*str_to_feat.keys()]}"
        features = [str_to_feat[feat] for feat in features]
    elif isinstance(features, str):
        assert features in freq_to_feats.keys(), \
            f"features should be one of {[*freq_to_feats.keys()]}"
        features = freq_to_feats[features]
    else:
        raise ValueError(f"features should be a list or str, not a {type(features)}")

    features = [feat(normalise, a, b)(dates) for feat in features]

    if len(features) == 0:
        return np.empty((dates.shape[0], 0))
    return np.stack(features, axis=1)


@torch.no_grad()
def validate(model: nn.Module,
             loader: DataLoader,
             loss_fn: Optional[Callable] = None,
             report_metrics: Optional[bool] = False,
             save_path: Optional[str] = None) -> Union[Dict[str, float], float]:
    model.eval()
    preds = []
    trues = []
    inps = []
    total_loss = []
    for it, data in enumerate(loader):
        x, y, x_time, y_time = map(to_tensor, data)

        if x.shape[0] == 1:
            # skip final batch if batch_size == 1
            # due to bug in torch.linalg.solve which raises error when batch_size == 1
            continue

        forecast = model(x, x_time, y_time)

        if report_metrics:
            preds.append(forecast)
            trues.append(y)
            if save_path is not None:
                inps.append(x)
        else:
            loss = loss_fn(forecast, y, reduction='none')
            total_loss.append(loss)

    if report_metrics:
        preds = torch.cat(preds, dim=0).detach().cpu().numpy()
        trues = torch.cat(trues, dim=0).detach().cpu().numpy()
        if save_path is not None:
            inps = torch.cat(inps, dim=0).detach().cpu().numpy()
            np.save(join(save_path, 'inps.npy'), inps)
            np.save(join(save_path, 'preds.npy'), preds)
            np.save(join(save_path, 'trues.npy'), trues)
        metrics = calc_metrics(preds, trues)
        return metrics

    total_loss = torch.cat(total_loss, dim=0).cpu()
    return np.average(total_loss)
