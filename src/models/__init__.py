from typing import Union

import gin
import torch

from .DeepTIMe import deeptime
# from .forecast import ForecastDataset, ForecastExperiment, Checkpoint, get_loss_fn, calc_metrics, Experiment, Optional
from .forecast import get_model, get_data, get_optimizer, get_loss_fn, get_scheduler


@gin.configurable
def get_model(model_type: str, **kwargs: Union[int, float]) -> torch.nn.Module:
    if model_type == 'deeptime':
        model = deeptime(datetime_feats=kwargs['datetime_feats'])
    else:
        raise ValueError(f"Unknown model type {model_type}")
    return model
