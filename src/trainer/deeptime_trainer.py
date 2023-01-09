import numpy as np
import gin
import pytorch_lightning as pl

from models import get_model
import random

import torch
import torch.nn.functional as F
from torch import optim

import math

from utils import Checkpoint, default_device, to_tensor
@gin.configurable
class DeepTimeTrainer(pl.LightningModule):

    def __init__(self,
                 lr,
                 lambda_lr,
                 weight_decay,
                 warmup_epochs,
                 random_seed,
                 T_max,
                 eta_min,
                 dim_size,
                 datetime_feats,
                 ):
        gin.parse_config_file('/home/reza/Projects/PL_DeepTime/DeepTime/config/config.gin')
        super(DeepTimeTrainer, self).__init__()
        self.lr = lr
        self.lambda_lr = lambda_lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs

        # Generator

        self.random_seed = random_seed

        #######
        self.lr = lr
        self.lambda_lr = lambda_lr
        self.weight_decay = weight_decay
        self.T_max = T_max
        self.warmup_epochs = warmup_epochs
        self.eta_min = eta_min

        self.model = get_model(
            model_type='deeptime',
            dim_size=dim_size,
            datetime_feats=datetime_feats
        )

    def on_fit_start(self):
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

    def training_step(self, batch, batch_idx):
        x, y, x_time, y_time = map(to_tensor, batch)
        forecast = self.model(x, x_time, y_time)

        if isinstance(forecast, tuple):
            # for models which require reconstruction + forecast loss
            loss = F.mse_loss(forecast[0], x) + \
                   F.mse_loss(forecast[1], y)
        else:
            loss = F.mse_loss(forecast, y)

        self.log('train_loss', loss, prog_bar=True, on_epoch=True)

        # remove lr scheduler
        return {'loss': loss, 'train_loss': loss, }

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["train_loss"] for x in outputs]).mean()

        self.log('avg_train_loss', avg_train_loss, on_epoch=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):

        x, y, x_time, y_time = map(to_tensor, batch)
        forecast = self.model(x, x_time, y_time)

        if isinstance(forecast, tuple):
            # for models which require reconstruction + forecast loss
            loss = F.mse_loss(forecast[0], x) + \
                   F.mse_loss(forecast[1], y)
        else:
            loss = F.mse_loss(forecast, y)

        self.log('val_loss', loss, prog_bar=True, on_epoch=True)

        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        return outputs

    def test_step(self, batch, batch_idx):
        x, y, x_time, y_time = map(to_tensor, batch)
        forecast = self.model(x, x_time, y_time)

        if isinstance(forecast, tuple):
            # for models which require reconstruction + forecast loss
            loss = F.mse_loss(forecast[0], x) + \
                   F.mse_loss(forecast[1], y)
        else:
            loss = F.mse_loss(forecast, y)

        self.log('test_loss', loss, prog_bar=True, on_epoch=True)

        return {'test_loss': loss}

    def test_epoch_end(self, outputs):
        return outputs

    @gin.configurable
    def configure_optimizers(self):
        group1 = []  # lambda
        group2 = []  # no decay
        group3 = []  # decay
        no_decay_list = ('bias', 'norm',)
        for param_name, param in self.model.named_parameters():
            if '_lambda' in param_name:
                group1.append(param)
            elif any([mod in param_name for mod in no_decay_list]):
                group2.append(param)
            else:
                group3.append(param)
        optimizer = optim.Adam([
            {'params': group1, 'weight_decay': 0, 'lr': self.lambda_lr, 'scheduler': 'cosine_annealing'},
            {'params': group2, 'weight_decay': 0, 'scheduler': 'cosine_annealing_with_linear_warmup'},
            {'params': group3, 'scheduler': 'cosine_annealing_with_linear_warmup'}
        ], lr=self.lr, weight_decay=self.weight_decay)

        scheduler_fns = []
        for param_group in optimizer.param_groups:
            scheduler = param_group['scheduler']
            if scheduler == 'none':
                fn = lambda T_cur: 1
            elif scheduler == 'cosine_annealing':
                lr = eta_max = param_group['lr']
                fn = lambda T_cur: (self.eta_min + 0.5 * (eta_max - self.eta_min) * (
                        1.0 + math.cos(
                    (T_cur - self.warmup_epochs) / (self.T_max - self.warmup_epochs) * math.pi))) / lr
            elif scheduler == 'cosine_annealing_with_linear_warmup':
                lr = eta_max = param_group['lr']
                fn = lambda T_cur: T_cur / self.warmup_epochs if T_cur < self.warmup_epochs else (self.eta_min + 0.5 * (
                        eta_max - self.eta_min) * (1.0 + math.cos(
                    (T_cur - self.warmup_epochs) / (self.T_max - self.warmup_epochs) * math.pi))) / lr
            else:
                raise ValueError(f'No such scheduler, {scheduler}')
            scheduler_fns.append(fn)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler_fns)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def forward(self, batch, z_0=None):
        z_0 = None
        Y = batch['Y'].to(default_device)
        sample_mask = batch['sample_mask'].to(default_device)
        available_mask = batch['available_mask'].to(default_device)

        # Forecasting
        forecasting_mask = available_mask.clone()
        if self.n_time_out > 0:
            forecasting_mask[:, 0, -self.n_time_out:] = 0

        Y, Y_hat, z = self.model(Y=Y, mask=forecasting_mask, idxs=None, z_0=z_0)

        if self.n_time_out > 0:
            Y = Y[:, :, -self.n_time_out:]
            Y_hat = Y_hat[:, :, -self.n_time_out:]
            sample_mask = sample_mask[:, :, -self.n_time_out:]

        return Y, Y_hat, sample_mask, z
