import torch
import numpy as np
import random
import sys
sys.path.append('/notebooks/cia')
from det3d.torchie import Config
from det3d.models import build_detector
from det3d.torchie.trainer.trainer import example_to_device
from det3d.torchie.parallel.collate import collate_kitti
from det3d.datasets.pipelines.compose import Compose
from det3d.torchie.apis.train import build_one_cycle_optimizer
from det3d.builder import _create_learning_rate_scheduler
from det3d.torchie.trainer.hooks import (CheckpointHook, Hook, IterTimerHook, LrUpdaterHook, OptimizerHook, lr_updater, TextLoggerHook, )
from det3d.torchie.apis.env import get_root_logger
from det3d.torchie.trainer.log_buffer import LogBuffer
from det3d.torchie.trainer.trainer import parse_second_losses
import warnings
import logging
warnings.filterwarnings('ignore')  # to remove warnings from numba
logging.getLogger('numba.transforms').setLevel(logging.ERROR)
from  det3d.torchie.parallel.collate import collate_kitti
from torch.nn.utils import clip_grad

class ODModel:
    """
    This class is the interface of the CIA-SSD model with the pseudo lidar project
    
    """  
    def __init__(self, cfg_path, logger = get_root_logger("INFO"), loader_len = 928):
        
        
        self.cfg = Config.fromfile(cfg_path)

        self.model = build_detector(self.cfg.model, train_cfg=self.cfg.train_cfg, test_cfg=self.cfg.test_cfg).cuda()
        
        self.optimizer = build_one_cycle_optimizer(self.model, self.cfg.optimizer)
        self._hooks = [OptimizerHook(**self.cfg.optimizer_config), CheckpointHook(self.cfg.checkpoint_config), TextLoggerHook(), IterTimerHook()]
        self.hooks = self._hooks
        self.data_loader_len = loader_len
        total_steps = self. cfg.total_epochs * self.data_loader_len
        
        self.lr_scheduler = _create_learning_rate_scheduler(self.optimizer, self.cfg.lr_config, total_steps)
        self.logger = logger
        self.log_buffer = LogBuffer()
        
        
        self.outputs = None
        self._iter = 0
        self._inner_iter = 0
        self._epoch = 0
        self._max_epochs = 60
        self.mode = "train"
        self.call_hook(self.hooks, "before_run")
        
    def get_pipline(self):
        pipeline =  Compose(self.cfg.train_pipeline)
        return pipeline
    
    def iter(self):
        return self._iter
    def inner_iter(self):
        return self._inner_iter
    def epoch(self):
        return self._epoch
    def current_lr(self):
        if self.optimizer is None:
            raise RuntimeError("lr is not applicable because optimizer does not exist.")
        return [group["lr"] for group in self.optimizer.param_groups]
    def max_iters(self):
        return self._max_epochs * self.data_loader_len
    
    def call_hook(self, hooks,fn_name):
        for hook in hooks:
            getattr(hook, fn_name)(self)  # self is the param (trainer/runner) of func hook.fn_name
    def before_train_epoch(self):
        self.call_hook(self.hooks,"before_train_epoch")
        
    def after_train_epoch(self):
        self.call_hook(self.hooks,"after_train_epoch")
        
        
    def cia_forward(self, batch, epoch, batch_idx):
        batch = collate_kitti(batch)
        base_step = epoch * self.data_loader_len
        global_step = base_step * batch_idx
        self.lr_scheduler.step(global_step)
        self._inner_iter = batch_idx
        self.call_hook(self._hooks, "before_train_iter")
        example = example_to_device(batch, torch.cuda.current_device(), non_blocking=False)
        self.call_hook(self.hooks, "after_data_to_device")
        losses =  self.model(example, return_loss = True)
        self.call_hook(self.hooks, "after_forward")
        loss, log_vars = parse_second_losses(losses)
        del losses
        
        self.outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(example["anchors"][0]))
        self.call_hook(self.hooks, "after_parse_loss")
        
        if "log_vars" in self.outputs:
            self.log_buffer.update(self.outputs["log_vars"], self.outputs["num_samples"])
        
        self.optimizer.zero_grad()
#         self.call_hook(self.hooks, "after_train_iter")
        self._iter += 1
        
        return self.outputs
    def clip_grad(self):
        grad_clip = self.hooks[0].grad_clip
        clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.parameters()), **grad_clip)
    def optimizer_step(self):
        self.optimizer.step()    
                
        
        