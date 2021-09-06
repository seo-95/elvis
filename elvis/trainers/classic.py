import copy
import math
import os
import pdb
import random
import socket
from datetime import datetime
from typing import Dict

import numpy as np
import torch
from torch import autograd
from torch.utils.data import DataLoader
from warmup_scheduler import GradualWarmupScheduler

from ..modeling import build_model
from ..utils import build_dataset
from .base import BaseTrainer
from .build import TRAINER_REGISTRY


class ClassicTrainer(BaseTrainer):
    """Classic single gpu (or cpu) training
    """

    def __init__(self,
                 cfg,
                 device=torch.device('cpu'),
                 resume_dict=None,
                 fp16=False,
                 walltime=None,
                 create_ckp_dir=True) -> None:
        super(ClassicTrainer, self).__init__()
        self.seed_everything(cfg.TRAINER.SEED)
        self._cfg       = cfg
        self._step      = 0
        self._epoch     = 0
        self._fp16      = fp16
        self._device    = device
        self._hparams   = cfg.TRAINER.HYPERPARAMS.get_as_dict()
        self._workers_n = cfg.TRAINER.WORKERS
        self.grad_clipv = cfg.TRAINER.GRADIENT_CLIPPING
        self._acc_steps = self._hparams['GRADIENT_ACCUMULATIONS_STEPS'] if 'GRADIENT_ACCUMULATIONS_STEPS' in self._hparams else 1
        assert self._acc_steps > 0
        self._ckp_flag  = cfg.TRAINER.CHECKPOINT.ENABLED
        self._best_loss = math.inf
        self._walltime  = walltime
        if self._ckp_flag and create_ckp_dir:
            self._ckp_dir = cfg.TRAINER.CHECKPOINT.DIR
            subdir_name   = '{}_{}'.format(datetime.today().strftime('%B%d_%H-%M-%S'), socket.gethostname())
            self._ckp_dir = os.path.join(self._ckp_dir, subdir_name)
            os.makedirs(self._ckp_dir, exist_ok=True)
            self._cfg.save_to_disk(save_path=os.path.join(self._ckp_dir, 'cfg.yaml'))
            self._resume_dir = os.path.join(self._ckp_dir, 'resume')
            os.makedirs(self._resume_dir, exist_ok=True)

        self._model, self._data_interface = build_model(cfg)
        self._model.to(self._device, non_blocking=True)
        (self._trainset, self._devset)    = build_dataset(cfg.TRAINER.DATASET,
                                                          worker_fn=self._data_interface.worker_fn)
        #TODO: here create build functions for optimizer and scheduler
        self._optimizer = torch.optim.AdamW(params=self._model.parameters(), 
                                            lr=self._hparams['LR'], 
                                            weight_decay=self._hparams['WEIGHT_DECAY'])
        self._scaler    = torch.cuda.amp.GradScaler(enabled=self._fp16)
    
    def train(self):
        #prepare data loaders
        params = {'batch_size': self._hparams['BATCH_SIZE'],
                  'shuffle': True,
                  'num_workers': self._workers_n,
                  'pin_memory': True if self._device.type == 'cuda' else False}
        self._trloader            = DataLoader(self._trainset, **params, collate_fn=self._data_interface.collate_fn)
        self._devloader           = DataLoader(self._devset, **params, collate_fn=self._data_interface.collate_fn)
        self._train_it            = range(self._epoch+1, self._hparams['MAX_EPOCHS']+1)
        self._tr_steps_per_epoch  = len(self._trloader)
        self._dev_steps_per_epoch = len(self._devloader)
        #init schedulers based on the total number of steps
        tot_steps       = math.ceil(self._tr_steps_per_epoch * self._hparams['MAX_EPOCHS'] /self._acc_steps)
        warmup_steps    = self._cfg.TRAINER.HYPERPARAMS.WARMUP_STEPS
        post_sched      = torch.optim.lr_scheduler.LambdaLR(self._optimizer, 
                                                            lr_lambda=lambda step : (tot_steps-step)/(tot_steps-warmup_steps))
        self._scheduler = GradualWarmupScheduler(self._optimizer,
                                                 multiplier=self._cfg.TRAINER.HYPERPARAMS.WARMUP_MULTIPLIER,
                                                 warmup_steps=warmup_steps,
                                                 post_warmup_scheduler=post_sched)
        if hasattr(self, '_resume_scheduler'):
            self._scheduler.load_state_dict(self._resume_scheduler)

        self.start_train_hook()
        self.train_loop()
        self.end_train_hook()

    def train_loop(self):
        self.start_t = datetime.now() #used to evaluate resume savings
        for _ in self._train_it:
            self._epoch += 1
            self._curr_dev_loss = 0
            self._model.train()
            self.start_epoch_hook()
            for batch in self._trloader:
                self._step += 1
                self.before_train_step_hook()
                self.move_dict_to_device(batch)
                self._last_out, self._last_loss = self.do_step(batch)
                self.after_train_step_hook()
            self._model.eval()
            with torch.no_grad():
                for dev_step, batch in enumerate(self._devloader):
                    self.move_dict_to_device(batch)
                    self.before_eval_step_hook(dev_step)
                    self._last_out, self._last_loss = self.do_step(batch, optimize=False)
                    self._curr_dev_loss            += self._last_loss['loss'].item()
                    self.after_eval_step_hook(dev_step)
            self.save_best_checkpoint()
            self.save_resume_checkpoint()
            if self._walltime and not self.enough_training_time():
                self.premature_end()
                return
            self.end_epoch_hook()

    def do_step(self, batch, optimize=True):
        #use mixed precision only at training time
        use_amp = self._fp16 and optimize
        #with autograd.set_detect_anomaly(False):
        with torch.cuda.amp.autocast(enabled=use_amp):
            self.move_dict_to_device(batch)
            out     = self._model(**batch)
            loss    = self._model.compute_loss(**out, **batch)
            #? in case of memory saturation, here we can freed gpu my moving batch back to cpu asynchronously
        if optimize:
            self._scaler.scale(loss['loss']/self._acc_steps).backward()
            #plot_grad_flow(self._model.model.named_parameters())
            #for item in self._model.parameters():
            #    pdb.set_trace()
            if (self._step+1) % self._acc_steps == 0 or (self._step+1) % self._tr_steps_per_epoch == 0:
                #unscale before clipping
                self._scaler.unscale_(self._optimizer)
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), self.grad_clipv)
                self._scaler.step(self._optimizer)
                self._scaler.update()
                self._optimizer.zero_grad()
                self._scheduler.step() #it calls warning if the optimizer is skipped due to not valid unscaled grads
        return out, loss

    def save_best_checkpoint(self):
        assert not math.isnan(self._curr_dev_loss)
        if self._curr_dev_loss < self._best_loss:
            self._best_loss = self._curr_dev_loss
            if self._ckp_flag:
                self._model.save_on_disk(self._ckp_dir)

    def save_resume_checkpoint(self):
        resume_dict = {'model_state': copy.deepcopy(self._model).cpu().state_dict(), #todo adapt to model.save_on_disk()
                       'optim_state': self._optimizer.state_dict(),
                       'sched_state': self._scheduler.state_dict(),
                       'epoch'      : self._epoch #? save also curr loss?
                    }
        torch.save(resume_dict, os.path.join(self._resume_dir, 'resume_dict_ep{}.pt'.format(self._epoch)))

    def resume(self, resume_path):
        res_dict = torch.load(resume_path)
        self._model.load_state_dict(res_dict['model_state']) #todo adapt to model.save_on_disk()
        self._optimizer.load_state_dict(res_dict['optim_state'])
        # scheduler was not yet created
        if 'sched_state' in res_dict:
            self._resume_scheduler = res_dict['sched_state']
        #self._scheduler.load_state_dict(res_dict['sched_state'])
        self._epoch = res_dict['epoch']

    def from_pretrained(self, state_dict):
        self._model.from_pretrained(state_dict)

    def enough_training_time(self) -> bool:
        elapsed_t = datetime.now() - self.start_t
        #remaining time should be more than the time for an epoch plus its 10%
        return self._walltime - elapsed_t > elapsed_t/self._epoch * (1+0.1)

    def premature_end(self) -> None:
        pass

    def move_dict_to_device(self, input_dict: Dict) -> Dict:
        """This method moves all the tensors inside the dictionary on the specified device

        Args:
            tensor_dict (Dict): dictionary to move on device
        """
        for k in input_dict.keys():
            if isinstance(input_dict[k], torch.Tensor,):
                input_dict[k] = input_dict[k].to(self._device, non_blocking=True)
            elif isinstance(input_dict[k], Dict):
                self.move_dict_to_device(input_dict[k])

    def seed_everything(self, seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

import matplotlib.pyplot as plt


def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for name, param in named_parameters:
        if(param.requires_grad) and ("bias" not in name):
            layers.append(name)
            if param.grad is None:
                ave_grads.append(torch.zeros(1).cuda())
            else:
                ave_grads.append(param.grad.abs().mean())

    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.show()
