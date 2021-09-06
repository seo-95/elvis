import copy
import math
import os
import pdb
import socket
from datetime import datetime

import torch
import torch.distributed as dist
from torch import autograd
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from warmup_scheduler import GradualWarmupScheduler

from .build import TRAINER_REGISTRY
from .classic import ClassicTrainer

#availables environment variables:
# os.environ['MASTER_ADDR']
# os.environ['MASTER_PORT']
# os.environ['WORLD_SIZE']
# os.environ['RANK']
# os.environ['LOCAL_RANK']

#Single node distributed training:
# python -m torch.distributed.launch --nproc_per_node=NUM_GPUS script.py --arg1 --arg2 ...

#Multi-node distributed training:
# python -m torch.distributed.launch --nproc_per_node=NUM_GPUS --nnodes=2 --node_rank=0 --master_addr="192.168.1.1" --master_port=1234 script.py --arg1 --arg2 ...

class DistributedTrainer(ClassicTrainer):
    """Trainer class implementing DistributedDataParallel with 1 process per GPUs.
        To use this class you need to set MASTER_ADDR and MASTER_PORT environment variables.
    """
    
    def __init__(self,
                 cfg,
                 fp16=False,
                 **kwargs) -> None:
        #set gpu and rank
        self._world_size = int(os.environ['WORLD_SIZE'])
        self._rank       = int(os.environ['RANK'])
        self._local_rank = int(os.environ['LOCAL_RANK'])
        self._gpu        = self._local_rank
        torch.cuda.set_device(self._gpu)

        #build model, optimizer etc. for each process but do not create the checkpoint folder (only if rank==0)
        super(DistributedTrainer, self).__init__(cfg,
                                                 device=torch.device('cuda:{}'.format(self._gpu)),
                                                 fp16=fp16,
                                                 walltime=kwargs['walltime'] if 'walltime' in kwargs else None,
                                                 create_ckp_dir=False)
        self._ckp_dir = cfg.TRAINER.CHECKPOINT.DIR
        subdir_name   = '{}_{}'.format(datetime.today().strftime('%B%d_%H-%M'), socket.gethostname()) if 'log_dir' not in kwargs else kwargs['log_dir']
        self._ckp_dir = os.path.join(self._ckp_dir, subdir_name)
        if self._ckp_flag and self._rank == 0:
            if self._rank == 0:
                os.makedirs(self._ckp_dir, exist_ok=False) #if already exists do not overwrite results
                self._cfg.save_to_disk(save_path=os.path.join(self._ckp_dir, 'cfg.yaml'))
                self._resume_dir = os.path.join(self._ckp_dir, 'resume')
                os.makedirs(self._resume_dir, exist_ok=True)

        #self._model.cuda()
        #initialize multiprocessing
        dist.init_process_group(backend='nccl', init_method='env://')
        self._model = DDP(self._model, device_ids=[self._gpu], find_unused_parameters=True)

    def train(self):
        assert self._hparams['BATCH_SIZE'] % self._world_size == 0, 'Batch must be divisible by the total number of processes'
        #prepare data loaders
        params = {'batch_size' : self._hparams['BATCH_SIZE']//self._world_size,
                  'shuffle'    : False, #DistributedDataParallel is not compatible with shuffle
                  'num_workers': self._workers_n,
                  'pin_memory' : True}
        self._tr_sampler  = DistributedSampler(self._trainset, num_replicas=self._world_size, rank=self._rank)
        self._dev_sampler = DistributedSampler(self._devset, num_replicas=self._world_size, rank=self._rank) 
            
        self._trloader    = DataLoader(self._trainset, **params, sampler=self._tr_sampler, collate_fn=self._data_interface.collate_fn)
        self._devloader   = DataLoader(self._devset, **params, sampler=self._dev_sampler, collate_fn=self._data_interface.collate_fn)
        self._train_it    = range(1, self._hparams['MAX_EPOCHS']+1)
        self._tr_steps_per_epoch  = len(self._trloader)
        self._dev_steps_per_epoch = len(self._devloader)
        #init schedulers based on the total number of steps
        tot_steps       = math.ceil(self._tr_steps_per_epoch * self._hparams['MAX_EPOCHS'] /self._acc_steps)
        warmup_steps    = self._cfg.TRAINER.HYPERPARAMS.WARMUP_STEPS
        #scheduler for linear decay
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

    def do_step(self, batch, optimize=True):
        #use mixed precision only at training time
        use_amp = self._fp16 and optimize
        with torch.cuda.amp.autocast(enabled=use_amp):
            self.move_dict_to_device(batch)
            out     = self._model(**batch)
            loss    = self._model.module.compute_loss(**out, **batch)
            #? in case of memory saturation, here we can freed gpu my moving batch back to cpu asynchronously
        if optimize:
            self._scaler.scale(loss['loss']/self._acc_steps).backward()
            if (self._step+1) % self._acc_steps == 0 or (self._step+1) % self._tr_steps_per_epoch == 0:
                #unscale before clipping
                self._scaler.unscale_(self._optimizer)
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), self.grad_clipv)
                self._scaler.step(self._optimizer)
                self._scaler.update()
                self._optimizer.zero_grad()
                self._scheduler.step() #it calls warning if the optimizer is skipped due to not valid unscaled grads
        return out, loss

    def save_resume_checkpoint(self):
        if self._rank == 0:
            resume_dict = {'model_state': copy.deepcopy(self._model.module).cpu().state_dict(), #todo adapt to model.save_on_disk()
                           'optim_state': self._optimizer.state_dict(),
                           'sched_state': self._scheduler.state_dict(),
                           'epoch'      : self._epoch #? save also curr loss?
                        }
            """
            #pytorch < 1.8.0 compatibility: state_dict() is not done recursively
            if 'post_wup_scheduler' in resume_dict['sched_state'] and type(resume_dict['sched_state']['post_wup_scheduler']) != dict:
                resume_dict['sched_state']['post_wup_scheduler'] = resume_dict['sched_state']['post_wup_scheduler'].state_dict()
            """
            torch.save(resume_dict, os.path.join(self._resume_dir, 'resume_dict_ep{}.pt'.format(self._epoch)))
        torch.distributed.barrier()


    def resume(self, resume_path):
        resume_dict = torch.load(resume_path, map_location='cuda:{}'.format(self._gpu))
        self._model.module.load_state_dict(resume_dict['model_state']) #todo adapt to model.save_on_disk()
        self._optimizer.load_state_dict(resume_dict['optim_state'])
        # scheduler was not yet created
        if 'sched_state' in resume_dict:
            self._resume_scheduler = resume_dict['sched_state']
        #self._scheduler.load_state_dict(res_dict['sched_state'])
        self._epoch = resume_dict['epoch']

    def from_pretrained(self, state_dict):
        self._model.module.from_pretrained(state_dict)

    def save_best_checkpoint(self):
        if self._rank == 0:
            assert not math.isnan(self._curr_dev_loss)
            if self._curr_dev_loss < self._best_loss:
                self._best_loss = self._curr_dev_loss
                if self._ckp_flag:
                    self._model.module.save_on_disk(self._ckp_dir)
        if self._world_size > 1:
            torch.distributed.barrier()

    def start_epoch_hook(self):
        #used to shuffle data between epochs with distributed training
        self._tr_sampler.set_epoch(self._epoch)

    """
    def enough_training_time(self) -> bool:
        if self._rank == 0:
            super().save_resume_checkpoint()
        #todo send results on all processes to make them exit together from training
    """

@TRAINER_REGISTRY.register()
def build_distributed_trainer(cfg, **kwargs):
    return DistributedTrainer(cfg=cfg, **kwargs)
