import argparse
import os
import pdb
import socket
from datetime import datetime, timedelta

import torch
import yaml
from torch import autograd
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from elvis.config import ConfigNode
from elvis.trainers import LogTrainer


class MyTrainer(LogTrainer):
    """Distributed trainer with log integration
        - tensorboard logger: only in master process (rank == 0)
        - output logger: only in master process (rank == 0)
        - file logger: one file for each process in the pool
    """
    def __init__(self, cfg, **kwargs):
        super(MyTrainer, self).__init__(cfg, **kwargs)
        #logging options
        self.tblog_flag   = cfg.TRAINER.LOGGER.TENSORBOARD
        self.pbar_flag    = cfg.TRAINER.LOGGER.PROGRESS_BAR

    def start_train_hook(self):
        super().start_train_hook()
        console_msg    = 'Using {} gpus for {} steps. Machine {}'.format(self._world_size, self.tot_tr_steps, socket.gethostname())
        if self._fp16:
            console_msg += '\nUsing mixed precision training'
        self.logger.log(console_msg, stream='stdout')
        if self._rank == 0:
            if self.tblog_flag:
                #open tensorboard handler here and not before the spawn (the handler is not pickable)
                tb_path        = os.path.join('runs', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
                tb_path        = os.path.join(self._ckp_dir, tb_path)
                self.tb_writer = SummaryWriter(tb_path)
                self.tb_writer.add_text('progress', 'Training started ({})'.format(self.train_start_t), self._step)
            if self.pbar_flag: 
                self._train_it = tqdm(self._train_it)
        else:
            #no tensorboard logs for other processes
            self.tblog_flag  = False

    def end_train_hook(self):
        super().end_train_hook()
        tot_time         = self.train_end_t - self.train_start_t
        if self._rank == 0 and self.tblog_flag:
            self.tb_writer.add_text('progress', 'Training ended ({}). Total time: {}'.format(self.train_end_t, tot_time), self.virtual_step)
            #todo here log hparams on tensorboard
            self.tb_writer.close()

    def end_epoch_hook(self):
        super().end_epoch_hook()
        if self._rank == 0 and self.tblog_flag:
            tr_loss  = self.loss_by_epoch['train'][-1]/self.effective_tr_steps
            dev_loss = self.loss_by_epoch['dev'][-1]/self.effective_dev_steps
            self.tb_writer.add_text('progress', 'Epochs {} ended ({})'.format(self._epoch, self.epoch_end_t), self._step)
            self.tb_writer.add_scalar('epoch_loss', dev_loss, self._epoch)
            self.tb_writer.add_scalars('loss_compare', {'train': tr_loss,'dev': dev_loss}, self._epoch)

    def after_train_step_hook(self):
        super().after_train_step_hook()
        if self._rank == 0 \
            and self._step % self._acc_steps == 0 or self._step % self._tr_steps_per_epoch == 0 \
            and self.tblog_flag:
            self.tb_writer.add_scalar('steps_loss', self.acc_loss, self.virtual_step)
            self.tb_writer.add_scalar('lr', self.curr_lr, self.virtual_step)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--local_rank',
        type=int,
        help="Local rank. Necessary for using the torch.distributed.launch utility."
    )
    parser.add_argument(
        '--config_file',
        type=str,
        required=True,
        help='Path to yaml config file'
    )
    parser.add_argument(
        '--fp16',
        action='store_true',
        default=False,
        required=False,
        help='Train with mixed precision'
    )
    parser.add_argument(
        '--walltime',
        type=str,
        default=None,
        required=False,
        help='Maximum walltime in HH:MM:SS (up to 23:59) for the training (resume checkpoint saving calibrated based on this value)'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        required=False,
        help='Path to folder containing the resume_dict.pt'
    )
    parser.add_argument(
        '--from_pretrained',
        type=str,
        default=None,
        required=False,
        help='Path to pretrained model parameters'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default=None,
        required=False,
        help='Path to folder containing the resume_dict.pt'
    )

    args = parser.parse_args()
    with open(args.config_file) as fp:
        cfg = yaml.safe_load(fp)
        cfg = ConfigNode(cfg)

    walltime = None
    if args.walltime:
        date_time   = datetime.strptime(args.walltime, "%H:%M:%S")
        a_timedelta = date_time - datetime(1900, 1, 1)
        seconds     = a_timedelta.total_seconds()
        walltime    = timedelta(seconds=seconds)
    kwargs = {'fp16'     : args.fp16,
              'walltime' : walltime
            }
    if args.log_dir is not None:
        kwargs.update({'log_dir': args.log_dir})
    trainer = MyTrainer(cfg, **kwargs)
    if args.resume:
        print('resuming from {}'.format(args.resume))
        trainer.resume(args.resume)
    if args.from_pretrained:
        print('from pretrained {}'.format(args.from_pretrained))
        state_dict = torch.load(args.from_pretrained)
        trainer.from_pretrained(state_dict)
    trainer.train()
