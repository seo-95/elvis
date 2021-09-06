import math
import os
import pdb
from datetime import datetime

from elvis.utils import TrainLogger

from .distributed import DistributedTrainer


class LogTrainer(DistributedTrainer):
    """Distributed trainer with log integration
        - output logger: only in master process (rank == 0)
        - file logger: one file for each process in the pool
    """
    def __init__(self, cfg, **kwargs):
        super(LogTrainer, self).__init__(cfg, **kwargs)
        #logging options
        self.stdout_flag  = cfg.TRAINER.LOGGER.STDOUT
        self.flog_flag    = cfg.TRAINER.LOGGER.FILE

        self.lr_values     = []
        self.acc_losses    = [] #loss accumulated over steps
        self.loss_by_step  = []
        self.loss_by_epoch = {'train': [], 'dev': []}
        self.virtual_step  = 0 #step that takes into account the accumulation steps
        #to compute estimated time to arrival over a window containing the last 50 values
        self.step_t_list  = []
        self.window_size  = 100

    def start_train_hook(self):
        super().start_train_hook()
        filename           = os.path.join(self._ckp_dir, 'train{}{}.log'.format(self._rank, self._device.index))
        self.logger        = TrainLogger(outstream=self.stdout_flag if self._rank == 0 else False,
                                         fstream=self.flog_flag,
                                         filename=filename)
        if self._rank == 0:
            self._cfg.save_to_disk(save_path=os.path.join(self._ckp_dir, 'cfg.yaml'))
        self.effective_tr_steps  = math.ceil(self._tr_steps_per_epoch / self._acc_steps)
        self.effective_dev_steps = math.ceil(self._dev_steps_per_epoch / self._acc_steps)
        self.tot_tr_steps        = self.effective_tr_steps * (self._hparams['MAX_EPOCHS'] - self._epoch)
        self.tot_dev_steps       = self.effective_dev_steps * (self._hparams['MAX_EPOCHS'] - self._epoch)
        self.train_start_t       = datetime.now()

    def start_epoch_hook(self):
        super().start_epoch_hook()
        #init curr epoch stats
        self.loss_by_epoch['train'].append(0)
        self.loss_by_epoch['dev'].append(0)
        self.epoch_start_t = datetime.now()

    def before_train_step_hook(self):
        super().before_train_step_hook()
        if (self._step-1) % self._acc_steps == 0 or (self._step-1) % self._tr_steps_per_epoch == 0:
            #save start time only if the previous step was an optimization step
            self.step_start_t = datetime.now()

    def after_train_step_hook(self):
        super().after_train_step_hook()
        self.acc_losses.append(self._last_loss['loss'].item())
        if self._step % self._acc_steps != 0 and self._step % self._tr_steps_per_epoch != 0:
            #print logs only when optimization occurs
            return
        self.step_end_t = datetime.now()
        #virtual_step starts from 1 and take in consideration the number of accumulation steps
        self.virtual_step  += 1 #self._step // self._acc_steps
        self.acc_loss      = sum(self.acc_losses)/len(self.acc_losses)
        self.curr_lr       = self._optimizer.param_groups[0]['lr']
        self.acc_losses    = [] #reset
        if self.stdout_flag or self.flog_flag:
            #estimate eta over a windows of N values for elapsed time
            eta = self.estimate_eta()
            log_msg = {'step'   : self.virtual_step,
                       'ep'     : self._epoch,
                       'tr_loss': self.acc_loss,
                       'lr'     : self.curr_lr,
                       'eta'    : str(eta).split('.')[0],
                       't'      : datetime.now().strftime("%b %d %Y %H:%M:%S")
                    }
            self.logger.log(log_msg, stream='file')
            self.logger.log(log_msg, stream='stdout')
        self.lr_values.append(self.curr_lr)
        self.loss_by_step.append(self.acc_loss)
        self.loss_by_epoch['train'][-1] += self.acc_loss

    def after_eval_step_hook(self, dev_step:int):
        super().after_eval_step_hook(dev_step)
        if dev_step == 0:
            assert len(self.acc_losses) == 0
        self.acc_losses.append(self._last_loss['loss'].item()) 
        if (dev_step+1) % self._acc_steps == 0 or (dev_step+1) % self._dev_steps_per_epoch == 0:
            acc_loss = sum(self.acc_losses)/len(self.acc_losses)
            self.loss_by_epoch['dev'][-1] += acc_loss

    def end_epoch_hook(self):
        super().end_epoch_hook()
        self.epoch_end_t = datetime.now()
        tot_time         = str(self.epoch_end_t - self.epoch_start_t).split('.')[0]
        curr_t           = datetime.now().strftime("%b %d %Y %H:%M:%S")
        log_msg = {'msg'      : 'Epoch ended',
                    'ep'      : self._epoch,
                    'tr_loss' : self.loss_by_epoch['train'][-1]/self.effective_tr_steps,
                    'dev_loss': self.loss_by_epoch['dev'][-1]/self.effective_dev_steps,
                    'tot_time': tot_time,
                    't'       : curr_t
                }
        console_msg = 'Epoch {} ended ({}). Total time: {}'.format(self._epoch, curr_t, tot_time)
        self.logger.log(log_msg, stream='file')
        self.logger.log(console_msg, stream='stdout')

    def end_train_hook(self):
        super().end_train_hook()
        self.train_end_t = datetime.now()
        tot_time         = str(self.train_end_t - self.train_start_t).split('.')[0]
        msg              = 'Training ended ({}). Total time: {}'.format(self.train_end_t, tot_time)
        self.logger.log(msg, stream='stdout')
        self.logger.log({'msg': msg}, stream='file')
        self.logger.close()

    def estimate_eta(self):
        step_elapse_t = self.step_end_t-self.step_start_t
        if hasattr(self, 'estimate_step_t'):
            if len(self.step_t_list) == 50:
                pop_val               = self.step_t_list.pop()
                self.estimate_step_t -= pop_val
            self.step_t_list.insert(0, step_elapse_t)
            self.estimate_step_t += step_elapse_t
        else:
            #init
            self.estimate_step_t = step_elapse_t
            self.step_t_list.append(step_elapse_t)
        tot_steps  = self.tot_tr_steps + self.tot_dev_steps
        done_steps = self.virtual_step + self.effective_dev_steps * self._epoch
        eta        = self.estimate_step_t/len(self.step_t_list) * (tot_steps - done_steps)
        return eta

    def enough_training_time(self) -> bool:
        is_enough = super().enough_training_time()
        if not is_enough:
            self.logger.log('Premature end of training due to walltime. Resume checkpoint saving...', stream='stdout')
            self.logger.log({'msg': 'Premature end of training due to walltime. Resume checkpoint saving...'}, stream='file')
        return is_enough

    def premature_end(self) -> None:
        self.logger.log({'msg': '-- premature end due to walltime -- '}, stream='file')
        self.logger.log({'msg': '-- premature end due to walltime -- '}, stream='stdout')
