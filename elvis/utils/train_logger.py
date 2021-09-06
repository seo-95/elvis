import json
import os
import sys
from os import write

from torch.utils.tensorboard import SummaryWriter


class TBLogger:
    """TensorBoard training logger
        - TensorBoard will save the logs inside the "runs" folder
        - To see the log just start the server with `tensorboard --logdir runs` and check it with a browser
    """
    def __init__(self, log_dir, rank=0):
        self.rank = rank
        if rank != 0:
            return
        self.logging_off = not len(log_dir)
        if self.logging_off:
            return
        #original name for tb folder
        import socket
        from datetime import datetime
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        tb_dir = os.path.join('runs', current_time + '_' + socket.gethostname())
        self.log_dir = os.path.join(log_dir, tb_dir)
        self.tb = SummaryWriter(self.log_dir)

    def log(self, heading, msg, step, log_type):
        if self.rank != 0 or self.logging_off:
            return
        if log_type == 'scalar':
            if type(msg) == float:
                self.tb.add_scalar(heading, msg, step)
            elif type(msg) == dict:
                self.tb.add_scalars(heading, msg, step)
            else:
                raise TypeError('{} message type is not allowed for log type {}'.format(type(msg), log_type))
        elif log_type == 'text':
            self.tb.add_text(heading, msg, step)
        elif log_type == 'histogram':
            self.tb.add_histogram(heading, msg, step)
        elif log_type == 'hparams':
            self.tb.add_hparams(heading, msg)
        else:
            raise ValueError('Type of log `{}` is unknown'.format(log_type))

    def close(self):
        if self.rank == 0:
            self.tb.close()


class TrainLogger:
    """Logger for multi-process training with multiple log streams
    """
    def __init__(self, outstream:bool=True, fstream:bool=None, filename:str=None):
        """Init function for the logger

        Args:
            filename ([type]): [description]
            rank (int, optional): [description]. Defaults to 0.
        """
        self.std_active = outstream
        self.f_active   = fstream is not None
        if self.f_active:
            assert filename, 'file stream active but filename not specified'
            self.logfile = open(filename, 'w')


    def log(self, log_msg, stream:str):
        """Log

        Args:
            log_msg (dict): [description]
        """
        assert stream in ['stdout', 'file'], 'Stream not supported'
        if stream == 'stdout' and self.std_active:
            print(log_msg)
        if stream == 'file' and self.f_active:
            self.logfile.write(json.dumps(log_msg)+'\n')

    def close(self):
        if self.f_active:
            self.logfile.close()
