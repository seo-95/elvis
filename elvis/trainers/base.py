from abc import abstractclassmethod


class BaseTrainer(object):

    def __init__(self) -> None:
        pass

    @abstractclassmethod
    def train(self):
        pass

    def train_loop(self):
        pass

    def compute_batch_loss(self):
        pass

    @abstractclassmethod
    def predict(self):
        pass

    @abstractclassmethod
    def save_checkpoint(self):
        pass

    #HOOKS TO CUSTOMIZE TRAINING
    def start_train_hook(self): pass
    def end_train_hook(self): pass
    def before_train_step_hook(self): pass
    def after_train_step_hook(self): pass
    def before_eval_step_hook(self, dev_step:int): pass
    def after_eval_step_hook(self, dev_step:int): pass
    def start_epoch_hook(self): pass
    def end_epoch_hook(self): pass

