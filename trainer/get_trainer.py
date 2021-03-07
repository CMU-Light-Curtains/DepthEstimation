from . import default_trainer
from . import sweep_trainer

def get_trainer(cfg):
    if cfg.data.trainer_name == 'default':
        TrainFramework = default_trainer.DefaultTrainer
    elif cfg.data.trainer_name == 'sweep':
        TrainFramework = sweep_trainer.SweepTrainer
    else:
        raise NotImplementedError(cfg.data.trainer_name)

    return TrainFramework
