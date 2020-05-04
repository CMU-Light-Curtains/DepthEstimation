from . import sintel_trainer

def get_trainer(cfg):
    if cfg.trainer_name == 'Sintel':
        TrainFramework = sintel_trainer.TrainFramework
    else:
        raise NotImplementedError(cfg.trainer_name)

    return TrainFramework
