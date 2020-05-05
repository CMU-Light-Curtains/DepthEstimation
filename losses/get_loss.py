from .losses import DefaultLoss

def get_loss(cfg):
    if cfg.data.loss_name == 'default':
        loss = DefaultLoss(cfg)
    else:
        raise NotImplementedError(cfg.data.loss_name)
    return loss
