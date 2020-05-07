from .losses import DefaultLoss, BaseLoss

def get_loss(cfg):
    if cfg.data.loss_name == 'default':
        loss = DefaultLoss(cfg)
    elif cfg.data.loss_name == "base":
        loss = BaseLoss(cfg)
    else:
        raise NotImplementedError(cfg.data.loss_name)
    return loss
