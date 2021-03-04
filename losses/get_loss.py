from .losses import DefaultLoss, BaseLoss, SweepLoss

def get_loss(cfg, id):
    if cfg.data.loss_name == 'default':
        loss = DefaultLoss(cfg, id)
    elif cfg.data.loss_name == "base":
        loss = BaseLoss(cfg, id)
    elif cfg.data.loss_name == "sweep":
        loss = SweepLoss(cfg, id)
    else:
        raise NotImplementedError(cfg.data.loss_name)
    return loss
