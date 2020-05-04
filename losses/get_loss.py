from .flow_loss import unFlowLoss

def get_loss(cfg):
    if cfg.data.loss_name == 'unflow':
        loss = unFlowLoss(cfg)
    else:
        raise NotImplementedError(cfg.data.loss_name)
    return loss
