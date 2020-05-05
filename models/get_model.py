from .pwclite import PWCLite
from .models import DefaultModel

def get_model(cfg):
    if cfg.data.model_name == 'default':
        model = DefaultModel(cfg)
    else:
        raise NotImplementedError(cfg.data.model_name)
    return model
