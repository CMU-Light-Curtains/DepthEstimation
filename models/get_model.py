from .pwclite import PWCLite
from .models import DefaultModel, BaseModel
from .packnet import PacknetModel

def get_model(cfg):
    if cfg.data.model_name == 'default':
        model = DefaultModel(cfg)
    elif cfg.data.model_name == 'base':
        model = BaseModel(cfg)
    elif cfg.data.model_name == 'packnet':
        model = PacknetModel(cfg)
    else:
        raise NotImplementedError(cfg.data.model_name)
    return model
