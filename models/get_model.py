from .pwclite import PWCLite

def get_model(cfg):
    if cfg.data.model_name == 'pwclite':
        model = PWCLite(cfg)
    else:
        raise NotImplementedError(cfg.data.model_name)
    return model
