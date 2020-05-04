import json
import pprint
import datetime
import argparse
from path import Path
from easydict import EasyDict

from logger import init_logger

import torch
from utils.torch_utils import init_seed

from models.get_model import get_model
from losses.get_loss import get_loss
from trainer.get_trainer import get_trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='configs/default.json')
    parser.add_argument('-e', '--evaluate', action='store_true')
    parser.add_argument('-m', '--model', default=None)

    args = parser.parse_args()

    with open(args.config) as f:
        cfg = EasyDict(json.load(f))

    if args.evaluate:
        raise("Not implemented")
        # cfg.train.update({
        #     'epochs': 1,
        #     'epoch_size': -1,
        #     'valid_size': 0,
        #     'workers': 1,
        #     'val_epoch_size': 1,
        # })

    if args.model is not None:
        cfg.train.pretrained_model = args.model

    # store files day by day
    exp_name = cfg.data.exp_name
    curr_time = datetime.datetime.now().strftime("%y%m%d%H%M%S")
    #cfg.save_root = Path('./outputs/checkpoints') / curr_time[:6] / curr_time[6:]
    cfg.save_root = Path('./outputs/checkpoints/') + exp_name
    cfg.save_root.makedirs_p()

    # init logger
    _log = init_logger(log_dir=cfg.save_root, filename=curr_time[6:] + '.log')
    _log.info('=> will save everything to {}'.format(cfg.save_root))

    # show configurations
    cfg_str = pprint.pformat(cfg)
    _log.info('=> configurations \n ' + cfg_str)

    # Train
    init_seed(cfg.seed)

    # Get Stuff
    model = get_model(cfg)
    loss = get_loss(cfg)
    trainer = get_trainer(cfg)(
        train_loader, valid_loader, model, loss, _log, cfg.save_root, cfg.train)

    # Train
    trainer.train()