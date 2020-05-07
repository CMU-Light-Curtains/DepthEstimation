import os
import json
import pprint
import datetime
import argparse
from path import Path
from easydict import EasyDict

from logger import init_logger

import torch
from utils.torch_utils import init_seed
import torch.multiprocessing as mp
import torch.distributed as dist

from models.get_model import get_model
from losses.get_loss import get_loss
from trainer.get_trainer import get_trainer

def worker(id, args): pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='configs/default.json')
    parser.add_argument('-e', '--evaluate', action='store_true')
    parser.add_argument('-m', '--model', default=None)
    parser.add_argument('--viz', action='store_true', help='viz', default=False)

    args = parser.parse_args()

    with open(args.config) as f:
        cfg = EasyDict(json.load(f))

    if args.viz:
        cfg.var.viz = True

    if args.model is not None:
        cfg.train.pretrained_model = args.model

    init_seed(cfg.seed)

    # store files day by day
    exp_name = cfg.data.exp_name
    curr_time = datetime.datetime.now().strftime("%y%m%d%H%M%S")
    cfg.save_root = Path('./outputs/checkpoints/') + exp_name
    cfg.save_root.makedirs_p()

    # Multiprocessing (Number of Workers = Number of GPU)
    if cfg.mp.enabled:
        # Checks
        if cfg.train.n_gpu > 0:
            if cfg.train.n_gpu > torch.cuda.device_count():
                raise RuntimeError("Total GPU size is incorrect")
            cfg.mp.workers = cfg.train.n_gpu
        else:
            if cfg.mp.workers <= 0:
                raise RuntimeError("Wrong number of workers")

        # Set Flags
        os.environ["MASTER_ADDR"] = cfg.mp.master_addr
        os.environ["MASTER_PORT"] = str(cfg.mp.master_port)
        os.environ["WORLD_SIZE"] = str(cfg.mp.workers)
        os.environ["RANK"] = str(0)
        shared = torch.zeros((cfg.mp.workers, 10)).share_memory_()

        # Spawn Worker
        mp.spawn(worker, nprocs=cfg.mp.workers, args=(cfg, shared))
    else:
        # Spawn Worker
        shared = torch.zeros((1, 10)).share_memory_()
        worker(0, cfg, shared)


def worker(id, cfg, shared):
    # init logger
    curr_time = datetime.datetime.now().strftime("%y%m%d%H%M%S")
    _log = init_logger(log_dir=cfg.save_root, filename=curr_time[6:] + '.log')
    if id == 0: _log.info(id, '=> will save everything to {}'.format(cfg.save_root))

    # show configurations
    cfg_str = pprint.pformat(cfg)
    if id == 0: _log.info(id, '=> configurations \n ' + cfg_str)

    # Distributed
    if cfg.mp.enabled:
        if cfg.train.n_gpu > 0:
            dist.init_process_group(backend="nccl", init_method="env://",
                                    world_size=cfg.mp.workers, rank=id)
        else:
            dist.init_process_group(backend="gloo", init_method="env://",
                                    world_size=cfg.mp.workers, rank=id)

    # Get Model and Loss
    model = get_model(cfg)
    loss = get_loss(cfg)

    # Create Trainer
    trainer = get_trainer(cfg)(id, model, loss, _log, cfg.save_root, cfg, shared)
    trainer.train()

    # Destroy
    if cfg.mp.enabled:
        dist.destroy_process_group()


    pass

if __name__ == '__main__':
    main()