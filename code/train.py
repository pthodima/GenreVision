import argparse
import os
import time
import datetime
import torch

import torch.backends.cudnn as cudnn

from pprint import pprint
from torch.utils.tensorboard import SummaryWriter

from libs import (
    load_config,
    build_dataset,
    build_dataloader,
    MovieClassifier,
    build_optimizer,
    build_scheduler,
)

########################################################################################
def main(args):
    """main function to handle training"""

    """1. Load config / setup folders"""
    # parse args
    args.start_epoch = 0
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    pprint(cfg)

    # prep for output folder (based on time stamp)
    if not os.path.exists(cfg["output_folder"]):
        os.mkdir(cfg["output_folder"])
    cfg_filename = os.path.basename(args.config).replace(".yaml", "")
    if len(args.output) == 0:
        ts = datetime.datetime.fromtimestamp(int(time.time()))
        ckpt_folder = os.path.join(cfg["output_folder"], cfg_filename + "_" + str(ts))
    else:
        ckpt_folder = os.path.join(
            cfg["output_folder"], cfg_filename + "_" + str(args.output)
        )
    if not os.path.exists(ckpt_folder):
        os.mkdir(ckpt_folder)
    # tensorboard writer
    tb_writer = SummaryWriter(os.path.join(ckpt_folder, "logs"))

    """2. Create dataset / dataloader"""
    train_dataset = build_dataset(
        cfg["dataset"]["name"],
        cfg["dataset"]["train"],
        cfg["dataset"]["img_folder"],
        cfg["dataset"]["ann_folder"],
    )
    # data loaders
    train_loader = build_dataloader(train_dataset, True, **cfg["loader"])

    """3. create model, optimizer, and scheduler"""
    # model
    model = MovieClassifier(**cfg["model"]).to(torch.device(cfg["devices"][0]))
    # optimizer
    optimizer = build_optimizer(model, cfg["opt"])
    # schedule
    num_iters_per_epoch = len(train_loader)
    scheduler = build_scheduler(optimizer, cfg["opt"], num_iters_per_epoch)
    # also disable cudnn benchmark, as the input size varies during training
    cudnn.benchmark = False


########################################################################################
if __name__ == "__main__":
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
        description="Train a model for movie genre classification from its poster"
    )
    parser.add_argument("config", metavar="DIR", help="path to a config file")
    parser.add_argument(
        "-p",
        "--print-freq",
        default=10,
        type=int,
        help="print frequency (default: 10 iterations)",
    )
    parser.add_argument(
        "-c",
        "--ckpt-freq",
        default=1,
        type=int,
        help="checkpoint frequency (default: every 1 epoch)",
    )
    parser.add_argument(
        "--output", default="", type=str, help="name of exp folder (default: none)"
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to a checkpoint (default: none)",
    )
    args = parser.parse_args()
    main(args)

