import time

import torch
from torch.utils.tensorboard import SummaryWriter

from .util import AverageMeter

def train_one_epoch (
    train_loader,
    model,
    optimizer,
    scheduler,
    curr_epoch,
    device,
    scaler=None,
    tb_writer: SummaryWriter=None,
    print_freq=10,
):
    """Training the model for one epoch"""
    # set up meters
    batch_time = AverageMeter()
    losses_tracker = {}
    # number of iterations per epoch
    num_iters = len(train_loader)
    # switch to train mode
    model.train()

    # main training loop
    print("\n[Train]: Epoch {:d} started".format(curr_epoch))
    start = time.time()
    for iter_idx, (imgs, targets) in enumerate(train_loader, 0):
        imgs.to(device)
        targets.to(device)

        optimizer.zero_grad()
        if scaler is not None:
            # mixed precision training
            # forward / backward the model
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                losses = model(imgs, targets)
            scaler.scale(losses["final_loss"]).backward()
            # step optimizer / scheduler
            scaler.step(optimizer)
            scheduler.step()
            # update the scaler
            scaler.update()
        else:
            # forward / backward the model
            losses = model(imgs, targets)
            losses["final_loss"].backward()
            # step optimizer / scheduler
            optimizer.step()
            scheduler.step()

        # printing (only check the stats when necessary to avoid extra cost)
        if (iter_idx != 0) and (iter_idx % print_freq) == 0:
            # measure elapsed time (sync all kernels)
            torch.cuda.synchronize()
            batch_time.update((time.time() - start) / print_freq)
            start = time.time()

            # track all losses
            for key, value in losses.items():
                # init meter if necessary
                if key not in losses_tracker:
                    losses_tracker[key] = AverageMeter()
                # update
                losses_tracker[key].update(value.item())

            # log to tensorboard
            lr = scheduler.get_last_lr()[0]
            global_step = curr_epoch * num_iters + iter_idx
            if tb_writer is not None:
                # learning rate (after stepping)
                tb_writer.add_scalar("train/learning_rate", lr, global_step)
                # all losses
                tag_dict = {}
                for key, value in losses_tracker.items():
                    if key != "final_loss":
                        tag_dict[key] = value.val
                tb_writer.add_scalars("train/all_losses", tag_dict, global_step)
                # final loss
                tb_writer.add_scalar(
                    "train/final_loss", losses_tracker["final_loss"].val, global_step
                )

            # print to terminal
            block1 = "Epoch: [{:03d}][{:05d}/{:05d}]".format(
                curr_epoch, iter_idx, num_iters
            )
            block2 = "Time {:.2f} ({:.2f})".format(batch_time.val, batch_time.avg)
            block3 = "Loss {:.2f} ({:.2f})\n".format(
                losses_tracker["final_loss"].val, losses_tracker["final_loss"].avg
            )
            block4 = ""
            for key, value in losses_tracker.items():
                if key != "final_loss":
                    block4 += "\t{:s} {:.2f} ({:.2f})".format(key, value.val, value.avg)

            print("\t".join([block1, block2, block3, block4]))

    # finish up and print
    lr = scheduler.get_last_lr()[0]
    print("[Train]: Epoch {:d} finished with lr={:.8f}\n".format(curr_epoch, lr))
    return

# def evaluate(
#         val_loader,
#         model,
#         output_file,
#         gt_file,
#         device,
#         print_freq = 10,
# ):
#     """Test the model on the validation set"""
#     # an output file will be used to save all results
#     assert output_file is not None
#
#     # set up meters
#     batch_time = AverageMeter()
#     # switch to evaluate mode
#     model.eval()
#     cpu_device = torch.device("cpu")
#
#     # loop over validation set
#     start = time.time()
#     det_results = []
#     for iter_idx, data in enumerate(val_loader, 0):
#         imgs, targets = data
#         imgs_device = list(img.to(device) for img in imgs)
#         with torch.no_grad()
#             outputs = model(imgs_device, None)
#
#         # unpack the results
#         outputs = []