import argparse
import sys
import os.path as osp
import os
import copy

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from time import time

from models import ESPCN
from datasets import TrainDataset, EvalDataset
from utils import AverageMeter, calc_psnr
from qsparse.fx import symbolic_trace
import cloudpickle


if __name__ == "__main__":
    start_time = time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=str, required=True)
    parser.add_argument("--eval-file", type=str, required=True)
    parser.add_argument("--outputs-dir", type=str, required=True)
    parser.add_argument("--weights-file", type=str)
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-epochs", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--compile", type=bool, action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--train-mode",
        help="different modes as in the experiments of the paper",
        default="float",
    )
    args = parser.parse_args()

    args.outputs_dir = os.path.join(args.outputs_dir, "x{}".format(args.scale))

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    logfile = open(osp.join(args.outputs_dir, "log.txt"), "w")

    class Unbuffered:
        def __init__(self, stream):
            self.stream = stream

        def write(self, data):
            self.stream.write(data)
            logfile.write(data)

        def flush(self):
            self.stream.flush()
            logfile.flush()

    sys.stdout = Unbuffered(sys.stdout)
    sys.stderr = Unbuffered(sys.stderr)

    cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)

    train_dataset = TrainDataset(args.train_file)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    eval_dataset = EvalDataset(args.eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    model = ESPCN(
        scale_factor=args.scale,
        train_mode=args.train_mode,
        epoch_size=len(train_dataloader),
        hardware_compat=args.compile,
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        [
            {"params": model.first_part.parameters()},
            {"params": model.last_part.parameters(), "lr": args.lr * 0.1},
        ],
        lr=args.lr,
    )

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0

    for epoch in range(args.num_epochs):
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr * (0.1 ** (epoch // int(args.num_epochs * 0.8)))

        model.train()
        epoch_losses = AverageMeter()

        for data in train_dataloader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            preds = model(inputs)

            loss = criterion(preds, labels)

            epoch_losses.update(loss.item(), len(inputs))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        torch.save(
            model.state_dict(),
            os.path.join(args.outputs_dir, "epoch_{}.pth".format(epoch)),
        )

        model.eval()
        epoch_psnr = AverageMeter()

        for data in eval_dataloader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                preds = model(inputs).clamp(0.0, 1.0)

            epoch_psnr.update(calc_psnr(preds, labels), len(inputs))

        print("epoch {} eval psnr: {:.2f}".format(epoch, epoch_psnr.avg))

        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())

    print("best epoch: {}, psnr: {:.2f}".format(best_epoch, best_psnr))
    torch.save(best_weights, os.path.join(args.outputs_dir, "best.pth"))
    if args.compile:
        torch.save(
            symbolic_trace(model),
            os.path.join(args.outputs_dir, "compiled.pkl"),
            pickle_module=cloudpickle,
        )
    end_time = time()
    print(f"total elapse {end_time-start_time:0.02f}s")
