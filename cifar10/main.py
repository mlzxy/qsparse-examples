"""Train CIFAR10 with PyTorch."""
import torch
import torch.nn as nn
from time import time
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import sys
from glob import glob
import os.path as osp
import argparse

from mobilenetv2 import MobileNetV2
from utils import progress_bar

global_start_time = time()

parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--checkpoint-dir", help="directory to save models", default=".")
parser.add_argument(
    "--data-root",
    help="dataset location",
    default="data",
)

parser.add_argument(
    "--train-mode",
    help="different modes as in the experiments of the paper",
    default="float",
)
args = parser.parse_args()

epochs = args.epochs





if not osp.exists(args.checkpoint_dir):
    os.makedirs(args.checkpoint_dir)

logfile = open(osp.join(args.checkpoint_dir, "log.txt"), "w")


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

device = "cuda" if torch.cuda.is_available() else "cpu"
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print("==> Preparing data..")
transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

trainset = torchvision.datasets.CIFAR10(
    root=args.data_root, train=True, download=True, transform=transform_train
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2
)

testset = torchvision.datasets.CIFAR10(
    root=args.data_root, train=False, download=True, transform=transform_test
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2
)

classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

# Model
print("==> Building model..")
net = MobileNetV2(
    train_mode=args.train_mode,
    epoch_size=len(trainloader)
)

has_data_parallel = False
# Load checkpoint.
if len(glob(osp.join(args.checkpoint_dir, "*.pth"))) > 0:
    print("==> Resuming from checkpoint..")
    # assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found!"
    ckpt_path = sorted(glob(osp.join(args.checkpoint_dir, "*.pth")))[-1]
    print(f"loading from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
    with torch.no_grad():
        for inputs, targets in testloader:
            break
        outputs = net(inputs)
    try:
        net.load_state_dict(checkpoint["net"])
    except:
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
        net.load_state_dict(checkpoint["net"])
        has_data_parallel = True
    # net.load_state_dict(checkpoint["net"])
    best_acc = checkpoint["acc"]
    start_epoch = checkpoint["epoch"]

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

if device == "cuda" and not has_data_parallel:
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

net = net.to(device)


if start_epoch > 0:
    print("calibrating learning rate")
    for i in range(start_epoch):
        scheduler.step()


# Training
def train(epoch):
    # print("\nEpoch: %d" % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # progress_bar(
        #     batch_idx,
        #     len(trainloader),
        #     "Loss: %.3f | Acc: %.3f%% (%d/%d)"
        #     % (train_loss / (batch_idx + 1), 100.0 * correct / total, correct, total),
        # )


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(
            #     batch_idx,
            #     len(testloader),
            #     "Loss: %.3f | Acc: %.3f%% (%d/%d)"
            #     % (
            #         test_loss / (batch_idx + 1),
            #         100.0 * correct / total,
            #         correct,
            #         total,
            #     ),
            # )

    # Save checkpoint.
    acc = 100.0 * correct / total
    print(f"Epoch-{epoch} Validation Acc [{acc}], Best Vac [{best_acc}]")
    state = {
        "net": net.module.state_dict(),
        "acc": acc,
        "epoch": epoch,
    }
    # torch.save(state, osp.join(args.checkpoint_dir, f"ckpt-{str(epoch).zfill(4)}.pth"))
    if acc > best_acc:
        best_acc = acc


for epoch in range(start_epoch, epochs):
    train(epoch)
    test(epoch)
    scheduler.step()


global_end_time = time()
print(
    f"Total elapse {global_end_time - global_start_time:.02f} seconds for the entire training"
)