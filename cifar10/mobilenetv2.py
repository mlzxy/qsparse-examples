"""MobileNetV2 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
"""
import torch
import json
import torch.nn as nn
import torch.nn.functional as F
from qsparse import prune, quantize
from enum import Enum


def identity(v):
    return v


def create_p_q(train_mode, epoch_size):
    def bypass(*args):
        if len(args) == 0:
            return identity
        else:
            return args[0]

    quantize_first = train_mode.startswith("quantize")
    hp = [[110, 235], [100, 230], [130, 100]]
    if "late" in train_mode:
        # experiments for quantize-then-prune, but on the late stage of training

        if "prune" not in train_mode:
            print("just quantization")
            hp[0][0] = 240
            hp[1][0] = 230
        else:
            hp[0][0] = 170
            hp[1][0] = 160
            hp[2][0] = 180

    ws = 2048

    def q(*args, c=0):
        if "quantize" in train_mode:
            return (
                quantize(
                    timeout=epoch_size * (hp[0][0] if quantize_first else hp[0][1]),
                    channelwise=-1,
                    window_size=ws,
                )
                if len(args) == 0
                else quantize(
                    args[0],
                    timeout=epoch_size * (hp[1][0] if quantize_first else hp[1][1]),
                    window_size=1,
                    channelwise=c or 1,
                )
            )
        else:
            return bypass(*args)

    def p(*args):
        if "prune" in train_mode:
            kw = {
                "start": epoch_size * (hp[2][0] if quantize_first else hp[2][1]),  # 200
                "interval": epoch_size * 15,  # 5
                "repetition": 4,
                "sparsity": 0.5,
            }
            if "weight" in train_mode:
                return (
                    identity if len(args) == 0 else prune(args[0], **kw, window_size=1)
                )
            elif "feat" in train_mode:
                return prune(**kw, window_size=ws) if len(args) == 0 else args[0]
            elif "both" in train_mode:
                return prune(**kw, window_size=ws) if len(args) == 0 else prune(args[0], **kw, window_size=1)
        else:
            return bypass(*args)

    return p, q


class Block(nn.Module):
    """expand + depthwise + pointwise"""

    def __init__(
        self, in_planes, out_planes, expansion, stride, train_mode, epoch_size
    ):
        super(Block, self).__init__()
        self.stride = stride
        p, q = create_p_q(train_mode, epoch_size)

        planes = expansion * in_planes
        self.conv1 = q(
            p(
                nn.Conv2d(
                    in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False
                )
            )
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.p1, self.q1 = p(), q()

        self.conv2 = q(
            p(
                nn.Conv2d(
                    planes,
                    planes,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=planes,
                    bias=False,
                )
            )
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.p2, self.q2 = p(), q()

        self.conv3 = q(
            p(
                nn.Conv2d(
                    planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False
                )
            )
        )
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                q(
                    p(
                        nn.Conv2d(
                            in_planes,
                            out_planes,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=False,
                        )
                    )
                ),
                nn.BatchNorm2d(out_planes),
            )

        self.p3, self.q3 = p(), q()

    def forward(self, x):
        out = F.relu(self.q1(self.p1(self.bn1(self.conv1(x)))))
        out = F.relu(self.q2(self.p2(self.bn2(self.conv2(out)))))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride == 1 else out
        return self.q3(
            self.p3(out)
        )  # should put p3 on top of shortcut instead of here, which will make results a little bit better


class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [
        (1, 16, 1, 1),
        (6, 24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
        (6, 32, 3, 2),
        (6, 64, 4, 2),
        (6, 96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1),
    ]

    def __init__(
        self, num_classes=10, train_mode="float", epoch_size=-1
    ):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        p, q = create_p_q(train_mode, epoch_size)
        self.qin = q()
        self.conv1 = q(nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False))
        self.bn1 = nn.BatchNorm2d(32)
        self.p1, self.q1 = p(), q()

        self.layers = self._make_layers(
            in_planes=32, train_mode=train_mode, epoch_size=epoch_size
        )
        self.conv2 = q(
            p(nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False))
        )
        self.bn2 = nn.BatchNorm2d(1280)
        self.q2 = q()
        self.linear = q(nn.Linear(1280, num_classes), c=-1)

    def _make_layers(self, in_planes, train_mode, epoch_size):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1] * (num_blocks - 1)
            for stride in strides:
                layers.append(
                    Block(
                        in_planes, out_planes, expansion, stride, train_mode, epoch_size
                    )
                )
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.qin(x)
        out = F.relu(self.q1(self.p1(self.bn1(self.conv1(x)))))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        out = self.q2(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def test():
    net = MobileNetV2()
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print(y.size())

