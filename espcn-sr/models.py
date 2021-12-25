import math
import torch
from torch import nn
from qsparse import prune, quantize


def create_p_q(train_mode, epoch_size, hardware_compat=False):
    def bypass(*args):
        if len(args) == 0:
            return nn.Identity()
        else:
            return args[0]

    quantize_first = train_mode.startswith("quantize")
    ws = 64

    def q(*args, c=0):
        if hardware_compat:
            c = -1
            bias_bits = 12
        else:
            bias_bits = -1

        if "quantize" in train_mode:
            return (
                quantize(
                    timeout=epoch_size * (150 if quantize_first else 170),
                    channelwise=-1,
                    window_size=ws,
                )
                if len(args) == 0
                else quantize(
                    args[0],
                    timeout=epoch_size * (140 if quantize_first else 160),
                    window_size=1,
                    channelwise=c or 1,
                    bias_bits=bias_bits,
                )
            )
        else:
            return bypass(*args)

    def p(*args):
        if "prune" in train_mode:
            kw = {
                "start": epoch_size * (155 if quantize_first else 140),
                "interval": epoch_size * 5,
                "repetition": 4,
                "sparsity": 0.5,
            }
            if "weight" in train_mode:
                return (
                    nn.Identity()
                    if len(args) == 0
                    else prune(args[0], **kw, window_size=1)
                )
            elif "feat" in train_mode:
                return (
                    prune(**kw, window_size=ws, strict=False)
                    if len(args) == 0
                    else args[0]
                )
            elif "both" in train_mode:
                return (
                    prune(**kw, window_size=ws, strict=False)
                    if len(args) == 0
                    else prune(args[0], **kw, window_size=1)
                )
        else:
            return bypass(*args)

    return p, q


class ESPCN(nn.Module):
    def __init__(
        self,
        scale_factor,
        num_channels=1,
        train_mode="float",
        epoch_size=-1,
        hardware_compat=False,
    ):
        super(ESPCN, self).__init__()
        p, q = create_p_q(train_mode, epoch_size, hardware_compat)
        self.qin = q()
        self.first_part = nn.Sequential(
            q(nn.Conv2d(num_channels, 64, kernel_size=5, padding=5 // 2)),
            q(),
            nn.ReLU() if hardware_compat else nn.Tanh(),
            q(nn.Conv2d(64, 32, kernel_size=3, padding=3 // 2))
            if hardware_compat
            else q(p(nn.Conv2d(64, 32, kernel_size=3, padding=3 // 2))),
            p(),
            q(),
            nn.ReLU() if hardware_compat else nn.Tanh(),
        )
        self.last_part = nn.Sequential(
            q(
                nn.Conv2d(
                    32,
                    num_channels * (scale_factor ** 2),
                    kernel_size=3,
                    padding=3 // 2,
                )
            ),
            q(),
            nn.PixelShuffle(scale_factor),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.in_channels == 32:
                    nn.init.normal_(m.weight.data, mean=0.0, std=0.001)
                    nn.init.zeros_(m.bias.data)
                else:
                    nn.init.normal_(
                        m.weight.data,
                        mean=0.0,
                        std=math.sqrt(
                            2 / (m.out_channels * m.weight.data[0][0].numel())
                        ),
                    )
                    nn.init.zeros_(m.bias.data)

    def forward(self, x):
        x = self.qin(x)
        x = self.first_part(x)
        x = self.last_part(x)
        return x
