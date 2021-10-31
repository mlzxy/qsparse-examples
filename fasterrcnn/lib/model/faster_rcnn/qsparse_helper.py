from qsparse import prune, quantize, structured_prune_callback
import torch.nn as nn
from enum import Enum


def create_p_q(train_mode, epoch_size):
    def bypass(*args):
        if len(args) == 0:
            return nn.Identity()
        else:
            return args[0]

    quantize_first = train_mode.startswith("quantize")
    ws = 32 

    hp = [[5.5, 7.5], [5.2, 7.2], [5.7, 3]]

    if "late" in train_mode:
        # experiments for quantize-then-prune, but on the late stage of training

        if "prune" not in train_mode:
            print("just quantization")
            hp[0][0] = 7.5
            hp[1][0] = 7.2
        else:
            hp[0][0] = 6.8
            hp[1][0] = 6.5
            hp[2][0] = 6.9

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
                "start": epoch_size * (hp[2][0] if quantize_first else hp[2][1]),
                "interval": epoch_size * 0.5,
                "repetition": 4,  # 3
                "sparsity": 0.5,
            }
            if "weight" in train_mode:
                return (
                    nn.Identity()
                    if len(args) == 0
                    else prune(args[0], **kw, window_size=1)
                )
            elif "feat" in train_mode:
                # since we already prune activations on the output channel dimension, activation pruning is equivalent to activation+weight pruning
                return (
                    prune(**kw, window_size=ws, callback=structured_prune_callback)
                    if len(args) == 0
                    else args[0]
                )
        else:
            return bypass(*args)

    return p, q
