from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
from model.utils.config import cfg
from model.faster_rcnn.faster_rcnn import _fasterRCNN
from model.faster_rcnn.qsparse_helper import create_p_q

from qsparse.sparse import PruneLayer
from qsparse.quantize import QuantizeLayer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.utils.model_zoo as model_zoo

import pdb

__all__ = ["ResNet", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]


model_urls = {
    "resnet18": "https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth",
    "resnet34": "https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth",
    "resnet50": "https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth",
    "resnet101": "https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth",
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(BasicBlock, self).__init__()
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         out += residual
#         out = self.relu(out)

#         return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        train_mode="float",
        epoch_size=-1,
    ):
        super(Bottleneck, self).__init__()
        p, q = create_p_q(train_mode, epoch_size)
        self.conv1 = p(
            q(nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False))
        )  # change
        self.bn1 = nn.BatchNorm2d(planes)
        self.p1, self.q1 = p(), q()

        self.conv2 = p(
            q(
                nn.Conv2d(
                    planes,
                    planes,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,  # change
                )
            )
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.p2, self.q2 = p(), q()

        self.conv3 = p(q(nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)))
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.q1(self.p1(out))
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.q2(self.p2(out))
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, train_mode, epoch_size, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        p, q = create_p_q(train_mode, epoch_size)

        self.conv1 = q(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=0, ceil_mode=True
        )  # change
        self.layer1 = self._make_layer(
            block, 64, layers[0], train_mode=train_mode, epoch_size=epoch_size
        )
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=2,
            train_mode=train_mode,
            epoch_size=epoch_size,
        )
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=2,
            train_mode=train_mode,
            epoch_size=epoch_size,
        )
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=2,
            train_mode=train_mode,
            epoch_size=epoch_size,
        )

        # it is slightly better whereas slower to set stride = 1
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=1)

        # we don't care about following layers since they are not used
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(
        self, block, planes, blocks, stride=1, train_mode="float", epoch_size=-1
    ):
        assert blocks > 1
        downsample = None
        p, q = create_p_q(train_mode, epoch_size)
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                q(
                    p(
                        nn.Conv2d(
                            self.inplanes,
                            planes * block.expansion,
                            kernel_size=1,
                            stride=stride,
                            bias=False,
                        )
                    )
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                train_mode=train_mode,
                epoch_size=epoch_size,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes, planes, train_mode=train_mode, epoch_size=epoch_size
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


# def resnet18(pretrained=False):
#     """Constructs a ResNet-18 model.
#     Args:
#       pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(BasicBlock, [2, 2, 2, 2])
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls["resnet18"]))
#     return model


# def resnet34(pretrained=False):
#     """Constructs a ResNet-34 model.
#     Args:
#       pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(BasicBlock, [3, 4, 6, 3])
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls["resnet34"]))
#     return model


def resnet50(pretrained=False):
    """Constructs a ResNet-50 model.
    Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet50"]))
    return model


def resnet101(pretrained, train_model, epoch_size):
    """Constructs a ResNet-101 model.
    Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], train_model, epoch_size)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet101"]))
    return model


def resnet152(pretrained=False):
    """Constructs a ResNet-152 model.
    Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet152"]))
    return model


class resnet(_fasterRCNN):
    def __init__(
        self,
        classes,
        num_layers=101,
        pretrained=False,
        class_agnostic=False,
        train_mode="float",
        epoch_size=-1,
    ):
        # /content/drive/MyDrive/PretrainedModels/
        self.model_path = osp.join(cfg.DATA_DIR, "resnet101_caffe.pth")
        self.dout_base_model = 1024
        self.pretrained = pretrained
        self.class_agnostic = class_agnostic
        self.train_mode = train_mode
        self.epoch_size = epoch_size

        _fasterRCNN.__init__(self, classes, class_agnostic, train_mode, epoch_size)

    def _init_modules(self):
        resnet = resnet101(False, self.train_mode, self.epoch_size)
        p, q = create_p_q(self.train_mode, self.epoch_size)

        if self.pretrained == True:
            print("Loading pretrained weights from %s" % (self.model_path))
            state_dict = torch.load(self.model_path)
            resnet.load_state_dict(
                {k: v for k, v in state_dict.items() if k in resnet.state_dict()},
                strict=False,
            )

        # part-1
        self.RCNN_base = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            p(),
            q(),
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            p(),
            q(),
            resnet.layer2,
            p(),
            q(),
            resnet.layer3,
            q(),
        )

        # part-2
        self.RCNN_top = nn.Sequential(resnet.layer4)

        # part-3
        self.RCNN_cls_score = q(nn.Linear(2048, self.n_classes), c=-1)
        # part-4
        if self.class_agnostic:
            self.RCNN_bbox_pred = q(nn.Linear(2048, 4), c=-1)
        else:
            self.RCNN_bbox_pred = q(nn.Linear(2048, 4 * self.n_classes), c=-1)
        # part-5: RCNN_rpn

        # Fix blocks
        for p in self.RCNN_base[0].parameters():
            p.requires_grad = False
        for p in self.RCNN_base[1].parameters():
            p.requires_grad = False

        assert 0 <= cfg.RESNET.FIXED_BLOCKS < 4
        if cfg.RESNET.FIXED_BLOCKS >= 3:
            for p in self.RCNN_base[6].parameters():
                p.requires_grad = False
        if cfg.RESNET.FIXED_BLOCKS >= 2:
            for p in self.RCNN_base[5].parameters():
                p.requires_grad = False
        if cfg.RESNET.FIXED_BLOCKS >= 1:
            for p in self.RCNN_base[4].parameters():
                p.requires_grad = False

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find("BatchNorm") != -1:
                for p in m.parameters():
                    p.requires_grad = False

        self.RCNN_base.apply(set_bn_fix)
        self.RCNN_top.apply(set_bn_fix)

    def train(self, mode=True):
        # Override train so that the training mode is set as we want
        nn.Module.train(self, mode)
        if mode:
            # Set fixed blocks to be in eval mode (we only prune/quantize in trainable layers)
            self.RCNN_base.eval()
            self.RCNN_base[5].train()
            self.RCNN_base[6].train()

            for name, mod in self.named_modules():
                if isinstance(mod, (PruneLayer, QuantizeLayer)):
                    mod.name = name  # for debug purpose

            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find("BatchNorm") != -1:
                    m.eval()

            self.RCNN_base.apply(set_bn_eval)
            self.RCNN_top.apply(set_bn_eval)

    def _head_to_tail(self, pool5):
        fc7 = self.RCNN_top(pool5).mean(3).mean(2)
        return fc7
