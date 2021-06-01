# -*- coding: UTF-8 -*-
"""Implements SRGAN models: https://arxiv.org/abs/1809.00219
@Project ：srgan-esrgan-pytorch
@File    ：esrgan_models.py
@IDE     ：PyCharm 
@Author  ：AmazingPeterZhu
@Date    ：2021/5/12 下午1:11 
"""

import torch.nn as nn
import torch.nn.functional as F

import utils
from .archs import UpSampleBlock, RRDB, SiLU


# 定义生成器（SRRDBNet）
class SRRDBNet(nn.Module):
    def __init__(self, in_channels=3,
                 num_features=64,
                 num_blocks=23,
                 num_grow=32,
                 gaussian_noise=False):
        super(SRRDBNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, num_features, 3, 1, 1)
        self.body = utils.stack_block(block=RRDB,
                                      num_blocks=num_blocks,
                                      num_features=num_features,
                                      num_grow=num_grow,
                                      gaussian_noise=gaussian_noise)
        self.conv2 = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.usb1 = UpSampleBlock(num_features, 4*num_features)
        self.usb2 = UpSampleBlock(num_features, 4*num_features)
        self.conv3 = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_features, in_channels, 3, 1, 1)

        self.silu = SiLU(inplace=True)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.body(out1)
        out3 = self.conv2(out2) + out1
        out4 = self.usb2(self.usb1(out3))
        out = self.conv4(self.silu(self.conv3(out4)))
        return out


# 定义辨别器（RERaD）
class RERaDiscriminator(nn.Module):
    def __init__(self, in_channels, num_features):
        super(RERaDiscriminator, self).__init__()

        self.conv_1st = nn.Conv2d(in_channels, num_features, 3, 1)
        self.conv_2nd = nn.Conv2d(num_features, num_features, 3, 2)
        self.bn_1st = nn.BatchNorm2d(num_features)
        self.conv_3rd = nn.Conv2d(num_features, 2*num_features, 3, 1)
        self.bn_2nd = nn.BatchNorm2d(2*num_features)
        self.conv_4th = nn.Conv2d(2*num_features, 2*num_features, 3, 2)
        self.bn_3rd = nn.BatchNorm2d(2*num_features)
        self.conv_5th = nn.Conv2d(2*num_features, 3*num_features, 3, 1)
        self.bn_4th = nn.BatchNorm2d(3*num_features)
        self.conv_6th = nn.Conv2d(3*num_features, 3*num_features, 3, 2)
        self.bn_5th = nn.BatchNorm2d(3*num_features)
        self.conv_7th = nn.Conv2d(3*num_features, 4*num_features, 3, 1)
        self.bn_6th = nn.BatchNorm2d(4*num_features)
        self.conv_8th = nn.Conv2d(4*num_features, 4*num_features, 3, 2)
        self.bn_7th = nn.BatchNorm2d(4*num_features)
        self.conv_9th = nn.Conv2d(4*num_features, 5*num_features, 3, 1)
        self.bn_8th = nn.BatchNorm2d(5*num_features)
        self.conv_10th = nn.Conv2d(5*num_features, 5*num_features, 3, 2)
        self.bn_9th = nn.BatchNorm2d(5*num_features)
        self.conv_11th = nn.Conv2d(5*num_features, 6*num_features, 3, 1)
        self.bn_10th = nn.BatchNorm2d(6*num_features)
        self.conv_12th = nn.Conv2d(6*num_features, 6*num_features, 3, 2)
        self.bn_11th = nn.BatchNorm2d(6*num_features)
        self.conv_13th = nn.Conv2d(6*num_features, 7*num_features, 3, 1)
        self.bn_12th = nn.BatchNorm2d(7*num_features)
        self.conv_14th = nn.Conv2d(7*num_features, 7*num_features, 3, 2)
        self.bn_13th = nn.BatchNorm2d(7*num_features)
        self.conv_15th = nn.Conv2d(7*num_features, 8*num_features, 3, 1)
        self.bn_14th = nn.BatchNorm2d(8*num_features)
        self.conv_16th = nn.Conv2d(8*num_features, 8*num_features, 3, 2)
        self.bn_15th = nn.BatchNorm2d(8*num_features)
        # 全连接
        # self.dense_1st = nn.Linear(8*num_features, 1024)
        # self.dense_2nd = nn.Linear(1024, 1)

        # 全卷积
        self.conv_9th = nn.Conv2d(8*num_features, 1, 1, 1)

        # 激活函数
        self.silu = SiLU(inplace=True)
        # 注意！不能直接nn.LeakyReLU(model)

    def forward(self, x):
        x = self.silu(self.conv_1st(x))
        x = self.silu(self.bn_1st(self.conv_2nd(x)))
        x = self.silu(self.bn_2nd(self.conv_3rd(x)))
        x = self.silu(self.bn_3rd(self.conv_4th(x)))
        x = self.silu(self.bn_4th(self.conv_5th(x)))
        x = self.silu(self.bn_5th(self.conv_6th(x)))
        x = self.silu(self.bn_6th(self.conv_7th(x)))
        x = self.silu(self.bn_7th(self.conv_8th(x)))

        x = self.silu(self.bn_8st(self.conv_9nd(x)))
        x = self.silu(self.bn_9nd(self.conv_10rd(x)))
        x = self.silu(self.bn_10rd(self.conv_11th(x)))
        x = self.silu(self.bn_11th(self.conv_12th(x)))
        x = self.silu(self.bn_12th(self.conv_13th(x)))
        x = self.silu(self.bn_13th(self.conv_14th(x)))
        x = self.silu(self.bn_14th(self.conv_15th(x)))
        x = self.silu(self.bn_15th(self.conv_16th(x)))
        # 全连接
        # return nn.Sigmoid(self.dense_1st(self.lrelu(self.dense_2nd(x))))
        # 全卷积
        x = self.conv_9th(x)
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
















