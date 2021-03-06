# -*- coding: UTF-8 -*-
"""Implements SRGAN models: https://arxiv.org/abs/1609.04802
@Project ：srgan-esrgan-pytorch
@File    ：model.py
@IDE     ：PyCharm 
@Author  ：AmazingPeterZhu
@Date    ：2021/4/23 上午9:42 
"""

import torch.nn as nn
import torch.nn.functional as F

import utils
from .archs import UpSampleBlock, ResidualBLock


# 定义生成器（SRResNet）
class SRResNet(nn.Module):
    def __init__(self, in_channels, num_features, num_blocks=16):
        super(SRResNet, self).__init__()
        
        self.conv_1st = nn.Conv2d(in_channels, num_features, 9, 1, 4)
        self.res_body = utils.stack_block(ResidualBLock, num_blocks)
        self.conv_2nd = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.bn_1st = nn.BatchNorm2d(num_features)
        self.usb_1st = UpSampleBlock(num_features, 4*num_features)
        self.usb_2nd = UpSampleBlock(num_features, 4*num_features)
        self.conv_end = nn.Conv2d(num_features, in_channels, 9, 1, 4)

        # 激活函数
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # 注意！不能直接nn.LeakyReLU(model)

    def forward(self, x):
        out1 = self.lrelu(self.conv_1st(x))
        out2 = self.res_body(out1)
        out2 = self.bn_1st(self.conv_2nd(out2))+out1
        return self.conv_end(self.usb_2nd(self.usb_1st(out2)))


# 定义辨别器
class Discriminator(nn.Module):
    def __init__(self, in_channels, num_features):
        super(Discriminator, self).__init__()

        self.conv_1st = nn.Conv2d(in_channels, num_features, 3, 1)
        self.conv_2nd = nn.Conv2d(num_features, num_features, 3, 2)
        self.bn_1st = nn.BatchNorm2d(num_features)
        self.conv_3rd = nn.Conv2d(num_features, 2*num_features, 3, 1)
        self.bn_2nd = nn.BatchNorm2d(2*num_features)
        self.conv_4th = nn.Conv2d(2*num_features, 2*num_features, 3, 2)
        self.bn_3rd = nn.BatchNorm2d(2*num_features)
        self.conv_5th = nn.Conv2d(2*num_features, 4*num_features, 3, 1)
        self.bn_4th = nn.BatchNorm2d(4*num_features)
        self.conv_6th = nn.Conv2d(4*num_features, 4*num_features, 3, 2)
        self.bn_5th = nn.BatchNorm2d(4*num_features)
        self.conv_7th = nn.Conv2d(4*num_features, 8*num_features, 3, 1)
        self.bn_6th = nn.BatchNorm2d(8*num_features)
        self.conv_8th = nn.Conv2d(8*num_features, 8*num_features, 3, 2)
        self.bn_7th = nn.BatchNorm2d(8*num_features)
        # 全连接
        # self.dense_1st = nn.Linear(8*num_features, 1024)
        # self.dense_2nd = nn.Linear(1024, 1)

        # 全卷积
        self.conv_9th = nn.Conv2d(8*num_features, 1, 1, 1)

        # 激活函数
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # 注意！不能直接nn.LeakyReLU(model)

    def forward(self, x):
        x = self.lrelu(self.conv_1st(x))
        x = self.lrelu(self.bn_1st(self.conv_2nd(x)))
        x = self.lrelu(self.bn_2nd(self.conv_3rd(x)))
        x = self.lrelu(self.bn_3rd(self.conv_4th(x)))
        x = self.lrelu(self.bn_4th(self.conv_5th(x)))
        x = self.lrelu(self.bn_5th(self.conv_6th(x)))
        x = self.lrelu(self.bn_6th(self.conv_7th(x)))
        x = self.lrelu(self.bn_7th(self.conv_8th(x)))
        # 全连接
        # return nn.Sigmoid(self.dense_1st(self.lrelu(self.dense_2nd(x))))
        # 全卷积
        x = self.conv_9th(x)
        return F.sigmoid(F.avg_pool2d(x, x.size()[2:])).view(x.size()[0], -1)