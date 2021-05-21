# -*- coding: UTF-8 -*-
"""
@Project ：srgan-esrgan-pytorch
@File    ：archs.py
@IDE     ：PyCharm 
@Author  ：AmazingPeterZhu
@Date    ：2021/5/12 下午6:53 
"""
import torch
import torch.nn as nn


# 定义上采样块
class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.shuffle = nn.PixelShuffle(2)
        self.prelu = nn.PReLU()

    def forward(self, x):
        return self.prelu(self.shuffle(self.conv(x)))


# 添加高斯噪声
class GaussianNoise(nn.Module):
    def __init__(self, sigma=0.1, is_relative_detach=False):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0, dtype=torch.float).to(torch.device('cuda'))

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x


# 定义残差块
class ResidualBLock(nn.Module):
    def __init__(self, num_features=64):
        super(ResidualBLock, self).__init__()
        self.conv1 = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(num_features)
        self.bn2 = nn.BatchNorm2d(num_features)

        self.prelu = nn.PReLU()

    def forward(self, x):
        out1 = self.prelu(self.bn1(self.conv1(x)))
        out2 = self.bn2(self.conv2(out1)) + x
        return out2


# 定义残差稠密块
class ResidualDenseBlock(nn.Module):
    def __init__(self, num_features=64, num_grow=32, gaussian_noise=False):
        super(ResidualDenseBlock, self).__init__()
        self.gaussian_noise = gaussian_noise
        self.noise = GaussianNoise() if gaussian_noise else None
        self.conv1 = nn.Conv2d(num_features, num_grow, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_features+num_grow*1, num_grow, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_features+num_grow*2, num_grow, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_features+num_grow*3, num_grow, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_features+num_grow*4, num_features, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        y1 = self.lrelu(self.conv1(x))
        y2 = self.lrelu(self.conv2(torch.cat((x, y1), 1)))
        y3 = self.lrelu(self.conv3(torch.cat((x, y1, y2), 1)))
        y4 = self.lrelu(self.conv4(torch.cat((x, y1, y2, y3), 1)))
        y5 = self.conv5(torch.cat((x, y1, y2, y3, y4), 1))
        if self.gaussian_noise:
            return self.noise(y5 * 0.2 + x)
        else:
            return y5 * 0.2 + x


# 定义RRDB
class RRDB(nn.Module):
    def __init__(self, num_features=64, num_grow=32, gaussian_noise=False):
        super(RRDB, self).__init__()
        self.b1 = ResidualDenseBlock(num_features, num_grow, gaussian_noise)
        self.b2 = ResidualDenseBlock(num_features, num_grow, gaussian_noise)
        self.b3 = ResidualDenseBlock(num_features, num_grow, gaussian_noise)

    def forward(self, x):
        out = self.b1(x)
        out = self.b2(out)
        out = self.b3(out)
        return out * 0.2 + x
















