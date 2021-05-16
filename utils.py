# -*- coding: UTF-8 -*-
"""
@Project ：srgan-esrgan-pytorch
@File    ：utils.py
@IDE     ：PyCharm
@Author  ：AmazingPeterZhu
@Date    ：2021/4/23 上午11:24 
"""
import torch.nn as nn
import torch
import numpy as np
from tqdm import trange


def stack_block(block, num_blocks, **kwarg):
    """
    将相同的层或者块堆到一起
    :param block:nn.module，要堆叠的层或者块
    :param num_blocks:int，堆叠的数量
    :param kwarg:
    :return:nn.sequential
    """
    blocks = []
    for _ in range(num_blocks):
        blocks.append(block(**kwarg))
    return nn.Sequential(*blocks)


class BicubicInterpolation:
    """
    双三次插值算法
    :param img: numpy数组，可以由opencv读取
    :param w: int,图片宽度，单位像素
    :param h: int,图片长度，单位像素
    """
    def __init__(self, img, w, h):
        self.img = img
        self.h = h
        self.w = w

    def S(self, x):
        x = np.abs(x)

        if 0 <= x < 1:
            return 1 - 2 * x * x + x * x * x
        if 1 <= x < 2:
            return 4 - 8 * x + 5 * x * x - x * x * x
        else:
            return 0

    def interpolate(self):
        height, width, channels = self.img.shape
        emptyImage = np.zeros((self.h, self.w, channels), np.uint8)
        sh = self.h / height
        sw = self.w / width
        for i in trange(self.h):
            for j in range(self.w):
                x = i / sh
                y = j / sw
                p = (i + 0.0) / sh - x
                q = (j + 0.0) / sw - y
                x = int(x) - 2
                y = int(y) - 2
                A = np.array([
                    [self.S(1 + p), self.S(p), self.S(1 - p), self.S(2 - p)]
                ])
                if x >= self.h - 3:
                    self.h - 1
                if y >= self.w - 3:
                    self.w - 1
                if 1 <= x <= (self.h - 3) and 1 <= y <= (self.w - 3):
                    B = np.array([
                        [self.img[x - 1, y - 1], self.img[x - 1, y],
                         self.img[x - 1, y + 1],
                         self.img[x - 1, y + 1]],
                        [self.img[x, y - 1], self.img[x, y],
                         self.img[x, y + 1], self.img[x, y + 2]],
                        [self.img[x + 1, y - 1], self.img[x + 1, y],
                         self.img[x + 1, y + 1], self.img[x + 1, y + 2]],
                        [self.img[x + 2, y - 1], self.img[x + 2, y],
                         self.img[x + 2, y + 1], self.img[x + 2, y + 1]],

                    ])
                    C = np.array([
                        [self.S(1 + q)],
                        [self.S(q)],
                        [self.S(1 - q)],
                        [self.S(2 - q)]
                    ])
                    blue = np.dot(np.dot(A, B[:, :, 0]), C)[0, 0]
                    green = np.dot(np.dot(A, B[:, :, 1]), C)[0, 0]
                    red = np.dot(np.dot(A, B[:, :, 2]), C)[0, 0]

                    # adjust the value to be in [0,255]
                    def adjust(value):
                        if value > 255:
                            value = 255
                        elif value < 0:
                            value = 0
                        return value

                    blue = adjust(blue)
                    green = adjust(green)
                    red = adjust(red)
                    emptyImage[i, j] = np.array([blue, green, red], dtype=np.uint8)

        return emptyImage


@torch.no_grad()
def init_weights_mini(m, scale=0.1):
    """
    权重初始化
    :param scale: Float, 初始权重放大或者缩小系数
    :param m: Module
    :return: Module, self
    """
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
        m.weight.data *= scale
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
        m.weight.data *= scale
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


@torch.no_grad()
def init_weights(m, scale=1):
    """
    权重初始化
    :param scale (Float): Float, 初始权重放大或者缩小系数
    :param m (Module): Module
    :return: Module, self
    """
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
        m.weight.data *= scale
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
        m.weight.data *= scale
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class FeatureExtractor(nn.Module):
    """
    特征提取器，截取模型从前往后数特定的层
    :param model(torch.nn.Module): nn.Module
    :param feature_layer(int): default:12
    """
    def __init__(self, model, feature_layer=12):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(model.features.children())[:feature_layer]).eval()

    def forward(self, x):
        return self.features(x)
