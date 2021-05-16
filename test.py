# -*- coding: UTF-8 -*-
"""
@Project ：srgan-esrgan-pytorch
@File    ：test.py
@IDE     ：PyCharm 
@Author  ：AmazingPeterZhu
@Date    ：2021/4/23 下午3:19 
"""
import yaml
import cv2
import torch
import os.path as osp

from models import SRResNet


# 读取配置文件
fs = open("options/srgan/test.yaml", encoding="UTF-8")
opt = yaml.load(fs, Loader=yaml.FullLoader)

# 读取模型
netG = SRResNet(opt['in_channels'], opt['ngf'], opt['num_blocks'])

# 加载数据
PATH = osp.join(opt['outroot'], opt['name'], 'models', 'netG.pth')
netG.load_state_dict(torch.load(PATH))

# 读取图片
img_n = cv2.imread("./LR.png")
img = torch.from_numpy(img_n)

# 测试
result = netG(img)
result_n = result.numpy()
cv2.imwrite('./results/001_SRGAN_x4_f64b16_DIV2K_1000k_B16G1/SRGAN_SR.png', result_n)

