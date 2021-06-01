# -*- coding: UTF-8 -*-
"""
@Project ：srgan-esrgan-pytorch 
@File    ：test_esrgan.py
@IDE     ：PyCharm 
@Author  ：AmazingPeterZhu
@Date    ：2021/5/28 上午11:14 
"""
import os
import os.path as osp
import glob
import cv2
import numpy as np
import torch
from models import *
import yaml

# 读取配置文件
fs = open("options/esrgan/test.yaml", encoding="UTF-8")
opt = yaml.load(fs, Loader=yaml.FullLoader)

# 创建输出文件夹
# 文件夹名字为配置文件中设置的name
try:
    os.makedirs(osp.join('results', opt['name']))
except OSError:
    pass

model_path = os.path.join('experiments', opt['name'], 'models', 'netG_e129.pth') # models path
device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
# device = torch.device('cpu')

test_img_folder = 'LR_test_img/*'

# 网络参数
in_channels = opt['in_channels']
num_features = opt['ngf']
num_blocks = opt['num_blocks']
num_grow = opt['num_grow']
gaussian_noise = opt['gaussian_noise']


model = RRDBNet(in_channels, num_features, num_blocks, num_grow)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

print('Model path {:s}. \nTesting...'.format(model_path))

idx = 0
for path in glob.glob(test_img_folder):
    idx += 1
    base = osp.splitext(osp.basename(path))[0]
    print(idx, base)
    # read images
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    cv2.imwrite(os.path.join('results', opt['name'], '{:s}_rlt.png'.format(base)), output)
