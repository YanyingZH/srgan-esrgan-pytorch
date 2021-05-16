# -*- coding: UTF-8 -*-
"""
@Project ：srgan-esrgan-pytorch
@File    ：datasets.py
@IDE     ：PyCharm 
@Author  ：AmazingPeterZhu
@Date    ：2021/5/13 下午2:26 
"""
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from random import random
from transforms import paired_random_crop
import numpy as np


class DIV2KDataset(Dataset):
    def __init__(self,
                 rootGT,
                 rootLQ,
                 RandomCrop=False,
                 RandomHorizontalFlip=False,
                 RandomRotation90Degree=False,
                 ToTensor=True,
                 crop_size=128
                 ):
        """
        适用于DIV2K，需要预先Crop to sub-images，运行脚本 scripts/extract_subimages.py
        :param rootGT: str, 高分辨率文件目录
        :param rootLQ: str, 低分辨率文件目录
        :param RandomCrop: bool, 是否进行随机裁剪
        :param RandomHorizontalFlip: bool, 是否进行随机水平变换
        :param RandomRotation90Degree: bool, 是否进行随机旋转90度
        :param ToTensor: bool, 是否进行随机裁剪
        :param crop_size: int, 随机裁剪尺寸
        """

        image_files_GT = os.listdir(rootGT)
        image_files_LQ = os.listdir(rootLQ)
        self.rootGT = rootGT
        self.rootLQ = rootLQ
        self.image_files_GT = sorted(image_files_GT)
        self.image_files_LQ = sorted(image_files_LQ)
        # transform 配置
        self.RandomCrop = RandomCrop
        self.RandomHorizontalFlip = RandomHorizontalFlip
        self.RandomRotation90Degree = RandomRotation90Degree
        self.ToTensor = ToTensor
        self.crop_size = crop_size

    def __getitem__(self, item):
        file_GT = self.image_files_GT[item]
        file_LQ = self.image_files_LQ[item]
        image_GT = Image.open(os.path.join(self.rootGT, file_GT))
        image_LQ = Image.open(os.path.join(self.rootLQ, file_LQ))
        crop_size = self.crop_size

        # 进行transform
        if self.RandomCrop:
            image_GT = np.array(image_GT)
            image_LQ = np.array(image_LQ)
            image_GT, image_LQ = paired_random_crop(image_GT, image_LQ, crop_size, 4, file_GT)
            image_GT = Image.fromarray(image_GT.astype('uint8')).convert('RGB')
            image_LQ = Image.fromarray(image_LQ.astype('uint8')).convert('RGB')

        if self.RandomHorizontalFlip:
            if random() > 0.5:
                image_GT = TF.hflip(image_GT)
                image_LQ = TF.hflip(image_LQ)

        if self.RandomRotation90Degree:
            if random() > 0.5:
                image_GT = TF.rotate(image_GT, 90)
                image_LQ = TF.rotate(image_LQ, 90)

        if self.ToTensor:
            image_GT = TF.to_tensor(image_GT)
            image_LQ = TF.to_tensor(image_LQ)

        return {'gt': image_GT, 'lq': image_LQ}

    def __len__(self):
        return len(self.image_files_GT)
