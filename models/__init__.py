# -*- coding: UTF-8 -*-
"""
@Project ：srgan-esrgan-pytorch
@File    ：__init__.py
@IDE     ：PyCharm 
@Author  ：AmazingPeterZhu
@Date    ：2021/5/12 下午1:17 
"""
from .srgan_models import SRResNet, Discriminator
from .esrgan_models import RRDBNet, RaDiscriminator
from .resrgan_models import SRRDBNet, RERaDiscriminator


__all__ = ['SRResNet', 'Discriminator', 'RRDBNet', 'RaDiscriminator',
           'SRRDBNet', 'RERaDiscriminator']
