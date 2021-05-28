# -*- coding: UTF-8 -*-
"""
@Project ：srgan-esrgan-pytorch 
@File    ：mark_area.py
@IDE     ：PyCharm 
@Author  ：AmazingPeterZhu
@Date    ：2021/5/28 上午9:41 
"""
import cv2
import os

# mark and crop area parameter
x = 0
y = 0
w = 0
h = 0

root_path = os.getcwd()
img_path = os.path.join(root_path, 'mark_img')
results_path = os.path.join(root_path, 'mark_results')

print(root_path)
print(img_path)
print(results_path)