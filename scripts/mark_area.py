# -*- coding: UTF-8 -*-
""" 标注区域画框并且裁剪
@Project ：srgan-esrgan-pytorch 
@File    ：mark_area.py
@IDE     ：PyCharm 
@Author  ：AmazingPeterZhu
@Date    ：2021/5/28 上午9:41 
"""
import cv2
import os

# 标注区的左上角坐标和长宽
x = 100
y = 600
w = 350
h = 350

root_path = os.getcwd()
img_path = os.path.join(root_path, 'mark_img')
results_path = os.path.join(root_path, 'mark_results')
crop_path = os.path.join(root_path, 'mark_results', 'crop_results')

img_list = os.listdir(img_path)

print("Start...")
for img_name_suffix in img_list:
    img = cv2.imread(os.path.join(img_path, img_name_suffix))
    img_name, img_suffix = img_name_suffix.split('.', 1)[0], img_name_suffix.split('.', 1)[1]

    # 裁剪
    img_name_cropped = img_name + '_cropped' + '.' + img_suffix
    img_cropped = img[y:y+h, x:x+w]
    cv2.imwrite(os.path.join(crop_path, img_name_cropped), img_cropped)

    # 标框
    left_top = (x, y)
    right_bottom = (x + w, y + h)
    img_name_marked = img_name + '_marked' + '.' + img_suffix
    cv2.rectangle(img, left_top, right_bottom, (0, 0, 255), 5)
    cv2.imwrite(os.path.join(results_path, img_name_marked), img)


print("Finished!!!")
