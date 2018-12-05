# 合并图片

import cv2 as cv
import numpy as np
import os

# 读文件
path = 'H:\\bin\\1120\\GRAY\\NG_PEDAL_rotated'
dir = os.listdir(path)
img_list = []
for i in dir:
    if os.path.splitext(i)[1] == '.bmp':
        img_list.append(i)
file_num = len(img_list)

# 每行图片数量（假设每张图片大小一致，目标图片为正方形）
side_num = int(np.sqrt(file_num) + 1)
# 新建一个空图片
img = cv.imread(os.path.join(path, img_list[0]))
img_side = img.shape[0]


# 改写图片

def merge():
    dst = np.zeros((img_side * side_num, img_side * side_num, 3))
    t = 1
    idx = 0
    for i in range(side_num):
        for j in range(side_num):
            img = cv.imread(os.path.join(path, img_list[idx]))
            cols = img.shape[1]
            rows = img.shape[0]
            roi = img[0:rows, 0:cols, :]
            dst[(i * img_side):((i + 1) * img_side), (j * img_side):((j + 1) * img_side), :] = roi
            idx += 1

            print('已完成： %s / %s' % (t, file_num))
            if t == file_num:
                return dst
            t += 1


dst = merge()

cv.imwrite('merge.bmp',dst)
