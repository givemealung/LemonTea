# 批量图片转为灰度图

import os
import cv2 as cv

path = 'H:\\gray\\Hole_gray'
new_path = 'H:\\gray\\Hole_gray\\GRAY'
if not os.path.exists(new_path):
    os.makedirs(new_path)
dir = os.listdir(path)
img_list = []
for i in dir:
    if os.path.splitext(i)[1] == '.bmp':
        img_list.append(i)
file_num = len(img_list)
t = 1

for file in img_list:
    img = cv.imread(os.path.join(path, file))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imwrite(os.path.join(new_path, file), gray)

    print('已完成：%s / %s' % (t, file_num))
    t += 1
