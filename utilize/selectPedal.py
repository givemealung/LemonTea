from utilize.lemon_tea import Lemon
import cv2 as cv
import numpy as np
import os

# 读文件
path = 'H:\\bin\\1125sample'
new_path = 'H:\\bin\\1125sample\\pedals'
if not os.path.exists(new_path):
    os.makedirs(new_path)
dir = os.listdir(path)
img_list = []
for i in dir:
    if os.path.splitext(i)[1] == '.bmp':
        img_list.append(i)
file_num = len(img_list)
t = 1

# 实例化
test = Lemon()

for file in img_list:
    test.readImage(os.path.join(path, file))
    test.circle_fitting()
    pts = test.circle_fitting()
    holes_num = 1
    # roi参数
    side_length = 60

    for i in pts:
        cX, cY = i[:]
        roi = test.img[int(cY - side_length / 2):int(cY + side_length / 2),
              int(cX - side_length / 2):int(cX + side_length / 2)]
        new_file = 'OK_' + file.split('.')[0] + '_%s' % holes_num
        new_file += '.bmp'
        # cv.imshow(new_file, roi)
        cv.imwrite(os.path.join(new_path, new_file), roi)
        print('已完成： %s_%s / %s' % (t, holes_num, file_num))
        holes_num += 1
    t += 1

