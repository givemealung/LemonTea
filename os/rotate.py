# 旋转图片

import cv2 as cv
import numpy as np
import os


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv.warpAffine(image, M, (nW, nH))


path = 'H:\\bin\\1120\\GRAY\\OK_PEDAL'
new_path = 'H:\\bin\\1120\\GRAY\\OK_PEDAL_rotated'
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
    dst = rotate_bound(img, 45)

    cv.imwrite(os.path.join(new_path, file), dst)
    print('已完成： %s / %s' % (t, file_num))
    t += 1
