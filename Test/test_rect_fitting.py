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


def rect_holes_fitting(img, mark=False):
    rotated = rotate_bound(img, 45)
    gray = cv.cvtColor(rotated, cv.COLOR_BGR2GRAY)

    retval, thre = cv.threshold(gray, 65, 255, cv.THRESH_BINARY)  # 阈值65
    _, contours, _ = cv.findContours(thre, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    cnt_list = []
    for i in contours:
        cnt = {'contour': i, 'area': cv.contourArea(i)}
        cnt_list.append(cnt)
    cnt_list.sort(key=lambda x: x['area'], reverse=True)
    cnt_list.pop(0)
    # cv.drawContours(rotated, cnt_list[0]['contour'], -1, (0, 0, 255), 2)

    rect = cv.minAreaRect(cnt_list[0]['contour'])
    box = cv.boxPoints(rect)

    # 点排序
    # final_pts_list(左上，左下，右上，右下)
    pts_list = []
    corePts = []
    for i in range(box.shape[0]):
        pts_list.append((box[i][0], box[i][1]))
    pts_list.sort(key=lambda x: x[0])
    corePts.append(pts_list.pop(0))
    corePts.append(pts_list.pop(0))
    corePts.sort(key=lambda x: x[1])
    pts_list.sort(key=lambda x: x[1])
    corePts.append(pts_list.pop(0))
    corePts.append(pts_list.pop(0))

    if mark == True:
        for i in corePts:
            cv.circle(rotated, i, 2, (0, 0, 255), -1)

    return rotated, corePts


# 批处理
path = 'H:\\bin\\1120\\GRAY\\OK_PEDAL'
new_path = 'H:\\bin\\1120\\GRAY\\OK_PEDAL_rect'
if not os.path.exists(new_path):
    os.mkdir(new_path)
dir = os.listdir(path)
img_list = []
for i in dir:
    if os.path.splitext(i)[1] == '.bmp':
        img_list.append(i)
file_num = len(img_list)
t = 1

for file in img_list:
    img = cv.imread(os.path.join(path, file))
    dst, _ = rect_holes_fitting(img, mark=True)
    cv.imwrite(os.path.join(new_path, file), dst)

    print('已完成：%s / %s' % (t, file_num))
    t += 1
