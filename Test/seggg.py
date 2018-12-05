import cv2 as cv
import numpy as np
from Test.test_pedal import rectttttttt, lineeeeeeee
import os


def segmenttttt(rotated, lined, corePts, borderPts):
    dst = lined.copy()
    # 重新二值化，用于统计像素数量
    gray = cv.cvtColor(rotated, cv.COLOR_BGR2GRAY)
    _, thre = cv.threshold(gray, 135, 255, cv.THRESH_BINARY)

    # mask制作
    # 分别绘制2,4,6,8区域的mask
    area_2 = np.array([corePts[0], corePts[2], borderPts[2], borderPts[0]], dtype=np.int32)
    area_4 = np.array([corePts[0], corePts[1], borderPts[6], borderPts[4]], dtype=np.int32)
    area_6 = np.array([corePts[2], corePts[3], borderPts[7], borderPts[5]], dtype=np.int32)
    area_8 = np.array([corePts[1], corePts[3], borderPts[3], borderPts[1]], dtype=np.int32)
    areaPts = [area_2, area_4, area_6, area_8]

    kernel = np.ones((5, 5))
    closed = cv.morphologyEx(thre, cv.MORPH_CLOSE, kernel)
    areas = []
    for idx in range(len(areaPts)):
        # 应用mask
        areaMask = np.zeros_like(closed, dtype=np.uint8)
        cv.fillPoly(areaMask, [areaPts[idx]], 1)
        seg_thre = areaMask * closed
        # 计算mask区域内花瓣面积
        # 闭运算
        # closed = cv.morphologyEx(seg_thre, cv.MORPH_CLOSE, kernel)
        cv.imshow(str(idx), seg_thre)
        # 筛选
        _, contours, _ = cv.findContours(seg_thre, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        cnt_list = []
        for i in contours:
            cnt = {'contour': i, 'area': cv.contourArea(i)}
            cnt_list.append(cnt)
        cnt_list.sort(key=lambda x: x['area'], reverse=True)
        cv.drawContours(dst, cnt_list[0]['contour'], -1, (0, 0, 255), 1)
        # 统计
        areas.append(cnt_list[0]['area'])

    return dst, areas


# # 单独图片处理
# if __name__ == '__main__':
#     img = cv.imread('H:\\bin\\1120\\GRAY\\NG_PEDAL_EXPANDED\\NG_9_1.bmp')
#     rotated, corePts = rectttttttt(img)
#     lined, borderPts = lineeeeeeee(rotated, corePts)
#     dst, areas = segmenttttt(rotated, lined, corePts, borderPts)
#     cv.imshow('test', dst)
#     cv.waitKey()

# 批处理
if __name__ == '__main__':
    path = 'H:\\bin\\1120\\GRAY\\OK_PEDAL_EXPANDED'
    new_path = 'H:\\bin\\1120\\OK_EXPANDED_segg'
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
        rotated, corePts = rectttttttt(img)
        lined, borderPts = lineeeeeeee(rotated, corePts)
        dst, areas = segmenttttt(rotated, lined, corePts, borderPts)

        cv.imwrite(os.path.join(new_path, file), dst)
        print('已完成： %s / %s' % (t, file_num))
        t += 1
