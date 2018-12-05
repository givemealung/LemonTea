# 11/28 修改
# 重新改写 只添加花瓣相关函数
# 12/03 修改
# 绘制环轮廓

import cv2 as cv
import numpy as np
import openpyxl
import os


class Lemon(object):

    def readImage(self, path):
        '''
        Load the image, save its BGR as well as GRAY form as property.
        :param path: Image path
        :return: None
        '''
        img = cv.imread(path)
        self.img = self.rotate_bound(img, 90)  # 向右旋转90度
        gray = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        self.gray = cv.GaussianBlur(gray, (3, 3), 0)

        self.fitting_param = {'file_name': path.split('\\')[-1]}

    def holes_fitting(self, mark=False):
        '''
        Find the circle contours, and fit into a line.
        :return: thre_masked(for func:line_fitting)
        '''
        # 二值化
        retval, thre = cv.threshold(self.gray, 20, 255, cv.THRESH_BINARY)

        # 寻找铝箔区域
        y = thre.shape[1] // 3
        line = thre[y, :]
        idx = np.where(line == 255)
        x1, x2 = idx[0][0], idx[0][-1]

        # 清除铝箔外区域
        thre[:, :x1] = 0
        thre[:, x2:] = 0

        # 去除最大轮廓
        image, contours, hierarchy = cv.findContours(thre, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        cnt_list = []
        for i in contours:
            cnt = {'contour': i, 'area': cv.contourArea(i)}
            cnt_list.append(cnt)
        cnt_list.sort(key=lambda x: x['area'], reverse=True)
        cnt_list.pop(0)

        # 每个轮廓找质心
        # 判断轮廓个数
        mass_center = []
        pre_area = cnt_list[0]['area']
        for i in cnt_list:
            if i['area'] >= pre_area / 2:
                pre_area = i['area']
                if mark == True:
                    cv.drawContours(self.img, i['contour'], -1, (0, 0, 255), 2)
                moments = cv.moments(i['contour'])
                x = moments['m10'] / moments['m00']
                y = moments['m01'] / moments['m00']
                mass_center.append((x, y))
            else:
                break

        mass_center.sort(key=lambda x: x[0], reverse=True)
        self.fitting_param['holes_center'] = mass_center

    def rotate_bound(self, image, angle):
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

    def pedalCheck(self, file, ptMark=False, lineMark=False, areaMark=False):
        '''
        检测花瓣
        :return:检测后的图片，孔洞数目
        '''

        side_length = 60  # roi参数
        holes_num = 0
        for pt in self.fitting_param['holes_center']:
            cX, cY = pt[:]
            roi = self.img.copy()[int(cY - side_length / 2):int(cY + side_length / 2),
                  int(cX - side_length / 2):int(cX + side_length / 2)]  # bgr，不经过高斯模糊
            name = file.split('.')[0] + '_' + str(holes_num + 1) + '.bmp'
            rotated, corePts, hole_area = self.rect_holes_fitting(roi)
            areas = self.segment(rotated, corePts, name, areaMark=True)
            self.fitting_param['hole_%s_areas' % (holes_num + 1)] = areas
            holes_num += 1

        return holes_num

    def rect_holes_fitting(self, img, ptMark=False):
        '''
        判定roi中心孔洞的四个角点
        :param img: 花瓣roi（非旋转）
        :return: 旋转后的图片，角点坐标，孔洞面积
        '''
        rotated = self.rotate_bound(img, 45)
        gray = cv.cvtColor(rotated, cv.COLOR_BGR2GRAY)

        retval, thre = cv.threshold(gray, 50, 255, cv.THRESH_BINARY)  # 阈值50
        _, contours, _ = cv.findContours(thre, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        cnt_list = []
        for i in contours:
            cnt = {'contour': i, 'area': cv.contourArea(i)}
            cnt_list.append(cnt)
        cnt_list.sort(key=lambda x: x['area'], reverse=True)
        cnt_list.pop(0)

        # 孔洞面积
        hole_area = cnt_list[0]['area']
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

        if ptMark == True:
            for i in corePts:
                cv.circle(rotated, i, 2, (0, 0, 255), -1)

        return rotated, corePts, hole_area

    def segment(self, rotated, corePts, name, lineMark=False, areaMark=False):
        '''
        根据孔洞的四个角点将花瓣切割成九个区域，并分别计算相应的面积
        :param rotated: 旋转后的roi
        :param corePts: 孔洞四个角点
        :return: 标记后的照片，九个区域的面积
        '''

        def lineSegment(rotated, thinkness=1):

            def vertLine(pt1, pt2, img):
                x1, y1, x2, y2 = pt1[0], pt1[1], pt2[0], pt2[1]
                rows, cols = img.shape[:2]
                if x1 == x2:
                    if lineMark:
                        cv.line(img, (x1, 0), (x1, rows - 1), (0, 255, 0), thinkness)  # Green line
                    topy = bottomy = x1
                else:
                    topy = np.round(x1 - (x2 - x1) / (y2 - y1) * y1).astype(np.uint8)
                    bottomy = np.round(x1 - (x2 - x1) / (y2 - y1) * (y1 + 1 - rows)).astype(np.uint8)
                    if lineMark:
                        cv.line(img, (topy, 0), (bottomy, rows - 1), (0, 255, 0), thinkness)  # Green line

                return (topy, 0), (bottomy, rows - 1)

            def horLine(pt1, pt2, img):
                x1, y1, x2, y2 = pt1[0], pt1[1], pt2[0], pt2[1]
                rows, cols = img.shape[:2]
                lefty = np.round(y1 - x1 * (y2 - y1) / (x2 - x1)).astype(np.uint8)
                righty = np.round(y1 - (x1 + 1 - cols) * (y2 - y1) / (x2 - x1)).astype(np.uint8)
                if lineMark:
                    cv.line(img, (cols - 1, righty), (0, lefty), (255, 0, 0), 1)  # Blue line

                return (0, lefty), (cols - 1, righty)

            lined = rotated.copy()
            # pts_list(左上，左下，右上，右下)
            vertPts = [(corePts[0], corePts[1]), (corePts[2], corePts[3])]
            horPts = [(corePts[0], corePts[2]), (corePts[1], corePts[3])]
            borderPts = []
            for i in vertPts:
                top, bottom = vertLine(i[0], i[1], lined)
                borderPts.append(top)
                borderPts.append(bottom)
            for j in horPts:
                left, right = horLine(j[0], j[1], lined)
                borderPts.append(left)
                borderPts.append(right)

            return lined, borderPts

        def pedalSegment(rotated, lined, corePts, borderPts):

            def calculateArea(ptsArea):
                areaMask = np.zeros_like(thre, dtype=np.uint8)
                cv.fillPoly(areaMask, [ptsArea], 1)
                seg_thre = areaMask * thre
                # 计算mask区域内面积
                _, contours, _ = cv.findContours(seg_thre, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
                if len(contours) == 0:
                    area = 0
                else:
                    cnt_list = []
                    for i in contours:
                        cnt = {'contour': i, 'area': cv.contourArea(i)}
                        cnt_list.append(cnt)
                    cnt_list.sort(key=lambda x: x['area'], reverse=True)
                    if areaMark:
                        cv.drawContours(dst, cnt_list[0]['contour'], -1, (0, 0, 255), 1)
                        cv.imwrite(name, dst)
                    area = cnt_list[0]['area']

                return area

            dst = lined.copy()
            # 重新二值化，用于统计像素数量
            gray = cv.cvtColor(rotated, cv.COLOR_BGR2GRAY)
            _, thre = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)  # 阈值150

            rows, cols = gray.shape[:2]
            # mask制作
            # 分别绘制花瓣，环，孔的mask
            area_1 = np.array([corePts[0], borderPts[0], (0, 0), borderPts[4]], dtype=np.int32)
            area_2 = np.array([corePts[0], corePts[2], borderPts[2], borderPts[0]], dtype=np.int32)
            area_3 = np.array([corePts[2], borderPts[2], (cols - 1, 0), borderPts[5]], dtype=np.int32)
            area_4 = np.array([corePts[0], corePts[1], borderPts[6], borderPts[4]], dtype=np.int32)
            area_6 = np.array([corePts[2], corePts[3], borderPts[7], borderPts[5]], dtype=np.int32)
            area_7 = np.array([corePts[1], borderPts[1], (0, rows - 1), borderPts[6]], dtype=np.int32)
            area_8 = np.array([corePts[1], corePts[3], borderPts[3], borderPts[1]], dtype=np.int32)
            area_9 = np.array([corePts[3], borderPts[3], (cols - 1, rows - 1), borderPts[7]], dtype=np.int32)
            area_list = (area_1)

            areas=calculateArea(area_3)

            return areas

        lined, borderPts = lineSegment(rotated)
        areas = pedalSegment(rotated, lined, corePts, borderPts)

        return areas


# # 单文件处理
# if __name__ == '__main__':
#     path='H:\\bin\\1120\\GRAY\\OK_gray\\1.bmp'
#
#     test=Lemon()
#     test.read_image(path)
#     test.holes_fitting()
#     holes_num = test.pedal_check()


# 批文件处理
if __name__ == '__main__':
    path = 'H:\\bin\\1203stat\\gray\\holes'
    dir = os.listdir(path)
    img_list = []
    for i in dir:
        if os.path.splitext(i)[1] == '.bmp':
            img_list.append(i)
    file_num = len(img_list)
    t = 1

    test = Lemon()

    for file in img_list:
        test.readImage(os.path.join(path, file))
        test.holes_fitting()
        holes_num = test.pedalCheck(file, )

        print('已完成：%s / %s' % (t, file_num))
        t += 1
