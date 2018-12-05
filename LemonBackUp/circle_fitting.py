# circle_fitting函数备份
# 适用于旧照片

#############################################################################
        #
        # # 再找轮廓
        # image, contours, hierarchy = cv.findContours(thre_masked, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        #
        # cnt_list = []
        # for i in contours:
        #     cnt = {'contour': i, 'area': cv.contourArea(i)}
        #     cnt_list.append(cnt)
        # cnt_list.sort(key=lambda x: x['area'], reverse=True)
        # cnt_list.pop(0)
        #
        # # 每个轮廓找质心
        # # 判断轮廓个数
        # mass_center = []
        # pre_area = cnt_list[0]['area']
        # holes_counts = 0
        # for i in cnt_list:
        #     if i['area'] >= pre_area / 2:
        #         holes_counts += 1
        #         pre_area = i['area']
        #         cv.drawContours(self.img, i['contour'], -1, (0, 0, 255), 2)
        #         moments = cv.moments(i['contour'])
        #         x = moments['m10'] / moments['m00']
        #         y = moments['m01'] / moments['m00']
        #         mass_center.append((x, y))
        #     else:
        #         break
        #
        # # 拟合直线（1108）
        # # 用左右端点拟合
        # pts = np.array(mass_center)
        # rows, cols = self.img.shape[:2]
        # line = cv.fitLine(pts, cv.DIST_L2, 0, 0.01, 0.01)
        # vx, vy, x, y = line[0][0], line[1][0], line[2][0], line[3][0]
        # lefty = int((-x * vy / vx) + y)
        # righty = int(((cols - x) * vy / vx) + y)
        # cv.line(self.img, (cols - 1, righty), (0, lefty), (0, 255, 0), 2)
        #
        # # 在纯黑图上画线
        # c_line = np.zeros_like(thre_masked)
        # cv.line(c_line, (cols - 1, righty), (0, lefty), 255, 2)
        #
        # # 返回拟合直线参数
        # # 数据类型：numpy.float32
        # angle = np.arctan2(vy, vx) * 180.0 / np.pi
        # intercept = -x * (vy / vx) + y
        #
        # self.fitting_param['holes_num'] = holes_counts
        # self.fitting_param['holes'] = (angle, intercept)
        # self.fitting_param['c_line'] = (lefty, righty)
        #
        # return thre_masked, c_line, mass_center
        ##########################################################