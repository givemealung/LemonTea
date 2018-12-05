import cv2 as cv
import numpy as np


def pts(useless):
    image = img.copy()
    detected_edges = cv.Canny(gray, 113, 232)

    nums = cv.getTrackbarPos('Numbers', 'CornerCheck')
    qual = cv.getTrackbarPos('Quality', 'CornerCheck') / 100

    crns = cv.goodFeaturesToTrack(detected_edges, nums, qual, 2)
    crns = np.int0(crns)

    for i in crns:
        x, y = i.ravel()
        cv.circle(image, (x, y), 2, (0.0, 255), -1)  # 在原图像上画出角点位置
    cv.imshow('CornerCheck', image)


img = cv.imread('H:\\bin\\1120\\GRAY\\NG_PEDAL_EXPANDED\\NG_9_1.bmp')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

corners, quality = 1, 1
cv.namedWindow('CornerCheck')
cv.createTrackbar('Numbers', 'CornerCheck', corners, 60, pts)
cv.createTrackbar('Quality', 'CornerCheck', quality, 100, pts)

pts(0)
cv.waitKey(0)
