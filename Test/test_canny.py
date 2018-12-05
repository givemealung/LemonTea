import cv2 as cv
import numpy as np
import os


def CannyThreshold(temp):
    lowThre = cv.getTrackbarPos('Low Thre', 'canny')
    highThre = cv.getTrackbarPos('High Thre', 'canny')
    print(lowThre, highThre)
    detected_edges = cv.Canny(gray, lowThre, highThre)
    cv.imshow('canny', detected_edges)


lowThre, highThre = 0, 0


img = cv.imread('H:\\bin\\1120\\GRAY\\NG_PEDAL_EXPANDED\\NG_9_1.bmp')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

cv.namedWindow('canny')

cv.createTrackbar('Low Thre', 'canny', lowThre, 255, CannyThreshold)
cv.createTrackbar('High Thre', 'canny', highThre, 255, CannyThreshold)

CannyThreshold(0)  # initialization
if cv.waitKey(0) == 27:
    cv.destroyAllWindows()
