import cv2 as cv


def Threshold(threshold):
    retval, thre = cv.threshold(gray, threshold, 255, cv.THRESH_OTSU)
    cv.imshow('threshold_adjust', thre)


threshold = 0

img = cv.imread('OK_PEDAL.bmp')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# dst = cv.GaussianBlur(gray, (3, 3), 0)

cv.namedWindow('threshold_adjust')

cv.createTrackbar('Thre', 'threshold_adjust', threshold, 255, Threshold)

Threshold(0)  # initialization

cv.imshow('Original',img)

if cv.waitKey(0) == 27:
    cv.destroyAllWindows()
