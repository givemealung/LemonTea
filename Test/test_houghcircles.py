import cv2 as cv
import numpy as np

img = cv.imread('../gray.bmp')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = cv.GaussianBlur(gray, (3, 3), 0)

canny=cv.Canny(gray,234,255)

circles = cv.HoughCircles(canny, cv.HOUGH_GRADIENT, 1, 10, param1=220, param2=30, minRadius=1, maxRadius=50)
print(circles)
circles = np.uint16(np.around(circles))

for i in circles[0, :]:
    cv.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
    cv.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)

cv.imshow('HoughCircles', img)
cv.waitKey()
cv.destroyAllWindows()
