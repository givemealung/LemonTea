from Test.test_pedal import zooooooooom
import cv2 as cv
import os

path = 'H:\\bin\\1120\\GRAY\\NG_PEDAL'
new_path = 'H:\\bin\\1120\\GRAY\\NG_PEDAL_EXPANDED'
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
    dst = zooooooooom(img, 2)
    cv.imwrite(os.path.join(new_path, file), dst)

    print('已完成： %s / %s' % (t, file_num))
    t += 1
