# 用于删除文件名中的中文

import os

path='H:\\bin\\1120\\NG_PEDAL'
dir=os.listdir(path)

for i in dir:
    os.rename(os.path.join(path,i),os.path.join(path,i.replace('OK','NG')))