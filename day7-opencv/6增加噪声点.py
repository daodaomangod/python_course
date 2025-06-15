import cv2 as cv
import matplotlib.pyplot as plt

import  numpy as np
# 均值滤波有可能模糊图片，高斯滤波没法去掉小白点
# 读取图片
img = cv.imread('Image/image_9_1.png', 0)

for i in range(1000):
    w = img.shape[0]
    h = img.shape[1]

    x=np.random.randint(0,w)
    y=np.random.randint(0,h)

    img[x,y]=0  #变成黑色





# 显示


cv.imshow('image', img)


cv.waitKey()
cv.destroyAllWindows()
