import cv2 as cv
import matplotlib.pyplot as plt

import  numpy as np
# 均值滤波有可能模糊图片，高斯滤波没法去掉小白点
# 读取图片
img = cv.imread('Image/image_9_1.png')

b,g,r=cv.split(img)

for i in range(100000):
    w = img.shape[0]
    h = img.shape[1]

    x=np.random.randint(0,w)
    y=np.random.randint(0,h)

    b[x,y]=0  #变成黑色
    g[x, y] = 0  # 变成黑色
    r[x, y] = 0  # 变成黑色


img_new=cv.merge([b,g,r])


# 显示


cv.imshow('image', img)
cv.imshow('img_new', img_new)


cv.waitKey()
cv.destroyAllWindows()
