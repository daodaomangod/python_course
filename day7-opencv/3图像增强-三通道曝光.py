import cv2 as cv
import copy
import numpy as np

# 读取图像  0就读灰度图
# img= cv.imread('Image/image_5_2.png',0)

img = cv.imread('Image/image_5_3.png')

# 拆分三通道
b,g,r=cv.split(img)

# 均衡化
img_b =cv.equalizeHist(b)
img_g =cv.equalizeHist(g)

img_r =cv.equalizeHist(r)

# 合并
img_new=cv.merge([img_b,img_g,img_r])
#
cv.imshow('image', img)

cv.imshow('img_new', img_new)
cv.waitKey()
cv.destroyAllWindows()
