import cv2 as cv
import copy
import numpy as np

# 读取图像  0就读灰度图
# img= cv.imread('Image/image_5_2.png',0)

img = cv.imread('Image/image_5_2.png')

# 灰度图
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 复制gray灰度图
img_new = copy.deepcopy(gray)

w, h, ch = img.shape
# 读取每一个像素点

for i in range(w):
    for j in range(h):
        # img_new[i,j] = 255 - img_new[i, j]   #取反过程
        # img_new[i, j] = 45 * np.log(1 + img_new[i, j])  # 对数变换
        img_new[i, j] = 2 * img_new[i, j]**0.5   # 指数变换

#
cv.imshow('image', img)
cv.imshow('gray', gray)
cv.imshow('img_new', img_new)
cv.waitKey()
cv.destroyAllWindows()
