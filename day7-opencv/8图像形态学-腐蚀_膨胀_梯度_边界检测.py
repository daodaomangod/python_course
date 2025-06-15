import cv2 as cv
import matplotlib.pyplot as plt

import numpy as np

# 均值滤波有可能模糊图片，高斯滤波没法去掉小白点
# 读取图片
# img = cv.imread('Image/image_8_1.png')
img = cv.imread('Image/image.png')

# 灰度
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 边界检测  提取边界 ！！！
canny = cv.Canny(gray, 50, 150)

# 二值图

ret, binary = cv.threshold(img, 127, 255, cv.THRESH_BINARY)

# kernel = np.ones((3, 3), np.uint8)
# 形状，
kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))

# 腐蚀 核函数kernel   迭代次数iterations表示腐蚀次数
erode = cv.erode(binary, kernel=kernel, iterations=2)

# 膨胀函数
dilate = cv.dilate(erode, kernel=kernel, iterations=2)
# 开运算  先腐蚀，再膨胀
img_open = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel=kernel)

# 闭运算 先膨胀在腐蚀

img_close = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel=kernel)
# 形态学梯度 提取轮廓
img_grad = cv.morphologyEx(img_open, cv.MORPH_GRADIENT, kernel=kernel)

# 显示


cv.imshow('image', binary)
cv.imshow('erode', erode)
cv.imshow('dilate', dilate)
cv.imshow('img_open', img_open)
cv.imshow('img_close', img_close)
cv.imshow('img_grad', img_grad)
cv.imshow('canny', canny)
cv.waitKey()
cv.destroyAllWindows()
