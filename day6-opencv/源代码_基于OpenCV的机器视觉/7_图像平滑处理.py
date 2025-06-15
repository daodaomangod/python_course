import cv2 as cv
import numpy as np

# img = cv.imread('Image/gear.png', 0)  # 灰度图像读取
# # 给图像加一些噪声
# for i in range(1000):
#     rows = img.shape[0]
#     cols = img.shape[1]
#     x = np.random.randint(0, rows)
#     y = np.random.randint(0, cols)
#     img[x, y] = 255  # 添加白点
#
# # 均值滤波:灰度值求平均值
# img_mean = cv.blur(img, (3, 3))
#
# # 高斯滤波: 使用高斯核每个像素作卷积运算，从而得到输出图像
# img_gaussian = cv.GaussianBlur(img, (3, 3), 0.5)
#
# # 中值滤波: 灰度值求中值
# img_median = cv.medianBlur(img, 3)
#
# cv.imshow('Image', img)
# cv.imshow('mean', img_mean)
# cv.imshow('gaussian', img_gaussian)
# cv.imshow('median', img_median)
#
# cv.waitKey()

# # 练习题目
# img = cv.imread('Image/cat_nosie.png', 0)  # 灰度图像读取
#
# # 均值滤波:灰度值求平均值
# img_mean = cv.blur(img, (3, 5))
#
# # 高斯滤波: 使用高斯核每个像素作卷积运算，从而得到输出图像
# img_gaussian = cv.GaussianBlur(img, (3, 3), 1)
#
# # 中值滤波: 灰度值求中值
# img_median = cv.medianBlur(img, 7)
#
# cv.imshow('Image', img)
# cv.imshow('mean', img_mean)
# cv.imshow('gaussian', img_gaussian)
# cv.imshow('median', img_median)
#
# cv.waitKey()

img = cv.imread('Image/cat_nosie.png')

# BGR通道分别滤波
b, g, r = cv.split(img)
b = cv.medianBlur(b, 5)
g = cv.medianBlur(g, 5)
r = cv.medianBlur(r, 5)
img_med = cv.merge([b, g, r])

# 显示结果
cv.imshow('Original', img)
cv.imshow('BGR Separate', img_med)

cv.waitKey(0)
cv.destroyAllWindows()
