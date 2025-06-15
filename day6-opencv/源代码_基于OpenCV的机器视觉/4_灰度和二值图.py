import cv2 as cv

# 灰度图
# 方式1：在图像读取的时候直接将彩色图像转换成灰度图像
# 图像读取: cv.imread('图片路径path'，读取图像的方式)
# 读取图像的方式, -1，表示读取原图；0，表示以灰度图方式读取原图；1，表示以RGB方式读取原图

# img_0 = cv.imread('Image/Lena.png')
# cv.imshow('0', img_0)
#
# img_1 = cv.imread('Image/Lena.png',0)
# cv.imshow('1', img_1)
#
# cv.waitKey()

# # 方式2：通过cvtColor将彩色图像转换成灰度图像
# img_BGR = cv.imread('Image/Lena.png')
# cv.imshow('BGR', img_BGR)
#
# img_RGB = cv.cvtColor(img_BGR, cv.COLOR_BGR2RGB)
# cv.imshow('RGB', img_RGB)
#
# img_Gray = cv.cvtColor(img_BGR, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray', img_Gray)
#
# img_HSV = cv.cvtColor(img_BGR, cv.COLOR_BGR2HSV)
# cv.imshow('HSV', img_Gray)
#
# cv.waitKey()

# 二值图

# 读取图像，灰度图
img = cv.imread('Image/Lena.png', 0)

# 简单阈值处理
# ret, img_1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
ret, img_1 = cv.threshold(img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

# 显示结果
cv.imshow('Image Gray', img)
cv.imshow('Image Threshold', img_1)

cv.waitKey()

