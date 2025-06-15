import copy

import cv2 as cv
import matplotlib.pyplot as plt

import numpy as np

# 均值滤波有可能模糊图片，高斯滤波没法去掉小白点
# 读取目标图片
img = cv.imread('Image/Lena.png')

# 灰度
target = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 模板
template = cv.imread('Image/temp_Lena_2.png', 0)

# 创建一个sift特征检测器
sift = cv.SIFT_create()

# 计算特征点

kp1, des1 = sift.detectAndCompute(target, None)

kp2, des3 = sift.detectAndCompute(template, None)

# 创建匹配实例
bf = cv.BFMatcher(cv.NORM_L2)

# 匹配
# match = bf.match(des3, des1)
match = bf.knnMatch(des3, des1, k=2)
print(len(match))
good_match = []

for m, n in match:
    if m.distance / n.distance < 0.7:
        good_match.append(m)


print(len(good_match))

# 可视化
# result = cv.drawMatches(template, kp2, target, kp1, match, None, flags=2)
result = cv.drawMatches(template, kp2, img, kp1, good_match, None, flags=2)

# 显示

cv.imshow('img', result)
# cv.imshow('SIFT', result)
cv.waitKey()
cv.destroyAllWindows()
