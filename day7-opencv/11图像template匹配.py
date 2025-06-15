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
template = cv.imread('Image/temp_Lena_1.png', 0)

# 模板匹配
result = cv.matchTemplate(target, template, cv.TM_CCOEFF_NORMED)
# print(result)
# 最佳匹配
minval, maxval, minloc, maxloc = cv.minMaxLoc(result)
w, h = template.shape
# 匹配区域
cv.rectangle(img, (maxloc), (maxloc[0] + h, maxloc[1] + w), (0, 0, 255), thickness=3)

cv.imshow('img', img)
cv.waitKey()
cv.destroyAllWindows()
