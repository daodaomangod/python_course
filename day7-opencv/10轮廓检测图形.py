import copy

import cv2 as cv
import matplotlib.pyplot as plt

import numpy as np

# 均值滤波有可能模糊图片，高斯滤波没法去掉小白点
# 读取图片
# img = cv.imread('Image/image_8_1.png')
img = cv.imread('Image/image_10_1.png')

# 灰度
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 二值化
# ret,canny  = cv.threshold(gray,127,255,cv.THRESH_BINARY_INV)
# 函数二值化
# ret, canny = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
# 边界检测  提取边界 ！！！
canny = cv.Canny(gray, 50, 150)
# 查找轮廓

contours, hie = cv.findContours(canny, cv.RETR_EXTERNAL,
                                cv.CHAIN_APPROX_SIMPLE)

# print(len(contours))


# 复制原图
result_img = img.copy()

# 轮廓绘制
for i in range(len(contours)):
    # 计算面积
    area = cv.contourArea(contours[i])
    # 计算周长
    length = cv.arcLength(contours[i], closed=True)

    #  大概多边形数量，识别  计算几边形
    approx = cv.approxPolyDP(contours[i], 0.02 * length, True)

    print(len(approx))
    x, y, w, h = cv.boundingRect(contours[i])

    if (len(approx) == 3):  # 三角形
        cv.drawContours(result_img, contours, i, (0, 0, 0), thickness=3)

        # 文字描述
        cv.putText(result_img, str(3), (int(x + w / 2), int(y + h / 2)), cv.FONT_HERSHEY_SIMPLEX,
                   1, (0, 0, 0), 1)
    elif (len(approx) == 8 and int(area) > 100):
        cv.drawContours(result_img, contours, i, (0, 0, 0), thickness=3)

        # 文字描述
        cv.putText(result_img, str(i), (int(x + w / 2), int(y + h / 2)), cv.FONT_HERSHEY_SIMPLEX,
                   1, (0, 0, 0), 1)
    elif 1.1 > w / h > 0.9 and len(approx) == 4:
        cv.drawContours(result_img, contours, i, (0, 0, 0), thickness=3)

        # 文字描述
        cv.putText(result_img, str(i), (int(x + w / 2), int(y + h / 2)), cv.FONT_HERSHEY_SIMPLEX,
                   1, (0, 0, 0), 1)

cv.imshow('result_img', result_img)
cv.waitKey()
cv.destroyAllWindows()
