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


#二值化
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
    # 绘制轮廓
    cv.drawContours(result_img, contours, i, (0, 0, 0), thickness=3)
    # 计算面积
    area = cv.contourArea(contours[i])
    print(f'计算面积:{area}')
    # 计算周长
    length = cv.arcLength(contours[i], closed=True)
    print(f'计算周长:{length:.2f}')

    x, y, w, h = cv.boundingRect(contours[i])
    # 文字描述
    cv.putText(result_img,str(i),(int(x+w/2),int(y+h/2)),cv.FONT_HERSHEY_SIMPLEX,
               1,(0,0,0),1)
    # 绘制图形
    cv.rectangle(result_img, (x - 10, y - 10), (x + w + 10, y + h + 10), color=(0, 0, 0))

# cv.imshow('image', img)
# cv.imshow('canny', canny)
cv.imshow('result_img', result_img)
cv.waitKey()
cv.destroyAllWindows()
