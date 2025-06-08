import cv2 as cv
import numpy as np

# 1. 读取图像

img = cv.imread("Image/image_girl.jpg")
# x y轴取像素点修改！！
img[25:125,25:125]=[0,0,0]
print(img.shape)
# 图像拆分  分割通道  顺序是bgr 不是标准的RGB
b, g, r = cv.split(img)
# 2.显示
cv.imshow('b', b)
cv.imshow('g', g)
cv.imshow('r', r)
cv.imshow('girl', img)

# 图像合成
img1=cv.merge([r,g,b])
cv.imshow('Image1',img1)

# 3.保存
cv.imwrite('save.jpg', img)
# 停留，单位是毫秒   0代表一直存在  或者不写也是一直存在
cv.waitKey()

cv.destroyAllWindows()  # 所有！窗口全部关闭
