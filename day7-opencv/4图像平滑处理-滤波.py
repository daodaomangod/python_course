import cv2 as cv
import matplotlib.pyplot as plt

# 均值滤波有可能模糊图片，高斯滤波没法去掉小白点
# 读取图片
img = cv.imread('Image/image_7_1.png', 0)

# 转化灰度图
# gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

# 均值滤波
img_1 = cv.blur(img, (7, 7))

# 高斯滤波
img_2 = cv.GaussianBlur(img, (7, 7), 0)

# 中值滤波
img_3 = cv.medianBlur(img, 7)

# 显示
plt.figure(figsize=(8, 6))
plt.subplot(2, 2, 1)
plt.imshow(cv.cvtColor(img,cv.COLOR_BGR2RGB))
plt.axis('off')
plt.subplot(2, 2, 2)
plt.imshow(cv.cvtColor(img_1,cv.COLOR_BGR2RGB))
plt.axis('off')
plt.subplot(2, 2, 3)
plt.imshow(cv.cvtColor(img_2,cv.COLOR_BGR2RGB))
plt.axis('off')
plt.subplot(2, 2, 4)
plt.imshow(cv.cvtColor(img_3,cv.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

cv.imshow('image', img)
cv.imshow('junzhiimage', img_1)
cv.imshow('gaosiimage', img_2)

cv.imshow('zhongzhiimage', img_2)

cv.waitKey()
cv.destroyAllWindows()
