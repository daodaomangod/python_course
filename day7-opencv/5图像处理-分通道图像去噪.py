import cv2 as cv
import matplotlib.pyplot as plt

# 均值滤波有可能模糊图片，高斯滤波没法去掉小白点
# 读取图片
img = cv.imread('Image/微信图片_20250615103058_273.jpg')

# 拆分通道
b,g,r = cv.split(img)

# 均值滤波
# img_1 = cv.blur(img, (7, 7))

# 高斯滤波
# img_2 = cv.GaussianBlur(img, (7, 7), 0)

# 中值滤波
img_b = cv.medianBlur(b, 7)
img_g = cv.medianBlur(g, 7)
img_r = cv.medianBlur(r, 7)
#合并
img_new = cv.merge([img_b,img_g,img_r])
# 显示
plt.figure(figsize=(8, 6))
plt.subplot(2, 2, 1)
plt.imshow(cv.cvtColor(img,cv.COLOR_BGR2RGB))
plt.axis('off')
plt.subplot(2, 2, 2)
plt.imshow(cv.cvtColor(img_new,cv.COLOR_BGR2RGB))
plt.axis('off')

plt.show()

cv.imshow('image', img)
cv.imshow('img_new', img_new)


cv.waitKey()
cv.destroyAllWindows()
