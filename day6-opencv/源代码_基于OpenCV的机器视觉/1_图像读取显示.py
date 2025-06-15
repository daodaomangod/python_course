import cv2 as cv
import numpy as np

# 1. 图片读取
image = cv.imread('Image/Lena.png')
# image = cv.imread('Lena.png')

# #访问某一个位置的图像像素
# print("像素值[200,150]:", image[200,150])
# print("像素值蓝色通道[200,150]:", image[200,150][0])
# print("像素值绿色通道[200,150]:", image[200,150][1])
# print("像素值红色通道[200,150]:", image[200,150][2])
#
# # 将一个区域的像素颜色修改为白色
# image[100:300,100:300] = (255, 255, 255)

# 2. 图像显示
cv.imshow('Display Image', image)

# 3. 图像保存
cv.imwrite('image_save.jpg', image)

# 4. 图像属性
print(image)
print(image.shape)
print(image.dtype)

# 5. 图像保持显示
# 等待用户命令, 参数: 等待时间（毫秒），0表示无限等待
cv.waitKey(0)

# 随机生成图片
# 颜色范围0-255，大小256*256,三通道彩色图片
# 数据类型，uint8： 8位无符号的整数
# 常用的图像数据类型uint8 和 float32
random_img = np.random.randint(2, 255, size=[256, 256, 3], dtype=np.uint8)

# 图像显示
cv.imshow('Random Image', random_img)

cv.waitKey()
