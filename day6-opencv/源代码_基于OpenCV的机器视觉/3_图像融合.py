import cv2 as cv
import numpy as np

# # 读取图片
# img1 = cv.imread('Image/image_1.jpg')
# img2 = cv.imread('Image/image_2.jpg')
#
# #图片叠加
# # add_img = cv.add(img1,img2)
# add_img = cv.addWeighted(img1, 0.3, img2, 0.8, 0)
#
# # 图片显示
# cv.imshow('add_image', add_img)
# cv.waitKey()

# 读取图片
obj = cv.imread('Image/airplane.png')  # 目标图片
background = cv.imread('Image/sky.png')  # 背景图片

# 创建一个白色的msak
# mask = 255*np.ones(obj.shape[:2], dtype=np.uint8)
mask = 255 * np.ones(obj.shape, obj.dtype)

# 图像融合
img_clone = cv.seamlessClone(obj, background, mask, (500, 300), cv.NORMAL_CLONE)

# 图片显示
cv.imshow('Image_clone', img_clone)
cv.waitKey()
