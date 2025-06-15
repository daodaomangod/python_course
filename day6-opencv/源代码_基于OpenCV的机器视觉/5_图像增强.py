import numpy as np
import cv2 as cv
import copy


# 定义reverse反转函数
def reserve(img):
    img_new = copy.deepcopy(img)  # 拷贝图像
    rows = img.shape[0]  # 宽度属性
    cols = img.shape[1]  # 高度属性
    # 对图像的每一个像素进行gamma函数变换
    for i in range(rows):
        for j in range(cols):
            img_new[i, j] = 255 - img_new[i, j]
    return img_new

# 定义Log变化函数
def log_tran(img):
    """log灰度变换"""
    img_new = copy.deepcopy(img).astype(np.float32)  # 拷贝图像,转换为浮点型
    rows = img.shape[0]
    cols = img.shape[1]
    # 对图像的每一个像素进行log函数变换
    for i in range(rows):
        for j in range(cols):
            img_new[i, j] = 45 * np.log(img_new[i, j] + 1)
            # 归一化到0-255范围
            cv.normalize(img_new, img_new, 0, 255, cv.NORM_MINMAX)
    return img_new.astype(np.uint8)

# 定义gamma变化函数
def gamma_tran(img):
    """gamma灰度变换"""
    img_new = copy.deepcopy(img)  # 拷贝图像
    rows = img.shape[0]  # 宽度属性
    cols = img.shape[1]  # 高度属性
    # 对图像的每一个像素进行gamma函数变换
    for i in range(rows):
        for j in range(cols):
            img_new[i, j] = 4* pow(img_new[i, j], 0.8)
    return img_new


# 灰度图
img = cv.imread('Image/gear.png', 0)
cv.imshow('Image', img)

# # 图像反转
# img_reserve = reserve(img)
# cv.imshow('Image_reserve', img_reserve)


# # log灰度增强
# img_log = log_tran(img)
# cv.imshow('Image_log', img_log)

# gamma灰度增强
img_gamma = gamma_tran(img)
cv.imshow('Image_gamma', img_gamma)

cv.waitKey()
