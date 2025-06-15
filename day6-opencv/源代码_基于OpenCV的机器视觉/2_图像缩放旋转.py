import cv2 as cv

# 1. 读取图片
img = cv.imread('image/gear.png')
print(img.shape)
width, height = img.shape[:2]
cv.imshow('Original Image', img)

# 2. 图像缩放
# 方式1：固定大小缩放
# img_dst = cv.resize(img,(128,256)))
# cv.imshow('img_dst',img_dst)

# # 方式2：固定比例缩放
# img_dst = cv.resize(img,None,fx=2,fy=2)
# cv.imshow('img_dst',img_dst)

# 3. 图像旋转
# 获取旋转矩阵
M = cv.getRotationMatrix2D((width/2, height/2), 45, 1)

# 仿射变换
rotated_img = cv.warpAffine(img, M,dsize=(width,height))
cv.imshow('Rotated Image', rotated_img)

cv.waitKey()
