import cv2 as cv

# 读取
img = cv.imread('Image/Lena.png')
# print(img.shape)

# 提取图像的宽度，高度，以及通道数
w, h, s = img.shape

# 缩放
img_1 = cv.resize(img, (256, 256))
# img_2 = cv.resize(img,None,fx=0.5,fy=0.8)

# 旋转矩阵
M = cv.getRotationMatrix2D((w / 2, h / 2), 25, 1)

# 旋转图像
img_3 = cv.warpAffine(img, M, (w, h))

#图像切割

img_4 =img[70:450,50:400]

# 保存
#cv.imwrite('Image/img_save_rot.png', img_3)

# 显示
cv.imshow('Image', img)
cv.imshow('img_4', img_4)

cv.waitKey()
cv.destroyAllWindows()