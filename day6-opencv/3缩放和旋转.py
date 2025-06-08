import cv2 as cv

# 读取
img = cv.imread('Image/Lena.png')
w ,h, s=img.shape
# 缩放
img1 = cv.resize(img, (125, 125))
img2 = cv.resize(img, None, fx=0.5, fy=0.8)

#旋转
# 旋转矩阵
M = cv.getRotationMatrix2D((w/2,h/2),45,1)

#旋转图像
img_3 =cv.warpAffine(img,M,(w,h))

# 显示
cv.imshow('Image', img)
cv.imshow('Image1', img1)
cv.imshow('Image2', img2)
cv.imshow('Image3', img_3)
cv.waitKey()
cv.destroyAllWindows()

'''
import cv2 as cv

# 读取图像
img = cv.imread('Image/Lena.png')

# print(img)

# 像素索引
print(img[25:125, 25:125])

img[200:256, 200:256] = [255, 255, 255]

# 图像拆分
b, g, r = cv.split(img)

# 图像合成
img_1 = cv.merge([r, g, b])

# 2. 显示
cv.imshow('b', b)
cv.imshow('r', r)
cv.imshow('g', g)

cv.imshow('Image', img)
cv.imshow('Image_1', img_1)

# 3.保存
cv.imwrite('Image/save.jpg', img)

cv.waitKey()
'''