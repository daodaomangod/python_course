import cv2 as cv
#彩色的
img=cv.imread('Image/gear.png')

#灰度图
img_gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

img_gray_2=img_gray

# 二值图 retval实际阀值，dst是处理后图像， thresh设置为0,采用otsus二值化
# retval,dst=cv.threshold(img_gray,127,255,cv.THRESH_BINARY)
# retval2,dst2=cv.threshold(img_gray,127,255,cv.THRESH_BINARY_INV)
retval,dst=cv.threshold(img_gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
retval2,dst2=cv.threshold(img_gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)


cv.imshow('Image1', img)
cv.imshow('img_gray', img_gray)
cv.imshow('img_th', dst)
cv.imshow('dst2', dst2)
cv.waitKey()
cv.destroyAllWindows()


'''
import cv2 as cv

# 彩色
img = cv.imread('Image/gear.png')
# 灰度图
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# w, h = img_gray.shape
# for i in range(w):
#     for j in range(h):
#         img_gray[i,j] = img_gray[i,j]*2
#
#         if img_gray[i,j]>255:
#             img_gray[i, j]=255

# 二值图
ret1, dst1 = cv.threshold(img_gray, 127, 255, cv.THRESH_BINARY)
ret2, dst2 = cv.threshold(img_gray, 127, 255, cv.THRESH_BINARY_INV)

cv.imshow('Image', img)
cv.imshow('Image Gray', img_gray)
cv.imshow('Image Threshould_1', dst1)
cv.imshow('Image Threshould_2', dst2)

cv.waitKey()
'''

