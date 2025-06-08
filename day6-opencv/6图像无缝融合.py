import cv2 as cv
import  numpy as np


#读取图片
img_obj = cv.imread('Image/airplane.png')

img_g = cv.imread('Image/sky.png')

#mask

w, h, s=img_obj.shape
mask = np.ones([w+1,h],dtype=np.uint8)*255
#融合
img_clone=cv.seamlessClone(img_obj,img_g,mask,(500,300),cv.NORMAL_CLONE)
img_clone2=cv.seamlessClone(img_obj,img_g,mask,(500,300),cv.MIXED_CLONE)
img_clone3=cv.seamlessClone(img_obj,img_g,mask,(500,300),cv.MONOCHROME_TRANSFER)
cv.imshow('img_clone',img_clone)
cv.imshow('img_clone2',img_clone2)
cv.imshow('img_clone3',img_clone3)
cv.waitKey()
cv.destroyAllWindows()