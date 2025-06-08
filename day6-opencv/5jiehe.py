import cv2 as cv

#读取图片
img_1 = cv.imread('Image/sky.png')

img_2 = cv.imread('Image/image_girl.jpg')

#缩放图片
img_1_1=cv.resize(img_1,(400,600))
img_2_1=cv.resize(img_2,(400,600))

# 图像融合
# img_add=cv.add(img_1,img_2)
# 按照权重的方式添加
img_add=cv.addWeighted(img_1_1,0.1,img_2_1,0.9,1)
cv.imshow('img_add',img_add)
cv.waitKey()
cv.destroyAllWindows()