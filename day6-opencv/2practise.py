import cv2 as cv
import numpy as np

# 生成随机bgr像素点
data = np.random.randint(0,255,size=[512,512,3],dtype=np.uint8)

cv.imshow('Image',data)

cv.waitKey()
cv.destroyAllWindows()

