import cv2 as cv
import numpy as np

img = cv.imread('save_0.jpg')

img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
# 色调 饱和度 亮度
# 全选，然后取亮度
value_channel = img_hsv[:, :, 2]

avg_brightness = np.mean(value_channel)
print(avg_brightness)

cv.imshow('image', img_hsv)

cv.waitKey()
cv.destroyAllWindows()
