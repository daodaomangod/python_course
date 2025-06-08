import cv2 as cv

# 灰度图 1
img_1 = cv.imread('Image/Lena.png', 0)

# 灰度图2
img = cv.imread('Image/Lena.png')
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

cv.imshow('Image1', img_1)

cv.waitKey()
cv.destroyAllWindows()
