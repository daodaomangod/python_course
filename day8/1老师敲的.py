import cv2 as cv
import numpy as np

# 1. 图像分割

img = cv.imread('image/img_2.png')

# 调整尺寸
img_1 = cv.resize(img, (600, 400))
# 转换灰度图
gray = cv.cvtColor(img_1, cv.COLOR_BGR2GRAY)
# 二值化
ret, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
# 查找轮廓
contours, hit = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# 图像副本
result_img = img_1.copy()
contours_item = []

for i, contour in enumerate(contours):
    # 轮廓位置
    x, y, w, h = cv.boundingRect(contour)
    # 保存数据
    contours_item.append((x, y, w, h))

# # 2. 缺陷判断
# 遍历
for index, (x, y, w, h) in enumerate(contours_item):
    # 分割
    img_rio = img_1[y:y + h, x:x + w]
    # 转换灰度图
    gray_rio = cv.cvtColor(img_rio, cv.COLOR_BGR2GRAY)
    # 二值化
    ret_rio, binary_rio = cv.threshold(gray_rio, 127, 255, cv.THRESH_BINARY)

    # 查找轮廓
    contours_crack, hit_crack = cv.findContours(binary_rio, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

    crack = False

    for j, cont in enumerate(contours_crack):
        # 计算面积
        area = cv.contourArea(cont)
        # 判定是否有缺陷
        if 50 < area < 5000:  # 依据
            crack = True
            # 缺陷全局坐标
            global_cont = cont + np.array([x, y])
            # 绘制缺陷
            cv.drawContours(result_img, global_cont, -1, (0, 0, 255), thickness=2)

    if crack:
        cv.rectangle(result_img, (x, y), (x + w, y + h), (0, 0, 255), 3)

        cv.putText(result_img, 'Crack', (x + 5, y + 20), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    else:
        cv.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 3)

        cv.putText(result_img, 'Good', (x + 5, y + 20), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

cv.imshow('img', result_img)
cv.waitKey(0)
cv.destroyAllWindows()