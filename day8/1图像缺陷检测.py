import cv2 as cv

import numpy as np

#  1. 图片分割

img = cv.imread('image/img_2.png')
# 调整尺寸
img_1 = cv.resize(img, (600, 500))

# 转化灰度图
gray = cv.cvtColor(img_1, cv.COLOR_BGR2GRAY)

# 二值化
# ret, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)

ret, binary=cv.threshold(gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)


# 查找轮廓
contours, hit = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

print(len(contours))
contours_item = []
# 图像副本
result_img = img_1.copy()
for i, contour in enumerate(contours):
    x, y, w, h = cv.boundingRect(contour)
    # # 画图
    # cv.drawContours(result_img, contours, i, (0, 0, 255), thickness=2)
    # # # 画方框
    # cv.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 3)

    contours_item.append((x, y, w, h))
    # 分割
    # img_rio = img_1[y - 10:y + h + 10, x - 10:x + w + 10]
    # cv.imwrite(f'save/save_img_{i}.png', img_rio)

# 2. 缺陷判断
# 遍历
for index, (x, y, w, h) in enumerate(contours_item):
    # 分割
    img_rio = img_1[y:y + h, x:x + w]
    # 转化灰度图
    gray_img_rio = cv.cvtColor(img_rio, cv.COLOR_BGR2GRAY)

    # 二值化
    ret_img_rio, binary_img_rio = cv.threshold(gray_img_rio, 127, 255, cv.THRESH_BINARY)

    # 查找轮廓 sIMPLE只绘制端点  CHAIN_APPROX_SIMPLE
    contours_img_rio, hit_img_rio = cv.findContours(binary_img_rio, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

    # for j, cont in enumerate(contours_img_rio):
    #     cv.drawContours(img_rio, contours_img_rio, j, (0, 0, 255), thickness=2)
    #
    # cv.imshow('1', gray_img_rio)
    # cv.imshow('2', binary_img_rio)
    # cv.imshow('3', img_rio)

    print(len(contours_img_rio))

    flag = False
    for j, contour_rio in enumerate(contours_img_rio):
        # 计算面积
        area = cv.contourArea(contour_rio)
        # 判断是否有缺陷
        if 5000 > area > 50:  # 依据面积大小
            flag = True
            # 缺陷全局坐标
            global_cont = contour_rio + np.array([x, y])

            # 画缺陷轮廓
            # 画图  绘制缺陷
            cv.drawContours(result_img, global_cont, -1, (0, 0, 255), thickness=2)
            # 画方框
        # cv.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 3)

    print(flag)
    if flag:
        cv.rectangle(result_img, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cv.putText(result_img,'Crack',(x+5,y+20),cv.FONT_HERSHEY_SIMPLEX,0.7,(0, 0, 255),1)
    else:
        cv.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv.putText(result_img,'Pass',(x+5,y+20),cv.FONT_HERSHEY_SIMPLEX,0.7,(0, 255, 0),1)



# cv.imshow('img', img_1)
#
# cv.imshow('gray', gray)
cv.imshow('result_img', result_img)
cv.waitKey()
cv.destroyAllWindows()
