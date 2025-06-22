import cv2 as cv
import easyocr

origin_img = cv.imread('card/credit_card_04.png')
# 调整尺寸
img = cv.resize(origin_img, (550, 350))
# 灰度
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# 二值图
ret, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
# 形态学处理 ！！！
kernel = cv.getStructuringElement(cv.MORPH_RECT, (9, 3))
dilate = cv.dilate(binary, kernel, iterations=3)

cv.imshow('dilate', dilate)

# 查找轮廓
contours, hit = cv.findContours(dilate, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

result = img.copy()

img_loc = []

for i, contour in enumerate(contours):
    #计算轮廓面积
    area = cv.contourArea(contour)
    # 过滤小尺寸
    if area < 500:
        continue
    # 轮廓尺寸
    x, y, w, h = cv.boundingRect(contour)
    # 轮廓宽高比
    ratio = w / float(h)
    # # 判定文字区域
    if 3 < ratio < 4 and 25 < h < 45:

        cv.rectangle(result, (x, y), (x + w, y + h), (0, 0, 225), 2)
        img_loc.append([x, y, w, h])

# 排序
img_loc = sorted(img_loc,key=lambda c:c[0])

# ocr识别器
reader = easyocr.Reader(['ch_sim','en'])
for j ,(x, y, w, h) in enumerate(img_loc):
    # 提取区域
    rio = img [y:y+h,x:x+w]
    # 读取结果
    result_ocr = reader.readtext(rio,detail=0)
    # 结果组合
    result_text = ''.join(result_ocr)

    print(f'第{j+1}个区域：结果是{result_text}')
    cv.putText(result,result_text,(x+10,y-10),cv.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

cv.imshow('Image', result)
cv.waitKey(0)
cv.destroyAllWindows()