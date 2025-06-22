import easyocr
import cv2 as cv

# 加载模型 指定语言中文和英文
reader = easyocr.Reader(['ch_sim', 'en'], gpu=True)

# 读取图像并识别文字
img = cv.imread('image/image.jpg')
result = reader.readtext(img, detail=0)

# for (bbox, text, prob) in result:
#     print(f'识别到的文本：{text},置信度：{prob}')
print(result)
