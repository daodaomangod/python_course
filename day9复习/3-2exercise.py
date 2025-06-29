import cv2 as cv

# 加载Haar模型

face_haarcascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 读取图片
image = cv.imread('image_people1.jpg')

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# 人脸识别
faces = face_haarcascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                          minSize=(20, 20), maxSize=(100, 100))

print(faces)
# 可视化
for (x, y, w, h) in faces:
    cv.rectangle(image, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=3)

# 结果展示
cv.imshow('faces Result',image)
cv.waitKey()
cv.destroyAllWindows()
