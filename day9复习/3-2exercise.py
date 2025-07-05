import cv2 as cv

# 加载Haar模型
face_haarcascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
# 读取图片
image = cv.imread('image_people2.jpg')
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
# 人脸识别
faces = face_haarcascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                          minSize=(50, 50))
print(faces)
face_count = len(faces)
if face_count > 10:
    density_status = 'High density'
else:
    density_status = 'Low density'
# 是否采取分流措施
if face_count > 20:
    action_status = 'Need to divert'
else:
    action_status = 'No action need'
cv.putText(image, f'Faces:{face_count}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv.putText(image, density_status, (10, 70), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv.putText(image, action_status, (10, 110), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv.imwrite('4-2.png', image)
# 可视化
for (x, y, w, h) in faces:
    cv.rectangle(image, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=3)
print('输出检测-判断结果')
print(f'人群数量:{face_count}')
print(f'人群密度:{density_status}')
print(f'是否分流:{action_status}')
# 结果展示
cv.imshow('faces Result', image)
cv.waitKey()
cv.destroyAllWindows()


