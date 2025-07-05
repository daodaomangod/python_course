import cv2 as cv

def detect_faces(image_path):
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    image = cv.imread(image_path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(20, 20))

    face_count = len(faces)

    for (x, y, w, h) in faces:
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # 人群密度
    if face_count > 10:
        density_status = 'High density'
    else:
        density_status = 'Low density'

    # density_status = 'High density' if face_count > 10 else 'Low density'

    # 是否采取分流措施
    if face_count > 20:
        action_status = 'Need to divert'
    else:
        action_status = 'No action needed'

    # action_status = 'Need to divert' if face_count > 20 else 'No action needed'

    cv.putText(image, f'Faces:{face_count}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv.putText(image, density_status, (10, 70), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv.putText(image, action_status, (10, 110), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv.imwrite('4-2.png', image)

    return face_count, density_status, action_status


image_path = 'image.png'
face_count, density_status, action_status = detect_faces(image_path)

print('输出检测-判断结果')
print(f'人群数量:{face_count}')
print(f'人群密度:{density_status}')
print(f'是否分流:{action_status}')
