import cv2 as cv

cap = cv.VideoCapture('video.mp4')

cap.set(cv.CAP_PROP_POS_MSEC, 15 * 1000)

ret, frame = cap.read()

if ret:
    cv.imwrite('4_1.png', frame)
    print('成功提取视频帧')
else:
    print('提取失败')

cap.release()

# 2.0 人脸识别

