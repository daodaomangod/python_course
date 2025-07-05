import cv2 as cv

# 创建视频捕获对象，加载视频文件
cap = cv.VideoCapture('video.mp4')

# 设置视频位置
cap.set(cv.CAP_PROP_POS_MSEC, 10 * 1000)

ret, frame = cap.read()

if ret:
    cv.imwrite('4-1.png', frame)
    print('成功提取视频帧')
else:
    print('提取失败')

# 释放视频捕获对象
cap.release()



