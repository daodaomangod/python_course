import cv2 as cv

# 确保保存文件夹存在
import os
os.makedirs('save', exist_ok=True)
# 1.读取视频
cap = cv.VideoCapture('video_test.mp4')

print(cap)

# 视频参数

w = cap.get(cv.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
print(f'视屏分辨率宽{w}*高{h}')

fps = cap.get(cv.CAP_PROP_FPS)
print(f'视频的FPS为{fps}')

frame_time = [1, 2, 5, 10, 16]

for i,sec in enumerate(frame_time):
    # 设置提取时间
    cap.set(cv.CAP_PROP_POS_MSEC, sec * 1000)

    # 提取当前帧数图像
    ret, frame = cap.read()

    print(ret)

    if ret:
        cv.imshow('image', frame)
        cv.imwrite(f'save/save_{i+1}.jpg', frame)
        print(f'图片提取成功：save/save_{i+1}.jpg')
    else:
        print("图像提取失败")

cv.waitKey(0)
cv.destroyAllWindows()
