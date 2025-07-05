import cv2 as cv


# 1. 定义提取视频帧的函数
def extract_frame(video_path, frame_times):
    video_frames = []
    cap = cv.VideoCapture(video_path)
    for i, sec in enumerate(frame_times):
        cap.set(cv.CAP_PROP_POS_MSEC, sec * 1000)
        ret, frame = cap.read()
        if ret:
            cv.imwrite(f'save/frame_{i + 1}.png', frame)
            video_frames.append(frame)
            print(f'成功提取了视频帧，保存图片为：frame_{i + 1}.png')
        else:
            print('视频帧提取不成功')
    print('视频帧提取完成')
    return video_frames
video_path = 'video_test.mp4'
frame_times = [1, 10, 15,20, 30]
frames = extract_frame(video_path, frame_times)

# 2.亮度分析
import numpy as np


def calculate_average_brightness(image_path):
    img = cv.imread(image_path)
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    value = img_hsv[:, :, 2]
    avg_brightness = np.mean(value)
    print(f'{image_path}的平均亮度为{avg_brightness}')
    return avg_brightness


frame_brightness = []
for i in range(len(frame_times)):
    img_path = f'save/frame_{i}.png'
    avg_value = calculate_average_brightness(img_path)
    frame_brightness.append(avg_value)
    print(f'图像frame_{i}.png的平均亮度：{avg_value}')

print('图像亮度分析完成')

# 3. 图片展示

import matplotlib.pyplot as plt

def display_image(image, brightness):
    n = len(image)
    plt.figure(figsize=(8, 6))
    for j, (frame, bv) in enumerate(zip(image, brightness)):
        plt.subplot(1, n, j + 1)
        img_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.axis('off')
        plt.title(f'frame_{j}.png')
        plt.text(400, 900, f'Avg Brightness:{bv:.2f}')

    plt.show()


display_image(frames, frame_brightness)


