import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# 1. 定义提取视频帧的函数
def extract_frame(video_path, frame_times):
    cap = cv.VideoCapture(video_path)
    video_frames = []
    for i, sec in enumerate(frame_times):
        cap.set(cv.CAP_PROP_POS_MSEC, sec * 1000)
        ret, frame = cap.read()
        if ret:
            video_frames.append(frame)
            cv.imwrite(f'frame_{i}.png', frame)
            print(f'成功提示了视频帧，保存图片为：frame_{i}.png')
        else:
            print('视频帧提取不成功')
    print('视频帧提取完成')
    return video_frames


video_path = 'video_test.mp4'
frame_times = [2, 4, 5, 6, 10]
frames = extract_frame(video_path, frame_times)
print()


# 2. 亮度分析
def calculate_average_brigntness(image_path):
    img = cv.imread(image_path)
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    value = img_hsv[:, :, 2]
    agv_brightness = np.mean(value)
    return agv_brightness


frame_brightness = []
for i in range(len(frame_times)):
    img_path = f'frame_{i}.png'
    avg_value = calculate_average_brigntness(img_path)
    frame_brightness.append(avg_value)
    print(f'图像frame_{i}.png的平均亮度：{avg_value}')
print('图像亮度分析完成')
print()


# 3. 图片展示
def display_image(images, brightnesses):
    n = len(images)

    plt.figure(figsize=(12, 6))
    for j, (image, bv) in enumerate(zip(images, brightnesses)):
        plt.subplot(1, n, j + 1)
        img_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.axis('off')
        plt.title(f'frame_{j}.png')
        plt.text(100, 900, f'Brightnesses= {bv:.2f}')
    plt.show()


display_image(frames, frame_brightness)
print('图片展示完成')
