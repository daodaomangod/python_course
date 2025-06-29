import cv2 as cv

cap = cv.VideoCapture('video_test.mp4')


frame_times = [1, 2, 4, 6, 8]

for i, sec in enumerate(frame_times):
    cap.set(cv.CAP_PROP_POS_MSEC, sec * 1000)
    ret, frame = cap.read()
    cv.imwrite(f'save_{i}.jpg', frame)
