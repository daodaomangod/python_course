import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('save_0.jpg')

img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

plt.figure(figsize=(12, 8))
plt.subplot(1, 5, 1)
plt.imshow(img_rgb)
plt.axis('off')
plt.title('save_0.jpg', fontsize=12)
# plt.xlabel('avg_brightness=123')
plt.text(600, 800, 'vg_brightness', color='r', fontsize=12, backgroundcolor='w')
plt.show()
