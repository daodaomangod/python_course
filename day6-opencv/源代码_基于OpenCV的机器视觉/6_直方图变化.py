import cv2 as cv
import matplotlib.pyplot as plt

#
# # 灰度图像读取
# img = cv.imread('Image/Lena.png', 0)
#
# # 直方图均衡化，像素个数多的灰度级拉的更宽，对像素个数少的灰度级进行压缩
# img_hist = cv.equalizeHist(img)
#
# # 显示原图和直方图均衡化之后的图
# cv.imshow('Gary Image', img)
# cv.imshow('Equalized Image', img_hist)
#
# cv.waitKey(0)
# cv.destroyAllWindows()
#
# # 计算出直方图cv.calcHist()
# # 原图
# hist = cv.calcHist(img, [0], None, [256], [0, 256])
# # 直方图均衡化
# hist_1 = cv.calcHist(img_hist, [0], None, [256], [0, 256])
#
# # 使用matplotlib绘制直方图
# plt.subplots_adjust(wspace=0.3)
# plt.subplot(1, 2, 1)  # 绘图设置
# plt.plot(hist)
# plt.title('Original Image')
# plt.xlabel("Pixel Intensity")
# plt.ylabel('Pixel Count')
#
# # 计算出直方图
# plt.subplot(1, 2, 2)
# plt.plot(hist_1, color='r')
# plt.title('Color Histogram')
# plt.xlabel("Pixel Intensity")
# plt.ylabel('Pixel Count')
#
# plt.show()

# 读取图像
img = cv.imread('Image/stone.png')
# RGB 分别均衡化
b, g, r = cv.split(img)  # 分别提取三通道数据
eq_b = cv.equalizeHist(b)
eq_g = cv.equalizeHist(g)
eq_r = cv.equalizeHist(r)
img_hist = cv.merge([eq_b, eq_g, eq_r])
# 显示图像
cv.imshow('Original', img)
cv.imshow('Equalized (BGR)', img_hist)
cv.waitKey(0)
cv.destroyAllWindows()

# 绘制直方图
def plot_histogram(img, title):
    colors = ('b', 'g', 'r')
    plt.figure(figsize=(8, 6))
    plt.title(title)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    for i, color in enumerate(colors):
        hist = cv.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
    plt.xlim([0, 256])
    plt.show()

plot_histogram(img, 'Original')
plot_histogram(img_hist, 'Histogram')
