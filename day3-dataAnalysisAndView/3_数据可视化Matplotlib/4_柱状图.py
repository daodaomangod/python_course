import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False

# # 生成数据
# labels = np.array(["Num-1", "Num-2", "Num-3", "C-Team"])
# y = np.array([12, 22, 6, 18])
# # 设置点的颜色
# colors = np.array(["r", "g", "b", "orange"])
# # 绘制状态图
# plt.bar(labels, y, color=colors, width=0.5)
# # 设置标题
# plt.title('Python绘制柱状图', fontsize=16)
# plt.xlabel('分组', fontsize=14)
# plt.ylabel('数据', fontsize=14)
# plt.show()

# # 生成数据
# labels = np.array(["Num-1", "Num-2", "Num-3", "C-Team"])
# y = np.array([12, 22, 6, 18])
# # 设置点的颜色
# colors = np.array(["r", "g", "b", "orange"])
# # 绘制状态图
# plt.barh(labels, y, color=colors, height=0.5)
# # 设置标题
# plt.title('Python绘制柱状图', fontsize=16)
# plt.xlabel('数据', fontsize=14)
# plt.ylabel('分组', fontsize=14)
# plt.show()
#
# 生成数据
labels = np.array(["Num-1", "Num-2", "Num-3", "C-Team"])
y1 = np.array([12, 22, 6, 18])
y2 = np.array([22, 10, 5, 28])
# x轴刻度标签位置
x = np.arange(len(labels))
width = 0.3
# 计算每个柱子在x轴上的位置，保证x轴刻度标签居中
# x - width/2，x + width/2即每组数据在x轴上的位置
# 绘制状态图
plt.bar(x - width / 2, y1, color='r', width=width, label='G1')
plt.bar(x + width / 2, y2, color='g', width=width, label='G2')
# 设置标题
plt.title('Python绘制柱状图', fontsize=16)
plt.xlabel('分组', fontsize=14)
plt.ylabel('数据', fontsize=14)
# x轴刻度标签位置不进行计算
plt.xticks(x, labels=labels)
plt.legend()
plt.show()
