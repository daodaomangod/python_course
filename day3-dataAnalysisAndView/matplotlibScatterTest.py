import matplotlib.pyplot as plt

import numpy as np

"""
散点图（Scatter Diagram）又称为散点分布图，是以一个特征为横坐标，以另一个特征为纵坐
标，利用坐标点（散点）的分布形态反映特征间的统计关系的一种图形。
 matplotlib.pyplot.scatter (x, y, s=None, c=None, marker=None, cmap=None, norm=None,
vmin=None, vmax=None, alpha=None)
必备参数
• x，y：长度相同的数组，也就是我们即将绘制散点图的数据点，输入数据。
• s：点的大小，默认 20，也可以是个数组，数组每个参数为对应点的大小。
• c：点的颜色，默认蓝色 'b'，也可以是个 RGB 或 RGBA 二维行数组。
• marker：点的样式，默认小圆圈 'o’。
• cmap：Colormap颜色映射集，默认 None，标量或者是一个 colormap 的名
"""


# x=np.linspace(0,10,100)
# y=np.sin(x)
#
# plt.scatter(x,y,color='r',label='Sin()')
# plt.xlabel("自变量X")
# plt.ylabel("Sin(x)")
# plt.title("函数图像")
#散点图
# x=np.array([1,2,3,4,5,6,7])
# y=np.array(([2,4,6,8,10,12,1]))
# size=np.array([20,40,30,50,80,40,40])
# color = np.array(['r','b','g','k','y','c','c'])
# plt.scatter(x,y,s=size,c=color)
# plt.show()
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
#柱状图
plt.subplot(121)
x= np.array(("A","B","C","D"))
y=np.array([10,15,32,4])
color = np.array(['r','b','g','k'])

plt.bar(x,y,width=0.5,color=color)
plt.subplot(122)
x= np.array(("A","B","C","D"))
y=np.array([10,15,32,4])
y2=np.array([30,5,18,30])
color = np.array(['r','b','g','k'])
# 设置柱状图宽度和位置
bar_width = 0.35
index = np.arange(len(x))

# 绘制双柱状图
plt.bar(index,y,width=bar_width,color=color)
plt.bar(index,y2,width=bar_width,color=color)

plt.xlabel("")






plt.show()
