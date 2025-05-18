import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False

# 设置画布
plt.figure('Python绘图', figsize=(10, 8))
# 定义子图之间的间隔
plt.subplots_adjust(hspace=0.4, wspace=0.4)

# 画第1个图：折线图
plt.subplot(221)
x = np.linspace(-2 * np.pi, 2 * np.pi, 50)
y = np.sin(x)
plt.plot(x, y, 'r--*')
plt.title('折线图')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.xlim(-6, 6)
plt.xticks(np.arange(-6, 6.5, 2))
plt.yticks(np.arange(-1, 1.1, 0.5))

# 画第2个图：散点图
plt.subplot(222)
colors = np.random.rand(10)
plt.scatter(np.arange(0, 10), np.random.rand(10),
            c=colors, marker='s', cmap='plasma')
plt.title('散点图')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.colorbar()

# 画第3个图：条形图
plt.subplot(223)
c = np.array(["r", "g", "b", "orange"])
plt.bar(['A', 'B', 'C', ' D'], [25, 15, 35, 30],
        color=c, width=0.5)
plt.title('条形图')
plt.xlabel('x-axis')
plt.ylabel('y-axis')

# 画第4个图：饼图
plt.subplot(224)
plt.pie(x=[16, 30, 42, 10], labels=list('ABCD'), explode=[0, 0.2, 0, 0],
        autopct='%.1f%%', shadow=True, startangle=0)
plt.title('饼图')

# 图像显示
plt.show()
