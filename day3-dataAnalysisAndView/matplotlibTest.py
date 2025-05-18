import matplotlib.pyplot as plt

import numpy as np

"""

https://matplotlib.org/3.3.0/tutorials/introductory/sample_plots.html
#调整字体设置 中文
plt.rcParams['font.sans-serif']=['SimHei’]
# 默认是使用Unicode负号，设置正常显示字符  负号
plt.rcParams['axes.unicode_minus'] =False


plt.figure ()创建一个空白画布，可以指定画布大小、像素
plt.subplot()创建并选中子图，可以指定子图的行数、列数和选中图片的编号
"""

# plt.figure('画布',figsize=(4,4),
# facecolor='green')
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] =False
#plt.subplot(221).set_title("图1")

# plt.subplot(222)
# plt.xlabel("A")
plt.subplot(121)

x=np.linspace(0,10,100)
y=np.sin(x)

plt.plot(x,y,color='r',label='Sin()')
plt.xlabel("自变量X")
plt.ylabel("Sin(x)")
plt.title("函数图像")
#图例开关 必须要label！！
plt.legend()
# 网格
plt.grid()
plt.subplot(522)
plt.subplot(524)
plt.subplot(526)
plt.subplot(528)


z=np.cos(x)

plt.plot(x,z,color='b',linestyle='--'
,marker='<',label='cos()')
plt.xlabel("自变量X")
plt.ylabel("cos(x)")
plt.title("函数图像")
#图例开关 必须要label！！
plt.legend()
# 网格
plt.grid()
plt.show()

