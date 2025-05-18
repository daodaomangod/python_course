import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False

# 读取csv数据
df = pd.read_csv('newbj_lianJia.csv', encoding='gbk')
print(df.head())
area = df['area'].values.tolist()  # 房屋面积
print(area)
rent = df['rent'].values.tolist()  # 房屋房租
print(rent)

plt.figure('面积和价格',figsize=(8, 6))
colors = np.random.rand(len(area))
plt.scatter(area,rent,s=40,c=colors,cmap='plasma')
plt.xlabel('面积',fontsize=16)
plt.ylabel('租金/元',fontsize=16)
plt.title('面积 vs 租金',fontsize=20)

plt.colorbar()
plt.show()

