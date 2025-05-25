import pandas as pd
import seaborn as sns
import  matplotlib.pyplot as plt

#1 读取数据

df=pd.read_csv('boston.csv')

print(df.head(5))

print(df.info())

#2 相关性的分析

cor = df.corr()

print(cor)

#热力图
# plt.figure(figsize=(8,6))
# sns.heatmap(cor,annot=True,fmt='.2f',cmap='Blues')
#
# plt.show()
#排序
print(cor['MEDV'].sort_values())
