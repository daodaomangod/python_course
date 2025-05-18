import pandas as pd

df5 = pd.read_csv('bj_lianJia.csv',encoding='gbk')
print(df5)
df5_5 = df5['rent']
print(df5_5)

print(df5_5.count())  # 非空元素计数
print(df5_5.min())  # 最小值
print(df5_5.max())  # 最大值

print(df5_5.idxmin())  # 最小值的位置
print(df5_5.idxmax())  # 最大值的位置

print(df5_5.sum())  # 求和
print(df5_5.mean())  # 均值
print(df5_5.median())  # 中位数

print(df5_5.var())  # 方差
print(df5_5.std())  # 标准差
print(df5_5.quantile(0.25))  # 25%分位数
print(df5_5.quantile(0.5))  # 50%分位数
print(df5_5.quantile(0.75))  # 75%分位数

print(df5_5.mode())  #众数是指一组数据中出现次数最多的数值

