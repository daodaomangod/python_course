import pandas as pd

# 1. 读取文件
df = pd.read_csv('bj_lianJia.csv', encoding='gbk')
print(df.info())
print(df.head())

# 2.提取信息
info=df[(df['rent'] > 3000) & (df['rent'] < 5000)]['rent'].mean().round(2)

print(f'平均房租：{info}')
