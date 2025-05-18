import pandas as pd

df = pd.read_csv('bj_lianJia.csv', encoding='gbk', index_col=['ID'])
# print('----打印含有空值的行数----')
# print(df.isnull())
# print(df.isnull().sum())


#dropna()函数删除带有缺失值的数据行
from copy import deepcopy
dff = deepcopy(df)

# print('----删除缺失值的那些行----')
# dff.dropna(how = 'any', inplace=True)
# print(dff.isnull().sum())

print('----删除lift列中有缺失值的行----')
dff.dropna(subset = ['lift'], inplace=True)
print(dff.isnull().sum())

