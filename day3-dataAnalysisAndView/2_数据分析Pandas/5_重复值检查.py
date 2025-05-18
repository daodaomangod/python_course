import pandas as pd

df = pd.read_csv('bj_lianJia.csv', encoding='gbk', index_col='ID')
# print(df.head())
# print(df.info())

# DataFrame.duplicates()用来检查重复数据
print('----总行数----')
print(len(df))
#
# print('----重复行----')
# print(df.duplicated())
# print(df[df.duplicated()])
#
# print('----指定列的重复项----')
# print(df[df.duplicated(['district','area','rent'])])
#
# print('----显示不重复的“district”值----')
# print(df[df.duplicated('district')==False]['district'])


# DataFrame.drop_duplicates()用来删除重复数据
from copy import deepcopy
dff = deepcopy(df)  #深复制，不影响原来的 df
print('----删除重复行（指定inplace参数）----')
df1 =  dff.drop_duplicates(inplace=False) #原地删除重复行
print(len(df1))