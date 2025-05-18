# <安装>：anaconda模式  conda install numpy
"""
Numpy是数据分析计算生态的基础库，是Pandas 、Matplotlib等支撑依赖库
array()将输入数据（列表、元组等序列类型）转换为一维或多维数组
arange()类似于range函数，返回一个一维数组，arange([start, ]stop, [step, ])
linspace()
设置起始和结束区间，均匀地产生指定个数的数字，组成一维数组
linspace ( start, stop, num= )
zeros( m, n )创建一个m行n列全1的二维数组(矩阵)，dtype控制数据类型
ones( m, n )创建一个m行n列全0的二维数组(矩阵)，dtype控制数据类型
empty()空数组，只申请空间，不初始化
diag()创建对角矩阵 np.diag([1,2,3])  对角是1-2-3的矩阵
identity()创建单位矩阵 np.diag(4)  4×4 单位矩阵

ndarray是同种元素的一维或多维数组对象，拥有如下基本属性：
ndim 数组维数
shape 形状
size 元素总数
dtype 数组中元素的数据类型


np.ramdom.random()生成指定形状的[0, 1)区间内的的随机数
np.ramdom.rand()生成指定形状的[0, 1)区间内服从均匀分布的随机数
np.ramdom.randn()生成指定形状的[0, 1)区间内服从正态分布的随机数
np. ramdom.randint(a,b,size=[2,3])
np.ramdom.normal()
生成一个2× 3的[a , b]区间内的随机整数

ndarrary. reshape (m.n) ： 改变现有数组的维度，返回维度(m , n)数组。
"""
import numpy as np

array1=np.array([[1,2,3,4],[2,3,4,5]])






