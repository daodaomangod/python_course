import math
import random


# 定义的函数
def count_x(x):
    if 0 <= x < 5:
        y = math.pow(x, 2) + math.sqrt(x) + 1
    elif 5 <= x < 10:
        y = math.exp(x) + math.cos(x)
    elif 10 <= x < 20:
        y = math.sin(math.pi * x)
    else:
        y = 0
    return y


# x = eval(input("请输入x的值:"))
# print(f"输入值:{x},函数的计算成果{count_x(x):.3f}")

"""
ramdom.seed()
ramdom.random()
描述
初始化随机数种子，默认为当前系统时间。
生成一个[0 , 1)之间的随机小数。
>>> ramdom.random()
random.uniform(a,b)生成一个[a , b]之间的随机小数。
ramdom.randint(a,b)生成一个[a , b]之间的随机整数。
"""

