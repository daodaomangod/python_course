'''
列表list的数据可以替换更改
元祖tuple的数据不可修改
数字类型 int（） float（） 复数complex（）
运算符  /表示除   //表示整除 %取余数    **幂运算
      +x 代表x的本身 -x代表x的复数
'''

a =1
b=1.234
c=a+b
print(isinstance (a,float))
print(type(c))
print(c)
print(a//b)
print(a%b)
print(-a)
print(not a)
print(a**b)
print(int(b))
print(complex(b))
print(complex(c))
print(a!=b)
print(a--b)
print(a++b)

'''
函数方法
正整数

'''
# 绝对值
print(abs(-a))
# 取商和余数
print(divmod(10,3))
# 幂运算
print(pow(10,4))
# 保留小数
print(round(b,1))
print(round(a,0))
# 进步与退步0.001 千分之一的数据
print((1+0.001)**365)
print((1-0.001)**365)
# 进步百分之一 37倍
print((1+0.01)**365)
result1 =pow((1+0.001),365)
result2=pow((1-0.001),365)
print(round(result1,3),round(result2,3),sep='   进步与退步0.001  ')


# changePercent=input("请输入变化的比例：")
#
# print("进步{:.2f}".format(eval(changePercent)),pow((1+eval(changePercent)),365),sep=",  ")

#字符串string

'''
x+ y 连接字符串
str[n:m]提取片段  >=   <  [)

'''

str1='hello world'
print(type(str1))

str2='打招呼'
print(str1*3+str2)
print("h" in str1)
print('h' == str1)
print('h' in str1)
print(str1[0])
print(str1[-2])

print(str1[0:4])
print(str1[:])
print(str1[6:])
print(str1[:6])
print("str1{},str2{}".format(str1,str2))
print(f"第一各字符{str1}---,第二个字符---{str2}")
print(f"缩短浮点数{3.13423232:.3f}")
print("str1{:.2f},str2{}".format(1.32323232,str2))

#list 列表
a=[1,2,3,4,5,5]
item=(1,2,232,232,3232)
print(type(a))
print(type(item))
print(a[2:4])

st=[[1,2,3],[3,4,5],'hellow world',4]
print(st[0][1])
print(st[1][2])
print(st[0:2])
# in   not in
print(1 in st[0])
a[2]=33
print(a)
print(a*4)
print(a+st)
# list添加元素
print(a.append(33333))
# list 复制
st_copy=st.copy();
del st_copy[:2]
print(st_copy)
st_copy.insert(0,'daodao')
print(st_copy)


