# 温度转换
# TempStr =input("请输入带有符号的温度值:")
# if TempStr[-1] in [ 'F', 'f']:
#     C=(eval(TempStr[ 0:-1])-32) / 1.8
#     print("转换后的温度是 {:.2f}C".format(C))
# elif TempStr[-1] in ['C', 'c' ]:
#     F=1.8*eval (TempStr[0:-1])+ 32
#     print("转换后的温度是{:.2f}F".format(F))
# else:
#     print("输入格式错误")
'''
这是我的注释
第二行注释
ctrl+/ 选中快速注释
shift + F10 快速运行
'''

d =input("请输入你的工资：")
if eval(d)>10000:
    print("工资不错哦",d,sep='$$$$',end=' ')
else:
    print("加油赚钱吧",d)