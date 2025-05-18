import random

"""
电脑随机生成区间[1,100]的整数，让用户去猜，用户每猜一次程序自动判断：
• 如果用户猜的数字小于电脑随机生成的数字则提示“你猜小了！”；
• 如果大于，则提示“你猜大了！”；
• 如果等于，则提示“恭喜你，猜对了！”；
同时统计猜对数字共花费的次数
"""


def start_game():
    print("猜数字游戏开始！")
    random_number = random.randint(1, 100)

    guess_count = 0  # 记录次数

    while True:
        user_input = input("请输入数字（输入exit退出游戏）：")
        if user_input.lower() == "exit":
            print("游戏结束，再见！")
            print("你猜了：", guess_count,'次')
            break

        if int(user_input) == random_number:
            print(f"猜对啦！数字是{random_number},你一共猜了{guess_count}次")

            guess_count += 1
            break
        elif int(user_input) > random_number:
            print(f"你猜大了")

            guess_count += 1
        else:
            print(f"你猜小了")

            guess_count += 1
"""
编写一个函数，实现摇骰子的功能，记录每次骰子点数，最后计算N次摇骰子的点数和。
输出N=6时，每次骰子点数和N次点数总和
"""

def dice_roller():
    list = []
    count =int(input("请输入投掷的次数："))
    for i in range(count):
        list.append(random.randint(1,6))
    print(f"每次投掷的点数为{list}，点数总和为{sum(list)}")






if __name__ == "__main__":
    # start_game()
   dice_roller()

