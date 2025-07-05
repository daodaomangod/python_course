import torch
from torch import nn  # 导入神经网络模块

# 判断是否可以用GPU
cuda_available = torch.cuda.is_available()
map_available = torch.backends.mps.is_available()  # 检查MPS的可用性
if cuda_available:
    device = 'cuda'
elif map_available:
    device = 'mps'     # mps是苹果m系列芯片的GPU
else:
    device = 'cpu'
print(f'Using device is {device}')

# 构建神经网络
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()  # 创建一个 nn.Flatten 层，用于将输入的二维图像数据展平成一维向量
        self.hidden1 = nn.Linear(28 * 28, 128)  # 第一个隐藏层，输入大小为 28x28，输出大小为 128
        self.hidden2 = nn.Linear(128, 64)  # 第二个隐藏层，输入大小为 128，输出大小为 64
        self.hidden3 = nn.Linear(64, 64)  # 第三个隐藏层，输入大小为 64，输出大小仍然为 64
        self.out = nn.Linear(64, 10)  # 创建输出层，输入大小为 64，输出大小为 10

    def forward(self, x):  # 定义了前向传播函数
        x = self.flatten(x)  # 将输入数据 x 展平成一维向量
        x = self.hidden1(x)  # 将展平后的数据传递给第一个隐藏层 hidden1
        x = torch.relu(x)  # 对第一个隐藏层的输出应用 ReLU（修正线性单元）激活函数。
        x = self.hidden2(x)  # 将经过 ReLU 激活的数据传递给第二个隐藏层 hidden2。
        x = torch.sigmoid(x)  # 对第二个隐藏层的输出应用 Sigmoid 激活函数
        x = self.hidden3(x)  # 将经过 Sigmoid 激活的数据传递给第三个隐藏层
        x = torch.relu(x)  # 对第三个隐藏层的输出应用 ReLU 激活函数
        x = self.out(x)  # 将经过 ReLU 激活的数据传递给输出层
        return x  # 返回模型的输出


model = NeuralNetwork().to(device)  # 创建一个神经网络模型的实例
print(model)  # 打印模型的结构
print('权重个数为：', ((28 * 28) + 1) * 128 + 129 * 256 + 257 * 10)
