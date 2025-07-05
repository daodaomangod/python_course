import torch
import torch.nn as nn
from torch.utils.data import DataLoader  # 数据包管理工具
from torchvision import datasets  # 数据处理工具，专门用于图像处理的包
from torchvision.transforms import ToTensor  # 数据转换

# 1. 加载数据集
training_data = datasets.MNIST(
    root='data',
    train=True,
    download=True,
    transform=ToTensor(),  # tensor张量
)

test_data = datasets.MNIST(
    root='data',
    train=False,
    download=True,
    transform=ToTensor()
)

# 2. 创建数据DataLoader（数据加载器）
train_dataloader = DataLoader(training_data, batch_size=64)  # 64张图片为一个包
test_dataloader = DataLoader(test_data, batch_size=64)

# 3. 判断设备是否支持GPU
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


# 4. 创建神经网络模型
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


# 5. 训练神经网络
def train(dataloader, model, loss_fn, optimizer):
    # 定义训练过程
    batch_size_num = 1     # 跟踪处理的批次数量
    for x, y in dataloader:  # 遍历数据加载器中的每个训练批次
        x, y = x.to(device), y.to(device)
        pred = model.forward(x)  # 前向传播
        loss = loss_fn(pred, y)   # 损失函数
        optimizer.zero_grad()    #清零优化器 optimizer 中存储的之前的梯度信息。
        loss.backward()          # 执行反向传播
        optimizer.step()         # 使用优化器来更新模型的参数，以减小损失
        loss_value = loss.item()
        print(f'loss:{loss_value:>7f}[num:{batch_size_num}]')
        batch_size_num += 1


loss_fn = nn.CrossEntropyLoss()    # 定义交叉熵损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.0015)  # 定义 Adam 优化器
train(train_dataloader, model, loss_fn, optimizer)   # 执行训练




