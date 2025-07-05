from torch.utils.data import DataLoader  # 数据包管理工具
from torchvision import datasets  # 数据处理工具，专门用于图像处理的包
from torchvision.transforms import ToTensor  # 数据转换
from matplotlib import pyplot as plt

# datasets.MNIST来加载MNIST数据集作为训练数据集。
# #root='data'：指定数据集存储的根目录，可以根据需要进行更改。
# #train=True：表示加载训练数据集#download=True：如果数据集在指定路径中不存在，将自动从官方源下载并保存。
# #transform=ToTensor()：指定数据转换操作，将图像数据转换为PyTorch中的Tensor张量格式。

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

# 绘制图像
figure = plt.figure()
for i in range(9):  # 从训练数据集（training_data）中获取样本的图像（img）和标签（lable）
    img, lable = training_data[i+100]
    figure.add_subplot(3, 3, i + 1)  # 绘制子画布
    plt.title(f'lable={lable}')
    plt.axis('off')
    plt.imshow(img.squeeze(), cmap='gray')  # img.squeeze  将维度进行压缩
plt.show()

# 创建数据DataLoader（数据加载器）
train_dataloader = DataLoader(training_data, batch_size=64)  # 64张图片为一个包
test_dataloader = DataLoader(test_data, batch_size=64)
for X, Y in train_dataloader:  # X表示打包好的每一个数据包
    print(f'Shape of X[N,C,H,W]:{X.shape}')
    print(f'Shape of Y:{Y.shape}{Y.dtype}')
    break
