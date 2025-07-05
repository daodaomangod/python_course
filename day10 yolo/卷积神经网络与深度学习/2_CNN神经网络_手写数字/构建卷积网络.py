import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets

# 1. 设置网络参数
batch_size = 64
learning_rate = 0.01
momentum = 0.5
EPOCH = 10

# 2.数据归一化处理
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
# softmax归一化指数函数,其中0.1307是mean均值和0.3081是std标准差

# 数据加载
train_dataset = datasets.MNIST(root='./data/mnist', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data/mnist', train=False, transform=transform, download=True)
# 实例化dataset
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# # 绘制图像
# fig = plt.figure()
# for i in range(12):
#     plt.subplot(3, 4, i + 1)
#     plt.tight_layout()
#     plt.imshow(train_dataset.train_data[i], cmap='gray', interpolation='none')
#     plt.title("Labels: {}".format(train_dataset.train_labels[i]))
#     plt.xticks([])
#     plt.yticks([])
# plt.show()


# 3. 构建模型(CNN)
class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 10, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(10, 20, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(320, 50),
            torch.nn.Linear(50, 10),
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)  # 一层卷积层,一层池化层,一层激活层
        x = self.conv2(x)  # 一层卷积层,一层池化层,一层激活层
        x = x.view(batch_size, -1)  # flatten变成全连接网络需要的输入 (batch, 20,4,4) ==> (batch,320), -1 自动算出的是320
        x = self.fc(x)
        return x  # 输出结果


model = NeuralNetwork()

# 4.损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)  # lr学习率，momentum冲量


# 4. 定义模型训练
def train(epoch):
    running_loss = 0.0  # 这整个epoch的loss清零
    running_total = 0
    running_correct = 0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        # 把运行中的loss累加起来
        running_loss += loss.item()
        # 把运行中的准确率acc算出来
        _, predicted = torch.max(outputs.data, dim=1)
        running_total += inputs.shape[0]
        running_correct += (predicted == target).sum().item()

        if batch_idx % 300 == 299:  # 选择每300次出一个平均损失,和准确率
            print(
                f'[{epoch + 1}, {batch_idx + 1:5d}]: loss: {running_loss / 300:.3f} , acc: {100 * running_correct / running_total:.2f} %')
            running_loss = 0.0  # 这小批300的loss清零
            running_total = 0
            running_correct = 0  # 这小批300的acc清零

        # torch.save(model.state_dict(), './model_Mnist.pth')
        # torch.save(optimizer.state_dict(), './optimizer_Mnist.pth')


# 5. 定义模型测试
def test():
    correct = 0
    total = 0
    with torch.no_grad():  # 测试集不用算梯度
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)  # dim = 1 列是第0个维度，行是第1个维度，沿着行(第1个维度)去找1.最大值和2.最大值的下标
            total += labels.size(0)  # 张量之间的比较运算
            correct += (predicted == labels).sum().item()
    acc = correct / total
    print(f'[{epoch + 1} / {EPOCH}]: Accuracy on test set: {100 * acc:.1f} %')  # 求测试的准确率，正确数/总数
    return acc


# 6. 执行模型的训练和测试
if __name__ == '__main__':
    acc_list_test = []
    for epoch in range(EPOCH):
        train(epoch)
        # if epoch % 10 == 9:  #每训练10轮 测试1次
        acc_test = test()
        acc_list_test.append(acc_test)

    plt.plot(acc_list_test)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy On TestSet')
    plt.show()
