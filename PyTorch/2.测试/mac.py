import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. 检查并设置设备 (Mac's MPS or CPU)
# 这是针对 Mac 优化的关键部分
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("检测到 MPS 设备，将使用 MPS 进行计算。")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("检测到 CUDA 设备，将使用 CUDA 进行计算。")
else:
    device = torch.device("cpu")
    print("未检测到 MPS 或 CUDA，将使用 CPU 进行计算。")

# 2. 定义超参数
n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
log_interval = 10

# 3. 加载 MNIST 数据集
# 定义数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 下载并加载训练数据
# PyTorch 会自动将数据下载到指定的 root 目录中, 这里我们设置为当前目录
train_dataset = datasets.MNIST(root='.', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)

# 下载并加载测试数据
test_dataset = datasets.MNIST(root='.', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)

# 4. 定义神经网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 卷积层 (1个输入通道, 10个输出通道, 5x5 卷积核)
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # 卷积层 (10个输入通道, 20个输出通道, 5x5 卷积核)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # Dropout层
        self.conv2_drop = nn.Dropout2d()
        # 全连接层
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # 卷积 -> 激活 -> 池化
        x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
        # 卷积 -> Dropout -> 激活 -> 池化
        x = torch.relu(torch.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # 展平数据
        x = x.view(-1, 320)
        # 全连接层 -> 激活
        x = torch.relu(self.fc1(x))
        # Dropout
        x = torch.dropout(x, training=self.training)
        # 输出层
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)

# 5. 初始化模型、优化器和损失函数
model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# 6. 训练函数
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # 将数据和标签移动到指定的设备
        data, target = data.to(device), target.to(device)
        # 梯度清零
        optimizer.zero_grad()
        # 前向传播
        output = model(data)
        # 计算损失
        loss = criterion(output, target)
        # 反向传播
        loss.backward()
        # 更新权重
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(f'训练周期: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                  f' ({100. * batch_idx / len(train_loader):.0f}%)]\t损失: {loss.item():.6f}')

# 7. 测试函数
def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            # 将数据和标签移动到指定的设备
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            # 获取预测结果
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    print(f'\n测试集: 平均损失: {test_loss:.4f}, 准确率: {correct}/{len(test_loader.dataset)}'
          f' ({100. * correct / len(test_loader.dataset):.2f}%)\n')

# 8. 执行训练和测试
if __name__ == '__main__':
    # 首次运行时会自动下载数据集，请耐心等待
    test() # 先看未经训练的模型的表现
    for epoch in range(1, n_epochs + 1):
        train(epoch)
        test()

    print("训练完成!")