# 定义数据集

# 定义数据特征
x_data = [1.0, 2.0, 3.0]  # 输入特征数据，表示自变量 x 的值
# 定义数据标签
y_data = [2.0, 4.0, 6.0]  # 目标标签数据，表示因变量 y 的值

# 初始化参数 W
w = 4  # 线性回归模型的初始权重参数

# 定义线性回归模型
def forward(x):
    """线性回归模型的前向传播函数"""
    return x * w  # 根据当前权重 w 计算预测值

# 定义损失函数
def loss(xs, ys):
    """计算预测值与实际值之间的均方误差损失"""
    loss_value = 0  # 初始化损失值
    for x, y in zip(xs, ys):  # 遍历特征和标签
        y_pred = forward(x)  # 计算预测值
        loss_value += (y_pred - y) ** 2  # 累加每个样本的损失
    return loss_value / len(xs)  # 返回平均损失

# 定义计算梯度的函数
def gradient(xs, ys):
    """计算损失函数对权重 w 的梯度"""
    gradient_value = 0  # 初始化梯度值
    for x, y in zip(xs, ys):  # 遍历特征和标签
        gradient_value += 2 * x * (x * w - y)  # 计算梯度
    return gradient_value / len(xs)  # 返回平均梯度

# 进行训练
for epoch in range(100):  # 进行 100 轮训练
    loss_val = loss(x_data, y_data)  # 计算当前损失值
    grad_val = gradient(x_data, y_data)  # 计算当前梯度值
    w = w - 0.01 * grad_val  # 更新权重 w，学习率为 0.01

    # 打印当前训练轮次、权重和损失值
    print("训练轮次:", epoch, "w=", w, "loss=", loss_val)

# 使用训练好的权重进行推理
print("100轮后w已经训练好了，此时我们用训练好的w进行推理，学习时间为4个小时的时候最终的得分为:", forward(4))
