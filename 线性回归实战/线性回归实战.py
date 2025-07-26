# --- 1. 数据准备 ---
# 定义一个简单的数据集，用于演示线性回归。
# 我们的目标是找到一个函数 y = w * x，使得它能最好地拟合这些数据。

# 定义输入特征 (x)。可以看作是学习的小时数。
x_data = [1.0, 2.0, 3.0]
# 定义对应的目标标签 (y)。可以看作是考试的分数。
y_data = [2.0, 4.0, 6.0]

# --- 2. 模型和参数初始化 ---

# 初始化模型的权重参数 w。
# 我们随机给 w 一个初始值，训练的目标就是找到最优的 w。
# 这里的真实 w 应该是 2，我们故意设置一个不准确的初始值。
w = 4.0

# 定义线性回归模型的前向传播。
# 这是最简单的线性模型 y_pred = w * x。
def forward(x):
    """
    根据输入的 x 和当前的权重 w，计算预测值 y_pred。
    """
    return x * w

# --- 3. 损失函数和梯度计算 ---

# 定义损失函数（Loss Function）。
# 损失函数用来衡量模型的预测值与真实值之间的差距。
# 这里我们使用均方误差（Mean Squared Error, MSE）。
def loss(xs, ys):
    """
    计算在一批数据 (xs, ys) 上的平均损失。
    """
    loss_value = 0
    # 遍历数据集中的每一个样本
    for x, y in zip(xs, ys):
        y_pred = forward(x)  # 得到预测值
        # 计算单个样本的平方误差，并累加到总损失中
        loss_value += (y_pred - y) ** 2
    # 返回所有样本的平均损失
    return loss_value / len(xs)

# 定义计算梯度的函数。
# 梯度表示损失函数在当前 w 值下变化最快的方向。
# 我们需要沿着梯度的反方向更新 w，以减小损失。
# 对于损失函数 L = (w*x - y)^2，它对 w 的偏导数（梯度）是 2*x*(w*x - y)。
def gradient(xs, ys):
    """
    计算损失函数关于权重 w 的平均梯度。
    """
    grad_value = 0
    # 遍历数据集中的每一个样本
    for x, y in zip(xs, ys):
        # 计算单个样本的梯度，并累加到总梯度中
        grad_value += 2 * x * (forward(x) - y)
    # 返回所有样本的平均梯度
    return grad_value / len(xs)

# --- 4. 模型训练 ---

print("开始训练...")
# 设置学习率 (learning rate)，它控制每次更新 w 的步长。
learning_rate = 0.01
# 进行 100 次迭代训练 (epoch)。
for epoch in range(100):
    # 1. 计算当前权重下的损失
    loss_val = loss(x_data, y_data)
    # 2. 计算当前权重下的梯度
    grad_val = gradient(x_data, y_data)
    # 3. 更新权重 w。这是梯度下降的核心步骤。
    #    我们让 w 朝着梯度的反方向移动一小步。
    w = w - learning_rate * grad_val

    # 打印每一轮训练的信息，观察 w 和 loss 的变化
    print(f"轮次: {epoch}, 权重 w = {w:.4f}, 损失 loss = {loss_val:.4f}")

# --- 5. 模型推理（预测） ---

print("\n训练完成!")
# 使用训练好的最终权重 w 来进行预测。
# 假设我们想预测学习 4 个小时的分数。
test_x = 4.0
predicted_y = forward(test_x)
print(f"使用训练好的模型进行预测：如果学习 {test_x} 小时，预测分数为: {predicted_y:.4f}")
