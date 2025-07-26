# 导入所需的库
import numpy as np  # 用于数值计算
import pandas as pd  # 用于数据处理和CSV文件I/O
import matplotlib.pyplot as plt  # 用于数据可视化
from sklearn.preprocessing import MinMaxScaler  # 用于数据归一化
from keras.layers import Dense  # Keras中的全连接层
from sklearn.model_selection import train_test_split  # 用于划分数据集
import keras  # Keras深度学习框架

# 解决matplotlib中文显示问题
# 提供一个字体列表，matplotlib会依次尝试，直到找到可用的字体
# 'PingFang HK' 是 macOS 上常见的字体, 'SimHei' 和 'Microsoft YaHei' 是 Windows 上常见的字体
plt.rcParams['font.sans-serif'] = ['PingFang HK', 'SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

# 1. 导入数据集
# 从CSV文件中读取数据
dataset = pd.read_csv("全连接神经网络空气质量回归预测/data.csv")
# print(dataset) # 可以取消注释以查看原始数据集

# 2. 数据预处理
# 将数据进行归一化处理，将所有特征值缩放到[0, 1]区间
sc = MinMaxScaler(feature_range=(0, 1))
scaled = sc.fit_transform(dataset)
# print(scaled) # 可以取消注释以查看归一化后的数据

# 将归一化好的数据转化为DataFrame格式，方便后续处理
dataset_sc = pd.DataFrame(scaled)
# print(dataset_sc) # 可以取消注释以查看转换后的DataFrame

# 3. 划分特征和标签
# 将数据集中的特征（所有行，除了最后一列）和标签（所有行，最后一列）分离开
X = dataset_sc.iloc[:, :-1]  # 特征
Y = dataset_sc.iloc[:, -1]   # 标签

# 4. 划分训练集和测试集
# 将数据集划分为训练集和测试集，测试集占20%，并设置随机种子以保证结果可复现
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 5. 构建神经网络模型
# 利用Keras的Sequential API搭建一个简单的全连接神经网络模型
model = keras.Sequential()
# 添加第一个隐藏层，包含10个神经元，激活函数为ReLU
model.add(Dense(10, activation='relu'))
# 添加第二个隐藏层，包含10个神经元，激活函数为ReLU
model.add(Dense(10, activation='relu'))
# 添加输出层，包含1个神经元（用于回归预测）
model.add(Dense(1))

# 6. 编译模型
# 对神经网络模型进行编译，配置学习过程
# loss='mse': 使用均方误差作为损失函数，适用于回归问题
# optimizer='SGD': 使用随机梯度下降作为优化器
model.compile(loss='mse', optimizer='SGD')

# 7. 训练模型
# 使用训练数据对模型进行训练
# epochs=100: 训练100个周期
# batch_size=24: 每个批次的大小为24
# verbose=2: 每个epoch输出一行记录
# validation_data: 在每个epoch结束时评估模型在测试集上的性能
history = model.fit(x_train, y_train, epochs=100, batch_size=24, verbose=2, validation_data=(x_test, y_test))
# 保存训练好的模型到文件 "model.h5"
model.save("全连接神经网络空气质量回归预测/model.h5")

# 8. 可视化训练过程
# 绘制模型的训练集和验证集的loss值对比图
plt.plot(history.history['loss'], label='train_loss')  # 训练集损失
plt.plot(history.history['val_loss'], label='val_loss')    # 验证集损失
plt.title("全连接神经网络loss值图")  # 图表标题
plt.xlabel("Epoch")  # x轴标签
plt.ylabel("Loss")   # y轴标签
plt.legend()  # 显示图例
plt.show()  # 显示图表


