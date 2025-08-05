# --------------------------------- 1. 导入所需的库 ---------------------------------
import pandas as pd  # 用于数据处理和分析，提供了DataFrame等数据结构
import numpy as np  # 用于进行高效的数值计算，特别是数组操作
import matplotlib.pyplot as plt  # 用于数据可视化，绘制图表
from sklearn.preprocessing import MinMaxScaler  # 从scikit-learn中导入MinMaxScaler，用于数据归一化
from keras.layers import Dense, LSTM  # 从Keras中导入所需的神经网络层：全连接层(Dense)和长短期记忆网络层(LSTM)
import keras  # 导入Keras深度学习框架

# --------------------------------- 2. Matplotlib中文显示设置 ---------------------------------
# 为了在图表中正确显示中文字符，需要进行以下设置
# plt.rcParams是一个字典，用于配置matplotlib的参数
# 'font.sans-serif'用于设置无衬线字体的列表，matplotlib会按顺序查找可用的字体
# 'PingFang HK' (苹果系统), 'SimHei' (黑体), 'Microsoft YaHei' (微软雅黑) 是常见的中文字体
plt.rcParams['font.sans-serif'] = ['PingFang HK', 'SimHei', 'Microsoft YaHei']
# 'axes.unicode_minus'设置为False，可以解决负号'-'在某些字体下显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False

# --------------------------------- 3. 加载和准备数据 ---------------------------------
# 使用pandas的read_csv函数从CSV文件中加载黄金价格历史数据
# index_col='Date'参数指定将'Date'列作为DataFrame的索引，方便按日期进行操作
dataset = pd.read_csv('循环神经网络项目实战/LBMA-GOLD.csv', index_col='Date')

# 定义训练集和测试集的划分点
# 数据集总长度为1256，我们选择前1056条数据作为训练集，剩余的200条作为测试集
training_len = 1256 - 200

# 提取训练集数据
# 使用.iloc方法通过整数位置索引来选择数据
# [0:training_len, [0]]表示选择从第0行到training_len-1行，以及第0列（即价格数据）
training_set = dataset.iloc[0:training_len, [0]]

# 提取测试集数据
# [training_len:, [0]]表示选择从training_len行到最后一行，以及第0列
test_set = dataset.iloc[training_len:, [0]]

# --------------------------------- 4. 数据归一化 ---------------------------------
# 归一化是将数据缩放到一个特定的范围（通常是0到1），这有助于神经网络更快、更稳定地收敛
# 创建一个MinMaxScaler实例，设置缩放范围为(0, 1)
sc = MinMaxScaler(feature_range=(0, 1))
# 使用fit_transform方法对训练集进行拟合和转换
# fit会计算出缩放所需的最小值和最大值，transform则应用这个缩放
train_set_scaled = sc.fit_transform(training_set)
# 使用之前在训练集上计算出的缩放器（sc）来转换测试集
# 这样做可以保证训练集和测试集使用完全相同的缩放标准，避免数据泄露
test_set = sc.transform(test_set)

# --------------------------------- 5. 创建时间序列数据集 ---------------------------------
# LSTM网络需要特定格式的输入数据，即(样本数, 时间步长, 特征数)
# 我们需要将一维的时间序列数据转换为这种格式
# 定义用于存放处理后的训练数据的列表
x_train = []  # 存放输入特征 (过去5天的价格)
y_train = []  # 存放标签 (第6天的价格)

# 定义用于存放处理后的测试数据的列表
x_test = []
y_test = []

# 使用滑动窗口的方法创建训练集的时间序列数据
# 窗口大小为5，意味着我们用过去5天的数据来预测第6天的数据
# 循环从第5个数据点开始，直到训练集结束
for i in range(5, len(train_set_scaled)):
    # 提取i-5到i-1的数据作为输入特征
    x_train.append(train_set_scaled[i - 5:i, 0])
    # 提取第i个数据作为标签
    y_train.append(train_set_scaled[i, 0])

# 将列表格式的训练数据转换为NumPy数组，这是神经网络库的标准输入格式
x_train, y_train = np.array(x_train), np.array(y_train)

# 重塑(reshape)x_train的形状以满足LSTM的输入要求
# x_train.shape[0] 是样本数量
# 5 是时间步长 (time steps)
# 1 是每个时间步的特征数量 (因为我们只用了价格这一个特征)
x_train = np.reshape(x_train, (x_train.shape[0], 5, 1))

# 对测试集进行同样的时间序列转换
for i in range(5, len(test_set)):
    x_test.append(test_set[i - 5:i, 0])
    y_test.append(test_set[i, 0])

# 将测试数据也转换为NumPy数组并重塑形状
x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], 5, 1))

# --------------------------------- 6. 构建LSTM神经网络模型 ---------------------------------
# 使用Keras的Sequential模型，它允许我们按顺序堆叠网络层
model = keras.Sequential()

# 添加第一个LSTM层
# 80是LSTM单元（或神经元）的数量
# return_sequences=True表示该层会返回每个时间步的输出，而不仅仅是最后一个
# 这对于堆叠多个LSTM层是必需的
# activation="relu"设置激活函数为ReLU，有助于缓解梯度消失问题
model.add(LSTM(80, return_sequences=True, activation="relu"))

# 添加第二个LSTM层
# 100是LSTM单元的数量
# return_sequences=False表示该层只返回最后一个时间步的输出，因为后面是全连接层
model.add(LSTM(100, return_sequences=False, activation="relu"))

# 添加一个全连接(Dense)层
# 10是该层的神经元数量
model.add(Dense(10, activation="relu"))

# 添加输出层
# 1个神经元，因为我们是在做一个回归任务，预测一个单一的数值（价格）
model.add(Dense(1))

# --------------------------------- 7. 编译模型 ---------------------------------
# 编译模型是配置其学习过程的步骤
# loss='mse' (均方误差) 是回归问题中常用的损失函数，它衡量预测值与真实值之间的平均平方差
# optimizer=keras.optimizers.Adam(0.01) 指定使用Adam优化器，并设置其学习率为0.01
# Adam优化器是一种常用的优化算法，它结合了Adaptive Moment Estimation (Adam)，比SGD（随机梯度下降法）和Momentum优化器更适合处理高维数据
# 学习率控制了模型权重更新的幅度
model.compile(loss='mse', optimizer=keras.optimizers.Adam(0.01))

# --------------------------------- 8. 训练模型 ---------------------------------
# 使用fit方法来训练模型
# x_train, y_train 是训练数据和对应的标签
# batch_size=32 表示每次权重更新使用32个样本，有助于提高训练效率和稳定性
# epochs=100 表示对整个训练数据集进行100次完整的迭代
# validation_data=(x_test, y_test) 提供了验证集，模型会在每个epoch结束后在该数据上评估性能
# 训练过程中的指标（如loss, val_loss）会被记录在history对象中
history = model.fit(x_train, y_train, batch_size=32, epochs=100, validation_data=(x_test, y_test))

# 训练完成后，将模型保存到文件中，以便将来可以直接加载使用，无需重新训练
model.save('循环神经网络项目实战/model.h5')

# --------------------------------- 9. 可视化训练结果 ---------------------------------
# 绘制训练过程中的损失(loss)和验证损失(val_loss)曲线，以评估模型性能
# history.history是一个字典，包含了训练期间的所有指标记录
plt.plot(history.history['loss'], label='train_loss')  # 绘制训练集损失曲线
plt.plot(history.history['val_loss'], label='val_loss')  # 绘制验证集损失曲线
plt.title("LSTM神经网络loss值")  # 设置图表标题
plt.xlabel("Epoch")  # 设置x轴标签为"周期"
plt.ylabel("Loss")  # 设置y轴标签为"损失"
plt.legend()  # 显示图例，区分不同曲线
plt.show()  # 显示图表