# 导入库
import pandas as pd  # 用于数据处理和分析
import numpy as np  # 用于数值计算
import matplotlib.pyplot as plt  # 用于数据可视化
from sklearn.preprocessing import MinMaxScaler  # 用于数据归一化
from keras.layers import Dense, LSTM  # 从Keras导入所需的层：全连接层和LSTM层
import keras  # 导入Keras深度学习框架

# 解决matplotlib中文显示问题
# 提供一个字体列表，matplotlib会依次尝试，直到找到可用的字体
# 'PingFang HK' 是 macOS 上常见的字体, 'SimHei' 和 'Microsoft YaHei' 是 Windows 上常见的字体
plt.rcParams['font.sans-serif'] = ['PingFang HK', 'SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

# 加载历史数据文件
# 使用pandas的read_csv函数读取CSV文件，并将'Date'列作为索引
dataset = pd.read_csv('循环神经网络项目实战/LBMA-GOLD.csv', index_col='Date')

# 设置训练集的长度
# 总数据长度为1256，这里取前1056条作为训练数据，后200条作为测试数据
training_len = 1256 - 200

# 获取训练集数据
# 使用.iloc基于整数位置进行索引，选择所有行和第0列（价格）作为训练集
training_set = dataset.iloc[0:training_len, [0]]

# 获取测试集数据
# 选择训练集之后的数据作为测试集
test_set = dataset.iloc[training_len:, [0]]

# 将数据集进行归一化，方便神经网络的训练
# 创建一个MinMaxScaler对象，将数据缩放到0到1的范围内
sc = MinMaxScaler(feature_range=(0, 1))
# 对训练集进行拟合和转换
train_set_scaled = sc.fit_transform(training_set)
# 使用相同的缩放器转换测试集，保证训练集和测试集使用相同的缩放标准
test_set = sc.transform(test_set)

# 设置放置训练数据特征和训练数据标签的列表
x_train = []  # 存放训练数据的输入特征
y_train = []  # 存放训练数据的标签

# 设置放置测试数据特征和测试数据标签的列表
x_test = []  # 存放测试数据的输入特征
y_test = []  # 存放测试数据的标签

# 利用for循环，遍历整个训练集，创建时间序列数据
# 提取训练集中连续5个采样点的数据作为输入特征x_train，第6个采样点的数据作为标签y_train。
# 这是一个创建滑动窗口的方法，窗口大小为5。
for i in range(5, len(train_set_scaled)):
    x_train.append(train_set_scaled[i - 5:i, 0])
    y_train.append(train_set_scaled[i, 0])

# 将训练集由list格式变为array格式，方便后续处理
x_train, y_train = np.array(x_train), np.array(y_train)

# 使x_train符合LSTM输入要求：[送入样本数, 循环核时间展开步数, 每个时间步输入特征个数]。
# x_train.shape[0]是样本数
# 5 是时间步长（time steps）
# 1 是每个时间步的特征数
x_train = np.reshape(x_train, (x_train.shape[0], 5, 1))

# 同理划分测试集数据
# 对测试集也进行同样的操作，创建滑动窗口
for i in range(5, len(test_set)):
    x_test.append(test_set[i - 5:i, 0])
    y_test.append(test_set[i, 0])

# 测试集变array并reshape为符合LSTM输入要求的格式
x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], 5, 1))

# # 搭建神经网络模型
# 使用Keras的Sequential模型，这是一个线性的层次堆栈。
model = keras.Sequential()
# 添加第一个LSTM层，80个神经元，return_sequences=True表示返回所有时间步的输出，以便下一个LSTM层可以处理
model.add(LSTM(80, return_sequences=True, activation="relu"))
# 添加第二个LSTM层，100个神经元，return_sequences=False表示只返回最后一个时间步的输出
model.add(LSTM(100, return_sequences=False, activation="relu"))
# 添加一个全连接层，10个神经元
model.add(Dense(10, activation="relu"))
# 添加输出层，1个神经元，用于预测价格
model.add(Dense(1))

# 对模型进行编译
# 使用均方误差（mse）作为损失函数，适用于回归问题
# 使用Adam优化器，并设置学习率为0.01
model.compile(loss='mse', optimizer=keras.optimizers.Adam(0.01))

# 将训练集和测试集放入网络进行训练
# batch_size=32：每次更新权重时使用32个样本
# epochs=100：对整个训练数据集进行100次完整的遍历
# validation_data=(x_test, y_test)：在每个epoch结束后，在测试集上评估模型的性能
history = model.fit(x_train, y_train, batch_size=32, epochs=100, validation_data=(x_test, y_test))
# 保存训练好的模型到文件'model.h5'
model.save('循环神经网络项目实战/model.h5')

# 绘制训练集和测试集的loss值对比图
# history.history 包含了训练过程中的所有指标
plt.plot(history.history['loss'], label='train_loss')  # 训练集损失
plt.plot(history.history['val_loss'], label='val_loss')  # 验证集（测试集）损失
plt.title("LSTM神经网络loss值")  # 设置图表标题
plt.xlabel("Epoch")  # 设置x轴标签
plt.ylabel("Loss")  # 设置y轴标签
plt.legend()  # 显示图例
plt.show()  # 显示图表