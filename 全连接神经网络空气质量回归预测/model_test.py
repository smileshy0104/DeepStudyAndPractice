# 导入所需的库
import numpy as np  # 用于数值计算
import pandas as pd  # 用于数据处理和CSV文件I/O
import matplotlib.pyplot as plt  # 用于数据可视化
from sklearn.preprocessing import MinMaxScaler  # 用于数据归一化
from keras.layers import Dense  # Keras中的全连接层
from sklearn.model_selection import train_test_split  # 用于划分数据集
import keras  # Keras深度学习框架
from keras.models import load_model  # 用于加载已保存的Keras模型
from math import sqrt  # 用于计算平方根（RMSE）
from numpy import concatenate  # 用于数组拼接
from sklearn.metrics import mean_squared_error  # 用于计算均方误差

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
# 注意：这里的归一化应该使用与训练时相同的scaler对象。
# 为了独立测试，这里重新创建并拟合了scaler，但在实际应用中，
# 应该保存训练时的scaler并在测试时加载使用，以确保数据转换的一致性。
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
# 将数据集划分为训练集和测试集，这里只取5%作为测试集来演示模型预测
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.05, random_state=42)

# 5. 加载模型
# 加载已经训练并保存好的模型
model = load_model("全连接神经网络空气质量回归预测/model.h5")

# 6. 进行预测
# 利用加载的模型对测试集特征进行预测
yhat = model.predict(x_test)
# print("归一化之前的值", yhat) # 预测出的值是归一化后的

# 7. 反归一化预测值
# 为了将预测值与真实值进行比较，需要将其反归一化到原始尺度
# 首先，将测试集的特征(x_test)和预测的标签(yhat)拼接起来
inv_yhat = concatenate((x_test, yhat), axis=1)
# 使用scaler的inverse_transform方法进行反归一化
inv_yhat = sc.inverse_transform(inv_yhat)
# 提取反归一化后的预测值（最后一列）
prediction = inv_yhat[:, -1]
# print("归一化之后的值", prediction)

# 8. 反归一化真实值
# 同样地，需要将测试集的真实标签(y_test)也反归一化
# 转换y_test的维度以匹配拼接要求
y_test = np.array(y_test)
y_test = np.reshape(y_test, (y_test.shape[0], 1))
# 将测试集的特征(x_test)和真实的标签(y_test)拼接起来
inv_y = concatenate((x_test, y_test), axis=1)
# 使用scaler的inverse_transform方法进行反归一化
inv_y = sc.inverse_transform(inv_y)
# 提取反归一化后的真实值（最后一列）
real = inv_y[:, -1]
# print(real)

# 9. 评估模型性能
# 计算均方根误差 (RMSE)
rmse = sqrt(mean_squared_error(real, prediction))
# 计算平均绝对百分比误差 (MAPE)
mape = np.mean(np.abs((real - prediction) / real)) # 注意分母是real

# 打印评估指标
print('Test RMSE: %.3f' % rmse)
print('Test MAPE: %.3f' % mape)

# 10. 可视化结果
# 画出真实值和预测值的对比图
plt.plot(prediction, label='预测值')
plt.plot(real, label="真实值")
plt.title("全连接神经网络空气质量预测对比图")
plt.legend()
plt.show()