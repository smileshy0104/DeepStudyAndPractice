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
import joblib  # 用于加载保存的scaler对象

# 解决matplotlib中文显示问题
# 提供一个字体列表，matplotlib会依次尝试，直到找到可用的字体
# 'PingFang HK' 是 macOS 上常见的字体, 'SimHei' 和 'Microsoft YaHei' 是 Windows 上常见的字体
plt.rcParams['font.sans-serif'] = ['PingFang HK', 'SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

# 1. 加载数据和预训练对象
# 从CSV文件中读取数据
dataset = pd.read_csv("全连接神经网络空气质量回归预测/data.csv")
# 加载在训练阶段保存的scaler对象，用于数据反归一化
sc = joblib.load('全连接神经网络空气质量回归预测/scaler.gz')
# 加载已经训练并保存好的模型
model = load_model("全连接神经网络空气质量回归预测/model.h5")

# 2. 准备测试数据
# 为了演示，我们从完整数据集中随机抽取一小部分作为“新的”未知数据
# 在实际应用中，这里会是全新的、模型从未见过的数据
# test_df = dataset.sample(n=100, random_state=43) # 随机抽取100条
_, test_df = train_test_split(dataset, test_size=0.2, random_state=42) # 或者使用和训练时相同的划分

# 3. 归一化测试数据
# 使用加载的scaler来转换测试数据
test_scaled = sc.transform(test_df)
test_scaled_df = pd.DataFrame(test_scaled, columns=dataset.columns, index=test_df.index)

# 4. 提取特征和真实标签
x_test = test_scaled_df.iloc[:, :-1]
y_test_scaled = test_scaled_df.iloc[:, -1] # 这是归一化后的真实标签

# 5. 进行预测
# 利用加载的模型对测试集特征进行预测
yhat_scaled = model.predict(x_test) # 预测出的值是归一化后的

# 6. 反归一化预测值和真实值
# 为了将预测值与真实值在原始尺度上进行比较，需要进行反归一化

# 反归一化预测值
# 创建一个与原始数据结构相同的临时DataFrame，用于反归一化
temp_df_pred = pd.DataFrame(x_test.copy())
temp_df_pred.columns = dataset.columns[:-1]
temp_df_pred[dataset.columns[-1]] = yhat_scaled # 将预测的标签加入
# 进行反归一化
inv_yhat = sc.inverse_transform(temp_df_pred)
# 提取反归一化后的预测值（最后一列）
prediction = inv_yhat[:, -1]

# 反归一化真实值
# 同样地，反归一化测试集的真实标签
temp_df_real = pd.DataFrame(x_test.copy())
temp_df_real.columns = dataset.columns[:-1]
temp_df_real[dataset.columns[-1]] = y_test_scaled # 将真实的标签加入
# 进行反归一化
inv_y = sc.inverse_transform(temp_df_real)
# 提取反归一化后的真实值（最后一列）
real = inv_y[:, -1]

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