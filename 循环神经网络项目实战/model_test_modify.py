# --------------------------------- 1. 导入所需的库 ---------------------------------
import pandas as pd  # 用于数据处理和分析
import numpy as np  # 用于数值计算
from sklearn.preprocessing import MinMaxScaler  # 用于数据归一化
from sklearn.metrics import mean_squared_error  # 用于计算均方误差
from math import sqrt  # 用于计算平方根
from keras.models import load_model  # 用于加载已保存的Keras模型
import matplotlib.pyplot as plt  # 用于数据可视化
import joblib  # 用于加载之前保存的scaler对象

# --------------------------------- 2. Matplotlib中文显示设置 ---------------------------------
# 设置matplotlib以正确显示中文字符和负号
plt.rcParams['font.sans-serif'] = ['PingFang HK', 'SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# --------------------------------- 3. 加载和准备测试数据 ---------------------------------
# 加载完整的黄金价格数据集
dataset = pd.read_csv('循环神经网络项目实战/LBMA-GOLD.csv', index_col='Date')

# 定义训练集长度，以确定测试集的起始位置
training_len = 1256 - 200

# 提取测试集数据（从训练集结束后的位置开始）
# 使用.iloc方法通过整数位置索引来选择数据
# 从training_len开始，到数据集末尾，[0] 表示只选择第一列数据
test_set_original = dataset.iloc[training_len:, [0]]

# --------------------------------- 4. 数据归一化 ---------------------------------
# 加载在训练阶段保存的MinMaxScaler对象
# 这是至关重要的一步，确保测试数据使用与训练数据完全相同的缩放标准
sc = joblib.load('循环神经网络项目实战/scaler.gz')

# 使用加载的scaler来转换（transform）测试数据，而不是拟合转换（fit_transform）
test_set_scaled = sc.transform(test_set_original)

# --------------------------------- 5. 创建时间序列测试集 ---------------------------------
# 初始化用于存放测试特征和标签的列表
x_test = []
y_test = [] # y_test 在此脚本中并未直接用于模型评估，但保留了其创建逻辑

# 使用与训练时相同的滑动窗口方法创建测试数据
# 用过去5天的数据作为输入特征，预测第6天的数据
for i in range(5, len(test_set_scaled)):
    x_test.append(test_set_scaled[i - 5:i, 0])
    y_test.append(test_set_scaled[i, 0])

# 将列表转换为NumPy数组
x_test, y_test = np.array(x_test), np.array(y_test)

# 重塑x_test以满足LSTM模型的输入要求 [样本数, 时间步长, 特征数]
x_test = np.reshape(x_test, (x_test.shape[0], 5, 1))

# --------------------------------- 6. 加载模型并进行预测 ---------------------------------
# 从 'model.h5' 文件加载已经训练好的LSTM模型
# 注意：load_model会加载模型的结构、权重和优化器状态
model = load_model('循环神经网络项目实战/model.h5')

# 使用加载的模型对测试数据x_test进行预测
predicted_scaled = model.predict(x_test)

# --------------------------------- 7. 反归一化处理 ---------------------------------
# 模型的预测输出是归一化后的值，需要将其转换回原始的价格尺度，以便进行比较和评估
# 使用之前创建的scaler对象(sc)的inverse_transform方法进行反归一化
prediction_original = sc.inverse_transform(predicted_scaled)

# 获取真实的黄金价格用于比较
# 注意：由于我们的滑动窗口大小为5，所以真实值应该从测试集的第5个索引开始
real_prices = test_set_original.values[5:] # 获取真实价格

# --------------------------------- 8. 评估模型性能 ---------------------------------
# 计算均方根误差 (RMSE)，这是衡量回归模型预测误差的常用指标
rmse = sqrt(mean_squared_error(prediction_original, real_prices))
# 计算平均绝对百分比误差 (MAPE)，它表示预测误差占真实值的百分比
mape = np.mean(np.abs((real_prices - prediction_original) / real_prices))

# 打印评估结果
print('均方根误差 (RMSE):', rmse)
print('平均绝对百分比误差 (MAPE):', mape)

# --------------------------------- 9. 可视化预测结果 ---------------------------------
# 绘制真实价格和模型预测价格的对比图
plt.plot(real_prices, color='red', label='真实黄金价格')
plt.plot(prediction_original, color='blue', label='预测黄金价格')
plt.title("基于LSTM神经网络的黄金价格预测")
plt.xlabel("时间（天）")
plt.ylabel("黄金价格")
plt.legend()  # 显示图例
plt.show()  # 显示图表