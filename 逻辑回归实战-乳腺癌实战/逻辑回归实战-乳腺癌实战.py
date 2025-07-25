# 导入所需的库
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split  # 用于划分数据集
from sklearn.preprocessing import MinMaxScaler  # 用于数据归一化
from sklearn.linear_model import LogisticRegression  # 逻辑回归模型
from sklearn.metrics import classification_report  # 用于评估模型性能

# --- 数据加载与预处理 ---

# 定义Excel文件路径
file_path = '逻辑回归实战-乳腺癌实战/breast_cancer_data.xlsx'

# 使用 try-except 块来处理可能的文件未找到错误
try:
    # 读取Excel文件到pandas DataFrame
    dataset = pd.read_excel(file_path)
    print(f"成功读取文件: {file_path}")
except FileNotFoundError:
    # 如果文件未找到，则打印错误消息并退出程序
    print(f"文件未找到: {file_path}")
    exit()

# --- 特征和标签提取 ---

# 提取特征 (X)，即除了最后一列 'target' 之外的所有列
X = dataset.iloc[:, : -1]
# 打印特征数据的前几行，以作检查
# print("特征 (X):")
# print(X.head())

# 提取标签 (Y)，即 'target' 列
Y = dataset['target']
# 打印标签数据的前几行
# print("\n标签 (Y):")
# print(Y.head())

# --- 数据集划分 ---

# 将数据集划分为训练集和测试集
# test_size=0.2 表示将20%的数据作为测试集，其余80%作为训练集
# random_state=42 确保每次划分结果都相同，便于复现
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# --- 数据归一化 ---

# 创建一个MinMaxScaler对象，将特征值缩放到 [0, 1] 区间
sc = MinMaxScaler(feature_range=(0, 1))
# 对训练集的特征进行拟合和转换
x_train = sc.fit_transform(x_train)
# 对测试集的特征进行转换 (使用训练集拟合的scaler)
x_test = sc.transform(x_test) # 注意这里是 transform 而不是 fit_transform

# --- 模型训练 ---

# 创建逻辑回归模型实例
lr = LogisticRegression()
# 使用训练数据来拟合（训练）模型
lr.fit(x_train, y_train)

# --- 模型评估与预测 ---

# 打印模型的学习到的权重（w）和偏置（b）
# print('权重 (w):', lr.coef_)
# print('偏置 (b):', lr.intercept_)

# 使用训练好的模型对测试集进行预测
pre_result = lr.predict(x_test)

# 打印每个样本属于各个类别的概率
pre_result_proba = lr.predict_proba(x_test)

# 提取样本为“恶性”肿瘤（类别1）的概率
pre_list = pre_result_proba[:, 1]

# --- 阈值调整与结果生成 ---

# 设置一个自定义的分类阈值
# 如果一个样本被预测为恶性的概率大于这个阈值，我们就将其分类为恶性
thresholds = 0.3

# 初始化用于存储最终分类结果的列表
result = []
result_name = []

# 遍历每个测试样本的预测概率
for i in range(len(pre_list)):
    if pre_list[i] > thresholds:
        # 如果概率大于阈值，分类为1（恶性）
        result.append(1)
        result_name.append('恶性')
    else:
        # 否则，分类为0（良性）
        result.append(0)
        result_name.append('良性')

# --- 性能报告 ---

# 生成并打印分类报告，其中包含精确率、召回率、F1分数等指标
# labels=[0, 1] 指定了类别的顺序
# target_names 指定了类别对应的名称
report = classification_report(y_test, result, labels=[0, 1], target_names=['良性肿瘤', '恶性肿瘤'])
print("\n分类报告:")
print(report)
