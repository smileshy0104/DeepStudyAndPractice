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
print('预测结果:', pre_result)

# 打印每个样本属于各个类别的概率
pre_result_proba = lr.predict_proba(x_test)
print('预测结果概率:', pre_result_proba)

# 提取样本为“恶性”肿瘤（类别1）的概率
pre_list = pre_result_proba[:, 1]
print('样本为“恶性”肿瘤的概率:', pre_list)

# --- 阈值调整与结果生成 ---

# 设置一个自定义的分类阈值
# 如果一个样本被预测为恶性的概率大于这个阈值，我们就将其分类为恶性
thresholds = 0.5

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
# 输出结果
report = classification_report(y_test, result, labels=[0, 1], target_names=['良性肿瘤', '恶性肿瘤'])
print("\n分类报告:")
print(report)



# 分类报告是评估分类模型性能的重要工具，它能显示每个类别的精确度、召回率、F1分数以及支持数。

# 为了更好地理解，我们先来看一个典型的分类报告示例，然后逐一解释其中的指标：

#               precision    recall  f1-score   support

#      Class 0       0.95      0.93      0.94       100
#      Class 1       0.94      0.96      0.95       120

#     accuracy                           0.95       220
#    macro avg       0.95      0.94      0.94       220
# weighted avg       0.95      0.95      0.95       220

# txt


# 核心指标解释
# 假设我们正在做一个垃圾邮件分类的任务，模型需要判断一封邮件是“垃圾邮件”（正类/Positive）还是“正常邮件”（负类/Negative）。

# Precision (精确率)

# 定义: 精确率 = (正确预测为正类的数量) / (所有被预测为正类的数量)
# 通俗解释: 在所有被模型预测为“垃圾邮件”的邮件中，有多少是真的垃圾邮件。
# 意义: 精确率高表示模型预测为正类的结果很准，“宁缺毋滥”，尽量不把正常邮件误判为垃圾邮件。
# Recall (召回率)

# 定义: 召回率 = (正确预测为正类的数量) / (所有真实为正类的数量)
# 通俗解释: 在所有真正的“垃圾邮件”中，有多少被模型成功地找了出来。
# 意义: 召回率高表示模型能把绝大部分的正类都找出来，“宁可错杀，不可放过”，尽量不漏掉任何一封垃圾邮件。
# F1-Score (F1分数)

# 定义: F1 = 2 * (精确率 * 召回率) / (精确率 + 召回率)
# 通俗解释: 这是精确率和召回率的调和平均值，是两者的综合考量。
# 意义: F1分数高表示模型的精确率和召回率都比较高，性能比较均衡。当精确率和召回率之间存在矛盾时（一个高另一个就低），F1分数是一个很好的综合评价指标。
# Support (支持数)

# 定义: 每个类别在真实数据中的样本数量。
# 通俗解释: 在我们的例子中，就是“Class 0”（比如正常邮件）和“Class 1”（比如垃圾邮件）各自有多少封。
# 意义: 它可以帮助我们判断样本分布是否均衡。如果某个类别的支持数非常少，那么该类别的指标可能不太可靠。
# 报告中的平均值
# Accuracy (准确率)

# 定义: (所有预测正确的样本数) / (总样本数)
# 通俗解释: 整个模型预测正确的比例。
# 注意: 在样本类别分布不均衡的情况下（例如，99%是正常邮件，1%是垃圾邮件），准确率这个指标可能会有误导性。一个把所有邮件都预测为“正常”的模型，准确率也能达到99%，但它毫无用处。
# Macro Avg (宏平均)

# 定义: 对每个类别的指标（如precision）直接求算术平均值。
# 通俗解释: 它平等地看待每一个类别，不管这个类别的样本多还是少。
# Weighted Avg (加权平均)

# 定义: 按照每个类别的支持数（support）作为权重，对每个类别的指标进行加权平均。
# 通俗解释: 样本多的类别在计算平均值时占的比重更大。在样本不均衡的情况下，加权平均更能反映模型在整体上的表现。
# 总结
# 当您关心**“预测的结果有多准”**时，关注 Precision。例如，在股票预测中，预测为“上涨”的股票，我们希望它真的上涨，这时精确率很重要。
# 当您关心**“是否把所有目标都找到了”**时，关注 Recall。例如，在癌症诊断中，我们希望把所有真正的病人都找出来，不能漏诊，这时召回率很重要。
# 当您需要一个均衡的评价时，或者当样本不均衡时，F1-Score 和 Weighted Avg 是非常重要的参考指标。