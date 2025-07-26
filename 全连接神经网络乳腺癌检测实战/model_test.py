# 1. 导入所需的库
import numpy as np  # 用于数值计算
import pandas as pd  # 用于数据处理和CSV文件I/O
import matplotlib.pyplot as plt  # 用于数据可视化
from sklearn.preprocessing import MinMaxScaler  # 用于数据归一化
from sklearn.model_selection import train_test_split  # 用于划分数据集
import keras  # 深度学习框架
from keras.layers import Dense  # 全连接层
from keras.utils.np_utils import to_categorical  # 用于将标签转换为one-hot编码
from sklearn.metrics import classification_report  # 用于生成分类报告
from keras.models import load_model  # 用于加载已保存的Keras模型

# 2. Matplotlib全局设置，解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

# 3. 数据加载与预处理
# 加载乳腺癌数据集
# dataset = pd.read_csv("breast_cancer_data.csv")

# 定义Excel文件路径
file_path = '全连接神经网络乳腺癌检测实战/breast_cancer_data.xlsx'

# 使用 try-except 块来处理可能的文件未找到错误
try:
    # 读取Excel文件到pandas DataFrame
    dataset = pd.read_excel(file_path)
    print(f"成功读取文件: {file_path}")
except FileNotFoundError:
    # 如果文件未找到，则打印错误消息并退出程序
    print(f"文件未找到: {file_path}")
    exit()

# 提取特征数据 (所有行，除了最后一列的所有列)
X = dataset.iloc[:, :-1]

# 提取标签数据 (所有行的 'target' 列)
Y = dataset['target']

# 将数据集划分为训练集和测试集
# 这里虽然没有使用训练集，但为了与训练过程保持一致的测试集，进行了相同的划分
# test_size=0.2 表示测试集占20%
# random_state=42 保证每次划分结果都一样，便于复现
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 对测试集的特征数据进行归一化处理
# 注意：在测试时，应该使用在训练集上学习到的归一化规则 (scaler)
# 但此处为了脚本独立运行，重新创建并拟合了测试数据。
# 更严谨的做法是保存训练时使用的scaler，在此处加载并使用。
sc = MinMaxScaler(feature_range=(0, 1))
x_test = sc.fit_transform(x_test)

# 4. 加载并使用模型进行预测
# 从 'model.h5' 文件加载已经训练好的神经网络模型
model = load_model("全连接神经网络乳腺癌检测实战/model.h5")

# 使用加载的模型对测试数据进行预测
# predict返回的是每个类别（良性/恶性）的概率
predict_probabilities = model.predict(x_test)

# 从预测的概率中找出最大概率对应的索引，作为最终的预测类别
# axis=1 表示沿着行的方向操作
# 对应的阈值为0.5
y_pred = np.argmax(predict_probabilities, axis=1) # y_pred 为预测标签

# 5. 结果处理与展示
# 将数字化的预测结果 (0 或 1) 转换成人类可读的标签 ("良性" 或 "恶性")
result = []
for i in range(len(y_pred)):
    if y_pred[i] == 0:
        result.append("良性")
    else:
        result.append("恶性")
# print(result) # 可以取消注释来查看具体的预测结果列表

# 6. 生成并打印分类报告
# classification_report 是一个强大的工具，可以清晰地展示模型在每个类别上的性能
# y_test: 真实的标签
# y_pred: 模型预测的标签
# labels=[0, 1]: 指定报告中要包含的类别
# target_names=["良性", '恶性']: 为每个类别指定名称
report = classification_report(y_test, y_pred, labels=[0, 1], target_names=["良性", '恶性'])
print("模型在测试集上的性能评估报告：")
print(report)