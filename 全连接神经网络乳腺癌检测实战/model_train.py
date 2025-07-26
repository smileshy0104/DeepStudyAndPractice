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
# test_size=0.2 表示测试集占20%
# random_state=42 保证每次划分结果都一样，便于复现
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 将标签数据转换为one-hot编码格式
# 例如，标签[0, 1, 0] 会被转换为 [[1, 0], [0, 1], [1, 0]]
# 告诉其，我们有两个类别，将其转换成one-hot编码（向量形式），这样模型才能处理
# 2 表示类别数量
y_train_one = to_categorical(y_train, 2) # y_train_one
y_test_one = to_categorical(y_test, 2) # y_test_one

# 对特征数据进行归一化处理
# 将特征值缩放到 (0, 1) 区间，有助于加速模型收敛和提升性能
sc = MinMaxScaler(feature_range=(0, 1))
x_train = sc.fit_transform(x_train) # 归一化训练集
x_test = sc.transform(x_test)  # 注意：测试集使用训练集学习到的规则进行转换

# 4. 构建神经网络模型
# 使用Keras的Sequential模型，这是一个线性的层次堆栈
model = keras.Sequential()
# 添加第一个隐藏层，包含10个神经元，使用ReLU激活函数
model.add(Dense(10, activation='relu'))
# 添加第二个隐藏层，同样是10个神经元和ReLU激活函数
model.add(Dense(10, activation='relu'))
# 添加输出层，包含2个神经元（对应2个类别），使用softmax激活函数
# softmax将输出转换为概率分布，适用于多分类问题
model.add(Dense(2, activation='softmax'))

# 5. 编译模型
# 配置模型的学习过程
# loss='categorical_crossentropy': 使用分类交叉熵作为--损失函数，适用于one-hot编码的标签
# optimizer='SGD': 使用随机梯度下降作为--优化器
# metrics=['accuracy']: 在训练和测试期间评估的指标为准确率
model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])

# 6. 训练模型
# history对象记录了训练过程中的所有信息
# epochs=110: 训练将进行110个周期
# batch_size=64: 每个批次包含64个样本
# verbose=2: 每个epoch输出一行日志
# validation_data: 用于在每个epoch结束时评估模型性能的验证数据
history = model.fit(x_train, y_train_one, epochs=150, batch_size=64, verbose=2, validation_data=(x_test, y_test_one))

# 将训练好的模型保存到文件 'model.h5'
model.save('全连接神经网络乳腺癌检测实战/model.h5')

# 7. 结果可视化
# 绘制训练过程中的损失值（loss）变化曲线
plt.plot(history.history['loss'], label='train_loss') # 训练集损失值
plt.plot(history.history['val_loss'], label='val_loss') # 验证集损失值
plt.title("全连接神经网络loss值图") # 标题
plt.xlabel("Epochs") # 横坐标
plt.ylabel("Loss") # 纵坐标
plt.legend() # 显示图例
plt.show() # 显示图像

# 绘制训练过程中的准确率（accuracy）变化曲线
plt.plot(history.history['accuracy'], label='train_accuracy')  # 训练集准确率
plt.plot(history.history['val_accuracy'], label='val_accuracy') # 验证集准确率
plt.title("全连接神经网络accuracy值图") # 标题
plt.xlabel("Epochs") # 横坐标
plt.ylabel("Accuracy") # 纵坐标
plt.legend() # 显示图例
plt.show() # 显示图像


# 8. 评估模型并打印分类报告
# 使用模型对测试集进行预测
y_pred_one_hot = model.predict(x_test) # 预测结果为one-hot编码
# 将one-hot编码的预测结果转换回类别标签 (0或1)
y_pred = np.argmax(y_pred_one_hot, axis=1) # axis=1 表示沿着行的方向操作

# 打印分类报告，详细展示每个类别的精确度、召回率和F1分数
print("\n分类报告:")
print(classification_report(y_test, y_pred)) # y_test为真实标签，y_pred为预测标签
