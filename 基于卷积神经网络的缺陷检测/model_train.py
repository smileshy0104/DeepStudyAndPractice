# 导入所需的库
import pathlib  # 用于处理文件路径
import numpy as np  # 用于数值运算
import pandas as pd  # 用于数据处理
import matplotlib.pyplot as plt  # 用于数据可视化
import keras  # 深度学习框架
import tensorflow as tf
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D  # 从Keras导入所需的层

# 解决matplotlib中文显示问题
# 提供一个字体列表，matplotlib会依次尝试，直到找到可用的字体
# 'PingFang HK' 是 macOS 上常见的字体, 'SimHei' 和 'Microsoft YaHei' 是 Windows 上常见的字体
plt.rcParams['font.sans-serif'] = ['PingFang HK', 'SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

# --- 数据准备 ---

# 定义训练集数据路径
data_train_path = '基于卷积神经网络的缺陷检测/data/train/'
data_train = pathlib.Path(data_train_path)  # 转换为pathlib对象以便更好地操作路径

# 定义验证集数据路径
data_val_path = '基于卷积神经网络的缺陷检测/data/val/'
data_val = pathlib.Path(data_val_path)  # 转换为pathlib对象

# 定义所有类别的名称
CLASS_NAMES = np.array(['Cr', 'In', 'Pa', 'PS', 'Rs', 'Sc'])

# --- 图像处理和数据生成器设置 ---

# 设置图像处理的参数
BATCH_SIZE = 64  # 每一批次处理的图片数量
IMG_HEIGHT = 32  # 图像高度
IM_WIDTH = 32  # 图像宽度

# 创建一个图像数据生成器，并进行归一化处理，加快数据处理速度
# 将像素值从 [0, 255] 缩放到 [0, 1] 区间
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

# 创建训练数据生成器
# 从目录中读取图片，并应用上述的转换
# 训练数据生成器默认从目录中读取图片，并应用上述的转换
# 生成一个批次的训练数据
train_data_gen = image_generator.flow_from_directory(
    directory=str(data_train),  # 训练数据路径（字符串）
    batch_size=BATCH_SIZE,  # 批次大小
    shuffle=True,  # 打乱数据
    target_size=(IMG_HEIGHT, IM_WIDTH),  # 调整图像大小为32*32
    classes=list(CLASS_NAMES)  # 指定类别列表
)

# 创建验证数据生成器
# 从目录中读取图片，并应用上述的转换
val_data_gen = image_generator.flow_from_directory(
    directory=str(data_val),  # 验证数据路径
    batch_size=BATCH_SIZE,  # 批次大小
    shuffle=True,  # 打乱数据
    target_size=(IMG_HEIGHT, IM_WIDTH),  # 调整图像大小
    classes=list(CLASS_NAMES)  # 指定类别列表
)

# --- 构建卷积神经网络 (CNN) 模型 ---

# 使用Keras的Sequential API来构建模型
model = keras.Sequential()

# 添加第一个卷积层
# filters=6: 卷积核数量
# kernel_size=5: 卷积核大小
# input_shape=(32, 32, 3): 输入图像的尺寸 (高度, 宽度, 通道数)
# activation='relu': 激活函数
model.add(Conv2D(filters=6, kernel_size=5, input_shape=(32, 32, 3), activation='relu'))
# 添加第一个最大池化层
# pool_size=(2, 2): 池化窗口大小
# strides=(2, 2): 步长
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

# 添加第二个卷积层
model.add(Conv2D(filters=16, kernel_size=5, activation='relu'))
# 添加第二个最大池化层
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

# 添加第三个卷积层
model.add(Conv2D(filters=120, kernel_size=5, activation='relu'))

# 将多维数据展平成一维
model.add(Flatten())

# 添加一个全连接层
# units=84: 神经元数量
model.add(Dense(84, activation='relu'))

# 添加输出层
# units=6: 输出类别的数量
# activation='softmax': Softmax函数用于多分类问题，输出每个类别的概率
model.add(Dense(6, activation='softmax'))

# --- 编译和训练模型 ---

# 编译卷积神经网络
# loss='categorical_crossentropy': 损失函数，适用于多分类问题
# optimizer='Adam': 优化器
# metrics=['accuracy']: 评估指标
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

# 传入数据集进行训练
# epochs=50: 训练轮数
history = model.fit(train_data_gen, validation_data=val_data_gen, epochs=50)

# --- 保存模型和可视化结果 ---

# 保存训练好的模型
model.save("基于卷积神经网络的缺陷检测/model.h5")

# 绘制训练过程中的损失值 (loss) 变化图
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title("CNN模型训练损失")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# 绘制训练过程中的准确率 (accuracy) 变化图
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title("CNN模型训练准确率")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
