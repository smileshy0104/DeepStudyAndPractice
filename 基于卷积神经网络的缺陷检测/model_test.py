# 导入所需的库
import numpy as np  # 用于数值运算
import tensorflow as tf  # TensorFlow 框架
from keras.models import load_model  # 从 Keras 加载训练好的模型
import cv2  # OpenCV 库，用于图像处理

# 定义类别名称，与训练时保持一致
CLASS_NAMES = np.array(['Cr', 'In', 'Pa', 'PS', 'Rs', 'Sc'])

# 设置图像的尺寸，必须与模型训练时使用的尺寸相同
IMG_HEIGHT = 32
IM_WIDTH = 32

# --- 模型加载 ---
# 从 .h5 文件加载已经训练好的模型
model = load_model("基于卷积神经网络的缺陷检测/model.h5")

# --- 图像预处理 ---
# 读取待测试的图像
# 注意：这里硬编码了测试图片路径，可以根据需要进行修改
src = cv2.imread("基于卷积神经网络的缺陷检测/data/val/Cr/Cr_48.bmp")
# 将图像尺寸调整为模型输入所需的尺寸
src = cv2.resize(src, (32, 32))
# 将图像数据类型转换为整数
src = src.astype("int32")
# 对图像进行归一化，将像素值从 [0, 255] 缩放到 [0, 1]
src = src / 255

# --- 模型预测 ---
# 在第0维增加一个维度，以匹配模型输入的形状 (batch_size, height, width, channels)
# 模型期望的是一个批次的图像，所以即使只有一张图片，也需要增加一个批次维度
test_img = tf.expand_dims(src, 0)

# 使用模型对预处理后的图像进行预测
preds = model.predict(test_img)

# 获取预测结果中的分数
# preds 是一个二维数组，[[...]]，我们取第一个（也是唯一一个）结果
score = preds[0]

# --- 结果输出 ---
# 找到分数最高的类别的索引
predicted_class_index = np.argmax(score)
# 获取最高的分数（概率）
max_score = np.max(score)
# 根据索引获取对应的类别名称
predicted_class_name = CLASS_NAMES[predicted_class_index]

# 打印预测结果
print('模型预测的结果为{}， 概率为{}'.format(predicted_class_name, max_score))