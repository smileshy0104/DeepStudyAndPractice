import numpy as np

# 1. 创建一个 NumPy 数组并进行切片
sliced_array = np.array([10, 20, 30, 40])[:3]  # 切片，获取前 3 个元素
print("切片结果:", sliced_array)  # 输出: [10 20 30]

# 2. 创建两个 NumPy 数组
a = np.array([10, 20, 30, 40])
b = np.array([1, 2, 3, 4])

# 3. 矩阵相加
sum_array = a + b  # 对应元素相加
print("矩阵相加结果:", sum_array)  # 输出: [11 22 33 44]

# 4. 数组减去标量
subtracted_array = a - 1  # 每个元素减去 1
print("减去 1 的结果:", subtracted_array)  # 输出: [ 9 19 29 39]

# 5. 数组与正弦函数的乘积
sin_array = 4 * np.sin(a)  # 对 a 中的每个元素计算正弦并乘以 4
print("4 * sin(a) 的结果:", sin_array)  # 输出: 可能是一个浮点数组

# 6. 数学函数示例
print("最大值:", a.max())  # 输出: 40
print("最小值:", a.min())  # 输出: 10
print("总和:", a.sum())  # 输出: 100
print("标准差:", a.std())  # 输出: 11.180339887498949
print("所有元素为真:", a.all())  # 输出: True

# 7. 累积和
cumsum_array = a.cumsum()  # 计算累积和（依次累加）
print("累积和:", cumsum_array)  # 输出: [ 10  30  60 100]

# 8. 多维数组的指定方向求和
b_2d = np.array([[1, 2, 3], [4, 5, 6]])  # 创建一个 2D 数组
sum_axis_1 = b_2d.sum(axis=1)  # 沿着行方向求和
print("沿着行方向求和:", sum_axis_1)  # 输出: [ 6 15]

# 9. 计算相反数
negative_value = np.negative(-1)  # 计算 -1 的相反数
print("相反数:", negative_value)  # 输出: 1
