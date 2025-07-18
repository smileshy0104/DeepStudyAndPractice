import numpy as np

# 创建一个 2x2 的示例数组
n = np.array([[1, 2], [3, 4]])
print("原数组:\n", n)

# 1. 数组的形状，返回值是一个元组。
# TODO 返回数组的形状，输出为一个元组，表示每个维度的大小。
shape = n.shape
print("n.shape:", shape)  # 输出: (2, 2)

# 2. 改变形状 直接修改数组的形状为 4 行 1 列。
n.shape = (4, 1)  # 将数组的形状更改为 4 行 1 列
print("改变形状后的数组:\n", n)  # 输出: [[1] [2] [3] [4]]

# 3. 改变原数组的形状创建一个新的
a = n.reshape((2, 2))  # 创建一个新的数组，其形状为 2x2
print("reshape((2, 2)) 创建的新数组:\n", a)  # 输出: [[1 2] [3 4]]

# 4. 数据类型
dtype = n.dtype
print("n.dtype:", dtype)  # 输出: int64 (或其他具体数据类型)

# 5. 维度数
ndim = n.ndim
print("n.ndim:", ndim)  # 输出: 2 (因为 n 是一个二维数组)

# 6. 元素数
size = n.size
print("n.size:", size)  # 输出: 4 (数组中总共有 4 个元素)
