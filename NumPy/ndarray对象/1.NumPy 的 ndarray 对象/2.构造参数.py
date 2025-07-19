# 在 NumPy 中，数组对象（ndarray）是用于————表示固定大小的多维同构数组的核心数据结构。
# 它提供了高效的存储和操作多维数据的功能。
# 下面是关于 NumPy 数组对象的一些详细信息，包括如何构造数组、其属性和方法。

# TODO 1.从列表或元组创建数组
import numpy as np

# 从列表创建一维数组
array_1d = np.array([1, 2, 3, 4])
print("一维数组:\n", array_1d)

# 从嵌套列表创建二维数组
array_2d = np.array([[1, 2, 3], [4, 5, 6]])
print("二维数组:\n", array_2d)

# TODO 2.使用 np.zeros() 和 np.ones()

# 创建一个 2x3 的全零数组
zeros_array = np.zeros((2, 3))
print("全零数组:\n", zeros_array)

# 创建一个 3x2 的全一数组
ones_array = np.ones((3, 2))
print("全一数组:\n", ones_array)

# 创建一个 3x2 的全一数组
ones_array1 = np.empty((3, 2))
print("全一数组:\n", ones_array1)


# TODO 3. 使用 np.arange() 和 np.linspace()

# 使用 np.arange() 创建数组
arange_array = np.arange(0, 10, 2)  # 从 0 到 10，步长为 2
print("等间隔数组:\n", arange_array)

# 使用 np.linspace() 创建数组
linspace_array = np.linspace(0, 1, 5)  # 从 0 到 1，包含 5 个点
print("线性间隔数组:\n", linspace_array)

# TODO 4. 使用 np.eye() 创建单位矩阵

# 创建 3x3 的单位矩阵
identity_matrix = np.eye(3)
print("单位矩阵:\n", identity_matrix)

# TODO 5. NumPy 数组的属性
# ndarray.ndim: 数组的维度数。
# ndarray.shape: 数组的形状（各维度的大小）。
# ndarray.size: 数组中元素的总数。
# ndarray.dtype: 数组中元素的数据类型。
# 创建一个数组
array = np.array([[1, 2, 3], [4, 5, 6]])

# 打印数组的属性
print("数组的维度:", array.ndim)
print("数组的形状:", array.shape)
print("数组的大小:", array.size)
print("数组的数据类型:", array.dtype)

# TODO 6. NumPy 数组的方法
# ndarray.reshape(): 改变数组的形状。
# ndarray.flatten(): 将多维数组展平为一维数组。
# ndarray.transpose(): 转置数组。

# 创建一个数组
array = np.array([[1, 2, 3], [4, 5, 6]])

# 改变数组形状
reshaped_array = array.reshape(3, 2)
print("改变形状后的数组:\n", reshaped_array)

# 展平数组
flattened_array = array.flatten()
print("展平后的数组:\n", flattened_array)

# 转置数组
transposed_array = array.transpose()
print("转置后的数组:\n", transposed_array)


