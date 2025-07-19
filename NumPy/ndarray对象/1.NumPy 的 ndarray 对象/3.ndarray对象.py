# TODO 创建 ndarray 对象
# TODO 1.从列表或元组创建
import numpy as np

# 从列表创建一维数组
array_1d = np.array([1, 2, 3, 4])
print("一维数组:\n", array_1d)

# 从嵌套列表创建二维数组
array_2d = np.array([[1, 2, 3], [4, 5, 6]])
print("二维数组:\n", array_2d)

# TODO 2.使用特殊函数创建
# 全零数组: np.zeros()
# 全一数组: np.ones()
# 单位矩阵: np.eye()
# 等间隔数组: np.arange()
# 线性间隔数组: np.linspace()
# 创建全零数组

zeros_array = np.zeros((2, 3))
print("全零数组:\n", zeros_array)

# 创建全一数组
ones_array = np.ones((3, 2))
print("全一数组:\n", ones_array)

# 创建单位矩阵
identity_matrix = np.eye(3)
print("单位矩阵:\n", identity_matrix)

# 创建等间隔数组
arange_array = np.arange(0, 10, 2)
print("等间隔数组:\n", arange_array)

# 创建线性间隔数组
linspace_array = np.linspace(0, 1, 5)
print("线性间隔数组:\n", linspace_array)

# TODO 3.ndarray 对象的属性
# ndarray.ndim: 数组的维度数（秩）。
# ndarray.shape: 数组的形状（各维度的大小）。
# ndarray.size: 数组中元素的总数。
# ndarray.dtype: 数组中元素的数据类型。
# ndarray.itemsize: 每个元素的字节大小。
# ndarray.nbytes: 数组占用的总字节数。
# 创建一个数组
array = np.array([[1, 2, 3], [4, 5, 6]])

# 打印数组的属性
print("数组的维度:", array.ndim)
print("数组的形状:", array.shape)
print("数组的大小:", array.size)
print("数组的数据类型:", array.dtype)
print("每个元素的字节大小:", array.itemsize) # 8
print("数组占用的总字节数:", array.nbytes) # 6*8 = 48

# TODO 4.ndarray 对象的方法
# ndarray.reshape(): 改变数组的形状。
# ndarray.flatten(): 将多维数组展平为一维数组。
# ndarray.transpose(): 转置数组。
# ndarray.sum(): 计算数组元素的和。
# ndarray.mean(): 计算数组元素的平均值。
# ndarray.max(): 查找数组中的最大值。
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

# 计算和、平均值和最大值
print("数组元素的和:", array.sum())
print("数组元素的平均值:", array.mean())
print("数组中的最大值:", array.max())

# TODO 5.视图与副本
# 视图: 通过切片或 reshape 创建的数组是原数组的视图，修改视图会影响原数组。
# 副本: 使用 copy() 方法创建的数组是原数组的副本，修改副本不会影响原数组。
# 创建一个数组
array = np.array([[1, 2, 3], [4, 5, 6]])

# 创建视图
view = array[0, :]
print("视图:\n", view)

# 修改视图
view[0] = 10
print("修改视图后的原数组:\n", array)

# 创建副本
copy = array.copy()
copy[0, 0] = 100
print("副本:\n", copy)
print("修改后的原数组:\n", array)
