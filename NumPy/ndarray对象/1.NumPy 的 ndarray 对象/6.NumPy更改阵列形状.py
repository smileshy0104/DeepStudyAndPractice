# 在 NumPy 中，数据变形操作是处理数组时非常重要的功能。它允许我们改变数组的形状、大小和维度，以便更好地适应不同的计算需求。

# TODO 1.reshape() 方法用于改变数组的形状，而不改变其数据。
import numpy as np

# 创建一个一维数组
array = np.arange(12)  # 创建包含 0 到 11 的一维数组
print("原始数组:", array)   # [ 0  1  2  3  4  5  6  7  8  9 10 11]

# 改变形状为 3x4
reshaped_array = array.reshape(3, 4)
print("改变形状后的数组:\n", reshaped_array)

# TODO 2.flatten() 方法将多维数组展平为一维数组。
# 展平数组
flattened_array = reshaped_array.flatten()
print("展平后的数组:", flattened_array)

# TODO 3.ravel() 方法也可以展平数组，但它返回的是原数组的视图（如果可能的话），而不是副本。
# 使用 ravel 展平数组
raveled_array = reshaped_array.ravel()
print("使用 ravel 展平后的数组:", raveled_array)

# TODO 4.transpose() 方法用于转置数组，即交换数组的行和列。
# 转置数组
transposed_array = reshaped_array.transpose()
print("转置后的数组:\n", transposed_array)

# 使用 .T 转置数组
transposed_array_T = reshaped_array.T
print("使用 .T 转置后的数组:\n", transposed_array_T)

# TODO 5.np.newaxis 可以用来增加数组的维度。
# 增加维度
array_3d = array[np.newaxis, :]
print("增加维度后的数组形状:", array_3d.shape)

# TODO 6.通过 reshape() 将数组的形状调整为更少的维度。
# 删除维度
array_2d = array.reshape(3, 4)
array_1d = array_2d.reshape(-1)  # 自动计算维度
print("删除维度后的数组形状:", array_1d.shape)

# TODO 7.切片可以用于选择数组的特定部分。
# 切片操作
sliced_array = reshaped_array[1:, 1:]  # 选择第二行及以后的行和第二列及以后的列
print("切片后的数组:\n", sliced_array)

# TODO 8.可以使用 np.concatenate() 将多个数组连接在一起。
# 创建两个数组
array1 = np.array([[1, 2], [3, 4]])
array2 = np.array([[5, 6]])

# 连接数组
concatenated_array = np.concatenate((array1, array2), axis=0)
print("连接后的数组:\n", concatenated_array)

# TODO 9.可以使用 np.split() 将数组分割成多个部分。
# 分割数组
split_arrays = np.split(concatenated_array, 2)  # 将数组分割成 2 个部分
print("分割后的数组:")
for i, arr in enumerate(split_arrays):
    print(f"部分 {i}:\n{arr}")
