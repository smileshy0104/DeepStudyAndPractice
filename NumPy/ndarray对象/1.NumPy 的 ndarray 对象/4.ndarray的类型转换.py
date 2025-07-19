# 在 NumPy 中，ndarray 对象的类型转换（或称为数据类型转换）非常常见，通常用于确保数组中的数据类型符合特定要求。
# NumPy 提供了多种方法来进行类型转换，包括使用 astype() 方法和其他相关函数。

# TODO 1.使用 astype() 方法进行类型转换
# 示例 1: 将整数数组转换为浮点数组

import numpy as np

# 创建一个整数数组
array_int = np.array([1, 2, 3, 4])
print("原始整数数组:", array_int)

# 转换为浮点数组
array_float = array_int.astype(float) # [1 2 3 4]
print("转换后的浮点数组:", array_float) # [1. 2. 3. 4.]

# 示例 2: 将浮点数组转换为整数数组

# 创建一个浮点数组
array_float = np.array([1.5, 2.7, 3.8, 4.9])
print("原始浮点数组:", array_float)

# 转换为整数数组（注意：小数部分会被截断）
array_int = array_float.astype(int) # [1.5 2.7 3.8 4.9]
print("转换后的整数数组:", array_int) # [1 2 3 4]

# 示例 3: 指定数据类型
# 创建一个数组并指定数据类型
array_float = np.array([1, 2, 3, 4], dtype=np.float32)
print("指定数据类型的数组:", array_float)
print("数据类型:", array_float.dtype)

# 示例 4: 使用 np.asarray()
# np.asarray(): 将输入转换为数组，如果输入已经是数组，则返回原数组。可以指定数据类型。
# 创建一个列表
list_data = [1, 2, 3, 4]

# 使用 asarray 转换为数组并指定数据类型
array = np.asarray(list_data, dtype=np.float64)
print("使用 asarray 转换的数组:", array)
print("数据类型:", array.dtype)

# 示例 5: 使用 np.copy()
# 创建一个整数数组
array_int = np.array([1, 2, 3, 4])

# 创建副本并转换为浮点类型
array_float_copy = array_int.copy().astype(float)
print("副本转换后的浮点数组:", array_float_copy)
print("数据类型:", array_float_copy.dtype)

# 示例 6: 布尔类型转换
# 创建一个数组
array = np.array([1, 0, 3, 0, 5])

# 转换为布尔类型
bool_array = array.astype(bool)
print("布尔类型数组:", bool_array) # [ True False  True False  True]

# TODO 基础案例
# 创建一个二维数组，包含两个子数组
a = np.array([[1,2,3],[4,5,6]])
# a[:, 1] 中的 : 表示选择所有行，而 1 表示选择第二列（索引从 0 开始）。
print(a[:,1]) # 输出第二列的元素，结果为 array([2, 5])
# 将第二列的元素转换为字符串类型
print(a[:,1].astype('str')) # 转换后生成此副本  array(['2', '5'], dtype='<U21')
# 打印原数组的数据类型
print(a.dtype)   # 原来的数组没有改变  dtype('int64')

# 表达式转换为 int 生成布尔数组（0/1）
print((a > 0.5).astype(int))

# 构造一个时间表达数据
arr = [2020, 12, 0.6552562894775783]
# 定义自定义数据类型，包括年份、一年中的第几天和反射率
custom_type = np.dtype([
                        ('YEAR',np.uint16),
                        ('DOY', np.uint16),
                        ('REF',np.float16)
                        ])
# 使用自定义类型创建数组
d = np.array([tuple(arr)], custom_type)
# 打印自定义数组及其数据类型
print(d) # [(2020, 12, 0.6553)]
print(d.dtype) # [('YEAR', '<u2'), ('DOY', '<u2'), ('REF', '<f2')]
# 打印数组中的年份字段
print(d['YEAR']) # [2020]


