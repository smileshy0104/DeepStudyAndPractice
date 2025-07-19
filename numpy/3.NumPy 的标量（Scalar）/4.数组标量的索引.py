# 在 NumPy 中，数组标量（如 numpy.int_、numpy.float_ 等）可以像 0 维数组一样进行索引。这种索引方式允许我们以类似于数组的方式访问和操作标量数据。下面是关于数组标量索引的详细说明和示例。
#
# 数组标量的索引
# 假设 x 是一个数组标量，我们可以使用以下方式进行索引：
#
# 使用 x[()]:
#
# 返回标量数组的副本。
# 这种方式用于获取标量的值。
# 使用 x[...]:
#
# 返回 0 维数组。
# 这是一种通用的方式，适用于获取数组的内容。
# 使用 x['field-name']:
#
# 当 x 是一个结构化数组标量时，可以通过字段名访问对应的数组标量。
# 这种方式用于访问结构化数据类型的特定字段。

import numpy as np

# 创建一个标量
scalar = np.float_(3.14)

# 使用 x[()] 获取标量副本
scalar_copy = scalar[()]
print("标量副本:", scalar_copy)            # 输出: 标量副本: 3.14

# 使用 x[...] 返回 0 维数组
zero_dim_array = scalar[...]
print("0维数组:", zero_dim_array)          # 输出: 0维数组: 3.14
print("0维数组类型:", type(zero_dim_array))  # 输出: <class 'numpy.float64'>

# 创建一个结构化数组
structured_array = np.array([(1, 2.0, 'A'), (2, 3.5, 'B')],
                             dtype=[('field1', 'i4'), ('field2', 'f4'), ('field3', 'U1')])

# 访问结构化数组的第一个元素
x = structured_array[0]

# 使用字段名访问
field_value = x['field2']
print("字段 field2 的值:", field_value)    # 输出: 字段 field2 的值: 2.0
