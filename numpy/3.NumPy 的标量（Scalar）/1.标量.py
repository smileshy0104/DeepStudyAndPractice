# NumPy 的标量（Scalar）是指单个数值数据，通常用于表示一个特定类型的数值。与数组不同，标量不包含多个元素，而是————仅表示一个值。
# NumPy 提供了多种标量类型，以支持不同的数据类型和精度。
#
# NumPy 标量的特点
# 单一值: 标量————只包含一个值，而不是像数组那样包含多个值。
# 类型: NumPy 支持多种标量类型，包括整数、浮点数、复数、布尔值等。
# 高效: NumPy 的标量类型在内存中占用更少的空间，并且在计算时通常比 Python 的内置类型更高效。

import numpy as np

# 1. 整数标量
scalar_int = np.int32(10)
print("整数标量:", scalar_int, "类型:", type(scalar_int))  # 输出: 整数标量: 10 类型: <class 'numpy.int32'>

# 2. 浮点数标量
scalar_float = np.float64(3.14)
print("浮点数标量:", scalar_float, "类型:", type(scalar_float))  # 输出: 浮点数标量: 3.14 类型: <class 'numpy.float64'>

# 3. 复数标量
scalar_complex = np.complex128(2 + 3j)
print("复数标量:", scalar_complex, "类型:", type(scalar_complex))  # 输出: 复数标量: (2+3j) 类型: <class 'numpy.complex128'>

# 4. 布尔标量
scalar_bool = np.bool_(True)
print("布尔标量:", scalar_bool, "类型:", type(scalar_bool))  # 输出: 布尔标量: True 类型: <class 'numpy.bool_'>

# 5. 字符串标量
scalar_string = np.str_('Hello')
print("字符串标量:", scalar_string, "类型:", type(scalar_string))  # 输出: 字符串标量: Hello 类型: <class 'numpy.str_'>


# 1. 标量加法
result_add = scalar_int + 5
print("加法结果:", result_add)  # 输出: 加法结果: 15

# 2. 标量乘法
result_mul = scalar_float * 2
print("乘法结果:", result_mul)  # 输出: 乘法结果: 6.28

# 3. 标量与数组的运算
array = np.array([1, 2, 3])
result_array_add = array + scalar_int
print("数组加法结果:", result_array_add)  # 输出: 数组加法结果: [11 12 13]

# TODO 错误示例 只能用于单个元素才可以转换为标量
a = np.array([1,2,3])
float(a)
# TypeError: only size-1 arrays can be converted to Python scalars

b = np.array([1])
float(b)
# 1.0