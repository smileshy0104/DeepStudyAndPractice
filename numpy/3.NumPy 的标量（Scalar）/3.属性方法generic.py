# TODO numpy.generic 是 numpy 标量类型的基类，大多数（所有？）numpy 派生标量的类型出自它。
# 在 NumPy 中，generic 类作为一个基础类，主要用于从 ndarray 类派生出 NumPy 标量（如 int_、float_ 等）。
# 它的设计目的是为了提供一个统一的 API，使得 NumPy 标量能够共享 ndarray 的属性和方法。尽管 generic 类本身并不直接实现这些属性和方法，但它确实为 NumPy 标量提供了一个一致的接口。
#
# generic 类的特点
# 统一接口: generic 类提供了一种方式，使得所有 NumPy 标量可以通过相同的接口进行访问和操作。这种设计使得用户在处理标量时能够享受与处理数组相似的体验。
#
# 继承自 ndarray: 所有的 NumPy 标量类型都是从 generic 类派生而来，因此它们可以使用 ndarray 的许多属性和方法。这种继承关系使得标量在功能上更为强大。
#
# 虚拟属性: generic 类定义了一些虚拟属性，尽管它们尚未实现，但这些属性为未来的扩展提供了基础。
#
# 主要属性和方法
# 虽然 generic 类本身没有具体实现，但它所提供的属性和方法在 NumPy 标量中是可用的。以下是一些常见的属性和方法：
#
# 属性:
#
# dtype: 返回标量的类型。
# itemsize: 返回标量在内存中占用的字节数。
# ndim: 返回标量的维度（对于标量，通常为0）。
# shape: 返回标量的形状（对于标量，通常为()）。
# 方法:
#
# astype(dtype): 将标量转换为指定的数据类型。
# real: 返回标量的实部。
# imag: 返回标量的虚部（对于复数标量）。

import numpy as np

# 创建一些 NumPy 标量
int_scalar = np.int_(42)
float_scalar = np.float_(3.14)
complex_scalar = np.complex_(2 + 3j)
bool_scalar = np.bool_(True)

# 访问标量的属性
print("整数标量:")
print("值:", int_scalar)                          # 输出: 值: 42
print("数据类型:", int_scalar.dtype)              # 输出: 数据类型: int32
print("内存占用:", int_scalar.itemsize)           # 输出: 内存占用: 4
print("维度:", int_scalar.ndim)                   # 输出: 维度: 0
print("形状:", int_scalar.shape)                   # 输出: 形状: ()

print("\n浮点数标量:")
print("值:", float_scalar)                        # 输出: 值: 3.14
print("数据类型:", float_scalar.dtype)            # 输出: 数据类型: float64
print("内存占用:", float_scalar.itemsize)         # 输出: 内存占用: 8
print("维度:", float_scalar.ndim)                 # 输出: 维度: 0
print("形状:", float_scalar.shape)                 # 输出: 形状: ()

print("\n复数标量:")
print("值:", complex_scalar)                      # 输出: 值: (2+3j)
print("数据类型:", complex_scalar.dtype)          # 输出: 数据类型: complex128
print("内存占用:", complex_scalar.itemsize)       # 输出: 内存占用: 16
print("维度:", complex_scalar.ndim)               # 输出: 维度: 0
print("形状:", complex_scalar.shape)               # 输出: 形状: ()
print("实部:", complex_scalar.real)                # 输出: 实部: 2.0
print("虚部:", complex_scalar.imag)                # 输出: 虚部: 3.0

print("\n布尔标量:")
print("值:", bool_scalar)                         # 输出: 值: True
print("数据类型:", bool_scalar.dtype)             # 输出: 数据类型: bool
print("内存占用:", bool_scalar.itemsize)          # 输出: 内存占用: 1
print("维度:", bool_scalar.ndim)                  # 输出: 维度: 0
print("形状:", bool_scalar.shape)                  # 输出: 形状: ()
