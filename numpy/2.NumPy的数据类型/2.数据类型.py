# 字节顺序概述
# 在 NumPy 中，数据类型对象的字节顺序由以下字符表示：
#
# '=': 本地字节顺序（native）
# '<': 小端法（little-endian）
# '>': 大端法（big-endian）
# '|': 不适用（not applicable）
# 字节顺序决定了数据在内存中的存储方式。小端法将最低有效字节存储在最低地址，而大端法将最高有效字节存储在最低地址。

import numpy as np

# 1. 创建不同字节顺序的数据类型
little_endian_dtype = np.dtype('<i4')  # 小端法 32 位整数
big_endian_dtype = np.dtype('>i4')     # 大端法 32 位整数
native_dtype = np.dtype('=i4')         # 本地字节顺序 32 位整数

# 2. 查看字节顺序
print("小端法字节顺序:", little_endian_dtype.byteorder)  # 输出: '<'
print("大端法字节顺序:", big_endian_dtype.byteorder)    # 输出: '>'
print("本地字节顺序:", native_dtype.byteorder)            # 输出: '='

# 3. 查看内置数据类型的字节顺序
print("np.int32 的字节顺序:", np.dtype(np.int32).byteorder)  # 输出可能为 '='

# 4. 创建一个包含字节顺序的数据类型
custom_dtype = np.dtype([('value', np.float32), ('label', np.unicode_, 10)])
print("自定义数据类型的字节顺序:", custom_dtype.byteorder)  # 输出可能为 '='

# 5. 通过字节顺序创建一个数组
data = np.array([1, 2, 3], dtype=little_endian_dtype)
print("小端法数组:", data)  # 输出: [1 2 3]

data_big_endian = np.array([1, 2, 3], dtype=big_endian_dtype)
print("大端法数组:", data_big_endian)  # 输出: [1 2 3]
