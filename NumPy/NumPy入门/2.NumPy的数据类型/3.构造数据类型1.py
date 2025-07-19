import numpy as np
# 每当 NumPy 函数或方法中需要某个数据类型时，就可以提供一个 dtype 对象或可以转换为一个的对象。
# 此类转换由 numpy.dtype 构造函数（ dtype(obj[, align, copy])）完成：
# 1. 使用数组标量类型
dtype_int16 = np.dtype(np.int16)
print("数组标量类型 int16:", dtype_int16)  # 输出: dtype('int16')

# 2. 结构化类型：一个字段名“f1”，包含 int16
# np.int16 表示 16 位整数。在 NumPy 中，int16 是一个有符号整数类型，占用 2 个字节。所以为<i2
dtype_struct1 = np.dtype([('f1', np.int16)])
print("结构化类型 f1 (int16):", dtype_struct1)  # 输出: dtype([('f1', '<i2')])
# 查看字段的字节顺序和类型
print("字段 'f1' 的字节顺序:", dtype_struct1['f1'].byteorder)  # 输出: <
print("字段 'f1' 的类型:", dtype_struct1['f1'].type)  # 输出: <class 'numpy.int16'>

# 3. 结构化类型：字段“f1”包含一个具有一个字段的结构化类型
dtype_struct2 = np.dtype([('f1', [('f1', np.int16)])])
print("结构化类型 f1 (包含结构化类型):", dtype_struct2)  # 输出: dtype([('f1', [('f1', '<i2')])])
# 查看字段的字节顺序和类型
print("字段 'f1' 的字节顺序:", dtype_struct2['f1'].byteorder)  # 输出: |
print("字段 'f1' 的类型:", dtype_struct2['f1'].type)  # 输出: <class 'numpy.void'>

# 4. 结构化类型：两个字段：无符号 int 和 int32
dtype_struct3 = np.dtype([('f1', np.uint64), ('f2', np.int32)])
print("结构化类型 f1 (uint64) 和 f2 (int32):", dtype_struct3)  # 输出: dtype([('f1', '<u8'), ('f2', '<i4')])
# 查看字段的字节顺序和类型
print("字段 'f1' 的字节顺序:", dtype_struct3['f1'].byteorder)  # 输出: =
print("字段 'f1' 的类型:", dtype_struct3['f1'].type)  # 输出: <class 'numpy.uint64'>
print("字段 'f2' 的字节顺序:", dtype_struct3['f2'].byteorder)  # 输出: =
print("字段 'f2' 的类型:", dtype_struct3['f2'].type)  # 输出: <class 'numpy.int32'>

# 5. 使用数组协议类型字符串
dtype_struct4 = np.dtype([('a', 'f8'), ('b', 'S10')])
print("结构化类型 a (float64) 和 b (字符串):", dtype_struct4)  # 输出: dtype([('a', '<f8'), ('b', 'S10')])
# 查看字段的字节顺序和类型
print("字段 'a' 的字节顺序:", dtype_struct4['a'].byteorder)  # 输出: =
print("字段 'a' 的类型:", dtype_struct4['a'].type)  # 输出: <class 'numpy.float64'>
print("字段 'b' 的字节顺序:", dtype_struct4['b'].byteorder)  # 输出: |
print("字段 'b' 的类型:", dtype_struct4['b'].type)  # 输出: <class 'numpy.bytes_'>

# 6. 使用逗号分隔的字段格式，形状为 (2,3)
dtype_struct5 = np.dtype("i4, (2,3)f8")
print("结构化类型 (i4, (2,3)f8):", dtype_struct5)  # 输出: dtype([('f0', '<i4'), ('f1', '<f8', (2, 3))])

# 7. 使用元组，int 是固定类型，3 表示字段的形状
dtype_struct6 = np.dtype([('hello', (np.int64, 3)), ('world', np.void, 10)])
print("结构化类型 hello (int64, 3) 和 world (void, 10):", dtype_struct6)  # 输出: dtype([('hello', '<i8', (3,)), ('world', 'V10')])

# 8. 将 int16 细分为 2 个 int8，称为 x 和 y
dtype_struct7 = np.dtype((np.int16, {'x': (np.int8, 0), 'y': (np.int8, 1)}))
print("细分 int16 为 x (int8) 和 y (int8):", dtype_struct7)  # 输出: dtype((numpy.int16, [('x', 'i1'), ('y', 'i1')]))

# 9. 使用字典，两个字段名“gender”和“age”
dtype_struct8 = np.dtype({'names': ['gender', 'age'], 'formats': ['S1', np.uint8]})
print("字典结构化类型 gender (S1) 和 age (uint8):", dtype_struct8)  # 输出: dtype([('gender', 'S1'), ('age', 'u1')])

# 10. 偏移量（字节），这里是0和25
dtype_struct9 = np.dtype({'surname': ('S25', 0), 'age': (np.uint8, 25)})
print("带偏移量的结构化类型 surname (S25) 和 age (uint8):", dtype_struct9)  # 输出: dtype([('surname', 'S25'), ('age', 'u1')])
