import numpy as np

# 1. 创建一个包含 32 位大端模式整数的数据类型
dt = np.dtype('>i4') # 使用 np.dtype('>i4') 创建一个包含 32 位大端模式整数的数据类型。

# 查看字节序
print("字节序:", dt.byteorder)  # 输出: '>'

# 查看项大小
print("项大小:", dt.itemsize)  # 输出: 4

# 查看数据类型名称
print("数据类型名称:", dt.name)  # 输出: 'int32'

# 检查类型
print("是否为 int32:", dt.type is np.int32)  # 输出: True

# 检查浮点数类型
print("np.float64 == np.dtype(np.float64) == np.dtype('float64'):", np.float64 == np.dtype(np.float64) == np.dtype('float64'))  # 输出: True
print("np.float64 == np.dtype(np.float64).type:", np.float64 == np.dtype(np.float64).type)  # 输出: True

# 2. 创建一个结构化数据类型，包含 16 个字符串和两个 64 位浮点数的子数组
dt_structured = np.dtype([('name', np.unicode_, 16), ('grades', np.float64, (2,))])

# 查看字段信息
print("name 字段的数据类型:", dt_structured['name'])  # 输出: dtype('<U16')
print("grades 字段的数据类型:", dt_structured['grades'])  # 输出: dtype(('<f8', (2,)))

# 3. 创建一个使用结构化数据类型的数组
x = np.array([('Sarah', (8.0, 7.0)), ('John', (6.0, 7.0))], dtype=dt_structured)

# 访问数组项
print("访问 x[1]:", x[1])  # 输出: ('John', [6., 7.])

# 访问 grades 字段
print("访问 x[1]['grades']:", x[1]['grades'])  # 输出: array([6., 7.])

# 检查类型
print("x[1] 的类型:", type(x[1]))  # 输出: <class 'numpy.void'>
print("x[1]['grades'] 的类型:", type(x[1]['grades']))  # 输出: <class 'numpy.ndarray'>
