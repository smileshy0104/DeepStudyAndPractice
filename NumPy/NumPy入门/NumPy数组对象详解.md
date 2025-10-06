# NumPy 数组对象详解：ndarray 完全指南

## 什么是 ndarray？

想象一下，ndarray 就像是 NumPy 的"超级容器"。与 Python 普通的列表容器不同，这个超级容器专门为数学计算设计，所有放在里面的元素都必须是相同类型的，并且容器本身知道自己的大小、形状和各种操作方法。

### 简单理解 ndarray

```python
import numpy as np

# Python普通列表 - 可以放不同类型的东西
python_list = [1, "hello", 3.14, True]
print("Python列表:", python_list)
print("元素类型各不相同")

# NumPy数组 - 必须是相同类型
numpy_array = np.array([1, 2, 3, 4, 5])
print("NumPy数组:", numpy_array)
print("所有元素都是整数类型:", numpy_array.dtype)

# ndarray的优势
print("数组形状:", numpy_array.shape)
print("数组维度:", numpy_array.ndim)
print("数组大小:", numpy_array.size)
print("内存使用:", numpy_array.nbytes, "字节")
```

## ndarray 的核心属性

### 1. 基本属性

```python
# ndarray的核心属性
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int32)

print("数组:")
print(arr)

print("\n基本属性:")
print("形状 (shape):", arr.shape)        # (2, 4) - 2行4列
print("维度 (ndim):", arr.ndim)          # 2 - 二维数组
print("大小 (size):", arr.size)          # 8 - 总共8个元素
print("数据类型 (dtype):", arr.dtype)    # int32 - 32位整数
print("每个元素大小 (itemsize):", arr.itemsize)  # 4字节
print("总内存占用 (nbytes):", arr.nbytes)        # 32字节
```

### 2. 维度相关属性

```python
# 不同维度数组的属性展示
print("\n不同维度数组的属性:")

# 0维数组（标量）
scalar = np.array(42)
print(f"0维数组: 值={scalar}, 形状={scalar.shape}, 维度={scalar.ndim}")
# 0维数组: 值=42, 形状=(), 维度=0

# 1维数组（向量）
vector = np.array([1, 2, 3, 4, 5])
print(f"1维数组: 形状={vector.shape}, 维度={vector.ndim}")
# 1维数组: 形状=(5,), 维度=1

# 2维数组（矩阵）
matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(f"2维数组: 形状={matrix.shape}, 维度={matrix.ndim}")
# 2维数组: 形状=(2, 3), 维度=2

# 3维数组（张量）
tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(f"3维数组: 形状={tensor.shape}, 维度={tensor.ndim}")
# 3维数组: 形状=(2, 2, 2), 维度=3

# 解释形状的含义
print("\n形状解释:")
print("scalar.shape = () - 没有维度，就是一个数值")
print("vector.shape = (5,) - 5个元素的一维数组")
print("matrix.shape = (2, 3) - 2行3列的二维数组")
print("tensor.shape = (2, 2, 2) - 2个2×2的矩阵")
```

## ndarray 的创建方法

### 1. 从 Python 数据结构创建

```python
# 从Python列表创建数组
print("从Python列表创建:")

# 1维数组
list_1d = [1, 2, 3, 4, 5]
arr_1d = np.array(list_1d)
print("1维数组:", arr_1d)

# 2维数组
list_2d = [[1, 2, 3], [4, 5, 6]]
arr_2d = np.array(list_2d)
print("2维数组:")
print(arr_2d)

# 指定数据类型
arr_float = np.array([1, 2, 3], dtype=np.float64)
print("指定浮点类型:", arr_float)
print("数据类型:", arr_float.dtype)

# 从嵌套列表创建（要求长度一致）
try:
    irregular = [[1, 2, 3], [4, 5]]  # 长度不一致
    arr_irregular = np.array(irregular)
except Exception as e:
    print("创建不规则数组会出错:", e)
```

### 2. 使用 NumPy 内置函数创建

```python
# NumPy内置创建函数
print("\nNumPy内置创建函数:")

# 创建等差数列
arange_arr = np.arange(0, 10, 2)  # 0到10，步长2
print("arange:", arange_arr)

# 创建等间隔数列
linspace_arr = np.linspace(0, 1, 5)  # 0到1，5个点
print("linspace:", linspace_arr)

# 创建全零数组
zeros_1d = np.zeros(5)
zeros_2d = np.zeros((2, 3))
print("1维零数组:", zeros_1d)
print("2维零数组:")
print(zeros_2d)

# 创建全一数组
ones_1d = np.ones(4)
ones_2d = np.ones((2, 2))
print("1维一数组:", ones_1d)
print("2维一数组:")
print(ones_2d)

# 创建单位矩阵
eye_matrix = np.eye(3)
print("单位矩阵:")
print(eye_matrix)

# 创建指定值的数组
full_array = np.full((2, 3), 7)
print("填充数组:")
print(full_array)
```

### 3. 创建随机数组

```python
# 随机数组创建
print("\n随机数组创建:")

# 设置随机种子以保证结果可重现
np.random.seed(42)

# 均匀分布随机数
uniform_random = np.random.random((2, 3))
print("均匀分布随机数:")
print(uniform_random)

# 标准正态分布随机数
normal_random = np.random.randn(2, 3)
print("标准正态分布随机数:")
print(normal_random)

# 指定范围的随机整数
int_random = np.random.randint(0, 10, (2, 3))
print("0-9随机整数:")
print(int_random)

# 从指定数组中随机选择
choices = np.array([1, 3, 5, 7, 9])
choice_random = np.random.choice(choices, (2, 4))
print("从指定值中随机选择:")
print(choice_random)
```

### 4. 从现有数组创建

```python
# 从现有数组创建新数组
print("\n从现有数组创建:")

original = np.array([1, 2, 3, 4, 5])

# 复制数组
copied = np.copy(original)
print("复制的数组:", copied) # [1 2 3 4 5]

# 创建类似形状的数组
zeros_like = np.zeros_like(original)
ones_like = np.ones_like(original)
empty_like = np.empty_like(original)

print("类似形状的零数组:", zeros_like) # [0 0 0 0 0]
print("类似形状的一数组:", ones_like) # [1 1 1 1 1]

# 改变数组形状但保持数据
reshaped = original.reshape((1, 5))
print("重塑后的数组:")
print(reshaped) # [[1 2 3 4 5]]
print("新形状:", reshaped.shape)
```

## 数组索引和切片

### 1. 基本索引

```python
# 数组索引和切片
print("数组索引和切片:")

# 创建测试数组
arr_1d = np.array([10, 20, 30, 40, 50])
arr_2d = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])

print("1维数组:", arr_1d)
print("2维数组:")
print(arr_2d)

# 1维数组索引
print("\n1维数组索引:")
print("arr_1d[0] =", arr_1d[0])      # 第一个元素
print("arr_1d[-1] =", arr_1d[-1])    # 最后一个元素
print("arr_1d[2] =", arr_1d[2])      # 第三个元素

# 2维数组索引
print("\n2维数组索引:")
print("arr_2d[0, 0] =", arr_2d[0, 0])  # 第一行第一列
print("arr_2d[1, 2] =", arr_2d[1, 2])  # 第二行第三列
print("arr_2d[-1, -1] =", arr_2d[-1, -1])  # 最后一个元素

# 整行/整列索引
print("\n整行/整列索引:")
print("第一行:", arr_2d[0])          # 等同于 arr_2d[0, :]
print("第二列:", arr_2d[:, 1])      # 所有行的第二列
```

### 2. 切片操作

```python
# 切片操作
print("\n切片操作:")

# 1维数组切片
print("1维数组切片:")
print("arr_1d[1:4] =", arr_1d[1:4])    # 索引1到3
print("arr_1d[:3] =", arr_1d[:3])      # 前三个元素
print("arr_1d[2:] =", arr_1d[2:])      # 从索引2开始到末尾
print("arr_1d[::2] =", arr_1d[::2])    # 每隔一个元素
print("arr_1d[::-1] =", arr_1d[::-1])  # 反向数组

# 2维数组切片
print("\n2维数组切片:")
print("arr_2d[0:2, 1:3] =")
print(arr_2d[0:2, 1:3])  # 前两行，第2-3列

print("\narr_2d[:, 2:] =")
print(arr_2d[:, 2:])      # 所有行，从第3列开始

print("\narr_2d[1:, :] =")
print(arr_2d[1:, :])      # 从第2行开始，所有列

# 步长切片
print("\n步长切片:")
print("arr_2d[::2, ::2] =")
print(arr_2d[::2, ::2])   # 每隔一行一列
```

### 3. 高级索引

```python
# 高级索引
print("\n高级索引:")

# 布尔索引
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# 创建布尔条件
condition = arr > 5
print("条件 (arr > 5):", condition)
# 条件 (arr > 5): [False False False False False  True  True  True  True  True]
print("满足条件的元素:", arr[condition])
# 满足条件的元素: [ 6  7  8  9 10]

# 多个条件
condition2 = (arr > 3) & (arr < 8)
print("3 < arr < 8:", arr[condition2])
# 3 < arr < 8: [4 5 6 7]

# 花式索引（使用整数数组索引）
arr_2d = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])

print("\n原数组:")
print(arr_2d)

# 选择特定行
print("第0行和第2行:")
print(arr_2d[[0, 2]])
# [[ 1  2  3  4]
#  [ 9 10 11 12]]

# 选择特定元素
print("特定位置 [0,1], [1,2], [2,3]:")
print(arr_2d[[0, 1, 2], [1, 2, 3]])
# [ 2  7 12]

# 使用np.ix_创建网格索引
print("\n使用np.ix_选择子矩阵:")
rows = np.array([0, 2])
cols = np.array([1, 3])
print(arr_2d[np.ix_(rows, cols)])
```

## 数组形状操作

### 1. 改变数组形状

```python
# 数组形状操作
print("数组形状操作:")

# 创建一维数组
arr = np.arange(12)
print("原数组:", arr) # [ 0  1  2  3  4  5  6  7  8  9 10 11]
print("原形状:", arr.shape) # (12,)

# reshape改变形状
reshaped = arr.reshape(3, 4)
print("\nreshape为(3,4):")
print(reshaped)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]
print("新形状:", reshaped.shape)

# reshape为其他形状
reshaped_2x6 = arr.reshape(2, 6)
print("\nreshape为(2,6):")
# [[ 0  1  2  3  4  5]
#  [ 6  7  8  9 10 11]]
print(reshaped_2x6)

# reshape为3维
reshaped_3d = arr.reshape(2, 2, 3)
print("\nreshape为(2,2,3):")
print(reshaped_3d)
# [[[ 0  1  2]
  # [ 3  4  5]]
#
#  [[ 6  7  8]
  # [ 9 10 11]]]

# 使用-1自动计算维度
auto_reshaped = arr.reshape(3, -1)  # 自动计算列数
print("\nreshape为(3,-1)自动计算:")
print(auto_reshaped)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]
```

### 2. 展平数组

```python
# 展平数组
print("\n展平数组:")

# 创建2维数组
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
print("原数组:")
print(arr_2d)

# flatten - 创建副本,将二维数组展平成一维。
# 返回一个新数组（副本），与原数组无关。
# 不会影响原数组
flattened = arr_2d.flatten()
print("\nflatten结果:")
print(flattened) # [1 2 3 4 5 6]

# ravel - 创建视图（可能）,将二维数组展平成一维。
# 尽可能返回原数组的视图（共享内存），只有在必要时（如非连续内存）才返回副本。
# 修改可能会反映到原数组中
raveled = arr_2d.ravel()
print("ravel结果:")
print(raveled)

# 区别flatten和ravel
flattened[0] = 999
print("\n修改flatten后的原数组:")
print(arr_2d)  # 原数组不变

raveled[0] = 888
print("修改ravel后的原数组:")
print(arr_2d)  # 原数组可能改变
```

### 3. 转置和轴变换

```python
# 转置和轴变换
print("\n转置和轴变换:")

# 创建2维数组
arr = np.array([[1, 2, 3], [4, 5, 6]])
print("原数组:")
print(arr)
# [[1 2 3]
#  [4 5 6]]

# 转置
transposed = arr.T
print("\n转置:")
print(transposed)
# [[1 4]
#  [2 5]
#  [3 6]]

# 使用transpose函数,功能与 .T 完全相同，只是写法不同。
# 但 np.transpose() 更通用——它可以指定多维数组的轴顺序。
transposed_func = np.transpose(arr)
print("使用transpose函数:")
print(transposed_func)
# [[1 4]
#  [2 5]
#  [3 6]]

# 3维数组的轴变换
arr_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print("\n3维数组:")
print(arr_3d)
# [
#    [[1, 2], [3, 4]],
#     [[5, 6], [7, 8]]
# ]
print("形状:", arr_3d.shape) # (2, 2, 2)

# 交换轴
swapped = np.swapaxes(arr_3d, 0, 2)
print("\n交换轴0和2:")
print(swapped)
print("新形状:", swapped.shape)

# 使用transpose指定轴的顺序
reordered = np.transpose(arr_3d, (2, 0, 1))
print("\n重新排列轴(2,0,1):")
print(reordered)
print("新形状:", reordered.shape)
```

### 4. 数组连接和分割

```python
# 数组连接和分割
print("\n数组连接和分割:")

# 创建测试数组
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

# 连接数组
concatenated = np.concatenate([arr1, arr2])
print("连接1维数组:", concatenated)
# [1 2 3 4 5 6]

# 连接2维数组
arr2d_1 = np.array([[1, 2], [3, 4]])
arr2d_2 = np.array([[5, 6], [7, 8]])

# 垂直连接
vstacked = np.vstack([arr2d_1, arr2d_2])
print("\n垂直连接:")
print(vstacked)
# [[1 2]
#  [3 4]
#  [5 6]
#  [7 8]]

# 水平连接
hstacked = np.hstack([arr2d_1, arr2d_2])
print("\n水平连接:")
print(hstacked)
# [[1 2 5 6]
#  [3 4 7 8]]

# 分割数组
to_split = np.array([1, 2, 3, 4, 5, 6])

# 等分分割
split_equal = np.split(to_split, 3)
print("\n等分分割为3部分:", split_equal)
# [array([1, 2]), array([3, 4]), array([5, 6])]

# 按位置分割
split_positions = np.split(to_split, [2, 4])
print("在位置2,4分割:", split_positions)
# [array([1, 2]), array([3, 4]), array([5, 6])]
```

## 数组运算

### 1. 数学运算

```python
# 数组数学运算
print("数组数学运算:")

# 创建测试数组
arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([5, 6, 7, 8])

# 基本运算
print("数组1:", arr1)
print("数组2:", arr2)

print("\n加法:", arr1 + arr2)
print("减法:", arr1 - arr2)
print("乘法:", arr1 * arr2)
print("除法:", arr1 / arr2)
print("幂运算:", arr1 ** 2)

# 与标量运算（广播）
print("\n与标量运算:")
print("arr1 + 10:", arr1 + 10)
print("arr1 * 2:", arr1 * 2)

# 数学函数
print("\n数学函数:")
print("平方根:", np.sqrt(arr1))
print("指数:", np.exp(arr1))
print("对数:", np.log(arr1))
print("正弦:", np.sin(arr1))
```

### 2. 统计运算

```python
# 统计运算
print("\n统计运算:")

# 创建测试数组
data = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print("数据:")
print(data)

# 基本统计
print("\n基本统计:")
print("总和:", np.sum(data))
print("均值:", np.mean(data))
print("标准差:", np.std(data))
print("方差:", np.var(data))
print("最小值:", np.min(data))
print("最大值:", np.max(data))
print("中位数:", np.median(data))

# 按轴统计
print("\n按轴统计:")
print("每行求和:", np.sum(data, axis=1))
print("每列求和:", np.sum(data, axis=0))
print("每行均值:", np.mean(data, axis=1))
print("每列均值:", np.mean(data, axis=0))

# 累积运算
print("\n累积运算:")
print("累积求和:", np.cumsum(data))
print("累积乘积:", np.cumprod(data))
```

### 3. 比较运算

```python
# 比较运算
print("\n比较运算:")

arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([2, 2, 2, 4])

print("数组1:", arr1)
print("数组2:", arr2)

# 元素级比较
print("\n元素级比较:")
print("arr1 > arr2:", arr1 > arr2)
print("arr1 < arr2:", arr1 < arr2)
print("arr1 == arr2:", arr1 == arr2)
print("arr1 != arr2:", arr1 != arr2)

# 数组级比较
print("\n数组级比较:")
print("arr1.all():", arr1.all())  # 所有元素都为True？
print("arr1.any():", arr1.any())  # 任意元素为True？

# 比较函数
print("\n比较函数:")
print("数组相等:", np.array_equal(arr1, arr2))
print("数组近似相等:", np.allclose(arr1, arr2, rtol=0.1))
```

## 数组的视图和副本

### 1. 视图 vs 副本

```python
# 视图和副本的区别
print("视图和副本的区别:")

original = np.array([1, 2, 3, 4, 5])
print("原数组:", original) # [1 2 3 4 5]

# 创建视图
view = original[1:4]
print("视图:", view) # [2 3 4]

# 创建副本
copy = original[1:4].copy()
print("副本:", copy) # [2 3 4]

# 修改视图
view[0] = 99
print("修改视图后:")
print("原数组:", original)  # 原数组被修改
print("视图:", view)

# 修改副本
copy[0] = 88
print("\n修改副本后:")
print("原数组:", original)  # 原数组不变
print("副本:", copy)

# 检查是否为视图
print("\n视图检查:")
print("view.base是original吗?", view.base is original)
print("copy.base是original吗?", copy.base is original)
print("view.owns_data:", view.owns_data)
print("copy.owns_data:", copy.owns_data)
```

### 2. 内存管理

```python
# 内存管理
print("\n内存管理:")

# 创建大数组
large_arr = np.arange(1000000)
print(f"大数组大小: {large_arr.nbytes / 1024 / 1024:.2f} MB")

# 创建视图（不增加内存使用）
view = large_arr[:100000]
print(f"视图大小: {view.nbytes / 1024 / 1024:.2f} MB")

# 创建副本（增加内存使用）
copy = large_arr[:100000].copy()
print(f"副本大小: {copy.nbytes / 1024 / 1024:.2f} MB")

# 删除数组释放内存
del large_arr, view, copy
print("\n删除数组后内存被释放")
```

## 数组的数据类型操作

### 1. 类型转换

```python
# 数据类型操作
print("数据类型操作:")

# 创建不同类型的数组
int_arr = np.array([1, 2, 3, 4, 5], dtype=np.int32)
float_arr = int_arr.astype(np.float64)
bool_arr = int_arr > 3

print("整数数组:", int_arr, "类型:", int_arr.dtype)
print("浮点数组:", float_arr, "类型:", float_arr.dtype)
print("布尔数组:", bool_arr, "类型:", bool_arr.dtype)

# 类型转换的影响
print("\n类型转换的影响:")
original = np.array([1.5, 2.7, 3.9])
to_int = original.astype(np.int32)
print("原数组:", original)
print("转换为整数:", to_int)  # 截断小数部分

# 安全转换
print("\n安全类型转换:")
def safe_convert(arr, target_dtype):
    """安全的类型转换"""
    try:
        converted = arr.astype(target_dtype)
        print(f"成功转换: {arr.dtype} -> {converted.dtype}")
        return converted
    except Exception as e:
        print(f"转换失败: {e}")
        return arr

# 测试安全转换
large_int = np.array([1000, 2000, 3000])
safe_convert(large_int, np.int8)  # 可能溢出
```

### 2. 结构化数组

```python
# 结构化数组
print("\n结构化数组:")

# 定义结构化数据类型
student_dtype = np.dtype([
    ('name', 'U20'),      # 姓名字符串
    ('age', 'i1'),        # 年龄整数
    ('height', 'f4'),     # 身高浮点数
    ('weight', 'f4')      # 体重浮点数
])

# 创建结构化数组
students = np.array([
    ('Alice', 18, 1.65, 55.0),
    ('Bob', 19, 1.75, 70.0),
    ('Charlie', 20, 1.80, 75.0)
], dtype=student_dtype)

print("学生记录:")
print(students)

# 访问字段
print("\n访问字段:")
print("所有姓名:", students['name'])
print("所有年龄:", students['age'])
print("平均身高:", np.mean(students['height']))

# 访问记录
print("\n访问记录:")
print("第一个学生:", students[0])
print("第一个学生姓名:", students[0]['name'])
```

## 数组的文件操作

### 1. 保存和加载

```python
# 数组的文件操作
print("数组文件操作:")

# 创建测试数组
data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)

# 保存为NumPy格式
np.save('test_array.npy', data)
print("保存为.npy文件")

# 加载NumPy格式文件
loaded_data = np.load('test_array.npy')
print("从.npy文件加载:")
print(loaded_data)
print("数据相同:", np.array_equal(data, loaded_data))

# 保存为文本格式
np.savetxt('test_array.txt', data, delimiter=',')
print("\n保存为文本文件")

# 从文本文件加载
loaded_text = np.loadtxt('test_array.txt', delimiter=',')
print("从文本文件加载:")
print(loaded_text)

# 清理文件
import os
os.remove('test_array.npy')
os.remove('test_array.txt')
print("\n清理测试文件")
```

## 数组的性能优化

### 1. 内存布局优化

```python
# 性能优化
print("性能优化:")

# C顺序 vs Fortran顺序
c_array = np.array([[1, 2, 3], [4, 5, 6]], order='C')  # 行优先
f_array = np.array([[1, 2, 3], [4, 5, 6]], order='F')  # 列优先

print("C顺序数组:")
print(c_array)
print("内存布局:", c_array.flags['C_CONTIGUOUS'])

print("\nF顺序数组:")
print(f_array)
print("内存布局:", f_array.flags['F_CONTIGUOUS'])

# 性能测试
import time

large_c = np.ones((1000, 1000), order='C')
large_f = np.ones((1000, 1000), order='F')

# 测试行访问性能
start = time.time()
for i in range(1000):
    row_sum = np.sum(large_c[i, :])
c_time = time.time() - start

start = time.time()
for i in range(1000):
    row_sum = np.sum(large_f[i, :])
f_time = time.time() - start

print(f"\n行访问性能: C顺序 {c_time:.6f}s, F顺序 {f_time:.6f}s")
print("C顺序更适合行访问")
```

### 2. 向量化操作

```python
# 向量化操作
print("\n向量化操作:")

# 创建测试数据
data = np.random.random(1000000)

# Python循环方式（慢）
def python_loop_sqrt(data):
    result = np.zeros_like(data)
    for i in range(len(data)):
        result[i] = data[i] ** 0.5
    return result

# NumPy向量化方式（快）
def numpy_vectorized_sqrt(data):
    return np.sqrt(data)

# 性能对比
start = time.time()
loop_result = python_loop_sqrt(data[:10000])  # 只测试前10000个
loop_time = time.time() - start

start = time.time()
vectorized_result = numpy_vectorized_sqrt(data)
vectorized_time = time.time() - start

print(f"循环方式 {loop_time:.6f}s (10000个元素)")
print(f"向量化方式 {vectorized_time:.6f}s (1000000个元素)")
print(f"向量化加速比: {(loop_time * 100) / vectorized_time:.1f}x")
```

## 常见问题和解决方案

### 1. 广播错误

```python
# 常见问题和解决方案
print("常见问题解决:")

# 问题1：广播错误
print("问题1：广播错误")
try:
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([1, 2])  # 形状不匹配
    result = a + b
except ValueError as e:
    print("广播错误:", e)
    print("解决方案：调整形状")
    b_reshaped = b.reshape(2, 1)
    result = a + b_reshaped
    print("调整后的结果:")
    print(result)
```

### 2. 内存不足

```python
# 问题2：内存不足
print("\n问题2：内存不足")
print("解决方案：")
print("1. 使用生成器处理大数据")
print("2. 使用内存映射文件")
print("3. 分块处理数据")
print("4. 使用更小的数据类型")

# 示例：内存映射
large_size = 10000000
try:
    # 尝试创建大数组
    large_array = np.arange(large_size, dtype=np.float64)
    print(f"成功创建 {large_size} 元素数组")
except MemoryError:
    print("内存不足，使用更小的数据类型")
    large_array = np.arange(large_size, dtype=np.float32)
    print(f"使用float32成功创建")
```

### 3. 数据类型不匹配

```python
# 问题3：数据类型不匹配
print("\n问题3：数据类型不匹配")

int_arr = np.array([1, 2, 3])
float_arr = np.array([1.5, 2.5, 3.5])

print("整数数组:", int_arr)
print("浮点数组:", float_arr)

# 自动类型提升
result = int_arr + float_arr
print("运算结果类型:", result.dtype)
print("运算结果:", result)

# 显式类型转换
manual_result = int_arr.astype(float_arr.dtype) + float_arr
print("显式转换结果:", manual_result.dtype)
print("结果相同:", np.array_equal(result, manual_result))
```

## 最佳实践

### 1. 选择合适的数据类型

```python
# 最佳实践
print("最佳实践:")

# 1. 选择合适的数据类型
print("1. 选择合适的数据类型:")
data_range = (0, 255)
if data_range[1] <= 255:
    dtype = np.uint8
    print(f"数据范围{data_range}，选择{dtype}")
elif data_range[1] <= 65535:
    dtype = np.uint16
    print(f"数据范围{data_range}，选择{dtype}")
else:
    dtype = np.uint32
    print(f"数据范围{data_range}，选择{dtype}")
```

### 2. 使用向量化操作

```python
# 2. 使用向量化操作
print("\n2. 使用向量化操作:")
print("✅ 好：result = np.sqrt(data)")
print("❌ 差：for i in range(len(data)): result[i] = sqrt(data[i])")
```

### 3. 预分配内存

```python
# 3. 预分配内存
print("\n3. 预分配内存:")
print("✅ 好：result = np.zeros(size)")
print("❌ 差：result = [] 然后循环append")
```

### 4. 使用视图而不是副本

```python
# 4. 使用视图而不是副本
print("\n4. 使用视图而不是副本:")
print("✅ 好：view = arr[1:4] (当不需要修改原数组时)")
print("❌ 差：copy = arr[1:4].copy() (浪费内存)")
```

## 总结

### ndarray 的核心优势

1. **统一数据类型**: 所有元素类型相同，内存布局紧凑
2. **多维支持**: 从 0 维标量到 N 维数组的完整支持
3. **高效运算**: 向量化操作，比 Python 循环快很多
4. **丰富的功能**: 数学、统计、线性代数等完整函数库
5. **内存效率**: 连续内存布局，缓存友好

### 关键概念

1. **形状(Shape)**: 数组的维度结构
2. **数据类型(dtype)**: 元素的类型和大小
3. **视图 vs 副本**: 内存管理的重要概念
4. **广播**: 不同形状数组的运算机制
5. **向量化**: 高效的数组操作方式

### 实用技能

1. **数组创建**: 多种创建方法适应不同需求
2. **索引切片**: 灵活的数据访问方式
3. **形状操作**: reshape、转置、连接等
4. **数学运算**: 元素级和数组级运算
5. **文件操作**: 数组的保存和加载

掌握 NumPy 的 ndarray 对象，将为您的科学计算和数据分析工作提供强大的基础！
