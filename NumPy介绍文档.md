# NumPy 完整介绍文档

## 目录
1. [NumPy 简介](#numpy-简介)
2. [NumPy 的特点和优势](#numpy-的特点和优势)
3. [安装和导入](#安装和导入)
4. [核心概念：ndarray](#核心概念ndarray)
5. [数组创建方法](#数组创建方法)
6. [数据类型](#数据类型)
7. [数组属性](#数组属性)
8. [数组操作](#数组操作)
9. [索引和切片](#索引和切片)
10. [数组运算](#数组运算)
11. [广播机制](#广播机制)
12. [常用函数](#常用函数)
13. [实际应用场景](#实际应用场景)
14. [最佳实践](#最佳实践)

## NumPy 简介

NumPy（Numerical Python）是 Python 科学计算的基础包，它提供了一个强大的 N 维数组对象（ndarray），以及大量的数学函数库。NumPy 是几乎所有 Python 科学计算库的基础，包括 pandas、scikit-learn、matplotlib 等。

## NumPy 的特点和优势

### 主要特点
- **高性能的多维数组对象**：比 Python 原生列表快 50-100 倍
- **丰富的数学函数库**：线性代数、傅里叶变换、随机数等
- **广播功能**：不同大小数组之间的算术运算
- **内存高效**：使用连续内存存储，减少内存碎片
- **与 C/C++/Fortran 集成**：可以轻松集成底层代码

### 性能优势
- 底层用 C 语言实现，运行速度快
- 向量化操作，避免 Python 循环的开销
- 支持并行计算和 SIMD 指令集

## 安装和导入

### 安装
```bash
pip install numpy
```

### 导入
```python
import numpy as np  # 约定俗成的别名
```

## 核心概念：ndarray

ndarray（N-dimensional array）是 NumPy 的核心数据结构，它是一个同质的多维容器，所有元素必须是相同类型。

### ndarray 的特点
- **同质性**：所有元素类型相同
- **固定大小**：创建后大小不可改变
- **多维**：支持任意维度
- **高效**：连续内存存储

## 数组创建方法

### 1. 从 Python 序列创建

```python
# 一维数组
arr1 = np.array([1, 2, 3, 4])
arr2 = np.array((1, 2, 3, 4))  # 使用元组

# 二维数组
arr3 = np.array([[1, 2], [3, 4]])
arr4 = np.array(((1, 2), (3, 4)))  # 使用元组
```

### 2. 使用内置函数创建

```python
# 等差数列
np.arange(10)           # [0, 1, 2, ..., 9]
np.arange(3, 10, 0.1)   # 从3到10，步长0.1

# 等间距数列
np.linspace(0, 10, 11)  # 从0到10，共11个点

# 特殊值数组
np.zeros(5)             # [0, 0, 0, 0, 0]
np.ones((2, 3))         # 2x3的全1数组
np.full((2, 3), 7)      # 2x3的全7数组
np.empty((2, 3))        # 2x3的未初始化数组

# 相似结构数组
np.zeros_like(arr)      # 与arr相同形状的零数组
np.ones_like(arr)       # 与arr相同形状的1数组

# 随机数组
np.random.randn(3, 4)           # 标准正态分布
np.random.randint(0, 10, (3, 4))  # 随机整数
np.random.random((3, 4))        # [0, 1)区间随机数
```

### 3. 特殊矩阵创建

```python
# 单位矩阵
np.eye(3)           # 3x3单位矩阵
np.identity(3)      # 3x3单位矩阵

# 对角矩阵
np.diag([1, 2, 3])  # 主对角线为[1,2,3]的矩阵
```

## 数据类型

NumPy 支持比 Python 更多的数值类型：

### 整型
- `int8, int16, int32, int64`
- `uint8, uint16, uint32, uint64`

### 浮点型
- `float16, float32, float64, float128`

### 复数型
- `complex64, complex128, complex256`

### 布尔型
- `bool`

### 字符串型
- `string_, unicode_`

### 指定数据类型
```python
# 创建时指定
arr = np.array([1, 2, 3], dtype=np.float32)

# 转换类型
arr.astype(np.int32)
```

## 数组属性

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

arr.shape       # (2, 3) - 形状
arr.ndim        # 2 - 维数
arr.size        # 6 - 元素总数
arr.dtype       # dtype('int64') - 数据类型
arr.itemsize    # 8 - 每个元素字节大小
arr.nbytes      # 48 - 总字节数
```

## 数组操作

### 改变形状
```python
arr = np.arange(12)

# 重塑形状
arr.reshape(3, 4)       # 变成3x4
arr.reshape(-1, 4)      # 自动计算行数

# 展平
arr.flatten()           # 返回副本
arr.ravel()            # 返回视图（如果可能）

# 转置
arr.T                  # 转置
arr.transpose()        # 转置
```

### 分割数组
```python
arr = np.arange(12).reshape(3, 4)

# 水平分割
np.hsplit(arr, 2)      # 分成2部分
np.split(arr, 2, axis=1)  # 沿轴1分割

# 垂直分割
np.vsplit(arr, 3)      # 分成3部分
np.split(arr, 3, axis=0)  # 沿轴0分割
```

### 合并数组
```python
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])

# 水平合并
np.hstack([arr1, arr2])
np.concatenate([arr1, arr2], axis=1)

# 垂直合并
np.vstack([arr1, arr2])
np.concatenate([arr1, arr2], axis=0)
```

## 索引和切片

### 基本索引
```python
arr = np.array([1, 2, 3, 4, 5])

arr[0]          # 第一个元素
arr[-1]         # 最后一个元素
arr[1:4]        # 切片：[2, 3, 4]
arr[::2]        # 步长为2：[1, 3, 5]
```

### 多维数组索引
```python
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

arr[0, 1]       # 第0行第1列：2
arr[1, :]       # 第1行：[4, 5, 6]
arr[:, 2]       # 第2列：[3, 6, 9]
arr[0:2, 1:3]   # 子矩阵：[[2, 3], [5, 6]]
```

### 布尔索引
```python
arr = np.array([1, 2, 3, 4, 5])

# 条件索引
mask = arr > 3
arr[mask]       # [4, 5]
arr[arr > 3]    # [4, 5]

# 多条件
arr[(arr > 2) & (arr < 5)]  # [3, 4]
```

### 花式索引
```python
arr = np.array([10, 20, 30, 40, 50])

# 整数数组索引
indices = [0, 2, 4]
arr[indices]    # [10, 30, 50]

# 二维数组花式索引
arr2d = np.arange(12).reshape(3, 4)
arr2d[[0, 2], [1, 3]]  # 取(0,1)和(2,3)位置的元素
```

## 数组运算

### 算术运算
```python
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

# 元素级运算
arr1 + arr2     # [5, 7, 9]
arr1 - arr2     # [-3, -3, -3]
arr1 * arr2     # [4, 10, 18]
arr1 / arr2     # [0.25, 0.4, 0.5]
arr1 ** 2       # [1, 4, 9]

# 与标量运算
arr1 + 10       # [11, 12, 13]
arr1 * 2        # [2, 4, 6]
```

### 比较运算
```python
arr1 = np.array([1, 2, 3])
arr2 = np.array([3, 2, 1])

arr1 > arr2     # [False, False, True]
arr1 == arr2    # [False, True, False]
np.any(arr1 > 2)  # True
np.all(arr1 > 0)  # True
```

### 逻辑运算
```python
arr = np.array([True, False, True])

np.logical_and(arr, [True, True, False])   # [True, False, False]
np.logical_or(arr, [False, True, False])   # [True, True, True]
np.logical_not(arr)                        # [False, True, False]
```

## 广播机制

广播允许不同形状的数组进行算术运算：

```python
# 标量与数组
arr = np.array([[1, 2, 3], [4, 5, 6]])
arr + 10  # 每个元素都加10

# 不同形状数组
arr1 = np.array([[1, 2, 3], [4, 5, 6]])  # (2, 3)
arr2 = np.array([10, 20, 30])           # (3,)
arr1 + arr2  # arr2被广播为(1, 3)，然后扩展为(2, 3)

# 广播规则：
# 1. 从右边开始比较维度
# 2. 维度大小相等或其中一个为1时可以广播
# 3. 缺失的维度视为1
```

## 常用函数

### 数学函数
```python
arr = np.array([1, 2, 3, 4])

# 三角函数
np.sin(arr)
np.cos(arr)
np.tan(arr)

# 指数和对数
np.exp(arr)      # e^x
np.log(arr)      # ln(x)
np.log10(arr)    # log10(x)

# 取整函数
np.round(arr)    # 四舍五入
np.floor(arr)    # 向下取整
np.ceil(arr)     # 向上取整

# 符号函数
np.abs(arr)      # 绝对值
np.sign(arr)     # 符号
```

### 统计函数
```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

# 基本统计
np.mean(arr)        # 平均值
np.median(arr)      # 中位数
np.std(arr)         # 标准差
np.var(arr)         # 方差

# 最值
np.max(arr)         # 最大值
np.min(arr)         # 最小值
np.argmax(arr)      # 最大值索引
np.argmin(arr)      # 最小值索引

# 求和
np.sum(arr)         # 总和
np.sum(arr, axis=0) # 按列求和
np.sum(arr, axis=1) # 按行求和

# 累积函数
np.cumsum(arr)      # 累积和
np.cumprod(arr)     # 累积积
```

### 线性代数
```python
# 矩阵运算
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# 矩阵乘法
np.dot(A, B)        # 矩阵乘法
A @ B               # Python 3.5+的矩阵乘法语法

# 矩阵属性
np.linalg.det(A)    # 行列式
np.linalg.inv(A)    # 逆矩阵
np.linalg.eig(A)    # 特征值和特征向量
np.trace(A)         # 迹（对角线元素之和）
```

### 随机数生成
```python
# 设置随机种子
np.random.seed(42)

# 各种分布
np.random.normal(0, 1, (3, 3))      # 正态分布
np.random.uniform(0, 1, (3, 3))     # 均匀分布
np.random.binomial(10, 0.5, (3, 3)) # 二项分布
np.random.poisson(3, (3, 3))        # 泊松分布

# 随机采样
arr = np.array([1, 2, 3, 4, 5])
np.random.choice(arr, 3)            # 随机选择3个元素
np.random.shuffle(arr)              # 就地打乱
```

## 实际应用场景

### 1. 数据预处理
```python
# 标准化
data = np.random.randn(100, 5)
normalized = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

# 缺失值处理
data[data < 0] = np.nan  # 设置缺失值
mask = ~np.isnan(data)   # 非缺失值掩码
```

### 2. 图像处理
```python
# 图像表示为3D数组 (高度, 宽度, 通道)
image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

# 灰度化
gray = np.mean(image, axis=2)

# 图像滤波（简单平均滤波）
from scipy import ndimage
filtered = ndimage.uniform_filter(image, size=3)
```

### 3. 信号处理
```python
# 生成信号
t = np.linspace(0, 1, 1000)
signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)

# FFT频谱分析
from numpy.fft import fft, fftfreq
spectrum = fft(signal)
freqs = fftfreq(len(signal), t[1] - t[0])
```

### 4. 机器学习特征工程
```python
# 多项式特征
def polynomial_features(X, degree=2):
    features = [X]
    for d in range(2, degree + 1):
        features.append(X ** d)
    return np.column_stack(features)

# 数据分箱
def bin_data(data, bins=10):
    bin_edges = np.linspace(data.min(), data.max(), bins + 1)
    return np.digitize(data, bin_edges) - 1
```

## 最佳实践

### 1. 性能优化
```python
# 使用向量化操作而不是循环
# 不好的方式
result = []
for i in range(len(arr)):
    result.append(arr[i] ** 2)

# 好的方式
result = arr ** 2

# 预分配数组大小
result = np.empty((1000, 1000))  # 预分配
# 而不是逐步增长数组
```

### 2. 内存管理
```python
# 使用视图而不是副本（当可能时）
arr_view = arr[::2]     # 视图
arr_copy = arr[::2].copy()  # 副本

# 及时释放大数组
del large_array
```

### 3. 数值稳定性
```python
# 避免下溢/上溢
def safe_log(x):
    return np.log(np.maximum(x, 1e-15))

# 使用适当的数据类型
arr = np.array([1, 2, 3], dtype=np.float64)  # 高精度
```

### 4. 代码可读性
```python
# 使用有意义的变量名
heights = np.array([170, 180, 165])  # 而不是 arr
weights = np.array([70, 85, 60])     # 而不是 arr2

# 添加维度检查
def matrix_multiply(A, B):
    assert A.shape[1] == B.shape[0], "维度不匹配"
    return np.dot(A, B)
```

### 5. 常见陷阱避免
```python
# 注意视图 vs 副本
arr = np.array([1, 2, 3, 4, 5])
view = arr[1:4]        # 视图
view[0] = 99           # 会改变原数组
copy = arr[1:4].copy() # 副本
copy[0] = 99           # 不会改变原数组

# 注意广播的意外行为
a = np.array([1, 2, 3]).reshape(-1, 1)  # (3, 1)
b = np.array([4, 5])                    # (2,)
result = a + b  # 结果是(3, 2)，可能不是预期的
```

## 总结

NumPy 是 Python 科学计算生态系统的基石，它提供了：

1. **高性能的多维数组对象**：比原生 Python 列表快得多
2. **丰富的数学函数库**：覆盖线性代数、统计、傅里叶变换等
3. **强大的广播机制**：使得不同形状数组间的运算成为可能
4. **与其他库的无缝集成**：是 pandas、scikit-learn 等库的基础

掌握 NumPy 是进行 Python 数据科学和机器学习的必备技能。通过合理使用 NumPy 的功能，可以写出高效、简洁、可读的科学计算代码。