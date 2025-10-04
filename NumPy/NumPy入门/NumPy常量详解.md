# NumPy 常量详解：通俗易懂指南

## 什么是 NumPy 常量？

NumPy 常量是一些预定义的、具有特殊数值的常量，类似于数学中的 π、e 等常数。这些常量在科学计算中非常常用，NumPy 为我们提供了高精度的实现。

## NumPy 主要常量分类

### 1. 数学常数

#### 自然常数 e

```python
import numpy as np

# 自然常数 e ≈ 2.71828
print("e的值:", np.e)          # 输出: 2.718281828459045
print("e的精度:", type(np.e))   # 输出: <class 'float'>

# 实际应用：计算指数函数
x = 1
result = np.e ** x  # e^1 = e
print(f"e^{x} = {result}")

# 计算连续复利
principal = 1000  # 本金
rate = 0.05       # 年利率
time = 10         # 年数
compound = principal * np.e ** (rate * time)
print(f"连续复利结果: {compound:.2f}")
```

#### 圆周率 π

```python
# 圆周率 π ≈ 3.14159
print("π的值:", np.pi)         # 输出: 3.141592653589793

# 实际应用：计算圆的面积和周长
radius = 5
area = np.pi * radius ** 2     # 面积 = πr²
circumference = 2 * np.pi * radius  # 周长 = 2πr
print(f"圆的面积: {area:.2f}")
print(f"圆的周长: {circumference:.2f}")

# 计算球体体积
volume = (4/3) * np.pi * radius ** 3
print(f"球体体积: {volume:.2f}")
```

#### 欧拉-马歇罗尼常数

```python
# 欧拉-马歇罗尼常数 γ ≈ 0.57721
print("欧拉常数:", np.euler_gamma)  # 输出: 0.5772156649015329

# 这个常数在数论和特殊函数中出现
# 主要用于高级数学计算
```

### 2. 无穷大常数

#### 正无穷大

```python
# 正无穷大
pos_inf = np.inf
print("正无穷大:", pos_inf)     # 输出: inf
print("类型:", type(pos_inf))   # 输出: <class 'float'>

# 实际应用：
# 1. 表示除以零的结果（正数除以零）
result = 1.0 / 0.0  # 结果会是 inf
print("1.0 / 0.0 =", result)

# 2. 在数值优化中表示无界的解
# 3. 在比较中作为最大值
values = [1, 5, 3, 8, 2]
max_value = max(values) if max(values) < np.inf else np.inf
print("有限的最大值:", max_value)

# 4. 数学运算
print("∞ + 100 =", np.inf + 100)     # inf
print("∞ × 5 =", np.inf * 5)         # inf
print("∞ / ∞ =", np.inf / np.inf)    # nan (不确定)
```

#### 负无穷大

```python
# 负无穷大
neg_inf = np.NINF
print("负无穷大:", neg_inf)     # 输出: -inf

# 实际应用：
# 1. 表示负数除以零的结果
result = -1.0 / 0.0
print("-1.0 / 0.0 =", result)

# 2. 在数值优化中表示下界
# 3. 在比较中作为最小值
print("-∞ + 100 =", np.NINF + 100)  # -inf
print("-∞ × -1 =", np.NINF * -1)    # inf
```

### 3. 非数字常数

#### NaN (Not a Number)

```python
# 非数字
nan_value = np.nan
print("NaN:", nan_value)        # 输出: nan

# NaN的产生情况：
# 1. 0除以0
print("0 / 0 =", 0.0 / 0.0)     # nan

# 2. 无穷大减无穷大
print("∞ - ∞ =", np.inf - np.inf)  # nan

# 3. 负数的平方根（在实数范围内）
print("√(-1) =", np.sqrt(-1))     # nan + 警告

# NaN的特性：
print("NaN == NaN:", np.nan == np.nan)  # False（NaN不等于任何值，包括自己）
print("NaN + 100:", np.nan + 100)       # NaN（任何涉及NaN的运算结果都是NaN）

# 检测NaN值
arr = np.array([1, 2, np.nan, 4, 5])
mask = np.isnan(arr)
print("NaN位置:", mask)         # [False False  True False False]
print("非NaN值:", arr[~mask])   # [1. 2. 4. 5.]
```

### 4. 零的变体

#### 正零和负零

```python
# 正零
pos_zero = np.PZERO
print("正零:", pos_zero)        # 输出: 0.0

# 负零
neg_zero = np.NZERO
print("负零:", neg_zero)        # 输出: -0.0

# 在大多数情况下，“正零和负零行为相同”
print("正零 == 负零:", pos_zero == neg_zero)  # True

# 但在某些数学运算中有区别
print("1 / 正零 =", 1.0 / pos_zero)    # inf
print("1 / 负零 =", 1.0 / neg_zero)    # -inf

# 实际应用场景：
# - 在浮点运算中追踪符号
# - 某些科学计算中需要区分零的符号
```

### 5. 维度扩展常量

#### newaxis

```python
# newaxis 用于增加数组的维度
arr = np.array([1, 2, 3, 4])
print("原数组形状:", arr.shape)  # (4,)

# 使用 newaxis 增加维度
row_vector = arr[np.newaxis, :]  # 增加行维度
print("行向量形状:", row_vector.shape)  # (1, 4)

col_vector = arr[:, np.newaxis]  # 增加列维度
print("列向量形状:", col_vector.shape)  # (4, 1)

# 实际应用：广播机制
# 创建一个4x3的矩阵，每行都是arr
matrix = arr * np.ones((3, 1)).T
print("广播结果形状:", matrix.shape)  # (3, 4)

# 使用 newaxis 更简洁
matrix_newaxis = arr[np.newaxis, :] * np.ones((3, 1))
print("使用newaxis的结果形状:", matrix_newaxis.shape)  # (3, 4)
```

## 常用检测函数

### 无穷大检测

```python
# 创建包含特殊值的数组
special_arr = np.array([1, 2, np.inf, -np.inf, np.nan, 0])

# 检测无穷大
inf_mask = np.isinf(special_arr)
print("哪些是无穷大:", inf_mask)  # [False False  True  True False False]

# 检测正无穷大
pos_inf_mask = np.isposinf(special_arr)
print("哪些是正无穷大:", pos_inf_mask)  # [False False  True False False False]

# 检测负无穷大
neg_inf_mask = np.isneginf(special_arr)
print("哪些是负无穷大:", neg_inf_mask)  # [False False False  True False False]
```

### 有限值检测

```python
# 检测有限值（非无穷大且非NaN）
finite_mask = np.isfinite(special_arr)
print("哪些是有限值:", finite_mask)  # [ True  True False False False  True]

# 过滤出有限值
finite_values = special_arr[finite_mask]
print("有限值:", finite_values)  # [1. 2. 0.]
```

### NaN 检测

```python
# 检测NaN
nan_mask = np.isnan(special_arr)
print("哪些是NaN:", nan_mask)  # [False False False False  True False]

# 过滤掉NaN值
valid_values = special_arr[~nan_mask]
print("有效值（非NaN）:", valid_values)  # [  1.   2.  inf -inf   0.]
```

## 实际应用示例

### 1. 数据清洗

```python
# 模拟一个包含缺失值和异常值的数据集
data = np.array([1.2, 3.4, np.nan, 5.6, np.inf, 7.8, -np.inf, 2.3])

# 清洗数据：去除NaN和无穷大值
clean_data = data[np.isfinite(data)]
print("清洗后的数据:", clean_data)  # [1.2 3.4 5.6 7.8 2.3]

# 替换NaN为均值
data_mean = np.nanmean(data)  # 计算时忽略NaN
data_filled = np.where(np.isnan(data), data_mean, data)
print("NaN填充后的数据:", data_filled)
```

### 2. 数学函数计算

```python
# 使用NumPy常量进行数学计算
x = np.array([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])

# 计算三角函数
sin_values = np.sin(x)
cos_values = np.cos(x)

print("x值:", x)
print("sin(x):", sin_values)
print("cos(x):", cos_values)

# 指数和对数函数
exp_values = np.exp([0, 1, 2])  # [e^0, e^1, e^2]
print("指数函数:", exp_values)
```

### 3. 概率统计

```python
# 正态分布概率密度函数
def normal_pdf(x, mu=0, sigma=1):
    """正态分布概率密度函数"""
    return (1 / (sigma * np.sqrt(2 * np.pi))) * \
           np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# 计算标准正态分布在某些点的概率密度
points = np.array([-2, -1, 0, 1, 2])
pdf_values = normal_pdf(points)
print("标准正态分布PDF:", pdf_values)
```

### 4. 数值极限处理

```python
# 处理数值计算的极限情况
def safe_divide(a, b):
    """安全除法，处理除零情况"""
    result = np.zeros_like(a, dtype=float)

    # 正常除法
    normal_mask = (b != 0)
    result[normal_mask] = a[normal_mask] / b[normal_mask]

    # 除以正无穷大的情况
    pos_inf_mask = (b == np.inf)
    result[pos_inf_mask] = 0

    # 除以负无穷大的情况
    neg_inf_mask = (b == -np.inf)
    result[neg_inf_mask] = 0

    # 除以零的情况
    zero_mask = (b == 0)
    result[zero_mask] = np.inf

    return result

# 测试安全除法
a = np.array([1, 2, 3, 4])
b = np.array([2, 0, np.inf, -np.inf])
result = safe_divide(a, b)
print("安全除法结果:", result)
```

## 最佳实践

### 1. 选择合适的常量

```python
# 对于数学计算，优先使用NumPy常量而不是硬编码
# 好的做法：
area = np.pi * radius ** 2
exp_value = np.exp(x)

# 而不是：
# area = 3.14159 * radius ** 2  # 精度较低
# exp_value = 2.71828 ** x       # 精度较低
```

### 2. 异常值处理

```python
# 在处理数据时，总是检查特殊值
def analyze_data(data):
    """分析数据，报告特殊值情况"""
    nan_count = np.sum(np.isnan(data))
    inf_count = np.sum(np.isinf(data))
    finite_count = np.sum(np.isfinite(data))

    print(f"数据点总数: {len(data)}")
    print(f"NaN数量: {nan_count}")
    print(f"无穷大数量: {inf_count}")
    print(f"有限值数量: {finite_count}")

    return {
        'total': len(data),
        'nan': nan_count,
        'inf': inf_count,
        'finite': finite_count
    }

# 测试数据分析
test_data = np.array([1, 2, np.nan, 4, np.inf, 6])
stats = analyze_data(test_data)
```

### 3. 性能优化

```python
# 使用NumPy常量通常比Python内置常量更快
import time

# 比较性能
size = 1000000
arr = np.random.random(size)

# 使用NumPi常量
start = time.time()
result1 = arr * np.pi
numpy_time = time.time() - start

# 使用Python常量
start = time.time()
result2 = arr * 3.141592653589793
python_time = time.time() - start

print(f"NumPy常量时间: {numpy_time:.6f}秒")
print(f"Python常量时间: {python_time:.6f}秒")
```

## 常见陷阱和注意事项

### 1. NaN 的比较

```python
# ❌ 错误：NaN不等于任何值，包括自己
if np.nan == np.nan:
    print("这永远不会执行")

# ✅ 正确：使用np.isnan()检测
if np.isnan(np.nan):
    print("这是NaN值")
```

### 2. 无穷大的运算

```python
# 注意无穷大的特殊运算规则
print("∞ × 0 =", np.inf * 0)     # nan
print("∞ / ∞ =", np.inf / np.inf) # nan
print("∞ - ∞ =", np.inf - np.inf) # nan
```

### 3. 数值精度

```python
# 浮点数精度限制
very_small = 1e-308
print("极小值:", very_small)
print("极小值 × 1e308:", very_small * 1e308)  # 可能会溢出
```

## 总结

NumPy 常量是科学计算中的重要工具：

1. **数学常数**: `np.e`, `np.pi`, `np.euler_gamma` 用于精确的数学计算
2. **无穷大**: `np.inf`, `np.NINF` 表示无界值
3. **非数字**: `np.nan` 表示无效或未定义的结果
4. **零的变体**: `np.PZERO`, `np.NZERO` 用于特殊浮点运算
5. **维度工具**: `np.newaxis` 用于数组操作

通过合理使用这些常量和相应的检测函数，我们可以：

- 提高数值计算的精度
- 正确处理异常情况
- 写出更健壮的科学计算代码
- 提高代码的可读性和维护性

掌握 NumPy 常量是进行高效科学计算的重要基础！
