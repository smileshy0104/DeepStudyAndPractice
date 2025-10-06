# NumPy 索引完全指南：从入门到精通

## 什么是 NumPy 索引？

想象一下，NumPy 数组就像一个有很多抽屉的大柜子，而索引就是找到特定抽屉的"地址"。NumPy 索引系统提供了多种方式来访问数组中的元素，从简单的单个元素访问到复杂的多维数据操作。

### 简单理解索引

```python
import numpy as np

# 创建一个简单的数组
arr = np.array([10, 20, 30, 40, 50])

# 索引就像书页码
print("第1页:", arr[0])    # 10 (第一个元素)
print("第3页:", arr[2])    # 30 (第三个元素)
print("最后一页:", arr[-1]) # 50 (最后一个元素)
```

## 基础索引 (Basic Indexing)

### 1. 一维数组索引

```python
# 一维数组索引详解
print("=== 一维数组索引 ===")

# 创建测试数组
arr_1d = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
print("原数组:", arr_1d)

# 正向索引（从0开始）
print("\n正向索引:")
print("arr[0] =", arr_1d[0])   # 10
print("arr[3] =", arr_1d[3])   # 40
print("arr[9] =", arr_1d[9])   # 100

# 负向索引（从-1开始）
print("\n负向索引:")
print("arr[-1] =", arr_1d[-1])  # 100 (最后一个)
print("arr[-3] =", arr_1d[-3])  # 80
print("arr[-10] =", arr_1d[-10]) # 10 (第一个)

# 索引范围演示
print("\n索引范围可视化:")
for i in range(len(arr_1d)):
    print(f"位置 {i:2d} (正向: {i}, 负向: {i-len(arr_1d):3d}): {arr_1d[i]}")
```

### 2. 二维数组索引

```python
# 二维数组索引详解
print("\n=== 二维数组索引 ===")

# 创建测试矩阵
matrix = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])
print("原矩阵:")
print(matrix)

# 单个元素索引 [行, 列]
print("\n单个元素索引:")
print("matrix[0, 0] =", matrix[0, 0])  # 第1行第1列: 1
print("matrix[1, 2] =", matrix[1, 2])  # 第2行第3列: 7
print("matrix[3, 3] =", matrix[3, 3])  # 第4行第4列: 16

# 使用负索引
print("\n负向索引:")
print("matrix[-1, -1] =", matrix[-1, -1])  # 最后一个元素: 16
print("matrix[-2, 1] =", matrix[-2, 1])    # 倒数第2行，第2列: 10

# 整行访问
print("\n整行访问:")
print("matrix[1] =", matrix[1])          # 第2行: [5 6 7 8]
print("matrix[1, :] =", matrix[1, :])    # 等同写法: [5 6 7 8]

# 整列访问
print("\n整列访问:")
print("matrix[:, 2] =", matrix[:, 2])    # 第3列: [ 3  7 11 15]
```

### 3. 三维及更高维数组索引

```python
# 三维数组索引
print("\n=== 三维数组索引 ===")

# 创建三维数组 (2层, 3行, 4列)
tensor_3d = np.array([
    [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ],
    [
        [13, 14, 15, 16],
        [17, 18, 19, 20],
        [21, 22, 23, 24]
    ]
])
print("三维数组形状:", tensor_3d.shape)
print("三维数组:")
print(tensor_3d)

# 三维索引 [深度, 行, 列]
print("\n三维索引:")
print("tensor_3d[0, 1, 2] =", tensor_3d[0, 1, 2])  # 第1层第2行第3列: 7
print("tensor_3d[1, 2, 3] =", tensor_3d[1, 2, 3])  # 第2层第3行第4列: 24

# 访问整个二维切片
print("\n访问二维切片:")
print("tensor_3d[0] (第1层):")
print(tensor_3d[0])

print("\ntensor_3d[:, 1, :] (所有层的第2行):")
print(tensor_3d[:, 1, :])
```

## 切片 (Slicing)

### 1. 基本切片语法

```python
# 切片基础：start:stop:step
print("=== 基本切片语法 ===")

arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print("原数组:", arr)

# 基本切片格式: [start:stop:step]
print("\n切片示例:")

# 获取子数组
print("arr[2:7] =", arr[2:7])    # 索引2到6: [2 3 4 5 6]
print("arr[:5] =", arr[:5])      # 开始到索引4: [0 1 2 3 4]
print("arr[3:] =", arr[3:])      # 索引3到结束: [3 4 5 6 7 8 9 10]
print("arr[::2] =", arr[::2])    # 每隔一个元素: [0 2 4 6 8 10]

# 负步长（反向）
print("arr[::-1] =", arr[::-1])  # 反向数组: [10  9  8  7  6  5  4  3  2  1  0]
print("arr[8:3:-1] =", arr[8:3:-1])  # 从索引8反向到索引4: [8 7 6 5 4]

# 步长切片
print("arr[1:9:3] =", arr[1:9:3])  # 索引1到8，步长3: [1 4 7]
print("arr[-5:-1] =", arr[-5:-1])  # 倒数第5个到倒数第2个: [6 7 8 9]
```

### 2. 二维数组切片

```python
# 二维数组切片
print("\n=== 二维数组切片 ===")

matrix = np.array([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20]
])
print("原矩阵:")
print(matrix)

# 行切片
print("\n行切片:")
print("matrix[1:3] =")
print(matrix[1:3])  # 第2-3行
print("matrix[::2] =")
print(matrix[::2])  # 每隔一行

# 列切片
print("\n列切片:")
print("matrix[:, 1:4] =")
print(matrix[:, 1:4])  # 所有行的第2-4列
print("matrix[:, ::2] =")
print(matrix[:, ::2])  # 所有行的每隔一列

# 行列同时切片
print("\n行列同时切片:")
print("matrix[1:3, 1:4] =")
print(matrix[1:3, 1:4])  # 第2-3行，第2-4列

print("matrix[::2, ::2] =")
print(matrix[::2, ::2])  # 每隔一行和一列

# 复杂切片示例
print("\n复杂切片示例:")
print("matrix[::-1, ::-1] =")  # 反向矩阵
print(matrix[::-1, ::-1])
```

### 3. 切片的内存特性

```python
# 切片的内存特性
print("\n=== 切片的内存特性 ===")

original = np.array([1, 2, 3, 4, 5])
print("原数组:", original)

# 切片创建视图（共享内存）
slice_view = original[1:4]
print("切片:", slice_view)

# 修改切片会影响原数组
slice_view[0] = 99
print("修改切片后 - 原数组:", original)  # [ 1 99  3  4  5]
print("修改切片后 - 切片:", slice_view)   # [99  3  4]

# 创建独立的副本
slice_copy = original[1:4].copy()
slice_copy[1] = 88
print("\n修改副本后 - 原数组:", original)  # 不变
print("修改副本后 - 副本:", slice_copy)   # [99 88  4]

# 检查是否为视图
print("\n内存检查:")
print("slice_view.base是original吗?", slice_view.base is original)  # True
print("slice_copy.base是original吗?", slice_copy.base is original)  # False
```

## 高级索引 (Advanced Indexing)

### 1. 整数数组索引 (Integer Array Indexing)

```python
# 整数数组索引（花式索引）
print("=== 整数数组索引 ===")

arr = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
print("原数组:", arr)

# 使用整数数组选择特定位置
indices = [0, 3, 7]
selected = arr[indices]
print(f"选择位置 {indices}:", selected)  # [10 40 80]

# 重复索引
indices = [1, 2, 2, 3]
selected = arr[indices]
print(f"重复索引 {indices}:", selected)  # [20 30 30 40]

# 二维数组的花式索引
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
print("\n原矩阵:")
print(matrix)

# 选择特定行
row_indices = [0, 2]
selected_rows = matrix[row_indices]
print(f"选择行 {row_indices}:")
print(selected_rows)

# 选择特定元素
row_indices = [0, 1, 2]
col_indices = [2, 0, 1]
selected_elements = matrix[row_indices, col_indices]
print(f"选择元素 [(0,2), (1,0), (2,1)]:", selected_elements)  # [3 4 8]

# 使用不同的索引数组
rows = np.array([[0, 0], [2, 2]])
cols = np.array([[0, 2], [0, 2]])
selected_2d = matrix[rows, cols]
print("使用二维索引数组:")
print(selected_2d)
# [[1 3]
#  [7 9]]
```

### 2. 布尔索引 (Boolean Indexing)

```python
# 布尔索引
print("\n=== 布尔索引 ===")

data = np.array([1, 5, 3, 8, 2, 7, 4, 6, 9, 0])
print("原数据:", data)

# 创建布尔条件
condition = data > 5
print("条件 (data > 5):", condition)
print("满足条件的元素:", data[condition])

# 多个条件
print("\n多个条件:")
condition1 = data > 3
condition2 = data < 8
combined = condition1 & condition2  # 逻辑与
print("3 < data < 8:", data[combined])

# 或条件
condition3 = (data < 2) | (data > 8)
print("data < 2 或 data > 8:", data[condition3])

# 非条件
condition4 = ~(data > 5)
print("非 (data > 5):", data[condition4])

# 布尔索引的实际应用
print("\n实际应用:")
scores = np.array([85, 92, 78, 65, 88, 95, 72, 80, 90, 68])

# 找出优秀成绩（>= 90）
excellent = scores[scores >= 90]
print("优秀成绩:", excellent)

# 找出及格但不够优秀的成绩（60-89）
passing = scores[(scores >= 60) & (scores < 90)]
print("及格成绩:", passing)

# 统计各等级
excellent_count = np.sum(scores >= 90)
passing_count = np.sum((scores >= 60) & (scores < 90))
failing_count = np.sum(scores < 60)

print(f"优秀人数: {excellent_count}, 及格人数: {passing_count}, 不及格人数: {failing_count}")
```

### 3. where 函数

```python
# np.where 函数
print("\n=== np.where 函数 ===")

arr = np.array([1, 5, 3, 8, 2, 7, 4, 6, 9, 0])
print("原数组:", arr)

# 找到满足条件的元素索引
indices = np.where(arr > 5)
print("arr > 5 的索引:", indices)  # (array([3, 5, 7, 8]),)
print("对应元素:", arr[indices])

# 条件替换
result = np.where(arr > 5, "大", "小")
print("大于5的标记为'大'，否则为'小':", result)

# 多条件替换
result = np.where(arr % 2 == 0, "偶数",
          np.where(arr > 5, "奇数-大", "奇数-小"))
print("复杂条件替换:", result)

# 二维数组的 where
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
print("\n原矩阵:")
print(matrix)

# 找到大于5的元素位置
row_indices, col_indices = np.where(matrix > 5)
print("大于5的元素位置:")
for r, c in zip(row_indices, col_indices):
    print(f"位置({r},{c}) = {matrix[r, c]}")
```

## 特殊索引技巧

### 1. np.ix_ 函数

```python
# np.ix_ 构建索引网格
print("\n=== np.ix_ 函数 ===")

matrix = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])
print("原矩阵:")
print(matrix)

# 选择特定的行和列的交集
rows = [0, 2]  # 选择第1行和第3行
cols = [1, 3]  # 选择第2列和第4列

# 使用 np.ix_ 选择子矩阵
selected = matrix[np.ix_(rows, cols)]
print(f"选择行{rows}和列{cols}的交集:")
print(selected)
# [[ 2  4]
#  [10 12]]

# 对比普通索引
print("\n普通索引 matrix[rows, cols]:")
print(matrix[rows, cols])  # 这会选择 (0,1) 和 (2,3) 位置的元素
# [ 2 12]
```

### 2. take 函数

```python
# np.take 函数
print("\n=== np.take 函数 ===")

arr = np.array([10, 20, 30, 40, 50])
indices = [0, 2, 4]

# 沿指定轴取元素
taken = np.take(arr, indices)
print("原数组:", arr)
print(f"取索引 {indices}:", taken)

# 二维数组的 take
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

# 按行取元素
row_elements = np.take(matrix, [0, 2], axis=1)  # 每行的第1和第3个元素
print("\n按行取第1和第3列:")
print(row_elements)

# 按列取元素
col_elements = np.take(matrix, [0, 2], axis=0)  # 第1和第3行的所有元素
print("\n取第1和第3行:")
print(col_elements)
```

### 3. choose 函数

```python
# np.choose 函数
print("\n=== np.choose 函数 ===")

choices = [
    [0, 1, 2, 3],    # 选择索引0的数组
    [10, 11, 12, 13], # 选择索引1的数组
    [20, 21, 22, 23]  # 选择索引2的数组
]

# 根据选择数组从choices中选择
choice_array = [0, 1, 2, 0]
result = np.choose(choice_array, choices)
print("选择数组:", choice_array)
print("选择结果:", result)  # [ 0 11 22  3]

# 实际应用：成绩等级转换
scores = np.array([85, 92, 78, 65, 88])
grade_levels = (scores >= 90).astype(int) + (scores >= 80).astype(int)
# 0: <60, 1: 60-79, 2: 80-89, 3: >=90

grade_names = ['不及格', '及格', '良好', '优秀']
# 由于np.choose的限制，我们需要调整方法
final_grades = np.where(scores >= 90, '优秀',
               np.where(scores >= 80, '良好',
               np.where(scores >= 60, '及格', '不及格')))

print("\n成绩等级转换:")
for score, grade in zip(scores, final_grades):
    print(f"{score}分: {grade}")
```

## 索引性能优化

### 1. 视图 vs 副本

```python
# 索引性能：视图 vs 副本
print("\n=== 索引性能优化 ===")

large_array = np.arange(1000000)

# 基础索引创建视图（快速）
import time

start_time = time.time()
view_result = large_array[100:200]  # 视图
view_time = time.time() - start_time

# 花式索引创建副本（较慢）
start_time = time.time()
fancy_result = large_array[[100, 101, 102, 103, 104]]  # 副本
fancy_time = time.time() - start_time

print(f"视图创建时间: {view_time:.8f}秒")
print(f"花式索引时间: {fancy_time:.8f}秒")

# 内存使用
print(f"\n内存使用:")
print(f"视图内存: {view_result.nbytes} 字节 (共享原数组内存)")
print(f"花式索引内存: {fancy_result.nbytes} 字节 (独立内存)")
```

### 2. 布尔索引优化

```python
# 布尔索引优化
print("\n=== 布尔索引优化 ===")

data = np.random.random(1000000)
threshold = 0.5

# 方法1：直接布尔索引
start_time = time.time()
result1 = data[data > threshold]
time1 = time.time() - start_time

# 方法2：使用np.where
start_time = time.time()
indices = np.where(data > threshold)[0]
result2 = data[indices]
time2 = time.time() - start_time

# 方法3：使用np.nonzero（与where类似）
start_time = time.time()
indices = np.nonzero(data > threshold)[0]
result3 = data[indices]
time3 = time.time() - start_time

print(f"直接布尔索引时间: {time1:.6f}秒")
print(f"np.where时间: {time2:.6f}秒")
print(f"np.nonzero时间: {time3:.6f}秒")
print("结果相同:", np.array_equal(result1, result2) and np.array_equal(result2, result3))
```

## 实际应用案例

### 1. 数据筛选和分析

```python
# 实际应用：数据分析
print("\n=== 实际应用：数据分析 ===")

# 模拟学生成绩数据
np.random.seed(42)
n_students = 100
scores = np.random.normal(75, 12, n_students)  # 均值75，标准差12
scores = np.clip(scores, 0, 100)  # 限制在0-100范围内

print("成绩统计:")
print(f"平均分: {np.mean(scores):.2f}")
print(f"标准差: {np.std(scores):.2f}")
print(f"最高分: {np.max(scores):.2f}")
print(f"最低分: {np.min(scores):.2f}")

# 成绩分级
excellent = scores[scores >= 90]    # 优秀
good = scores[(scores >= 80) & (scores < 90)]    # 良好
passing = scores[(scores >= 60) & (scores < 80)]  # 及格
failing = scores[scores < 60]      # 不及格

print(f"\n成绩分布:")
print(f"优秀(>=90): {len(excellent)}人 ({len(excellent)/len(scores)*100:.1f}%)")
print(f"良好(80-89): {len(good)}人 ({len(good)/len(scores)*100:.1f}%)")
print(f"及格(60-79): {len(passing)}人 ({len(passing)/len(scores)*100:.1f}%)")
print(f"不及格(<60): {len(failing)}人 ({len(failing)/len(scores)*100:.1f}%)")

# 找出需要特别关注的学生（不及格或优秀）
need_attention = np.where(scores < 60)[0]  # 不及格学生索引
top_students = np.where(scores >= 90)[0]   # 优秀学生索引

print(f"\n需要帮助的学生索引: {need_attention[:10]}{'...' if len(need_attention) > 10 else ''}")
print(f"优秀学生索引: {top_students[:10]}{'...' if len(top_students) > 10 else ''}")
```

### 2. 图像处理索引应用

```python
# 图像处理中的索引应用
print("\n=== 图像处理索引应用 ===")

# 模拟RGB图像数据 (5x5像素，3通道)
image = np.random.randint(0, 256, (5, 5, 3), dtype=np.uint8)
print("图像形状:", image.shape)

# 提取红色通道
red_channel = image[:, :, 0]
print("红色通道形状:", red_channel.shape)

# 找出亮像素（所有通道都 > 200）
bright_pixels = np.where(np.all(image > 200, axis=2))
print("亮像素位置:", list(zip(bright_pixels[0], bright_pixels[1])))

# 创建遮罩：选择中心区域
mask = np.zeros((5, 5), dtype=bool)
mask[1:4, 1:4] = True  # 中心3x3区域

# 应用遮罩
center_region = image[mask]  # 这会返回一维数组
print("中心区域像素形状:", center_region.shape)

# 修改特定区域
image[mask] = [255, 255, 255]  # 将中心区域设为白色
print("修改后的图像统计:")
print(f"平均像素值: {np.mean(image):.2f}")
```

### 3. 时间序列数据处理

```python
# 时间序列数据索引
print("\n=== 时间序列数据处理 ===")

# 创建模拟时间序列数据
dates = np.arange('2024-01-01', '2024-01-31', dtype='datetime64[D]')
prices = 100 + np.cumsum(np.random.randn(30) * 0.02)  # 随机价格走势

print("日期范围:", dates[0], "到", dates[-1])
print("价格范围:", np.min(prices).round(2), "到", np.max(prices).round(2))

# 找出价格最高的日期
max_price_idx = np.argmax(prices)
max_price_date = dates[max_price_idx]
max_price = prices[max_price_idx]

print(f"最高价格: {max_price:.2f} 在 {max_price_date}")

# 找出价格最低的日期
min_price_idx = np.argmin(prices)
min_price_date = dates[min_price_idx]
min_price = prices[min_price_idx]

print(f"最低价格: {min_price:.2f} 在 {min_price_date}")

# 筛选价格上涨的日期
price_increases = np.where(np.diff(prices) > 0)[0]
print(f"价格上涨天数: {len(price_increases)}")

# 筛选特定时间段的数据
# 选择前两周的数据
first_two_weeks = prices[:14]
print(f"前两周平均价格: {np.mean(first_two_weeks):.2f}")

# 选择每周的数据（每7天取一个点）
weekly_data = prices[::7]
print("每周数据点:", weekly_data.round(2))
```

## 常见错误和解决方案

### 1. 索引越界

```python
# 常见错误1：索引越界
print("\n=== 常见错误和解决方案 ===")

arr = np.array([1, 2, 3, 4, 5])

print("原数组:", arr)

# 错误示例
try:
    invalid_index = arr[10]  # 索引超出范围
except IndexError as e:
    print("索引越界错误:", str(e))

# 安全的索引访问
def safe_get(arr, index, default=None):
    """安全的数组索引访问"""
    try:
        return arr[index]
    except IndexError:
        return default

print("安全访问arr[10]:", safe_get(arr, 10, "索引超出范围"))

# 处理负索引
print("安全访问arr[-10]:", safe_get(arr, -10, "索引超出范围"))
```

### 2. 维度不匹配

```python
# 常见错误2：维度不匹配
print("\n维度不匹配错误:")

matrix = np.array([[1, 2, 3], [4, 5, 6]])
print("矩阵形状:", matrix.shape)

# 错误示例
try:
    wrong_index = matrix[0, 3]  # 列索引超出范围
except IndexError as e:
    print("维度不匹配错误:", str(e))

# 正确的索引方式
print("正确索引 matrix[0, 2]:", matrix[0, 2])

# 安全的多维索引
def safe_multi_get(arr, *indices):
    """安全的多维索引访问"""
    try:
        return arr[indices]
    except IndexError:
        return f"索引 {indices} 超出范围"

print("安全访问 matrix[1, 5]:", safe_multi_get(matrix, 1, 5))
```

### 3. 布尔索引形状不匹配

```python
# 常见错误3：布尔索引形状不匹配
print("\n布尔索引形状错误:")

arr = np.array([1, 2, 3, 4, 5])
wrong_mask = np.array([True, False, True])  # 长度不匹配

# 错误示例
try:
    result = arr[wrong_mask]
except IndexError as e:
    print("布尔索引形状错误:", str(e))

# 正确的布尔索引
correct_mask = np.array([True, False, True, False, True])
print("正确布尔索引结果:", arr[correct_mask])

# 自动处理形状不匹配
def safe_boolean_index(arr, mask):
    """安全的布尔索引"""
    if len(mask) != len(arr):
        print(f"警告：掩码长度({len(mask)})与数组长度({len(arr)})不匹配")
        return arr
    return arr[mask]

print("安全布尔索引:", safe_boolean_index(arr, wrong_mask))
```

## 最佳实践总结

### 1. 索引选择指南

```python
# 索引选择指南
print("\n=== 最佳实践总结 ===")

print("1. 索引选择指南:")
print("   - 基础索引：访问单个或连续的元素，创建视图（快速）")
print("   - 切片：访问子数组，创建视图（快速）")
print("   - 布尔索引：基于条件筛选，创建副本（较慢但灵活）")
print("   - 花式索引：访问不连续的元素，创建副本（较慢）")

print("\n2. 性能建议:")
print("   - 优先使用基础索引和切片")
print("   - 避免在循环中使用复杂索引")
print("   - 重复使用索引时，考虑预先计算")
print("   - 大数组操作时注意内存使用")

print("\n3. 安全建议:")
print("   - 检查索引范围")
print("   - 使用try-except处理索引错误")
print("   - 验证布尔掩码的形状")
print("   - 考虑使用np.where进行条件操作")

print("\n4. 可读性建议:")
print("   - 使用描述性的变量名")
print("   - 复杂索引添加注释")
print("   - 将复杂索引分解为多个步骤")
print("   - 使用中间变量存储索引")
```

### 2. 性能对比模板

```python
# 性能对比模板
def benchmark_indexing_methods(arr_size=1000000):
    """索引方法性能对比"""
    import time

    # 创建测试数组
    arr = np.random.random(arr_size)

    # 测试不同索引方法
    methods = {
        "基础索引": lambda: arr[1000:2000],
        "步长切片": lambda: arr[::10],
        "布尔索引": lambda: arr[arr > 0.5],
        "花式索引": lambda: arr[np.random.randint(0, arr_size, 1000)],
        "where索引": lambda: arr[np.where(arr > 0.5)]
    }

    print("索引方法性能对比:")
    for name, method in methods.items():
        start_time = time.time()
        result = method()
        elapsed = time.time() - start_time
        print(f"{name:12s}: {elapsed:.6f}秒, 结果长度: {len(result)}")

# 运行性能测试
print("\n性能对比:")
benchmark_indexing_methods(1000000)
```

## 总结

NumPy 索引系统提供了强大而灵活的数据访问方式：

### 核心概念

1. **基础索引**：访问单个元素，速度快，创建视图
2. **切片**：访问连续子数组，速度快，创建视图
3. **花式索引**：访问不连续元素，灵活性高，创建副本
4. **布尔索引**：基于条件筛选，功能强大，创建副本

### 关键技巧

- 使用负索引从末尾访问元素
- 利用步长创建复杂的切片模式
- 结合布尔索引进行数据筛选
- 使用 `np.where` 进行条件操作
- 注意内存和性能影响

### 应用场景

- 数据分析和筛选
- 图像处理
- 时间序列分析
- 科学计算
- 机器学习特征工程

掌握 NumPy 索引是高效数据处理的基础，通过合理选择索引方法，可以编写出既高效又易读的代码！