# Pandas 完全入门指南：数据分析的瑞士军刀

## 什么是 Pandas？

想象一下，Pandas 就像是数据处理领域的"瑞士军刀"。它能让您像处理 Excel 表格一样轻松地处理数据，但功能比 Excel 强大一百倍！无论您是数据分析师、科学家，还是程序员，Pandas 都能让数据处理变得简单高效。

### 简单理解 Pandas

```python
import pandas as pd

# Pandas 让数据处理变得简单
# 就像 Excel 表格，但功能更强大

# 创建一个简单的表格（DataFrame）
data = {
    '姓名': ['张三', '李四', '王五', '赵六'],
    '年龄': [25, 30, 35, 28],
    '城市': ['北京', '上海', '广州', '深圳'],
    '工资': [8000, 12000, 15000, 10000]
}

df = pd.DataFrame(data)
print("我们的第一个数据表:")
print(df)

# 瞬间就能进行各种操作
print("\n平均工资:", df['工资'].mean())
print("年龄最大的员工:", df.loc[df['年龄'].idxmax(), '姓名'])
```

## Pandas 快速入门

### 1. 安装和导入

```python
# Pandas 安装和导入
print("=== Pandas 安装和导入 ===")

# 在终端/命令行中安装 Pandas
# pip install pandas
# 或者
# conda install pandas

# 导入 Pandas（惯例用法）
import pandas as pd
import numpy as np  # Pandas 通常配合 NumPy 使用

print("Pandas 版本:", pd.__version__)
print("导入成功！")

# Pandas 的核心优势
print("\nPandas 的核心优势:")
print("✅ 类似 Excel 的表格操作")
print("✅ 强大的数据清洗功能")
print("✅ 丰富的数据分析工具")
print("✅ 灵活的数据导入导出")
print("✅ 高效的时间序列处理")
print("✅ 完善的数据可视化支持")
```

### 2. Pandas 的三大核心数据结构

```python
# Pandas 三大核心数据结构
print("\n=== Pandas 三大核心数据结构 ===")

# 1. Series - 一维数据，就像带标签的列表
print("\n1. Series (一维数据):")
series_ages = pd.Series([25, 30, 35, 28],
                       index=['张三', '李四', '王五', '赵六'],
                       name='年龄')
print(series_ages)

# 2. DataFrame - 二维数据，就像表格
print("\n2. DataFrame (二维数据):")
df_employees = pd.DataFrame({
    '姓名': ['张三', '李四', '王五', '赵六'],
    '年龄': [25, 30, 35, 28],
    '部门': ['技术', '销售', '技术', '市场']
})
print(df_employees)

# 3. Index - 索引，就像行号或书签
print("\n3. Index (索引):")
index_names = pd.Index(['张三', '李四', '王五', '赵六'], name='员工姓名')
print("索引对象:", index_names)
print("索引类型:", type(index_names))
```

## Pandas 数据结构详解

### 1. Series 深度解析

```python
# Series 深度解析
print("=== Series 深度解析 ===")

# 创建 Series 的多种方式
print("\n创建 Series:")

# 方式1：从列表创建
series1 = pd.Series([10, 20, 30, 40])
print("从列表创建:", series1)

# 方式2：带索引创建
series2 = pd.Series([10, 20, 30, 40],
                    index=['a', 'b', 'c', 'd'],
                    name='数值')
print("带索引创建:")
print(series2)

# 方式3：从字典创建
series3 = pd.Series({'语文': 85, '数学': 92, '英语': 78})
print("从字典创建:")
print(series3)

# 方式4：从 NumPy 数组创建
series4 = pd.Series(np.arange(5), name='数字序列')
print("从 NumPy 数组创建:", series4)

# Series 的基本操作
print("\nSeries 基本操作:")

# 访问数据
print("series2['b']:", series2['b'])  # 通过标签访问
print("series2[1]:", series2[1])      # 通过位置访问

# 切片
print("series2[1:3]:")
print(series2[1:3])  # 位置切片

print("series2['b':'d']:")
print(series2['b':'d'])  # 标签切片（包含结束点）

# 统计信息
print("\nSeries 统计信息:")
scores = pd.Series([85, 92, 78, 95, 88, 76, 90])
print("成绩:", scores.values)
print("平均分:", scores.mean())
print("最高分:", scores.max())
print("最低分:", scores.min())
print("标准差:", scores.std())
print("成绩描述:")
print(scores.describe())

# 条件筛选
print("\n条件筛选:")
high_scores = scores[scores > 85]
print("高于85分的成绩:", high_scores.values)

# Series 的向量化操作
print("\n向量化操作:")
series_a = pd.Series([1, 2, 3, 4])
series_b = pd.Series([10, 20, 30, 40])

print("series_a + 5:", series_a + 5)
print("series_a * series_b:", series_a * series_b)
print("series_a > 2:", series_a > 2)
```

### 2. DataFrame 深度解析

```python
# DataFrame 深度解析
print("\n=== DataFrame 深度解析 ===")

# 创建 DataFrame 的多种方式
print("\n创建 DataFrame:")

# 方式1：从字典创建（列为主）
data1 = {
    '姓名': ['张三', '李四', '王五'],
    '年龄': [25, 30, 35],
    '城市': ['北京', '上海', '广州']
}
df1 = pd.DataFrame(data1)
print("从字典创建:")
print(df1)

# 方式2：从列表创建（行为主）
data2 = [
    ['张三', 25, '北京', '技术'],
    ['李四', 30, '上海', '销售'],
    ['王五', 35, '广州', '技术']
]
df2 = pd.DataFrame(data2,
                   columns=['姓名', '年龄', '城市', '部门'])
print("\n从列表创建:")
print(df2)

# 方式3：从 NumPy 数组创建
data3 = np.random.randint(0, 100, (4, 3))
df3 = pd.DataFrame(data3,
                   columns=['语文', '数学', '英语'],
                   index=['学生A', '学生B', '学生C', '学生D'])
print("\n从 NumPy 数组创建:")
print(df3)

# 方式4：从字典列表创建
data4 = [
    {'姓名': '张三', '年龄': 25, '城市': '北京'},
    {'姓名': '李四', '年龄': 30, '城市': '上海'},
    {'姓名': '王五', '年龄': 35, '城市': '广州'}
]
df4 = pd.DataFrame(data4)
print("\n从字典列表创建:")
print(df4)

# DataFrame 的基本属性
print("\nDataFrame 基本属性:")
df = pd.DataFrame({
    '姓名': ['张三', '李四', '王五', '赵六', '钱七'],
    '年龄': [25, 30, 35, 28, 32],
    '部门': ['技术', '销售', '技术', '市场', '技术'],
    '工资': [8000, 12000, 15000, 10000, 13000]
})

print("数据形状:", df.shape)  # (行数, 列数)
print("数据类型:")
print(df.dtypes)
print("列名:", df.columns.tolist())
print("索引:", df.index.tolist())
print("数据值:")
print(df.values)

# DataFrame 的数据访问
print("\nDataFrame 数据访问:")

# 访问列
print("访问单个列 (df['姓名']):", df['姓名'].tolist())
print("访问多个列 (df[['姓名', '工资']]):")
print(df[['姓名', '工资']])

# 访问行
print("\n访问行:")
print("第0行 (df.iloc[0]):")
print(df.iloc[0])

print("姓名为'张三'的行 (df.loc[df['姓名'] == '张三']):")
print(df.loc[df['姓名'] == '张三'])

# 访问特定单元格
print("\n访问单元格:")
print("第1行第2列 (df.iloc[1, 2]):", df.iloc[1, 2])
print("姓名为'李四'的工资 (df.loc[1, '工资']):", df.loc[1, '工资'])
```

### 3. Index 深度解析

```python
# Index 深度解析
print("\n=== Index 深度解析 ===")

# 创建不同的索引
print("\n创建索引:")

# 简单索引
simple_index = pd.Index(['A', 'B', 'C', 'D'])
print("简单索引:", simple_index)

# 带名称的索引
named_index = pd.Index(['张三', '李四', '王五'], name='姓名')
print("带名称的索引:", named_index)

# 范围索引
range_index = pd.RangeIndex(0, 10, 2)  # 0, 2, 4, 6, 8
print("范围索引:", range_index)

# 日期时间索引
date_index = pd.date_range('2024-01-01', periods=5, freq='D')
print("日期索引:", date_index)

# 索引操作
print("\n索引操作:")
idx = pd.Index(['苹果', '香蕉', '橙子', '葡萄'])

print("索引包含'苹果'吗?", '苹果' in idx)
print("索引长度:", len(idx))
print("索引类型:", type(idx))

# 索引查找
print("查找'香蕉'的位置:", idx.get_loc('香蕉'))
print("查找多个位置:", idx.get_indexer(['香蕉', '葡萄']))

# 索引运算
idx1 = pd.Index(['A', 'B', 'C', 'D'])
idx2 = pd.Index(['C', 'D', 'E', 'F'])

print("\n索引运算:")
print("交集:", idx1.intersection(idx2))
print("并集:", idx1.union(idx2))
print("差集 (idx1 - idx2):", idx1.difference(idx2))
print("对称差集:", idx1.symmetric_difference(idx2))

# 在 DataFrame 中使用自定义索引
print("\n在 DataFrame 中使用自定义索引:")
df_custom = pd.DataFrame({
    '数学': [85, 92, 78, 95],
    '英语': [88, 85, 92, 80],
    '物理': [90, 88, 85, 92]
}, index=['张三', '李四', '王五', '赵六'])

print("自定义索引的 DataFrame:")
print(df_custom)

print("按姓名查找成绩:")
print(df_custom.loc['李四'])
```

## 创建 Pandas 数据

### 1. 创建 Series

```python
# 创建 Series 的完整方法
print("=== 创建 Series 方法大全 ===")

# 从不同数据源创建 Series
print("\n1. 从基础数据类型创建:")

# 从列表
s1 = pd.Series([10, 20, 30, 40, 50])
print("从列表:", s1.values)

# 从元组
s2 = pd.Series((1, 2, 3, 4, 5))
print("从元组:", s2.values)

# 从 NumPy 数组
s3 = pd.Series(np.arange(5))
print("从 NumPy 数组:", s3.values)

# 从字典
s4 = pd.Series({'a': 1, 'b': 2, 'c': 3})
print("从字典:")
print(s4)

print("\n2. 带参数创建 Series:")

# 带索引
s5 = pd.Series([100, 200, 300],
               index=['语文', '数学', '英语'],
               name='考试成绩')
print("带索引和名称:")
print(s5)

# 指定数据类型
s6 = pd.Series([1.1, 2.2, 3.3], dtype='float32')
print("指定数据类型:", s6.dtype, s6.values)

# 从标量创建
s7 = pd.Series(5, index=['a', 'b', 'c'])
print("从标量创建:")
print(s7)

print("\n3. 高级创建方法:")

# 从函数创建
def square(x):
    return x ** 2

s8 = pd.Series([square(i) for i in range(5)],
               index=[f'item_{i}' for i in range(5)])
print("从函数创建:")
print(s8)

# 从日期创建
dates = pd.date_range('2024-01-01', periods=5)
s9 = pd.Series(range(1, 6), index=dates)
print("日期索引 Series:")
print(s9)

# 从多级索引创建
arrays = [['A', 'A', 'B', 'B'], [1, 2, 1, 2]]
tuples = list(zip(*arrays))
index = pd.MultiIndex.from_tuples(tuples, names=['字母', '数字'])
s10 = pd.Series([10, 20, 30, 40], index=index)
print("多级索引 Series:")
print(s10)
```

### 2. 创建 DataFrame

```python
# 创建 DataFrame 的完整方法
print("\n=== 创建 DataFrame 方法大全 ===")

print("\n1. 从字典创建:")

# 字典的键为列名
df1 = pd.DataFrame({
    '姓名': ['张三', '李四', '王五'],
    '年龄': [25, 30, 35],
    '城市': ['北京', '上海', '广州']
})
print("字典方式1 (键为列名):")
print(df1)

# 字典的键为索引名
data_by_row = {
    '张三': {'年龄': 25, '城市': '北京', '工资': 8000},
    '李四': {'年龄': 30, '城市': '上海', '工资': 12000},
    '王五': {'年龄': 35, '城市': '广州', '工资': 15000}
}
df2 = pd.DataFrame.from_dict(data_by_row, orient='index')
print("\n字典方式2 (键为索引名):")
print(df2)

print("\n2. 从列表创建:")

# 列表套列表
df3 = pd.DataFrame([
    ['张三', 25, '北京', 8000],
    ['李四', 30, '上海', 12000],
    ['王五', 35, '广州', 15000]
], columns=['姓名', '年龄', '城市', '工资'])
print("列表套列表:")
print(df3)

# 列表套字典
df4 = pd.DataFrame([
    {'姓名': '张三', '年龄': 25, '城市': '北京'},
    {'姓名': '李四', '年龄': 30, '城市': '上海'},
    {'姓名': '王五', '年龄': 35, '城市': '广州', '工资': 15000}
])
print("\n列表套字典:")
print(df4)

print("\n3. 从 NumPy 数组创建:")

# 从 NumPy 数组
data_np = np.random.randint(0, 100, (4, 4))
df5 = pd.DataFrame(data_np,
                   columns=['语文', '数学', '英语', '物理'],
                   index=['学生A', '学生B', '学生C', '学生D'])
print("从 NumPy 数组:")
print(df5)

# 从多个 NumPy 数组
names = np.array(['张三', '李四', '王五', '赵六'])
ages = np.array([25, 30, 35, 28])
scores = np.array([85, 92, 78, 88])

df6 = pd.DataFrame({
    '姓名': names,
    '年龄': ages,
    '成绩': scores
})
print("\n从多个 NumPy 数组:")
print(df6)

print("\n4. 从外部数据创建:")

# 从 CSV 字符串创建
csv_data = """姓名,年龄,城市
张三,25,北京
李四,30,上海
王五,35,广州"""
from io import StringIO
df7 = pd.read_csv(StringIO(csv_data))
print("从 CSV 字符串:")
print(df7)

print("\n5. 特殊用途的 DataFrame:")

# 空DataFrame
df_empty = pd.DataFrame()
print("空DataFrame:", df_empty.shape)

# 指定结构的空DataFrame
df_structured = pd.DataFrame(columns=['姓名', '年龄', '城市'])
print("指定结构的空DataFrame:")
print(df_structured.columns.tolist())

# 重复数据
df_repeat = pd.DataFrame({
    '类别': ['A', 'B', 'C'],
    '值': [10, 20, 30]
}).repeat(3, ignore_index=True)
print("\n重复数据:")
print(df_repeat)
```

### 3. 创建多级索引数据

```python
# 创建多级索引数据
print("\n=== 创建多级索引数据 ===")

print("\n1. 多级索引 Series:")

# 创建多级索引 Series
arrays = [
    ['北京', '北京', '上海', '上海', '广州', '广州'],
    ['语文', '数学', '语文', '数学', '语文', '数学']
]
index = pd.MultiIndex.from_arrays(arrays, names=['城市', '科目'])
multi_series = pd.Series([85, 92, 78, 88, 95, 82], index=index)
print("多级索引 Series:")
print(multi_series)

# 访问多级索引数据
print("\n访问北京的数据:")
print(multi_series['北京'])

print("\n访问所有城市的语文成绩:")
print(multi_series[:, '语文'])

print("\n2. 多级索引 DataFrame:")

# 创建多级索引 DataFrame
data = {
    ('销售', 'Q1'): [100, 120, 90],
    ('销售', 'Q2'): [110, 130, 95],
    ('成本', 'Q1'): [60, 70, 55],
    ('成本', 'Q2'): [65, 75, 58]
}
df_multi = pd.DataFrame(data,
                       index=['北京', '上海', '广州'])
print("多级列名 DataFrame:")
print(df_multi)

# 创建多级行索引
index_arrays = [
    ['2024', '2024', '2024', '2025', '2025', '2025'],
    ['Q1', 'Q2', 'Q3', 'Q1', 'Q2', 'Q3'],
    ['A产品', 'B产品', 'C产品', 'A产品', 'B产品', 'C产品']
]
multi_col_index = pd.MultiIndex.from_arrays(index_arrays,
                                           names=['年份', '季度', '产品'])

df_multi2 = pd.DataFrame({
    '销售额': [100, 120, 90, 110, 130, 95],
    '利润': [20, 25, 15, 22, 28, 18]
}, index=multi_col_index)
print("\n多级行索引 DataFrame:")
print(df_multi2)
```

## Pandas 数据类型详解

### 1. 基础数据类型

```python
# Pandas 数据类型详解
print("=== Pandas 数据类型详解 ===")

print("\n1. 基础数据类型:")

# 创建包含各种数据类型的 DataFrame
df_types = pd.DataFrame({
    '整数列': [1, 2, 3, 4, 5],
    '浮点数列': [1.1, 2.2, 3.3, 4.4, 5.5],
    '字符串列': ['A', 'B', 'C', 'D', 'E'],
    '布尔列': [True, False, True, False, True],
    '日期列': pd.date_range('2024-01-01', periods=5)
})

print("DataFrame 及其数据类型:")
print(df_types)
print("\n各列数据类型:")
print(df_types.dtypes)

# 数值类型
print("\n数值类型详解:")

# 整数类型
df_int = pd.DataFrame({
    'int8_col': pd.Series([1, 2, 3], dtype='int8'),
    'int16_col': pd.Series([100, 200, 300], dtype='int16'),
    'int32_col': pd.Series([1000, 2000, 3000], dtype='int32'),
    'int64_col': pd.Series([10000, 20000, 30000], dtype='int64')
})
print("整数类型:")
print(df_int.dtypes)
# 整数类型:
# int8_col      int8
# int16_col    int16
# int32_col    int32
# int64_col    int64
# dtype: object

# 浮点数类型
df_float = pd.DataFrame({
    'float16_col': pd.Series([1.1, 2.2, 3.3], dtype='float16'),
    'float32_col': pd.Series([1.11, 2.22, 3.33], dtype='float32'),
    'float64_col': pd.Series([1.111, 2.222, 3.333], dtype='float64')
})
print("\n浮点数类型:")
print(df_float.dtypes)
# 浮点数类型:
# float16_col    float16
# float32_col    float32
# float64_col    float64
# dtype: object

# 字符串类型
print("\n字符串类型:")
df_str = pd.DataFrame({
    'object_col': ['Apple', 'Banana', 'Cherry'],  # object 类型
    'string_col': pd.Series(['Apple', 'Banana', 'Cherry'], dtype='string')  # string 类型
})
print(df_str.dtypes)
# 字符串类型:
# object_col     object
# string_col     string
# dtype: object
```

### 2. 日期时间类型

```python
# 日期时间数据类型
print("\n=== 日期时间数据类型 ===")

print("\n1. 时间戳类型 (datetime64[ns]):")

# 创建日期时间数据
df_dates = pd.DataFrame({
    '日期': pd.date_range('2024-01-01', periods=5),
    '销售额': [100, 120, 90, 110, 130]
})
print("日期时间 DataFrame:")
print(df_dates)
print("日期列类型:", df_dates['日期'].dtype)
# 日期时间 DataFrame:
#           日期  销售额
# 0 2024-01-01  100
# 1 2024-01-02  120
# 2 2024-01-03   90
# 3 2024-01-04  110
# 4 2024-01-05  130
# 日期列类型: datetime64[ns]


# 日期时间操作
print("\n日期时间操作:")
df_dates['年份'] = df_dates['日期'].dt.year
df_dates['月份'] = df_dates['日期'].dt.month
df_dates['星期'] = df_dates['日期'].dt.day_name()
print("添加时间属性后:")
print(df_dates)
# 添加时间属性后:
#           日期  销售额    年份  月份         星期
# 0 2024-01-01  100  2024    1      Monday
# 1 2024-01-02  120  2024    1     Tuesday
# 2 2024-01-03   90  2024    1   Wednesday
# 3 2024-01-04  110  2024    1    Thursday
# 4 2024-01-05  130  2024    1      Friday

print("\n2. 时间差类型 (timedelta64[ns]):")

# 创建时间差数据
start_dates = pd.date_range('2024-01-01', periods=3)
end_dates = pd.date_range('2024-01-05', periods=3)

df_timedelta = pd.DataFrame({
    '开始日期': start_dates,
    '结束日期': end_dates
})
df_timedelta['时间差'] = df_timedelta['结束日期'] - df_timedelta['开始日期']
print("时间差 DataFrame:")
print(df_timedelta)
print("时间差类型:", df_timedelta['时间差'].dtype)
# 时间差 DataFrame:
#          开始日期        结束日期    时间差
# 0 2024-01-01 2024-01-05 4 days
# 1 2024-01-02 2024-01-06 4 days
# 2 2024-01-03 2024-01-07 4 days
# 时间差类型: timedelta64[ns]

print("\n3. 时间周期类型:")

# 创建周期数据
periods_data = pd.period_range('2024-01', periods=4, freq='M')
df_periods = pd.DataFrame({
    '月份': periods_data,
    '销售额': [1000, 1200, 900, 1100]
})
print("周期数据:")
print(df_periods)
print("周期类型:", df_periods['月份'].dtype)
# 周期数据:
#        月份   销售额
# 0  2024-01   1000
# 1  2024-02   1200
# 2  2024-03    900
# 3  2024-04   1100
# 周期类型: period[M]

```

### 3. 分类数据类型

```python
# 分类数据类型
print("\n=== 分类数据类型 ===")

print("\n1. 基础分类数据:")

# 创建分类数据
categories = ['小', '中', '大']
sizes = ['中', '小', '大', '中', '小', '大', '中']

df_categorical = pd.DataFrame({
    '产品名称': ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
    '尺寸': sizes
})

# 转换为分类类型
df_categorical['尺寸分类'] = pd.Categorical(df_categorical['尺寸'],
                                          categories=categories,
                                          ordered=True)
print("分类数据:")
print(df_categorical)
print("\n数据类型:")
print(df_categorical.dtypes)

# 分类数据的优势
print("\n分类数据的优势:")
print("内存使用对比:")
print("object 类型内存:", df_categorical['尺寸'].memory_usage())
print("category 类型内存:", df_categorical['尺寸分类'].memory_usage())

print("\n分类统计:")
print("尺寸分布:")
print(df_categorical['尺寸分类'].value_counts())

print("\n2. 自定义分类数据:")

# 自定义顺序的分类
education_levels = ['小学', '初中', '高中', '本科', '硕士', '博士']
education = ['本科', '硕士', '本科', '高中', '博士', '本科', '硕士']

df_education = pd.DataFrame({
    '姓名': ['张三', '李四', '王五', '赵六', '钱七', '孙八', '周九'],
    '学历': pd.Categorical(education, categories=education_levels, ordered=True)
})
print("学历分类数据:")
print(df_education)
print("\n学历排序:")
print(df_education.sort_values('学历'))
```

### 4. 特殊数据类型

```python
# 特殊数据类型
print("\n=== 特殊数据类型 ===")

print("\n1. 空值处理:")

# 创建包含空值的数据
df_nulls = pd.DataFrame({
    '整数列': [1, 2, None, 4, 5],
    '浮点数列': [1.1, None, 3.3, 4.4, None],
    '字符串列': ['A', 'B', None, 'D', 'E'],
    '布尔列': [True, False, None, True, False]
})
print("包含空值的数据:")
print(df_nulls)
print("\n数据类型 (包含空值):")
print(df_nulls.dtypes)

# 空值检测
print("\n空值检测:")
print("各列空值数量:")
print(df_nulls.isnull().sum())

print("\n2. 混合类型数据:")

# 包含混合类型的数据
mixed_data = [
    [1, 'A', True],
    [2.5, 'B', False],
    [3, 'C', None],
    [None, 'D', True]
]
df_mixed = pd.DataFrame(mixed_data, columns=['数字', '字母', '布尔'])
print("混合类型数据:")
print(df_mixed)
print("\n混合类型 dtype:")
print(df_mixed.dtypes)

print("\n3. 稀疏数据:")

# 稀疏数据示例
df_sparse = pd.DataFrame({
    'ID': [1, 2, 3, 4, 5],
    '值': [10, 0, 30, 0, 50],
    '标签': ['A', None, 'B', None, 'C']
})
print("稀疏数据:")
print(df_sparse)

# 只显示非空值
print("\n非空行:")
print(df_sparse.dropna())
```

### 5. 数据类型转换

```python
# 数据类型转换
print("\n=== 数据类型转换 ===")

# 创建原始数据
df_convert = pd.DataFrame({
    'ID': ['001', '002', '003', '004'],
    '价格': ['100.50', '200.75', '150.25', '300.00'],
    '数量': ['10', '20', '15', '25'],
    '日期': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04'],
    '是否完成': ['是', '否', '是', '是']
})

print("原始数据:")
print(df_convert)
print("\n原始数据类型:")
print(df_convert.dtypes)

# 数据类型转换
print("\n数据类型转换:")

# 字符串转整数
df_convert['ID_new'] = df_convert['ID'].astype(int)

# 字符串转浮点数
df_convert['价格_new'] = pd.to_numeric(df_convert['价格'])

# 字符串转整数
df_convert['数量_new'] = df_convert['数量'].astype(int)

# 字符串转日期
df_convert['日期_new'] = pd.to_datetime(df_convert['日期'])

# 字符串转布尔值
df_convert['是否完成_new'] = df_convert['是否完成'].map({'是': True, '否': False})

print("\n转换后的数据:")
print(df_convert[['ID_new', '价格_new', '数量_new', '日期_new', '是否完成_new']])
print("\n转换后的数据类型:")
print(df_convert[['ID_new', '价格_new', '数量_new', '日期_new', '是否完成_new']].dtypes)

# 智能类型转换
print("\n智能类型转换:")
df_auto = df_convert[['ID', '价格', '数量', '日期']].copy()
df_auto = df_auto.infer_objects()
print("自动推断类型后:")
print(df_auto.dtypes)

# 处理转换错误
print("\n处理转换错误:")
problematic_data = pd.Series(['100', '200', 'invalid', '300'])
print("问题数据:", problematic_data.tolist())

# 使用 errors 参数处理错误
safe_conversion = pd.to_numeric(problematic_data, errors='coerce')
print("安全转换结果:", safe_conversion.tolist())
print("转换失败的设为:", safe_conversion.isna().sum(), "个空值")
```

## Pandas 实际应用案例

### 1. 数据清洗示例

```python
# 实际应用：数据清洗
print("\n=== 实际应用：数据清洗 ===")

# 创建包含问题的模拟数据
data_messy = {
    '姓名': ['张三', '李四  ', '  王五', '赵六', '钱七', None],
    '年龄': [25, '30岁', 35, '未知', 28, 32],
    '工资': ['8000', '12,000', '15000元', '10,000元', '13000', 'invalid'],
    '入职日期': ['2024-01-01', '2024/01/02', None, '2024-01-04', 'invalid', '2024-01-06'],
    '部门': ['技术部', '销售部', '技术部', '市场部', None, '技术部']
}

df_messy = pd.DataFrame(data_messy)
print("原始脏数据:")
print(df_messy)

# 数据清洗步骤
print("\n=== 数据清洗过程 ===")

# 步骤1：处理姓名列（去除空格，处理空值）
df_clean = df_messy.copy()
df_clean['姓名'] = df_clean['姓名'].str.strip()
df_clean = df_clean.dropna(subset=['姓名'])  # 删除姓名为空的行
print("1. 清洗姓名后:")
print(df_clean[['姓名']])

# 步骤2：处理年龄列（提取数字，处理无效值）
def clean_age(age_str):
    if pd.isna(age_str):
        return None
    if isinstance(age_str, (int, float)):
        return age_str
    if isinstance(age_str, str):
        # 提取数字
        import re
        match = re.search(r'\d+', str(age_str))
        if match:
            return int(match.group())
    return None

df_clean['年龄_clean'] = df_clean['年龄'].apply(clean_age)
df_clean['年龄_clean'] = pd.to_numeric(df_clean['年龄_clean'], errors='coerce')
print("2. 清洗年龄后:")
print(df_clean[['年龄', '年龄_clean']])

# 步骤3：处理工资列（提取数字，转换为数值）
def clean_salary(salary_str):
    if pd.isna(salary_str):
        return None
    if isinstance(salary_str, (int, float)):
        return float(salary_str)
    if isinstance(salary_str, str):
        # 移除非数字字符，除了小数点
        import re
        clean_str = re.sub(r'[^\d.]', '', salary_str)
        if clean_str:
            try:
                return float(clean_str)
            except:
                return None
    return None

df_clean['工资_clean'] = df_clean['工资'].apply(clean_salary)
print("3. 清洗工资后:")
print(df_clean[['工资', '工资_clean']])

# 步骤4：处理日期列
def clean_date(date_str):
    if pd.isna(date_str):
        return None
    try:
        # 尝试不同的日期格式
        for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%Y年%m月%d日']:
            try:
                return pd.to_datetime(date_str, format=fmt)
            except:
                continue
        # 如果都失败了，让 pandas 自动解析
        return pd.to_datetime(date_str, errors='coerce')
    except:
        return None

df_clean['入职日期_clean'] = df_clean['入职日期'].apply(clean_date)
print("4. 清洗日期后:")
print(df_clean[['入职日期', '入职日期_clean']])

# 步骤5：处理部门列（填充空值）
df_clean['部门_clean'] = df_clean['部门'].fillna('未知')
print("5. 清洗部门后:")
print(df_clean[['部门', '部门_clean']])

# 最终清洗结果
df_final = df_clean[['姓名', '年龄_clean', '工资_clean', '入职日期_clean', '部门_clean']]
df_final.columns = ['姓名', '年龄', '工资', '入职日期', '部门']

print("\n=== 最终清洗结果 ===")
print("清洗后的数据:")
print(df_final)
print("\n清洗后数据类型:")
print(df_final.dtypes)
print("\n数据质量报告:")
print(f"总记录数: {len(df_final)}")
print(f"完整记录数: {len(df_final.dropna())}")
print(f"各列空值数:")
print(df_final.isnull().sum())
```

### 2. 数据分析示例

```python
# 实际应用：数据分析
print("\n=== 实际应用：数据分析 ===")

# 创建模拟销售数据
np.random.seed(42)
n_records = 100

sales_data = {
    '订单ID': [f'ORD{str(i+1).zfill(4)}' for i in range(n_records)],
    '客户ID': [f'CUST{str(np.random.randint(1, 21)).zfill(3)}' for _ in range(n_records)],
    '产品类别': np.random.choice(['电子产品', '服装', '食品', '图书', '家居'], n_records),
    '产品名称': np.random.choice([
        '手机', '笔记本电脑', '耳机',  # 电子产品
        'T恤', '牛仔裤', '连衣裙',    # 服装
        '面包', '牛奶', '水果',       # 食品
        '小说', '教材', '杂志',      # 图书
        '沙发', '桌子', '灯具'       # 家居
    ], n_records),
    '数量': np.random.randint(1, 11, n_records),
    '单价': np.random.uniform(10, 1000, n_records).round(2),
    '订单日期': pd.date_range('2024-01-01', periods=n_records, freq='D'),
    '销售员': np.random.choice(['张三', '李四', '王五', '赵六'], n_records)
}

df_sales = pd.DataFrame(sales_data)
df_sales['总金额'] = df_sales['数量'] * df_sales['单价']

print("销售数据样本:")
print(df_sales.head())
print(f"\n数据形状: {df_sales.shape}")
print(f"数据时间范围: {df_sales['订单日期'].min()} 到 {df_sales['订单日期'].max()}")

# 数据分析
print("\n=== 销售数据分析 ===")

# 1. 基本统计
print("\n1. 基本统计信息:")
print("总销售额:", df_sales['总金额'].sum().round(2))
print("平均订单金额:", df_sales['总金额'].mean().round(2))
print("最大订单金额:", df_sales['总金额'].max().round(2))
print("最小订单金额:", df_sales['总金额'].min().round(2))
print("总订单数:", len(df_sales))

# 2. 按产品类别分析
print("\n2. 按产品类别分析:")
category_analysis = df_sales.groupby('产品类别').agg({
    '总金额': ['sum', 'mean', 'count'],
    '数量': ['sum', 'mean']
}).round(2)
print(category_analysis)

# 3. 按销售员分析
print("\n3. 按销售员分析:")
salesperson_analysis = df_sales.groupby('销售员').agg({
    '订单ID': 'count',
    '总金额': ['sum', 'mean']
}).round(2)
salesperson_analysis.columns = ['订单数', '总销售额', '平均订单金额']
print(salesperson_analysis)

# 4. 时间趋势分析
print("\n4. 时间趋势分析:")
df_sales['月份'] = df_sales['订单日期'].dt.month
monthly_sales = df_sales.groupby('月份')['总金额'].agg(['sum', 'count']).round(2)
monthly_sales.columns = ['月销售额', '订单数']
print(monthly_sales)

# 5. 客户分析
print("\n5. 客户分析:")
customer_analysis = df_sales.groupby('客户ID').agg({
    '订单ID': 'count',
    '总金额': 'sum'
}).round(2)
customer_analysis.columns = ['购买次数', '总消费金额']
customer_analysis = customer_analysis.sort_values('总消费金额', ascending=False)
print("前10名客户:")
print(customer_analysis.head(10))

# 6. 产品分析
print("\n6. 产品分析:")
product_analysis = df_sales.groupby(['产品类别', '产品名称']).agg({
    '总金额': 'sum',
    '数量': 'sum'
}).round(2)
product_analysis = product_analysis.sort_values('总金额', ascending=False)
print("最畅销产品 (前10):")
print(product_analysis.head(10))
```

### 3. 数据可视化准备

```python
# 实际应用：为可视化准备数据
print("\n=== 实际应用：为可视化准备数据 ===")

# 准备图表数据
print("\n1. 准备月度销售趋势图数据:")
monthly_trend = df_sales.groupby(df_sales['订单日期'].dt.to_period('M')).agg({
    '总金额': 'sum',
    '订单ID': 'count'
}).reset_index()
monthly_trend['月份'] = monthly_trend['订单日期'].dt.to_timestamp()
monthly_trend.columns = ['月份', '销售额', '订单数']
print(monthly_trend)

print("\n2. 准备产品类别饼图数据:")
category_pie = df_sales.groupby('产品类别')['总金额'].sum().sort_values(ascending=False)
print("各类别销售额:")
print(category_pie)

print("\n3. 准备销售员业绩柱状图数据:")
salesperson_bar = df_sales.groupby('销售员')['总金额'].sum().sort_values(ascending=False)
print("销售员业绩:")
print(salesperson_bar)

print("\n4. 准备客户分布直方图数据:")
customer_dist = df_sales.groupby('客户ID')['总金额'].sum()
print("客户消费分布统计:")
print(f"客户数: {len(customer_dist)}")
print(f"平均消费: {customer_dist.mean():.2f}")
print(f"消费中位数: {customer_dist.median():.2f}")
print(f"最高消费: {customer_dist.max():.2f}")
print(f"最低消费: {customer_dist.min():.2f}")

print("\n5. 准备时间序列数据:")
daily_sales = df_sales.groupby('订单日期')['总金额'].sum().reset_index()
print("每日销售额 (前10天):")
print(daily_sales.head(10))
```

## Pandas 最佳实践

### 1. 性能优化技巧

```python
# Pandas 性能优化
print("\n=== Pandas 性能优化 ===")

print("\n1. 使用向量化操作代替循环:")

# 创建大数据集
large_df = pd.DataFrame({
    'A': np.random.randn(100000),
    'B': np.random.randn(100000),
    'C': np.random.randn(100000)
})

# 慢速方法：循环
import time
start_time = time.time()
result_slow = []
for i in range(len(large_df)):
    result_slow.append(large_df.iloc[i]['A'] + large_df.iloc[i]['B'])
slow_time = time.time() - start_time

# 快速方法：向量化
start_time = time.time()
result_fast = large_df['A'] + large_df['B']
fast_time = time.time() - start_time

print(f"循环方法: {slow_time:.4f}秒")
print(f"向量化方法: {fast_time:.4f}秒")
print(f"性能提升: {slow_time/fast_time:.1f}倍")

print("\n2. 使用适当的数据类型:")

# 内存使用对比
df_memory = pd.DataFrame({
    '整数': range(10000),
    '浮点数': np.random.randn(10000),
    '字符串': ['test'] * 10000
})

print("原始内存使用:")
print(df_memory.memory_usage(deep=True))

# 优化数据类型
df_optimized = df_memory.copy()
df_optimized['整数'] = df_optimized['整数'].astype('int32')
df_optimized['浮点数'] = df_optimized['浮点数'].astype('float32')
df_optimized['字符串'] = df_optimized['字符串'].astype('category')

print("\n优化后内存使用:")
print(df_optimized.memory_usage(deep=True))

memory_reduction = (df_memory.memory_usage(deep=True).sum() -
                  df_optimized.memory_usage(deep=True).sum())
print(f"内存节省: {memory_reduction/1024:.2f} KB")

print("\n3. 使用 query() 方法:")

# 比较筛选方法
df_test = pd.DataFrame({
    'A': np.random.randn(10000),
    'B': np.random.randn(10000),
    'C': np.random.choice(['X', 'Y', 'Z'], 10000)
})

# 传统方法
start_time = time.time()
result_traditional = df_test[(df_test['A'] > 0) & (df_test['B'] < 0) & (df_test['C'] == 'X')]
traditional_time = time.time() - start_time

# query 方法
start_time = time.time()
result_query = df_test.query('A > 0 and B < 0 and C == "X"')
query_time = time.time() - start_time

print(f"传统筛选: {traditional_time:.6f}秒")
print(f"query筛选: {query_time:.6f}秒")
print("结果相同:", len(result_traditional) == len(result_query))
```

### 2. 代码可读性技巧

```python
# 代码可读性技巧
print("\n=== 代码可读性技巧 ===")

# 创建示例数据
df_example = pd.DataFrame({
    'employee_id': range(1, 11),
    'name': [f'员工{i}' for i in range(1, 11)],
    'department': np.random.choice(['技术', '销售', '市场', '人事'], 10),
    'salary': np.random.randint(5000, 20000, 10),
    'experience_years': np.random.randint(0, 10, 10)
})

print("示例数据:")
print(df_example)

print("\n1. 使用描述性的变量名:")

# ❌ 不好的命名方式
a = df_example[df_example['salary'] > 10000]
b = a['department'].value_counts()

# ✅ 好的命名方式
high_salary_employees = df_example[df_example['salary'] > 10000]
department_distribution = high_salary_employees['department'].value_counts()

print("高薪员工部门分布:")
print(department_distribution)

print("\n2. 链式操作的可读性:")

# ❌ 难以阅读的链式操作
complex_result = df_example.groupby('department').agg({'salary': ['mean', 'max'], 'experience_years': 'mean'}).reset_index()

# ✅ 可读性更好的分步操作
step1 = df_example.groupby('department')
step2 = step1.agg({
    'salary': ['mean', 'max'],
    'experience_years': 'mean'
})
readable_result = step2.reset_index()

print("部门分析结果:")
print(readable_result)

print("\n3. 添加注释和文档:")

def analyze_employee_performance(df, min_salary=10000):
    """
    分析员工表现

    参数:
    df: 员工数据DataFrame
    min_salary: 最低薪资阈值

    返回:
    分析结果字典
    """
    # 筛选高薪员工
    high_earners = df[df['salary'] >= min_salary]

    # 按部门分析
    dept_analysis = high_earners.groupby('department').agg({
        'salary': ['count', 'mean'],
        'experience_years': 'mean'
    }).round(2)

    # 计算整体统计
    overall_stats = {
        'total_employees': len(df),
        'high_earners_count': len(high_earners),
        'high_earner_percentage': len(high_earners) / len(df) * 100,
        'average_salary': df['salary'].mean(),
        'high_earner_avg_salary': high_earners['salary'].mean()
    }

    return {
        'department_analysis': dept_analysis,
        'overall_statistics': overall_stats
    }

analysis_result = analyze_employee_performance(df_example, min_salary=12000)
print("\n员工表现分析:")
print("整体统计:")
for key, value in analysis_result['overall_statistics'].items():
    print(f"  {key}: {value}")
```

### 3. 错误处理和调试

```python
# 错误处理和调试
print("\n=== 错误处理和调试 ===")

print("\n1. 安全的数据操作:")

# 创建可能出错的数据
df_problematic = pd.DataFrame({
    'A': [1, 2, None, 4, 5],
    'B': ['X', 'Y', 'Z', None, 'W'],
    'C': [1.1, 'invalid', 3.3, 4.4, None]
})

print("问题数据:")
print(df_problematic)

# 安全的数据操作函数
def safe_column_operation(df, column_name, operation='sum'):
    """
    安全的列操作函数

    参数:
    df: DataFrame
    column_name: 列名
    operation: 操作类型

    返回:
    操作结果或错误信息
    """
    try:
        if column_name not in df.columns:
            return f"错误: 列 '{column_name}' 不存在"

        column_data = df[column_name]

        if operation == 'sum':
            if pd.api.types.is_numeric_dtype(column_data):
                return column_data.sum()
            else:
                return f"错误: 列 '{column_name}' 不是数值类型，无法求和"

        elif operation == 'count':
            return len(column_data.dropna())

        elif operation == 'unique':
            return column_data.nunique()

        else:
            return f"错误: 不支持的操作 '{operation}'"

    except Exception as e:
        return f"操作失败: {str(e)}"

# 测试安全操作
print("\n安全操作测试:")
print("A列求和:", safe_column_operation(df_problematic, 'A', 'sum'))
print("A列计数:", safe_column_operation(df_problematic, 'A', 'count'))
print("B列计数:", safe_column_operation(df_problematic, 'B', 'count'))
print("B列求和:", safe_column_operation(df_problematic, 'B', 'sum'))  # 会报错
print("不存在的列:", safe_column_operation(df_problematic, 'X', 'count'))

print("\n2. 数据验证:")

def validate_dataframe(df, required_columns=None, check_nulls=True):
    """
    验证DataFrame的质量

    参数:
    df: 要验证的DataFrame
    required_columns: 必需的列名列表
    check_nulls: 是否检查空值

    返回:
    验证结果字典
    """
    validation_result = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'info': {}
    }

    # 检查必需列
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"缺少必需列: {missing_columns}")

    # 检查空值
    if check_nulls:
        null_counts = df.isnull().sum()
        columns_with_nulls = null_counts[null_counts > 0]

        if len(columns_with_nulls) > 0:
            validation_result['warnings'].append("发现空值")
            validation_result['info']['null_counts'] = columns_with_nulls.to_dict()

    # 检查数据类型
    validation_result['info']['dtypes'] = df.dtypes.to_dict()
    validation_result['info']['shape'] = df.shape

    return validation_result

# 数据验证示例
validation = validate_dataframe(df_problematic, required_columns=['A', 'B'])
print("\n数据验证结果:")
print("验证通过:", validation['is_valid'])
if validation['errors']:
    print("错误:", validation['errors'])
if validation['warnings']:
    print("警告:", validation['warnings'])
print("信息:", validation['info'])
```

## 总结

### Pandas 核心优势

1. **直观的数据结构**: Series 和 DataFrame 就像带标签的数组和表格
2. **强大的数据处理**: 筛选、排序、分组、聚合等操作简单高效
3. **丰富的数据类型**: 支持数值、字符串、日期、分类等多种数据类型
4. **灵活的数据导入**: 支持 CSV、Excel、数据库等多种数据源
5. **优秀的性能**: 基于 NumPy 构建，处理大数据效率高

### 关键概念回顾

1. **Series**: 一维带标签数组，类似于带索引的列表
2. **DataFrame**: 二维表格数据，类似于 Excel 表格或数据库表
3. **Index**: 数据标签系统，提供快速的数据访问
4. **数据类型**: 理解各种数据类型及其转换
5. **向量化操作**: 使用数组操作代替循环提高性能

### 实用技能

1. **数据创建**: 多种方式创建 Series 和 DataFrame
2. **数据访问**: 索引、切片、条件筛选
3. **数据清洗**: 处理空值、重复值、格式转换
4. **数据分析**: 分组聚合、统计分析、时间序列处理
5. **数据可视化准备**: 为图表准备合适的数据格式

### 最佳实践

1. **性能优化**: 使用向量化操作、合适的数据类型
2. **代码可读性**: 使用描述性命名、适当的注释
3. **错误处理**: 安全的数据操作、数据验证
4. **内存管理**: 注意大数据集的内存使用

掌握 Pandas 是数据分析的基础技能，通过系统的学习和实践，您将能够高效地处理各种数据任务！