# Pandas数据合并完全指南

## 目录
1. [概述](#概述)
2. [concat - 数据拼接](#concat---数据拼接)
3. [merge - 数据合并](#merge---数据合并)
4. [join - 索引连接](#join---索引连接)
5. [append - 数据追加](#append---数据追加)
6. [时间序列合并](#时间序列合并)
7. [多文件合并](#多文件合并)
8. [combine - 数据组合](#combine---数据组合)
9. [compare - 数据比较](#compare---数据比较)
10. [最佳实践和性能优化](#最佳实践和性能优化)

---

## 概述

Pandas提供了多种数据合并的方法，每种方法都有其特定的应用场景：

- **concat()**: 最通用的拼接方法，可以在任意轴上拼接
- **merge()**: 类似SQL的表连接操作，基于列值进行合并
- **join()**: 基于索引的合并操作
- **append()**: 向DataFrame添加行（已弃用，推荐使用concat）
- **combine_first()**: 用非NaN值填充缺失值
- **compare()**: 比较两个DataFrame的差异

---

## concat - 数据拼接

`pd.concat()` 是最灵活的数据合并方法，可以沿任意轴拼接DataFrame或Series。

### 基础语法
```python
pd.concat(objs, axis=0, join='outer', ignore_index=False, keys=None, sort=False)
```

### 主要参数说明
- **objs**: 要合并的DataFrame或Series列表
- **axis**: 拼接轴（0=垂直/行，1=水平/列）
- **join**: 连接方式（'outer'=并集，'inner'=交集）
- **ignore_index**: 是否忽略原索引
- **keys**: 创建多级索引的键
- **sort**: 是否对非连接轴进行排序

### 垂直拼接（axis=0）
```python
import pandas as pd
import numpy as np

# 创建示例数据
df1 = pd.DataFrame({
    'A': ['A0', 'A1', 'A2'],
    'B': ['B0', 'B1', 'B2'],
    'C': ['C0', 'C1', 'C2']
}, index=[0, 1, 2])

df2 = pd.DataFrame({
    'A': ['A3', 'A4', 'A5'],
    'B': ['B3', 'B4', 'B5'],
    'C': ['C3', 'C4', 'C5']
}, index=[3, 4, 5])

# 垂直拼接
result = pd.concat([df1, df2])
print("垂直拼接结果:")
print(result)
```

### 水平拼接（axis=1）
```python
# 创建列名不同的DataFrame
df3 = pd.DataFrame({
    'D': ['D0', 'D1', 'D2'],
    'E': ['E0', 'E1', 'E2']
}, index=[0, 1, 2])

# 水平拼接
result = pd.concat([df1, df3], axis=1)
print("水平拼接结果:")
print(result)
```

### 使用keys创建多级索引
```python
# 使用keys参数
result = pd.concat([df1, df2], keys=['表1', '表2'])
print("带多级索引的拼接结果:")
print(result)

# 访问特定表的数据
print("表1的数据:")
print(result.loc['表1'])
```

### 处理重复索引
```python
# 创建索引重叠的DataFrame
df4 = pd.DataFrame({
    'A': ['A0', 'A1', 'A2'],
    'B': ['B0', 'B1', 'B2']
}, index=[0, 1, 2])

df5 = pd.DataFrame({
    'A': ['A3', 'A4', 'A5'],
    'B': ['B3', 'B4', 'B5']
}, index=[1, 2, 3])

# 外连接（默认）
result_outer = pd.concat([df4, df5], axis=1, join='outer')
print("外连接结果（包含所有索引）:")
print(result_outer)

# 内连接
result_inner = pd.concat([df4, df5], axis=1, join='inner')
print("内连接结果（只包含共同索引）:")
print(result_inner)
```

---

## merge - 数据合并

`pd.merge()` 类似于SQL的JOIN操作，基于一个或多个键值进行合并。

### 基础语法
```python
pd.merge(left, right, how='inner', on=None, left_on=None, right_on=None,
         left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'))
```

### 主要参数说明
- **how**: 连接方式（'inner', 'outer', 'left', 'right'）
- **on**: 连接的列名
- **left_on/right_on**: 左右两侧不同的连接列
- **left_index/right_index**: 使用索引作为连接键
- **suffixes**: 重名列的后缀

### 创建示例数据
```python
# 创建员工表
employees = pd.DataFrame({
    'emp_id': [1, 2, 3, 4, 5],
    'name': ['张三', '李四', '王五', '赵六', '钱七'],
    'dept_id': [1, 2, 1, 3, 2]
})

# 创建部门表
departments = pd.DataFrame({
    'dept_id': [1, 2, 3, 4],
    'dept_name': ['技术部', '销售部', '市场部', '人事部'],
    'manager': ['技术总监', '销售总监', '市场总监', '人事总监']
})

print("员工表:")
print(employees)
print("\n部门表:")
print(departments)
```

### 内连接（inner join）
```python
# 内连接 - 只保留两个表都存在的记录
result_inner = pd.merge(employees, departments, on='dept_id', how='inner')
print("内连接结果:")
print(result_inner)
```

### 左连接（left join）
```python
# 左连接 - 保留左表所有记录
result_left = pd.merge(employees, departments, on='dept_id', how='left')
print("左连接结果:")
print(result_left)
```

### 右连接（right join）
```python
# 右连接 - 保留右表所有记录
result_right = pd.merge(employees, departments, on='dept_id', how='right')
print("右连接结果:")
print(result_right)
```

### 外连接（outer join）
```python
# 外连接 - 保留两个表所有记录
result_outer = pd.merge(employees, departments, on='dept_id', how='outer')
print("外连接结果:")
print(result_outer)
```

### 多键连接
```python
# 创建包含多键的数据
performance = pd.DataFrame({
    'emp_id': [1, 2, 3, 4, 1, 2],
    'year': [2023, 2023, 2023, 2023, 2024, 2024],
    'score': [85, 92, 78, 88, 90, 95],
    'project': ['A', 'B', 'A', 'C', 'B', 'C']
})

# 基于多键连接
result_multi = pd.merge(employees, performance,
                       on=['emp_id'], how='inner')
print("多键连接结果:")
print(result_multi)
```

### 不同列名连接
```python
# 创建列名不同的数据
salaries = pd.DataFrame({
    'employee_id': [1, 2, 3, 4, 5],
    'salary': [8000, 12000, 15000, 10000, 13000],
    'bonus': [1000, 2000, 3000, 1500, 2500]
})

# 使用不同列名连接
result_diff = pd.merge(employees, salaries,
                      left_on='emp_id', right_on='employee_id', how='inner')
print("不同列名连接结果:")
print(result_diff)
```

---

## join - 索引连接

`DataFrame.join()` 是一个便捷方法，用于基于索引进行合并操作。

### 基础语法
```python
df.join(other, on=None, how='left', lsuffix='', rsuffix='', sort=False)
```

### 示例数据
```python
# 创建基础DataFrame
df1 = pd.DataFrame({
    'A': ['A0', 'A1', 'A2'],
    'B': ['B0', 'B1', 'B2']
}, index=['K0', 'K1', 'K2'])

df2 = pd.DataFrame({
    'C': ['C0', 'C1', 'C2'],
    'D': ['D0', 'D1', 'D2']
}, index=['K0', 'K2', 'K3'])

print("DataFrame 1:")
print(df1)
print("\nDataFrame 2:")
print(df2)
```

### 左连接（默认）
```python
# 左连接 - 保留左侧索引
result_left = df1.join(df2, how='left')
print("左连接结果:")
print(result_left)
```

### 右连接
```python
# 右连接 - 保留右侧索引
result_right = df1.join(df2, how='right')
print("右连接结果:")
print(result_right)
```

### 内连接
```python
# 内连接 - 只保留共同索引
result_inner = df1.join(df2, how='inner')
print("内连接结果:")
print(result_inner)
```

### 外连接
```python
# 外连接 - 保留所有索引
result_outer = df1.join(df2, how='outer')
print("外连接结果:")
print(result_outer)
```

### 连接多个DataFrame
```python
df3 = pd.DataFrame({
    'E': ['E0', 'E1'],
    'F': ['F0', 'F1']
}, index=['K0', 'K1'])

# 连接多个DataFrame
result_multi = df1.join([df2, df3], how='outer')
print("多DataFrame连接结果:")
print(result_multi)
```

---

## append - 数据追加

`DataFrame.append()` 用于向DataFrame添加行，但已在Pandas 2.0版本中被弃用。

### ⚠️ 弃用警告
从Pandas 1.4.0开始弃用，2.0版本已删除。推荐使用`pd.concat()`替代。

### 旧方法（已弃用）
```python
# 创建示例数据
df1 = pd.DataFrame({
    'A': ['A0', 'A1'],
    'B': ['B0', 'B1']
}, index=[0, 1])

df2 = pd.DataFrame({
    'A': ['A2', 'A3'],
    'B': ['B2', 'B3']
}, index=[2, 3])

# 旧的append方法（已弃用）
result = df1.append(df2)
print("使用append的结果:")
print(result)
```

### 推荐的替代方法
```python
# 方法1: 使用concat
result = pd.concat([df1, df2], ignore_index=True)
print("使用concat替代append:")
print(result)

# 方法2: 使用loc添加单行
new_row = pd.Series(['A4', 'B4'], index=['A', 'B'], name=4)
df1.loc[4] = new_row
print("使用loc添加单行:")
print(df1)
```

### 追加字典
```python
# 追加字典列表
dict_data = [{'A': 'A5', 'B': 'B5'}, {'A': 'A6', 'B': 'B6'}]
result = pd.concat([df1, pd.DataFrame(dict_data)], ignore_index=True)
print("追加字典数据:")
print(result)
```

---

## 时间序列合并

Pandas提供了专门用于时间序列合并的方法。

### merge_ordered - 有序合并
```python
# 创建时间序列数据
stock_prices = pd.DataFrame({
    'date': pd.date_range('2023-01-01', periods=5),
    'stock': ['AAPL', 'AAPL', 'AAPL', 'GOOGL', 'GOOGL'],
    'price': [150, 152, 151, 2800, 2850]
})

stock_volumes = pd.DataFrame({
    'date': pd.date_range('2023-01-02', periods=4),
    'stock': ['AAPL', 'AAPL', 'GOOGL', 'GOOGL'],
    'volume': [1000000, 1200000, 500000, 600000]
})

# 有序合并
from pandas import merge_ordered

result = merge_ordered(stock_prices, stock_volumes, on=['date', 'stock'],
                       fill_method='ffill')
print("有序合并结果:")
print(result)
```

### merge_asof - 时间点匹配合并
```python
# 创建交易数据和报价数据
trades = pd.DataFrame({
    'time': pd.to_datetime(['2023-01-01 09:30:00', '2023-01-01 09:31:00',
                           '2023-01-01 09:32:00', '2023-01-01 09:33:00']),
    'ticker': ['AAPL', 'GOOGL', 'AAPL', 'MSFT'],
    'price': [150.0, 2800.0, 152.0, 250.0],
    'quantity': [100, 50, 200, 75]
})

quotes = pd.DataFrame({
    'time': pd.to_datetime(['2023-01-01 09:30:01', '2023-01-01 09:30:30',
                           '2023-01-01 09:31:30', '2023-01-01 09:32:30']),
    'ticker': ['AAPL', 'AAPL', 'GOOGL', 'AAPL'],
    'bid': [149.8, 151.8, 2798.0, 152.1],
    'ask': [150.2, 152.2, 2802.0, 152.4]
})

# asof合并
from pandas import merge_asof

result = merge_asof(trades, quotes, on='time', by='ticker', direction='nearest')
print("asof合并结果:")
print(result)
```

---

## 多文件合并

在实际工作中，经常需要合并多个文件的数据。

### 批量读取CSV文件
```python
import glob
import os

# 创建模拟文件目录结构
files_data = {
    'data1.csv': pd.DataFrame({'id': [1, 2], 'value': [10, 20]}),
    'data2.csv': pd.DataFrame({'id': [3, 4], 'value': [30, 40]}),
    'data3.csv': pd.DataFrame({'id': [5, 6], 'value': [50, 60]})
}

# 保存模拟文件
os.makedirs('temp_data', exist_ok=True)
for filename, df in files_data.items():
    df.to_csv(f'temp_data/{filename}', index=False)

# 批量读取和合并
csv_files = glob.glob('temp_data/*.csv')
dataframes = []

for file in csv_files:
    df = pd.read_csv(file)
    dataframes.append(df)

# 合并所有数据
combined_data = pd.concat(dataframes, ignore_index=True)
print("合并后的数据:")
print(combined_data)
```

### 批量读取Excel文件
```python
import glob

# 创建模拟Excel文件
excel_data = {
    'sales.xlsx': pd.DataFrame({'month': ['Jan', 'Feb'], 'sales': [1000, 1200]}),
    'costs.xlsx': pd.DataFrame({'month': ['Jan', 'Feb'], 'costs': [800, 900]})
}

os.makedirs('temp_excel', exist_ok=True)
for filename, df in excel_data.items():
    df.to_excel(f'temp_excel/{filename}', index=False)

# 批量读取Excel
excel_files = glob.glob('temp_excel/*.xlsx')
excel_dfs = [pd.read_excel(file) for file in excel_files]

# 按月份合并
final_data = excel_dfs[0]
for df in excel_dfs[1:]:
    final_data = pd.merge(final_data, df, on='month', how='outer')

print("Excel文件合并结果:")
print(final_data)
```

### 多文件夹数据合并
```python
import os
from pathlib import Path

# 创建多层级目录结构
base_dir = Path('multi_level_data')
categories = ['electronics', 'clothing']
months = ['jan', 'feb']

for category in categories:
    for month in months:
        dir_path = base_dir / category / month
        dir_path.mkdir(parents=True, exist_ok=True)

        # 创建示例数据
        sample_data = pd.DataFrame({
            'product': [f'{category[:3]}_product_1', f'{category[:3]}_product_2'],
            'sales': np.random.randint(100, 1000, 2),
            'category': category,
            'month': month
        })
        sample_data.to_csv(dir_path / 'sales.csv', index=False)

# 递归读取所有CSV文件
all_data = []
for csv_file in base_dir.rglob('*.csv'):
    df = pd.read_csv(csv_file)
    all_data.append(df)

# 合并所有数据
complete_data = pd.concat(all_data, ignore_index=True)
print("多层级目录数据合并结果:")
print(complete_data.head())
```

### 性能优化技巧
```python
# 使用生成器减少内存占用
def read_files_generator(file_paths):
    """生成器方式读取文件，减少内存使用"""
    for file_path in file_paths:
        yield pd.read_csv(file_path)

# 使用生成器
csv_files = glob.glob('temp_data/*.csv')
data_generator = read_files_generator(csv_files)

# 使用chunksize处理大文件
chunk_size = 1000  # 根据实际情况调整
large_files = glob.glob('large_data/*.csv')

all_chunks = []
for file in large_files:
    for chunk in pd.read_csv(file, chunksize=chunk_size):
        all_chunks.append(chunk)

final_data = pd.concat(all_chunks, ignore_index=True)
```

---

## combine - 数据组合

### combine_first - 填充缺失值
```python
# 创建有缺失值的DataFrame
df1 = pd.DataFrame({
    'A': [1, np.nan, 3, 4],
    'B': [np.nan, 2, np.nan, 4]
})

df2 = pd.DataFrame({
    'A': [10, 20, 30, 40],
    'B': [100, np.nan, 300, np.nan]
})

print("DataFrame 1:")
print(df1)
print("\nDataFrame 2:")
print(df2)

# 使用combine_first填充缺失值
result = df1.combine_first(df2)
print("\ncombine_first结果:")
print(result)
```

### combine - 函数式组合
```python
# 创建示例数据
df1 = pd.DataFrame({
    'A': [1, 2, 3, 4],
    'B': [10, 20, 30, 40]
})

df2 = pd.DataFrame({
    'A': [5, 6, 7, 8],
    'B': [50, 60, 70, 80]
})

# 定义组合函数
def take_max(x, y):
    """取较大值"""
    return np.maximum(x, y)

def take_min(x, y):
    """取较小值"""
    return np.minimum(x, y)

# 使用combine进行元素级操作
result_max = df1.combine(df2, take_max)
print("取最大值结果:")
print(result_max)

result_min = df1.combine(df2, take_min)
print("\n取最小值结果:")
print(result_min)
```

### update - 更新数据
```python
# 创建基础数据
df1 = pd.DataFrame({
    'A': [1, 2, 3, 4],
    'B': [10, 20, 30, 40],
    'C': [100, 200, 300, 400]
})

df2 = pd.DataFrame({
    'A': [5, 6],
    'B': [50, np.nan],
    'C': [500, 600]
}, index=[1, 3])

print("原始DataFrame:")
print(df1)
print("\n更新DataFrame:")
print(df2)

# 使用update更新数据
df1.update(df2)
print("\nupdate后的结果:")
print(df1)
```

---

## compare - 数据比较

### DataFrame比较
```python
# 创建相似的DataFrame
df1 = pd.DataFrame({
    'A': [1, 2, 3, 4],
    'B': [10, 20, 30, 40],
    'C': [100, 200, 300, 400]
})

df2 = pd.DataFrame({
    'A': [1, 2, 4, 4],  # 第3行不同
    'B': [10, 25, 30, 45],  # 第2、4行不同
    'C': [100, 200, 300, 401]  # 第4行不同
}, index=[0, 1, 2, 3])

print("DataFrame 1:")
print(df1)
print("\nDataFrame 2:")
print(df2)

# 比较两个DataFrame
comparison = df1.compare(df2)
print("\nDataFrame比较结果:")
print(comparison)
```

### 数据一致性检查
```python
# 使用equals检查完全相同
are_equal = df1.equals(df2)
print(f"两个DataFrame完全相同: {are_equal}")

# 手动检查差异
differences = {}
for column in df1.columns:
    if column not in df2.columns:
        differences[column] = f"列 {column} 在df2中不存在"
        continue

    if not df1[column].equals(df2[column]):
        # 找出不同的行
        diff_mask = df1[column] != df2[column]
        diff_rows = df1.index[diff_mask].tolist()
        differences[column] = f"行 {diff_rows} 有差异"

print("\n手动检查差异:")
for col, diff in differences.items():
    print(f"{col}: {diff}")
```

### 统计差异
```python
def detailed_comparison(df1, df2):
    """详细的DataFrame比较"""
    comparison = df1.compare(df2)

    # 统计差异
    self_diff = comparison.xs('self', axis=1, level=1)
    other_diff = comparison.xs('other', axis=1, level=1)

    print("差异统计:")
    print(f"总差异单元格数: {comparison.size}")
    print(f"df1中的值:")
    print(self_diff.count())
    print(f"df2中的值:")
    print(other_diff.count())

    # 按列统计差异
    print("\n按列统计差异:")
    for column in comparison.columns.get_level_values(0).unique():
        col_diff = comparison[column]
        print(f"{column}: {col_diff.size} 个差异")

    return comparison

# 使用详细比较
detailed_comparison(df1, df2)
```

---

## 最佳实践和性能优化

### 1. 选择合适的合并方法

```python
# 场景1: 简单的行追加
# ❌ 错误：使用append（已弃用）
# result = df1.append(df2)

# ✅ 正确：使用concat
result = pd.concat([df1, df2], ignore_index=True)

# 场景2: 基于键值连接
# ✅ 使用merge
result = pd.merge(df1, df2, on='key', how='inner')

# 场景3: 基于索引连接
# ✅ 使用join
result = df1.join(df2, how='left')
```

### 2. 性能优化技巧

```python
# 处理大量小文件时
import glob
from concurrent.futures import ThreadPoolExecutor

def read_file(file_path):
    """读取单个文件"""
    return pd.read_csv(file_path)

# 使用多线程读取
files = glob.glob('data/*.csv')
with ThreadPoolExecutor(max_workers=4) as executor:
    dataframes = list(executor.map(read_file, files))

# 批量合并（避免多次concat）
result = pd.concat(dataframes, ignore_index=True)
```

### 3. 内存管理

```python
# 处理大文件时使用chunking
chunk_size = 10000
results = []

for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    # 处理每个chunk
    processed_chunk = process_chunk(chunk)  # 自定义处理函数
    results.append(processed_chunk)

# 合并结果
final_result = pd.concat(results, ignore_index=True)

# 及时清理内存
del results
import gc
gc.collect()
```

### 4. 数据类型优化

```python
# 在合并前优化数据类型
def optimize_dtypes(df):
    """优化DataFrame的数据类型"""
    for col in df.select_dtypes(include=['object']):
        if df[col].nunique() / len(df) < 0.5:  # 唯一值比例低于50%
            df[col] = df[col].astype('category')

    for col in df.select_dtypes(include=['int64']):
        if df[col].min() >= 0 and df[col].max() < 255:
            df[col] = df[col].astype('uint8')
        elif df[col].min() >= -128 and df[col].max() < 127:
            df[col] = df[col].astype('int8')

    return df

# 优化数据类型后再合并
df1_opt = optimize_dtypes(df1)
df2_opt = optimize_dtypes(df2)
result = pd.merge(df1_opt, df2_opt, on='key')
```

### 5. 错误处理

```python
def safe_merge(df1, df2, **kwargs):
    """安全的合并操作，包含错误处理"""
    try:
        result = pd.merge(df1, df2, **kwargs)
        return result
    except KeyError as e:
        print(f"连接键错误: {e}")
        print(f"df1的列: {df1.columns.tolist()}")
        print(f"df2的列: {df2.columns.tolist()}")
        return None
    except MemoryError:
        print("内存不足，尝试使用chunk处理")
        return None

# 使用安全合并
result = safe_merge(df1, df2, on='id', how='inner')
```

### 6. 数据验证

```python
def validate_merge(left_df, right_df, result_df, on, how='inner'):
    """验证合并结果的正确性"""
    print(f"原始左表行数: {len(left_df)}")
    print(f"原始右表行数: {len(right_df)}")
    print(f"合并结果行数: {len(result_df)}")

    # 检查连接键
    missing_left = set(left_df[on]) - set(result_df[on])
    missing_right = set(right_df[on]) - set(result_df[on])

    if missing_left:
        print(f"左表中未匹配的键: {missing_left}")
    if missing_right:
        print(f"右表中未匹配的键: {_right}")

    # 检查重复值
    duplicates = result_df.duplicated(subset=on).sum()
    if duplicates > 0:
        print(f"警告: 发现 {duplicates} 个重复连接键")

# 验证合并结果
validate_merge(df1, df2, result, on='id', how='inner')
```

---

## 总结

### 方法选择指南

| 方法 | 适用场景 | 优点 | 缺点 |
|------|----------|------|------|
| **concat** | 堆叠数据、简单合并 | 灵活、支持多轴 | 可能产生重复索引 |
| **merge** | 数据库式连接 | 功能强大、类型安全 | 语法较复杂 |
| **join** | 索引连接 | 简单易用 | 功能有限 |
| **combine_first** | 填充缺失值 | 处理缺失值方便 | 仅限元素级操作 |
| **compare** | 数据比较 | 详细差异分析 | 功能单一 |

### 性能排序（从快到慢）
1. **基于索引的操作** (join)
2. **基于列的操作** (merge)
3. **通用拼接** (concat)
4. **逐行操作** (避免使用)

### 最佳实践

1. **选择正确的方法**：根据数据结构和需求选择合适的合并方法
2. **注意内存使用**：处理大数据时使用chunking和生成器
3. **优化数据类型**：在合并前优化数据类型可以显著提升性能
4. **处理重复值**：注意连接键的重复问题
5. **验证结果**：合并后验证结果的正确性
6. **错误处理**：添加适当的错误处理和日志记录

掌握这些数据合并技术将使您能够高效地处理各种数据整合任务！