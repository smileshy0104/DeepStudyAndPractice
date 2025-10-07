# Pandas 高级操作完全指南：从数据处理大师到数据分析专家

## 概述

如果说 Pandas 是数据分析的"瑞士军刀"，那么这篇指南就是您的"武功秘籍"！我们将从数据输入输出开始，逐步掌握数据处理的所有高级技巧，让您成为真正的 Pandas 高手。

### 学习路径

1. **数据输入输出** - 让数据自由流动
2. **索引和选择** - 精准定位所需数据
3. **数据信息获取** - 深入了解数据特征
4. **统计分析** - 揭示数据背后的规律
5. **数据计算** - 进行各种数学运算
6. **数据筛选** - 找到真正需要的数据
7. **数据排序** - 让数据井然有序
8. **数据修改** - 灵活调整数据内容
9. **数据删除** - 清理不需要的数据
10. **数据迭代** - 遍历数据的艺术
11. **函数应用** - 批量处理的威力

## 1. 数据输入输出 (I/O 操作)

### 1.1 文件读写基础

```python
import pandas as pd
import numpy as np

print("=== Pandas 数据输入输出 ===")

# 创建示例数据
sample_data = pd.DataFrame({
    '姓名': ['张三', '李四', '王五', '赵六', '钱七'],
    '年龄': [25, 30, 35, 28, 32],
    '部门': ['技术', '销售', '技术', '市场', '技术'],
    '工资': [8000, 12000, 15000, 10000, 13000],
    '入职日期': pd.date_range('2024-01-01', periods=5)
})

print("示例数据:")
print(sample_data)

# === CSV 文件操作 ===
print("\n=== CSV 文件操作 ===")

# 保存为 CSV
csv_filename = 'employees.csv'
sample_data.to_csv(csv_filename, index=False, encoding='utf-8-sig')
print(f"已保存到 {csv_filename}")

# 读取 CSV
df_from_csv = pd.read_csv(csv_filename)
print("从 CSV 读取的数据:")
print(df_from_csv)

# 带参数的 CSV 操作
print("\n带参数的 CSV 操作:")
sample_data.to_csv('employees_with_index.csv', index=True, encoding='utf-8')
df_with_index = pd.read_csv('employees_with_index.csv', index_col=0)
print("带索引的 CSV 读取:")
print(df_with_index)

# === Excel 文件操作 ===
print("\n=== Excel 文件操作 ===")

# 保存为 Excel
excel_filename = 'employees.xlsx'
sample_data.to_excel(excel_filename, index=False, sheet_name='员工信息')
print(f"已保存到 {excel_filename}")

# 读取 Excel
df_from_excel = pd.read_excel(excel_filename, sheet_name='员工信息')
print("从 Excel 读取的数据:")
print(df_from_excel)

# 多工作表操作
print("\n多工作表操作:")
with pd.ExcelWriter('multi_sheet.xlsx') as writer:
    sample_data.to_excel(writer, sheet_name='基本信息', index=False)
    sample_data.groupby('部门')['工资'].sum().to_excel(writer, sheet_name='部门汇总')

# 读取多工作表
multi_sheets = pd.read_excel('multi_sheet.xlsx', sheet_name=['基本信息', '部门汇总'])
print("读取的多工作表:")
for sheet_name, df in multi_sheets.items():
    print(f"\n{sheet_name}:")
    print(df)
```

### 1.2 高级 I/O 操作

```python
# 高级 I/O 操作
print("\n=== 高级 I/O 操作 ===")

# === JSON 文件操作 ===
print("\n=== JSON 文件操作 ===")

# 转换为 JSON
json_data = sample_data.to_json(orient='records', force_ascii=False, indent=2)
print("JSON 格式数据:")
print(json_data)

# 保存 JSON
with open('employees.json', 'w', encoding='utf-8') as f:
    f.write(json_data)

# 读取 JSON
df_from_json = pd.read_json('employees.json', encoding='utf-8')
print("从 JSON 读取的数据:")
print(df_from_json)

# === 数据库操作 ===
print("\n=== 数据库操作 ===")

# 模拟数据库操作（需要 sqlite3）
import sqlite3

# 创建内存数据库
conn = sqlite3.connect(':memory:')

# 保存到数据库
sample_data.to_sql('employees', conn, index=False, if_exists='replace')
print("数据已保存到数据库")

# 从数据库读取
df_from_db = pd.read_sql('SELECT * FROM employees WHERE 工资 > 10000', conn)
print("从数据库读取的高薪员工:")
print(df_from_db)

# === 分块读写大数据 ===
print("\n=== 分块读写大数据 ===")

# 创建大数据集
large_data = pd.DataFrame({
    'ID': range(10000),
    '值': np.random.randn(10000),
    '类别': np.random.choice(['A', 'B', 'C', 'D'], 10000)
})

# 分块写入
chunk_size = 2000
for i, chunk in enumerate(np.array_split(large_data, len(large_data) // chunk_size)):
    chunk.to_csv(f'large_data_chunk_{i}.csv', index=False)
    print(f"已写入第 {i+1} 块数据，大小: {len(chunk)} 行")

# 分块读取
chunk_list = []
for chunk_file in ['large_data_chunk_0.csv', 'large_data_chunk_1.csv']:
    chunk = pd.read_csv(chunk_file)
    chunk_list.append(chunk)
    print(f"读取 {chunk_file}: {len(chunk)} 行")

combined_data = pd.concat(chunk_list, ignore_index=True)
print(f"合并后数据大小: {len(combined_data)} 行")
```

## 2. 索引和选择 (Indexing and Selection)

### 2.1 基础索引操作

```python
# 索引和选择操作
print("\n=== 索引和选择操作 ===")

# 创建示例数据
df = pd.DataFrame({
    '姓名': ['张三', '李四', '王五', '赵六', '钱七', '孙八'],
    '年龄': [25, 30, 35, 28, 32, 27],
    '部门': ['技术', '销售', '技术', '市场', '技术', '销售'],
    '工资': [8000, 12000, 15000, 10000, 13000, 11000],
    '城市': ['北京', '上海', '广州', '深圳', '杭州', '成都']
}, index=['emp1', 'emp2', 'emp3', 'emp4', 'emp5', 'emp6'])

print("示例数据:")
print(df)

# === 列选择 ===
print("\n=== 列选择 ===")

# 单列选择
print("单列选择 (df['姓名']):")
print(df['姓名'])

# 多列选择
print("\n多列选择 (df[['姓名', '年龄', '工资']]):")
print(df[['姓名', '年龄', '工资']])

# 使用 loc 和 iloc 选择列
print("使用 loc 选择列:")
print(df.loc[:, ['姓名', '工资']])

# === 行选择 ===
print("\n=== 行选择 ===")

# 使用位置索引 (iloc)
print("使用 iloc[2] (第3行):")
print(df.iloc[2])

print("使用 iloc[1:4] (第2-4行):")
print(df.iloc[1:4])

# 使用标签索引 (loc)
print("\n使用 loc['emp3'] (标签为emp3的行):")
print(df.loc['emp3'])

print("使用 loc['emp2':'emp5'] (标签范围):")
print(df.loc['emp2':'emp5'])

# === 混合选择 ===
print("\n=== 混合选择 ===")

# 选择特定行列
print("选择 emp2-emp4 的姓名和工资:")
print(df.loc['emp2':'emp4', ['姓名', '工资']])

print("使用 iloc 选择第2-4行的第1,3列:")
print(df.iloc[1:4, [0, 2]])
```

### 2.2 高级索引技巧

```python
# 高级索引技巧
print("\n=== 高级索引技巧 ===")

# === 条件索引 ===
print("=== 条件索引 ===")

# 单条件筛选
print("年龄大于30的员工:")
print(df[df['年龄'] > 30])

# 多条件筛选 (与)
print("\n年龄在25-30之间且部门为技术的员工:")
condition = (df['年龄'] >= 25) & (df['年龄'] <= 30) & (df['部门'] == '技术')
print(df[condition])

# 多条件筛选 (或)
print("\n部门为技术或销售的员工:")
print(df[(df['部门'] == '技术') | (df['部门'] == '销售')])

# 使用 isin 方法
print("\n城市在北京或上海的员工:")
print(df[df['城市'].isin(['北京', '上海'])])

# === 字符串条件索引 ===
print("\n=== 字符串条件索引 ===")

# 姓名包含'张'或'李'的员工
print("姓名包含'张'或'李'的员工:")
name_pattern = df['姓名'].str.contains('张|李')
print(df[name_pattern])

# 部门以'技'开头的员工
print("\n部门以'技'开头的员工:")
tech_dept = df['部门'].str.startswith('技')
print(df[tech_dept])

# === 使用 query 方法 ===
print("\n=== 使用 query 方法 ===")

print("使用 query 筛选:")
print(df.query("年龄 > 30 and 工资 > 10000"))

# 使用变量
min_age = 28
max_salary = 14000
print(f"\n使用变量筛选 (年龄 > {min_age} and 工资 < {max_salary}):")
print(df.query(f"年龄 > {min_age} and 工资 < {max_salary}"))
```

### 2.3 多级索引操作

```python
# 多级索引操作
print("\n=== 多级索引操作 ===")

# 创建多级索引数据
arrays = [
    ['2024', '2024', '2024', '2025', '2025', '2025'],
    ['Q1', 'Q2', 'Q3', 'Q1', 'Q2', 'Q3'],
    ['技术部', '销售部', '技术部', '市场部', '技术部', '销售部']
]
index = pd.MultiIndex.from_arrays(arrays, names=['年份', '季度', '部门'])

multi_df = pd.DataFrame({
    '收入': [100, 120, 110, 80, 130, 140],
    '支出': [60, 70, 65, 50, 75, 80],
    '利润': [40, 50, 45, 30, 55, 60]
}, index=index)

print("多级索引数据:")
print(multi_df)

# 访问多级索引
print("\n访问多级索引:")

# 选择特定年份
print("2024年的数据:")
print(multi_df.loc['2024'])

# 选择特定年份和季度
print("\n2024年Q2的数据:")
print(multi_df.loc[('2024', 'Q2')])

# 使用 xs 方法
print("\n使用 xs 选择所有Q1数据:")
print(multi_df.xs('Q1', level='季度'))

# 选择所有技术部数据
print("\n所有技术部数据:")
print(multi_df.xs('技术部', level='部门'))

# 多级索引切片
print("\n多级索引切片:")
print("2024年Q1-Q2的技术部数据:")
print(multi_df.loc[('2024', slice('Q1', 'Q2'), '技术部'), :])
```

## 3. 数据信息获取 (Data Information)

### 3.1 基础信息查看

```python
# 数据信息获取
print("\n=== 数据信息获取 ===")

# 使用之前的示例数据
print("=== 基础信息查看 ===")

# 基本形状信息
print("数据基本信息:")
print(f"数据形状: {df.shape}")
print(f"数据大小: {df.size}")
print(f"数据维度: {df.ndim}")

# 列信息
print("\n列信息:")
print("列名:", df.columns.tolist())
print("数据类型:")
print(df.dtypes)

# 索引信息
print("\n索引信息:")
print("索引:", df.index.tolist())
print("索引名称:", df.index.name)

# 数据概览
print("\n数据概览:")
print("前3行:")
print(df.head(3))

print("\n后3行:")
print(df.tail(3))

print("\n随机3行:")
print(df.sample(3, random_state=42))
```

### 3.2 详细统计信息

```python
# 详细统计信息
print("\n=== 详细统计信息 ===")

# 数值列统计
print("数值列统计信息:")
print(df.describe())

# 包含非数值列的统计
print("\n完整统计信息:")
print(df.describe(include='all'))

# 各列的唯一值
print("\n各列唯一值信息:")
for col in df.columns:
    unique_count = df[col].nunique()
    unique_values = df[col].unique() if unique_count <= 5 else f"共{unique_count}个唯一值"
    print(f"{col}: {unique_count}个唯一值")
    if unique_count <= 5:
        print(f"  值: {unique_values}")

# 数据类型统计
print("\n数据类型统计:")
print(df.dtypes.value_counts())

# 内存使用
print("\n内存使用信息:")
print(df.memory_usage(deep=True))

# 空值信息
print("\n空值信息:")
print(df.isnull().sum())

# 重复值信息
print("\n重复值信息:")
print(f"总重复行数: {df.duplicated().sum()}")
print(f"重复行索引: {df[df.duplicated()].index.tolist()}")
```

### 3.3 相关性和分布信息

```python
# 相关性和分布信息
print("\n=== 相关性和分布信息 ===")

# 创建数值数据用于相关性分析
numeric_df = pd.DataFrame({
    '数学': [85, 92, 78, 95, 88, 76, 90, 82, 87, 91],
    '物理': [88, 85, 82, 92, 86, 78, 88, 80, 85, 89],
    '化学': [82, 88, 80, 90, 84, 75, 85, 78, 83, 87],
    '英语': [90, 85, 88, 82, 91, 79, 87, 84, 89, 86]
}, index=['学生A', '学生B', '学生C', '学生D', '学生E', '学生F', '学生G', '学生H', '学生I', '学生J'])

print("学生成绩数据:")
print(numeric_df)

# 相关性矩阵
print("\n相关性矩阵:")
correlation_matrix = numeric_df.corr()
print(correlation_matrix)

# 相关系数热力图数据准备
print("\n最高相关系数对:")
# 找到最高的相关系数对（除了自相关）
max_corr = 0
max_pair = ('', '')
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        corr_val = correlation_matrix.iloc[i, j]
        if abs(corr_val) > abs(max_corr):
            max_corr = corr_val
            max_pair = (correlation_matrix.columns[i], correlation_matrix.columns[j])

print(f"最高相关性: {max_pair[0]} 和 {max_pair[1]}, 相关系数: {max_corr:.3f}")

# 分布统计
print("\n分布统计:")
for col in numeric_df.columns:
    series = numeric_df[col]
    print(f"\n{col}:")
    print(f"  均值: {series.mean():.2f}")
    print(f"  中位数: {series.median():.2f}")
    print(f"  标准差: {series.std():.2f}")
    print(f"  偏度: {series.skew():.3f}")
    print(f"  峰度: {series.kurtosis():.3f}")
    print(f"  四分位数: Q1={series.quantile(0.25):.2f}, Q3={series.quantile(0.75):.2f}")
```

## 4. 统计分析 (Statistics)

### 4.1 基础统计计算

```python
# 统计分析
print("\n=== 统计分析 ===")

# 使用之前的员工数据
print("=== 基础统计计算 ===")

# 单列统计
print("年龄统计:")
age_stats = df['年龄'].describe()
print(age_stats)

print("\n工资统计:")
salary_stats = df['工资'].describe()
print(salary_stats)

# 自定义统计函数
def custom_stats(series):
    """自定义统计函数"""
    return pd.Series({
        '计数': series.count(),
        '总和': series.sum(),
        '均值': series.mean(),
        '中位数': series.median(),
        '众数': series.mode().iloc[0] if not series.mode().empty else np.nan,
        '标准差': series.std(),
        '方差': series.var(),
        '最小值': series.min(),
        '最大值': series.max(),
        '极差': series.max() - series.min(),
        '四分位距': series.quantile(0.75) - series.quantile(0.25),
        '变异系数': series.std() / series.mean() if series.mean() != 0 else 0
    })

print("\n年龄自定义统计:")
print(custom_stats(df['年龄']))

print("\n工资自定义统计:")
print(custom_stats(df['工资']))
```

### 4.2 分组统计

```python
# 分组统计
print("\n=== 分组统计 ===")

# 按部门分组统计
print("按部门分组统计:")
dept_stats = df.groupby('部门').agg({
    '年龄': ['count', 'mean', 'std'],
    '工资': ['mean', 'median', 'min', 'max', 'std']
}).round(2)

print(dept_stats)

# 按城市分组统计
print("\n按城市分组统计:")
city_stats = df.groupby('城市').agg({
    '姓名': 'count',
    '年龄': ['mean', 'min', 'max'],
    '工资': ['mean', 'sum']
}).round(2)
city_stats.columns = ['员工数', '平均年龄', '最小年龄', '最大年龄', '平均工资', '总工资']
print(city_stats)

# 多级分组统计
print("\n多级分组统计 (部门 + 年龄段):")
df['年龄段'] = pd.cut(df['年龄'], bins=[20, 25, 30, 35, 40], labels=['20-25', '25-30', '30-35', '35-40'])
multi_group = df.groupby(['部门', '年龄段']).agg({
    '姓名': 'count',
    '工资': 'mean'
}).round(2)
multi_group.columns = ['员工数', '平均工资']
print(multi_group)

# 透视表统计
print("\n透视表统计:")
pivot_table = pd.pivot_table(df,
                            values='工资',
                            index='部门',
                            columns='城市',
                            aggfunc='mean',
                            fill_value=0)
print(pivot_table.round(2))
```

### 4.3 高级统计分析

```python
# 高级统计分析
print("\n=== 高级统计分析 ===")

# 创建更多示例数据
np.random.seed(42)
extended_df = pd.DataFrame({
    '产品类别': np.random.choice(['A', 'B', 'C'], 100),
    '销售额': np.random.normal(1000, 200, 100),
    '成本': np.random.normal(600, 100, 100),
    '客户满意度': np.random.uniform(3.0, 5.0, 100),
    '员工ID': np.random.choice(['E001', 'E002', 'E003', 'E004'], 100)
})

# 计算利润率
extended_df['利润率'] = (extended_df['销售额'] - extended_df['成本']) / extended_df['销售额'] * 100

print("扩展数据样本:")
print(extended_df.head())

# 按产品类别的详细统计
print("\n按产品类别的详细统计:")
detailed_stats = extended_df.groupby('产品类别').agg({
    '销售额': ['count', 'mean', 'std', 'min', 'max'],
    '成本': ['mean', 'std'],
    '利润率': ['mean', 'std'],
    '客户满意度': ['mean', 'std']
}).round(2)

print(detailed_stats)

# 置信区间计算
print("\n95%置信区间计算:")
def confidence_interval(series, confidence=0.95):
    """计算置信区间"""
    import scipy.stats as stats
    n = len(series)
    mean = series.mean()
    std_err = stats.sem(series)
    h = std_err * stats.t.ppf((1 + confidence) / 2, n-1)
    return (mean - h, mean + h)

for category in extended_df['产品类别'].unique():
    sales_data = extended_df[extended_df['产品类别'] == category]['销售额']
    ci_lower, ci_upper = confidence_interval(sales_data)
    print(f"{category} 销售额 95% 置信区间: ({ci_lower:.2f}, {ci_upper:.2f})")

# 异常值检测
print("\n异常值检测 (IQR 方法):")
def detect_outliers(series):
    """使用IQR方法检测异常值"""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    return outliers

for category in extended_df['产品类别'].unique():
    outliers = detect_outliers(extended_df[extended_df['产品类别'] == category]['销售额'])
    print(f"{category} 销售额异常值: {len(outliers)} 个")
    if len(outliers) > 0:
        print(f"  异常值: {outliers.values}")
```

## 5. 数据计算 (Computations)

### 5.1 数学运算

```python
# 数据计算
print("\n=== 数据计算 ===")

# 创建数值数据
calc_df = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [10, 20, 30, 40, 50],
    'C': [100, 200, 300, 400, 500]
})

print("基础数据:")
print(calc_df)

# 基础数学运算
print("\n=== 基础数学运算 ===")

print("加法 (A + B):")
print(calc_df['A'] + calc_df['B'])

print("\n乘法 (A * B):")
print(calc_df['A'] * calc_df['B'])

print("\n除法 (C / A):")
print(calc_df['C'] / calc_df['A'])

print("\n幂运算 (A ** 2):")
print(calc_df['A'] ** 2)

# 广播运算
print("\n=== 广播运算 ===")

print("每列加上标量 10:")
print(calc_df + 10)

print("\n每列乘以标量 2:")
print(calc_df * 2)

# 数学函数应用
print("\n=== 数学函数应用 ===")

import math

# 对数运算
print("自然对数 ln(A+1):")
print(np.log(calc_df['A'] + 1))

print("\n以10为底的对数 log10(B):")
print(np.log10(calc_df['B']))

# 三角函数
angles = pd.Series([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2])
print("\n角度和三角函数:")
trig_df = pd.DataFrame({
    '角度(弧度)': angles,
    '正弦': np.sin(angles),
    '余弦': np.cos(angles),
    '正切': np.tan(angles)
})
print(trig_df.round(4))
```

### 5.2 字符串运算

```python
# 字符串运算
print("\n=== 字符串运算 ===")

# 创建字符串数据
str_df = pd.DataFrame({
    '姓名': ['张三', '李四', '王五', '赵六'],
    '邮箱': ['zhang@email.com', 'li@email.com', 'wang@email.com', 'zhao@email.com'],
    '地址': ['北京市朝阳区', '上海市浦东新区', '广州市天河区', '深圳市南山区']
})

print("字符串数据:")
print(str_df)

# 字符串操作
print("\n=== 字符串操作 ===")

# 大小写转换
print("姓名大写:")
print(str_df['姓名'].str.upper())

print("\n邮箱小写:")
print(str_df['邮箱'].str.lower())

# 字符串长度
print("\n姓名长度:")
print(str_df['姓名'].str.len())

# 字符串分割
print("\n地址分割 (城市和区域):")
address_split = str_df['地址'].str.split('市', expand=True)
address_split.columns = ['城市', '区域']
print(address_split)

# 字符串替换
print("\n邮箱域名替换:")
print(str_df['邮箱'].str.replace('@email.com', '@company.com'))

# 字符串查找和提取
print("\n提取城市名称:")
print(str_df['地址'].str.extract(r'(\w+)[市区]'))

# 字符串条件判断
print("\n邮箱是否包含 'li':")
print(str_df['邮箱'].str.contains('li'))

print("\n姓名长度是否大于2:")
print(str_df['姓名'].str.len() > 2)
```

### 5.3 日期时间运算

```python
# 日期时间运算
print("\n=== 日期时间运算 ===")

# 创建日期时间数据
date_df = pd.DataFrame({
    '项目': ['项目A', '项目B', '项目C', '项目D'],
    '开始日期': pd.date_range('2024-01-01', periods=4),
    '结束日期': pd.date_range('2024-02-01', periods=4),
    '工时': [160, 200, 180, 220]
})

print("日期时间数据:")
print(date_df)

# 日期时间计算
print("\n=== 日期时间计算 ===")

# 计算项目持续时间
date_df['持续时间'] = date_df['结束日期'] - date_df['开始日期']
print("项目持续时间:")
print(date_df[['项目', '持续时间']])

# 计算工作日（简单估算）
date_df['工作日'] = date_df['持续时间'].dt.days * 5/7  # 假设每周5个工作日
print("\n估算工作日:")
print(date_df[['项目', '工作日']])

# 日期时间组件提取
print("\n=== 日期时间组件提取 ===")

date_df['开始年份'] = date_df['开始日期'].dt.year
date_df['开始月份'] = date_df['开始日期'].dt.month
date_df['开始季度'] = date_df['开始日期'].dt.quarter
date_df['开始星期'] = date_df['开始日期'].dt.day_name()

print("添加时间组件后:")
print(date_df[['项目', '开始年份', '开始月份', '开始季度', '开始星期']])

# 日期时间运算
print("\n=== 日期时间运算 ===")

# 计算效率 (工时/工作日)
date_df['效率'] = date_df['工时'] / date_df['工作日']
print("项目效率:")
print(date_df[['项目', '效率']].round(2))

# 日期偏移
print("\n日期偏移 (开始日期后30天):")
date_df['30天后'] = date_df['开始日期'] + pd.Timedelta(days=30)
print(date_df[['项目', '开始日期', '30天后']])
```

## 6. 数据筛选 (Selecting Data)

### 6.1 条件筛选

```python
# 数据筛选
print("\n=== 数据筛选 ===")

# 使用之前的员工数据
print("=== 条件筛选 ===")

# 基础条件筛选
print("年龄大于30的员工:")
high_age = df[df['年龄'] > 30]
print(high_age)

print("\n工资在10000-14000之间的员工:")
salary_range = df[(df['工资'] >= 10000) & (df['工资'] <= 14000)]
print(salary_range)

print("\n部门为技术且年龄小于30的员工:")
tech_young = df[(df['部门'] == '技术') & (df['年龄'] < 30)]
print(tech_young)

# 使用 isin 筛选
print("\n=== 使用 isin 筛选 ===")

cities = ['北京', '上海', '深圳']
print(f"城市在 {cities} 中的员工:")
city_filtered = df[df['城市'].isin(cities)]
print(city_filtered)

print("\n部门为技术或销售的员工:")
dept_filtered = df[df['部门'].isin(['技术', '销售'])]
print(dept_filtered)

# 使用 isin 的反向操作
print("\n部门不为技术和销售的员工:")
other_dept = df[~df['部门'].isin(['技术', '销售'])]
print(other_dept)
```

### 6.2 字符串筛选

```python
# 字符串筛选
print("\n=== 字符串筛选 ===")

# 姓名包含特定字符
print("姓名包含'张'或'李'的员工:")
name_filtered = df[df['姓名'].str.contains('张|李')]
print(name_filtered)

# 城市以特定字符开头
print("\n城市以'北'开头的员工:")
north_city = df[df['城市'].str.startswith('北')]
print(north_city)

# 城市以特定字符结尾
print("\n城市以'海'结尾的员工:")
sea_city = df[df['城市'].str.endswith('海')]
print(sea_city)

# 正则表达式筛选
print("\n使用正则表达式筛选:")
print("姓名包含数字的员工 (模拟):")
# 模拟一些包含数字的姓名
df_with_nums = df.copy()
df_with_nums.loc['emp7'] = ['员工1号', 29, '技术', 11500, '南京']
num_names = df_with_nums[df_with_nums['姓名'].str.contains(r'\d')]
print(num_names)
```

### 6.3 复杂筛选

```python
# 复杂筛选
print("\n=== 复杂筛选 ===")

# 多条件组合
print("复杂条件筛选:")
complex_condition = (
    (df['年龄'] >= 25) &
    (df['年龄'] <= 35) &
    (df['工资'] > 10000) &
    (df['部门'].isin(['技术', '销售']))
)
complex_filtered = df[complex_condition]
print(complex_filtered)

# 使用 query 方法
print("\n使用 query 方法:")
query_result = df.query("25 <= 年龄 <= 35 and 工资 > 10000 and 部门 in ['技术', '销售']")
print(query_result)

# 动态查询
print("\n动态查询:")
def dynamic_query(dataframe, conditions):
    """动态构建查询条件"""
    query_parts = []

    for column, operator, value in conditions:
        if operator in ['==', '!=', '>', '<', '>=', '<=']:
            if isinstance(value, str):
                query_parts.append(f"{column} {operator} '{value}'")
            else:
                query_parts.append(f"{column} {operator} {value}")
        elif operator == 'in':
            if isinstance(value, list):
                value_str = str(value).replace("'", '"')
                query_parts.append(f"{column} in {value_str}")
        elif operator == 'contains':
            query_parts.append(f"{column}.str.contains('{value}')")

    query_string = ' and '.join(query_parts)
    return dataframe.query(query_string)

# 使用动态查询
conditions = [
    ('部门', 'in', ['技术', '销售']),
    ('工资', '>', 10000),
    ('年龄', '<=', 32)
]

dynamic_result = dynamic_query(df, conditions)
print("动态查询结果:")
print(dynamic_result)

# 自定义筛选函数
print("\n=== 自定义筛选函数 ===")

def smart_filter(dataframe, filters):
    """智能筛选函数"""
    result = dataframe.copy()

    for column, filter_config in filters.items():
        filter_type = filter_config.get('type', 'equals')
        value = filter_config.get('value')

        if filter_type == 'equals':
            result = result[result[column] == value]
        elif filter_type == 'range':
            min_val, max_val = value
            result = result[(result[column] >= min_val) & (result[column] <= max_val)]
        elif filter_type == 'in_list':
            result = result[result[column].isin(value)]
        elif filter_type == 'contains':
            result = result[result[column].str.contains(value, na=False)]
        elif filter_type == 'greater_than':
            result = result[result[column] > value]
        elif filter_type == 'less_than':
            result = result[result[column] < value]

    return result

# 使用自定义筛选函数
filter_configs = {
    '部门': {'type': 'in_list', 'value': ['技术', '销售']},
    '年龄': {'type': 'range', 'value': [25, 35]},
    '工资': {'type': 'greater_than', 'value': 10000}
}

smart_result = smart_filter(df, filter_configs)
print("智能筛选结果:")
print(smart_result)
```

## 7. 数据排序 (Sorting)

### 7.1 基础排序

```python
# 数据排序
print("\n=== 数据排序 ===")

# 使用示例数据
print("=== 基础排序 ===")

# 按单列排序
print("按年龄升序排序:")
age_sorted = df.sort_values('年龄')
print(age_sorted)

print("\n按工资降序排序:")
salary_sorted = df.sort_values('工资', ascending=False)
print(salary_sorted)

# 按多列排序
print("\n按部门升序，工资降序排序:")
multi_sorted = df.sort_values(['部门', '工资'], ascending=[True, False])
print(multi_sorted)
```

### 7.2 高级排序

```python
# 高级排序
print("\n=== 高级排序 ===")

# 按索引排序
print("按索引降序排序:")
index_sorted = df.sort_index(ascending=False)
print(index_sorted)

# 按多个条件排序，并保持原始顺序
print("\n稳定排序 (保持原始顺序):")
stable_sorted = df.sort_values(['部门', '年龄'], kind='stable')
print(stable_sorted)

# 自定义排序
print("\n自定义排序:")
df_sorted = df.copy()
# 自定义排序规则：技术部优先，然后按年龄排序
dept_priority = {'技术': 1, '销售': 2, '市场': 3}
df_sorted['部门优先级'] = df_sorted['部门'].map(dept_priority)
custom_sorted = df_sorted.sort_values(['部门优先级', '年龄'])
print(custom_sorted[['姓名', '部门', '年龄', '部门优先级']])

# 按字符串长度排序
print("\n按姓名长度排序:")
name_length_sorted = df.copy()
name_length_sorted['姓名长度'] = name_length_sorted['姓名'].str.len()
print(name_length_sorted.sort_values('姓名长度')[['姓名', '姓名长度']])
```

### 7.3 排名和排序

```python
# 排名和排序
print("\n=== 排名和排序 ===")

# 创建带分数的数据
score_df = pd.DataFrame({
    '姓名': ['张三', '李四', '王五', '赵六', '钱七', '孙八'],
    '数学': [85, 92, 78, 95, 88, 76],
    '英语': [90, 85, 88, 82, 91, 79],
    '物理': [88, 89, 92, 85, 87, 83]
})

print("成绩数据:")
print(score_df)

# 计算总分
score_df['总分'] = score_df[['数学', '英语', '物理']].sum(axis=1)
print("\n添加总分后:")
print(score_df)

# 不同排名方法
print("\n=== 不同排名方法 ===")

# 默认排名 (平均排名)
score_df['总分排名'] = score_df['总分'].rank(method='average', ascending=False)
print("平均排名:")
print(score_df[['姓名', '总分', '总分排名']])

# 最小排名
score_df['最小排名'] = score_df['总分'].rank(method='min', ascending=False)
print("\n最小排名:")
print(score_df[['姓名', '总分', '最小排名']])

# 密集排名
score_df['密集排名'] = score_df['总分'].rank(method='dense', ascending=False)
print("\n密集排名:")
print(score_df[['姓名', '总分', '密集排名']])

# 按多列排名
print("\n=== 按多列排名 ===")

# 先按数学排名，数学相同时按英语排名
score_df['综合排名'] = score_df[['数学', '英语']].apply(
    lambda x: (x['数学'], x['英语']), axis=1
).rank(method='min', ascending=False)

print("按数学和英语综合排名:")
print(score_df[['姓名', '数学', '英语', '综合排名']].sort_values('综合排名'))
```

## 8. 数据修改 (Data Modification)

### 8.1 基础修改操作

```python
# 数据修改
print("\n=== 数据修改 ===")

# 复制数据用于修改
df_modified = df.copy()

print("=== 基础修改操作 ===")

# 修改单个值
print("修改前 emp2 的年龄:")
print(f"emp2 年龄: {df_modified.loc['emp2', '年龄']}")

df_modified.loc['emp2', '年龄'] = 31
print(f"修改后 emp2 年龄: {df_modified.loc['emp2', '年龄']}")

# 修改整列
print("\n给所有人涨薪10%:")
df_modified['工资'] = df_modified['工资'] * 1.1
print(df_modified[['姓名', '工资']])

# 修改满足条件的值
print("\n给技术部门员工额外涨薪5%:")
tech_mask = df_modified['部门'] == '技术'
df_modified.loc[tech_mask, '工资'] = df_modified.loc[tech_mask, '工资'] * 1.05
print(df_modified[['姓名', '部门', '工资']])
```

### 8.2 高级修改操作

```python
# 高级修改操作
print("\n=== 高级修改操作 ===")

# 使用 map 方法修改
print("使用 map 修改部门名称:")
dept_mapping = {
    '技术': '研发部',
    '销售': '市场部',
    '市场': '营销部'
}
df_modified['部门'] = df_modified['部门'].map(dept_mapping)
print(df_modified[['姓名', '部门']])

# 使用 apply 方法修改
print("\n使用 apply 修改工资等级:")
def salary_grade(salary):
    if salary < 10000:
        return '初级'
    elif salary < 15000:
        return '中级'
    else:
        return '高级'

df_modified['工资等级'] = df_modified['工资'].apply(salary_grade)
print(df_modified[['姓名', '工资', '工资等级']])

# 使用 replace 方法
print("\n使用 replace 修改城市名称:")
city_mapping = {
    '北京': '北京市',
    '上海': '上海市',
    '广州': '广州市',
    '深圳': '深圳市',
    '杭州': '杭州市',
    '成都': '成都市'
}
df_modified['城市'] = df_modified['城市'].replace(city_mapping)
print(df_modified[['姓名', '城市']])

# 条件修改
print("\n条件修改:")
# 年龄大于30的员工标记为'资深'
df_modified['经验等级'] = '普通'
df_modified.loc[df_modified['年龄'] > 30, '经验等级'] = '资深'
print(df_modified[['姓名', '年龄', '经验等级']])
```

### 8.3 批量修改

```python
# 批量修改
print("\n=== 批量修改 ===")

# 批量添加新列
print("批量添加新列:")
df_modified['入职年份'] = df_modified['入职日期'].dt.year
df_modified['工作年限'] = 2024 - df_modified['入职年份']
df_modified['绩效工资'] = df_modified['工资'] * 0.2
df_modified['总薪酬'] = df_modified['工资'] + df_modified['绩效工资']

print(df_modified[['姓名', '入职年份', '工作年限', '绩效工资', '总薪酬']])

# 根据其他列批量修改
print("\n根据其他列批量修改:")
# 根据工作年限和部门计算奖金
def calculate_bonus(row):
    base_bonus = row['总薪酬'] * 0.1
    experience_bonus = row['工作年限'] * 200
    dept_bonus = {'研发部': 1000, '市场部': 800, '营销部': 600}

    return base_bonus + experience_bonus + dept_bonus.get(row['部门'], 0)

df_modified['奖金'] = df_modified.apply(calculate_bonus, axis=1)
print(df_modified[['姓名', '部门', '工作年限', '奖金']])

# 使用 eval 进行动态计算
print("\n使用 eval 进行动态计算:")
df_modified['税前收入'] = df_modified.eval('总薪酬 + 奖金')
df_modified['税率'] = df_modified['税前收入'].apply(lambda x: 0.1 if x < 15000 else 0.15)
df_modified['税后收入'] = df_modified.eval('税前收入 * (1 - 税率)')

print(df_modified[['姓名', '税前收入', '税率', '税后收入']])
```

## 9. 数据删除 (Dropping Data)

### 9.1 删除行和列

```python
# 数据删除
print("\n=== 数据删除 ===")

# 创建包含需要删除数据的数据框
df_drop = df.copy()

print("=== 删除行和列 ===")

# 删除行
print("删除 emp3 和 emp5:")
df_drop_rows = df_drop.drop(['emp3', 'emp5'])
print(df_drop_rows)

# 按条件删除行
print("\n删除年龄大于30的员工:")
df_drop_filtered = df_drop.drop(df_drop[df_drop['年龄'] > 30].index)
print(df_drop_filtered)

# 删除列
print("\n删除城市列:")
df_drop_cols = df_drop.drop('城市', axis=1)
print(df_drop_cols)

# 删除多列
print("\n删除入职日期和城市列:")
df_drop_multi_cols = df_drop.drop(['入职日期', '城市'], axis=1)
print(df_drop_multi_cols)
```

### 9.2 高级删除操作

```python
# 高级删除操作
print("\n=== 高级删除操作 ===")

# 删除重复行
df_with_duplicates = pd.concat([df, df.iloc[2:4]], ignore_index=True)
print("包含重复行的数据:")
print(df_with_duplicates)

print("\n删除重复行 (保留第一个):")
df_no_duplicates = df_with_duplicates.drop_duplicates()
print(df_no_duplicates)

print("\n删除重复行 (保留最后一个):")
df_no_duplicates_last = df_with_duplicates.drop_duplicates(keep='last')
print(df_no_duplicates_last)

# 按特定列删除重复
print("\n按部门删除重复 (保留第一个):")
df_dept_unique = df.drop_duplicates(subset=['部门'], keep='first')
print(df_dept_unique[['姓名', '部门']])

# 删除空值
print("\n=== 删除空值 ===")

# 创建包含空值的数据
df_with_nulls = df.copy()
df_with_nulls.loc['emp1', '年龄'] = np.nan
df_with_nulls.loc['emp3', '工资'] = np.nan
df_with_nulls.loc['emp5', '城市'] = np.nan

print("包含空值的数据:")
print(df_with_nulls)

print("\n删除包含空值的行:")
df_no_nulls = df_with_nulls.dropna()
print(df_no_nulls)

print("\n删除特定列为空的行:")
df_no_nulls_subset = df_with_nulls.dropna(subset=['年龄', '工资'])
print(df_no_nulls_subset)

# 删除全为空的行
print("\n删除全为空的行:")
df_no_all_nulls = df_with_nulls.dropna(how='all')
print(df_no_all_nulls)

# 删除空值列
print("\n删除空值列:")
df_no_null_cols = df_with_nulls.dropna(axis=1, how='all')
print(df_no_null_cols.columns.tolist())
```

### 9.3 条件删除

```python
# 条件删除
print("\n=== 条件删除 ===")

# 创建扩展数据
df_extended = pd.DataFrame({
    'ID': range(1, 11),
    '姓名': [f'员工{i}' for i in range(1, 11)],
    '年龄': [25, 30, 35, 28, 32, 45, 22, 38, 50, 27],
    '工资': [8000, 12000, 15000, 10000, 13000, 18000, 7000, 14000, 20000, 9000],
    '部门': ['技术', '销售', '技术', '市场', '技术', '管理', '技术', '销售', '管理', '市场'],
    '状态': ['在职', '在职', '离职', '在职', '在职', '在职', '离职', '在职', '在职', '离职']
})

print("扩展数据:")
print(df_extended)

# 删除离职员工
print("\n删除离职员工:")
active_employees = df_extended.drop(df_extended[df_extended['状态'] == '离职'].index)
print(active_employees)

# 删除年龄过大的员工 (大于45岁)
print("\n删除年龄大于45岁的员工:")
age_filtered = df_extended.drop(df_extended[df_extended['年龄'] > 45].index)
print(age_filtered)

# 删除工资过低的员工 (小于8000)
print("\n删除工资小于8000的员工:")
salary_filtered = df_extended.drop(df_extended[df_extended['工资'] < 8000].index)
print(salary_filtered)

# 组合条件删除
print("\n组合条件删除 (年龄>40或工资>15000):")
complex_filter = (df_extended['年龄'] > 40) | (df_extended['工资'] > 15000)
complex_filtered = df_extended.drop(df_extended[complex_filter].index)
print(complex_filtered)

# 使用 query 删除
print("\n使用 query 删除年龄在25-30之外的员工:")
query_filtered = df_extended.query("25 <= 年龄 <= 30")
print(query_filtered)
```

## 10. 数据迭代 (Data Iteration)

### 10.1 基础迭代方法

```python
# 数据迭代
print("\n=== 数据迭代 ===")

# 使用示例数据
print("=== 基础迭代方法 ===")

# iterrows() - 逐行迭代 (慢但不推荐)
print("使用 iterrows() 逐行迭代:")
count = 0
for index, row in df.iterrows():
    if count < 3:  # 只显示前3行
        print(f"索引: {index}, 姓名: {row['姓名']}, 年龄: {row['年龄']}")
        count += 1
    else:
        break

# itertuples() - 更快的行迭代
print("\n使用 itertuples() 逐行迭代:")
count = 0
for row in df.itertuples():
    if count < 3:  # 只显示前3行
        print(f"索引: {row.Index}, 姓名: {row.姓名}, 年龄: {row.年龄}")
        count += 1
    else:
        break

# items() - 按列迭代
print("\n使用 items() 按列迭代:")
for column_name, column_data in df.items():
    print(f"列名: {column_name}, 数据类型: {column_data.dtype}")
    if column_name in ['姓名', '年龄']:  # 只显示前两列
        print(f"  数据: {column_data.tolist()}")
        break
```

### 10.2 高效迭代方法

```python
# 高效迭代方法
print("\n=== 高效迭代方法 ===")

# 使用 apply 进行高效操作
print("使用 apply 计算工资等级:")
def get_salary_grade(salary):
    if salary < 10000:
        return 'C'
    elif salary < 13000:
        return 'B'
    else:
        return 'A'

df['工资等级'] = df['工资'].apply(get_salary_grade)
print(df[['姓名', '工资', '工资等级']])

# 使用向量化操作替代迭代
print("\n使用向量化操作:")
# 计算BMI指数
bmi_data = pd.DataFrame({
    '姓名': ['张三', '李四', '王五'],
    '体重(kg)': [70, 80, 65],
    '身高(m)': [1.75, 1.80, 1.70]
})

# 慢方法 (迭代)
def calculate_bmi_slow(df):
    bmi_list = []
    for _, row in df.iterrows():
        bmi = row['体重(kg)'] / (row['身高(m)'] ** 2)
        bmi_list.append(round(bmi, 2))
    return bmi_list

# 快方法 (向量化)
def calculate_bmi_fast(df):
    return (df['体重(kg)'] / (df['身高(m)'] ** 2)).round(2)

print("BMI数据:")
print(bmi_data)

bmi_slow = calculate_bmi_slow(bmi_data)
bmi_fast = calculate_bmi_fast(bmi_data)

print(f"迭代方法: {bmi_slow}")
print(f"向量化方法: {bmi_fast.tolist()}")
print(f"结果相同: {bmi_slow == bmi_fast.tolist()}")
```

### 10.3 条件迭代

```python
# 条件迭代
print("\n=== 条件迭代 ===")

# 创建条件数据
condition_df = pd.DataFrame({
    '产品': ['A', 'B', 'C', 'D', 'E'],
    '销量': [100, 200, 50, 300, 150],
    '价格': [10, 20, 30, 15, 25]
})

print("产品数据:")
print(condition_df)

# 使用 loc 进行条件赋值
print("\n使用 loc 进行条件赋值:")
condition_df.loc[condition_df['销量'] > 150, '状态'] = '热销'
condition_df.loc[condition_df['销量'] <= 150, '状态'] = '一般'
print(condition_df)

# 使用 where 方法
print("\n使用 where 方法:")
condition_df['调整价格'] = condition_df['价格'].where(
    condition_df['销量'] > 100,
    condition_df['价格'] * 0.9
)
print(condition_df)

# 使用 mask 方法 (where 的反向)
print("\n使用 mask 方法:")
condition_df['高价标记'] = condition_df['价格'].mask(
    condition_df['价格'] < 20,
    '低价'
).mask(
    condition_df['价格'] >= 20,
    '高价'
)
print(condition_df)
```

## 11. 函数应用 (Function Application)

### 11.1 apply 方法

```python
# 函数应用
print("\n=== 函数应用 ===")

# 创建示例数据
func_df = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [10, 20, 30, 40, 50],
    'C': [100, 200, 300, 400, 500]
})

print("示例数据:")
print(func_df)

# === apply 方法详解 ===
print("\n=== apply 方法详解 ===")

# 对列应用函数
print("对每列求和:")
col_sum = func_df.apply(lambda x: x.sum())
print(col_sum)

print("\n对每列求均值:")
col_mean = func_df.apply(lambda x: x.mean())
print(col_mean)

# 对行应用函数
print("\n对每行求和:")
row_sum = func_df.apply(lambda x: x.sum(), axis=1)
print(row_sum)

print("\n对每行求均值:")
row_mean = func_df.apply(lambda x: x.mean(), axis=1)
print(row_mean)

# 应用自定义函数
print("\n应用自定义函数:")
def custom_operation(row):
    """自定义行操作"""
    return row['A'] * row['B'] + row['C']

func_df['计算结果'] = func_df.apply(custom_operation, axis=1)
print(func_df)
```

### 11.2 applymap 方法

```python
# applymap 方法
print("\n=== applymap 方法 ===")

# applymap 对每个元素应用函数
print("对每个元素应用平方根函数:")
sqrt_data = func_df.applymap(lambda x: x ** 0.5)
print(sqrt_data.round(3))

print("\n对每个元素应用条件函数:")
def conditional_value(x):
    if x < 100:
        return x * 2
    elif x < 300:
        return x * 1.5
    else:
        return x

conditional_data = func_df.applymap(conditional_value)
print(conditional_data)
```

### 11.3 pipe 方法

```python
# pipe 方法
print("\n=== pipe 方法 ===")

# pipe 方法可以链式应用多个函数
print("使用 pipe 进行链式操作:")

def add_prefix(df, prefix):
    """添加前缀到列名"""
    df.columns = [prefix + col for col in df.columns]
    return df

def multiply_by_factor(df, factor):
    """所有数值乘以因子"""
    return df * factor

def add_constant(df, constant):
    """所有数值加上常数"""
    return df + constant

# 链式应用函数
result = (func_df
          .pipe(multiply_by_factor, 2)
          .pipe(add_constant, 10)
          .pipe(add_prefix, 'processed_'))

print("链式处理结果:")
print(result)

# 复杂的 pipe 操作
print("\n复杂的 pipe 操作:")
def data_processing_pipeline(df):
    """数据处理管道"""
    # 步骤1: 标准化
    df_normalized = (df - df.mean()) / df.std()

    # 步骤2: 计算每行的总和
    df_normalized['row_sum'] = df_normalized.sum(axis=1)

    # 步骤3: 按总和排序
    df_sorted = df_normalized.sort_values('row_sum', ascending=False)

    return df_sorted

processed_result = func_df.pipe(data_processing_pipeline)
print("处理管道结果:")
print(processed_result.round(3))
```

### 11.4 transform 方法

```python
# transform 方法
print("\n=== transform 方法 ===")

# 创建分组数据
group_df = pd.DataFrame({
    '部门': ['技术', '销售', '技术', '市场', '技术', '销售', '市场', '技术'],
    '姓名': ['张三', '李四', '王五', '赵六', '钱七', '孙八', '周九', '吴十'],
    '工资': [8000, 12000, 15000, 10000, 13000, 11000, 9000, 14000]
})

print("分组数据:")
print(group_df)

# transform 与 groupby 结合
print("\n使用 transform 计算部门平均工资:")
group_df['部门平均工资'] = group_df.groupby('部门')['工资'].transform('mean')
print(group_df[['姓名', '部门', '工资', '部门平均工资']])

print("\n计算工资与部门平均的差异:")
group_df['工资差异'] = group_df['工资'] - group_df['部门平均工资']
print(group_df[['姓名', '部门', '工资', '部门平均工资', '工资差异']])

# 标准化每个部门内的工资
print("\n标准化部门内工资:")
group_df['标准化工资'] = group_df.groupby('部门')['工资'].transform(
    lambda x: (x - x.mean()) / x.std()
)
print(group_df[['姓名', '部门', '工资', '标准化工资']].round(3))
```

## 总结

### Pandas 高级操作技能树

1. **数据输入输出 (I/O)**
   - CSV、Excel、JSON 文件读写
   - 数据库连接和操作
   - 大数据分块处理

2. **索引和选择**
   - 基础索引 (loc, iloc)
   - 条件筛选和布尔索引
   - 多级索引操作

3. **数据信息获取**
   - 基础信息 (shape, dtypes, info)
   - 统计信息 (describe, correlation)
   - 数据质量检查

4. **统计分析**
   - 基础统计计算
   - 分组聚合 (groupby)
   - 透视表和交叉表

5. **数据计算**
   - 数学运算和广播
   - 字符串操作
   - 日期时间计算

6. **数据筛选**
   - 条件筛选
   - 字符串模式匹配
   - 复杂查询构建

7. **数据排序**
   - 单列和多列排序
   - 自定义排序规则
   - 排名计算

8. **数据修改**
   - 值修改和批量更新
   - 条件修改
   - 函数映射

9. **数据删除**
   - 行列删除
   - 重复值处理
   - 空值处理

10. **数据迭代**
    - 行列遍历方法
    - 高效的向量化操作
    - 条件迭代

11. **函数应用**
    - apply、applymap、transform
    - 自定义函数应用
    - 链式操作 (pipe)

### 最佳实践建议

1. **性能优先**: 优先使用向量化操作，避免循环
2. **内存管理**: 处理大数据时注意内存使用
3. **代码可读**: 使用描述性变量名，适当注释
4. **错误处理**: 处理缺失值和异常情况
5. **链式操作**: 合理使用链式操作，保持代码清晰

掌握这些高级操作，您将能够高效地处理各种复杂的数据分析任务！