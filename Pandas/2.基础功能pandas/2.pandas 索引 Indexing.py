import pandas as pd
import numpy as np
from datetime import timedelta

# 1. 创建一个示例 DataFrame
data = {
    'name': ['Liver', 'Arry', 'Ack', 'Eorge', 'Oah'],
    'team': ['E', 'C', 'A', 'C', 'D'],
    'Q1': [89, 36, 57, 93, 65],
    'Q2': [21, 37, 60, 96, 49],
    'Q3': [24, 37, 18, 71, 61],
    'Q4': [64, 57, 84, 78, 86]
}

df = pd.DataFrame(data)  # 创建 DataFrame
# data = 'https://gairuo.com/file/data/dataset/team.xlsx'
# df = pd.read_excel(data, index_col='name') # 从 Excel 文件读取数据并将 'name' 设置为索引
print(df)  # 打印 DataFrame

# 2. 设置索引
print("设置团队为索引:")  # 输出提示
print(df.set_index('team'))  # 将 'team' 列设置为索引并打印结果

print("设置团队和 Q1 为多层索引:")  # 输出提示
print(df.set_index(['team', 'Q1']))  # 将 'team' 和 'Q1' 列设置为多层索引并打印结果

# 3. 使用 Series 设置索引
s = pd.Series([1, 2, 3, 4, 5])  # 创建一个 Series
print("设置索引为 s:")
print(df.set_index(s))  # 将 Series s 设置为索引并打印结果

print("指定的索引和现有字段同时指定 s:")
print(df.set_index([s, 'team']))  # 将 Series s 和 'team' 列同时设置为索引并打印结果

print("计算索引 s:")
print(df.set_index([s, s**2]))  # 将 Series s 和其平方设置为索引并打印结果

print("不保留原列 s:")
print(df.set_index('name', drop=True))  # 将 'name' 列设置为索引并丢弃原列

print("保留原列 s:")
print(df.set_index('name', drop=False))  # 将 'name' 列设置为索引并保留原列

print("保留原来的索引 s:")
print(df.set_index('name', append=True))  # 将 'name' 列设置为索引并保留原来的索引

print("不保留原来的索引 s:")
print(df.set_index('name', append=False))  # 将 'name' 列设置为索引，不保留原来的索引

print("建立索引并重写覆盖 df s:")
print(df.set_index('name', inplace=False))  # 建立索引并重写覆盖 df


# 4. 设置索引为 'name'
df.set_index('name', append=True)  # 建立索引并重写覆盖 df
print("设置索引为 'name':")
print(df)  # 打印当前 DataFrame

# 5. 设置多层索引（以 'team' 和 'Q1' 为索引）
df_multi = df.set_index(['team', 'Q1'])  # 设置多层索引
print("\n设置多层索引（'team' 和 'Q1'）:")
print(df_multi)  # 打印多层索引的 DataFrame

# 6. 创建时间索引 date_range
date_rng = pd.date_range(start='2023-01-01', end='2023-01-05', freq='D')  # 创建日期范围
df_time = pd.DataFrame(date_rng, columns=['date'])  # 创建 DataFrame
df_time['data'] = np.random.randint(0, 100, size=(len(date_rng)))  # 添加随机数据列
df_time.set_index('date', inplace=True)  # 将 'date' 列设置为索引
print("\n时间索引 DataFrame:")
print(df_time)  # 打印带有时间索引的 DataFrame

# 7. 创建类别索引 Categorical
category_data = pd.Categorical(['apple', 'banana', 'apple', 'orange'])  # 创建类别数据
df_category = pd.DataFrame({'fruit': category_data, 'quantity': [10, 20, 15, 30]})  # 创建 DataFrame
df_category.set_index('fruit', inplace=True)  # 将 'fruit' 列设置为索引
print("\n类别索引 DataFrame:")
print(df_category)  # 打印类别索引的 DataFrame

# 8. 创建 IntervalIndex 间隔索引
interval_index = pd.interval_range(start=0, end=5, freq=1)  # 创建间隔索引
df_interval = pd.DataFrame({'value': [1, 2, 3, 4, 5]}, index=interval_index)  # 创建 DataFrame
print("\n间隔索引 DataFrame:")
print(df_interval)  # 打印间隔索引的 DataFrame

# 9. 设置新的索引
df_interval['new_index'] = ['A', 'B', 'C', 'D', 'E']  # 添加新列
df_interval.set_index('new_index', inplace=True)  # 将新列设置为索引
print("\n设置新索引后的 DataFrame:")
print(df_interval)  # 打印设置新索引后的 DataFrame

# 10. 创建 TimedeltaIndex 时间差索引
timedelta_index = pd.TimedeltaIndex(['1 days', '2 days', '3 days'])  # 创建时间差索引
df_timedelta = pd.DataFrame({'duration': [10, 20, 30]}, index=timedelta_index)  # 创建 DataFrame
print("\n时间差索引 DataFrame:")
print(df_timedelta)  # 打印时间差索引的 DataFrame

# 11. 创建 PeriodIndex 周期索引
period_index = pd.period_range('2020-01-01', periods=5, freq='M')  # 创建周期索引
df_period = pd.DataFrame({'sales': [100, 150, 200, 250, 300]}, index=period_index)  # 创建 DataFrame
print("\n周期索引 DataFrame:")
print(df_period)  # 打印周期索引的 DataFrame
