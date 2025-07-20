# Pandas 的 DataFrame 是一种二维的标签数组，类似于电子表格或 SQL 表。
# 它是 Pandas 中最重要的数据结构之一，广泛用于数据分析和处理。
# 在 Pandas 中，使用 DataFrame 或 Series 时，copy 参数用于控制数据的复制行为。
# 具体来说，当你创建一个 DataFrame 或 Series 时，可以通过设置 copy=False 来“避免复制数据”，从而提高性能，尤其是在处理大型数据集时。
import pandas as pd
import numpy as np

# 原始数据字典
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35]
}

# 创建 DataFrame，copy=False
# 这里使用 copy=False 意味着 DataFrame 将直接引用传入的数据字典，而不是复制它。
# 这样做的好处是性能更好，但需要注意的是，如果原始数据发生变化，DataFrame 中的数据也会相应改变。
df = pd.DataFrame(data, copy=False)

print("创建的 DataFrame:")
print(df)

# 修改原始数据
# 这里修改原始字典中的数据，由于 df 是直接引用的 data，所以 df 中的数据也会受到影响。
data['Age'][0] = 26  # 修改原始字典中的数据

print("\n修改原始数据后的 DataFrame:")
print(df)  # df 中的 Age 列也会反映这个变化

# 如果希望不影响原始数据，可以使用 copy=True
# 这里创建 DataFrame 时使用 copy=True，会复制传入的数据，从而避免原始数据的改变影响到 DataFrame。
df_copy = pd.DataFrame(data, copy=True)
data['Age'][1] = 31  # 修改原始字典中的数据

print("\n使用 copy=True 创建的 DataFrame:")
print(df_copy)  # df_copy 不受影响
print("\n修改原始数据后的 df_copy:")
print(df_copy)



# 创建 DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'Los Angeles', 'Chicago']
}
df = pd.DataFrame(data)

# 访问和选择数据
print("访问 'Name' 列:")
print(df['Name'])

print("访问前两行:")
print(df.head(2))

# 修改数据
df.at[0, 'Age'] = 26
print("\n修改后的 DataFrame:")
print(df)

# 添加新列
df['Country'] = ['USA', 'USA', 'USA']
print("\n添加新列后的 DataFrame:")
print(df)

# 筛选数据
filtered_df = df[df['Age'] > 25]
print("\n筛选年龄大于 25 的行:")
print(filtered_df)

# 查看基本信息
print("\nDataFrame 信息:")
print(df.info())

# 描述性统计
print("\n描述性统计:")
print(df.describe())

# 排序
sorted_df = df.sort_values(by='Age')
print("\n按年龄排序后的 DataFrame:")
print(sorted_df)

# 分组
grouped_df = df.groupby('City')['Age'].mean()
print("\n按照城市分组的平均年龄:")
print(grouped_df)

# 合并和连接
data2 = {
    'Name': ['David', 'Eve'],
    'Age': [40, 28],
    'City': ['Miami', 'Houston']
}
df2 = pd.DataFrame(data2)

combined_df = pd.concat([df, df2], ignore_index=True)
print("\n合并后的 DataFrame:")
print(combined_df)

