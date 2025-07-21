import pandas as pd

# TODO 主要去看文档！！！
# 创建一个示例 DataFrame
data = {
    'month': ['January', 'February', 'March'],
    'year': [2021, 2022, 2023],
    'sale': [100, 150, 200]
}
df = pd.DataFrame(data)

print("DataFrame:")
print(df)

# 创建索引对象
int_index = pd.Index([1, 2, 3])
print("\n整数索引:")
print(int_index)
char_index = pd.Index(list('abc'))
print("\n字符索引:")
print(char_index)
named_index = pd.Index(['e', 'd', 'a', 'b'], name='something')
print("\n命名索引:")
print(named_index)
print("\n命名索引的索引名称:", named_index.name)
print("\n")

# 查看 DataFrame 的索引和列
print("DataFrame 索引:", df.index)  # RangeIndex(start=0, stop=3, step=1)
print("DataFrame 列:", df.columns)  # Index(['month', 'year', 'sale'], dtype='object')

print("\n")
# 索引的属性
df.index.name = 'number1'
print("索引名称:", df.index.name)  # None
print("索引数组:", df.index.array)  # <PandasArray>
print("索引数据类型:", df.index.dtype)  # dtype('int64')
print("索引形状:", df.index.shape)  # (3,)
print("索引大小:", df.index.size)   # 3
print("索引值:", df.index.values)  # array([0, 1, 2])

print("\n")
# 索引的操作方法
df.index = df.index.astype('int64')  # 转换类型
print("转换后的索引:", df.index)

# 检查是否存在
print("索引是否包含 1:", df.index.isin([0, 1]))  # array([ True,  True])

# 重命名索引
df.index.rename('number2', inplace=True)
print("重命名后的索引:", df.index)

# 去重和计数
print("不重复值的数量:", df.index.nunique())  # 3

# 排序
sorted_index = df.index.sort_values(ascending=False)
print("排序后的索引:", sorted_index)

# 转换为列表
index_list = df.index.to_list()  # [0, 1, 2]
print("索引列表:", index_list)

# 转为 Series
index_series = df.index.to_series()
print("索引转为 Series:", index_series)

# 筛选索引
filtered_index = df.index.where(df.index == 1)
print("筛选后的索引:", filtered_index)

# 获取最大值和最小值
print("最大值:", df.index.max())  # 最大值
print("最小值:", df.index.min())  # 最小值

# 删除索引
df = df.drop(index=1, errors='ignore')  # 删除索引为 1 的行
print("删除索引后的 DataFrame:\n", df)

# 连接索引
new_index = df.index.append(pd.Index([3, 4]))  # 直接连接索引
print("连接后的新索引:", new_index)

# 检查是否有重复值
print("索引是否有重复值:", df.index.has_duplicates)  # False

