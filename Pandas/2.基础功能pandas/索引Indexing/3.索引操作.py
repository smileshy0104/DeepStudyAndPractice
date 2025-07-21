import pandas as pd

# 创建一个示例 DataFrame
data = {
    'month': ['January', 'February', 'March'],
    'year': [2021, 2022, 2023],
    'sale': [100, 150, 200]
}
df = pd.DataFrame(data)

# 重置索引
print("原始 DataFrame:\n", df)

# 清除索引
df_reset = df.reset_index()
print("\n清除索引后的 DataFrame:\n", df_reset)

# 设置 'month' 列为索引并重置索引
df_with_month_index = df.set_index('month').reset_index()
print("\n设置 'month' 列为索引后再重置:\n", df_with_month_index)

# 删除原索引，'month' 列没了
df_reset_drop = df.set_index('month').reset_index(drop=True)
print("\n删除原索引后的 DataFrame:\n", df_reset_drop)

# 创建一个新 DataFrame 用于后续操作
df2 = df.copy()

# 覆盖使重置生效
df2.reset_index(inplace=True)
print("\n重置索引并覆盖后的 DataFrame:\n", df2)

# 设置多级索引
df_multi_index = df.set_index(['month', 'year'])
print("\n设置多级索引后的 DataFrame:\n", df_multi_index)

# 取消 'year' 一级索引
df_year_index = df_multi_index.reset_index(level='year')
print("\n取消 'year' 一级索引后的 DataFrame:\n", df_year_index)

# 使用层级索引名重置索引
df_year_index_level = df_multi_index.reset_index(level='year', drop=False)
print("\n使用层级索引名重置索引后的 DataFrame:\n", df_year_index_level)

# 修改索引内容
new_index = ['a', 'b', 'c']
df_reindexed = df.reindex(new_index)
print("\n修改索引后的 DataFrame:\n", df_reindexed)

# 索引重命名
df_rename_columns = df.rename(columns={"month": "月", "year": "年", "sale": "销售"})
print("\n重命名列索引后的 DataFrame:\n", df_rename_columns)

# 一一对应修改列索引
df_rename_columns_specific = df.rename(columns={"month": "月份", "year": "年份"})
print("\n一一对应修改列索引后的 DataFrame:\n", df_rename_columns_specific)

# 修改行索引
df_rename_index = df.rename(index={0: "x", 1: "y", 2: "z"})
print("\n修改行索引后的 DataFrame:\n", df_rename_index)

# 修改数据类型
df_rename_type = df.rename(index=str)
print("\n修改索引数据类型后的 DataFrame:\n", df_rename_type)

# 列名加前缀
df_add_prefix = df.rename(lambda x: 't_' + x, axis=1)
print("\n列名加前缀后的 DataFrame:\n", df_add_prefix)

# 修改索引
df.set_axis(['a', 'b', 'c'], axis='index', inplace=True)
print("\n修改索引后的 DataFrame:\n", df)

# 修改列名
df.set_axis(list('abc'), axis=1, inplace=True)
print("\n修改列名后的 DataFrame:\n", df)

# 修改索引名
df.rename_axis("动物", axis="index", inplace=True)
print("\n修改索引名后的 DataFrame:\n", df)

# 修改多层索引名
df_multi_index.rename_axis(index={'month': '月份', 'year': '年份'}, inplace=True)
print("\n修改多层索引名后的 DataFrame:\n", df_multi_index)

# 修改多层列索引名
df_multi_index.rename_axis(columns={'sale': '销售额'}, inplace=True)
print("\n修改多层列索引名后的 DataFrame:\n", df_multi_index)
