import pandas as pd
import numpy as np

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
