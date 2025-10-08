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
