import pandas as pd
import numpy as np

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
