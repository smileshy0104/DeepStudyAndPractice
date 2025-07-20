# pandas 将读取到的数据加载到自己的叫做 Series 和 DataFrame 的数据结构框架
# 1	Series	带标签的一维同构数组
# 2	DataFrame	带标签的，大小可变的，二维异构表格

# Numpy 是一个高效的科学计算库，Panas 的这些数据结构是构建在 Numpy 数组之上，所以处理速度非常快
# Series 和 DataFrame 里边的"值"都是"可变"的，它们都可以增加行，并排序，Series 只有一列不能再增加，DataFrame 可以增加列。
# TODO 1.Series
# 定义：一维数组，类似于 Python 的列表或 NumPy 的一维数组。
# 特点：
# 每个元素都有一个索引（index），可以是数字或字符串。
# 可以存储任意数据类型（整数、浮点数、字符串等）。
import pandas as pd

# 创建一个 Series
s = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
print(s)

# TODO 2.DataFrame
# 定义：二维表格数据结构，类似于电子表格或 SQL 表。
# 特点：
# 由行和列组成，可以看作是多个 Series 的集合。
# 每列可以是不同的数据类型（整数、浮点数、字符串等）。
# 支持多种操作，如筛选、排序、分组等。

# 创建一个 DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'Los Angeles', 'Chicago']
}
df = pd.DataFrame(data)
print(df)

# TODO 4.Index
# 定义：用于标识 Series 和 DataFrame 中的行和列。
# 特点：
# 提供了快速的查找和选择功能。
# 可以使用自定义索引，包括时间序列索引。

# 创建一个 DataFrame 并查看其索引
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
print(df)
print("查看行索引:",df.index)  # 查看行索引
print("查看列索引:",df.columns)  # 查看列索引
