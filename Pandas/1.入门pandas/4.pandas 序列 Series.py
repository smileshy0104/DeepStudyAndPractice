# Series 是一个一维的带有标签的数组，这个数据可以由任何类型数据构成，包括整型、浮点、字符、Python 对象等。
# 它的轴标签被称为「索引」，它是 Pandas 最基础的数据结构。
# 创建方式：s = pd.Series(data, index=index)
import pandas as pd

# 列表和元组可以直接放入 pd.Series()：
print(pd.Series(['a', 'b', 'c', 'd', 'e']))
print(pd.Series(('a', 'b', 'c', 'd', 'e')))

# ndarray 创建 Series，由索引为 a、b.. ， 五个随机浮点数数组组成
data = [1, 2, 3, 4, 5]
s = pd.Series(data, index=['a', 'b', 'c', 'd', 'e'])

# 输出 Series
print("Index:", s.index) # Index(['a', 'b', 'c', 'd', 'e'], dtype='object')
print("原始 Series:")
print(s)

# 字典创建 Series
d = {'b': 1, 'a': 0, 'c': 2}
s0 = pd.Series(d)
print("字典创建 Series:")
print(s0)

#  如果指定索引，则会按索引顺序，如无法与索引对应的会产生缺失值
s1 = pd.Series(d, index=['b', 'c', 'd', 'a'])
print("按索引顺序创建 Series:")
print(s1)

# 访问和修改
print("\n访问元素:")
print(s['c'])  # 访问索引为 'c' 的元素

s['b'] = 20  # 修改元素
print("\n修改后的 Series:")
print(s)

# 统计操作
print("\n统计信息:")
print(f"总和: {s.sum()}")
print(f"平均值: {s.mean()}")

# 应用函数
s_squared = s.apply(lambda x: x ** 2)
print("\n每个元素平方:")
print(s_squared)

print("————标量————")
# 标量（scalar value）
print(pd.Series(5.))
# Out:
# 0    5.0
# dtype: float64
# 指定索引
print(pd.Series(5., index=['a', 'b', 'c', 'd', 'e']))
# Out:
# a    5.0
# b    5.0
# c    5.0
# d    5.0
# e    5.0
# dtype: float64