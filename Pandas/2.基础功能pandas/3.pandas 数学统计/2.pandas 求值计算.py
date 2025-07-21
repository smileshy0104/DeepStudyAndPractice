import pandas as pd
import numpy as np

data = {
    'Q1': [89, 36, 57, 93, 65],
    'Q2': [21, 37, 60, 96, 49],
    'Q3': [24, 37, 18, 71, 61],
    'Q4': [64, 57, 84, 78, 86]
}
df = pd.DataFrame(data)  # 创建 DataFrame
print("原始数据:\n", df)
# 数据操作
print("数据操作:")
# 相邻元素之间的差值
print("\n相邻元素之间的差值:\n", df.diff())
print("\n向右一列的差值:\n", df.diff(axis=1))
print("\n向前两行的差值:\n", df.diff(2))
print("\n新本行为本行减去后一行:\n", df.diff(-1))

# 数据移位
print("\n整体下移一行:\n", df.shift())
print("\n整体上移一行:\n", df.shift(-1))
print("\n向右移动一位:\n", df.shift(axis=1))
print("\n向左移动一位:\n", df.shift(-1, axis=1))

# 检查是否存在至少一个元素为 True
print("\n检查是否存在至少一个元素为 True:\n", df.any())
print("\n检查所有元素是否为 True:\n", df.all())

# 表达式计算
df.eval('Q2Q3 = Q2 + Q3', inplace=True)
print("\n添加新列 Q2Q3:\n", df[['Q2', 'Q3', 'Q2Q3']].head())

# 四舍五入
print("\n四舍五入保留两位小数:\n", df.round(2))
print("\n四舍五入 Q1 保留两位小数，Q2 保留0位:\n", df.round({'Q1': 2, 'Q2': 0}))

# 去重值的数量
print("\n每个列的去重值的数量:\n", df.nunique())

# 真假检测
print("\n缺失值检测:\n", df.isna())
print("\n非缺失值检测:\n", df.notna())

# 数值计算
print("\n对 df 整体所有元素做加法:\n", df + 1)
print("\n对 df 整体所有元素做乘法:\n", df.mul(2))

# 数据降维
print("\n单列 DataFrame 转为 Series:\n", df[['Q1']].squeeze())
print("\n单个值挤压为标量:\n", df.loc[1, 'Q1'].squeeze())

# Series 专门函数
s = pd.Series(['a', 'b', 'c', 'c', 'a', 'b'])

print("\n不重复的值及数量:\n", s.value_counts())
print("\n重复值的频率:\n", s.value_counts(normalize=True))

# Series 专门函数
s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print("\n最大的前两个值:\n", s.nlargest(2))
print("\n最小的前两个值:\n", s.nsmallest(2))
print("\n计算与前一行的变化百分比:\n", s.pct_change())
