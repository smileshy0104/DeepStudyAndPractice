# 在 Pandas 中，可以方便地将 DataFrame 或 Series 转换为 NumPy 数组。
# 注意事项
# 数据类型: 转换后的 NumPy 数组的————数据类型将与原 DataFrame 或 Series 的数据类型相同。
# 如果 DataFrame 中包含不同的数据类型，NumPy 数组将使用最通用的数据类型（通常是 object）。
# 缺失值: 如果 DataFrame 或 Series 包含缺失值（NaN），这些值在转换后仍然会保留。
# 性能: to_numpy() 方法通常比 .values 属性更快且更可靠，因此建议使用 to_numpy()。
# TODO 1.使用 DataFrame 的 to_numpy() 方法将整个 DataFrame 转换为 NumPy 数组。
import pandas as pd

# 创建一个 DataFrame
data = {
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': [7, 8, 9]
}
df = pd.DataFrame(data)
print("原始 DataFrame:\n", df)

# 将 DataFrame 转换为 NumPy 数组
numpy_array = df.to_numpy()
print("转换后的 NumPy 数组:\n", numpy_array)

# TODO 2.对于 Pandas Series，可以使用 to_numpy() 方法将其转换为 NumPy 数组。
# 创建一个 Series
series = pd.Series([1, 2, 3, 4, 5])
print("原始 Series:\n", series)

# 将 Series 转换为 NumPy 数组
numpy_array_from_series = series.to_numpy()
print("转换后的 NumPy 数组:\n", numpy_array_from_series)
