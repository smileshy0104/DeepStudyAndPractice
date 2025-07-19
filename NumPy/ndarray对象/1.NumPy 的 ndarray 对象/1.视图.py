# 在 NumPy 中，**视图（view）**是指对同一数据的不同视角。
# 通过视图，你可以在不复制原始数据的情况下，创建新的数组对象。这意味着————对视图的修改会影响原始数组，反之亦然。
#
# 视图的特点
# 共享数据: 视图与原始数组————共享相同的数据内存，因此内存使用效率更高。
# 修改影响: 对视图的修改————会直接影响到原始数组，反之亦然。
# 不同形状: 视图可以具有不同的形状和维度，但它们仍然指向相同的数据。

# 视图的应用场景
# 节省内存: 在处理大型数据集时，使用视图可以节省内存，因为不需要复制数据。
# 数据处理: 在数据分析和处理过程中，视图可以方便地对数据进行不同的切片和变换。
# 高效计算: 通过视图，可以高效地进行批量操作，而无需复制数据。
import numpy as np

# TODO 示例 1: 使用切片创建视图
# 创建一个数组
array = np.array([[1, 2, 3],
                  [4, 5, 6]])

# 使用切片创建视图，view 是通过切片创建的，它共享了 array 的数据
view = array[0, :]
print("原始数组:\n", array)
print("视图:\n", view)

# 修改视图，当我们修改 view 中的元素时，array 中对应的元素也发生了变化。
view[0] = 10
print("修改视图后的原始数组:\n", array)

# TODO 示例 2: 使用 np.reshape 创建视图
# 创建一个一维数组
array = np.arange(6)

# 使用 reshape 创建视图
# 通过 reshape 创建的 view 共享了 array 的数据。
view = array.reshape(2, 3)
print("原始数组:\n", array)
print("视图:\n", view)

# 修改视图
# 当我们修改 view 的元素时，array 中对应的元素也会变化。
view[0, 0] = 100
print("修改视图后的原始数组:\n", array) # [100   1   2   3   4   5]
print("修改视图后的原始数组:\n", view)

# TODO 示例 3: 使用 np.copy 创建副本
# 创建一个数组
array = np.array([[1, 2, 3],
                  [4, 5, 6]])

# 使用 copy 创建副本
copy = array.copy()
print("原始数组:\n", array)
print("副本:\n", copy)
print("副本类型:\n", type(array))  # <class 'numpy.ndarray'>

# 修改副本
copy[0, 0] = 10
print("修改副本后的原始数组:\n", array)
print("修改后的副本:\n", copy)
