# 当 NumPy 数组的选择对象 obj 是非元组序列对象、ndarray（数据类型为 integer 或 bool）或至少有一个序列对象或 ndarray（数据类型为integer或bool）的元组时，会触发高级索引。
# 与基础索引不同，高级索引可以使用——整数数组、布尔数组或切片的组合——来选择数组中的元素。
# NumPy 有两种类型的高级索引：整型和布尔型。
# 高级索引总是返回数据的副本（与基本索引切片返回视图的不同）。

# TODO 1.花式索引允许使用整数数组来访问数组中的特定元素。它可以用来————选择不连续的元素。
import numpy as np

# 创建一个一维数组
array_1d = np.array([10, 20, 30, 40, 50])

# 使用花式索引访问特定元素
indices = [0, 2, 4]
selected_elements = array_1d[indices]
print("使用花式索引获得的元素:", selected_elements)  # 输出 [10 30 50]

# 创建一个二维数组
array_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 使用花式索引访问二维数组的特定元素
row_indices = [0, 2]
col_indices = [1, 2]
selected_elements_2d = array_2d[row_indices, col_indices] # [0, 1]  [2, 2]
print("使用花式索引获得的二维元素:", selected_elements_2d)  # 输出 [2 9]
# 输出
# 花式索引：通过指定 row_indices 和 col_indices，选择了：
# 第一行的第二列（2）
# 第二行的第二列（9）

# TODO 2.布尔索引允许使用布尔数组来选择符合条件的元素。这种方法非常————适合筛选数据。
# 创建一个包含随机数的数组
array_random = np.array([10, 20, 30, 40, 50])

# 创建布尔索引
bool_index = array_random > 25
print("大于 25 的元素:", array_random[bool_index])  # 输出 [30 40 50]

# 直接使用条件创建布尔索引
print("小于 35 的元素:", array_random[array_random < 35])  # 输出 [10 20 30]


# TODO 3.在多维数组中，可以使用多个索引数组来选择特定的元素。
# 创建一个二维数组
array_2d = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])

# 使用花式索引选择特定元素
row_indices = [0, 1, 2]
col_indices = [1, 0, 2]
selected_elements = array_2d[row_indices, col_indices] #  [0, 1] [1, 0] [2, 2]
print("多维数组的花式索引结果:", selected_elements)  # 输出 [20 40 90]
# 输出
# 花式索引：通过指定 row_indices 和 col_indices，选择了：
# 第一行的第二列（20）
# 第二行的第一列（40）
# 第三行的第三列（90）


# TODO 4.可以将切片和花式索引结合使用，进行更复杂的选择。
# 创建一个二维数组
array_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 选择前两行和特定的列
selected_columns = array_2d[:2, [0, 2]]  # 前两行，——选择第 0 列和第 2 列
print("组合索引结果:\n", selected_columns)
# 输出:
# [[1 3]
#  [4 6]]


# TODO 5.np.ix_() 函数可以创建一个用于多维数组的网格索引。
# 创建一个二维数组
array_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 使用 np.ix_() 选择特定的行和列
rows = [0, 2]
cols = [1, 2]
grid_selected = array_2d[np.ix_(rows, cols)]
print("网格索引结果:\n", grid_selected)
# 输出:
# [[2 3]
#  [8 9]]

# TODO 6.通过高级索引，可以直接修改数组中的元素。
# 创建一个数组
array_modifiable = np.array([10, 20, 30, 40, 50])

# 使用花式索引修改元素
array_modifiable[[1, 3]] = [99, 88]
print("修改后的数组:", array_modifiable)  # 输出 [10 99 30 88 50]

# TODO 7.通过高级索引，使用高级索引选择角元素。
# 创建一个4x3的二维数组x
x = np.array([[ 0,  1,  2],
              [ 3,  4,  5],
              [ 6,  7,  8],
              [ 9, 10, 11]])

# 创建一个二维数组rows，用于指定高级索引的行位置
rows = np.array([[0, 0], [3, 3]], dtype=np.intp)

# 创建一个二维数组columns，用于指定高级索引的列位置
columns = np.array([[0, 2], [0, 2]], dtype=np.intp)

# 使用高级索引选择角元素
# 通过rows和columns数组来选择数组x中的特定元素
# 这种索引方式允许我们直接访问数组中的非连续元素
print(x[rows, columns]) # [0,0] [0,2] [3,0] [3,2]
# 输出
# [[ 0  2]
#  [ 9 11]]

# 定义一个整型数组rows，用于指定矩阵行的索引
rows = np.array([0, 3], dtype=np.intp)
# 定义一个整型数组columns，用于指定矩阵列的索引
columns = np.array([0, 2], dtype=np.intp)

# 打印rows数组的列向量形式
# 这里使用np.newaxis来增加一个新的轴，使rows从1-D数组变为2-D数组
# 这样做是为了能够进行后续的2-D索引操作
print(rows[:, np.newaxis])
# 输出
# [[0]
#  [3]]
# 打印x矩阵中由rows和columns指定的元素
# 这里使用高级索引来获取x矩阵中特定位置的元素
# rows[:, np.newaxis]保持了rows的列向量形式，与columns数组配合使用，实现了对x的2-D索引
print(x[rows[:, np.newaxis], columns])
print(x[np.ix_(rows, columns)])