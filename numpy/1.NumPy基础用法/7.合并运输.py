import numpy as np

# 创建示例数组
a1 = np.array([1, 2, 3])
a2 = np.array([4, 5, 6])
a3 = np.array([[7, 8, 9], [10, 11, 12]])

# 1. 使用 np.append 追加数组
appended_array = np.append(a1, a2)  # 将 a2 追加到 a1 的后面
print("np.append(a1, a2):", appended_array)  # 输出: [1 2 3 4 5 6]

# 2. 使用 np.stack 进行堆叠
stacked_array = np.stack((a1, a2))  # 沿着新轴堆叠
print("np.stack((a1, a2)):\n", stacked_array)
# 输出:
# [[1 2 3]
#  [4 5 6]]

# 3. 使用 np.dstack 进行深度堆叠
dstacked_array = np.dstack((a1, a2))  # 沿着深度轴堆叠
print("np.dstack((a1, a2)):\n", dstacked_array)
# 输出:
# [[[1 4]]
#  [[2 5]]
#  [[3 6]]]

# 4. 使用 np.vstack 进行垂直合并
vstacked_array = np.vstack((a1, a2))  # 垂直合并 a1 和 a2
print("np.vstack((a1, a2)):\n", vstacked_array)
# 输出:
# [[1 2 3]
#  [4 5 6]]

# 5. 使用 np.hstack 进行水平合并
hstacked_array = np.hstack((a1, a2))  # 水平合并 a1 和 a2
print("np.hstack((a1, a2)):", hstacked_array)  # 输出: [1 2 3 4 5 6]

# 6. 使用 np.newaxis 增加新维度
newaxis_array = a1[np.newaxis, :]  # 在第一个维度增加新轴
print("a1[np.newaxis, :]:\n", newaxis_array)
# 输出:
# [[1 2 3]]

# 7. 使用 np.concatenate 进行合并
concatenated_array = np.concatenate((a1, a2))  # 合并 a1 和 a2
print("np.concatenate((a1, a2)):", concatenated_array)  # 输出: [1 2 3 4 5 6]

# 8. 使用 np.split 分隔数组
ab = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])  # 创建一个 4x3 的数组
split_arrays = np.split(ab, 4, axis=0)  # 沿着第 0 轴将 ab 分成 4 个部分
print("np.split(ab, 4, axis=0):")
for i, arr in enumerate(split_arrays):
    print(f"部分 {i}:\n", arr)
# 输出:
# 部分 0:
# [[1 2 3]]
# 部分 1:
# [[4 5 6]]
# 部分 2:
# [[7 8 9]]
# 部分 3:
# [[10 11 12]]
