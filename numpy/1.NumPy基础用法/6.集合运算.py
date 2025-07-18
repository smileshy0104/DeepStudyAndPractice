import numpy as np

# 定义两个示例数组 x 和 y
x = np.array([1, 2, 3, 4, 5])
y = np.array([4, 5, 6, 7, 8])

# 1. 输出 x 和 y 的交集
intersection = np.intersect1d(x, y)  # 找到 x 和 y 的共同元素
print("x 和 y 的交集:", intersection)  # 输出: [4 5]

# 2. 输出在 x 中不在 y 中的元素
difference = np.setdiff1d(x, y)  # 找到在 x 中但不在 y 中的元素
print("在 x 中不在 y 中的元素:", difference)  # 输出: [1 2 3]

# 3. 输出 x 和 y 的并集
union = np.union1d(x, y)  # 找到 x 和 y 的所有唯一元素
print("x 和 y 的并集:", union)  # 输出: [1 2 3 4 5 6 7 8]

# 4. 输出 x 和 y 的异或集
symmetric_difference = np.setxor1d(x, y)  # 找到在 x 或 y 中但不同时在两者中的元素
print("x 和 y 的异或集:", symmetric_difference)  # 输出: [1 2 3 6 7 8]
