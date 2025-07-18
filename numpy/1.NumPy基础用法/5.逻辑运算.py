import numpy as np

# 设定变量 x
x = 10

# 1. 使用 np.where
result_where = np.where(x > 0, 1, 0)  # 如果 x > 0，则返回 1，否则返回 0
print("np.where(x > 0, 1, 0):", result_where)  # 输出: array(1)

# 2. 逻辑与
result_and = np.logical_and(x > 0, x > 5)  # 检查 x 是否大于 0 且大于 5
print("np.logical_and(x > 0, x > 5):", result_and)  # 输出: True

# 3. 逻辑或
result_or = np.logical_or(x > 0, x < 5)  # 检查 x 是否大于 0 或小于 5
print("np.logical_or(x > 0, x < 5):", result_or)  # 输出: True

# 4. 逻辑非
result_not = np.logical_not(x > 5)  # 检查 x 是否不大于 5
print("np.logical_not(x > 5):", result_not)  # 输出: False

# 5. 逻辑异或
result_xor = np.logical_xor(x > 5, x == 0)  # 检查 x 是否大于 5 与 x 是否等于 0 之间的异或关系
print("np.logical_xor(x > 5, x == 0):", result_xor)  # 输出: True
