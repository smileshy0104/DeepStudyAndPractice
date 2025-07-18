# 导入 numpy 库, 约定俗成别名为 np
import numpy as np

# TODO nparray 的生成创建方法：
# 创建一维数组
array1 = np.array([1, 2, 3])
print("一维数组:", array1)  # 输出: array([1, 2, 3])

# 使用元组创建一维数组
array2 = np.array((1, 2, 3))  # 同上
print("使用元组创建的一维数组:", array2)  # 输出: array([1, 2, 3])

# 创建二维数组
array3 = np.array(((1, 2), (1, 2)))
print("二维数组 (使用元组):")
print(array3)  # 输出: array([[1, 2], [1, 2]])

# 使用列表创建二维数组
array4 = np.array(([1, 2], [1, 2]))  # 同上
print("二维数组 (使用列表):")
print(array4)  # 输出: array([[1, 2], [1, 2]])

# TODO 使用函数创建：
# 1. 创建包含 0 到 9 的一维数组
array1 = np.arange(10)  # 10个, 不包括10，步长为 1
print("arange(10):", array1)  # 输出: [0 1 2 3 4 5 6 7 8 9]

# 2. 创建从 3 到 10 的数组，步长为 0.1
array2 = np.arange(3, 10, 0.1)  # 从 3 开始到 10，步长为 0.1
print("arange(3, 10, 0.1):", array2)  # 输出: [3.  3.1 3.2 ... 9.9]

# 3. 从 2.0 到 3.0，生成均匀的 5 个值，不包括终止值 3.0
array3 = np.linspace(2.0, 3.0, num=5, endpoint=False)
print("linspace(2.0, 3.0, num=5, endpoint=False):", array3)  # 输出: [2.  2.2 2.4 2.6 2.8]

# 4. 返回一个 6x4 的随机数组，float 型
array4 = np.random.randn(6, 4)
print("随机数组 (6x4):\n", array4)  # 输出: 6x4 的随机数数组

# 5. 指定范围和形状的整型随机数组
array5 = np.random.randint(3, 7, size=(2, 4))
print("整型随机数组 (2x4):\n", array5)  # 输出: 2x4 的随机整数数组，值在 3 到 6 之间

# 6. 创建从 0 到 20 随机的 5 个数组
array6 = np.random.randint(0, 20, 5)
print("随机整型数组 (5个):", array6)  # 输出: [9 10 14  6 14]（示例）

# 7. 创建值为 0 的数组
array7 = np.zeros(6)  # 6个浮点 0
print("zeros(6):", array7)  # 输出: [0. 0. 0. 0. 0. 0.]

# 8. 创建 5 x 6 整型 0 数组
array8 = np.zeros((5, 6), dtype=int)
print("zeros((5, 6), dtype=int):\n", array8)  # 输出: 5x6 的整型 0 数组

# 9. 创建值为 1 的数组
array9 = np.ones(4)  # 4个浮点 1
print("ones(4):", array9)  # 输出: [1. 1. 1. 1.]

# 10. 创建未初始化的数组
array10 = np.empty(4)  # 4个未初始化的值
print("empty(4):", array10)  # 输出: [0. 0. 0. 0.]（值可能是随机的）

# 11. 创建与目标结构相同的 0 值数组
array11 = np.zeros_like(np.arange(6))
print("zeros_like(np.arange(6)):", array11)  # 输出: [0 0 0 0 0 0]

# 12. 创建与目标结构相同的 1 值数组
array12 = np.ones_like(np.arange(6))
print("ones_like(np.arange(6)):", array12)  # 输出: [1 1 1 1 1 1]

# 13. 创建与目标结构相同的未初始化数组
array13 = np.empty_like(np.arange(6))
print("empty_like(np.arange(6)):", array13)  # 输出: [0 0 0 0 0 0]（值可能是随机的）

# 14. 将 0 到 3 的值依次重复四次
array14 = np.arange(4).repeat(4)  # 将4个值依次重复四次，共16个
print("arange(4).repeat(4):", array14)  # 输出: [0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3]

# 15. 创建 2 行 4 列，值全为 9 的数组
array15 = np.full((2, 4), 9)  # 两行四列值全为 9
print("full((2, 4), 9):\n", array15)  # 输出: [[9 9 9 9] [9 9 9 9]]
