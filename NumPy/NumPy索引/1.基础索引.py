# 在 NumPy 中，数组的基础索引是非常重要的概念，它允许我们访问和操作数组中的元素。
import numpy as np
from numpy.typing import NDArray


def integer_indexing_1d() -> None:
    """
    展示一维数组的整数索引。
    对于一维数组，可以使用"整数索引"来访问特定的元素。
    """
    print("--- 1. 一维数组整数索引 ---")
    # 创建一维数组
    array_1d: NDArray[np.int_] = np.array([10, 20, 30, 40, 50])

    # 访问特定元素
    print(f"原始数组: {array_1d}")
    print(f"第一个元素: {array_1d[0]}")  # 索引从 0 开始
    print(f"第三个元素: {array_1d[2]}")

    # 负索引访问
    print(f"最后一个元素: {array_1d[-1]}")
    print(f"倒数第二个元素: {array_1d[-2]}")
    print()


def integer_indexing_2d() -> None:
    """
    展示二维数组的整数索引。
    对于二维数组，可以使用"元组形式"的索引来访问特定的元素。
    """
    print("--- 2. 二维数组整数索引 ---")
    # 创建二维数组
    array_2d: NDArray[np.int_] = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(f"原始二维数组:\n{array_2d}")

    # 访问特定元素
    print(f"第一行第一列的元素: {array_2d[0, 0]}")
    print(f"第二行第三列的元素: {array_2d[1, 2]}")

    # 负索引访问
    print(f"最后一行最后一列的元素: {array_2d[-1, -1]}")
    print(f"倒数第二行第一列的元素: {array_2d[-2, 0]}")

    # 如果对索引数少于维度的多维数组进行索引，则会得到一个子维度数组。
    # 注意：返回的数组是一个视图（view），它不是原始数组的副本，而是指向内存中与原始数组相同的值。
    print(f"第一行的元素 (视图): {array_2d[0]}")
    # 两种方式等价
    print(f"第二行第三列的元素 (方式一): {array_2d[1, 2]}")
    print(f"第二行第三列的元素 (方式二): {array_2d[1][2]}")
    print()


def slicing() -> None:
    """
    展示数组的切片操作。
    切片的语法为 start:stop:step。
    """
    print("--- 3. 切片 (Slicing) ---")
    array_1d: NDArray[np.int_] = np.array([10, 20, 30, 40, 50])
    array_2d: NDArray[np.int_] = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    print(f"原始一维数组: {array_1d}")
    # 一维数组切片
    print(f"从第二个到第四个元素: {array_1d[1:4]}")
    print(f"每隔一个元素: {array_1d[::2]}")

    print(f"\n原始二维数组:\n{array_2d}")
    # 二维数组切片
    print(f"获取前两行:\n{array_2d[:2]}")
    print(f"获取第二列: {array_2d[:, 1]}")
    print(f"获取第一行的前两个元素: {array_2d[0, :2]}")
    print()


def boolean_indexing() -> None:
    """
    展示布尔索引的使用。
    """
    print("--- 4. 布尔索引 ---")
    # 创建一个包含随机数的数组
    array_random: NDArray[np.int_] = np.array([10, 20, 30, 40, 50])
    print(f"原始数组: {array_random}")

    # 布尔索引
    bool_index = array_random > 25  # 创建布尔数组
    print(f"布尔索引 (大于25): {bool_index}")
    print(f"大于 25 的元素: {array_random[bool_index]}")

    # 直接使用条件
    print(f"大于 25 的元素 (直接使用条件): {array_random[array_random > 25]}")
    print()


def fancy_indexing() -> None:
    """
    展示花式索引的使用。
    """
    print("--- 5. 花式索引 ---")
    # 创建一个数组
    array_fancy: NDArray[np.int_] = np.array([10, 20, 30, 40, 50])
    print(f"原始数组: {array_fancy}")

    # 使用花式索引
    indices = [0, 2, 4]
    print(f"索引列表: {indices}")
    print(f"使用花式索引获得的元素: {array_fancy[indices]}")
    print()


if __name__ == "__main__":
    integer_indexing_1d()
    integer_indexing_2d()
    slicing()
    boolean_indexing()
    fancy_indexing()
