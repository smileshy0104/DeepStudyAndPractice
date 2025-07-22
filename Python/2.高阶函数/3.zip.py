# Python 的内置函数 zip() 上用来创建一个"聚合了来自每个可迭代对象中的元素的迭代器"。
# 它获取 iterables（可以是零或更多），将它们--聚合成多个-元组-形成的可迭代对象--，然后返回。

# zip() 函数返回一个基于参数 iterable 对象的元组迭代器（zip 对象）。
# 不传递任何参数，将返回一个空迭代器（zip 对象）
# 传递一个可迭代对象，返回一个元组迭代器，每个元组只有一个元素
# 传递多个可迭代对象，返回一个元组迭代器，每个元组都包含来自所有 iterables 的元素, 迭代器在最短 iterable 耗尽时停止。

# 定义每个人的身高、体重和年龄
lily = [168, 50, 22]  # 身高、体重、年龄
lucy = [170, 55, 25]
amy = [175, 53, 24]

# 使用 zip 打包身高数据
zipped_data = zip(lily, lucy, amy)
zipped_list = list(zipped_data)

# 打印打包后的数据
print("打包后的数据:", zipped_list) # [(168, 170, 175), (50, 55, 53), (22, 25, 24)]

# 计算平均身高
heights = list(zip(lily, lucy, amy))[0]  # 获取身高数据 (168, 170, 175)
average_height = sum(heights) / len(heights)
print("平均身高:", average_height) # 171.0

# 定义姓名和年龄
names = ['Alice', 'Bob', 'Charlie']
ages = [24, 50, 18]

# 使用 zip 结合姓名和年龄并打印
print("\n姓名和年龄:")
for name, age in zip(names, ages):
    print(name, age)

# 示例：解包 zip 对象
coordinate = ['x', 'y', 'z']
value = [3, 4, 5]

result = zip(coordinate, value)
result_list = list(result)

print("\n原始数据:",result_list)

# TODO 解包 zip 对象
# 打印解包后的结果
print("\n解包后的结果:")
c, v = zip(*result_list)
print("坐标:", c)  # 输出: ('x', 'y', 'z')
print("值:", v)   # 输出: (3, 4, 5)
