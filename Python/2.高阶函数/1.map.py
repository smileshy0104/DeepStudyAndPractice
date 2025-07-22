# TODO 单个可迭代对象
def add_1(x):
    return x + 1

m = map(add_1, [1,2,3,4])
print(list(m)) # [2, 3, 4, 5]

# TODO 多个可迭代对象
def add_num(x, y):
    return x + y

m = map(add_num, [1,2,3,4], [1,2,3,4])
print(list(m)) # [2, 4, 6, 8]


# 对象不同长度 多个序列如果长度不一样，只会————处理到最短的元素位置：
def add_num(x, y):
    return x + y

m = map(add_num, [1,2,3,4], [1, 2])
print(list(m)) # [2, 4]


def add_num(x, y, z):
    return x + y + z

l1 = [1, 2, 3]
var = [*map(add_num, l1, l1, l1)]
print(var)

# TODO 使用匿名函数
# 匿名函数
m = map(lambda x,y: x+y, [1,2,3,4], [1, 2])
print(list(m)) # [2, 4]

# TODO itertools.starmap(function, iterable) 创建一个迭代器，使用从可迭代对象中获取的参数来计算该函数。
# 当参数对应的形参已从一个单独可迭代对象组合为---元组---时（数据已被“预组对”）可用此函数代替 map() 。
# map() 与 starmap() 之间的区别可以类比 function(a,b) 与 function(*c) (元组)的区别
import itertools

# 定义一个简单的函数
def square(x, y):
    return x ** y

# 使用 starmap 对每对元组执行 square 函数
data = [(2, 3), (4, 2), (3, 4)]
results = itertools.starmap(square, data)

# 打印结果
print(list(results))  # 输出 [8, 16, 81]

data = [(2, 3), (4, 2), (3, 4)]

# 使用 lambda 函数计算每对元组的乘积
results = itertools.starmap(lambda x, y: x * y, data)

# 打印结果
print(list(results))  # 输出 [6, 8, 12]