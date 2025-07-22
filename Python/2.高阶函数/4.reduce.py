# Python reduce 将两个参数的 function 从左至右积累地应用到 iterable 的条目，以便将该可迭代对象缩减为单一的值

from functools import reduce
from operator import truediv

def add(x, y):
    return x + y

numbers = [1, 2, 3, 4, 5]
sum_of_numbers = reduce(add, numbers)

print(sum_of_numbers)  # 输出: 15

print(reduce(truediv, [4, 3, 2, 1]))
print(reduce(lambda x,y: x/y, [4, 3, 2, 1]))