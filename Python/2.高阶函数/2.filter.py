# filter() 方法的语法是：
# filter(function, iterable)
# filter() 有两个参数：
# function - 测试 iterable 元素是否返回 true 或 false 的函数
# iterable - 要过滤的 iterable，可以是集合、列表、元组或任何迭代器的容器

# TODO filter 对象
# 创建一个函数，判断一个数是否是偶数
def is_even(x):
    return x % 2 == 0

numbers = [1, 2, 3, 4, 5]
even_numbers = list(filter(is_even, numbers))

print(even_numbers)  # 输出: [2, 4]
print(type(filter(is_even, numbers))) # <class 'filter'>

# 筛选大于 5 的值
f = filter(lambda x: x>5, [2,3,5,7,9])
print(f) # <filter at 0x7fe33ea36730>
print(list(f)) # [7, 9]

# 函数为 None
f = filter(None, [2,False,5,None,9])
print(list(f)) # [2, 5, 9]

# 元音筛选
letters = ['a', 'b', 'd', 'e', 'i', 'j', 'o']

def filter_vowels(letter):
    vowels = ['a', 'e', 'i', 'o', 'u']

    if letter in vowels:
        return True
    else:
        return False

filtered_vowels = filter(filter_vowels, letters)

print('The filtered vowels are:')
for vowel in filtered_vowels:
    print(vowel)


strings = ['apple', '', 'banana', '', 'cherry']

non_empty_strings = filter(None, strings)
print(list(non_empty_strings))
# ['apple', 'banana', 'cherry']

# TODO 列表表达式
l = [-2, -1, 0, 1, 2]
print([x for x in l if x % 2 == 0])
# [-2, 0, 2]

print([x for x in l if x % 2 != 0])
# [-1, 1]

l_s = ['apple', 'orange', 'strawberry']
print([x for x in l_s if x.endswith('e')])
# ['apple', 'orange']

print([x for x in l_s if not x.endswith('e')])
# ['strawberry']

l = [-2, -1, 0, 1, 2]
print([x for x in l if x])
# [-2, -1, 1, 2]

l_2d = [[0, 1, 2], [], [3, 4, 5]]
print([x for x in l_2d if x])
# [[0, 1, 2], [3, 4, 5]]