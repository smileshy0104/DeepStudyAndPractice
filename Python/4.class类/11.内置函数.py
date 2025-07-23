# 内置函数
# setattr(object, name, value): 设置对象的指定属性值。
# getattr(object, name[, default]): 获取对象的指定属性值。如果属性不存在且提供了默认值，则返回默认值；如果未提供默认值且属性不存在，则抛出 AttributeError 异常。
# delattr(object, name): 删除对象的指定属性。
# hasattr(object, name): 检查对象是否具有指定的属性。
# vars([object]): 返回对象的 __dict__ 属性，或者返回当前局部作用域的字典。如果没有提供对象参数，则返回当前作用域的符号表。

class Person:
    def __init__(self, name):
        self.name = name

# 创建一个 Person 对象
person = Person("Alice")

# 1. 使用 getattr 获取属性
print("获取属性:")
print(getattr(person, 'name'))  # 输出: Alice
print(getattr(person, 'age', 25))  # 输出: 25，因为 'age' 属性不存在，默认值是 25

# 2. 使用 setattr 设置属性
print("\n设置属性:")
setattr(person, 'age', 30)
print(person.age)  # 输出: 30

# 3. 使用 delattr 删除属性
print("\n删除属性:")
delattr(person, 'age')
# print(person.age)  # Uncommenting this line will raise AttributeError

# 4. 使用 hasattr 检查属性
print("\n检查属性:")
print(hasattr(person, 'name'))  # 输出: True
print(hasattr(person, 'age'))   # 输出: False

# 5. 使用 vars 获取对象的属性字典
class MyClass:
    class_var = 10

    def __init__(self, x):
        self.x = x

obj = MyClass(20)
print("\n对象的属性字典:")
print(vars(obj))  # 输出: {'x': 20}
print(vars())     # 输出: 当前作用域的符号表
