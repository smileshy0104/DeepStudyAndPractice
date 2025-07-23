# TODO 静态方法（static method）是类的一种方法，它不依赖于类的实例或类本身。
#  静态方法使用 @staticmethod 装饰器来定义，不需要额外的 self 或 cls 参数。
#  静态方法通常用于一些逻辑上属于类的方法，但在实现上————不需要访问类的实例属性或类属性。

# 静态方法的用途：
# 工具方法：静态方法经常用于实现一些工具函数，这些函数不需要访问类或实例的状态。
# 逻辑分组：将一些相关的函数放在一个类中，使用静态方法来实现，可以使代码结构更加清晰。
# 工厂方法：虽然类方法更常用于工厂方法，有时也可以用静态方法来创建类的实例。
class MyClass:
    @staticmethod
    def static_method():
        print("This is a static method.")

# 调用静态方法
print("调用静态方法：")
# 通过类调用静态方法
MyClass.static_method()  # 输出: This is a static method.

# 通过实例调用静态方法
obj = MyClass()
obj.static_method()      # 输出: This is a static method.


# 示例：工具方法
class MathUtils:
    @staticmethod
    def add(a, b):
        return a + b

    @staticmethod
    def subtract(a, b):
        return a - b

print("\n工具方法示例：")
print(MathUtils.add(5, 3))       # 输出: 8
print(MathUtils.subtract(5, 3))  # 输出: 2


# 示例：逻辑分组
class StringUtils:
    @staticmethod
    def is_palindrome(s):
        return s == s[::-1]

    @staticmethod
    def to_uppercase(s):
        return s.upper()

print("\n逻辑分组示例：")
print(StringUtils.is_palindrome("radar"))  # 输出: True
print(StringUtils.to_uppercase("hello"))   # 输出: HELLO


# 示例：静态方法作为工厂方法
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    @staticmethod
    def create_child(name):
        return Person(name, 0)

print("\n工厂方法示例：")
child = Person.create_child("Alice")
print(child.name)  # 输出: Alice
print(child.age)   # 输出: 0
