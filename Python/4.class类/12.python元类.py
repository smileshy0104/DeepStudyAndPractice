# TODO 元类是类的类，可以用于控制类的创建行为。
# 在 Python 中，类也是对象，因此它们的创建也是通过其他类来完成的。这个“其他类”就是元类。
# 元类在 Python 中的主要作用是控制类的创建过程。
# 通过定义自己的元类，你可以控制类的初始化、属性赋值、方法添加等行为。
# 元类最常用的场景是创建 API，框架，或者实现ORM（对象关系映射）等功能。

class MyMeta(type):
    def __new__(cls, name, bases, dct):
        # 在创建类之前调用
        print("Creating class:", name)
        print("Bases:", bases)
        print("Attributes:", dct)
        return super().__new__(cls, name, bases, dct)

class MyClass(metaclass=MyMeta):
    pass

# 输出：
# Creating class: MyClass
# Bases: ()
# Attributes: {'__module__': '__main__', '__qualname__': 'MyClass'}

class HelloMeta(type):
    # 接收到name（类的名称）、bases（基类）和attrs（属性字典）
    # TODO 在attrs中添加了一个名为"hello"的方法
    def __new__(cls, name, bases, attrs):
        attrs['hello'] = lambda self: print(f"Hello, {self.name}!")
        return super().__new__(cls, name, bases, attrs)

# 通过metaclass参数将HelloMeta指定为它的元类。
class MyClass(metaclass=HelloMeta):
    name = "World"

# 调用obj.hello() 时，它会打印出"Hello, World!"，其中"World"是我们在MyClass中定义的name属性的值。
obj = MyClass()

obj.hello()  # 输出：Hello, World!


# TODO 类创建过程可通过在定义行传入 metaclass 关键字参数，或是通过继承一个包含此参数的现有类来进行定制。
class Meta(type):
    pass

class MyClass(metaclass=Meta):
    pass

class MySubclass(MyClass):
    pass