def class_decorator(cls):
    # 在类中添加一个新的方法
    cls.new_method = lambda self: "这是一个新方法"

    # 修改类的 __init__ 方法
    original_init = cls.__init__

    def new_init(self, *args, **kwargs):
        print("创建类实例...")
        original_init(self, *args, **kwargs)  # 调用原始的 __init__ 方法

    cls.__init__ = new_init  # 替换原始的 __init__ 方法
    return cls


@class_decorator
class ExampleClass:
    def __init__(self, name):
        self.name = name

    def greet(self):
        return f"你好, {self.name}!"


# 使用装饰过的类
example = ExampleClass("小明")
print(example.greet())  # 输出: 你好, 小明!
print(example.new_method())  # 输出: 这是一个新方法
