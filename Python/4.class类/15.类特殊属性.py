class MyClass:
    """这是一个示例类"""

    def __init__(self, name):
        self.name = name

    def greet(self):
        return f'你好，我是 {self.name}。'

# 创建一个 MyClass 的实例
obj = MyClass('示例')

# 展示特殊属性
print(f"对象的文档字符串: {MyClass.__doc__}")  # 输出: 这是一个示例类
print(f"对象的名称: {MyClass.__name__}")  # 输出: MyClass
print(f"对象的模块: {MyClass.__module__}")  # 输出: __main__
print(f"对象的字典: {obj.__dict__}")  # 输出: {'name': '示例'}
print(f"对象的类: {obj.__class__}")  # 输出: <class '__main__.MyClass'>
print(f"对象的限定名称: {MyClass.__qualname__}")  # 输出: MyClass
print(f"对象的 MRO: {MyClass.__mro__}")  # 输出: (<class '__main__.MyClass'>, <class 'object'>)
# obj.__globals__ 不适用于实例对象，需在函数内使用
# print(f"对象的全局变量: {obj.__globals__}")  # 这行会引发错误
print(f"对象的代码对象: {obj.greet.__code__}")  # 输出: <code object ...>
print(f"对象的函数: {obj.greet.__func__}")  # 输出: <function MyClass.greet at ...>
print(f"模块文件路径: {__file__}")  # 输出: 模块文件的路径
# 默认值和注解信息
print(f"对象的默认值: {MyClass.__defaults__}")  # 输出: None
print(f"对象的注解信息: {MyClass.__annotations__}")  # 输出: {}

# 测试 greet 方法
print(obj.greet())  # 输出: 你好，我是 示例。

# 测试 __slots__
try:
    obj.age = 30  # 尝试添加未声明的属性
except AttributeError as e:
    print(f"错误: {e}")  # 输出: 'MyClass' object has no attribute 'age'
