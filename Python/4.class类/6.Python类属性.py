# 作用范围：类属性通常用于定义所有实例共享的数据。例如，动物种类、常量值、静态配置信息等。

# TODO 与实例属性区别
# 实例属性 是在类的__init__ 方法中定义的，属于特定实例，“每个实例都有自己的——独立属性”。
# 类属性 是在类体中定义的，属于类本身，在——所有实例之间共享。
class Counter:
    # 类属性，用于跟踪创建的实例数
    count = 0

    def __init__(self, name):
        # 实例属性
        self.name = name
        # 每创建一个新实例，类属性 count 增加
        Counter.count += 1

    def __del__(self):
        # 实例被销毁时，类属性 count 减少
        Counter.count -= 1

print("——————与实例属性区别———————")
# 创建三个 Counter 实例
c1 = Counter("Counter 1")
c2 = Counter("Counter 2")
c3 = Counter("Counter 3")

# 输出类属性 count
print(Counter.count)  # 输出: 3

# 销毁一个实例
del c1

# 输出类属性 count
print(Counter.count)  # 输出: 2

# TODO 增加修改
class Example:
    class_attr = 10

# 创建实例
e1 = Example()
e2 = Example()

# 访问类属性
print(Example.class_attr)  # 输出: 10
print(e1.class_attr)       # 输出: 10
print(e2.class_attr)       # 输出: 10

# 修改类属性
Example.class_attr = 20
print(Example.class_attr)  # 输出: 20
print(e1.class_attr)       # 输出: 20
print(e2.class_attr)       # 输出: 20

# 动态增加类属性
Example.new_class_attr = 30
print(Example.new_class_attr)  # 输出: 30
print(e1.new_class_attr)       # 输出: 30
print(e2.new_class_attr)       # 输出: 30

# 删除类属性
del Example.class_attr

# 尝试访问已删除的类属性
try:
    print(Example.class_attr)
except AttributeError as e:
    print(e)  # 输出: type object 'Example' has no attribute 'class_attr'

# 检查实例是否还能访问已删除的类属性
try:
    print(e1.class_attr)
except AttributeError as e:
    print(e)  # 输出: type object 'Example' has no attribute 'class_attr'