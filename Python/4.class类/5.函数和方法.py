# 定义一个独立的函数
def add(a, b):
    return a + b

# 调用函数
result = add(2, 3)
print(result)  # 输出: 5


# 定义一个类，包含实例方法、类方法和静态方法
class MyClass:
    # 实例方法
    def instance_method(self):
        print("这是实例方法")

    # 类方法
    @classmethod
    def class_method(cls):
        print("这是类方法")

    # 静态方法
    @staticmethod
    def static_method():
        print("这是静态方法")


# 实例化 MyClass 类
obj = MyClass()

# 调用实例方法：实例方法是绑定到类实例的方法，必须通过类的实例调用，并且“第一个参数通常是 self，表示实例本身”。
obj.instance_method()  # 输出: 这是实例方法

# 调用类方法：类方法是绑定到类的方法，可以通过类或类的实例调用，”第一个参数通常是 cls，表示类本身“。类方法使用 @classmethod 装饰器定义。
MyClass.class_method()  # 输出: 这是类方法
obj.class_method()      # 输出: 这是类方法

# 调用静态方法：静态方法是绑定到类的方法，不绑定任何实例或类。静态方法使用 @staticmethod 装饰器定义。
MyClass.static_method()  # 输出: 这是静态方法
obj.static_method()      # 输出: 这是静态方法
