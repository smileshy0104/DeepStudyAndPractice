class MyClass:
    # 类属性：类属性是由类所有实例共享的属性，定义在类体内部但在任何方法之外。
    class_attribute = "I am a class attribute"

    # 实例属性：实例属性是与具体实例关联的属性，通常在 init 方法中定义。
    def __init__(self, instance_attribute):
        # 实例属性
        self.instance_attribute = instance_attribute

    # 实例方法：实例方法是与具体实例关联的方法，第一个参数通常是 self，表示实例本身。
    def instance_method(self):
        print(f"This is an instance method. Instance attribute: {self.instance_attribute}")

    # 类方法：类方法是与类关联的方法，第一个参数通常是 cls，表示类本身。——类方法使用
    @classmethod
    def class_method(cls):
        print(f"This is a class method. Class attribute: {cls.class_attribute}")

    # 静态方法：静态方法是与类关联的方法，但——不绑定任何实例或类——。静态方法使用
    @staticmethod
    def static_method():
        print("This is a static method.")

# 创建类的实例
obj = MyClass("I am an instance attribute")

# 访问实例方法
obj.instance_method() # This is an instance method. Instance attribute: I am an instance attribute

# 访问类方法
MyClass.class_method() # This is a class method. Class attribute: I am a class attribute

# 访问静态方法
MyClass.static_method() # This is a static method.
