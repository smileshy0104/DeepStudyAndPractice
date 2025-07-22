class Dog:
    """这是一个🐶类"""

    # 类属性
    species = "Canis lupus familiaris"

    def __init__(self, name, age):
        # 实例属性
        self.name = name
        self.age = age

    # 实例方法
    def bark(self):
        print(f"{self.name} says woof!")

    # 类方法
    @classmethod
    def get_species(cls):
        return cls.species

    # 静态方法
    @staticmethod
    def info():
        print("Dogs are domesticated mammals, not natural wild animals.")


# 创建类的实例
my_dog = Dog("Buddy", 3)

# 访问实例属性
print(f"My dog's name is {my_dog.name} and he is {my_dog.age} years old.")

# 调用实例方法
my_dog.bark()

# 调用类方法
print(f"Dog species: {Dog.get_species()}")

# 调用静态方法
Dog.info()

print(my_dog.__doc__)  # '这是一个学生类'
print(my_dog.__dict__) # 查看类的属性，是一个字典
print(my_dog.__class__) # 所属类
print(my_dog.__class__.__name__) # 类名
print(my_dog.__module__) # 类定义所在的模块
