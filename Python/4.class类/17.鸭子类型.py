# TODO 鸭子类型（Duck Typing）是 Python 中的一种动态类型特性，它指的是对象的————类型不重要，重要的是对象是否具有某些特定的方法或属性。
# 这种特性源于一句流行的说法：“如果它走起来像鸭子，叫起来像鸭子，那么它就是鸭子。”

# 在鸭子类型中，程序员并不关心对象的具体类型，而是关心对象是否实现了所需的方法和属性。这使得 Python 的代码更加灵活和可重用。
class Duck:
    def quack(self):
        return "嘎嘎叫"

class Dog:
    def quack(self):
        return "汪汪叫"

class Cat:
    def meow(self):
        return "喵喵叫"

def make_it_quack(animal):
    print(animal.quack())  # 不关心 animal 的具体类型，只关心是否有 quack 方法

# 创建对象
duck = Duck()
dog = Dog()
cat = Cat()

# 调用函数
make_it_quack(duck)  # 输出: 嘎嘎叫
make_it_quack(dog)   # 输出: 汪汪叫

# 下面的调用将会引发 AttributeError，因为 Cat 类没有 quack 方法
try:
    make_it_quack(cat)
except AttributeError as e:
    print(f"错误: {e}")  # 输出: 'Cat' object has no attribute 'quack'
