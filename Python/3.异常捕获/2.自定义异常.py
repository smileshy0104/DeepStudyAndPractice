# 定义一个异常类Gte10Error，用于处理幼儿园水平大于等于10的情况
class Gte10Error(Exception):
    # 初始化异常类，可以接收一个错误信息参数
    def __init__(self, ErrorInfo='幼儿园水平不能大于等于10'):
        super().__init__(self) # 初始化父类
        self.errorinfo=ErrorInfo
    # 返回错误信息的字符串表示
    def __str__(self):
        return self.errorinfo

# 定义一个加法函数add，接收两个参数x和y
def add(x, y):
    # 检查x或y是否大于等于10，如果是，则抛出Gte10Error异常
    if x >=10 or x >=10:
        raise Gte10Error
    # 如果x和y都小于10，则返回它们的和
    else:
        return x + y

# 测试add函数
print(add(1,1)) # 期望输出：2
print(add(12,1)) # 这将抛出Gte10Error异常
