import time
from functools import wraps

# 定义一个装饰器
def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 记录开始时间
        result = func(*args, **kwargs)  # 调用原函数
        end_time = time.time()  # 记录结束时间
        execution_time = end_time - start_time  # 计算执行时间
        print(f"{func.__name__} 执行时间: {execution_time:.4f} 秒")
        return result  # 返回原函数的结果
    return wrapper

# 使用装饰器
@timing_decorator
def example_function(n):
    # 模拟一些耗时的操作
    time.sleep(n)
    return f"完成 {n} 秒的工作"

# 调用装饰过的函数
result = example_function(2)
print(result)
