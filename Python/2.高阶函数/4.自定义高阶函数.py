def apply_function(func, value):
    return func(value)

def double(x):
    return x * 2

result = apply_function(double, 5)
print(result)  # 输出: 10
