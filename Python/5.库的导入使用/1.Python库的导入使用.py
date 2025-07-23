# 导入内置的 math 模块
import math

# 导入第三方库 requests 和 pandas，并为 pandas 起别名
import requests
import pandas as pd

# 1. 使用 math 模块
print("Math 模块的常用函数:")
print(f"圆周率: {math.pi}")
print(f"平方根 (16): {math.sqrt(16)}")
print(f"正弦 (30度): {math.sin(math.radians(30))}")

# 2. 使用 requests 库
try:
    response = requests.get('https://api.github.com')
    print("\nRequests 库的 GET 请求:")
    print(f"状态码: {response.status_code}")
    print(f"响应内容: {response.json()}")
except requests.exceptions.RequestException as e:
    print(f"请求错误: {e}")

# 3. 使用 pandas 库
# 创建一个简单的 DataFrame
data = {
    '名字': ['Alice', 'Bob', 'Charlie'],
    '年龄': [24, 30, 22]
}
df = pd.DataFrame(data)

print("\nPandas 数据框:")
print(df)

# 4. 精确导入 datetime 模块
from datetime import datetime

# 获取当前时间
current_time = datetime.now()
print(f"\n当前时间: {current_time}")

# 5. 使用 dir() 函数查看库的功能
print("\nMath 模块的可用函数:")
print(dir(math))

print("\nRequests 库的可用函数:")
print(dir(requests))

print("\nPandas 库的可用函数:")
print(dir(pd))

# 6. 查看库的常用属性
print("\nRequests 库的版本:")
print(requests.__version__)

print("\nPandas 库的文档说明:")
print(pd.__doc__)

print("\nRequests 库的文件路径:")
print(requests.__file__)

# 7. 使用别名调用 pandas 的方法
csv_data = pd.DataFrame({
    '城市': ['北京', '上海', '广州'],
    '人口': [21540000, 24150000, 14000000]
})

# 打印 DataFrame
print("\n城市人口数据:")
print(csv_data)
