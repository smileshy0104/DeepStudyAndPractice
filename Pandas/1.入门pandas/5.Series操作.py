import pandas as pd
import numpy as np

# 1. 类似 ndarray 的操作
s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8])
print("——————————切片操作——————————")
# 切片操作
print("s[3]:", s[3])  # 访问索引为 3 的元素
print("s[2:]:", s[2:])  # 从索引 2 开始到结束
print("Median:", s.median())  # 计算中位数
print("筛选大于中位数的元素:", s[s > s.median()])  # 筛选大于中位数的元素
print("s[[1, 2, 1]]:", s[[1, 2, 1]])  # 指定索引的内容
print("数据类型:", s.dtype)  # 获取数据类型
print("返回值的数列:", s.array)  # 返回值的数列（内容、length、dtype）
print("转为 NumPy ndarray:", s.to_numpy())  # 转为 NumPy ndarray
print("3 in s:", 3 in s)  # 检测值是否在 Series 中


# 2. 类似字典的操作
print("——————————类似字典的操作——————————")
s = pd.Series([14.22, 21.34, 5.18], index=['中国', '美国', '日本'], name='人口')

print("\n中国的人口:", s['中国'])  # 通过 key 取值
# s['印度']  # 如果没有该索引会报 KeyError
s['印度'] = 13.54  # 直接增加一个新数据
print("印度的人口:", s['印度'])  # 输出印度的人口
print("'法国' in s:", '法国' in s)  # 检测索引

# 3. 向量计算和标签对齐
print("——————————向量计算和标签对齐——————————")
s = pd.Series([1, 2, 3, 4])

print("\ns + s:", s + s)  # 同索引相加
print("s * 2:", s * 2)  # 同索引相乘
print("s[1:] + s[:-1]:", s[1:] + s[:-1])  # 部分计算
print("np.exp(s):", np.exp(s))  # 求 e 的幂次方

# 4. 名称属性
print("——————————名称属性——————————")
s = pd.Series([1, 2, 3, 4], name='数字')
print("\ns.name:", s.name)  # 获取 Series 名称
s = s.rename("number")  # 修改名称
print("修改后的名称:", s.name)  # 输出修改后的名称
s2 = s.rename("number")  # 修改名称并赋值给新变量

# 5. 其他操作
print("——————————其他操作——————————")
s = pd.Series([1, 2, 3, 4], name='数字')
print("\ns.add(1):", s.add(1))  # 每个元素加 1
print("s.add_prefix(3):", s.add_prefix('3_'))  # 给索引前加个 3，升位
print("s.add_suffix(4):", s.add_suffix('_4'))  # 在后增加 4
print("总和:", s.sum())  # 计算总和
print("数量:", s.count())  # 计算数量
print("标准差:", s.agg('std'))  # 聚合，返回标准差
print("最大最小值:", s.agg(['min', 'max']))  # 聚合，返回最大最小值

# 创建另一个 Series
s2 = pd.Series([5, 6, 7, 8], name='数字2')

# 连接 Series
aligned_s1, aligned_s2 = s.align(s2)  # 联接
print("\n联接后的 Series:")
print(aligned_s1)
print(aligned_s2)

# 逻辑运算
print("是否有为假的:", s.any())  # 是否有为假的
print("是否全是真:", s.all())  # 是否全是真

# 追加另一个 Series
s_appended = s.append(s2, ignore_index=True)  # 追加 Series
print("\n追加后的 Series:")
print(s_appended)

# 应用方法
print("\n每个元素加 1:")
print(s.apply(lambda x: x + 1))  # 应用方法
print("是否为空:", s.empty)  # 检查是否为空

# 深拷贝
s3 = s.copy()  # 深拷贝
print("\n深拷贝的 Series:")
print(s3)
