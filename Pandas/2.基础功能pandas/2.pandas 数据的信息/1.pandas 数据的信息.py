import pandas as pd

# 从 Excel 文件读取数据
df = pd.read_excel('https://gairuo.com/file/data/team.xlsx')

# 查看数据的基本属性
print("数据的前五行:\n", df.head())
print("\n数据的后五行:\n", df.tail())
print("\n数据的基本信息:")
df.info()

# 数据的形状
print("\n数据的形状:", df.shape)  # 返回 (行数, 列数)

# 每列的数据类型
print("\n每列的数据类型:\n", df.dtypes)

# 行列索引内容
print("\n行列索引内容:\n", df.axes)

# 索引对象
print("\n索引对象:", df.index)
print("\n列索引:", df.columns)

# 数据的 NumPy 数组形式
print("\n数据的 NumPy 数组形式:\n", df.values)

# 数据的维度和总元素数量
print("\n数据的维度:", df.ndim)
print("\n数据的总元素数量:", df.size)

# 检查数据是否为空
print("\n数据是否为空:", df.empty)

# 随机查看数据
print("\n随机查看一条数据:\n", df.sample(1))
print("\n随机查看三条数据:\n", df.sample(3))

# 取出某一列形成 Series
s = df['Q1']
print("\n取出的 Series:\n", s)

# Series 的基本属性
print("\nSeries 的数据类型:", s.dtype)
print("\nSeries 的索引:", s.index)
print("\nSeries 是否包含缺失值:", s.hasnans)

# DataFrame 的其他信息
print("\n第一个非NA/空值的索引:", df.first_valid_index())
print("最后一个非NA/空值的索引:", df.last_valid_index())

# 数据的标记信息
print("\n数据的标记信息:\n", df.flags)

# 设置元信息
df.attrs = {'info': '学生成绩表'}
print("\n元信息:\n", df.attrs)

# Series 独有的属性
print("\nSeries 的名称:", s.name)
print("\nSeries 的数组形式:\n", s.array)
print("\nSeries 是否有空值:", s.hasnans)
