import pandas as pd
import numpy as np

# DataFrame 深度解析
print("\n=== DataFrame 深度解析 ===")

# 创建 DataFrame 的多种方式
print("\n创建 DataFrame:")

# 方式1：从字典创建（列为主）
data1 = {
    '姓名': ['张三', '李四', '王五'],
    '年龄': [25, 30, 35],
    '城市': ['北京', '上海', '广州']
}
df1 = pd.DataFrame(data1)
print("从字典创建:")
print(df1)

# 方式2：从列表创建（行为主）
data2 = [
    ['张三', 25, '北京', '技术'],
    ['李四', 30, '上海', '销售'],
    ['王五', 35, '广州', '技术']
]
df2 = pd.DataFrame(data2,
                   columns=['姓名', '年龄', '城市', '部门'])
print("\n从列表创建:")
print(df2)

# 方式3：从 NumPy 数组创建
data3 = np.random.randint(0, 100, (4, 3))
df3 = pd.DataFrame(data3,
                   columns=['语文', '数学', '英语'],
                   index=['学生A', '学生B', '学生C', '学生D'])
print("\n从 NumPy 数组创建:")
print(df3)

# 方式4：从字典列表创建
data4 = [
    {'姓名': '张三', '年龄': 25, '城市': '北京'},
    {'姓名': '李四', '年龄': 30, '城市': '上海'},
    {'姓名': '王五', '年龄': 35, '城市': '广州'}
]
df4 = pd.DataFrame(data4)
print("\n从字典列表创建:")
print(df4)

# DataFrame 的基本属性
print("\nDataFrame 基本属性:")
df = pd.DataFrame({
    '姓名': ['张三', '李四', '王五', '赵六', '钱七'],
    '年龄': [25, 30, 35, 28, 32],
    '部门': ['技术', '销售', '技术', '市场', '技术'],
    '工资': [8000, 12000, 15000, 10000, 13000]
})

print("数据形状:", df.shape)  # (行数, 列数)
print("数据类型:")
print(df.dtypes)
print("列名:", df.columns.tolist())
print("索引:", df.index.tolist())
print("数据值:")
print(df.values)

# DataFrame 的数据访问
print("\nDataFrame 数据访问:")

# 访问列
print("访问单个列 (df['姓名']):", df['姓名'].tolist())
print("访问多个列 (df[['姓名', '工资']]):")
print(df[['姓名', '工资']])

# 访问行
print("\n访问行:")
print("第0行 (df.iloc[0]):")
print(df.iloc[0])

print("姓名为'张三'的行 (df.loc[df['姓名'] == '张三']):")
print(df.loc[df['姓名'] == '张三'])

# 访问特定单元格
print("\n访问单元格:")
print("第1行第2列 (df.iloc[1, 2]):", df.iloc[1, 2])
print("姓名为'李四'的工资 (df.loc[1, '工资']):", df.loc[1, '工资'])
