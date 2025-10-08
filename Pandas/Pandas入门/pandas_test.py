import pandas as pd

# Pandas 让数据处理变得简单
# 就像 Excel 表格，但功能更强大

# 创建一个简单的表格（DataFrame）
data = {
    '姓名': ['张三', '李四', '王五', '赵六'],
    '年龄': [25, 30, 35, 28],
    '城市': ['北京', '上海', '广州', '深圳'],
    '工资': [8000, 12000, 15000, 10000]
}

df = pd.DataFrame(data)
print("我们的第一个数据表:")
print(df)

print("\n数据表的统计摘要:")
print(df.describe())

# 瞬间就能进行各种操作
print("\n平均工资:", df['工资'].mean())
print("年龄最大的员工:", df.loc[df['年龄'].idxmax(), '姓名'])