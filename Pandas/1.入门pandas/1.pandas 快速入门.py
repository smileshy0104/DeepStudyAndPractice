import pandas as pd  # 引入 pandas 库，按惯例起别名 pd

# 读取 Excel 文件
file_path = '../team.xlsx'  # 文件路径
# df = pd.read_excel('https://www.gairuo.com/file/data/team.xlsx')  # 从网址读取

# 尝试读取文件并处理异常
try:
    df = pd.read_excel(file_path)  # 从当前目录读取文件
    print(f"成功读取文件: {file_path}")
except FileNotFoundError:
    print(f"文件未找到: {file_path}")
    exit()

# 查看数据框的形状 (行数, 列数)
print(f"数据框形状: {df.shape}")  # 输出形状，例如 (100, 6)

# 数据探索
print("\n数据框基本信息:")
df.info()  # 显示有数据类型、索引情况、行列数、各字段数据类型、内存占用等

print("\n数值型列的汇总统计:")
# 计算出各数字字段的总数、平均数、标准差、最大最小值和四分位数：
print(df.describe())  # 查看数值型列的汇总统计

print("\n各字段类型:")
print(df.dtypes)  # 查看各字段类型

print("\n数据行和列名:")
print(f"行名: {df.index}")
print(f"列名: {df.columns}")

# 检查缺失值
missing_values = df.isnull().sum()
print("\n每列缺失值数量:")
print(missing_values[missing_values > 0])  # 只显示有缺失值的列

# 数据清洗示例（可选）
# df.dropna(inplace=True)  # 移除含有缺失值的行
# df.fillna(0, inplace=True)  # 用 0 填充缺失值

# 保存处理后的数据（可选）
# df.to_excel('cleaned_team.xlsx', index=False)  # 保存清洗后的数据
