import pandas as pd
import numpy as np
import os

# 设置输出文件目录
OUTPUT_DIR = 'io_files'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=== Pandas 数据输入输出 ===")
print(f"输出文件将保存到: {OUTPUT_DIR}/")

# 创建示例数据
sample_data = pd.DataFrame({
    '姓名': ['张三', '李四', '王五', '赵六', '钱七'],
    '年龄': [25, 30, 35, 28, 32],
    '部门': ['技术', '销售', '技术', '市场', '技术'],
    '工资': [8000, 12000, 15000, 10000, 13000],
    '入职日期': pd.date_range('2024-01-01', periods=5)
})

print("示例数据:")
print(sample_data)

# === CSV 文件操作 ===
print("\n=== CSV 文件操作 ===")

# 保存为 CSV，默认不保存索引
csv_filename = os.path.join(OUTPUT_DIR, 'employees.csv')
sample_data.to_csv(csv_filename, index=False, encoding='utf-8-sig')
print(f"已保存到 {csv_filename}")

# 读取 CSV
df_from_csv = pd.read_csv(csv_filename)
print("从 CSV 读取的数据:")
print(df_from_csv)

# 带参数的 CSV 操作，保持索引列
print("\n带参数的 CSV 操作:")
csv_with_index = os.path.join(OUTPUT_DIR, 'employees_with_index.csv')
sample_data.to_csv(csv_with_index, index=True, encoding='utf-8')
df_with_index = pd.read_csv(csv_with_index, index_col=0)
print("带索引的 CSV 读取:")
print(df_with_index)

# === Excel 文件操作 ===
print("\n=== Excel 文件操作 ===")

# 保存为 Excel
excel_filename = os.path.join(OUTPUT_DIR, 'employees.xlsx')
sample_data.to_excel(excel_filename, index=False, sheet_name='员工信息')
print(f"已保存到 {excel_filename}")

# 读取 Excel
df_from_excel = pd.read_excel(excel_filename, sheet_name='员工信息')
print("从 Excel 读取的数据:")
print(df_from_excel)

# 多工作表操作
print("\n多工作表操作:")
multi_sheet_file = os.path.join(OUTPUT_DIR, 'multi_sheet.xlsx')
with pd.ExcelWriter(multi_sheet_file) as writer:
    sample_data.to_excel(writer, sheet_name='基本信息', index=False)
    sample_data.groupby('部门')['工资'].sum().to_excel(writer, sheet_name='部门汇总')

# 读取多工作表
multi_sheets = pd.read_excel(multi_sheet_file, sheet_name=['基本信息', '部门汇总'])
print("读取的多工作表:")
for sheet_name, df in multi_sheets.items():
    print(f"\n{sheet_name}:")
    print(df)


# 高级 I/O 操作
print("\n=== 高级 I/O 操作 ===")

# === JSON 文件操作 ===
print("\n=== JSON 文件操作 ===")

# 转换为 JSON
json_data = sample_data.to_json(orient='records', force_ascii=False, indent=2)
print("JSON 格式数据:")
print(json_data)

# 保存 JSON
json_file = os.path.join(OUTPUT_DIR, 'employees.json')
with open(json_file, 'w', encoding='utf-8') as f:
    f.write(json_data)

# 读取 JSON
df_from_json = pd.read_json(json_file, encoding='utf-8')
print("从 JSON 读取的数据:")
print(df_from_json)

# === 数据库操作 ===
print("\n=== 数据库操作 ===")

# 模拟数据库操作（需要 sqlite3）
import sqlite3

# 创建内存数据库
conn = sqlite3.connect(':memory:')

# 保存到数据库
sample_data.to_sql('employees', conn, index=False, if_exists='replace')
print("数据已保存到数据库")

# 从数据库读取
df_from_db = pd.read_sql('SELECT * FROM employees WHERE 工资 > 10000', conn)
print("从数据库读取的高薪员工:")
print(df_from_db)

# === 分块读写大数据 ===
print("\n=== 分块读写大数据 ===")

# 创建大数据集
large_data = pd.DataFrame({
    'ID': range(10000),
    '值': np.random.randn(10000),
    '类别': np.random.choice(['A', 'B', 'C', 'D'], 10000)
})

# 分块写入
chunk_size = 2000
chunk_files = []
for i, chunk in enumerate(np.array_split(large_data, len(large_data) // chunk_size)):
    chunk_file = os.path.join(OUTPUT_DIR, f'large_data_chunk_{i}.csv')
    chunk.to_csv(chunk_file, index=False)
    chunk_files.append(chunk_file)
    print(f"已写入第 {i+1} 块数据，大小: {len(chunk)} 行")

# 分块读取
chunk_list = []
for chunk_file in chunk_files[:2]:  # 只读取前两块作为示例
    chunk = pd.read_csv(chunk_file)
    chunk_list.append(chunk)
    print(f"读取 {os.path.basename(chunk_file)}: {len(chunk)} 行")

combined_data = pd.concat(chunk_list, ignore_index=True)
print(f"合并后数据大小: {len(combined_data)} 行")