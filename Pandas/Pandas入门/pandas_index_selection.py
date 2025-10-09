#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pandas索引和选择操作完全演示
基于《Pandas高级操作完全指南》的"索引和选择"章节

作者: AI Assistant
日期: 2024
"""

import pandas as pd
import numpy as np

def main():
    print("=== Pandas 索引和选择操作完全演示 ===\n")

    # 创建示例数据
    df = pd.DataFrame({
        '姓名': ['张三', '李四', '王五', '赵六', '钱七', '孙八'],
        '年龄': [25, 30, 35, 28, 32, 27],
        '部门': ['技术', '销售', '技术', '市场', '技术', '销售'],
        '工资': [8000, 12000, 15000, 10000, 13000, 11000],
        '城市': ['北京', '上海', '广州', '深圳', '杭州', '成都']
    }, index=['emp1', 'emp2', 'emp3', 'emp4', 'emp5', 'emp6'])

    print("示例数据:")
    print(df)
    print(f"\n数据形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")
    print(f"索引: {df.index.tolist()}")

    # ==================== 2.1 基础索引操作 ====================
    print("\n" + "="*60)
    print("2.1 基础索引操作")
    print("="*60)

    # === 列选择 ===
    print("\n--- 列选择 ---")

    # 单列选择
    print("1. 单列选择 (df['姓名']):")
    print(df['姓名'])
    print(f"类型: {type(df['姓名'])}")

    # 多列选择
    print("\n2. 多列选择 (df[['姓名', '年龄', '工资']]):")
    print(df[['姓名', '年龄', '工资']])
    print(f"类型: {type(df[['姓名', '年龄', '工资']])}")

    # 使用 loc 和 iloc 选择列
    print("\n3. 使用 loc 选择所有行的姓名和工资列:")
    print(df.loc[:, ['姓名', '工资']])

    print("\n4. 使用 iloc 选择所有行的第1,3列 (姓名和部门):")
    print(df.iloc[:, [0, 2]])

    # === 行选择 ===
    print("\n--- 行选择 ---")

    # 使用位置索引 (iloc)
    print("1. 使用 iloc[2] (第3行，索引为2):")
    print(df.iloc[2])
    print(f"类型: {type(df.iloc[2])}")

    print("\n2. 使用 iloc[1:4] (第2-4行，索引1到3):")
    print(df.iloc[1:4])

    # 使用标签索引 (loc)
    print("\n3. 使用 loc['emp3'] (标签为emp3的行):")
    print(df.loc['emp3'])

    print("\n4. 使用 loc['emp2':'emp5'] (标签范围emp2到emp5):")
    print(df.loc['emp2':'emp5'])

    # === 混合选择 ===
    print("\n--- 混合选择 ---")

    # 选择特定行列
    print("1. 选择 emp2-emp4 的姓名和工资:")
    print(df.loc['emp2':'emp4', ['姓名', '工资']])

    print("\n2. 使用 iloc 选择第2-4行的第1,3列:")
    print(df.iloc[1:4, [0, 2]])

    print("\n3. 使用 loc 选择 emp2, emp4, emp6 的年龄和城市:")
    print(df.loc[['emp2', 'emp4', 'emp6'], ['年龄', '城市']])

    print("\n4. 使用 iloc 选择第1,3,5行的第2,4列:")
    print(df.iloc[[0, 2, 4], [1, 3]])

    # === 索引切片技巧 ===
    print("\n--- 索引切片技巧 ---")

    print("1. 选择前3行 (head(3)):")
    print(df.head(3))

    print("\n2. 选择后3行 (tail(3)):")
    print(df.tail(3))

    print("\n3. 选择偶数行 (步长为2):")
    print(df.iloc[::2])

    print("\n4. 选择奇数行:")
    print(df.iloc[1::2])

    # ==================== 2.2 高级索引技巧 ====================
    print("\n" + "="*60)
    print("2.2 高级索引技巧")
    print("="*60)

    # === 条件索引 ===
    print("\n--- 条件索引 ---")

    # 单条件筛选
    print("1. 年龄大于30的员工:")
    age_above_30 = df[df['年龄'] > 30]
    print(age_above_30)

    # 多条件筛选 (与)
    print("\n2. 年龄在25-30之间且部门为技术的员工:")
    condition = (df['年龄'] >= 25) & (df['年龄'] <= 30) & (df['部门'] == '技术')
    tech_age_range = df[condition]
    print(tech_age_range)

    # 多条件筛选 (或)
    print("\n3. 部门为技术或销售的员工:")
    tech_sales = df[(df['部门'] == '技术') | (df['部门'] == '销售')]
    print(tech_sales)

    # 使用 isin 方法
    print("\n4. 城市在北京或上海的员工:")
    cities = ['北京', '上海']
    city_filtered = df[df['城市'].isin(cities)]
    print(city_filtered)

    print("\n5. 年龄不在25-30范围的员工:")
    age_outside = df[~((df['年龄'] >= 25) & (df['年龄'] <= 30))]
    print(age_outside)

    # === 字符串条件索引 ===
    print("\n--- 字符串条件索引 ---")

    # 姓名包含特定字符
    print("1. 姓名包含'张'或'李'的员工:")
    name_pattern = df['姓名'].str.contains('张|李')
    print(df[name_pattern])

    # 部门以特定字符开头
    print("\n2. 部门以'技'开头的员工:")
    tech_dept = df['部门'].str.startswith('技')
    print(df[tech_dept])

    # 城市以特定字符结尾
    print("\n3. 城市以'海'结尾的员工:")
    sea_city = df['城市'].str.endswith('海')
    print(df[sea_city])

    print("\n4. 姓名长度等于2的员工:")
    name_len_2 = df['姓名'].str.len() == 2
    print(df[name_len_2])

    # === 使用 query 方法 ===
    print("\n--- 使用 query 方法 ---")

    print("1. 使用 query 筛选年龄>30且工资>10000:")
    query_result = df.query("年龄 > 30 and 工资 > 10000")
    print(query_result)

    # 使用变量
    min_age = 28
    max_salary = 14000
    print(f"\n2. 使用变量筛选 (年龄 > {min_age} and 工资 < {max_salary}):")
    query_vars = df.query(f"年龄 > {min_age} and 工资 < {max_salary}")
    print(query_vars)

    # 复杂query表达式
    print("\n3. 复杂query表达式 (部门 in ['技术', '销售'] and 年龄 < 30):")
    complex_query = df.query("部门 in ['技术', '销售'] and 年龄 < 30")
    print(complex_query)

    # === 布尔索引组合 ===
    print("\n--- 布尔索引组合 ---")

    # 多个条件组合
    print("1. 复杂条件组合:")
    complex_mask = (
        (df['年龄'] > 25) &
        (df['年龄'] < 35) &
        (df['工资'] > 9000) &
        (df['城市'].isin(['北京', '上海', '深圳']))
    )
    complex_result = df[complex_mask]
    print(complex_result)

    # 使用布尔运算符
    print("\n2. 嵌套条件:")
    nested_condition = df[
        (df['部门'] == '技术') &
        ((df['年龄'] < 30) | (df['工资'] > 12000))
    ]
    print(nested_condition)

    # ==================== 2.3 多级索引操作 ====================
    print("\n" + "="*60)
    print("2.3 多级索引操作")
    print("="*60)

    # 创建多级索引数据
    print("创建多级索引数据...")
    arrays = [
        ['2024', '2024', '2024', '2025', '2025', '2025'],
        ['Q1', 'Q2', 'Q3', 'Q1', 'Q2', 'Q3'],
        ['技术部', '销售部', '技术部', '市场部', '技术部', '销售部']
    ]
    index = pd.MultiIndex.from_arrays(arrays, names=['年份', '季度', '部门'])

    multi_df = pd.DataFrame({
        '收入': [100, 120, 110, 80, 130, 140],
        '支出': [60, 70, 65, 50, 75, 80],
        '利润': [40, 50, 45, 30, 55, 60],
        '员工数': [10, 15, 12, 8, 14, 16]
    }, index=index)

    print("\n多级索引数据:")
    print(multi_df)
    print(f"索引层级数: {multi_df.index.nlevels}")
    print(f"层级名称: {multi_df.index.names}")

    # === 访问多级索引 ===
    print("\n--- 访问多级索引 ---")

    # 选择特定年份
    print("1. 2024年的数据:")
    data_2024 = multi_df.loc['2024']
    print(data_2024)

    # 选择特定年份和季度
    print("\n2. 2024年Q2的数据:")
    data_2024_q2 = multi_df.loc[('2024', 'Q2')]
    print(data_2024_q2)

    # 选择完整路径
    print("\n3. 2024年Q2销售部的完整数据:")
    specific_data = multi_df.loc[('2024', 'Q2', '销售部')]
    print(specific_data)

    # === 使用 xs 方法 ===
    print("\n--- 使用 xs 方法 ---")

    print("1. 使用 xs 选择所有Q1数据:")
    q1_data = multi_df.xs('Q1', level='季度')
    print(q1_data)

    print("\n2. 选择所有技术部数据:")
    tech_data = multi_df.xs('技术部', level='部门')
    print(tech_data)

    print("\n3. 选择2024年技术部数据:")
    tech_2024 = multi_df.xs(('2024', '技术部'), level=['年份', '部门'])
    print(tech_2024)

    # === 多级索引切片 ===
    print("\n--- 多级索引切片 ---")

    # 使用 slice 对象
    print("1. 2024年Q1-Q2的所有数据:")
    slice_data = multi_df.loc[('2024', slice('Q1', 'Q2')), :]
    print(slice_data)

    print("\n2. 所有年份Q1的技术部数据:")
    tech_q1 = multi_df.loc[(slice(None), 'Q1', '技术部'), :]
    print(tech_q1)

    # 选择特定列
    print("\n3. 2024年所有季度的收入和利润:")
    income_profit = multi_df.loc['2024', ['收入', '利润']]
    print(income_profit)

    # === 高级多级索引操作 ===
    print("\n--- 高级多级索引操作 ---")

    # 重排层级
    print("1. 重排层级顺序:")
    reordered = multi_df.swaplevel(0, 1)  # 交换年份和季度
    print(reordered.head())

    # 按特定层级排序
    print("\n2. 按季度排序:")
    sorted_by_q = multi_df.sort_index(level='季度')
    print(sorted_by_q)

    # 获取层级值
    print("\n3. 获取所有部门:")
    departments = multi_df.index.get_level_values('部门').unique()
    print(f"部门列表: {departments.tolist()}")

    print("\n4. 获取所有年份:")
    years = multi_df.index.get_level_values('年份').unique()
    print(f"年份列表: {years.tolist()}")

    # ==================== 实用技巧演示 ====================
    print("\n" + "="*60)
    print("实用技巧演示")
    print("="*60)

    # 技巧1: 链式选择
    print("\n1. 链式选择 (先筛选部门，再选择列):")
    chain_selection = df[df['部门'] == '技术'][['姓名', '工资']]
    print(chain_selection)

    # 技巧2: 使用between方法
    print("\n2. 使用between方法选择年龄范围:")
    age_between = df[df['年龄'].between(25, 30)]
    print(age_between)

    # 技巧3: 使用nlargest/nsmallest
    print("\n3. 工资最高的3名员工:")
    top3_salary = df.nlargest(3, '工资')
    print(top3_salary)

    print("\n4. 年龄最小的2名员工:")
    young2 = df.nsmallest(2, '年龄')
    print(young2)

    # 技巧4: 条件选择后重置索引
    print("\n5. 条件选择后重置索引:")
    filtered_reset = df[df['工资'] > 10000].reset_index(drop=True)
    print(filtered_reset)

    # 技巧5: 选择数据类型
    print("\n6. 选择数值类型的列:")
    numeric_cols = df.select_dtypes(include=[np.number])
    print(f"数值列: {numeric_cols.columns.tolist()}")
    print(numeric_cols)

    print("\n7. 选择字符串类型的列:")
    string_cols = df.select_dtypes(include=['object'])
    print(f"字符串列: {string_cols.columns.tolist()}")
    print(string_cols)

    # 技巧6: 复杂条件选择
    print("\n8. 复杂条件选择示例:")
    complex_example = df[
        (df['年龄'].between(25, 35)) &  # 年龄范围
        (df['工资'] > df['工资'].median()) &  # 工资高于中位数
        (~df['城市'].isin(['成都']))  # 不在成都
    ]
    print(complex_example)

    # ==================== 性能对比演示 ====================
    print("\n" + "="*60)
    print("性能对比演示")
    print("="*60)

    # 创建大数据集进行性能对比
    print("创建大数据集进行性能对比...")
    large_df = pd.DataFrame({
        'ID': range(10000),
        'Category': np.random.choice(['A', 'B', 'C', 'D'], 10000),
        'Value1': np.random.randn(10000),
        'Value2': np.random.randint(1, 100, 10000),
        'Flag': np.random.choice([True, False], 10000)
    })

    # 方法1: 直接索引
    print("\n1. 直接索引方法:")
    result1 = large_df[large_df['Value1'] > 0]
    print(f"结果行数: {len(result1)}")

    # 方法2: query方法
    print("\n2. query方法:")
    result2 = large_df.query('Value1 > 0')
    print(f"结果行数: {len(result2)}")

    # 方法3: loc方法
    print("\n3. loc方法:")
    result3 = large_df.loc[large_df['Value1'] > 0]
    print(f"结果行数: {len(result3)}")

    # 验证结果一致性
    print(f"\n结果一致性验证: {result1.equals(result2) and result2.equals(result3)}")

    # ==================== 常见错误和解决方案 ====================
    print("\n" + "="*60)
    print("常见错误和解决方案")
    print("="*60)

    # 错误1: SettingWithCopyWarning
    print("\n1. SettingWithCopyWarning 演示:")
    print("错误做法: df[df['年龄'] > 30]['工资'] = 99999  # 会产生警告")

    # 正确做法
    print("正确做法: df.loc[df['年龄'] > 30, '工资'] = 99999")
    df_correct = df.copy()
    df_correct.loc[df_correct['年龄'] > 30, '工资'] = 99999
    print("修改后的结果:")
    print(df_correct[df_correct['年龄'] > 30][['姓名', '年龄', '工资']])

    # 错误2: 混淆标签和位置索引
    print("\n2. 标签索引 vs 位置索引:")
    print("df.iloc[0] - 按位置选择第1行")
    print("df.loc[df.index[0]] - 按标签选择第一个索引对应的行")

    # 错误3: 多条件时忘记括号
    print("\n3. 多条件时的括号问题:")
    print("错误: df['年龄'] > 25 & df['工资'] < 15000  # 语法错误")
    print("正确: (df['年龄'] > 25) & (df['工资'] < 15000)  # 需要括号")

    # 错误4: 使用or/and而不是|/&
    print("\n4. 逻辑运算符选择:")
    print("错误: df['年龄'] > 25 or df['工资'] > 10000  # 不适用")
    print("正确: (df['年龄'] > 25) | (df['工资'] > 10000)  # 使用|和&")

    print("\n" + "="*60)
    print("索引和选择操作演示完成!")
    print("="*60)

    # 总结
    print("\n【操作总结】")
    print("✓ 基础索引: loc (标签), iloc (位置), [] (列选择)")
    print("✓ 条件筛选: 布尔索引, query方法, isin方法")
    print("✓ 字符串操作: str.contains, str.startswith, str.endswith")
    print("✓ 多级索引: xs方法, slice对象, 层级操作")
    print("✓ 实用技巧: between, nlargest/nsmallest, select_dtypes")
    print("✓ 性能优化: 向量化操作, 避免链式赋值")
    print("✓ 错误避免: 正确使用loc/iloc, 注意运算符优先级")

    print(f"\n原始数据形状: {df.shape}")
    print(f"多级索引数据形状: {multi_df.shape}")
    print("所有演示均成功完成! 🎉")

if __name__ == "__main__":
    main()