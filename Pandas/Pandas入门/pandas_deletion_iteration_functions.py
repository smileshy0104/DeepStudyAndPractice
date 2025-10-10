#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pandas数据删除、迭代和函数应用完全演示
基于《Pandas高级操作完全指南》的第9、10、11章

作者: AI Assistant
日期: 2024
"""

import pandas as pd
import numpy as np
import time

def main():
    print("=== Pandas 数据删除、迭代和函数应用完全演示 ===\n")

    # 创建员工数据
    df = pd.DataFrame({
        '姓名': ['张三', '李四', '王五', '赵六', '钱七', '孙八'],
        '年龄': [25, 30, 35, 28, 32, 27],
        '部门': ['技术', '销售', '技术', '市场', '技术', '销售'],
        '工资': [8000, 12000, 15000, 10000, 13000, 11000],
        '城市': ['北京', '上海', '广州', '深圳', '杭州', '成都'],
        '入职日期': pd.date_range('2024-01-01', periods=6)
    }, index=['emp1', 'emp2', 'emp3', 'emp4', 'emp5', 'emp6'])

    print("原始员工数据:")
    print(df)
    print(f"\n数据形状: {df.shape}")

    # ==================== 第9章: 数据删除 ====================
    print("\n" + "="*60)
    print("第9章: 数据删除 (Dropping Data)")
    print("="*60)

    # 9.1 删除行和列
    print("\n9.1 删除行和列")
    print("-" * 30)

    # 创建用于删除操作的数据副本
    df_drop = df.copy()

    # 删除行
    print("1. 删除指定行 (emp3 和 emp5):")
    df_drop_rows = df_drop.drop(['emp3', 'emp5'])
    print(df_drop_rows[['姓名', '年龄', '部门']])

    # 按条件删除行
    print("\n2. 删除年龄大于30的员工:")
    df_drop_filtered = df_drop.drop(df_drop[df_drop['年龄'] > 30].index)
    print(df_drop_filtered[['姓名', '年龄', '部门']])

    # 删除列
    print("\n3. 删除城市列:")
    df_drop_cols = df_drop.drop('城市', axis=1)
    # 显示结果
    print(df_drop_cols.head(3))

    # 删除多列
    print("\n4. 删除多列 (入职日期和城市):")
    df_drop_multi_cols = df_drop.drop(['入职日期', '城市'], axis=1)
    print(df_drop_multi_cols.head(3))

    # 按位置删除列
    print("\n5. 按位置删除列 (删除第2、4列):")
    df_drop_by_position = df_drop.drop(df_drop.columns[[1, 3]], axis=1)
    print(df_drop_by_position.head(3))

    # 9.2 高级删除操作
    print("\n9.2 高级删除操作")
    print("-" * 30)

    # 删除重复行
    print("1. 删除重复行:")
    # 创建包含重复行的数据
    df_with_duplicates = pd.concat([df, df.iloc[2:4]], ignore_index=True)
    print("包含重复行的数据:")
    print(df_with_duplicates)

    print("\n删除重复行 (保留第一个):")
    df_no_duplicates = df_with_duplicates.drop_duplicates()
    print(df_no_duplicates)

    print("\n删除重复行 (保留最后一个):")
    df_no_duplicates_last = df_with_duplicates.drop_duplicates(keep='last')
    print(df_no_duplicates_last)

    # 按特定列删除重复
    print("\n2. 按部门删除重复 (保留第一个):")
    df_dept_unique = df.drop_duplicates(subset=['部门'], keep='first')
    print(df_dept_unique[['姓名', '部门']])

    print("\n按部门和年龄删除重复:")
    df_dept_age_unique = df.drop_duplicates(subset=['部门', '年龄'], keep='first')
    print(df_dept_age_unique[['姓名', '部门', '年龄']])

    # 删除空值
    print("\n3. 删除空值:")
    # 创建包含空值的数据
    df_with_nulls = df.copy()
    df_with_nulls.loc['emp1', '年龄'] = np.nan
    df_with_nulls.loc['emp3', '工资'] = np.nan
    df_with_nulls.loc['emp5', '城市'] = np.nan

    print("包含空值的数据:")
    print(df_with_nulls)

    print("\n删除包含空值的行:")
    df_no_nulls = df_with_nulls.dropna()
    print(df_no_nulls)

    print("\n删除特定列为空的行 (年龄和工资):")
    df_no_nulls_subset = df_with_nulls.dropna(subset=['年龄', '工资'])
    print(df_no_nulls_subset)

    print("\n删除全为空的行:")
    df_no_all_nulls = df_with_nulls.dropna(how='all')
    print(df_no_all_nulls)

    print("\n删除空值超过50%的列:")
    df_few_nulls = df_with_nulls.dropna(axis=1, thresh=len(df_with_nulls) * 0.5)
    print(f"剩余列: {df_few_nulls.columns.tolist()}")

    # 9.3 条件删除
    print("\n9.3 条件删除")
    print("-" * 30)

    # 创建扩展数据
    df_extended = pd.DataFrame({
        'ID': range(1, 11),
        '姓名': [f'员工{i}' for i in range(1, 11)],
        '年龄': [25, 30, 35, 28, 32, 45, 22, 38, 50, 27],
        '工资': [8000, 12000, 15000, 10000, 13000, 18000, 7000, 14000, 20000, 9000],
        '部门': ['技术', '销售', '技术', '市场', '技术', '管理', '技术', '销售', '管理', '市场'],
        '状态': ['在职', '在职', '离职', '在职', '在职', '在职', '离职', '在职', '在职', '离职']
    })

    print("扩展数据:")
    print(df_extended)

    # 删除离职员工
    print("\n1. 删除离职员工:")
    active_employees = df_extended.drop(df_extended[df_extended['状态'] == '离职'].index)
    print(active_employees)

    # 删除年龄过大的员工
    print("\n2. 删除年龄大于45岁的员工:")
    age_filtered = df_extended.drop(df_extended[df_extended['年龄'] > 45].index)
    print(age_filtered)

    # 删除工资过低的员工
    print("\n3. 删除工资小于8000的员工:")
    salary_filtered = df_extended.drop(df_extended[df_extended['工资'] < 8000].index)
    print(salary_filtered)

    # 组合条件删除
    print("\n4. 组合条件删除 (年龄>40或工资>15000):")
    complex_filter = (df_extended['年龄'] > 40) | (df_extended['工资'] > 15000)
    complex_filtered = df_extended.drop(df_extended[complex_filter].index)
    print(complex_filtered)

    # 使用 query 删除
    print("\n5. 使用 query 保留年龄在25-30之间的员工:")
    query_filtered = df_extended.query("25 <= 年龄 <= 30")
    print(query_filtered)

    # ==================== 第10章: 数据迭代 ====================
    print("\n" + "="*60)
    print("第10章: 数据迭代 (Data Iteration)")
    print("="*60)

    # 10.1 基础迭代方法
    print("\n10.1 基础迭代方法")
    print("-" * 30)

    # iterrows() - 逐行迭代 (慢但不推荐)
    print("1. 使用 iterrows() 逐行迭代 (前3行):")
    count = 0
    for index, row in df.iterrows():
        if count < 3:
            print(f"索引: {index}, 姓名: {row['姓名']}, 年龄: {row['年龄']}")
            count += 1
        else:
            break

    # itertuples() - 更快的行迭代
    print("\n2. 使用 itertuples() 逐行迭代 (前3行):")
    count = 0
    for row in df.itertuples():
        if count < 3:
            print(f"索引: {row.Index}, 姓名: {row.姓名}, 年龄: {row.年龄}")
            count += 1
        else:
            break

    # items() - 按列迭代
    print("\n3. 使用 items() 按列迭代:")
    print("\n列数据类型:")
    for column_name, column_data in df.items():
        print(f"列名: {column_name}, 数据类型: {column_data.dtype}")
        if column_name in ['姓名', '年龄']:
            print(f"  数据: {column_data.tolist()}")
            break

    # 10.2 高效迭代方法
    print("\n10.2 高效迭代方法")
    print("-" * 30)

    # 使用 apply 进行高效操作
    print("1. 使用 apply 计算工资等级:")
    def get_salary_grade(salary):
        if salary < 10000:
            return 'C'
        elif salary < 13000:
            return 'B'
        else:
            return 'A'

    df['工资等级'] = df['工资'].apply(get_salary_grade)
    print(df[['姓名', '工资', '工资等级']])

    # 使用向量化操作替代迭代
    print("\n2. 使用向量化操作替代迭代:")
    # 计算BMI指数
    bmi_data = pd.DataFrame({
        '姓名': ['张三', '李四', '王五'],
        '体重(kg)': [70, 80, 65],
        '身高(m)': [1.75, 1.80, 1.70]
    })

    # 慢方法 (迭代)
    def calculate_bmi_slow(df):
        bmi_list = []
        for _, row in df.iterrows():
            bmi = row['体重(kg)'] / (row['身高(m)'] ** 2)
            bmi_list.append(round(bmi, 2))
        return bmi_list

    # 快方法 (向量化)
    def calculate_bmi_fast(df):
        return (df['体重(kg)'] / (df['身高(m)'] ** 2)).round(2)

    print("BMI数据:")
    print(bmi_data)

    bmi_slow = calculate_bmi_slow(bmi_data)
    bmi_fast = calculate_bmi_fast(bmi_data)

    print(f"\n迭代方法: {bmi_slow}")
    print(f"向量化方法: {bmi_fast.tolist()}")
    print(f"结果相同: {bmi_slow == bmi_fast.tolist()}")

    # 性能对比
    print("\n3. 性能对比:")
    # 创建大数据集
    large_df = pd.DataFrame({
        'A': np.random.randn(1000),
        'B': np.random.randn(1000),
        'C': np.random.randn(1000)
    })

    # iterrows方法
    start_time = time.time()
    result_slow = []
    for _, row in large_df.iterrows():
        result_slow.append(row['A'] + row['B'] + row['C'])
    slow_time = time.time() - start_time

    # 向量化方法
    start_time = time.time()
    result_fast = large_df['A'] + large_df['B'] + large_df['C']
    fast_time = time.time() - start_time

    print(f"iterrows方法耗时: {slow_time:.4f}秒")
    print(f"向量化方法耗时: {fast_time:.4f}秒")
    print(f"性能提升: {slow_time/fast_time:.2f}倍")

    # 10.3 条件迭代
    print("\n10.3 条件迭代")
    print("-" * 30)

    # 创建条件数据
    condition_df = pd.DataFrame({
        '产品': ['A', 'B', 'C', 'D', 'E'],
        '销量': [100, 200, 50, 300, 150],
        '价格': [10, 20, 30, 15, 25]
    })

    print("产品数据:")
    print(condition_df)

    # 使用 loc 进行条件赋值
    print("\n1. 使用 loc 进行条件赋值:")
    condition_df.loc[condition_df['销量'] > 150, '状态'] = '热销'
    condition_df.loc[condition_df['销量'] <= 150, '状态'] = '一般'
    print(condition_df)

    # 使用 where 方法
    print("\n2. 使用 where 方法:")
    condition_df['调整价格'] = condition_df['价格'].where(
        condition_df['销量'] > 100,
        condition_df['价格'] * 0.9
    )
    print(condition_df)

    # 使用 mask 方法 (where 的反向)
    print("\n3. 使用 mask 方法:")
    condition_df['高价标记'] = condition_df['价格'].mask(
        condition_df['价格'] < 20,
        '低价'
    ).mask(
        condition_df['价格'] >= 20,
        '高价'
    )
    print(condition_df)

    # ==================== 第11章: 函数应用 ====================
    print("\n" + "="*60)
    print("第11章: 函数应用 (Function Application)")
    print("="*60)

    # 11.1 apply 方法
    print("\n11.1 apply 方法")
    print("-" * 30)

    # 创建示例数据
    func_df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50],
        'C': [100, 200, 300, 400, 500]
    })

    print("示例数据:")
    print(func_df)

    # 对列应用函数
    print("\n1. 对每列应用函数:")
    print("对每列求和:")
    col_sum = func_df.apply(lambda x: x.sum())
    print(col_sum)

    print("\n对每列求均值:")
    col_mean = func_df.apply(lambda x: x.mean())
    print(col_mean)

    print("\n对每列求最大值:")
    col_max = func_df.apply(lambda x: x.max())
    print(col_max)

    # 对行应用函数
    print("\n2. 对每行应用函数:")
    print("对每行求和:")
    row_sum = func_df.apply(lambda x: x.sum(), axis=1)
    print(row_sum)

    print("\n对每行求均值:")
    row_mean = func_df.apply(lambda x: x.mean(), axis=1)
    print(row_mean)

    # 应用自定义函数
    print("\n3. 应用自定义函数:")
    def custom_operation(row):
        """自定义行操作"""
        return row['A'] * row['B'] + row['C']

    func_df['计算结果'] = func_df.apply(custom_operation, axis=1)
    print(func_df)

    # 应用多个函数
    print("\n4. 对列应用多个函数:")
    multi_func = func_df[['A', 'B', 'C']].apply(['sum', 'mean', 'std'])
    print(multi_func)

    # 11.2 applymap 方法
    print("\n11.2 applymap 方法")
    print("-" * 30)

    print("1. 对每个元素应用平方根函数:")
    sqrt_data = func_df[['A', 'B', 'C']].map(lambda x: x ** 0.5)
    print(sqrt_data.round(3))

    print("\n2. 对每个元素应用条件函数:")
    def conditional_value(x):
        if x < 100:
            return x * 2
        elif x < 300:
            return x * 1.5
        else:
            return x

    conditional_data = func_df[['A', 'B', 'C']].map(conditional_value)
    print(conditional_data)

    print("\n3. 对每个元素应用字符串操作:")
    string_df = pd.DataFrame({
        'first_name': ['John', 'Jane', 'Bob'],
        'last_name': ['Doe', 'Smith', 'Johnson']
    })
    for col in string_df.columns:
        string_df[f'uppercase_{col}'] = string_df[col].map(lambda x: x.upper() if isinstance(x, str) else x)
    print(string_df)

    # 11.3 pipe 方法
    print("\n11.3 pipe 方法")
    print("-" * 30)

    print("1. 使用 pipe 进行链式操作:")

    def add_prefix(df, prefix):
        """添加前缀到列名"""
        df.columns = [prefix + col for col in df.columns]
        return df

    def multiply_by_factor(df, factor):
        """所有数值乘以因子"""
        return df * factor

    def add_constant(df, constant):
        """所有数值加上常数"""
        return df + constant

    # 链式应用函数
    result = (func_df[['A', 'B', 'C']]
              .pipe(multiply_by_factor, 2)
              .pipe(add_constant, 10)
              .pipe(add_prefix, 'processed_'))

    print("链式处理结果:")
    print(result)

    # 复杂的 pipe 操作
    print("\n2. 复杂的 pipe 操作:")
    def data_processing_pipeline(df):
        """数据处理管道"""
        # 步骤1: 标准化
        df_normalized = (df - df.mean()) / df.std()

        # 步骤2: 计算每行的总和
        df_normalized['row_sum'] = df_normalized.sum(axis=1)

        # 步骤3: 按总和排序
        df_sorted = df_normalized.sort_values('row_sum', ascending=False)

        return df_sorted

    processed_result = func_df[['A', 'B', 'C']].pipe(data_processing_pipeline)
    print("处理管道结果:")
    print(processed_result.round(3))

    # 11.4 transform 方法
    print("\n11.4 transform 方法")
    print("-" * 30)

    # 创建分组数据
    group_df = pd.DataFrame({
        '部门': ['技术', '销售', '技术', '市场', '技术', '销售', '市场', '技术'],
        '姓名': ['张三', '李四', '王五', '赵六', '钱七', '孙八', '周九', '吴十'],
        '工资': [8000, 12000, 15000, 10000, 13000, 11000, 9000, 14000]
    })

    print("分组数据:")
    print(group_df)

    # transform 与 groupby 结合
    print("\n1. 使用 transform 计算部门平均工资:")
    group_df['部门平均工资'] = group_df.groupby('部门')['工资'].transform('mean')
    print(group_df[['姓名', '部门', '工资', '部门平均工资']])

    print("\n2. 计算工资与部门平均的差异:")
    group_df['工资差异'] = group_df['工资'] - group_df['部门平均工资']
    print(group_df[['姓名', '部门', '工资', '部门平均工资', '工资差异']])

    print("\n3. 标准化每个部门内的工资:")
    group_df['标准化工资'] = group_df.groupby('部门')['工资'].transform(
        lambda x: (x - x.mean()) / x.std()
    )
    print(group_df[['姓名', '部门', '工资', '标准化工资']].round(3))

    print("\n4. 计算部门内工资排名:")
    group_df['部门内排名'] = group_df.groupby('部门')['工资'].transform('rank', ascending=False)
    print(group_df[['姓名', '部门', '工资', '部门内排名']])

    # ==================== 实用技巧和最佳实践 ====================
    print("\n" + "="*60)
    print("实用技巧和最佳实践")
    print("="*60)

    # 技巧1: 高效的数据清洗函数
    print("\n1. 高效的数据清洗函数:")
    def clean_data(df):
        """数据清洗管道"""
        # 删除重复行
        df_clean = df.drop_duplicates()

        # 处理缺失值
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())

        # 标准化列名
        df_clean.columns = df_clean.columns.str.lower().str.replace(' ', '_')

        return df_clean

    # 创建带问题的数据进行演示
    dirty_data = pd.DataFrame({
        'Name': ['Alice', 'Bob', 'Alice', None],
        'Age': [25, 30, 25, 35],
        'Salary': [50000, None, 50000, 60000]
    })

    print("原始数据:")
    print(dirty_data)

    cleaned_data = clean_data(dirty_data)
    print("\n清洗后数据:")
    print(cleaned_data)

    # 技巧2: 性能优化的函数应用
    print("\n2. 性能优化的函数应用:")

    # 创建大数据集
    performance_df = pd.DataFrame({
        'value1': np.random.randn(10000),
        'value2': np.random.randn(10000),
        'category': np.random.choice(['A', 'B', 'C'], 10000)
    })

    # 慢方法 - apply
    start_time = time.time()
    slow_result = performance_df.apply(lambda row: row['value1'] * row['value2'], axis=1)
    slow_time = time.time() - start_time

    # 快方法 - 向量化
    start_time = time.time()
    fast_result = performance_df['value1'] * performance_df['value2']
    fast_time = time.time() - start_time

    print(f"apply方法耗时: {slow_time:.4f}秒")
    print(f"向量化方法耗时: {fast_time:.4f}秒")
    print(f"性能提升: {slow_time/fast_time:.2f}倍")

    # 技巧3: 自定义聚合函数
    print("\n3. 自定义聚合函数:")

    def custom_aggregation(group):
        """自定义聚合函数"""
        return pd.Series({
            'count': len(group),
            'mean': group['value1'].mean(),
            'std': group['value1'].std(),
            'range': group['value1'].max() - group['value1'].min()
        })

    # 对性能数据按类别分组
    agg_result = performance_df.groupby('category').apply(custom_aggregation)
    print("自定义聚合结果:")
    print(agg_result.round(3))

    # 技巧4: 链式操作的最佳实践
    print("\n4. 链式操作的最佳实践:")

    chained_result = (performance_df
                     .query('value1 > 0')
                     .assign(product=lambda x: x['value1'] * x['value2'])
                     .groupby('category')
                     .agg({
                         'value1': ['mean', 'std'],
                         'product': 'sum'
                     })
                     .round(3))

    print("链式操作结果:")
    print(chained_result)

    print("\n" + "="*60)
    print("数据删除、迭代和函数应用演示完成!")
    print("="*60)

    # 总结
    print("\n【操作总结】")
    print("🗑️ 数据删除:")
    print("  ✓ 基础删除: drop() - 行列删除/条件删除/按位置删除")
    print("  ✓ 高级删除: drop_duplicates() - 重复值处理")
    print("  ✓ 空值处理: dropna() - 多种空值删除策略")
    print("  ✓ 条件删除: query() + 索引操作")

    print("\n🔄 数据迭代:")
    print("  ✓ 基础迭代: iterrows(), itertuples(), items()")
    print("  ✓ 高效方法: apply() - 替代显式循环")
    print("  ✓ 向量化: 优先使用向量化操作")
    print("  ✓ 条件迭代: where(), mask() - 条件式操作")

    print("\n⚙️ 函数应用:")
    print("  ✓ apply: 按行/列应用函数")
    print("  ✓ applymap: 按元素应用函数")
    print("  ✓ pipe: 链式函数应用")
    print("  ✓ transform: 分组变换操作")

    print("\n⚡ 性能优化:")
    print("  ✓ 优先使用向量化操作而非循环")
    print("  ✓ 合理选择迭代方法")
    print("  ✓ 使用链式操作提高可读性")
    print("  ✓ 自定义聚合函数提高效率")

    print(f"\n原始数据形状: {df.shape}")
    print("所有删除、迭代和函数应用演示均成功完成! 🚀")

if __name__ == "__main__":
    main()