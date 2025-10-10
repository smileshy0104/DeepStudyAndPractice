#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pandas数据筛选完全演示
基于《Pandas高级操作完全指南》的"数据筛选"章节

作者: AI Assistant
日期: 2024
"""

import pandas as pd
import numpy as np

def main():
    print("=== Pandas 数据筛选完全演示 ===\n")

    # 创建员工数据
    df = pd.DataFrame({
        '姓名': ['张三', '李四', '王五', '赵六', '钱七', '孙八'],
        '年龄': [25, 30, 35, 28, 32, 27],
        '部门': ['技术', '销售', '技术', '市场', '技术', '销售'],
        '工资': [8000, 12000, 15000, 10000, 13000, 11000],
        '城市': ['北京', '上海', '广州', '深圳', '杭州', '成都']
    }, index=['emp1', 'emp2', 'emp3', 'emp4', 'emp5', 'emp6'])

    print("员工数据:")
    print(df)
    print(f"\n数据形状: {df.shape}") # 数据形状: (6, 5)

    # ==================== 6.1 条件筛选 ====================
    print("\n" + "="*60)
    print("6.1 条件筛选")
    print("="*60)

    # 基础条件筛选
    print("\n--- 基础条件筛选 ---")

    print("1. 年龄大于30的员工:")
    # 筛选
    print(df['年龄'] > 30)
    # emp1    False
    # emp2    False
    # emp3     True
    # emp4    False
    # emp5     True
    # emp6    False
    high_age = df[df['年龄'] > 30]
    print(high_age)

    print("\n2. 工资在10000-14000之间的员工:")
    salary_range = df[(df['工资'] >= 10000) & (df['工资'] <= 14000)]
    print(salary_range)

    print("\n3. 部门为技术且年龄小于30的员工:")
    tech_young = df[(df['部门'] == '技术') & (df['年龄'] < 30)]
    print(tech_young)

    print("\n4. 年龄大于30或工资高于12000的员工:")
    condition = (df['年龄'] > 30) | (df['工资'] > 12000)
    complex_or = df[condition]
    print(complex_or)

    # 使用 isin 筛选
    print("\n--- 使用 isin 筛选 ---")

    cities = ['北京', '上海', '深圳']
    print(f"1. 城市在 {cities} 中的员工:")
    city_filtered = df[df['城市'].isin(cities)]
    print(city_filtered)

    print("\n2. 部门为技术或销售的员工:")
    dept_filtered = df[df['部门'].isin(['技术', '销售'])]
    print(dept_filtered)

    print("\n3. 年龄在特定列表中的员工:")
    target_ages = [25, 28, 32]
    age_filtered = df[df['年龄'].isin(target_ages)]
    print(age_filtered)

    # 使用 isin 的反向操作
    print("\n--- 反向筛选 ---")

    print("1. 部门不为技术和销售的员工:")
    other_dept = df[~df['部门'].isin(['技术', '销售'])]
    print(other_dept)

    print("\n2. 年龄不在25-30范围的员工:")
    age_outside = df[~df['年龄'].between(25, 30)]
    print(age_outside)

    # ==================== 6.2 字符串筛选 ====================
    print("\n" + "="*60)
    print("6.2 字符串筛选")
    print("="*60)

    # 姓名包含特定字符
    print("\n--- 字符串模式匹配 ---")

    print("1. 姓名包含'张'或'李'的员工:")
    name_filtered = df[df['姓名'].str.contains('张|李')]
    print(name_filtered)

    print("\n2. 城市以'北'开头的员工:")
    north_city = df[df['城市'].str.startswith('北')]
    print(north_city)

    print("\n3. 城市以'海'结尾的员工:")
    sea_city = df[df['城市'].str.endswith('海')]
    print(sea_city)

    print("\n4. 姓名长度等于2的员工:")
    name_len_2 = df[df['姓名'].str.len() == 2]
    print(name_len_2)

    print("\n5. 部门包含'技'字的员工:")
    tech_contains = df[df['部门'].str.contains('技')]
    print(tech_contains)

    # 正则表达式筛选
    print("\n--- 正则表达式筛选 ---")

    print("1. 姓名包含数字的员工 (模拟):")
    # 模拟一些包含数字的姓名
    df_with_nums = df.copy()
    df_with_nums.loc['emp7'] = ['员工1号', 29, '技术', 11500, '南京']
    df_with_nums.loc['emp8'] = ['张小三', 26, '销售', 9500, '西安']
    print("扩展数据:")
    print(df_with_nums)

    num_names = df_with_nums[df_with_nums['姓名'].str.contains(r'\d')]
    print("\n包含数字的姓名:")
    print(num_names)

    print("\n2. 城市名以'北'或'南'开头的员工:")
    north_south = df[df['城市'].str.match(r'^[南北]')]
    print(north_south)

    # 高级字符串操作
    print("\n--- 高级字符串操作 ---")

    print("1. 姓名中包含'三'的员工:")
    contains_three = df[df['姓名'].str.contains('三')]
    print(contains_three)

    print("\n2. 城市名包含'州'的员工:")
    contains_zhou = df[df['城市'].str.contains('州')]
    print(contains_zhou)

    # ==================== 6.3 复杂筛选 ====================
    print("\n" + "="*60)
    print("6.3 复杂筛选")
    print("="*60)

    # 多条件组合
    print("\n--- 多条件组合 ---")

    print("1. 复杂条件筛选:")
    complex_condition = (
        (df['年龄'] >= 25) &
        (df['年龄'] <= 35) &
        (df['工资'] > 10000) &
        (df['部门'].isin(['技术', '销售']))
    )
    complex_filtered = df[complex_condition]
    print(complex_filtered)

    print("\n2. 嵌套条件组合:")
    nested_condition = df[
        (df['部门'] == '技术') &
        ((df['年龄'] < 30) | (df['工资'] > 12000))
    ]
    print(nested_condition)

    print("\n3. 多重OR条件:")
    multiple_or = df[
        (df['年龄'] > 30) |
        (df['工资'] > 13000) |
        (df['城市'] == '北京')
    ]
    print(multiple_or)

    # 使用 query 方法
    print("\n--- 使用 query 方法 ---")

    print("1. 基础query筛选:")
    query_result = df.query("25 <= 年龄 <= 35 and 工资 > 10000 and 部门 in ['技术', '销售']")
    print(query_result)

    # 使用变量
    min_age = 28
    max_salary = 14000
    print(f"\n2. 使用变量的query筛选 (年龄 > {min_age} and 工资 < {max_salary}):")
    query_vars = df.query(f"年龄 > {min_age} and 工资 < {max_salary}")
    print(query_vars)

    print("\n3. 复杂query表达式:")
    complex_query = df.query("部门 == '技术' and (年龄 < 30 or 工资 > 12000)")
    print(complex_query)

    # 动态查询
    print("\n--- 动态查询 ---")

    def dynamic_query(dataframe, conditions):
        """动态构建查询条件"""
        query_parts = []

        for column, operator, value in conditions:
            if operator in ['==', '!=', '>', '<', '>=', '<=']:
                if isinstance(value, str):
                    query_parts.append(f"{column} {operator} '{value}'")
                else:
                    query_parts.append(f"{column} {operator} {value}")
            elif operator == 'in':
                if isinstance(value, list):
                    value_str = str(value).replace("'", '"')
                    query_parts.append(f"{column} in {value_str}")
            elif operator == 'contains':
                query_parts.append(f"{column}.str.contains('{value}')")

        query_string = ' and '.join(query_parts)
        return dataframe.query(query_string)

    # 使用动态查询
    conditions = [
        ('部门', 'in', ['技术', '销售']),
        ('工资', '>', 10000),
        ('年龄', '<=', 32)
    ]

    dynamic_result = dynamic_query(df, conditions)
    print("4. 动态查询结果:")
    print(dynamic_result)

    # ==================== 自定义筛选函数 ====================
    print("\n" + "="*60)
    print("自定义筛选函数")
    print("="*60)

    def smart_filter(dataframe, filters):
        """智能筛选函数"""
        result = dataframe.copy()

        for column, filter_config in filters.items():
            filter_type = filter_config.get('type', 'equals')
            value = filter_config.get('value')

            if filter_type == 'equals':
                result = result[result[column] == value]
            elif filter_type == 'range':
                min_val, max_val = value
                result = result[(result[column] >= min_val) & (result[column] <= max_val)]
            elif filter_type == 'in_list':
                result = result[result[column].isin(value)]
            elif filter_type == 'contains':
                result = result[result[column].str.contains(value, na=False)]
            elif filter_type == 'greater_than':
                result = result[result[column] > value]
            elif filter_type == 'less_than':
                result = result[result[column] < value]
            elif filter_type == 'not_equals':
                result = result[result[column] != value]
            elif filter_type == 'not_in_list':
                result = result[~result[column].isin(value)]

        return result

    # 使用自定义筛选函数
    filter_configs = {
        '部门': {'type': 'in_list', 'value': ['技术', '销售']},
        '年龄': {'type': 'range', 'value': [25, 35]},
        '工资': {'type': 'greater_than', 'value': 10000}
    }

    smart_result = smart_filter(df, filter_configs)
    print("智能筛选结果:")
    print(smart_result)

    # # 高级自定义筛选
    # print("\n--- 高级自定义筛选 ---")

    # def advanced_filter(dataframe, criteria):
    #     """高级筛选函数，支持复杂条件"""
    #     result = dataframe.copy()

    #     for criterion in criteria:
    #         criterion_type = criterion.get('type')

    #         if criterion_type == 'and_condition':
    #             # AND条件组
    #             and_mask = pd.Series([True] * len(result))
    #             for condition in criterion.get('conditions', []):
    #                 field = condition['field']
    #                 operator = condition['operator']
    #                 value = condition['value']

    #                 if operator == '>':
    #                     and_mask &= (result[field] > value)
    #                 elif operator == '<':
    #                     and_mask &= (result[field] < value)
    #                 elif operator == '>=':
    #                     and_mask &= (result[field] >= value)
    #                 elif operator == '<=':
    #                     and_mask &= (result[field] <= value)
    #                 elif operator == '==':
    #                     and_mask &= (result[field] == value)
    #                 elif operator == '!=':
    #                     and_mask &= (result[field] != value)
    #                 elif operator == 'in':
    #                     and_mask &= result[field].isin(value)
    #                 elif operator == 'contains':
    #                     and_mask &= result[field].str.contains(value, na=False)

    #             result = result[and_mask]

    #         elif criterion_type == 'or_condition':
    #             # OR条件组
    #             or_mask = pd.Series([False] * len(result))
    #             for condition in criterion.get('conditions', []):
    #                 field = condition['field']
    #                 operator = condition['operator']
    #                 value = condition['value']

    #                 if operator == '>':
    #                     or_mask |= (result[field] > value)
    #                 elif operator == '<':
    #                     or_mask |= (result[field] < value)
    #                 elif operator == '>=':
    #                     or_mask |= (result[field] >= value)
    #                 elif operator == '<=':
    #                     or_mask |= (result[field] <= value)
    #                 elif operator == '==':
    #                     or_mask |= (result[field] == value)
    #                 elif operator == '!=':
    #                     or_mask |= (result[field] != value)
    #                 elif operator == 'in':
    #                     or_mask |= result[field].isin(value)
    #                 elif operator == 'contains':
    #                     or_mask |= result[field].str.contains(value, na=False)

    #             result = result[or_mask]

    #     return result

    # # 使用高级筛选
    # advanced_criteria = [
    #     {
    #         'type': 'and_condition',
    #         'conditions': [
    #             {'field': '年龄', 'operator': '>=', 'value': 25},
    #             {'field': '年龄', 'operator': '<=', 'value': 35}
    #         ]
    #     },
    #     {
    #         'type': 'or_condition',
    #         'conditions': [
    #             {'field': '部门', 'operator': '==', 'value': '技术'},
    #             {'field': '工资', 'operator': '>', 'value': 12000}
    #         ]
    #     }
    # ]

    # advanced_result = advanced_filter(df, advanced_criteria)
    # print("\n高级筛选结果 (年龄25-35 且 (部门=技术 或 工资>12000)):")
    # print(advanced_result)

    # ==================== 实用筛选技巧 ====================
    print("\n" + "="*60)
    print("实用筛选技巧")
    print("="*60)

    # 技巧1: 链式筛选
    print("\n--- 链式筛选 ---")

    chain_result = df[df['部门'] == '技术'][df['年龄'] < 30]
    print("1. 链式筛选 (技术部门且年龄<30):")
    print(chain_result)

    # 技巧2: 使用between方法
    print("\n2. 使用between方法:")
    age_between = df[df['年龄'].between(25, 30)]
    print("年龄在25-30之间的员工:")
    print(age_between)

    # 技巧3: 使用nlargest/nsmallest结合筛选
    print("\n3. 高薪员工筛选:")
    high_salary = df[df['工资'] > df['工资'].median()]
    print("工资高于中位数的员工:")
    print(high_salary)

    # 技巧4: 空值筛选
    print("\n4. 空值筛选演示:")
    df_with_null = df.copy()
    df_with_null.loc['emp1', '年龄'] = np.nan
    df_with_null.loc['emp3', '工资'] = np.nan

    print("包含空值的数据:")
    print(df_with_null)

    print("\n年龄不为空的员工:")
    age_not_null = df_with_null[df_with_null['年龄'].notna()]
    print(age_not_null)

    print("\n年龄或工资为空的员工:")
    null_salary_or_age = df_with_null[df_with_null['年龄'].isna() | df_with_null['工资'].isna()]
    print(null_salary_or_age)

    # 技巧5: 重复值筛选
    print("\n5. 重复值筛选:")
    df_with_duplicates = pd.concat([df, df.iloc[2:4]], ignore_index=True)
    print("包含重复的数据:")
    print(df_with_duplicates)

    print("\n重复的行:")
    duplicate_rows = df_with_duplicates[df_with_duplicates.duplicated()]
    print(duplicate_rows)

    print("\n不重复的行:")
    unique_rows = df_with_duplicates[~df_with_duplicates.duplicated()]
    print(unique_rows)

    # ==================== 性能对比演示 ====================
    print("\n" + "="*60)
    print("性能对比演示")
    print("="*60)

    # 创建大数据集
    print("创建大数据集进行性能测试...")
    large_df = pd.DataFrame({
        'ID': range(10000),
        'Category': np.random.choice(['A', 'B', 'C', 'D'], 10000),
        'Value1': np.random.randn(10000),
        'Value2': np.random.randint(1, 100, 10000),
        'Flag': np.random.choice([True, False], 10000),
        'Name': [f'Item_{i}' for i in range(10000)]
    })

    # 添加一些特殊条件的数据
    large_df.loc[large_df['Value1'] > 2, 'Category'] = 'Special'

    # 性能测试
    conditions = large_df['Value1'] > 0

    print("\n1. 直接布尔索引:")
    result1 = large_df[conditions]
    print(f"筛选结果: {len(result1)} 行")

    print("\n2. 使用query方法:")
    result2 = large_df.query('Value1 > 0')
    print(f"筛选结果: {len(result2)} 行")

    print("\n3. 使用loc:")
    result3 = large_df.loc[conditions]
    print(f"筛选结果: {len(result3)} 行")

    print(f"\n结果一致性: {result1.equals(result2) and result2.equals(result3)}")

    # ==================== 常见错误和最佳实践 ====================
    print("\n" + "="*60)
    print("常见错误和最佳实践")
    print("="*60)

    # 错误1: 忘记括号
    print("\n1. 运算符优先级问题:")
    print("❌ 错误: df['年龄'] > 25 & df['工资'] < 12000  # 会导致语法错误")
    print("✅ 正确: (df['年龄'] > 25) & (df['工资'] < 12000)  # 使用括号")

    # 错误2: 使用python的and/or
    print("\n2. 逻辑运算符选择:")
    print("❌ 错误: df['年龄'] > 25 and df['工资'] < 12000  # 不适用")
    print("✅ 正确: (df['年龄'] > 25) & (df['工资'] < 12000)  # 使用&和|")

    # 错误3: 字符串比较引号问题
    print("\n3. 字符串比较:")
    print("✅ 正确: df['部门'] == '技术'")
    print("✅ 正确: df.query(\"部门 == '技术'\")")

    # 最佳实践演示
    print("\n--- 最佳实践演示 ---")

    # 实践1: 可读性优先
    print("1. 提高可读性:")
    # 将复杂条件分解
    age_condition = df['年龄'] > 25
    salary_condition = df['工资'] < 12000
    dept_condition = df['部门'].isin(['技术', '销售'])

    readable_result = df[age_condition & salary_condition & dept_condition]
    print("可读性筛选结果:")
    print(readable_result)

    # 实践2: 使用变量
    print("\n2. 使用变量提高灵活性:")
    target_departments = ['技术', '销售']
    min_salary = 10000
    max_age = 35

    flexible_result = df[
        (df['部门'].isin(target_departments)) &
        (df['工资'] >= min_salary) &
        (df['年龄'] <= max_age)
    ]
    print("灵活筛选结果:")
    print(flexible_result)

    # 实践3: 函数封装
    print("\n3. 函数封装复用:")
    def filter_by_criteria(data, departments=None, salary_range=None, age_range=None):
        """通用筛选函数"""
        result = data.copy()

        if departments:
            result = result[result['部门'].isin(departments)]

        if salary_range:
            min_sal, max_sal = salary_range
            result = result[(result['工资'] >= min_sal) & (result['工资'] <= max_sal)]

        if age_range:
            min_age, max_age = age_range
            result = result[(result['年龄'] >= min_age) & (result['年龄'] <= max_age)]

        return result

    reusable_result = filter_by_criteria(
        df,
        departments=['技术', '销售'],
        salary_range=(10000, 15000),
        age_range=(25, 35)
    )
    print("函数封装筛选结果:")
    print(reusable_result)

    print("\n" + "="*60)
    print("数据筛选演示完成!")
    print("="*60)

    # 总结
    print("\n【筛选方法总结】")
    print("✓ 基础筛选: >, <, >=, <=, ==, !=")
    print("✓ 逻辑组合: & (AND), | (OR), ~ (NOT)")
    print("✓ 成员判断: isin(), between()")
    print("✓ 字符串匹配: str.contains, str.startswith, str.endswith")
    print("✓ 空值处理: isna(), notna()")
    print("✓ 查询方法: query() - 可读性更好")
    print("✓ 高级技巧: 链式筛选, 函数封装, 性能优化")
    print("✓ 最佳实践: 括号优先, 变量分离, 可读性优先")

    print(f"\n原始数据形状: {df.shape}")
    print("所有筛选演示均成功完成! 🔍")

if __name__ == "__main__":
    main()