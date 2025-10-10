#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pandas数据排序和修改完全演示
基于《Pandas高级操作完全指南》的"数据排序"和"数据修改"章节

作者: AI Assistant
日期: 2024
"""

import pandas as pd
import numpy as np

def main():
    print("=== Pandas 数据排序和修改完全演示 ===\n")

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

    # ==================== 第7章: 数据排序 ====================
    print("\n" + "="*60)
    print("第7章: 数据排序 (Sorting)")
    print("="*60)

    # 7.1 基础排序
    print("\n7.1 基础排序")
    print("-" * 30)

    # 按单列排序
    print("1. 按年龄升序排序:")
    age_sorted = df.sort_values('年龄')
    print(age_sorted[['姓名', '年龄', '部门']])

    print("\n2. 按工资降序排序:")
    salary_sorted = df.sort_values('工资', ascending=False)
    print(salary_sorted[['姓名', '工资', '部门']])

    print("\n3. 按城市升序排序:")
    city_sorted = df.sort_values('城市')
    print(city_sorted[['姓名', '城市', '工资']])

    # 按多列排序
    print("\n4. 按部门升序，工资降序排序:")
    multi_sorted = df.sort_values(['部门', '工资'], ascending=[True, False])
    print(multi_sorted[['姓名', '部门', '工资']])

    print("\n5. 按年龄降序，工资升序排序:")
    multi_sorted2 = df.sort_values(['年龄', '工资'], ascending=[False, True])
    print(multi_sorted2[['姓名', '年龄', '工资']])

    # 7.2 高级排序
    print("\n7.2 高级排序")
    print("-" * 30)

    # 按索引排序
    print("1. 按索引降序排序:")
    index_sorted = df.sort_index(ascending=False)
    print(index_sorted[['姓名', '年龄']])

    print("\n2. 按索引升序排序:")
    index_sorted_asc = df.sort_index(ascending=True)
    print(index_sorted_asc[['姓名', '年龄']])

    # 稳定排序
    print("\n3. 稳定排序 (保持原始顺序):")
    stable_sorted = df.sort_values(['部门', '年龄'], kind='stable')
    print(stable_sorted[['姓名', '部门', '年龄']])

    # 自定义排序
    print("\n4. 自定义排序:")
    df_sorted = df.copy()
    # 自定义排序规则：技术部优先，然后按年龄排序
    dept_priority = {'技术': 1, '销售': 2, '市场': 3}
    df_sorted['部门优先级'] = df_sorted['部门'].map(dept_priority)
    custom_sorted = df_sorted.sort_values(['部门优先级', '年龄'])
    print(custom_sorted[['姓名', '部门', '年龄', '部门优先级']])

    # 按字符串长度排序
    print("\n5. 按姓名长度排序:")
    name_length_sorted = df.copy()
    name_length_sorted['姓名长度'] = name_length_sorted['姓名'].str.len()
    name_sorted_result = name_length_sorted.sort_values('姓名长度')
    print(name_sorted_result[['姓名', '姓名长度']])

    # 按日期排序
    print("\n6. 按入职日期排序:")
    date_sorted = df.sort_values('入职日期')
    print(date_sorted[['姓名', '入职日期']])

    # 7.3 排名和排序
    print("\n7.3 排名和排序")
    print("-" * 30)

    # 创建带分数的数据
    score_df = pd.DataFrame({
        '姓名': ['张三', '李四', '王五', '赵六', '钱七', '孙八'],
        '数学': [85, 92, 78, 95, 88, 76],
        '英语': [90, 85, 88, 82, 91, 79],
        '物理': [88, 89, 92, 85, 87, 83]
    })

    print("成绩数据:")
    print(score_df)

    # 计算总分
    score_df['总分'] = score_df[['数学', '英语', '物理']].sum(axis=1)
    print("\n添加总分后:")
    print(score_df)

    # 不同排名方法
    print("\n不同排名方法:")

    # 默认排名 (平均排名)
    score_df['总分排名'] = score_df['总分'].rank(method='average', ascending=False)
    print("\n1. 平均排名 (method='average'):")
    print(score_df[['姓名', '总分', '总分排名']])

    # 最小排名
    score_df['最小排名'] = score_df['总分'].rank(method='min', ascending=False)
    print("\n2. 最小排名 (method='min'):")
    print(score_df[['姓名', '总分', '最小排名']])

    # 密集排名
    score_df['密集排名'] = score_df['总分'].rank(method='dense', ascending=False)
    print("\n3. 密集排名 (method='dense'):")
    print(score_df[['姓名', '总分', '密集排名']])

    # 第一排名
    score_df['第一排名'] = score_df['总分'].rank(method='first', ascending=False)
    print("\n4. 第一排名 (method='first'):")
    print(score_df[['姓名', '总分', '第一排名']])

    # 按多列排名
    print("\n5. 按多列排名:")
    # 先按数学排名，数学相同时按英语排名
    score_df['综合得分'] = score_df['数学'] * 0.4 + score_df['英语'] * 0.3 + score_df['物理'] * 0.3
    score_df['综合排名'] = score_df['综合得分'].rank(method='min', ascending=False)
    print("按综合得分排名:")
    print(score_df[['姓名', '数学', '英语', '物理', '综合得分', '综合排名']].sort_values('综合排名'))

    # 分组排名
    print("\n6. 分组排名:")
    # 为成绩数据添加班级信息
    score_df['班级'] = ['A班', 'B班', 'A班', 'B班', 'A班', 'B班']
    score_df['班级内排名'] = score_df.groupby('班级')['总分'].rank(ascending=False, method='dense')
    print("班级内排名:")
    print(score_df[['姓名', '班级', '总分', '班级内排名']].sort_values(['班级', '班级内排名']))


    # ==================== 第8章: 数据修改 ====================
    print("\n" + "="*60)
    print("第8章: 数据修改 (Data Modification)")
    print("="*60)

    # 8.1 基础修改操作
    print("\n8.1 基础修改操作")
    print("-" * 30)

    # 复制数据用于修改
    df_modified = df.copy()
    print("原始数据:")
    print(df_modified)
    # 修改单个值
    print("1. 修改单个值:")
    print(f"修改前 emp2 的年龄: {df_modified.loc['emp2', '年龄']}")
    df_modified.loc['emp2', '年龄'] = 31
    print(f"修改后 emp2 的年龄: {df_modified.loc['emp2', '年龄']}")

    print("\n2. 修改单个值 (使用iat):")
    print(f"修改前 emp1 的工资: {df_modified.iloc[0, 3]}")
    df_modified.iat[0, 3] = 8500  # 第一行第四列
    print(f"修改后 emp1 的工资: {df_modified.iloc[0, 3]}")

    # 修改整列
    print("\n3. 修改整列 - 给所有人涨薪10%:")
    print("修改前工资:")
    print(df_modified[['姓名', '工资']])
    df_modified['工资'] = df_modified['工资'] * 1.1
    print("\n修改后工资:")
    print(df_modified[['姓名', '工资']])

    # 修改满足条件的值
    print("\n4. 条件修改 - 给技术部门员工额外涨薪5%:")
    tech_mask = df_modified['部门'] == '技术'
    print("技术部门员工涨薪前:")
    print(df_modified[tech_mask][['姓名', '部门', '工资']])

    df_modified.loc[tech_mask, '工资'] = df_modified.loc[tech_mask, '工资'] * 1.05
    print("\n技术部门员工涨薪后:")
    print(df_modified[tech_mask][['姓名', '部门', '工资']])

    # 8.2 高级修改操作
    print("\n8.2 高级修改操作")
    print("-" * 30)

    # 使用 map 方法修改
    print("1. 使用 map 修改部门名称:")
    dept_mapping = {
        '技术': '研发部',
        '销售': '市场部',
        '市场': '营销部'
    }
    print("修改前部门:")
    print(df_modified[['姓名', '部门']].head())

    df_modified['部门'] = df_modified['部门'].map(dept_mapping)
    print("\n修改后部门:")
    print(df_modified[['姓名', '部门']])

    # 使用 apply 方法修改
    print("\n2. 使用 apply 修改工资等级:")
    def salary_grade(salary):
        if salary < 10000:
            return '初级'
        elif salary < 15000:
            return '中级'
        else:
            return '高级'

    df_modified['工资等级'] = df_modified['工资'].apply(salary_grade)
    print(df_modified[['姓名', '工资', '工资等级']])

    # 使用 replace 方法
    print("\n3. 使用 replace 修改城市名称:")
    city_mapping = {
        '北京': '北京市',
        '上海': '上海市',
        '广州': '广州市',
        '深圳': '深圳市',
        '杭州': '杭州市',
        '成都': '成都市'
    }
    print("修改前城市:")
    print(df_modified[['姓名', '城市']].head())

    df_modified['城市'] = df_modified['城市'].replace(city_mapping)
    print("\n修改后城市:")
    print(df_modified[['姓名', '城市']])

    # 条件修改
    print("\n4. 条件修改:")
    # 年龄大于30的员工标记为'资深'
    df_modified['经验等级'] = '普通'
    df_modified.loc[df_modified['年龄'] > 30, '经验等级'] = '资深'
    print(df_modified[['姓名', '年龄', '经验等级']])

    # 多条件修改
    print("\n5. 多条件修改:")
    # 研发部且工资高于12000的员工标记为'核心员工'
    df_modified['员工类型'] = '普通员工'
    conditions = (df_modified['部门'] == '研发部') & (df_modified['工资'] > 12000)
    df_modified.loc[conditions, '员工类型'] = '核心员工'
    print(df_modified[['姓名', '部门', '工资', '员工类型']])

    # 8.3 批量修改
    print("\n8.3 批量修改")
    print("-" * 30)

    # 批量添加新列
    print("1. 批量添加新列:")
    df_modified['入职年份'] = df_modified['入职日期'].dt.year
    df_modified['工作年限'] = 2024 - df_modified['入职年份']
    df_modified['绩效工资'] = df_modified['工资'] * 0.2
    df_modified['总薪酬'] = df_modified['工资'] + df_modified['绩效工资']

    print("添加的列:")
    print(df_modified[['姓名', '入职年份', '工作年限', '绩效工资', '总薪酬']])

    # 根据其他列批量修改
    print("\n2. 根据其他列批量修改 - 计算奖金:")
    def calculate_bonus(row):
        base_bonus = row['总薪酬'] * 0.1
        experience_bonus = row['工作年限'] * 200
        dept_bonus = {'研发部': 1000, '市场部': 800, '营销部': 600}

        return base_bonus + experience_bonus + dept_bonus.get(row['部门'], 0)

    df_modified['奖金'] = df_modified.apply(calculate_bonus, axis=1)
    print(df_modified[['姓名', '部门', '工作年限', '奖金']])

    # 使用 eval 进行动态计算
    print("\n3. 使用 eval 进行动态计算:")
    df_modified['税前收入'] = df_modified.eval('总薪酬 + 奖金')
    df_modified['税率'] = df_modified['税前收入'].apply(lambda x: 0.1 if x < 15000 else 0.15)
    df_modified['税后收入'] = df_modified.eval('税前收入 * (1 - 税率)')

    print("动态计算结果:")
    print(df_modified[['姓名', '总薪酬', '奖金', '税前收入', '税率', '税后收入']])

    # ==================== 高级修改技巧 ====================
    print("\n" + "="*60)
    print("高级修改技巧")
    print("="*60)

    # 技巧1: 使用where方法
    print("\n1. 使用 where 方法进行条件修改:")
    df_where = df.copy()
    # 如果年龄大于30，保持原值，否则改为30
    df_where['调整年龄'] = df_where['年龄'].where(df_where['年龄'] > 30, 30)
    print("年龄调整 (小于30的设为30):")
    print(df_where[['姓名', '年龄', '调整年龄']])

    # 技巧2: 使用mask方法 (where的反向)
    print("\n2. 使用 mask 方法:")
    df_mask = df.copy()
    # 如果年龄小于30，保持原值，否则改为30
    df_mask['限制年龄'] = df_mask['年龄'].mask(df_mask['年龄'] > 30, 30)
    print("年龄限制 (大于30的设为30):")
    print(df_mask[['姓名', '年龄', '限制年龄']])

    # 技巧3: 批量替换
    print("\n3. 批量替换操作:")
    df_replace = df.copy()
    # 批量替换多个值
    df_replace['部门'] = df_replace['部门'].replace({
        '技术': 'Technology',
        '销售': 'Sales',
        '市场': 'Marketing'
    })
    print("部门英文化:")
    print(df_replace[['姓名', '部门']])

    # 技巧4: 分类转换
    print("\n4. 分类转换:")
    df_cut = df.copy()
    # 将年龄分为不同段
    df_cut['年龄段'] = pd.cut(df_cut['年龄'],
                             bins=[0, 25, 30, 35, 100],
                             labels=['25岁以下', '25-30岁', '31-35岁', '35岁以上'])
    print("年龄分段:")
    print(df_cut[['姓名', '年龄', '年龄段']])

    # 技巧5: 字符串操作批量修改
    print("\n5. 字符串操作批量修改:")
    df_str = df.copy()
    # 批量添加前缀
    df_str['姓名_格式化'] = '员工_' + df_str['姓名']
    # 批量提取城市首字母
    df_str['城市代码'] = df_str['城市'].str[0]
    print("字符串格式化:")
    print(df_str[['姓名', '姓名_格式化', '城市', '城市代码']])

    # ==================== 性能优化技巧 ====================
    print("\n" + "="*60)
    print("性能优化技巧")
    print("="*60)

    # 创建大数据集进行性能对比
    print("创建大数据集进行性能测试...")
    large_df = pd.DataFrame({
        'ID': range(10000),
        'Value': np.random.randn(10000),
        'Category': np.random.choice(['A', 'B', 'C', 'D'], 10000),
        'Score': np.random.randint(50, 100, 10000)
    })

    # 性能对比1: apply vs 向量化操作
    print("\n1. apply vs 向量化操作性能对比:")

    # 方法1: apply
    import time
    start_time = time.time()
    large_df['Score_Apply'] = large_df['Score'].apply(lambda x: x * 2 + 10)
    apply_time = time.time() - start_time

    # 方法2: 向量化
    start_time = time.time()
    large_df['Score_Vectorized'] = large_df['Score'] * 2 + 10
    vectorized_time = time.time() - start_time

    print(f"apply方法耗时: {apply_time:.4f}秒")
    print(f"向量化方法耗时: {vectorized_time:.4f}秒")
    print(f"性能提升: {apply_time/vectorized_time:.2f}倍")

    # 性能对比2: loc vs where
    print("\n2. loc vs where 性能对比:")
    threshold = large_df['Score'].median()

    # 方法1: loc
    start_time = time.time()
    large_df_copy1 = large_df.copy()
    large_df_copy1.loc[large_df_copy1['Score'] > threshold, 'HighScore_Loc'] = 1
    large_df_copy1.loc[large_df_copy1['Score'] <= threshold, 'HighScore_Loc'] = 0
    loc_time = time.time() - start_time

    # 方法2: where
    start_time = time.time()
    large_df_copy2 = large_df.copy()
    large_df_copy2['HighScore_Where'] = (large_df_copy2['Score'] > threshold).astype(int)
    where_time = time.time() - start_time

    print(f"loc方法耗时: {loc_time:.4f}秒")
    print(f"where方法耗时: {where_time:.4f}秒")
    print(f"性能提升: {loc_time/where_time:.2f}倍")

    # ==================== 常见错误和最佳实践 ====================
    print("\n" + "="*60)
    print("常见错误和最佳实践")
    print("="*60)

    # 错误1: SettingWithCopyWarning
    print("\n1. SettingWithCopyWarning 演示:")
    print("❌ 错误做法 (会产生警告):")
    print("   df_filtered = df[df['年龄'] > 30]")
    print("   df_filtered['工资'] = 99999  # 可能在副本上操作")

    print("\n✅ 正确做法:")
    print("   df.loc[df['年龄'] > 30, '工资'] = 99999")
    print("   # 或者使用 copy()")
    print("   df_filtered = df[df['年龄'] > 30].copy()")
    print("   df_filtered['工资'] = 99999")

    # 演示正确做法
    df_correct = df.copy()
    df_correct.loc[df_correct['年龄'] > 30, '工资'] = 99999
    print("\n正确修改结果:")
    print(df_correct[df_correct['工资'] == 99999][['姓名', '年龄', '工资']])

    # 错误2: 链式赋值
    print("\n2. 链式赋值问题:")
    print("❌ 错误做法:")
    print("   df[df['部门'] == '技术']['工资'] = df[df['部门'] == '技术']['工资'] * 1.1")

    print("\n✅ 正确做法:")
    print("   df.loc[df['部门'] == '技术', '工资'] *= 1.1")

    # 最佳实践1: 使用assign方法
    print("\n3. 使用 assign 方法进行链式操作:")
    result_assign = df.assign(
        工资调整=lambda x: x['工资'] * 1.1,
        工资等级=lambda x: pd.cut(x['工资调整'],
                                bins=[0, 10000, 15000, 20000],
                                labels=['初级', '中级', '高级'])
    )
    print("assign方法结果:")
    print(result_assign[['姓名', '工资', '工资调整', '工资等级']].head())

    # 最佳实践2: 使用eval进行复杂计算
    print("\n4. 使用 eval 进行复杂计算:")
    df_eval = df.copy()
    df_eval['绩效指数'] = df_eval.eval('(工资 / 年龄) * (城市 == "北京") * 1.2')
    print("eval计算结果:")
    print(df_eval[['姓名', '工资', '年龄', '城市', '绩效指数']])

    print("\n" + "="*60)
    print("数据排序和修改演示完成!")
    print("="*60)

    # 总结
    print("\n【操作总结】")
    print("📊 数据排序:")
    print("  ✓ 基础排序: sort_values() - 单列/多列/升序/降序")
    print("  ✓ 索引排序: sort_index() - 按标签排序")
    print("  ✓ 高级排序: 自定义排序/稳定排序/字符串长度排序")
    print("  ✓ 排名计算: rank() - 多种排名方法/分组排名")

    print("\n🔧 数据修改:")
    print("  ✓ 基础修改: loc/iat - 单值/整列/条件修改")
    print("  ✓ 高级修改: map/apply/replace - 函数式修改")
    print("  ✓ 批量修改: 多列添加/动态计算/eval表达式")
    print("  ✓ 条件修改: where/mask - 条件式值替换")

    print("\n⚡ 性能优化:")
    print("  ✓ 优先使用向量化操作而非apply")
    print("  ✓ 合理使用where/mask替代loc条件赋值")
    print("  ✓ 使用assign进行链式操作")
    print("  ✓ 避免SettingWithCopyWarning")

    print(f"\n原始数据形状: {df.shape}")
    print(f"修改后数据形状: {df_modified.shape}")
    print("所有排序和修改演示均成功完成! 🎯")

if __name__ == "__main__":
    main()