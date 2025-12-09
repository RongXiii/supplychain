#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
高级库存策略示例
演示ABC分类、动态安全库存和最小起订量的使用方法
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.system.main import ReplenishmentSystem

def generate_test_data():
    """
    生成测试数据用于演示高级库存策略
    """
    # 生成模拟销售数据
    def generate_sales_data(num_products, num_days):
        """生成模拟销售数据"""
        sales_data = []
        end_date = datetime.now()
        dates = [end_date - timedelta(days=i) for i in range(num_days)][::-1]
        
        for i in range(num_products):
            product_id = f'P{i+1}'
            for date in dates:
                # 生成随机销售数量，模拟波动
                if i < 20:  # A类产品，需求较高且稳定
                    sales_quantity = np.random.randint(50, 200)
                elif i < 50:  # B类产品，需求中等
                    sales_quantity = np.random.randint(20, 100)
                else:  # C类产品，需求较低
                    sales_quantity = np.random.randint(0, 50)
                
                sales_data.append({
                    'product_id': product_id,
                    'date': date.strftime('%Y-%m-%d'),
                    'sales_quantity': sales_quantity
                })
        
        return pd.DataFrame(sales_data)
    
    # 生成100个产品的销售数据
    sales_data = generate_sales_data(num_products=100, num_days=90)
    
    # 生成产品主数据
    products_data = {
        'product_id': [],
        'product_name': [],
        'revenue': [],
        'unit_cost': [],
        'category': []
    }
    
    # 模拟产品数据
    for i in range(100):
        product_id = f'P{i+1}'
        # 创建A类产品（高销售额）
        if i < 20:
            revenue = np.random.uniform(50000, 200000)
            unit_cost = np.random.uniform(100, 500)
            category = 'Electronics'
        # 创建B类产品（中等销售额）
        elif i < 50:
            revenue = np.random.uniform(10000, 50000)
            unit_cost = np.random.uniform(50, 150)
            category = 'Clothing'
        # 创建C类产品（低销售额）
        else:
            revenue = np.random.uniform(1000, 10000)
            unit_cost = np.random.uniform(10, 100)
            category = 'Accessories'
            
        products_data['product_id'].append(product_id)
        products_data['product_name'].append(f'Product {i+1}')
        products_data['revenue'].append(round(revenue, 2))
        products_data['unit_cost'].append(round(unit_cost, 2))
        products_data['category'].append(category)
    
    # 转换为DataFrame
    products_df = pd.DataFrame(products_data)
    
    # 生成历史需求数据（按产品ID分组）
    historical_demand = {}
    for _, row in sales_data.iterrows():
        product_id = row['product_id']
        if product_id not in historical_demand:
            historical_demand[product_id] = []
        historical_demand[product_id].append(row['sales_quantity'])
    
    return products_df, historical_demand

def main():
    """
    演示高级库存策略的主函数
    """
    print("=" * 70)
    print("🎯 高级库存策略演示")
    print("=" * 70)
    print()
    
    # 创建补货系统实例
    system = ReplenishmentSystem()
    
    # 步骤1: 生成测试数据
    print("📊 步骤1: 生成测试数据")
    print("-" * 40)
    products_df, historical_demand = generate_test_data()
    print(f"✓ 生成了 {len(products_df)} 个产品的数据")
    print(f"✓ 生成了 {len(historical_demand)} 个产品的历史需求数据")
    print()
    
    # 步骤2: 执行ABC分类
    print("🏷️  步骤2: 执行ABC分类")
    print("-" * 40)
    abc_classes = system.inventory_strategies.abc_classification(products_df)
    
    # 统计ABC分类结果
    abc_counts = {}
    for _, class_ in abc_classes.items():
        abc_counts[class_] = abc_counts.get(class_, 0) + 1
    
    print(f"ABC分类结果:")
    for class_, count in sorted(abc_counts.items()):
        percentage = (count / len(abc_classes)) * 100
        print(f"  • {class_}类产品: {count}个 ({percentage:.1f}%)")
    print()
    
    # 步骤3: 运行高级库存分析
    print("🔍 步骤3: 运行高级库存分析")
    print("-" * 40)
    analysis_result = system.run_advanced_inventory_analysis(products_df, historical_demand)
    
    print(f"分析结果摘要:")
    print(f"  • 分析产品数量: {analysis_result['total_products']}")
    print(f"  • 分析日期: {analysis_result['analysis_date']}")
    print()
    
    # 步骤4: 显示详细分析结果
    print("📋 步骤4: 显示详细分析结果")
    print("-" * 40)
    
    # 将产品分析结果转换为DataFrame以便查看
    product_analysis_df = pd.DataFrame(analysis_result['product_analysis'])
    
    # 按ABC分类分组，每组显示前3个产品
    abc_groups = product_analysis_df.groupby('abc_class')
    
    for abc_class, group in abc_groups:
        print(f"\n{abc_class}类产品分析示例:")
        print("-" * 30)
        # 按安全库存从高到低排序
        group_sorted = group.sort_values('safety_stock', ascending=False).head(3)
        
        # 格式化显示
        for _, product in group_sorted.iterrows():
            print(f"产品ID: {product['product_id']}")
            print(f"  • 安全库存: {product['safety_stock']:.2f}")
            print(f"  • 最小起订量: {product['min_order_qty']:.2f}")
            print(f"  • 月需求量: {product['monthly_demand']:.2f}")
            print(f"  • 需求CV: {product['demand_cv']:.2f}")
            print()
    
    # 步骤5: 演示与MILP优化器的集成
    print("🔄 步骤5: 与MILP优化器集成演示")
    print("-" * 40)
    
    # 准备简单的优化数据
    products = ['P1', 'P2', 'P3', 'P4', 'P5']
    warehouses = ['WH1', 'WH2']
    time_periods = 5
    
    # 生成简单的需求预测
    forecast_demands = {}
    for product in products:
        forecast_demands[product] = []
        for period in range(time_periods):
            forecast_demands[product].append(np.random.randint(10, 100))
    
    # 初始库存
    inventory_data = {
        'WH1': {'P1': 50, 'P2': 30, 'P3': 20, 'P4': 10, 'P5': 5},
        'WH2': {'P1': 20, 'P2': 40, 'P3': 30, 'P4': 20, 'P5': 15}
    }
    
    # 提前期
    lead_times = {'P1': 2, 'P2': 1, 'P3': 3, 'P4': 2, 'P5': 1}
    
    # 成本数据
    costs = {
        'ordering_cost': {'P1': 50, 'P2': 40, 'P3': 30, 'P4': 40, 'P5': 20},
        'holding_cost': {'P1': 10, 'P2': 8, 'P3': 5, 'P4': 8, 'P5': 3},
        'unit_cost': {'P1': 200, 'P2': 150, 'P3': 100, 'P4': 150, 'P5': 50}
    }
    
    # 约束条件
    constraints = {
        'budget_constraint': 10000,
        'max_order_qty': {'P1': 200, 'P2': 200, 'P3': 200, 'P4': 200, 'P5': 200}
    }
    
    # 仓库库存
    warehouse_inventory = {
        'WH1': {'capacity': 1000},
        'WH2': {'capacity': 800}
    }
    
    # 调拨成本
    transfer_costs = {
        'WH1': {'WH2': 5},  # WH1到WH2的调拨成本
        'WH2': {'WH1': 5}   # WH2到WH1的调拨成本
    }
    
    # 运行多仓库存优化（包含ABC分类）
    optimized_inventory, transfers, gaps = system.optimize_multi_warehouse_inventory(
        forecast_demands=forecast_demands,
        inventory_data=inventory_data,
        lead_times=lead_times,
        costs=costs,
        constraints=constraints,
        warehouse_inventory=warehouse_inventory,
        transfer_costs=transfer_costs,
        products_data=products_df  # 传递产品数据以进行ABC分类
    )
    
    print("✓ 库存优化完成，优化结果包含ABC分类信息")
    
    # 显示优化结果
    print("\n📊 库存优化结果：")
    print(f"  • 总调拨次数: {len(transfers)}")
    print(f"  • 平均调拨成本: {np.mean([t['cost'] for t in transfers]):.2f} 元")
    print(f"  • 总缺口: {sum(gaps):.2f}")
    
    print()
    print("=" * 70)
    print("✅ 高级库存策略演示完成")
    print("=" * 70)
    print()
    
    # 总结
    print("📝 演示总结")
    print("-" * 40)
    print("1. ABC分类: 根据产品销售额进行了A/B/C三级分类")
    print("2. 动态安全库存: 根据需求波动性和历史数据计算安全库存")
    print("3. 最小起订量: 基于ABC分类和月需求量计算最小起订量")
    print("4. 集成优化: 将高级库存策略与MILP优化器集成")
    print()
    
    return 0

if __name__ == "__main__":
    main()
