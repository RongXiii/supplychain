import sys
import os
from dotenv import load_dotenv

# 加载.env文件
load_dotenv()

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from src.data.data_warehouse import DataWarehouse
from src.mlops.real_time_processor import RealTimeDataProcessor
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 配置日志
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sample_data():
    """
    创建示例数据
    """
    # 创建示例产品数据
    items = pd.DataFrame({
        'sku': ['SKU001', 'SKU002', 'SKU003', 'SKU004', 'SKU005'],
        'name': ['产品A', '产品B', '产品C', '产品D', '产品E'],
        'category': ['类别1', '类别2', '类别1', '类别3', '类别2'],
        'cost': [10.0, 15.0, 20.0, 25.0, 30.0],
        'price': [20.0, 30.0, 40.0, 50.0, 60.0]
    })
    
    # 创建示例位置数据
    locations = pd.DataFrame({
        'location_id': ['LOC001', 'LOC002', 'LOC003'],
        'name': ['仓库A', '仓库B', '门店C'],
        'address': ['地址1', '地址2', '地址3']
    })
    
    # 创建示例供应商数据
    suppliers = pd.DataFrame({
        'supplier_id': ['SUP001', 'SUP002', 'SUP003'],
        'name': ['供应商A', '供应商B', '供应商C'],
        'contact': ['联系人A', '联系人B', '联系人C']
    })
    
    # 创建示例库存数据
    inventory_daily = []
    start_date = datetime.now() - timedelta(days=30)
    
    for sku in items['sku']:
        for i in range(30):
            date = start_date + timedelta(days=i)
            inventory_daily.append({
                'date': date.strftime('%Y-%m-%d'),
                'sku': sku,
                'location_id': np.random.choice(locations['location_id']),
                'inventory_on_hand': np.random.randint(50, 200),
                'inventory_in_transit': np.random.randint(0, 50),
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
    
    inventory_daily = pd.DataFrame(inventory_daily)
    
    # 创建示例采购订单数据
    purchase_orders = []
    for i in range(20):
        purchase_orders.append({
            'purchase_order_id': f'PO{i+1:04d}',
            'supplier_id': np.random.choice(suppliers['supplier_id']),
            'sku': np.random.choice(items['sku']),
            'order_quantity': np.random.randint(50, 200),
            'order_date': (start_date + timedelta(days=np.random.randint(0, 30))).strftime('%Y-%m-%d'),
            'expected_delivery_date': (start_date + timedelta(days=np.random.randint(5, 15))).strftime('%Y-%m-%d'),
            'status': np.random.choice(['pending', 'shipped', 'delivered']),
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    
    purchase_orders = pd.DataFrame(purchase_orders)
    
    # 创建示例预测数据
    forecast_output = []
    for sku in items['sku']:
        for i in range(7):
            date = datetime.now() + timedelta(days=i)
            forecast_output.append({
                'date': date.strftime('%Y-%m-%d'),
                'sku': sku,
                'forecasted_demand': np.random.randint(10, 50),
                'model_used': np.random.choice(['arima', 'holt_winters', 'prophet']),
                'confidence_interval_lower': np.random.randint(5, 25),
                'confidence_interval_upper': np.random.randint(30, 70),
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
    
    forecast_output = pd.DataFrame(forecast_output)
    
    return {
        'items': items,
        'locations': locations,
        'suppliers': suppliers,
        'inventory_daily': inventory_daily,
        'purchase_orders': purchase_orders,
        'forecast_output': forecast_output
    }

def main():
    """
    演示数据仓库、数据血缘追踪和实时数据支持功能
    """
    print("=" * 70)
    print("📦 供应链数据管理系统演示")
    print("=" * 70)
    print("\n本演示展示了以下功能：")
    print("1. 统一数据仓库：建立集中式数据仓库，整合所有供应链数据")
    print("2. 数据血缘追踪：记录数据来源、处理过程和使用情况")
    print("3. 实时数据支持：整合实时数据流，支持动态更新")
    print("=" * 70)
    
    # 1. 初始化数据仓库
    print("\n\n🚀 第1步：初始化数据仓库")
    print("-" * 40)
    data_warehouse = DataWarehouse()
    print("✅ 数据仓库已初始化")
    
    # 2. 创建示例数据
    print("\n\n🚀 第2步：创建示例数据")
    print("-" * 40)
    simulated_tables = create_sample_data()
    print(f"✅ 创建了 {len(simulated_tables)} 个示例数据表")
    
    for table_name, df in simulated_tables.items():
        print(f"  • {table_name}: {df.shape[0]:>4} 行 × {df.shape[1]:>2} 列")
    
    # 3. 加载数据到数据仓库
    print("\n\n🚀 第3步：加载数据到统一数据仓库")
    print("-" * 40)
    
    for table_name, df in simulated_tables.items():
        data_warehouse.update_data(table_name, df, update_type='replace')
        print(f"  ✅ {table_name} 已加载到数据仓库")
    
    # 4. 数据血缘追踪
    print("\n\n🚀 第4步：数据血缘追踪")
    print("-" * 40)
    
    # 获取数据血缘信息
    lineage = data_warehouse.get_data_lineage()
    print(f"✅ 生成了 {len(lineage)} 条数据血缘记录")
    
    # 展示数据血缘记录
    print("\n🔗 数据血缘记录详情：")
    for i, record in enumerate(lineage[:3]):  # 只展示前3条
        print(f"\n  记录 {i+1}:")
        print(f"    - 表名: {record['table_name']}")
        print(f"    - 开始时间: {record['start_time']}")
        print(f"    - 结束时间: {record['end_time']}")
        print(f"    - 总耗时: {record['total_time']:.3f} 秒")
        print(f"    - 操作数: {len(record['operations'])}")
    
    if len(lineage) > 3:
        print(f"\n  ... 还有 {len(lineage) - 3} 条记录")
    
    # 5. 数据质量监控
    print("\n\n🚀 第5步：数据质量监控")
    print("-" * 40)
    
    for table_name in simulated_tables.keys():
        metrics = data_warehouse.get_data_quality_metrics(table_name)
        print(f"\n  • {table_name}:")
        print(f"    - 总行数: {metrics['total_rows']}")
        print(f"    - 总列数: {metrics['total_columns']}")
        print(f"    - 空值数量: {metrics['missing_values']}")
        print(f"    - 重复行数: {metrics['duplicate_rows']}")
    
    # 6. 实时数据支持
    print("\n\n🚀 第6步：实时数据支持")
    print("-" * 40)
    
    # 创建实时数据处理器
    real_time_processor = RealTimeDataProcessor(data_warehouse)
    
    # 向数据仓库添加实时数据流
    data_warehouse.add_real_time_data_stream('inventory', real_time_processor)
    print("✅ 实时数据流已添加到数据仓库")
    
    # 处理实时数据
    print("\n⏱️  正在处理实时数据...")
    data_warehouse.process_real_time_data()
    print("✅ 实时数据处理完成")
    
    # 展示更新后的数据
    print("\n📊 更新后的数据表详情：")
    for table_name in ['inventory_daily', 'purchase_orders']:
        df = data_warehouse.get_data(table_name)
        print(f"  • {table_name}: {df.shape[0]:>4} 行 × {df.shape[1]:>2} 列")
    
    # 7. 数据查询示例
    print("\n\n🚀 第7步：数据查询示例")
    print("-" * 40)
    
    # 示例1：查询库存数据
    print("\n🔍 示例1：查询库存数据")
    inventory_df = data_warehouse.get_data('inventory_daily')
    print(f"库存数据前5行：")
    print(inventory_df.head())
    
    # 示例2：查询特定SKU的库存数据
    print("\n🔍 示例2：查询特定SKU（SKU001）的库存数据")
    sku_inventory = inventory_df[inventory_df['sku'] == 'SKU001'].tail(7)
    print(sku_inventory[['date', 'sku', 'inventory_on_hand']])
    
    # 示例3：查询预测数据
    print("\n🔍 示例3：查询预测数据")
    forecast_df = data_warehouse.get_data('forecast_output')
    print(f"预测数据前5行：")
    print(forecast_df.head())
    
    # 8. 系统总结
    print("\n\n" + "=" * 70)
    print("🎉 演示完成！")
    print("=" * 70)
    print("\n📋 系统功能总结：")
    print("1. ✅ 统一数据仓库：已整合所有供应链数据")
    print("2. ✅ 数据血缘追踪：已生成数据血缘记录")
    print("3. ✅ 实时数据支持：已实现实时数据处理")
    print("4. ✅ 数据质量监控：已生成数据质量报告")
    print("5. ✅ 数据查询功能：支持灵活的数据查询")
    print("\n📌 系统优势：")
    print("- 集中式管理：所有数据存储在统一数据仓库中")
    print("- 数据可追溯：完整的数据血缘记录，提高数据可信度")
    print("- 实时更新：支持动态数据更新，提高决策时效性")
    print("- 数据质量保证：内置数据质量监控，确保数据可靠性")
    print("- 灵活扩展：支持多种数据源和数据类型")
    print("=" * 70)

if __name__ == "__main__":
    main()
