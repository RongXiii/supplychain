import sys
import os
import pandas as pd
import numpy as np

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from feature_store import FeatureStore
from data_processor import DataProcessor
from simulated_data import generate_simulated_data

def test_feature_store():
    """
    测试Feature Store的功能
    """
    print("=== 测试Feature Store功能 ===")
    
    # 1. 生成模拟数据
    print("1. 生成模拟数据...")
    generate_simulated_data()
    
    # 2. 加载数据
    print("2. 加载数据...")
    data_processor = DataProcessor()
    inventory_df = data_processor.load_data('data/inventory_daily.csv')
    forecast_df = data_processor.load_data('data/forecast_output.csv')
    
    # 3. 创建Feature Store实例
    print("3. 创建Feature Store实例...")
    feature_store = FeatureStore()
    
    # 4. 提取SKU×仓库组合
    print("4. 提取SKU×仓库组合...")
    sku_location_combinations = data_processor.extract_sku_location_combinations(inventory_df, 'item_id', 'location_id')
    print(f"共提取到 {len(sku_location_combinations)} 个SKU×仓库组合")
    
    # 5. 测试单个SKU×仓库的特征更新
    print("5. 测试单个SKU×仓库的特征更新...")
    if sku_location_combinations:
        sku_id, location_id = sku_location_combinations[0]
        demand_series = data_processor.get_demand_series_by_sku_location(inventory_df, sku_id, location_id, 'item_id', 'location_id', 'demand_qty')
        feature_store.update_features(sku_id, location_id, demand_series)
        
        # 获取并打印特征
        features = feature_store.get_features(sku_id, location_id)
        print(f"SKU {sku_id} × 仓库 {location_id} 的特征:")
        for key, value in features.items():
            print(f"  {key}: {value}")
        
        # 获取并打印模型选择标签
        model_tag = feature_store.get_model_selection_tag(sku_id, location_id)
        print(f"SKU {sku_id} × 仓库 {location_id} 的模型选择标签: {model_tag}")
    
    # 6. 测试批量更新特征
    print("6. 测试批量更新特征...")
    # 准备需求数据
    demand_data = data_processor.prepare_demand_data_for_features(inventory_df, 'item_id', 'location_id', 'demand_qty')
    # 只取前2个SKU和前2个仓库进行测试，避免数据量过大
    sample_demand_data = demand_data[(demand_data['item_id'].isin([1, 2])) & (demand_data['location_id'].isin(['WH1', 'WH2']))]
    feature_store.batch_update_features(sample_demand_data)
    print("批量更新特征完成")
    
    # 7. 生成特征报告
    print("7. 生成特征报告...")
    feature_report = feature_store.generate_feature_report()
    print("特征统计信息:")
    for key, value in feature_report.items():
        print(f"  {key}: {value}")
    
    # 8. 测试多个SKU×仓库的特征
    print("8. 测试多个SKU×仓库的特征...")
    for i, (sku_id, location_id) in enumerate(sku_location_combinations[:5]):
        features = feature_store.get_features(sku_id, location_id)
        model_tag = feature_store.get_model_selection_tag(sku_id, location_id)
        print(f"SKU {sku_id} × 仓库 {location_id}: 模型标签={model_tag}, CV={features['coefficient_of_variation']:.2f}, 季节强度={features['seasonal_strength']:.2f}")
    
    print("\n=== Feature Store功能测试完成 ===")

if __name__ == "__main__":
    test_feature_store()
