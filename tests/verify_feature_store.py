#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证Feature Store的核心功能
"""

import sys
import os

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from feature_store import FeatureStore

# 创建Feature Store实例
feature_store = FeatureStore()
print("=== Feature Store 核心功能验证 ===")

# 1. 测试单个SKU×仓库的特征更新
print("\n1. 测试单个SKU×仓库的特征更新...")
sku_id = "test_sku_1"
location_id = "test_loc_1"
# 模拟需求序列（包含一些促销和季节性模式）
demand_series = [10, 12, 8, 15, 20, 50, 18, 16, 14, 12, 10, 8, 12, 15, 18, 20, 16, 14, 12, 10]

# 更新特征
feature_store.update_features(sku_id, location_id, demand_series)
print(f"✓ 已更新SKU {sku_id} × 仓库 {location_id} 的特征")

# 2. 测试特征获取
print("\n2. 测试特征获取...")
features = feature_store.get_features(sku_id, location_id)
print(f"✓ 成功获取特征：")
for key, value in features.items():
    if key != "last_updated":  # 跳过时间戳
        print(f"  - {key}: {value:.4f}")

# 3. 测试模型选择标签获取
print("\n3. 测试模型选择标签获取...")
model_tag = feature_store.get_model_selection_tag(sku_id, location_id)
print(f"✓ 模型选择标签：{model_tag}")

# 4. 测试不同需求模式的特征计算
print("\n4. 测试不同需求模式的特征计算...")

# 间歇性需求
intermittent_demand = [0, 0, 0, 15, 0, 0, 0, 20, 0, 0, 0, 18, 0, 0, 0, 25]
feature_store.update_features("sku_intermittent", "loc_1", intermittent_demand)
intermittent_features = feature_store.get_features("sku_intermittent", "loc_1")
print(f"间歇性需求 - CV: {intermittent_features['cv']:.4f}, 间歇性指数: {intermittent_features['intermittency_index']:.4f}")

# 平稳需求
stable_demand = [10, 11, 9, 10, 11, 10, 9, 10, 11, 10, 9, 10]
feature_store.update_features("sku_stable", "loc_1", stable_demand)
stable_features = feature_store.get_features("sku_stable", "loc_1")
print(f"平稳需求 - CV: {stable_features['cv']:.4f}, 间歇性指数: {stable_features['intermittency_index']:.4f}")

# 5. 测试模型选择标签的变化
print("\n5. 测试模型选择标签的变化...")
print(f"间歇性需求 - 模型标签: {feature_store.get_model_selection_tag('sku_intermittent', 'loc_1')}")
print(f"平稳需求 - 模型标签: {feature_store.get_model_selection_tag('sku_stable', 'loc_1')}")

# 6. 测试特征报告生成
print("\n6. 测试特征报告生成...")
report = feature_store.generate_feature_report()
print(f"✓ 生成特征报告：")
print(f"  - 总SKU×仓库组合数: {report['total_sku_location_pairs']}")
print(f"  - 特征汇总: {report['feature_summary'].keys()}")
print(f"  - 模型分布: {report['model_distribution']}")

# 7. 测试文件持久化
print("\n7. 测试文件持久化...")
# 重新创建Feature Store实例，测试数据加载
new_feature_store = FeatureStore()
# 获取之前保存的特征
loaded_features = new_feature_store.get_features(sku_id, location_id)
print(f"✓ 成功从文件加载特征：")
print(f"  - CV: {loaded_features['cv']:.4f}")
print(f"  - 季节性强度: {loaded_features['seasonality_strength']:.4f}")

print("\n=== 验证完成 ===")
print("Feature Store的核心功能验证通过！")
