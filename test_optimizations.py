import time
import pandas as pd
import sys
import os
from concurrent.futures import ThreadPoolExecutor
import random

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from data_processor import DataProcessor
from feature_store import FeatureStore
from cache_manager import cache_manager

def test_cache_performance():
    """测试缓存机制的性能提升"""
    print("=== 测试缓存机制性能 ===")
    
    # 初始化数据处理器
    data_dir = "data"
    processor = DataProcessor()
    
    # 第一次加载数据（没有缓存）
    start_time = time.time()
    demand_data = pd.read_csv(os.path.join(data_dir, "sample_demand_data.csv"))
    load_time_1 = time.time() - start_time
    print(f"第一次加载需求数据耗时: {load_time_1:.4f}秒")
    
    # 缓存数据
    cache_manager.set("test:demand_data", demand_data, expire_seconds=300)
    
    # 第二次加载数据（从缓存）
    start_time = time.time()
    cached_demand_data = cache_manager.get("test:demand_data", data_type='dataframe')
    load_time_2 = time.time() - start_time
    print(f"从缓存加载需求数据耗时: {load_time_2:.4f}秒")
    
    # 计算性能提升
    if load_time_2 > 0:
        improvement = (load_time_1 / load_time_2) * 100
        print(f"缓存性能提升: {improvement:.2f}%")
    else:
        print("缓存性能提升: 无限大%")
    
    print()

def test_parallel_data_loading():
    """测试并行数据加载性能"""
    print("\n=== 测试并行数据加载性能 ===")
    
    # 初始化数据处理器
    processor = DataProcessor()
    
    # 获取数据文件路径
    data_dir = os.path.join(os.getcwd(), "data")
    sample_file = os.path.join(data_dir, "sample_demand_data.csv")
    
    # 测试1：串行加载数据（模拟多个文件串行加载）
    start_time = time.time()
    for _ in range(5):
        data = processor.load_data(sample_file)
    serial_time = time.time() - start_time
    print(f"串行加载数据耗时: {serial_time:.4f} 秒")
    
    # 测试2：并行加载数据（使用并行机制）
    start_time = time.time()
    # 模拟并行加载多个文件
    file_paths = [sample_file for _ in range(5)]
    with ThreadPoolExecutor(max_workers=processor.num_workers) as executor:
        results = list(executor.map(processor.load_data, file_paths))
    parallel_time = time.time() - start_time
    print(f"并行加载数据耗时: {parallel_time:.4f} 秒")
    
    print("并行数据加载实现完成")

def test_parallel_feature_generation():
    """测试并行特征生成性能"""
    print("\n=== 测试并行特征生成性能 ===")
    
    # 初始化特征存储
    feature_store = FeatureStore()
    
    # 准备测试数据（创建数据框格式）
    data = []
    for sku_id in range(1, 11):  # 10个SKU
        for location_id in range(1, 6):  # 每个SKU 5个仓库
            # 生成30天的随机需求数据
            for day in range(30):
                data.append({
                    'item_id': sku_id,
                    'location_id': location_id,
                    'date': f'2023-01-{day+1:02d}',
                    'demand_qty': random.randint(50, 200)
                })
    
    demand_df = pd.DataFrame(data)
    
    # 测试：并行生成特征
    start_time = time.time()
    # 使用batch_update_features方法测试并行处理
    feature_store.batch_update_features(demand_df)
    parallel_time = time.time() - start_time
    print(f"并行生成特征耗时: {parallel_time:.4f} 秒")
    
    print("并行特征生成实现完成")

def test_api_performance():
    """测试API性能（需要运行API服务器）"""
    print("=== 测试API性能 ===")
    print("注意：此测试需要先启动API服务器")
    print("请运行：python start_api.py")
    print()
    
    try:
        import requests
        
        # 测试URL
        base_url = "http://localhost:8000"
        
        # 测试获取产品列表
        print("测试获取产品列表API...")
        
        # 第一次请求（没有缓存）
        start_time = time.time()
        response = requests.get(f"{base_url}/api/items")
        time_1 = time.time() - start_time
        print(f"第一次请求耗时: {time_1:.4f}秒")
        
        # 第二次请求（有缓存）
        start_time = time.time()
        response = requests.get(f"{base_url}/api/items")
        time_2 = time.time() - start_time
        print(f"第二次请求耗时: {time_2:.4f}秒")
        
        # 计算性能提升
        if time_2 > 0:
            improvement = (time_1 / time_2) * 100
            print(f"API缓存性能提升: {improvement:.2f}%")
        else:
            print(f"API缓存性能提升: 无限大%")
    except ImportError:
        print("未安装requests库，无法测试API性能")
    except requests.exceptions.ConnectionError:
        print("无法连接到API服务器，请确保服务器已启动")
    
    print()

def main():
    """主测试函数"""
    print("供应链智能补货系统 - 优化测试")
    print("=" * 50)
    print()
    
    # 测试缓存性能
    test_cache_performance()
    
    # 测试并行数据加载
    test_parallel_data_loading()
    
    # 测试并行特征生成
    test_parallel_feature_generation()
    
    # 测试API性能
    test_api_performance()
    
    print("测试完成！")
    print("=" * 50)

if __name__ == "__main__":
    main()
