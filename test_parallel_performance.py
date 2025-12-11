import time
import numpy as np
import pandas as pd
import sys
import os

# 将src目录添加到Python路径
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, src_path)

from system.main import ReplenishmentSystem

# 创建测试数据
def create_test_data(num_products=10, num_periods=365):
    """创建测试数据，模拟多个产品的销售历史"""
    products_data = {}
    
    for i in range(1, num_products + 1):
        # 创建不同类型的时间序列数据
        product_id = f"P{i:03d}"
        
        # 生成时间索引
        dates = pd.date_range(start='2023-01-01', periods=num_periods, freq='D')
        
        # 生成基础需求（带有趋势和季节性）
        base_demand = 50 + 0.1 * np.arange(num_periods)
        seasonality = 20 * np.sin(2 * np.pi * np.arange(num_periods) / 7)  # 周季节性
        trend = 0.05 * np.arange(num_periods)
        noise = np.random.normal(0, 5, num_periods)
        
        # 根据产品类型调整需求模式
        if i % 4 == 0:
            # 间歇性需求（高变异性）
            demand = np.random.poisson(lam=2, size=num_periods)
        elif i % 4 == 1:
            # 平稳需求
            demand = base_demand + noise
        elif i % 4 == 2:
            # 趋势需求
            demand = base_demand + trend + noise
        else:
            # 季节性需求
            demand = base_demand + seasonality + noise
        
        # 确保需求非负
        demand = np.maximum(demand, 0)
        
        # 创建数据框
        df = pd.DataFrame({
            'date': dates,
            'quantity': demand,
            'product_id': product_id
        })
        
        products_data[product_id] = df
    
    return products_data

# 测试批量预测性能
def test_batch_forecast_performance():
    """测试批量预测的并行和串行性能"""
    # 初始化系统
    system = ReplenishmentSystem()
    
    # 测试不同规模的数据
    test_scales = [10, 50, 100, 200]
    results = []
    
    for num_products in test_scales:
        print(f"\n=== 测试 {num_products} 个产品 ===")
        
        # 创建测试数据
        products_data = create_test_data(num_products=num_products, num_periods=365)
        
        # 1. 测试串行模式
        start_time = time.time()
        result_serial = system.batch_run_forecast(
            products_data=products_data,
            steps=7,
            parallel=False
        )
        serial_time = time.time() - start_time
        
        # 2. 测试并行模式（4核）
        start_time = time.time()
        result_parallel = system.batch_run_forecast(
            products_data=products_data,
            steps=7,
            parallel=True,
            n_jobs=4
        )
        parallel_time = time.time() - start_time
        
        # 3. 测试并行模式（8核）
        start_time = time.time()
        result_parallel_8 = system.batch_run_forecast(
            products_data=products_data,
            steps=7,
            parallel=True,
            n_jobs=8
        )
        parallel_time_8 = time.time() - start_time
        
        # 计算性能指标
        speedup = serial_time / parallel_time if parallel_time > 0 else 0
        speedup_8 = serial_time / parallel_time_8 if parallel_time_8 > 0 else 0
        avg_time_per_product_serial = serial_time / num_products
        avg_time_per_product_parallel = parallel_time / num_products
        avg_time_per_product_parallel_8 = parallel_time_8 / num_products
        
        # 保存结果
        results.append({
            'num_products': num_products,
            'serial_time': serial_time,
            'parallel_time_4': parallel_time,
            'parallel_time_8': parallel_time_8,
            'speedup_4': speedup,
            'speedup_8': speedup_8,
            'avg_time_per_product_serial': avg_time_per_product_serial,
            'avg_time_per_product_parallel_4': avg_time_per_product_parallel,
            'avg_time_per_product_parallel_8': avg_time_per_product_parallel_8
        })
        
        # 打印结果
        print(f"串行时间: {serial_time:.2f}秒")
        print(f"并行时间(4核): {parallel_time:.2f}秒")
        print(f"并行时间(8核): {parallel_time_8:.2f}秒")
        print(f"加速比(4核): {speedup:.2f}x")
        print(f"加速比(8核): {speedup_8:.2f}x")
        print(f"平均时间/产品(串行): {avg_time_per_product_serial:.4f}秒")
        print(f"平均时间/产品(并行4核): {avg_time_per_product_parallel:.4f}秒")
        print(f"平均时间/产品(并行8核): {avg_time_per_product_parallel_8:.4f}秒")
    
    # 输出汇总结果
    print("\n=== 性能测试汇总 ===")
    print("产品数量 | 串行时间(s) | 并行时间4核(s) | 并行时间8核(s) | 加速比4核 | 加速比8核")
    print("-" * 80)
    for result in results:
        print(f"{result['num_products']:8} | {result['serial_time']:11.2f} | {result['parallel_time_4']:14.2f} | {result['parallel_time_8']:14.2f} | {result['speedup_4']:8.2f} | {result['speedup_8']:8.2f}")

if __name__ == "__main__":
    test_batch_forecast_performance()