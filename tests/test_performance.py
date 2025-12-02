import time
import numpy as np
import pandas as pd
from src.data_processor import DataProcessor
from src.forecast_models import ForecastModelSelector
from src.cache_manager import CacheManager

# 初始化组件
cache_manager = CacheManager(distributed=False)  # 禁用Redis连接，只使用内存缓存
data_processor = DataProcessor()
model_selector = ForecastModelSelector(use_gpu=False)  # 先测试CPU版本

# 加载测试数据
df = pd.read_csv('./data/sample_demand_data.csv')

# 1. 测试数据预处理性能
print("\n=== 测试数据预处理性能 ===")
start_time = time.time()
loaded_df = data_processor.load_data('./data/sample_demand_data.csv')
processed_data = data_processor.preprocess_data(loaded_df)
preprocess_time = time.time() - start_time
print(f"数据预处理耗时: {preprocess_time:.4f}秒")
print(f"数据形状: {processed_data.shape}")

# 2. 测试数据分割和特征缩放性能
print("\n=== 测试数据分割和特征缩放性能 ===")
# 准备测试数据，确保有目标列
if 'demand' in processed_data.columns:
    # 使用需求作为目标变量，其他列作为特征
    test_df = processed_data.copy()
    # 确保最后一列是目标变量
    if test_df.columns[-1] != 'demand':
        cols = [col for col in test_df.columns if col != 'demand'] + ['demand']
        test_df = test_df[cols]
    
    start_time = time.time()
    X_train, X_test, y_train, y_test = data_processor.split_data(test_df, test_size=0.2)
    split_time = time.time() - start_time
    print(f"数据分割和特征缩放耗时: {split_time:.4f}秒")
    print(f"训练集形状: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"测试集形状: X_test={X_test.shape}, y_test={y_test.shape}")
else:
    print("数据中没有'demand'列，跳过数据分割测试")

# 3. 测试并行计算性能（简化版）
print("\n=== 测试并行计算性能 ===")
# 使用numpy的并行计算测试，避免pickle问题

def expensive_computation(x):
    # 模拟昂贵的计算
    return np.sum(np.sin(x) * np.cos(x) * np.exp(-x**2))

# 创建测试数据
big_array = np.random.rand(10000)

# 测试单线程计算
start_time = time.time()
single_results = [expensive_computation(x) for x in big_array[:1000]]
single_time = time.time() - start_time
print(f"单线程计算耗时: {single_time:.4f}秒")

# 测试多线程计算
from concurrent.futures import ThreadPoolExecutor
start_time = time.time()
with ThreadPoolExecutor(max_workers=4) as executor:
    thread_results = list(executor.map(expensive_computation, big_array[:1000]))
thread_time = time.time() - start_time
print(f"多线程计算耗时: {thread_time:.4f}秒")
print(f"线程加速比: {single_time / thread_time:.2f}x")

# 测试多进程计算
from concurrent.futures import ProcessPoolExecutor
start_time = time.time()
if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=4) as executor:
        process_results = list(executor.map(expensive_computation, big_array[:1000]))
    process_time = time.time() - start_time
    print(f"多进程计算耗时: {process_time:.4f}秒")
    print(f"进程加速比: {single_time / process_time:.2f}x")

# 4. 测试模型训练性能（简化版）
print("\n=== 测试模型训练性能 ===")
# 创建简单的训练数据
X_simple = np.random.rand(1000, 10)  # 1000个样本，10个特征
y_simple = np.random.rand(1000)  # 目标变量

start_time = time.time()
# 只测试一个简单模型，避免复杂的模型选择过程
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_simple, y_simple)
train_time = time.time() - start_time
print(f"简单模型训练耗时: {train_time:.4f}秒")

# 5. 测试缓存性能
print("\n=== 测试缓存性能 ===")
# 测试大数据缓存
big_data = np.random.rand(1000, 100)  # 减小数据规模以加快测试

# 测试写入缓存
start_time = time.time()
cache_manager.set('test_big_data', big_data, expire_seconds=3600)
write_time = time.time() - start_time
print(f"缓存写入耗时: {write_time:.4f}秒")

# 测试读取缓存
start_time = time.time()
cached_data = cache_manager.get('test_big_data')
read_time = time.time() - start_time
print(f"缓存读取耗时: {read_time:.4f}秒")

# 测试直接计算vs缓存读取
def compute_expensive_operation(data):
    # 模拟昂贵的计算
    return np.linalg.inv(data.T @ data + 0.01 * np.eye(data.shape[1]))

# 直接计算
start_time = time.time()
result1 = compute_expensive_operation(big_data)
direct_time = time.time() - start_time

# 缓存计算结果
cache_manager.set('expensive_result', result1, expire_seconds=3600)

# 从缓存读取
start_time = time.time()
result2 = cache_manager.get('expensive_result')
cache_compute_time = time.time() - start_time

print(f"直接计算耗时: {direct_time:.4f}秒")
print(f"缓存读取耗时: {cache_compute_time:.4f}秒")
if cache_compute_time > 0:
    print(f"缓存加速比: {direct_time / cache_compute_time:.2f}x")

# 6. 测试GPU加速效果（如果可用）
try:
    print("\n=== 测试XGBoost GPU加速效果 ===")
    from xgboost import XGBRegressor
    
    # 测试CPU版本
    start_time = time.time()
    model_cpu = XGBRegressor(n_estimators=100, random_state=42, tree_method='auto')
    model_cpu.fit(X_simple, y_simple)
    cpu_time = time.time() - start_time
    print(f"XGBoost CPU训练耗时: {cpu_time:.4f}秒")
    
    # 测试GPU版本
    start_time = time.time()
    model_gpu = XGBRegressor(n_estimators=100, random_state=42, tree_method='gpu_hist', device='cuda')
    model_gpu.fit(X_simple, y_simple)
    gpu_time = time.time() - start_time
    print(f"XGBoost GPU训练耗时: {gpu_time:.4f}秒")
    print(f"CPU vs GPU加速比: {cpu_time / gpu_time:.2f}x")
except Exception as e:
    print(f"GPU测试失败: {e}")
    print("请确保已正确安装GPU驱动和XGBoost GPU版本")

# 清理缓存
cache_manager.delete('test_big_data')
cache_manager.delete('expensive_result')

print("\n=== 性能测试完成 ===")
