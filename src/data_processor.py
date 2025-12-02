import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing

class DataProcessor:
    """
    数据处理类，用于处理需求数据、预测数据和实际到货数据
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.num_workers = max(1, multiprocessing.cpu_count() - 1)  # 使用除了1个CPU外的所有可用CPU
    
    def load_data(self, file_path):
        """
        加载数据文件
        
        Args:
            file_path: 数据文件路径
            
        Returns:
            df: 加载的数据框
        """
        try:
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            print(f"加载数据失败: {e}")
            return None
    
    def preprocess_data(self, df, target_column='demand', date_column='date'):
        """
        预处理数据，包括缺失值处理、特征工程等
        
        Args:
            df: 原始数据框
            target_column: 目标列名
            date_column: 日期列名
            
        Returns:
            processed_df: 预处理后的数据框
        """
        # 复制数据以避免修改原始数据
        processed_df = df.copy()
        
        # 处理日期列
        if date_column in processed_df.columns:
            processed_df[date_column] = pd.to_datetime(processed_df[date_column])
            processed_df.set_index(date_column, inplace=True)
        
        # 处理缺失值
        processed_df = processed_df.fillna(0)
        
        # 添加时间特征
        if hasattr(processed_df.index, 'month'):
            processed_df['month'] = processed_df.index.month
            processed_df['quarter'] = processed_df.index.quarter
            processed_df['year'] = processed_df.index.year
        
        return processed_df
    
    def split_data(self, df, test_size=0.2):
        """
        分割训练集和测试集
        
        Args:
            df: 数据框
            test_size: 测试集比例
            
        Returns:
            X_train, X_test, y_train, y_test: 训练集和测试集
        """
        # 假设最后一列是目标变量
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        split_idx = int(len(df) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # 标准化特征
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def prepare_features(self, df):
        """
        准备用于预测的特征
        
        Args:
            df: 数据框
            
        Returns:
            features: 特征数组
        """
        features = df.iloc[:, :-1]
        features_scaled = self.scaler.transform(features)
        return features_scaled
    
    def extract_sku_location_combinations(self, df, sku_column='item_id', location_column='location_id'):
        """
        从数据中提取SKU×仓库组合
        
        Args:
            df: 数据框
            sku_column: SKU列名
            location_column: 仓库列名
            
        Returns:
            combinations: SKU×仓库组合列表
        """
        combinations = df[[sku_column, location_column]].drop_duplicates().values.tolist()
        return combinations
    
    def get_demand_series_by_sku_location(self, df, sku_id, location_id, sku_column='item_id', location_column='location_id', demand_column='demand'):
        """
        获取指定SKU×仓库的需求序列
        
        Args:
            df: 数据框
            sku_id: SKU ID
            location_id: 仓库ID
            sku_column: SKU列名
            location_column: 仓库列名
            demand_column: 需求列名
            
        Returns:
            demand_series: 需求序列
        """
        filtered_df = df[(df[sku_column] == sku_id) & (df[location_column] == location_id)]
        demand_series = filtered_df[demand_column].tolist()
        return demand_series
    
    def prepare_demand_data_for_features(self, demand_df, sku_column='item_id', location_column='location_id', demand_column='demand', parallel_mode='thread', num_workers=None):
        """
        准备用于特征更新的需求数据
        
        Args:
            demand_df: 需求数据框
            sku_column: SKU列名
            location_column: 仓库列名
            demand_column: 需求列名
            parallel_mode: 并行模式，可选值: 'thread'(线程), 'process'(进程), 'distributed'(分布式)
            num_workers: 并行工作者数量，默认使用CPU核心数
            
        Returns:
            demand_data: 整理后的需求数据，格式为{sku_id: {location_id: demand_series}}
        """
        demand_data = {}
        
        # 提取所有SKU×仓库组合
        combinations = self.extract_sku_location_combinations(demand_df, sku_column, location_column)
        
        # 定义并行处理函数
        def process_combination(sku_id, location_id):
            demand_series = self.get_demand_series_by_sku_location(
                demand_df, sku_id, location_id, sku_column, location_column, demand_column
            )
            return sku_id, location_id, demand_series
        
        # 设置默认工作者数量
        if num_workers is None:
            num_workers = os.cpu_count() if parallel_mode == 'process' else min(20, os.cpu_count() * 4)
        
        # 根据不同的并行模式选择执行器
        if parallel_mode == 'thread':
            # 使用线程池进行并行处理（适合I/O密集型任务）
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # 提交所有任务
                futures = [executor.submit(process_combination, sku_id, location_id) for sku_id, location_id in combinations]
                
                # 收集结果
                for future in concurrent.futures.as_completed(futures):
                    sku_id, location_id, demand_series = future.result()
                    if sku_id not in demand_data:
                        demand_data[sku_id] = {}
                    demand_data[sku_id][location_id] = demand_series
        
        elif parallel_mode == 'process':
            # 使用进程池进行并行处理（适合CPU密集型任务）
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                # 提交所有任务
                futures = [executor.submit(process_combination, sku_id, location_id) for sku_id, location_id in combinations]
                
                # 收集结果
                for future in concurrent.futures.as_completed(futures):
                    sku_id, location_id, demand_series = future.result()
                    if sku_id not in demand_data:
                        demand_data[sku_id] = {}
                    demand_data[sku_id][location_id] = demand_series
        
        elif parallel_mode == 'distributed':
            # 分布式处理（使用Dask或Ray）
            try:
                # 尝试使用Dask进行分布式处理
                import dask
                from dask import delayed
                
                print("使用Dask进行分布式处理")
                
                # 创建延迟任务列表
                results = []
                for sku_id, location_id in combinations:
                    demand_series = delayed(self.get_demand_series_by_sku_location)(
                        demand_df, sku_id, location_id, sku_column, location_column, demand_column
                    )
                    results.append(delayed((sku_id, location_id, demand_series)))
                
                # 执行并行计算
                processed_results = dask.compute(*results, scheduler='processes', num_workers=num_workers)
                
                # 收集结果
                for sku_id, location_id, demand_series in processed_results:
                    if sku_id not in demand_data:
                        demand_data[sku_id] = {}
                    demand_data[sku_id][location_id] = demand_series
            except ImportError:
                try:
                    # 尝试使用Ray进行分布式处理
                    import ray
                    
                    print("使用Ray进行分布式处理")
                    
                    # 初始化Ray
                    if not ray.is_initialized():
                        ray.init(num_cpus=num_workers, log_to_driver=False)
                    
                    # 定义远程函数
                    @ray.remote
                    def remote_process_combination(sku_id, location_id):
                        return process_combination(sku_id, location_id)
                    
                    # 创建远程任务
                    remote_tasks = [remote_process_combination.remote(sku_id, location_id) for sku_id, location_id in combinations]
                    
                    # 获取结果
                    processed_results = ray.get(remote_tasks)
                    
                    # 收集结果
                    for sku_id, location_id, demand_series in processed_results:
                        if sku_id not in demand_data:
                            demand_data[sku_id] = {}
                        demand_data[sku_id][location_id] = demand_series
                    
                    # 关闭Ray
                    ray.shutdown()
                except ImportError:
                    print("未安装Dask或Ray，回退到线程池处理")
                    # 回退到线程池
                    with ThreadPoolExecutor(max_workers=num_workers) as executor:
                        # 提交所有任务
                        futures = [executor.submit(process_combination, sku_id, location_id) for sku_id, location_id in combinations]
                        
                        # 收集结果
                        for future in concurrent.futures.as_completed(futures):
                            sku_id, location_id, demand_series = future.result()
                            if sku_id not in demand_data:
                                demand_data[sku_id] = {}
                            demand_data[sku_id][location_id] = demand_series
        
        else:
            raise ValueError(f"不支持的并行模式: {parallel_mode}")
        
        return demand_data
    
    def compare_demand(self, actual_demand, predicted_demand):
        """
        比较实际需求和预测需求
        
        Args:
            actual_demand: 实际需求数组
            predicted_demand: 预测需求数组
            
        Returns:
            metrics: 包含各种误差指标的字典
        """
        # 计算误差指标
        metrics = {
            'mae': np.mean(np.abs(actual_demand - predicted_demand)),
            'mse': np.mean((actual_demand - predicted_demand) ** 2),
            'rmse': np.sqrt(np.mean((actual_demand - predicted_demand) ** 2)),
            'mape': np.mean(np.abs((actual_demand - predicted_demand) / actual_demand)) * 100
        }
        
        return metrics
    
    def update_with_actual_data(self, model, actual_data, product_id):
        """
        使用实际到货数据更新模型
        
        Args:
            model: 当前模型
            actual_data: 实际到货数据
            product_id: 产品ID
            
        Returns:
            updated_model: 更新后的模型
        """
        # 这里可以实现模型的在线更新逻辑
        # 简单示例：重新训练模型
        return model
