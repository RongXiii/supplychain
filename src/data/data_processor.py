import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing
from datetime import datetime, timedelta
import json
from .data_source import DataSourceFactory
import itertools

class DataProcessor:
    """
    数据处理类，用于数据加载、预处理、特征提取等操作
    """
    def __init__(self, data_dir: str = "data", parallel_mode: str = "single", max_workers: Optional[int] = None):
        """
        初始化数据处理类
        
        Args:
            data_dir: 数据目录
            parallel_mode: 并行模式，可选值: single(单线程), thread(多线程), process(多进程)
            max_workers: 最大工作线程/进程数，None表示自动计算
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.data_dir = data_dir
        self.parallel_mode = parallel_mode
        self.max_workers = max_workers if max_workers is not None else multiprocessing.cpu_count()
        self.scaler = StandardScaler()
        
        # 初始化数据源工厂
        self.data_source_factory = DataSourceFactory()
        
        # 特征工程配置
        self.feature_config = {
            'time_features': True,
            'lag_features': True,
            'rolling_features': True,
            'interaction_features': False,
            'polynomial_features': False,
            'embedding_features': False,
            'external_features': True,  # 外部特征集成
            'dynamic_feature_selection': True  # 动态特征选择
        }
        
        # 特征工程参数
        self.feature_params = {
            'lag_values': [1, 3, 7, 14, 30],
            'rolling_windows': [7, 14, 30, 60],
            'rolling_functions': ['mean', 'median', 'std', 'min', 'max'],
            'polynomial_degree': 2,
            'interaction_depth': 2,
            # 动态特征选择参数
            'feature_selection_method': 'mutual_info',  # 可选: mutual_info, f_regression
            'top_k_features': 20,  # 保留的特征数量
            'min_importance_threshold': 0.0  # 特征重要性阈值
        }
        
        # 并行执行器缓存
        self.executor_cache = {}
        
        self.logger.info(f"DataProcessor initialized with parallel_mode={parallel_mode}, max_workers={self.max_workers}")
        
    def _get_data_source(self):
        """
        获取数据源实例
        
        Returns:
            数据源实例
        """
        # 首先检查环境变量中是否配置了数据源类型
        data_source_type = os.getenv("DATA_SOURCE_TYPE", "csv")
        
        # 配置数据源参数
        if data_source_type == "csv":
            data_source_config = {
                'data_dir': self.data_dir
            }
        elif data_source_type == "database":
            data_source_config = {
                'connection_string': os.getenv("DATABASE_CONNECTION_STRING", "")
            }
        elif data_source_type == "api":
            data_source_config = {
                'base_url': os.getenv("API_BASE_URL", ""),
                'headers': {"Authorization": f"Bearer {os.getenv('API_TOKEN', '')}"}
            }
        else:
            data_source_config = {}
        
        # 创建数据源实例
        return self.data_source_factory.create_data_source(data_source_type, data_source_config)
    
    def load_data(self, table_name: str, cache: bool = False, cache_expire: int = 3600) -> pd.DataFrame:
        """
        加载数据文件，支持多种数据源
        
        Args:
            table_name: 表名或文件名
            cache: 是否使用缓存
            cache_expire: 缓存过期时间(秒)
        
        Returns:
            加载的数据DataFrame
        """
        start_time = datetime.now()
        
        try:
            # 获取数据源实例
            data_source = self._get_data_source()
            
            # 根据表名获取对应的数据
            if table_name == "items.csv" or table_name == "items":
                df = data_source.get_items()
            elif table_name == "locations.csv" or table_name == "locations":
                df = data_source.get_locations()
            elif table_name == "suppliers.csv" or table_name == "suppliers":
                df = data_source.get_suppliers()
            elif table_name == "inventory_daily.csv" or table_name == "inventory_daily":
                df = data_source.get_inventory_daily()
            elif table_name == "purchase_orders.csv" or table_name == "purchase_orders":
                df = data_source.get_purchase_orders()
            elif table_name == "forecast_output.csv" or table_name == "forecast_output":
                df = data_source.get_forecast_output()
            elif table_name == "optimal_plan.csv" or table_name == "optimal_plan":
                df = data_source.get_optimal_plan()
            else:
                # 对于其他表，尝试从CSV文件加载
                file_path = os.path.join(self.data_dir, table_name)
                if not os.path.exists(file_path):
                    self.logger.error(f"File not found: {file_path}")
                    raise FileNotFoundError(f"File not found: {file_path}")
                df = pd.read_csv(file_path)
        except Exception as e:
            # 如果从数据源获取失败，尝试从CSV文件加载（降级策略）
            self.logger.warning(f"Failed to load data from {data_source_type} source, falling back to CSV: {e}")
            file_path = os.path.join(self.data_dir, table_name)
            if not os.path.exists(file_path):
                self.logger.error(f"File not found: {file_path}")
                raise FileNotFoundError(f"File not found: {file_path}")
            df = pd.read_csv(file_path)
        
        load_time = (datetime.now() - start_time).total_seconds()
        self.logger.info(f"Loaded data from {table_name} in {load_time:.4f} seconds, shape: {df.shape}")
        
        return df
    
    def preprocess_data(self, df, target_column='demand', date_column='date', feature_config=None):
        """
        预处理数据，包括缺失值处理、自动化特征工程等
        
        Args:
            df: 原始数据框
            target_column: 目标列名
            date_column: 日期列名
            feature_config: 特征工程配置，覆盖默认配置
            
        Returns:
            processed_df: 预处理后的数据框
        """
        start_time = datetime.now()
        
        # 复制数据以避免修改原始数据
        processed_df = df.copy()
        
        # 更新特征配置
        current_feature_config = self.feature_config.copy()
        if feature_config:
            current_feature_config.update(feature_config)
        
        # 处理日期列
        if date_column in processed_df.columns:
            processed_df[date_column] = pd.to_datetime(processed_df[date_column])
            processed_df.set_index(date_column, inplace=True)
            
            # 尝试设置日期频率
            try:
                processed_df = processed_df.asfreq('D')  # 默认为天频率
            except Exception as e:
                # 如果设置频率失败，保持原样
                self.logger.warning(f"无法设置日期频率: {e}")
        
        # 处理缺失值
        processed_df = self._handle_missing_values(processed_df)
        
        # 自动化特征工程
        processed_df = self._automated_feature_engineering(processed_df, current_feature_config)
        
        # 处理异常值
        processed_df = self._handle_outliers(processed_df, target_column)
        
        preprocess_time = (datetime.now() - start_time).total_seconds()
        self.logger.info(f"预处理完成，耗时 {preprocess_time:.4f} 秒，原始特征数: {len(df.columns)}, 生成特征数: {len(processed_df.columns)}")
        
        return processed_df
    
    def _handle_missing_values(self, df):
        """
        智能处理缺失值
        
        Args:
            df: 数据框
            
        Returns:
            处理缺失值后的数据框
        """
        # 复制数据以避免修改原始数据
        df_copy = df.copy()
        
        # 分离数值列和类别列
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        categorical_cols = df_copy.select_dtypes(include=['object', 'category']).columns
        
        # 处理数值列缺失值（使用中位数填充）
        if len(numeric_cols) > 0:
            df_copy[numeric_cols] = df_copy[numeric_cols].fillna(df_copy[numeric_cols].median())
        
        # 处理类别列缺失值（使用众数填充）
        if len(categorical_cols) > 0:
            df_copy[categorical_cols] = df_copy[categorical_cols].fillna(df_copy[categorical_cols].mode().iloc[0])
        
        return df_copy
    
    def _handle_outliers(self, df, target_column):
        """
        处理异常值
        
        Args:
            df: 数据框
            target_column: 目标列名
            
        Returns:
            处理异常值后的数据框
        """
        # 复制数据以避免修改原始数据
        df_copy = df.copy()
        
        # 仅处理数值列
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        if target_column in numeric_cols:
            # 使用IQR方法处理目标列异常值
            Q1 = df_copy[target_column].quantile(0.25)
            Q3 = df_copy[target_column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # 用边界值替换异常值
            df_copy[target_column] = df_copy[target_column].clip(lower_bound, upper_bound)
        
        return df_copy
    
    def _automated_feature_engineering(self, df, feature_config, target_column=None):
        """
        自动化特征工程
        
        Args:
            df: 数据框
            feature_config: 特征工程配置
            target_column: 目标列名，如果为None则使用最后一列
            
        Returns:
            生成特征后的数据框
        """
        # 复制数据以避免修改原始数据
        df_copy = df.copy()
        
        # 使用字典存储所有新特征，最后一次性添加，减少DataFrame碎片化
        new_features = {}
        
        # 添加时间特征
        if feature_config['time_features'] and hasattr(df_copy.index, 'month'):
            time_features = {
                'month': df_copy.index.month,
                'quarter': df_copy.index.quarter,
                'year': df_copy.index.year,
                'day_of_week': df_copy.index.dayofweek,
                'day_of_month': df_copy.index.day,
                'is_weekend': df_copy.index.dayofweek.isin([5, 6]).astype(int),
                'is_month_start': df_copy.index.is_month_start.astype(int),
                'is_month_end': df_copy.index.is_month_end.astype(int),
                'sin_month': np.sin(2 * np.pi * df_copy.index.month / 12),
                'cos_month': np.cos(2 * np.pi * df_copy.index.month / 12),
                'sin_day_of_week': np.sin(2 * np.pi * df_copy.index.dayofweek / 7),
                'cos_day_of_week': np.cos(2 * np.pi * df_copy.index.dayofweek / 7)
            }
            new_features.update(time_features)
        
        # 添加滞后特征
        if feature_config['lag_features']:
            numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                for lag in self.feature_params['lag_values']:
                    new_features[f'{col}_lag_{lag}'] = df_copy[col].shift(lag)
        
        # 添加所有新特征
        if new_features:
            df_copy = df_copy.assign(**new_features)
            # 移除包含NaN的行（由于滞后操作产生）
            df_copy = df_copy.dropna()
        
        # 添加滚动统计特征 - 单独处理，因为需要基于之前添加的特征
        if feature_config['rolling_features']:
            rolling_features = {}
            numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                for window in self.feature_params['rolling_windows']:
                    for func in self.feature_params['rolling_functions']:
                        rolling_features[f'{col}_roll_{window}_{func}'] = df_copy[col].rolling(window=window).agg(func)
            
            if rolling_features:
                df_copy = df_copy.assign(**rolling_features)
                df_copy = df_copy.dropna()
        
        # 添加多项式特征
        if feature_config['polynomial_features']:
            poly_features = {}
            numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                for degree in range(2, self.feature_params['polynomial_degree'] + 1):
                    poly_features[f'{col}_poly_{degree}'] = df_copy[col] ** degree
            
            if poly_features:
                df_copy = df_copy.assign(**poly_features)
        
        # 添加交互特征
        if feature_config['interaction_features']:
            numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
            col_pairs = list(itertools.combinations(numeric_cols, 2))[:self.feature_params['max_interaction_pairs']]
            
            interaction_features = {}
            for col1, col2 in col_pairs:
                interaction_features[f'{col1}_x_{col2}'] = df_copy[col1] * df_copy[col2]
            
            if interaction_features:
                df_copy = df_copy.assign(**interaction_features)
        
        # 添加外部特征
        if feature_config['external_features']:
            df_copy = self._add_external_features(df_copy)
        
        # 动态特征选择
        if feature_config['dynamic_feature_selection']:
            # 使用指定的目标列或默认使用最后一列
            if len(df_copy.columns) > 1:
                if target_column is None:
                    target_column = df_copy.columns[-1]
                df_copy, selected_features = self._dynamic_feature_selection(df_copy, target_column)
                self.logger.info(f"动态特征选择完成，保留 {len(selected_features)} 个特征: {selected_features}")
        
        return df_copy
    
    def _dynamic_feature_selection(self, df, target_column):
        """
        动态特征选择：基于互信息或F统计量自动选择最优特征集
        
        Args:
            df: 包含特征和目标列的数据框
            target_column: 目标列名
            
        Returns:
            selected_df: 只包含选择后特征的数据框
            selected_features: 选择的特征列表
        """
        # 分离特征和目标
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # 仅保留数值类型的列
        X = X.select_dtypes(include=['number'])
        
        # 处理可能的缺失值
        X = X.fillna(0)
        y = y.fillna(0)
        
        # 特征选择方法
        method = self.feature_params['feature_selection_method']
        top_k = min(self.feature_params['top_k_features'], len(X.columns))
        
        if method == 'mutual_info':
            # 基于互信息的特征选择
            selector = SelectKBest(score_func=mutual_info_regression, k=top_k)
        elif method == 'f_regression':
            # 基于F统计量的特征选择
            selector = SelectKBest(score_func=f_regression, k=top_k)
        else:
            self.logger.warning(f"未知的特征选择方法: {method}，使用默认的mutual_info")
            selector = SelectKBest(score_func=mutual_info_regression, k=top_k)
        
        # 训练选择器并转换数据
        X_selected = selector.fit_transform(X, y)
        
        # 获取选择的特征
        selected_feature_indices = selector.get_support(indices=True)
        selected_features = list(X.columns[selected_feature_indices])
        
        # 重建数据框，包含目标列
        selected_df = pd.DataFrame(X_selected, index=X.index, columns=selected_features)
        selected_df[target_column] = y
        
        return selected_df, selected_features
    
    def calculate_feature_importance(self, df, target_column, method='mutual_info'):
        """
        计算特征重要性
        
        Args:
            df: 包含特征和目标列的数据框
            target_column: 目标列名
            method: 特征重要性计算方法，可选: mutual_info, f_regression
            
        Returns:
            feature_importance: 特征重要性字典，键为特征名，值为重要性得分
        """
        # 分离特征和目标
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # 处理可能的缺失值
        X = X.fillna(0)
        y = y.fillna(0)
        
        # 只保留数值类型的列
        X = X.select_dtypes(include=['number'])
        
        # 计算特征重要性
        if method == 'mutual_info':
            importance_scores = mutual_info_regression(X, y)
        elif method == 'f_regression':
            _, p_values = f_regression(X, y)
            # 将p值转换为重要性得分（p值越小，得分越高）
            importance_scores = -np.log10(p_values)
        else:
            self.logger.warning(f"未知的特征重要性计算方法: {method}，使用默认的mutual_info")
            importance_scores = mutual_info_regression(X, y)
        
        # 创建特征重要性字典
        feature_importance = {}
        for feature, score in zip(X.columns, importance_scores):
            if feature != target_column:
                feature_importance[feature] = score
        
        # 按重要性排序
        feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        return feature_importance
    
    def update_feature_set(self, df, target_column, feature_importance=None):
        """
        定期更新特征集，基于最新的特征重要性
        
        Args:
            df: 包含特征和目标列的数据框
            target_column: 目标列名
            feature_importance: 预计算的特征重要性，如果为None则自动计算
            
        Returns:
            updated_features: 更新后的特征列表
        """
        if feature_importance is None:
            feature_importance = self.calculate_feature_importance(df, target_column)
        
        # 应用特征重要性阈值
        min_threshold = self.feature_params['min_importance_threshold']
        updated_features = [feature for feature, score in feature_importance.items() if score >= min_threshold]
        
        # 确保不超过最大特征数量
        top_k = self.feature_params['top_k_features']
        updated_features = updated_features[:top_k]
        
        self.logger.info(f"特征集已更新，当前特征数量: {len(updated_features)}")
        
        return updated_features
    
    def visualize_feature_importance(self, feature_importance, output_path=None):
        """
        可视化特征贡献度
        
        Args:
            feature_importance: 特征重要性字典
            output_path: 可视化结果保存路径，如果为None则不保存
            
        Returns:
            None
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # 设置中文字体（如果需要）
            plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
            plt.rcParams['axes.unicode_minus'] = False    # 负号显示
            
            # 准备数据
            features = list(feature_importance.keys())
            scores = list(feature_importance.values())
            
            # 绘制条形图
            plt.figure(figsize=(12, 8))
            sns.barplot(x=scores, y=features, palette='viridis')
            plt.title('特征重要性分布', fontsize=16)
            plt.xlabel('重要性得分', fontsize=14)
            plt.ylabel('特征', fontsize=14)
            plt.tight_layout()
            
            # 保存可视化结果
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                plt.savefig(output_path, dpi=300)
                self.logger.info(f"特征重要性可视化已保存到: {output_path}")
            else:
                plt.show()
                
            plt.close()
        except ImportError as e:
            self.logger.warning(f"无法导入可视化库: {e}，跳过可视化步骤")
        except Exception as e:
            self.logger.error(f"特征重要性可视化失败: {e}")
    
    def monitor_feature_importance(self, df, target_column, output_path=None):
        """
        特征重要性监控：计算、更新和可视化特征重要性
        
        Args:
            df: 包含特征和目标列的数据框
            target_column: 目标列名
            output_path: 可视化结果保存路径
            
        Returns:
            feature_importance: 特征重要性字典
            updated_features: 更新后的特征列表
        """
        # 计算特征重要性
        feature_importance = self.calculate_feature_importance(df, target_column)
        
        # 更新特征集
        updated_features = self.update_feature_set(df, target_column, feature_importance)
        
        # 可视化特征重要性
        self.visualize_feature_importance(feature_importance, output_path)
        
        return feature_importance, updated_features
    
    def _add_external_features(self, df):
        """
        添加外部特征，包括天气数据、竞争对手价格和宏观经济指标
        
        Args:
            df: 主数据框
            
        Returns:
            添加外部特征后的数据框
        """
        df_copy = df.copy()
        
        try:
            # 获取数据源实例
            data_source = self._get_data_source()
            
            # 确定日期范围
            start_date = df_copy.index.min()
            end_date = df_copy.index.max()
            
            # 获取天气数据（示例：假设所有位置使用同一个天气站的数据）
            weather_data = data_source.get_weather_data(
                location_id='all',
                start_date=start_date,
                end_date=end_date
            )
            
            # 将天气数据与主数据合并
            if not weather_data.empty:
                # 将weather_data的日期列设置为索引
                if 'date' in weather_data.columns:
                    weather_data.set_index('date', inplace=True)
                df_copy = df_copy.merge(weather_data, left_index=True, right_index=True, how='left')
            
            # 获取竞争对手价格数据
            competitor_prices = data_source.get_competitor_prices(
                product_id='all',
                start_date=start_date,
                end_date=end_date
            )
            
            # 将竞争对手价格数据与主数据合并
            if not competitor_prices.empty:
                if 'date' in competitor_prices.columns:
                    competitor_prices.set_index('date', inplace=True)
                # 如果有product_id列，可能需要更复杂的合并逻辑
                if 'product_id' in competitor_prices.columns:
                    # 假设主数据有product_id列
                    if 'product_id' in df_copy.columns:
                        df_copy = df_copy.merge(competitor_prices, 
                                               left_on=['date', 'product_id'], 
                                               right_on=['date', 'product_id'], 
                                               how='left')
                else:
                    df_copy = df_copy.merge(competitor_prices, left_index=True, right_index=True, how='left')
            
            # 获取宏观经济数据
            macro_data = data_source.get_macro_economic_data(
                indicator_ids=['GDP', 'CPI', 'interest_rate'],
                start_date=start_date,
                end_date=end_date
            )
            
            # 将宏观经济数据与主数据合并
            if not macro_data.empty:
                if 'date' in macro_data.columns:
                    macro_data.set_index('date', inplace=True)
                df_copy = df_copy.merge(macro_data, left_index=True, right_index=True, how='left')
            
            # 填充外部特征中的缺失值
            numeric_external_cols = df_copy.select_dtypes(include=[np.number]).columns.difference(df.columns)
            df_copy[numeric_external_cols] = df_copy[numeric_external_cols].fillna(method='ffill').fillna(method='bfill')
            
            self.logger.info(f"Successfully added external features. New features: {list(numeric_external_cols)}")
            
        except Exception as e:
            self.logger.warning(f"Failed to add external features: {e}")
        
        return df_copy
    
    def _add_time_features(self, df):
        """
        添加时间特征
        
        Args:
            df: 数据框
            
        Returns:
            添加时间特征后的数据框
        """
        import holidays
        df_copy = df.copy()
        
        # 基本时间特征
        df_copy['month'] = df_copy.index.month
        df_copy['quarter'] = df_copy.index.quarter
        df_copy['year'] = df_copy.index.year
        df_copy['day_of_week'] = df_copy.index.dayofweek
        df_copy['day_of_month'] = df_copy.index.day
        df_copy['is_weekend'] = df_copy.index.dayofweek.isin([5, 6]).astype(int)
        df_copy['is_month_start'] = df_copy.index.is_month_start.astype(int)
        df_copy['is_month_end'] = df_copy.index.is_month_end.astype(int)
        
        # 添加季节性特征
        df_copy['sin_month'] = np.sin(2 * np.pi * df_copy['month'] / 12)
        df_copy['cos_month'] = np.cos(2 * np.pi * df_copy['month'] / 12)
        df_copy['sin_day_of_week'] = np.sin(2 * np.pi * df_copy['day_of_week'] / 7)
        df_copy['cos_day_of_week'] = np.cos(2 * np.pi * df_copy['day_of_week'] / 7)
        
        # 添加节假日特征
        try:
            # 中国节假日
            cn_holidays = holidays.CountryHoliday('CN')
            df_copy['is_holiday'] = df_copy.index.map(lambda x: 1 if x in cn_holidays else 0)
            df_copy['holiday_name'] = df_copy.index.map(lambda x: cn_holidays.get(x, ''))
            
            # 国际节假日
            int_holidays = holidays.CountryHoliday('US')
            df_copy['is_international_holiday'] = df_copy.index.map(lambda x: 1 if x in int_holidays else 0)
        except Exception as e:
            self.logger.warning(f"无法加载节假日数据: {e}")
            df_copy['is_holiday'] = 0
            df_copy['holiday_name'] = ''
            df_copy['is_international_holiday'] = 0
        
        # 添加时间衰减特征（近期数据权重更高）
        df_copy['days_since_start'] = (df_copy.index - df_copy.index.min()).days
        df_copy['time_decay'] = np.exp(-0.01 * df_copy['days_since_start'])
        
        # 添加趋势特征
        for window in [7, 14, 30]:
            if len(df_copy) >= window + 1:
                # 计算移动平均线
                df_copy[f'trend_{window}d'] = df_copy.index.to_series().rolling(window=window).apply(
                    lambda x: np.polyfit(range(len(x)), x.map(lambda d: d.dayofyear), 1)[0], raw=False
                )
        
        return df_copy
    
    def _add_lag_features(self, df):
        """
        添加滞后特征
        
        Args:
            df: 数据框
            
        Returns:
            添加滞后特征后的数据框
        """
        df_copy = df.copy()
        
        # 获取数值列
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        
        # 为每个数值列添加滞后特征
        for col in numeric_cols:
            for lag in self.feature_params['lag_values']:
                df_copy[f'{col}_lag_{lag}'] = df_copy[col].shift(lag)
        
        # 移除包含NaN的行（由于滞后操作产生）
        df_copy = df_copy.dropna()
        
        return df_copy
    
    def _add_rolling_features(self, df):
        """
        添加滚动统计特征
        
        Args:
            df: 数据框
            
        Returns:
            添加滚动统计特征后的数据框
        """
        df_copy = df.copy()
        
        # 获取数值列
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        
        # 为每个数值列添加滚动统计特征
        for col in numeric_cols:
            for window in self.feature_params['rolling_windows']:
                for func in self.feature_params['rolling_functions']:
                    df_copy[f'{col}_roll_{window}_{func}'] = df_copy[col].rolling(window=window).agg(func)
        
        # 移除包含NaN的行（由于滚动操作产生）
        df_copy = df_copy.dropna()
        
        return df_copy
    
    def _add_polynomial_features(self, df):
        """
        添加多项式特征
        
        Args:
            df: 数据框
            
        Returns:
            添加多项式特征后的数据框
        """
        df_copy = df.copy()
        
        # 获取数值列
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        
        # 为每个数值列添加多项式特征
        for col in numeric_cols:
            for degree in range(2, self.feature_params['polynomial_degree'] + 1):
                df_copy[f'{col}_poly_{degree}'] = df_copy[col] ** degree
        
        return df_copy
    
    def _add_interaction_features(self, df):
        """
        添加交互特征
        
        Args:
            df: 数据框
            
        Returns:
            添加交互特征后的数据框
        """
        df_copy = df.copy()
        
        # 获取数值列
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        
        # 生成所有可能的数值列对
        col_pairs = list(itertools.combinations(numeric_cols, 2))
        
        # 为每对列添加交互特征
        for col1, col2 in col_pairs:
            df_copy[f'{col1}_x_{col2}'] = df_copy[col1] * df_copy[col2]
        
        return df_copy
    
    def feature_selection(self, df, target_column, method='kbest', k=10):
        """
        特征选择
        
        Args:
            df: 数据框
            target_column: 目标列名
            method: 特征选择方法，可选值: kbest, mutual_info
            k: 选择的特征数量
            
        Returns:
            选择的特征列名列表
        """
        # 分离特征和目标
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # 仅选择数值列
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_cols]
        
        # 根据方法选择特征
        if method == 'kbest':
            selector = SelectKBest(score_func=f_regression, k=k)
        elif method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_regression, k=k)
        else:
            raise ValueError(f"不支持的特征选择方法: {method}")
        
        # 拟合选择器
        selector.fit(X_numeric, y)
        
        # 获取选择的特征列名
        selected_features = X_numeric.columns[selector.get_support()].tolist()
        
        return selected_features
    
    def split_data(self, df, target_column, test_size=0.2, validation_size=0.1):
        """
        分割训练集、验证集和测试集
        
        Args:
            df: 数据框
            target_column: 目标列名
            test_size: 测试集比例
            validation_size: 验证集比例（相对于训练集）
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test: 训练集、验证集和测试集
        """
        # 分离特征和目标
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # 第一次分割：训练集 + 验证集 和 测试集
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False  # 时间序列数据不打乱
        )
        
        # 第二次分割：训练集 和 验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=validation_size, shuffle=False
        )
        
        # 标准化特征
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test
    
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
    
    def prepare_demand_data_for_features(self, demand_df, sku_column='item_id', location_column='location_id', demand_column='demand', parallel_mode=None, num_workers=None):
        """
        准备用于特征更新的需求数据
        
        Args:
            demand_df: 需求数据框
            sku_column: SKU列名
            location_column: 仓库列名
            demand_column: 需求列名
            parallel_mode: 并行模式，可选值: 'thread'(线程), 'process'(进程), 'distributed'(分布式)，None表示使用默认配置
            num_workers: 并行工作者数量，默认使用CPU核心数
            
        Returns:
            demand_data: 整理后的需求数据，格式为{sku_id: {location_id: demand_series}}
        """
        start_time = datetime.now()
        demand_data = {}
        
        # 使用默认并行模式如果没有指定
        if parallel_mode is None:
            parallel_mode = self.parallel_mode
        
        # 提取所有SKU×仓库组合
        combinations = self.extract_sku_location_combinations(demand_df, sku_column, location_column)
        
        # 如果组合数量较少，直接串行处理
        if len(combinations) < 10:
            self.logger.info(f"SKU×仓库组合数量较少({len(combinations)}), 使用串行处理")
            for sku_id, location_id in combinations:
                demand_series = self.get_demand_series_by_sku_location(
                    demand_df, sku_id, location_id, sku_column, location_column, demand_column
                )
                if sku_id not in demand_data:
                    demand_data[sku_id] = {}
                demand_data[sku_id][location_id] = demand_series
            
            prepare_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"需求数据准备完成，耗时 {prepare_time:.4f} 秒")
            return demand_data
        
        # 定义并行处理函数
        def process_combination(sku_id, location_id):
            demand_series = self.get_demand_series_by_sku_location(
                demand_df, sku_id, location_id, sku_column, location_column, demand_column
            )
            return sku_id, location_id, demand_series
        
        # 设置默认工作者数量
        if num_workers is None:
            num_workers = os.cpu_count() if parallel_mode == 'process' else min(20, os.cpu_count() * 4)
        
        # 执行并行处理
        results = self._execute_parallel(
            process_combination, 
            [(sku_id, location_id) for sku_id, location_id in combinations],
            parallel_mode, 
            num_workers
        )
        
        # 收集结果
        for sku_id, location_id, demand_series in results:
            if sku_id not in demand_data:
                demand_data[sku_id] = {}
            demand_data[sku_id][location_id] = demand_series
        
        prepare_time = (datetime.now() - start_time).total_seconds()
        self.logger.info(f"需求数据准备完成，耗时 {prepare_time:.4f} 秒")
        
        return demand_data
    
    def _execute_parallel(self, func: Callable, args_list: List[Tuple], parallel_mode: str, num_workers: int):
        """
        通用并行执行函数
        
        Args:
            func: 要并行执行的函数
            args_list: 函数参数列表
            parallel_mode: 并行模式
            num_workers: 并行工作者数量
            
        Returns:
            执行结果列表
        """
        results = []
        
        # 根据不同的并行模式选择执行器
        if parallel_mode == 'thread':
            # 使用线程池进行并行处理（适合I/O密集型任务）
            self.logger.info(f"使用线程池并行处理，工作线程数: {num_workers}")
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # 提交所有任务
                futures = [executor.submit(func, *args) for args in args_list]
                
                # 收集结果
                for future in as_completed(futures):
                    results.append(future.result())
        
        elif parallel_mode == 'process':
            # 使用进程池进行并行处理（适合CPU密集型任务）
            self.logger.info(f"使用进程池并行处理，工作进程数: {num_workers}")
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                # 提交所有任务
                futures = [executor.submit(func, *args) for args in args_list]
                
                # 收集结果
                for future in as_completed(futures):
                    results.append(future.result())
        
        elif parallel_mode == 'distributed':
            # 分布式处理（使用Dask或Ray）
            try:
                # 尝试使用Dask进行分布式处理
                import dask
                from dask import delayed
                
                self.logger.info("使用Dask进行分布式处理")
                
                # 创建延迟任务列表
                delayed_results = []
                for args in args_list:
                    delayed_results.append(delayed(func)(*args))
                
                # 执行并行计算
                results = dask.compute(*delayed_results, scheduler='processes', num_workers=num_workers)
            except ImportError:
                try:
                    # 尝试使用Ray进行分布式处理
                    import ray
                    
                    self.logger.info("使用Ray进行分布式处理")
                    
                    # 初始化Ray
                    if not ray.is_initialized():
                        ray.init(num_cpus=num_workers, log_to_driver=False)
                    
                    # 定义远程函数
                    @ray.remote
                    def remote_func(*args):
                        return func(*args)
                    
                    # 创建远程任务
                    remote_tasks = [remote_func.remote(*args) for args in args_list]
                    
                    # 获取结果
                    results = ray.get(remote_tasks)
                    
                    # 关闭Ray
                    ray.shutdown()
                except ImportError:
                    self.logger.warning("未安装Dask或Ray，回退到线程池处理")
                    # 回退到线程池
                    with ThreadPoolExecutor(max_workers=num_workers) as executor:
                        # 提交所有任务
                        futures = [executor.submit(func, *args) for args in args_list]
                        
                        # 收集结果
                        for future in as_completed(futures):
                            results.append(future.result())
        
        else:
            raise ValueError(f"不支持的并行模式: {parallel_mode}")
        
        return results
    
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
