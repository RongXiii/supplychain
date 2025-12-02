import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import json
import os
from datetime import datetime
import concurrent.futures
import multiprocessing

class FeatureStore:
    """
    Feature Store 用于存储 SKU×仓库级别的统计特征和模型选择标签
    特征包括：CV（变异系数）、季节强度、间歇性指数、促销标记等
    """
    
    def __init__(self, features_dir='features', config_dir='config'):
        self.features_dir = features_dir
        self.config_dir = config_dir
        self.features_file = os.path.join(features_dir, 'sku_location_features.json')
        self.model_selection_file = os.path.join(features_dir, 'model_selection_tags.json')
        self.num_workers = max(1, multiprocessing.cpu_count() - 1)  # 使用除了1个CPU外的所有可用CPU
        
        # 创建必要的目录
        if not os.path.exists(features_dir):
            os.makedirs(features_dir)
        
        # 初始化特征和模型选择标签
        self.features = self._load_features()
        self.model_selection_tags = self._load_model_selection_tags()
    
    def _load_features(self):
        """
        加载已存储的特征
        """
        if os.path.exists(self.features_file):
            try:
                with open(self.features_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"加载特征失败: {e}")
        return {}
    
    def _load_model_selection_tags(self):
        """
        加载已存储的模型选择标签
        """
        if os.path.exists(self.model_selection_file):
            try:
                with open(self.model_selection_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"加载模型选择标签失败: {e}")
        return {}
    
    def _save_features(self):
        """
        保存特征到文件
        """
        try:
            with open(self.features_file, 'w') as f:
                json.dump(self.features, f, indent=2)
        except Exception as e:
            print(f"保存特征失败: {e}")
    
    def _save_model_selection_tags(self):
        """
        保存模型选择标签到文件
        """
        try:
            with open(self.model_selection_file, 'w') as f:
                json.dump(self.model_selection_tags, f, indent=2)
        except Exception as e:
            print(f"保存模型选择标签失败: {e}")
    
    def calculate_cv(self, demand_series):
        """
        计算变异系数（Coefficient of Variation）
        CV = 标准差 / 平均值
        """
        mean_demand = np.mean(demand_series)
        if mean_demand == 0:
            return 0.0
        return float(np.std(demand_series) / mean_demand)
    
    def calculate_seasonality_strength(self, demand_series, period=7):
        """
        计算季节强度
        基于STL分解的季节性强度指标
        """
        from statsmodels.tsa.seasonal import STL
        
        if len(demand_series) < 2 * period:
            return 0.0
        
        try:
            stl = STL(demand_series, period=period)
            result = stl.fit()
            seasonal_std = np.std(result.seasonal)
            residual_std = np.std(result.resid)
            
            if (seasonal_std + residual_std) == 0:
                return 0.0
            
            return float(seasonal_std / (seasonal_std + residual_std))
        except Exception:
            return 0.0
    
    def calculate_intermittency_index(self, demand_series):
        """
        计算间歇性指数
        间歇性指数 = 零需求天数 / 总天数
        """
        zero_days = sum(1 for d in demand_series if d == 0)
        return float(zero_days / len(demand_series))
    
    def calculate_promotion_flag(self, demand_series, threshold=1.5):
        """
        计算促销标记
        如果某一天的需求超过平均值的threshold倍，则标记为促销
        """
        mean_demand = np.mean(demand_series)
        if mean_demand == 0:
            return 0.0
        
        promotion_days = sum(1 for d in demand_series if d > threshold * mean_demand)
        return float(promotion_days / len(demand_series))
    
    def calculate_demand_features(self, demand_series):
        """
        计算需求系列的所有统计特征
        """
        if len(demand_series) == 0:
            return {
                'cv': 0.0,
                'seasonality_strength': 0.0,
                'intermittency_index': 0.0,
                'promotion_flag': 0.0,
                'mean_demand': 0.0,
                'std_demand': 0.0,
                'min_demand': 0.0,
                'max_demand': 0.0,
                'skewness': 0.0,
                'kurtosis': 0.0
            }
        
        return {
            'cv': self.calculate_cv(demand_series),
            'seasonality_strength': self.calculate_seasonality_strength(demand_series),
            'intermittency_index': self.calculate_intermittency_index(demand_series),
            'promotion_flag': self.calculate_promotion_flag(demand_series),
            'mean_demand': float(np.mean(demand_series)),
            'std_demand': float(np.std(demand_series)),
            'min_demand': float(np.min(demand_series)),
            'max_demand': float(np.max(demand_series)),
            'skewness': float(stats.skew(demand_series)),
            'kurtosis': float(stats.kurtosis(demand_series))
        }
    
    def determine_model_selection_tag(self, features):
        """
        根据特征确定模型选择标签
        基于规则的模型选择逻辑
        """
        cv = features['cv']
        intermittency = features['intermittency_index']
        seasonality = features['seasonality_strength']
        
        # 高间歇性需求（间歇指数 > 0.3）
        if intermittency > 0.3:
            return 'croston' if seasonality < 0.2 else 'tbats'
        
        # 低间歇性需求
        if cv < 0.2:
            return 'holt_winters' if seasonality > 0.3 else 'arima'
        elif cv < 0.5:
            return 'sarimax' if seasonality > 0.3 else 'hybrid'
        else:
            return 'xgboost' if seasonality > 0.3 else 'lightgbm'
    
    def update_features(self, sku_id, location_id, demand_series, timestamp=None):
        """
        更新特定SKU×仓库的特征
        """
        key = f"{sku_id}_{location_id}"
        
        # 计算特征
        features = self.calculate_demand_features(demand_series)
        
        # 确定模型选择标签
        model_tag = self.determine_model_selection_tag(features)
        
        # 添加时间戳
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        features['last_updated'] = timestamp
        
        # 保存特征
        self.features[key] = features
        self.model_selection_tags[key] = model_tag
        
        # 持久化到文件
        self._save_features()
        self._save_model_selection_tags()
    
    def get_features(self, sku_id, location_id):
        """
        获取特定SKU×仓库的特征
        """
        key = f"{sku_id}_{location_id}"
        return self.features.get(key, {})
    
    def get_model_selection_tag(self, sku_id, location_id):
        """
        获取特定SKU×仓库的模型选择标签
        """
        key = f"{sku_id}_{location_id}"
        return self.model_selection_tags.get(key, 'default')
    
    def get_all_features(self):
        """
        获取所有SKU×仓库的特征
        """
        return self.features
    
    def get_all_model_tags(self):
        """
        获取所有SKU×仓库的模型选择标签
        """
        return self.model_selection_tags
    
    def batch_update_features(self, demand_data):
        """
        批量更新特征（并行计算）
        demand_data 应该是包含 'item_id', 'location_id', 'date', 'demand_qty' 列的数据框
        """
        # 按SKU×仓库分组
        grouped = demand_data.groupby(['item_id', 'location_id'])
        
        # 准备所有需要处理的组合
        all_combinations = []
        for (item_id, location_id), group in grouped:
            all_combinations.append((item_id, location_id, group['demand_qty'].values))
        
        # 并行计算所有特征
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # 定义处理单个组合的函数
            def process_combination(sku_id, location_id, demand_series):
                self.update_features(sku_id, location_id, demand_series)
            
            # 提交所有任务
            futures = [executor.submit(process_combination, sku_id, location_id, demand_series) 
                      for sku_id, location_id, demand_series in all_combinations]
            
            # 等待所有任务完成
            for future in concurrent.futures.as_completed(futures):
                future.result()
    
    def generate_feature_report(self):
        """
        生成特征报告
        """
        report = {
            'total_sku_location_pairs': len(self.features),
            'feature_summary': {},
            'model_distribution': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # 计算特征摘要
        if self.features:
            feature_df = pd.DataFrame.from_dict(self.features, orient='index')
            report['feature_summary'] = feature_df.describe().to_dict()
        
        # 计算模型分布
        if self.model_selection_tags:
            from collections import Counter
            model_counts = Counter(self.model_selection_tags.values())
            report['model_distribution'] = dict(model_counts)
        
        return report
