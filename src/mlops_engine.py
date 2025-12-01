import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from scipy import stats
import json
import os
from datetime import datetime
import copy

class MLOpsEngine:
    """
    MLOps引擎，负责误差分析、漂移检测、模型重训、参数自适应、策略回滚与灰度上线
    """
    
    def __init__(self, models_dir='models', metrics_dir='metrics', config_dir='config'):
        self.models_dir = models_dir
        self.metrics_dir = metrics_dir
        self.config_dir = config_dir
        
        # 创建必要的目录
        for dir_path in [models_dir, metrics_dir, config_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
        
        # 初始化模型历史记录和指标
        self.model_history = {}
        self.error_metrics = {}
        self.drift_detection_results = {}
        self.current_policies = {}
        self.gray_release_config = {
            'enabled': False,
            'products': [],
            'traffic_split': 0.5
        }
        
    def calculate_error_metrics(self, actual, forecast, product_id):
        """
        计算预测误差指标：MAPE、SMAPE、RMSE
        
        Args:
            actual: 实际值列表
            forecast: 预测值列表
            product_id: 产品ID
            
        Returns:
            metrics: 包含MAPE、SMAPE、RMSE的字典
        """
        actual = np.array(actual)
        forecast = np.array(forecast)
        
        # 确保数据不为空且长度一致
        if len(actual) == 0 or len(forecast) == 0 or len(actual) != len(forecast):
            return None
        
        # 移除实际值为0的数据点，避免除零错误
        non_zero_mask = actual != 0
        actual = actual[non_zero_mask]
        forecast = forecast[non_zero_mask]
        
        if len(actual) == 0:
            return None
        
        # 计算MAPE
        mape = mean_absolute_percentage_error(actual, forecast) * 100
        
        # 计算SMAPE
        smape = (100 / len(actual)) * np.sum(2 * np.abs(forecast - actual) / (np.abs(actual) + np.abs(forecast)))
        
        # 计算RMSE
        rmse = np.sqrt(mean_squared_error(actual, forecast))
        
        # 将numpy类型转换为Python原生类型
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        metrics = {
            'mape': float(round(mape, 2)),
            'smape': float(round(smape, 2)),
            'rmse': float(round(rmse, 2)),
            'timestamp': datetime.now().isoformat()
        }
        
        # 确保所有值都是Python原生类型
        metrics = convert_numpy_types(metrics)
        
        # 保存指标
        self._save_metrics(product_id, metrics)
        
        return metrics
    
    def detect_drift(self, baseline_data, current_data, product_id, alpha=0.1):
        """
        检测数据漂移，使用KS检验检测分布变化
        
        Args:
            baseline_data: 基线数据（训练数据）
            current_data: 当前数据（新数据）
            product_id: 产品ID
            alpha: 显著性水平（调整为0.1以降低灵敏度）
            
        Returns:
            drift_result: 漂移检测结果
        """
        # 确保数据不为空且有足够的样本量
        if len(baseline_data) < 10 or len(current_data) < 10:
            return {
                'drift_detected': False,
                'p_value': 1.0,
                'test_statistic': 0.0,
                'reason': '样本量不足',
                'timestamp': datetime.now().isoformat()
            }
        
        # 使用KS检验检测分布漂移
        test_statistic, p_value = stats.ks_2samp(baseline_data, current_data)
        
        # 对于小样本数据，调整漂移检测逻辑：
        # 1. 提高显著性水平（降低灵敏度）
        # 2. 只有当p值非常小时才认为检测到漂移
        drift_detected = p_value < alpha and test_statistic > 0.3
        
        drift_result = {
            'drift_detected': drift_detected,
            'p_value': round(p_value, 4),
            'test_statistic': round(test_statistic, 4),
            'alpha': alpha,
            'sample_size': {
                'baseline': len(baseline_data),
                'current': len(current_data)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # 保存漂移检测结果
        self.drift_detection_results[product_id] = drift_result
        
        return drift_result
    
    def adaptive_params_update(self, product_id, historical_data, safety_stock_params):
        """
        参数自适应：动态更新安全库存参数（z值和标准差σ）
        
        Args:
            product_id: 产品ID
            historical_data: 历史数据
            safety_stock_params: 当前安全库存参数
            
        Returns:
            updated_params: 更新后的安全库存参数
        """
        # 计算历史需求的标准差
        historical_demands = np.array(historical_data)
        std_demand = np.std(historical_demands)
        
        # 计算需求的变异系数
        cv = std_demand / np.mean(historical_demands) if np.mean(historical_demands) != 0 else 0
        
        # 基于变异系数动态调整z值
        # 需求波动越大，z值越大（越保守）
        if cv < 0.2:
            z_value = 1.28  # 90% 服务水平
        elif cv < 0.5:
            z_value = 1.65  # 95% 服务水平
        else:
            z_value = 2.33  # 99% 服务水平
        
        updated_params = {
            'z_value': z_value,
            'std_demand': std_demand,
            'cv': cv,
            'timestamp': datetime.now().isoformat()
        }
        
        # 保存更新后的参数
        self._save_params(product_id, updated_params)
        
        return updated_params
    
    def should_retrain_model(self, product_id, metrics_history, drift_results):
        """
        决定是否需要重训模型
        
        Args:
            product_id: 产品ID
            metrics_history: 指标历史记录
            drift_results: 漂移检测结果
            
        Returns:
            should_retrain: 是否需要重训模型
            reason: 重训原因
        """
        # 检查漂移检测结果
        if drift_results and drift_results['drift_detected']:
            return True, '数据漂移检测到'
        
        # 检查指标是否恶化
        if len(metrics_history) < 3:
            return False, '指标历史数据不足'
        
        # 计算最近3个周期的MAPE变化
        recent_mapes = [m['mape'] for m in metrics_history[-3:]]
        
        # 如果MAPE连续上升且最后一个MAPE超过阈值（例如20%），则需要重训
        if (recent_mapes[1] > recent_mapes[0] and recent_mapes[2] > recent_mapes[1] and 
            recent_mapes[2] > 20):
            return True, '指标持续恶化'
        
        # 检查是否超过最大训练周期（例如30天）
        last_metric_time = datetime.fromisoformat(metrics_history[-1]['timestamp'])
        current_time = datetime.now()
        if (current_time - last_metric_time).days > 30:
            return True, '超过最大训练周期'
        
        return False, '无需重训'
    
    def save_model(self, product_id, model, model_name, metrics):
        """
        保存模型及其元数据
        
        Args:
            product_id: 产品ID
            model: 模型对象
            model_name: 模型名称
            metrics: 模型指标
        """
        import joblib
        
        # 将numpy类型转换为Python原生类型的辅助函数
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        # 确保product_id是Python原生类型
        product_id = convert_numpy_types(product_id)
        
        # 保存模型文件
        model_path = os.path.join(self.models_dir, f'model_{product_id}.pkl')
        joblib.dump(model, model_path)
        
        # 转换指标中的numpy类型
        converted_metrics = convert_numpy_types(metrics)
        
        # 保存模型元数据
        model_metadata = {
            'product_id': product_id,
            'model_name': model_name,
            'metrics': converted_metrics,
            'save_time': datetime.now().isoformat()
        }
        
        metadata_path = os.path.join(self.models_dir, f'model_{product_id}_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(model_metadata, f, indent=2)
        
        # 更新模型历史记录
        if product_id not in self.model_history:
            self.model_history[product_id] = []
        
        self.model_history[product_id].append({
            'model_path': model_path,
            'metadata_path': metadata_path,
            'save_time': datetime.now().isoformat()
        })
    
    def load_model(self, product_id, version='latest'):
        """
        加载模型
        
        Args:
            product_id: 产品ID
            version: 模型版本，'latest'表示最新版本
            
        Returns:
            model: 模型对象
            metadata: 模型元数据
        """
        import joblib
        
        if version == 'latest':
            # 加载最新模型
            metadata_path = os.path.join(self.models_dir, f'model_{product_id}_metadata.json')
            model_path = os.path.join(self.models_dir, f'model_{product_id}.pkl')
        else:
            # 加载指定版本模型
            # 这里简化处理，实际可以实现版本管理
            metadata_path = os.path.join(self.models_dir, f'model_{product_id}_metadata_{version}.json')
            model_path = os.path.join(self.models_dir, f'model_{product_id}_{version}.pkl')
        
        if not os.path.exists(model_path) or not os.path.exists(metadata_path):
            return None, None
        
        # 加载模型
        model = joblib.load(model_path)
        
        # 加载元数据
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return model, metadata
    
    def enable_gray_release(self, products, traffic_split=0.5):
        """
        启用灰度上线
        
        Args:
            products: 参与灰度上线的产品列表
            traffic_split: 新策略的流量比例（0-1）
        """
        self.gray_release_config = {
            'enabled': True,
            'products': products,
            'traffic_split': traffic_split
        }
        
        # 保存灰度配置
        self._save_gray_config()
    
    def disable_gray_release(self):
        """
        禁用灰度上线
        """
        self.gray_release_config['enabled'] = False
        self._save_gray_config()
    
    def get_policy_for_product(self, product_id):
        """
        获取产品的当前策略（根据灰度配置）
        
        Args:
            product_id: 产品ID
            
        Returns:
            policy: 当前策略
        """
        if not self.gray_release_config['enabled'] or product_id not in self.gray_release_config['products']:
            # 不使用灰度，返回当前策略
            return self.current_policies.get(product_id, {})
        
        # 使用灰度，根据流量分配选择策略
        if np.random.random() < self.gray_release_config['traffic_split']:
            # 使用新策略
            return self.current_policies.get(product_id, {})
        else:
            # 使用旧策略（回滚）
            return self._load_rollback_policy(product_id)
    
    def rollback_policy(self, product_id, version='previous'):
        """
        策略回滚：恢复到之前的策略版本
        
        Args:
            product_id: 产品ID
            version: 回滚版本，'previous'表示上一个版本
            
        Returns:
            success: 是否回滚成功
        """
        # 加载回滚策略
        rollback_policy = self._load_rollback_policy(product_id, version)
        
        if rollback_policy:
            # 保存当前策略作为回滚点
            self._save_rollback_point(product_id, self.current_policies.get(product_id, {}))
            
            # 回滚到之前的策略
            self.current_policies[product_id] = rollback_policy
            return True
        
        return False
    
    def _save_metrics(self, product_id, metrics):
        """
        保存指标到文件
        """
        metrics_file = os.path.join(self.metrics_dir, f'metrics_{product_id}.json')
        
        # 读取现有指标
        existing_metrics = []
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                existing_metrics = json.load(f)
        
        # 添加新指标
        existing_metrics.append(metrics)
        
        # 保存更新后的指标
        with open(metrics_file, 'w') as f:
            json.dump(existing_metrics, f, indent=2)
    
    def _save_params(self, product_id, params):
        """
        保存参数到文件
        """
        params_file = os.path.join(self.config_dir, f'safety_stock_params_{product_id}.json')
        
        # 读取现有参数
        existing_params = []
        if os.path.exists(params_file):
            with open(params_file, 'r') as f:
                existing_params = json.load(f)
        
        # 添加新参数
        existing_params.append(params)
        
        # 保存更新后的参数
        with open(params_file, 'w') as f:
            json.dump(existing_params, f, indent=2)
    
    def _save_gray_config(self):
        """
        保存灰度配置
        """
        gray_config_file = os.path.join(self.config_dir, 'gray_release_config.json')
        with open(gray_config_file, 'w') as f:
            json.dump(self.gray_release_config, f, indent=2)
    
    def _save_rollback_point(self, product_id, policy):
        """
        保存回滚点
        """
        rollback_dir = os.path.join(self.config_dir, 'rollback_policies')
        if not os.path.exists(rollback_dir):
            os.makedirs(rollback_dir)
        
        rollback_file = os.path.join(rollback_dir, f'policy_{product_id}_rollback.json')
        with open(rollback_file, 'w') as f:
            json.dump(policy, f, indent=2)
    
    def _load_rollback_policy(self, product_id, version='previous'):
        """
        加载回滚策略
        """
        rollback_file = os.path.join(self.config_dir, 'rollback_policies', f'policy_{product_id}_rollback.json')
        
        if os.path.exists(rollback_file):
            with open(rollback_file, 'r') as f:
                return json.load(f)
        
        return {}
    
    def get_model_performance_report(self, product_id, time_range='30d'):
        """
        获取模型性能报告
        
        Args:
            product_id: 产品ID
            time_range: 时间范围，如'7d'、'30d'、'90d'
            
        Returns:
            report: 模型性能报告
        """
        # 读取指标文件
        metrics_file = os.path.join(self.metrics_dir, f'metrics_{product_id}.json')
        
        if not os.path.exists(metrics_file):
            return None
        
        with open(metrics_file, 'r') as f:
            all_metrics = json.load(f)
        
        # 过滤指定时间范围内的指标
        end_time = datetime.now()
        if time_range == '7d':
            start_time = end_time - pd.Timedelta(days=7)
        elif time_range == '30d':
            start_time = end_time - pd.Timedelta(days=30)
        elif time_range == '90d':
            start_time = end_time - pd.Timedelta(days=90)
        else:
            start_time = datetime.min
        
        filtered_metrics = []
        for metrics in all_metrics:
            metric_time = datetime.fromisoformat(metrics['timestamp'])
            if metric_time >= start_time and metric_time <= end_time:
                filtered_metrics.append(metrics)
        
        if not filtered_metrics:
            return None
        
        # 计算平均指标
        avg_mape = np.mean([m['mape'] for m in filtered_metrics])
        avg_smape = np.mean([m['smape'] for m in filtered_metrics])
        avg_rmse = np.mean([m['rmse'] for m in filtered_metrics])
        
        # 计算指标趋势（最近7天与之前的对比）
        if len(filtered_metrics) >= 14:
            recent_metrics = filtered_metrics[-7:]
            previous_metrics = filtered_metrics[-14:-7]
            
            recent_mape = np.mean([m['mape'] for m in recent_metrics])
            previous_mape = np.mean([m['mape'] for m in previous_metrics])
            mape_trend = recent_mape - previous_mape
        else:
            mape_trend = 0
        
        # 生成报告
        report = {
            'product_id': product_id,
            'time_range': time_range,
            'metrics_count': len(filtered_metrics),
            'average_metrics': {
                'mape': round(avg_mape, 2),
                'smape': round(avg_smape, 2),
                'rmse': round(avg_rmse, 2)
            },
            'mape_trend': round(mape_trend, 2),  # 正数表示指标恶化，负数表示指标改善
            'drift_detection': self.drift_detection_results.get(product_id, {}),
            'report_time': datetime.now().isoformat()
        }
        
        return report
