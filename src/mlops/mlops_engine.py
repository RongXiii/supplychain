import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from scipy import stats
import json
import os
from datetime import datetime
import copy
import time

# 添加日志管理器
from src.system.logging_manager import get_logger, log_performance
# 导入A/B测试框架
from src.mlops.ab_testing import ABTestManager, ABTestConfigManager

class MLOpsEngine:
    """
    MLOps引擎，负责误差分析、漂移检测、模型重训、参数自适应、策略回滚与灰度上线
    """
    
    def __init__(self, models_dir='models', metrics_dir='metrics', config_dir='config'):
        # 初始化日志记录器
        self.logger = get_logger('mlops_engine')
        self.logger.info("Initializing MLOpsEngine")
        
        self.models_dir = models_dir
        self.metrics_dir = metrics_dir
        self.config_dir = config_dir
        
        # 创建必要的目录
        for dir_path in [models_dir, metrics_dir, config_dir]:
            if not os.path.exists(dir_path):
                self.logger.debug(f"Creating directory: {dir_path}")
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
        
        # 初始化A/B测试框架
        self.ab_test_manager = ABTestManager(test_id='default')
        self.ab_test_config_manager = ABTestConfigManager()
        
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
        start_time = time.time()
        self.logger.info(f"Calculating error metrics for product: {product_id}")
        
        actual = np.array(actual)
        forecast = np.array(forecast)
        
        # 确保数据不为空且长度一致
        if len(actual) == 0 or len(forecast) == 0 or len(actual) != len(forecast):
            self.logger.warning(f"Invalid data for error metrics calculation: product={product_id}, actual_len={len(actual)}, forecast_len={len(forecast)}")
            return None
        
        # 移除实际值为0的数据点，避免除零错误
        non_zero_mask = actual != 0
        actual = actual[non_zero_mask]
        forecast = forecast[non_zero_mask]
        
        if len(actual) == 0:
            self.logger.warning(f"No non-zero actual values for error metrics calculation: product={product_id}")
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
        
        self.logger.debug(f"Calculated error metrics for product: {product_id}, MAPE: {metrics['mape']}, SMAPE: {metrics['smape']}, RMSE: {metrics['rmse']}")
        log_performance("calculate_error_metrics", time.time() - start_time, product_id=product_id, data_points=len(actual))
        
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
        start_time = time.time()
        self.logger.info(f"Detecting drift for product: {product_id}")
        
        # 确保数据不为空且有足够的样本量
        if len(baseline_data) < 10 or len(current_data) < 10:
            self.logger.warning(f"Insufficient sample size for drift detection: product={product_id}, baseline={len(baseline_data)}, current={len(current_data)}")
            result = {
                'drift_detected': False,
                'p_value': 1.0,
                'test_statistic': 0.0,
                'reason': '样本量不足',
                'timestamp': datetime.now().isoformat()
            }
            self.drift_detection_results[product_id] = result
            log_performance("detect_drift", time.time() - start_time, product_id=product_id, baseline_size=len(baseline_data), current_size=len(current_data))
            return result
        
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
        
        self.logger.info(f"Drift detection result for product {product_id}: drift_detected={drift_detected}, p_value={p_value:.4f}, test_statistic={test_statistic:.4f}")
        log_performance("detect_drift", time.time() - start_time, product_id=product_id, baseline_size=len(baseline_data), current_size=len(current_data), drift_detected=drift_detected)
        
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
    
    def save_model(self, product_id, model, model_name, metrics, version=None):
        """
        保存模型及其元数据，支持版本管理
        
        Args:
            product_id: 产品ID
            model: 模型对象
            model_name: 模型名称
            metrics: 模型指标
            version: 版本号，若为None则自动生成
        
        Returns:
            version: 保存的模型版本号
        """
        start_time = time.time()
        self.logger.info(f"Saving model for product: {product_id}, model_name: {model_name}")
        
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
        
        # 获取或生成版本号
        if version is None:
            version = self._generate_version_number(product_id)
        
        # 保存特定版本的模型文件
        version_model_path = os.path.join(self.models_dir, f'model_{product_id}_{version}.pkl')
        joblib.dump(model, version_model_path)
        self.logger.debug(f"Saved versioned model file: {version_model_path}")
        
        # 同时保持最新模型的引用
        latest_model_path = os.path.join(self.models_dir, f'model_{product_id}.pkl')
        joblib.dump(model, latest_model_path)
        self.logger.debug(f"Updated latest model file: {latest_model_path}")
        
        # 转换指标中的numpy类型
        converted_metrics = convert_numpy_types(metrics)
        
        # 保存模型元数据，包含版本信息
        model_metadata = {
            'product_id': product_id,
            'model_name': model_name,
            'version': version,
            'metrics': converted_metrics,
            'save_time': datetime.now().isoformat()
        }
        
        # 保存特定版本的元数据
        version_metadata_path = os.path.join(self.models_dir, f'model_{product_id}_metadata_{version}.json')
        with open(version_metadata_path, 'w') as f:
            json.dump(model_metadata, f, indent=2)
        self.logger.debug(f"Saved versioned model metadata: {version_metadata_path}")
        
        # 更新最新元数据
        latest_metadata_path = os.path.join(self.models_dir, f'model_{product_id}_metadata.json')
        with open(latest_metadata_path, 'w') as f:
            json.dump(model_metadata, f, indent=2)
        self.logger.debug(f"Updated latest model metadata: {latest_metadata_path}")
        
        # 更新模型历史记录
        if product_id not in self.model_history:
            self.model_history[product_id] = []
        
        self.model_history[product_id].append({
            'version': version,
            'model_path': version_model_path,
            'metadata_path': version_metadata_path,
            'save_time': datetime.now().isoformat()
        })
        
        # 保存模型历史记录到文件
        self._save_model_history(product_id)
        
        self.logger.info(f"Model saved successfully for product: {product_id}, version: {version}")
        log_performance("save_model", time.time() - start_time, product_id=product_id, model_name=model_name, version=version)
        
        return version
    
    def load_model(self, product_id, version='latest'):
        """
        加载模型，支持版本管理
        
        Args:
            product_id: 产品ID
            version: 模型版本，'latest'表示最新版本
            
        Returns:
            model: 模型对象
            metadata: 模型元数据
        """
        start_time = time.time()
        self.logger.info(f"Loading model for product: {product_id}, version: {version}")
        
        import joblib
        
        # 获取模型文件路径
        if version == 'latest':
            # 加载最新模型
            metadata_path = os.path.join(self.models_dir, f'model_{product_id}_metadata.json')
            model_path = os.path.join(self.models_dir, f'model_{product_id}.pkl')
        else:
            # 加载指定版本模型
            metadata_path = os.path.join(self.models_dir, f'model_{product_id}_metadata_{version}.json')
            model_path = os.path.join(self.models_dir, f'model_{product_id}_{version}.pkl')
        
        if not os.path.exists(model_path) or not os.path.exists(metadata_path):
            self.logger.error(f"Model files not found for product: {product_id}, version: {version}")
            return None, None
        
        # 加载模型
        model = joblib.load(model_path)
        self.logger.debug(f"Loaded model file: {model_path}")
        
        # 加载元数据
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        self.logger.debug(f"Loaded model metadata: {metadata_path}")
        
        log_performance("load_model", time.time() - start_time, product_id=product_id, version=version, model_name=metadata.get('model_name', 'unknown'))
        return model, metadata
    
    def _generate_version_number(self, product_id):
        """
        为新模型生成版本号
        
        Args:
            product_id: 产品ID
            
        Returns:
            version: 生成的版本号
        """
        # 加载模型历史记录
        self._load_model_history(product_id)
        
        if product_id not in self.model_history or len(self.model_history[product_id]) == 0:
            return 'v1'
        
        # 找到最新版本号并递增
        versions = [entry['version'] for entry in self.model_history[product_id]]
        versions.sort()
        
        # 解析最新版本号，如v1 -> 1
        latest_version = versions[-1]
        version_num = int(latest_version[1:]) if latest_version.startswith('v') else int(latest_version)
        
        # 生成新版本号
        return f'v{version_num + 1}'
    
    def _save_model_history(self, product_id):
        """
        保存模型历史记录到文件
        
        Args:
            product_id: 产品ID
        """
        history_file = os.path.join(self.models_dir, f'model_history_{product_id}.json')
        with open(history_file, 'w') as f:
            json.dump(self.model_history.get(product_id, []), f, indent=2)
    
    def _load_model_history(self, product_id):
        """
        从文件加载模型历史记录
        
        Args:
            product_id: 产品ID
        """
        history_file = os.path.join(self.models_dir, f'model_history_{product_id}.json')
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                self.model_history[product_id] = json.load(f)
    
    def list_model_versions(self, product_id):
        """
        列出产品的所有模型版本
        
        Args:
            product_id: 产品ID
            
        Returns:
            versions: 版本列表，包含版本号和元数据
        """
        # 加载模型历史记录
        self._load_model_history(product_id)
        
        if product_id not in self.model_history:
            return []
        
        # 返回按版本号排序的版本列表
        versions = sorted(self.model_history[product_id], key=lambda x: x['version'])
        return versions
    
    def rollback_model(self, product_id, version):
        """
        回滚模型到指定版本
        
        Args:
            product_id: 产品ID
            version: 要回滚到的版本号
            
        Returns:
            success: 是否回滚成功
        """
        self.logger.info(f"Rolling back model for product: {product_id} to version: {version}")
        
        # 加载指定版本的模型
        model, metadata = self.load_model(product_id, version)
        if not model or not metadata:
            self.logger.error(f"Failed to load model version: {version} for product: {product_id}")
            return False
        
        # 保存为最新版本
        latest_model_path = os.path.join(self.models_dir, f'model_{product_id}.pkl')
        latest_metadata_path = os.path.join(self.models_dir, f'model_{product_id}_metadata.json')
        
        import joblib
        joblib.dump(model, latest_model_path)
        
        with open(latest_metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Model rolled back successfully for product: {product_id} to version: {version}")
        return True
    
    def delete_model_version(self, product_id, version):
        """
        删除指定版本的模型
        
        Args:
            product_id: 产品ID
            version: 要删除的版本号
            
        Returns:
            success: 是否删除成功
        """
        self.logger.info(f"Deleting model version: {version} for product: {product_id}")
        
        # 加载模型历史记录
        self._load_model_history(product_id)
        
        if product_id not in self.model_history:
            self.logger.error(f"No model history found for product: {product_id}")
            return False
        
        # 查找要删除的版本
        version_entry = None
        for entry in self.model_history[product_id]:
            if entry['version'] == version:
                version_entry = entry
                break
        
        if not version_entry:
            self.logger.error(f"Model version: {version} not found for product: {product_id}")
            return False
        
        # 删除模型文件和元数据文件
        try:
            if os.path.exists(version_entry['model_path']):
                os.remove(version_entry['model_path'])
            if os.path.exists(version_entry['metadata_path']):
                os.remove(version_entry['metadata_path'])
        except Exception as e:
            self.logger.error(f"Error deleting model files: {e}")
            return False
        
        # 从历史记录中移除该版本
        self.model_history[product_id].remove(version_entry)
        
        # 保存更新后的历史记录
        self._save_model_history(product_id)
        
        self.logger.info(f"Model version: {version} deleted successfully for product: {product_id}")
        return True
    
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
        获取产品的当前策略（根据灰度配置或A/B测试配置）
        
        Args:
            product_id: 产品ID
            
        Returns:
            policy: 当前策略
        """
        # 首先检查是否有正在进行的A/B测试
        active_test = self.ab_test_manager.get_active_test_for_product(product_id)
        if active_test:
            # 使用A/B测试的流量分配
            return self.ab_test_manager.get_treatment_for_product(product_id)
        
        # 否则使用灰度配置
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
    
    # A/B测试相关方法
    def create_ab_test(self, test_id, name, description, treatments, traffic_allocation, products):
        """
        创建A/B测试
        
        Args:
            test_id: 测试ID
            name: 测试名称
            description: 测试描述
            treatments: 处理组列表，每个处理组包含treatment_id和对应的策略
            traffic_allocation: 流量分配比例
            products: 参与测试的产品列表
            
        Returns:
            test_id: 创建的测试ID
        """
        self.logger.info(f"Creating A/B test: {test_id} for products: {products}")
        
        # 创建测试配置
        test_config = {
            'test_id': test_id,
            'name': name,
            'description': description,
            'treatments': treatments,
            'traffic_allocation': traffic_allocation,
            'products': products
        }
        
        # 使用ABTestConfigManager创建测试
        self.ab_test_config_manager.create_test(test_config)
        
        return test_id
    
    def start_ab_test(self, test_id):
        """
        启动A/B测试
        
        Args:
            test_id: 测试ID
            
        Returns:
            success: 是否启动成功
        """
        self.logger.info(f"Starting A/B test: {test_id}")
        
        # 获取测试配置
        test_config = self.ab_test_config_manager.get_test_config(test_id)
        if not test_config:
            self.logger.error(f"Test configuration not found: {test_id}")
            return False
        
        # 使用ABTestManager启动测试
        return self.ab_test_manager.start_test(test_config)
    
    def end_ab_test(self, test_id):
        """
        结束A/B测试
        
        Args:
            test_id: 测试ID
            
        Returns:
            results: 测试结果
        """
        self.logger.info(f"Ending A/B test: {test_id}")
        
        # 结束测试并获取结果
        results = self.ab_test_manager.end_test(test_id)
        
        if results:
            # 保存测试结果
            results_file = os.path.join(self.metrics_dir, 'ab_tests', f'results_{test_id}.json')
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
        return results
    
    def record_ab_test_result(self, test_id, product_id, treatment_id, actual, forecast):
        """
        记录A/B测试结果
        
        Args:
            test_id: 测试ID
            product_id: 产品ID
            treatment_id: 处理组ID
            actual: 实际值
            forecast: 预测值
            
        Returns:
            success: 是否记录成功
        """
        self.logger.debug(f"Recording A/B test result: test_id={test_id}, product_id={product_id}, treatment_id={treatment_id}")
        
        # 计算指标
        metrics = self.calculate_error_metrics(actual, forecast, product_id)
        if not metrics:
            self.logger.warning(f"Failed to calculate metrics for A/B test result: test_id={test_id}, product_id={product_id}")
            return False
        
        # 记录结果
        return self.ab_test_manager.record_result(test_id, product_id, treatment_id, metrics)
    
    def get_ab_test_results(self, test_id):
        """
        获取A/B测试结果
        
        Args:
            test_id: 测试ID
            
        Returns:
            results: 测试结果
        """
        return self.ab_test_manager.get_test_results(test_id)
    
    def get_all_ab_tests(self):
        """
        获取所有A/B测试
        
        Returns:
            tests: 所有测试的列表
        """
        return self.ab_test_config_manager.get_all_tests()
    
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
