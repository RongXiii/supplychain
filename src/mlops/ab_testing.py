import pandas as pd
import numpy as np
import hashlib
import logging
from typing import Dict, List, Tuple, Optional
from scipy import stats
import json
import os
from datetime import datetime

class ABTestManager:
    """
    A/B测试管理器，负责流量分配、实验结果收集和统计分析
    """
    def __init__(self, experiment_id: str, config: Dict = None):
        """
        初始化A/B测试管理器
        
        Args:
            experiment_id: 实验ID
            config: 实验配置
        """
        self.logger = logging.getLogger(__name__)
        self.experiment_id = experiment_id
        
        # 默认配置
        default_config = {
            'traffic_allocation': {'control': 0.5, 'variant': 0.5},
            'sampling_method': 'random',  # 'random', 'sticky', 'stratified'
            'metrics': ['rmse', 'mae', 'mape'],
            'significance_level': 0.05,
            'min_sample_size': 100,
            'results_file': f'logs/ab_test_results_{experiment_id}.json'
        }
        
        # 合并配置
        self.config = {**default_config, **(config or {})}
        
        # 实验结果存储
        self.results = {
            'experiment_id': experiment_id,
            'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'end_time': None,
            'config': self.config,
            'control_results': [],
            'variant_results': []
        }
        
        # 确保结果文件目录存在
        os.makedirs(os.path.dirname(self.config['results_file']), exist_ok=True)
        
        self.logger.info(f"A/B测试管理器初始化: experiment_id={experiment_id}")
    
    def assign_traffic(self, user_id: str, product_id: str = None) -> str:
        """
        根据流量分配策略为用户/产品分配实验组
        
        Args:
            user_id: 用户ID或请求ID
            product_id: 产品ID（可选）
            
        Returns:
            分配的实验组: 'control' 或 'variant'
        """
        sampling_method = self.config['sampling_method']
        
        if sampling_method == 'sticky':
            # 粘性分配：相同的user_id始终分配到相同的组
            return self._sticky_allocation(user_id, product_id)
        elif sampling_method == 'stratified':
            # 分层分配：根据产品属性进行分层采样
            return self._stratified_allocation(user_id, product_id)
        else:
            # 随机分配
            return self._random_allocation()
    
    def _random_allocation(self) -> str:
        """
        随机分配流量
        
        Returns:
            分配的实验组: 'control' 或 'variant'
        """
        allocation = self.config['traffic_allocation']
        groups = list(allocation.keys())
        weights = list(allocation.values())
        return np.random.choice(groups, p=weights)
    
    def _sticky_allocation(self, user_id: str, product_id: str = None) -> str:
        """
        粘性分配：相同的user_id始终分配到相同的组
        
        Args:
            user_id: 用户ID或请求ID
            product_id: 产品ID（可选）
            
        Returns:
            分配的实验组: 'control' 或 'variant'
        """
        # 使用hash值进行粘性分配
        combined_id = f"{user_id}_{product_id}" if product_id else user_id
        hash_value = hashlib.md5(combined_id.encode()).hexdigest()
        # 将hash值转换为0-1之间的浮点数
        float_hash = int(hash_value, 16) / (2**128 - 1)
        
        allocation = self.config['traffic_allocation']
        cumulative = 0.0
        for group, weight in allocation.items():
            cumulative += weight
            if float_hash <= cumulative:
                return group
        
        return list(allocation.keys())[-1]  # 默认返回最后一个组
    
    def _stratified_allocation(self, user_id: str, product_id: str = None) -> str:
        """
        分层分配：根据产品属性进行分层采样
        
        Args:
            user_id: 用户ID或请求ID
            product_id: 产品ID（可选）
            
        Returns:
            分配的实验组: 'control' 或 'variant'
        """
        # 简化实现：仅根据产品ID的奇偶性分层
        # 在实际应用中，应该根据产品的ABC分类、需求模式等属性进行分层
        if product_id:
            # 使用产品ID的最后一位数字进行分层
            product_hash = int(product_id[-1]) if product_id[-1].isdigit() else 0
            if product_hash % 2 == 0:
                # 偶数产品分配到control组
                return 'control'
            else:
                # 奇数产品分配到variant组
                return 'variant'
        else:
            # 如果没有产品ID，退回到随机分配
            return self._random_allocation()
    
    def record_result(self, group: str, actual: float, predicted: float, 
                     metadata: Optional[Dict] = None) -> None:
        """
        记录实验结果
        
        Args:
            group: 实验组 ('control' 或 'variant')
            actual: 实际值
            predicted: 预测值
            metadata: 元数据，如时间戳、产品ID等
        """
        result = {
            'actual': actual,
            'predicted': predicted,
            'error': abs(actual - predicted),
            'squared_error': (actual - predicted) ** 2,
            'absolute_percentage_error': abs((actual - predicted) / actual) if actual != 0 else 0,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'metadata': metadata or {}
        }
        
        # 将结果添加到对应组
        if group == 'control':
            self.results['control_results'].append(result)
        elif group == 'variant':
            self.results['variant_results'].append(result)
        else:
            self.logger.warning(f"未知的实验组: {group}")
        
        # 定期保存结果
        self._save_results()
    
    def _save_results(self) -> None:
        """
        保存实验结果到文件
        """
        results_file = self.config['results_file']
        try:
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"保存实验结果失败: {e}")
    
    def get_results_summary(self) -> Dict:
        """
        获取实验结果摘要
        
        Returns:
            实验结果摘要
        """
        summary = {
            'experiment_id': self.experiment_id,
            'config': self.config,
            'sample_sizes': {
                'control': len(self.results['control_results']),
                'variant': len(self.results['variant_results'])
            },
            'metrics': {}
        }
        
        # 计算各指标
        for metric in self.config['metrics']:
            control_metric = self._calculate_metric(self.results['control_results'], metric)
            variant_metric = self._calculate_metric(self.results['variant_results'], metric)
            
            summary['metrics'][metric] = {
                'control': control_metric,
                'variant': variant_metric,
                'difference': variant_metric - control_metric,
                'improvement': ((control_metric - variant_metric) / control_metric * 100) if control_metric != 0 else 0
            }
        
        return summary
    
    def _calculate_metric(self, results: List[Dict], metric: str) -> float:
        """
        计算指定指标
        
        Args:
            results: 结果列表
            metric: 指标名称
            
        Returns:
            指标值
        """
        if not results:
            return 0.0
        
        if metric == 'rmse':
            # 均方根误差
            mse = np.mean([r['squared_error'] for r in results])
            return np.sqrt(mse)
        elif metric == 'mae':
            # 平均绝对误差
            return np.mean([r['error'] for r in results])
        elif metric == 'mape':
            # 平均绝对百分比误差
            return np.mean([r['absolute_percentage_error'] for r in results]) * 100
        else:
            self.logger.warning(f"未知的指标: {metric}")
            return 0.0
    
    def run_significance_test(self) -> Dict:
        """
        运行显著性检验，比较两组的性能差异
        
        Returns:
            显著性检验结果
        """
        control_size = len(self.results['control_results'])
        variant_size = len(self.results['variant_results'])
        min_size = self.config['min_sample_size']
        
        if control_size < min_size or variant_size < min_size:
            return {
                'status': 'insufficient_sample',
                'message': f"样本量不足，需要至少 {min_size} 个样本",
                'control_sample_size': control_size,
                'variant_sample_size': variant_size
            }
        
        significance_level = self.config['significance_level']
        test_results = {
            'status': 'completed',
            'significance_level': significance_level,
            'control_sample_size': control_size,
            'variant_sample_size': variant_size,
            'metrics': {}
        }
        
        # 对每个指标进行显著性检验
        for metric in self.config['metrics']:
            # 获取两组的指标值
            control_metric_values = [r[self._get_metric_field(metric)] for r in self.results['control_results']]
            variant_metric_values = [r[self._get_metric_field(metric)] for r in self.results['variant_results']]
            
            # 运行t检验
            t_statistic, p_value = stats.ttest_ind(control_metric_values, variant_metric_values)
            
            # 计算效应量（Cohen's d）
            cohens_d = self._calculate_cohens_d(control_metric_values, variant_metric_values)
            
            test_results['metrics'][metric] = {
                't_statistic': t_statistic,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'is_significant': p_value < significance_level,
                'conclusion': '差异显著' if p_value < significance_level else '差异不显著'
            }
        
        return test_results
    
    def _get_metric_field(self, metric: str) -> str:
        """
        根据指标名称获取结果字典中的字段名
        
        Args:
            metric: 指标名称
            
        Returns:
            字段名
        """
        metric_field_map = {
            'rmse': 'squared_error',  # RMSE基于平方误差计算
            'mae': 'error',
            'mape': 'absolute_percentage_error'
        }
        return metric_field_map.get(metric, 'error')
    
    def _calculate_cohens_d(self, group1: List[float], group2: List[float]) -> float:
        """
        计算Cohen's d效应量
        
        Args:
            group1: 第一组数据
            group2: 第二组数据
            
        Returns:
            Cohen's d效应量
        """
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        n1, n2 = len(group1), len(group2)
        
        # 合并标准差
        pooled_std = np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))
        
        return (mean1 - mean2) / pooled_std
    
    def end_experiment(self) -> Dict:
        """
        结束实验，保存最终结果
        
        Returns:
            实验最终结果
        """
        self.results['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self._save_results()
        
        # 生成完整报告
        final_results = {
            'experiment_summary': self.get_results_summary(),
            'significance_test': self.run_significance_test()
        }
        
        self.logger.info(f"A/B测试结束: experiment_id={self.experiment_id}")
        return final_results
    
    def get_experiment_results(self) -> Dict:
        """
        获取完整的实验结果
        
        Returns:
            完整的实验结果
        """
        return self.results

# A/B测试配置管理器
class ABTestConfigManager:
    """
    A/B测试配置管理器，用于管理多个A/B测试实验
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.experiments = {}
        self.config_file = 'config/ab_test_config.json'
        
        # 加载配置
        self._load_config()
        self.logger.info("A/B测试配置管理器初始化")
    
    def _load_config(self) -> None:
        """
        加载A/B测试配置
        """
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    self.config = json.load(f)
            else:
                self.config = {
                    'active_experiments': [],
                    'experiment_configs': {}
                }
        except Exception as e:
            self.logger.error(f"加载A/B测试配置失败: {e}")
            self.config = {
                'active_experiments': [],
                'experiment_configs': {}
            }
    
    def _save_config(self) -> None:
        """
        保存A/B测试配置
        """
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"保存A/B测试配置失败: {e}")
    
    def create_experiment(self, experiment_id: str, config: Dict) -> ABTestManager:
        """
        创建新的A/B测试实验
        
        Args:
            experiment_id: 实验ID
            config: 实验配置
            
        Returns:
            A/B测试管理器实例
        """
        if experiment_id in self.experiments:
            self.logger.warning(f"实验 {experiment_id} 已存在")
            return self.experiments[experiment_id]
        
        # 创建实验
        ab_test_manager = ABTestManager(experiment_id, config)
        self.experiments[experiment_id] = ab_test_manager
        
        # 更新配置
        if experiment_id not in self.config['active_experiments']:
            self.config['active_experiments'].append(experiment_id)
        
        self.config['experiment_configs'][experiment_id] = config
        self._save_config()
        
        self.logger.info(f"创建新实验: experiment_id={experiment_id}")
        return ab_test_manager
    
    def get_experiment(self, experiment_id: str) -> Optional[ABTestManager]:
        """
        获取A/B测试实验实例
        
        Args:
            experiment_id: 实验ID
            
        Returns:
            A/B测试管理器实例或None
        """
        return self.experiments.get(experiment_id)
    
    def list_active_experiments(self) -> List[str]:
        """
        获取所有活跃的实验ID
        
        Returns:
            活跃实验ID列表
        """
        return self.config['active_experiments']
    
    def end_experiment(self, experiment_id: str) -> Dict:
        """
        结束指定的A/B测试实验
        
        Args:
            experiment_id: 实验ID
            
        Returns:
            实验最终结果
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"实验不存在: {experiment_id}")
        
        ab_test_manager = self.experiments[experiment_id]
        final_results = ab_test_manager.end_experiment()
        
        # 从活跃实验列表中移除
        if experiment_id in self.config['active_experiments']:
            self.config['active_experiments'].remove(experiment_id)
        
        # 保存配置
        self._save_config()
        
        # 从内存中移除实验实例
        del self.experiments[experiment_id]
        
        self.logger.info(f"结束实验: experiment_id={experiment_id}")
        return final_results
