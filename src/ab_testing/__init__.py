import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from datetime import datetime

class TestVariant:
    """测试变体类，代表A/B测试中的一个模型或策略"""
    def __init__(self, variant_id, name, model=None, strategy=None, sample_size=0):
        """
        初始化测试变体
        
        Args:
            variant_id: 变体唯一标识符
            name: 变体名称
            model: 变体对应的模型对象
            strategy: 变体对应的策略函数
            sample_size: 分配的样本量
        """
        self.variant_id = variant_id
        self.name = name
        self.model = model
        self.strategy = strategy
        self.sample_size = sample_size
        self.metrics = {}
        self.performance_data = []
        
    def record_performance(self, **kwargs):
        """记录变体的性能数据
        
        Args:
            **kwargs: 键值对，表示指标名称和对应的值
        """
        self.performance_data.append(kwargs)
        
    def calculate_metrics(self):
        """计算变体的统计指标"""
        if not self.performance_data:
            return
        
        df = pd.DataFrame(self.performance_data)
        for metric in df.columns:
            self.metrics[metric] = {
                'mean': df[metric].mean(),
                'std': df[metric].std(),
                'count': len(df[metric]),
                'median': df[metric].median(),
                'min': df[metric].min(),
                'max': df[metric].max()
            }

class StatisticalAnalyzer:
    """统计显著性分析类"""
    
    @staticmethod
    def t_test(variant_a, variant_b, metric, alpha=0.05):
        """
        执行双样本t检验
        
        Args:
            variant_a: 第一个变体
            variant_b: 第二个变体
            metric: 要比较的指标
            alpha: 显著性水平
            
        Returns:
            dict: 包含t统计量、p值、置信区间等结果
        """
        # 从变体中提取数据
        data_a = [item[metric] for item in variant_a.performance_data if metric in item]
        data_b = [item[metric] for item in variant_b.performance_data if metric in item]
        
        # 执行t检验
        t_stat, p_value = stats.ttest_ind(data_a, data_b, equal_var=False)
        
        # 计算置信区间
        ci_a = stats.t.interval(1-alpha, len(data_a)-1, 
                               loc=np.mean(data_a), scale=stats.sem(data_a))
        ci_b = stats.t.interval(1-alpha, len(data_b)-1, 
                               loc=np.mean(data_b), scale=stats.sem(data_b))
        
        # 计算效应量（Cohen's d）
        pooled_std = np.sqrt(((len(data_a)-1)*np.var(data_a) + (len(data_b)-1)*np.var(data_b)) / 
                            (len(data_a) + len(data_b) - 2))
        cohens_d = (np.mean(data_a) - np.mean(data_b)) / pooled_std
        
        return {
            'test_type': 't_test',
            'metric': metric,
            't_statistic': t_stat,
            'p_value': p_value,
            'alpha': alpha,
            'significant': p_value < alpha,
            'cohens_d': cohens_d,
            'confidence_interval_a': ci_a,
            'confidence_interval_b': ci_b,
            'sample_size_a': len(data_a),
            'sample_size_b': len(data_b)
        }
    
    @staticmethod
    def chi_square_test(variant_a, variant_b, success_metric, total_metric):
        """
        执行卡方检验（用于比例比较）
        
        Args:
            variant_a: 第一个变体
            variant_b: 第二个变体
            success_metric: 成功指标名称
            total_metric: 总样本量指标名称
            
        Returns:
            dict: 包含卡方统计量、p值等结果
        """
        # 提取数据
        successes_a = variant_a.metrics[success_metric]['mean'] * variant_a.metrics[total_metric]['mean']
        successes_b = variant_b.metrics[success_metric]['mean'] * variant_b.metrics[total_metric]['mean']
        
        total_a = variant_a.metrics[total_metric]['mean']
        total_b = variant_b.metrics[total_metric]['mean']
        
        # 构建列联表
        contingency_table = [[successes_a, total_a - successes_a],
                           [successes_b, total_b - successes_b]]
        
        # 执行卡方检验
        chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        return {
            'test_type': 'chi_square',
            'success_metric': success_metric,
            'total_metric': total_metric,
            'chi2_statistic': chi2_stat,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'significant': p_value < 0.05,
            'contingency_table': contingency_table,
            'expected_values': expected
        }
    
    @staticmethod
    def bootstrap_test(variant_a, variant_b, metric, n_bootstrap=10000, alpha=0.05):
        """
        执行自助法检验
        
        Args:
            variant_a: 第一个变体
            variant_b: 第二个变体
            metric: 要比较的指标
            n_bootstrap: 自助采样次数
            alpha: 显著性水平
            
        Returns:
            dict: 自助法检验结果
        """
        # 从变体中提取数据
        data_a = np.array([item[metric] for item in variant_a.performance_data if metric in item])
        data_b = np.array([item[metric] for item in variant_b.performance_data if metric in item])
        
        # 计算原始均值差
        original_diff = np.mean(data_a) - np.mean(data_b)
        
        # 执行自助采样
        bootstrap_diffs = []
        for _ in range(n_bootstrap):
            sample_a = np.random.choice(data_a, size=len(data_a), replace=True)
            sample_b = np.random.choice(data_b, size=len(data_b), replace=True)
            bootstrap_diffs.append(np.mean(sample_a) - np.mean(sample_b))
        
        # 计算置信区间
        bootstrap_diffs = np.array(bootstrap_diffs)
        ci_lower = np.percentile(bootstrap_diffs, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2))
        
        # 计算p值
        p_value = 2 * min(np.mean(bootstrap_diffs <= original_diff), 
                         np.mean(bootstrap_diffs >= original_diff))
        
        return {
            'test_type': 'bootstrap',
            'metric': metric,
            'original_diff': original_diff,
            'bootstrap_diffs': bootstrap_diffs,
            'confidence_interval': (ci_lower, ci_upper),
            'p_value': p_value,
            'alpha': alpha,
            'significant': p_value < alpha,
            'n_bootstrap': n_bootstrap
        }

class ResultVisualizer:
    """测试结果可视化类"""
    
    def __init__(self, output_dir='ab_test_results'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def plot_metric_comparison(self, variants, metrics, test_id=None):
        """
        绘制不同变体的指标对比图
        
        Args:
            variants: 变体列表
            metrics: 要比较的指标列表
            test_id: 测试ID，用于命名输出文件
        """
        for metric in metrics:
            plt.figure(figsize=(12, 6))
            
            # 准备数据
            variant_names = []
            means = []
            stds = []
            counts = []
            
            for variant in variants:
                if metric in variant.metrics:
                    variant_names.append(variant.name)
                    means.append(variant.metrics[metric]['mean'])
                    stds.append(variant.metrics[metric]['std'])
                    counts.append(variant.metrics[metric]['count'])
            
            # 绘制柱状图
            x_pos = np.arange(len(variant_names))
            plt.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
            
            # 添加数值标签
            for i, (mean, count) in enumerate(zip(means, counts)):
                plt.text(i, mean + stds[i] * 0.1, f'{mean:.2f}\n(n={count})', 
                        ha='center', va='bottom')
            
            plt.title(f'{metric} Comparison Across Variants', fontsize=14)
            plt.xlabel('Variant', fontsize=12)
            plt.ylabel(metric, fontsize=12)
            plt.xticks(x_pos, variant_names, rotation=45)
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            
            # 保存图像
            if test_id:
                filename = f'{test_id}_{metric}_comparison.png'
            else:
                filename = f'{metric}_comparison.png'
            plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_confidence_intervals(self, variants, metrics, test_id=None):
        """
        绘制不同变体指标的置信区间图
        
        Args:
            variants: 变体列表
            metrics: 要比较的指标列表
            test_id: 测试ID，用于命名输出文件
        """
        for metric in metrics:
            plt.figure(figsize=(12, 6))
            
            variant_names = []
            means = []
            cis = []
            
            for variant in variants:
                if metric in variant.metrics:
                    variant_names.append(variant.name)
                    means.append(variant.metrics[metric]['mean'])
                    std = variant.metrics[metric]['std']
                    count = variant.metrics[metric]['count']
                    # 计算95%置信区间
                    ci = 1.96 * (std / np.sqrt(count))
                    cis.append(ci)
            
            x_pos = np.arange(len(variant_names))
            plt.errorbar(x_pos, means, yerr=cis, fmt='o', capsize=5, 
                        markersize=8, capthick=2, linestyle='none')
            
            plt.title(f'{metric} with 95% Confidence Intervals', fontsize=14)
            plt.xlabel('Variant', fontsize=12)
            plt.ylabel(metric, fontsize=12)
            plt.xticks(x_pos, variant_names, rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # 保存图像
            if test_id:
                filename = f'{test_id}_{metric}_confidence_intervals.png'
            else:
                filename = f'{metric}_confidence_intervals.png'
            plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_significance_matrix(self, variants, metrics, test_results, test_id=None):
        """
        绘制显著性矩阵图
        
        Args:
            variants: 变体列表
            metrics: 要比较的指标列表
            test_results: 测试结果字典
            test_id: 测试ID，用于命名输出文件
        """
        for metric in metrics:
            # 创建显著性矩阵
            n_variants = len(variants)
            sig_matrix = np.zeros((n_variants, n_variants))
            
            for i in range(n_variants):
                for j in range(n_variants):
                    if i != j:
                        key = f"{variants[i].variant_id}_vs_{variants[j].variant_id}_{metric}"
                        if key in test_results:
                            sig_matrix[i, j] = 1 if test_results[key]['significant'] else 0
                    else:
                        sig_matrix[i, j] = np.nan
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(sig_matrix, annot=True, cmap='RdYlGn', 
                       xticklabels=[v.name for v in variants],
                       yticklabels=[v.name for v in variants],
                       mask=np.isnan(sig_matrix),
                       cbar_kws={'label': 'Significant Difference (p < 0.05)'})
            
            plt.title(f'Significance Matrix for {metric}', fontsize=14)
            plt.tight_layout()
            
            # 保存图像
            if test_id:
                filename = f'{test_id}_{metric}_significance_matrix.png'
            else:
                filename = f'{metric}_significance_matrix.png'
            plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
            plt.close()
    
    def generate_report(self, test_id, variants, metrics, test_results, start_time, end_time):
        """
        生成A/B测试报告
        
        Args:
            test_id: 测试ID
            variants: 变体列表
            metrics: 指标列表
            test_results: 测试结果字典
            start_time: 测试开始时间
            end_time: 测试结束时间
        """
        # 生成HTML报告
        report_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>A/B Test Report - {test_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #3498db; }}
                .test-info {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .variant {{ background-color: #e8f4f8; padding: 15px; margin-bottom: 15px; border-radius: 5px; }}
                .metric {{ margin-left: 20px; margin-bottom: 10px; }}
                .significant {{ color: #27ae60; font-weight: bold; }}
                .not-significant {{ color: #e74c3c; }}
                img {{ max-width: 100%; height: auto; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <h1>A/B Test Report</h1>
            <div class="test-info">
                <p><strong>Test ID:</strong> {test_id}</p>
                <p><strong>Start Time:</strong> {start_time}</p>
                <p><strong>End Time:</strong> {end_time}</p>
                <p><strong>Number of Variants:</strong> {len(variants)}</p>
                <p><strong>Metrics:</strong> {', '.join(metrics)}</p>
            </div>
            
            <h2>Variants Performance</h2>
        """
        
        # 添加变体性能数据
        for variant in variants:
            report_content += f"""
            <div class="variant">
                <h3>{variant.name} (ID: {variant.variant_id})</h3>
                <p><strong>Sample Size:</strong> {variant.sample_size}</p>
                <p><strong>Actual Data Points:</strong> {len(variant.performance_data)}</p>
            """
            
            for metric, stats in variant.metrics.items():
                report_content += f"""
                <div class="metric">
                    <strong>{metric}:</strong> {stats['mean']:.2f} ± {stats['std']:.2f} (n={stats['count']})
                    <br>Min: {stats['min']:.2f}, Median: {stats['median']:.2f}, Max: {stats['max']:.2f}
                </div>
                """
            
            report_content += "</div>"
        
        # 添加显著性测试结果
        report_content += "<h2>Statistical Significance Results</h2>"
        
        for metric in metrics:
            report_content += f"<h3>{metric}</h3>"
            
            for i in range(len(variants)):
                for j in range(i+1, len(variants)):
                    key = f"{variants[i].variant_id}_vs_{variants[j].variant_id}_{metric}"
                    if key in test_results:
                        result = test_results[key]
                        sig = "Significant" if result['significant'] else "Not Significant"
                        sig_class = "significant" if result['significant'] else "not-significant"
                        
                        report_content += f"""
                        <p>
                            <strong>{variants[i].name} vs {variants[j].name}:</strong> 
                            {sig} (p={result['p_value']:.4f}, {result['test_type']})
                        </p>
                        """
        
        # 添加图表
        report_content += "<h2>Visualizations</h2>"
        for metric in metrics:
            report_content += f"""
            <h3>{metric} Visualizations</h3>
            <img src="{test_id}_{metric}_comparison.png" alt="{metric} Comparison">
            <img src="{test_id}_{metric}_confidence_intervals.png" alt="{metric} Confidence Intervals">
            <img src="{test_id}_{metric}_significance_matrix.png" alt="{metric} Significance Matrix">
            """
        
        report_content += "</body></html>"
        
        # 保存报告
        report_path = os.path.join(self.output_dir, f"{test_id}_report.html")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        return report_path

class ABTestManager:
    """A/B测试管理器，负责整个测试流程"""
    
    def __init__(self, test_id=None):
        """
        初始化A/B测试管理器
        
        Args:
            test_id: 测试唯一标识符，若为None则自动生成
        """
        self.test_id = test_id or f"ab_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.variants = {}
        self.metrics = []
        self.start_time = None
        self.end_time = None
        self.test_results = {}
        self.visualizer = ResultVisualizer()
        self.analyzer = StatisticalAnalyzer()
    
    def add_variant(self, variant):
        """添加测试变体"""
        self.variants[variant.variant_id] = variant
    
    def set_metrics(self, metrics):
        """设置要跟踪的指标"""
        self.metrics = metrics
    
    def start_test(self):
        """开始测试"""
        self.start_time = datetime.now()
        print(f"A/B测试 {self.test_id} 已开始")
    
    def end_test(self):
        """结束测试"""
        self.end_time = datetime.now()
        print(f"A/B测试 {self.test_id} 已结束")
        # 计算所有变体的指标
        for variant in self.variants.values():
            variant.calculate_metrics()
    
    def assign_variant(self, user_id):
        """
        根据用户ID分配变体（使用哈希分配算法）
        
        Args:
            user_id: 用户唯一标识符
            
        Returns:
            TestVariant: 分配给用户的变体
        """
        if not self.variants:
            raise ValueError("没有可用的测试变体")
        
        # 使用哈希函数将用户ID映射到变体
        hash_val = hash(str(user_id))
        variant_index = hash_val % len(self.variants)
        variant_id = list(self.variants.keys())[variant_index]
        return self.variants[variant_id]
    
    def run_statistical_tests(self, alpha=0.05):
        """
        运行统计显著性测试
        
        Args:
            alpha: 显著性水平
        """
        variant_list = list(self.variants.values())
        
        # 对每对变体进行比较
        for i in range(len(variant_list)):
            for j in range(i+1, len(variant_list)):
                variant_a = variant_list[i]
                variant_b = variant_list[j]
                
                for metric in self.metrics:
                    # 执行t检验
                    result = self.analyzer.t_test(variant_a, variant_b, metric, alpha)
                    key = f"{variant_a.variant_id}_vs_{variant_b.variant_id}_{metric}"
                    self.test_results[key] = result
    
    def generate_visualizations(self):
        """生成测试结果可视化"""
        variant_list = list(self.variants.values())
        
        # 绘制指标对比图
        self.visualizer.plot_metric_comparison(variant_list, self.metrics, self.test_id)
        
        # 绘制置信区间图
        self.visualizer.plot_confidence_intervals(variant_list, self.metrics, self.test_id)
        
        # 绘制显著性矩阵
        self.visualizer.plot_significance_matrix(variant_list, self.metrics, 
                                               self.test_results, self.test_id)
    
    def generate_report(self):
        """
        生成完整的测试报告
        
        Returns:
            str: 报告文件路径
        """
        # 首先生成可视化
        self.generate_visualizations()
        
        # 生成报告
        return self.visualizer.generate_report(
            self.test_id,
            list(self.variants.values()),
            self.metrics,
            self.test_results,
            self.start_time,
            self.end_time
        )
    
    def get_test_summary(self):
        """
        获取测试摘要
        
        Returns:
            dict: 测试摘要信息
        """
        return {
            'test_id': self.test_id,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': str(self.end_time - self.start_time) if self.end_time else None,
            'num_variants': len(self.variants),
            'metrics': self.metrics,
            'variants': {v.variant_id: v.name for v in self.variants.values()},
            'num_significant_results': sum(1 for r in self.test_results.values() if r['significant']),
            'total_comparisons': len(self.test_results)
        }
