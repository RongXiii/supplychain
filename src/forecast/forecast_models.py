import numpy as np
import pandas as pd
# 设置matplotlib使用非交互式后端，避免弹出窗口
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
import os
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from xgboost import XGBRegressor
from scipy import stats
from prophet import Prophet
import optuna
from sklearn.base import BaseEstimator, RegressorMixin

# 间歇性需求模型 - Croston方法
class Croston:
    """Croston方法用于处理间歇性需求，优化版支持自适应平滑参数和趋势检测"""
    
    def __init__(self, alpha=None, beta=None, trend=True, seasonal=False, seasonal_periods=12):
        """
        初始化Croston模型
        
        Args:
            alpha: 需求水平平滑参数，默认为None（自适应）
            beta: 趋势平滑参数，默认为None（自适应）
            trend: 是否考虑趋势
            seasonal: 是否考虑季节性
            seasonal_periods: 季节性周期长度
        """
        self.alpha = alpha
        self.beta = beta
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        
        self.demand_level = None
        self.demand_interval = None
        self.demand_trend = None
        self.seasonal_factors = None
        self.last_demand = None
        self.last_interval = None
        self.last_period = None
        self.is_fitted = False
    
    def _detect_trend(self, demand_values, demand_times):
        """检测需求趋势"""
        if len(demand_values) < 3:
            return 0.0
        
        # 使用线性回归检测趋势
        X = np.array(demand_times).reshape(-1, 1)
        y = demand_values
        reg = LinearRegression().fit(X, y)
        return reg.coef_[0]
    
    def _detect_seasonality(self, y):
        """检测季节性"""
        from statsmodels.tsa.stattools import adfuller
        
        if len(y) < 2 * self.seasonal_periods:
            return False
        
        # 使用ADF检验检测季节性
        result = adfuller(y)
        return result[1] > 0.05  # 如果不平稳，可能存在季节性
    
    def _calculate_seasonal_factors(self, y):
        """计算季节性因子"""
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        # 使用移动平均分解季节性
        decomposition = seasonal_decompose(y, model='additive', period=self.seasonal_periods, extrapolate_trend='freq')
        return decomposition.seasonal
    
    def fit(self, y):
        """训练模型，支持自适应平滑参数和趋势检测"""
        # 初始化
        demand_mask = y > 0
        demand_indices = np.where(demand_mask)[0]
        
        if len(demand_indices) < 2:
            # 数据点不足，使用简单平均值
            self.demand_level = np.mean(y)
            self.demand_interval = 1
            self.demand_trend = 0.0
            self.is_fitted = True
            return self
        
        # 提取需求和间隔
        demand_values = y[demand_mask]
        intervals = np.diff(demand_indices)
        
        # 自适应计算alpha（基于需求波动性）
        if self.alpha is None:
            demand_std = np.std(demand_values) if len(demand_values) > 1 else 0
            demand_mean = np.mean(demand_values)
            volatility = demand_std / (demand_mean + 1e-8)
            self.alpha = max(0.05, min(0.3, 0.1 + volatility * 0.2))
        
        # 自适应计算beta（基于趋势强度）
        if self.beta is None:
            trend_strength = abs(self._detect_trend(demand_values, demand_indices))
            self.beta = max(0.01, min(0.2, trend_strength * 0.1))
        
        # 检测季节性
        has_seasonality = self._detect_seasonality(y) if self.seasonal else False
        
        if has_seasonality:
            self.seasonal_factors = self._calculate_seasonal_factors(y)
        
        # 初始化平滑值
        self.demand_level = demand_values[0]
        self.demand_interval = intervals[0]
        self.demand_trend = self._detect_trend(demand_values[:min(3, len(demand_values))], 
                                             demand_indices[:min(3, len(demand_values))])
        
        # 平滑处理
        for i in range(1, len(demand_values)):
            # 更新需求水平和趋势
            old_level = self.demand_level
            self.demand_level = self.alpha * demand_values[i] + (1 - self.alpha) * (old_level + self.demand_trend)
            
            if self.trend:
                self.demand_trend = self.beta * (self.demand_level - old_level) + (1 - self.beta) * self.demand_trend
            
            # 更新间隔
            if i < len(intervals):
                self.demand_interval = self.alpha * intervals[i] + (1 - self.alpha) * self.demand_interval
        
        self.last_demand = demand_values[-1]
        self.last_interval = intervals[-1] if len(intervals) > 0 else 1
        self.last_period = demand_indices[-1]
        self.is_fitted = True
        return self
    
    def forecast(self, steps=1):
        """预测未来值，支持趋势和季节性"""
        if not self.is_fitted:
            raise ValueError("模型未训练")
        
        forecasts = []
        
        for step in range(steps):
            # 基础预测值：需求水平 / 平均间隔
            base_forecast = self.demand_level / (self.demand_interval + 1e-8)
            
            # 添加趋势影响
            if self.trend:
                base_forecast += self.demand_trend * step
            
            # 添加季节性影响
            if self.seasonal and self.seasonal_factors is not None:
                period = (self.last_period + step + 1) % self.seasonal_periods
                seasonal_factor = self.seasonal_factors[period] if period < len(self.seasonal_factors) else 0
                base_forecast += seasonal_factor
            
            # 确保预测值非负
            forecast_value = max(0, base_forecast)
            forecasts.append(forecast_value)
        
        return np.array(forecasts)

# 间歇性需求模型 - SBA方法
class SBA:
    """SBA (Syntetos-Boylan Approximation) 方法用于处理间歇性需求，优化版支持自适应平滑参数和趋势检测"""
    
    def __init__(self, alpha=None, beta=None, trend=True, seasonal=False, seasonal_periods=12, method='SBA'):
        """
        初始化SBA模型
        
        Args:
            alpha: 需求水平平滑参数，默认为None（自适应）
            beta: 趋势平滑参数，默认为None（自适应）
            trend: 是否考虑趋势
            seasonal: 是否考虑季节性
            seasonal_periods: 季节性周期长度
            method: SBA变体，可选'SBA'或'SBA-M'
        """
        self.alpha = alpha
        self.beta = beta
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.method = method
        
        self.demand_level = None
        self.demand_interval = None
        self.demand_trend = None
        self.seasonal_factors = None
        self.last_demand = None
        self.last_interval = None
        self.last_period = None
        self.is_fitted = False
    
    def _detect_trend(self, demand_values, demand_times):
        """检测需求趋势"""
        if len(demand_values) < 3:
            return 0.0
        
        # 使用线性回归检测趋势
        X = np.array(demand_times).reshape(-1, 1)
        y = demand_values
        reg = LinearRegression().fit(X, y)
        return reg.coef_[0]
    
    def _detect_seasonality(self, y):
        """检测季节性"""
        from statsmodels.tsa.stattools import adfuller
        
        if len(y) < 2 * self.seasonal_periods:
            return False
        
        # 使用ADF检验检测季节性
        result = adfuller(y)
        return result[1] > 0.05  # 如果不平稳，可能存在季节性
    
    def _calculate_seasonal_factors(self, y):
        """计算季节性因子"""
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        # 使用移动平均分解季节性
        decomposition = seasonal_decompose(y, model='additive', period=self.seasonal_periods, extrapolate_trend='freq')
        return decomposition.seasonal
    
    def fit(self, y):
        """训练模型，支持自适应平滑参数和趋势检测"""
        # 初始化
        demand_mask = y > 0
        demand_indices = np.where(demand_mask)[0]
        
        if len(demand_indices) < 2:
            # 数据点不足，使用简单平均值
            self.demand_level = np.mean(y)
            self.demand_interval = 1
            self.demand_trend = 0.0
            self.is_fitted = True
            return self
        
        # 提取需求和间隔
        demand_values = y[demand_mask]
        intervals = np.diff(demand_indices)
        
        # 自适应计算alpha（基于需求波动性）
        if self.alpha is None:
            demand_std = np.std(demand_values) if len(demand_values) > 1 else 0
            demand_mean = np.mean(demand_values)
            volatility = demand_std / (demand_mean + 1e-8)
            self.alpha = max(0.05, min(0.3, 0.1 + volatility * 0.2))
        
        # 自适应计算beta（基于趋势强度）
        if self.beta is None:
            trend_strength = abs(self._detect_trend(demand_values, demand_indices))
            self.beta = max(0.01, min(0.2, trend_strength * 0.1))
        
        # 检测季节性
        has_seasonality = self._detect_seasonality(y) if self.seasonal else False
        
        if has_seasonality:
            self.seasonal_factors = self._calculate_seasonal_factors(y)
        
        # 初始化平滑值
        self.demand_level = demand_values[0]
        self.demand_interval = intervals[0]
        self.demand_trend = self._detect_trend(demand_values[:min(3, len(demand_values))], 
                                             demand_indices[:min(3, len(demand_values))])
        
        # 平滑处理
        for i in range(1, len(demand_values)):
            # 更新需求水平和趋势
            old_level = self.demand_level
            self.demand_level = self.alpha * demand_values[i] + (1 - self.alpha) * (old_level + self.demand_trend)
            
            if self.trend:
                self.demand_trend = self.beta * (self.demand_level - old_level) + (1 - self.beta) * self.demand_trend
            
            # 更新间隔
            if i < len(intervals):
                self.demand_interval = self.alpha * intervals[i] + (1 - self.alpha) * self.demand_interval
        
        self.last_demand = demand_values[-1]
        self.last_interval = intervals[-1] if len(intervals) > 0 else 1
        self.last_period = demand_indices[-1]
        self.is_fitted = True
        return self
    
    def forecast(self, steps=1):
        """预测未来值，支持趋势和季节性"""
        if not self.is_fitted:
            raise ValueError("模型未训练")
        
        forecasts = []
        
        for step in range(steps):
            # 计算SBA修正因子
            a = self.alpha
            if self.method == 'SBA-M':
                # SBA-M变体，使用不同的gamma计算
                gamma = (1 - a) / 2 if a < 1 else 0.5
            else:
                # 标准SBA
                gamma = (2 - a) / (2 * (1 - a)) if a < 1 else 1
            
            # 基础预测值：需求水平 / (平均间隔 - gamma)
            base_forecast = self.demand_level / (max(self.demand_interval - gamma, 0.5) + 1e-8)
            
            # 添加趋势影响
            if self.trend:
                base_forecast += self.demand_trend * step
            
            # 添加季节性影响
            if self.seasonal and self.seasonal_factors is not None:
                period = (self.last_period + step + 1) % self.seasonal_periods
                seasonal_factor = self.seasonal_factors[period] if period < len(self.seasonal_factors) else 0
                base_forecast += seasonal_factor
            
            # 确保预测值非负
            forecast_value = max(0, base_forecast)
            forecasts.append(forecast_value)
        
        return np.array(forecasts)

# 模型融合类
class ModelEnsemble(BaseEstimator, RegressorMixin):
    """模型融合器，结合多种模型的优势，支持动态权重调整和实时更新"""
    
    def __init__(self, models=None, weights=None, dynamic_weights=False, weight_decay=0.9, 
                 performance_metric='rmse', smoothing_factor=0.1, diversity_weight=0.1):
        """
        初始化模型融合器
        
        参数：
        - models: 基础模型列表
        - weights: 初始权重列表
        - dynamic_weights: 是否启用动态权重调整
        - weight_decay: 权重衰减因子，控制历史表现的影响
        - performance_metric: 性能评估指标 ('rmse', 'mae', 'mape')
        - smoothing_factor: 权重更新平滑因子，避免权重突变
        - diversity_weight: 多样性权重，鼓励模型多样性
        """
        self.models = models or []
        self.weights = weights or [1/len(models) for _ in models] if models else []
        self.dynamic_weights = dynamic_weights
        self.weight_decay = weight_decay
        self.performance_metric = performance_metric
        self.smoothing_factor = smoothing_factor
        self.diversity_weight = diversity_weight
        self.is_fitted = False
        self.performance_history = []  # 记录模型历史表现
        self.update_count = 0
    
    def fit(self, X, y):
        """训练所有基础模型"""
        for model in self.models:
            model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """使用融合模型进行预测"""
        if not self.is_fitted:
            raise ValueError("模型未训练")
        
        # 获取各模型预测结果
        predictions = np.array([model.predict(X) for model in self.models]).T
        
        if self.dynamic_weights:
            # 使用当前动态权重
            weights = np.array(self.weights)
        else:
            weights = np.array(self.weights)
        
        return np.sum(predictions * weights, axis=1)
    
    def update_weights(self, new_weights):
        """更新模型权重"""
        # 平滑过渡到新权重
        if self.update_count > 0:
            new_weights = np.array(new_weights)
            current_weights = np.array(self.weights)
            self.weights = list(current_weights * (1 - self.smoothing_factor) + new_weights * self.smoothing_factor)
        else:
            self.weights = new_weights
        self.update_count += 1
        
    def _calculate_performance(self, y_true, y_pred):
        """计算模型性能指标"""
        if self.performance_metric == 'rmse':
            return np.sqrt(mean_squared_error(y_true, y_pred))
        elif self.performance_metric == 'mae':
            return mean_absolute_error(y_true, y_pred)
        elif self.performance_metric == 'mape':
            return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100
        else:
            raise ValueError(f"不支持的性能指标: {self.performance_metric}")
    
    def _calculate_diversity(self, predictions):
        """计算模型多样性"""
        if len(predictions) < 2:
            return 0.0
        
        # 使用相关性作为多样性度量（相关性越低，多样性越高）
        correlations = []
        for i in range(len(predictions)):
            for j in range(i+1, len(predictions)):
                corr = np.corrcoef(predictions[i], predictions[j])[0, 1]
                correlations.append(1 - corr)  # 转换为多样性分数
        
        return np.mean(correlations) if correlations else 0.0
    
    def update_dynamic_weights(self, X_val, y_val):
        """根据验证集表现动态调整权重，考虑性能和多样性"""
        if not self.is_fitted:
            raise ValueError("模型未训练")
        
        # 获取各模型预测
        predictions = []
        performances = []
        
        for model in self.models:
            y_pred = model.predict(X_val)
            predictions.append(y_pred)
            
            # 计算性能分数（使用倒数，越高越好）
            perf = self._calculate_performance(y_val, y_pred)
            perf_score = 1 / (perf + 1e-6)
            performances.append(perf_score)
        
        # 计算多样性分数
        diversity_score = self._calculate_diversity(predictions)
        
        # 调整性能分数，加入多样性考虑
        adjusted_performances = []
        for perf in performances:
            adjusted_perf = perf * (1 - self.diversity_weight) + diversity_score * self.diversity_weight
            adjusted_performances.append(adjusted_perf)
        
        # 归一化权重
        total = sum(adjusted_performances)
        new_weights = [p / total for p in adjusted_performances]
        
        # 更新权重历史
        self.performance_history.append(adjusted_performances)
        
        # 应用权重衰减，更重视近期表现
        if len(self.performance_history) > 1:
            weighted_history = []
            for i, history in enumerate(reversed(self.performance_history)):
                weight = self.weight_decay ** i
                weighted_history.append([h * weight for h in history])
            
            # 聚合历史表现
            aggregated_performance = [sum(h) for h in zip(*weighted_history)]
            total_aggregated = sum(aggregated_performance)
            new_weights = [p / total_aggregated for p in aggregated_performance]
        
        # 更新权重，应用平滑过渡
        self.update_weights(new_weights)
        
        return self.weights
    
    def update_model(self, model_index, new_model, retrain=False, X=None, y=None):
        """更新单个模型并重新训练（如果需要）"""
        if model_index < 0 or model_index >= len(self.models):
            raise ValueError(f"模型索引超出范围: {model_index}")
        
        self.models[model_index] = new_model
        if retrain and X is not None and y is not None:
            self.models[model_index].fit(X, y)
        
        # 更新权重以适应新模型
        if self.dynamic_weights and X is not None and y is not None:
            self.update_dynamic_weights(X, y)
    
    def add_model(self, model, fit_data=None, weight=None):
        """添加新模型并训练"""
        self.models.append(model)
        
        if fit_data is not None:
            X, y = fit_data
            model.fit(X, y)
        
        # 初始化权重
        if weight is not None:
            self.weights.append(weight)
        else:
            # 平均分配权重
            total = sum(self.weights) + 1
            self.weights = [w / total for w in self.weights] + [1 / total]
        
        # 重新归一化权重
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]

# 超参数优化器
class HyperparameterOptimizer:
    """超参数自动优化器，支持自适应优化策略"""
    
    def __init__(self, model_name, n_trials=10, use_gpu=False, optimization_strategy='adaptive'):
        self.model_name = model_name
        self.n_trials = n_trials
        self.use_gpu = use_gpu
        self.optimization_strategy = optimization_strategy
        self.best_params_history = {}  # 保存历史最优参数，用于自适应优化
    
    def _get_adaptive_search_space(self):
        """根据历史最优参数和模型类型获取自适应搜索空间"""
        # 默认搜索空间
        default_space = {
            'xgb': {
                'n_estimators': (50, 500),
                'max_depth': (3, 10),
                'learning_rate': (0.01, 0.3),
                'subsample': (0.7, 1.0),
                'colsample_bytree': (0.7, 1.0),
                'gamma': (0, 1),
                'reg_alpha': (0, 1),
                'reg_lambda': (1, 3)
            },
            'random_forest': {
                'n_estimators': (50, 500),
                'max_depth': (3, 15),
                'min_samples_split': (2, 20),
                'min_samples_leaf': (1, 10),
                'max_features': (0.5, 1.0)
            },
            'gradient_boosting': {
                'n_estimators': (50, 500),
                'max_depth': (3, 10),
                'learning_rate': (0.01, 0.3),
                'subsample': (0.7, 1.0),
                'min_samples_split': (2, 20),
                'min_samples_leaf': (1, 10)
            },
            'svr': {
                'C': (0.1, 100),
                'epsilon': (0.01, 1.0),
                'gamma': ('scale', 'auto')
            }
        }
        
        # 如果没有历史最优参数，返回默认搜索空间
        if self.model_name not in self.best_params_history:
            return default_space.get(self.model_name, default_space['xgb'])
        
        # 基于历史最优参数调整搜索空间（缩小范围，提高搜索效率）
        best_params = self.best_params_history[self.model_name]
        adaptive_space = {}
        
        for param, (min_val, max_val) in default_space[self.model_name].items():
            if param in best_params:
                # 缩小搜索空间到最优值的±20%范围内
                best_val = best_params[param]
                if isinstance(best_val, (int, float)):
                    range_val = (max_val - min_val) * 0.2
                    new_min = max(min_val, best_val - range_val)
                    new_max = min(max_val, best_val + range_val)
                    adaptive_space[param] = (new_min, new_max)
                else:
                    # 对于分类参数，保持不变
                    adaptive_space[param] = default_space[self.model_name][param]
            else:
                adaptive_space[param] = default_space[self.model_name][param]
        
        return adaptive_space
    
    def optimize(self, X_train, y_train, X_val=None, y_val=None):
        """优化模型超参数，支持自适应优化和早停策略"""
        # 自适应调整搜索空间
        search_space = self._get_adaptive_search_space()
        
        def objective(trial):
            if self.model_name == 'xgb':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', *search_space['n_estimators']),
                    'max_depth': trial.suggest_int('max_depth', *search_space['max_depth']),
                    'learning_rate': trial.suggest_float('learning_rate', *search_space['learning_rate']),
                    'subsample': trial.suggest_float('subsample', *search_space['subsample']),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', *search_space['colsample_bytree']),
                    'gamma': trial.suggest_float('gamma', *search_space['gamma']),
                    'reg_alpha': trial.suggest_float('reg_alpha', *search_space['reg_alpha']),
                    'reg_lambda': trial.suggest_float('reg_lambda', *search_space['reg_lambda']),
                    'tree_method': 'gpu_hist' if self.use_gpu else 'auto',
                    'device': 'cuda' if self.use_gpu else 'cpu'
                }
                model = XGBRegressor(**params, random_state=42)
            elif self.model_name == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', *search_space['n_estimators']),
                    'max_depth': trial.suggest_int('max_depth', *search_space['max_depth']),
                    'min_samples_split': trial.suggest_int('min_samples_split', *search_space['min_samples_split']),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', *search_space['min_samples_leaf']),
                    'max_features': trial.suggest_float('max_features', *search_space['max_features']),
                    'random_state': 42
                }
                model = RandomForestRegressor(**params)
            elif self.model_name == 'gradient_boosting':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', *search_space['n_estimators']),
                    'max_depth': trial.suggest_int('max_depth', *search_space['max_depth']),
                    'learning_rate': trial.suggest_float('learning_rate', *search_space['learning_rate']),
                    'subsample': trial.suggest_float('subsample', *search_space['subsample']),
                    'min_samples_split': trial.suggest_int('min_samples_split', *search_space['min_samples_split']),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', *search_space['min_samples_leaf']),
                    'random_state': 42
                }
                model = GradientBoostingRegressor(**params)
            elif self.model_name == 'svr':
                params = {
                    'C': trial.suggest_float('C', *search_space['C']),
                    'epsilon': trial.suggest_float('epsilon', *search_space['epsilon']),
                    'gamma': trial.suggest_categorical('gamma', search_space['gamma'])
                }
                model = SVR(**params)
            else:
                # 默认使用XGBoost
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', *search_space['n_estimators']),
                    'max_depth': trial.suggest_int('max_depth', *search_space['max_depth']),
                    'learning_rate': trial.suggest_float('learning_rate', *search_space['learning_rate']),
                    'subsample': trial.suggest_float('subsample', *search_space['subsample']),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', *search_space['colsample_bytree']),
                    'gamma': trial.suggest_float('gamma', *search_space['gamma']),
                    'tree_method': 'gpu_hist' if self.use_gpu else 'auto',
                    'device': 'cuda' if self.use_gpu else 'cpu'
                }
                model = XGBRegressor(**params, random_state=42)
            
            # 训练模型并评估
            if X_val is not None and y_val is not None:
                # 使用验证集进行早停和评估
                if self.model_name == 'xgb':
                    model.fit(X_train, y_train, 
                             eval_set=[(X_val, y_val)], 
                             early_stopping_rounds=50, 
                             verbose=False)
                    y_pred = model.predict(X_val)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            else:
                # 使用交叉验证进行评估
                from sklearn.model_selection import cross_val_score
                scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
                rmse = -np.mean(scores)
            
            return rmse
        
        # 创建Optuna研究并运行优化
        study = optuna.create_study(direction='minimize')
        
        # 添加早停回调
        def callback(study, trial):
            # 当连续5次 trial 没有改进时停止优化
            if len(study.trials) > 5:
                recent_trials = study.trials[-5:]
                if all(trial.value >= study.best_value for trial in recent_trials):
                    study.stop()
        
        study.optimize(objective, n_trials=self.n_trials, callbacks=[callback])
        
        # 保存最优参数到历史记录
        self.best_params_history[self.model_name] = study.best_params
        
        return study.best_params

class HybridModel:
    """
    混合模型：结合统计模型和机器学习模型
    先用ARIMA捕捉趋势，再用ML模型修正
    """
    
    def __init__(self, ml_model=None):
        self.arima_model = None
        self.ml_model = ml_model or XGBRegressor(n_estimators=50, random_state=42)
        self.is_fitted = False
    
    def fit(self, y_train):
        """
        训练混合模型
        
        Args:
            y_train: 训练数据
        """
        # 检查数据长度，至少需要10个数据点
        if len(y_train) < 10:
            # 数据不足时，使用简单的平均模型
            self.arima_model = None
            self.ml_model = None
            self.is_fitted = True
            return
        
        try:
            # 1. 用ARIMA模型捕捉趋势和季节性
            # 根据数据长度调整ARIMA阶数
            p = min(1, len(y_train) // 10)
            d = 1
            q = min(1, len(y_train) // 10)
            self.arima_model = ARIMA(y_train, order=(p,d,q))
            self.arima_model = self.arima_model.fit()
            
            # 2. 计算ARIMA残差
            # 使用索引位置而不是日期来避免频率问题
            arima_pred = self.arima_model.predict(start=0, end=len(y_train)-1, typ='levels')
            residuals = y_train - arima_pred
            
            # 3. 用ML模型预测残差
            # 准备特征：滞后残差和时间特征
            n_lags = min(3, len(y_train) // 5)
            X_train = []
            y_res = []
            
            for i in range(n_lags, len(residuals)):
                # 滞后残差作为特征
                lag_features = residuals[i-n_lags:i]
                # 添加时间特征（归一化的时间索引）
                time_feature = i / len(residuals)
                X_train.append(list(lag_features) + [time_feature])
                y_res.append(residuals[i])
            
            if len(X_train) > 0:
                X_train = np.array(X_train)
                y_res = np.array(y_res)
                self.ml_model.fit(X_train, y_res)
        except Exception as e:
            # ARIMA训练失败时，使用简单的平均模型
            print(f"ARIMA训练失败，使用简单模型: {str(e)}")
            self.arima_model = None
            self.ml_model = None
        
        self.is_fitted = True
    
    def forecast(self, steps=1):
        """
        预测未来值
        
        Args:
            steps: 预测步数
        
        Returns:
            predictions: 预测结果
        """
        if not self.is_fitted:
            raise ValueError("模型未训练，请先调用fit方法")
        
        # 如果ARIMA模型未训练成功，返回简单的预测
        if self.arima_model is None:
            return np.zeros(steps)
        
        try:
            # 1. ARIMA预测
            arima_forecast = self.arima_model.forecast(steps=steps)
            
            # 确保arima_forecast是1维数组
            if hasattr(arima_forecast, 'values'):
                arima_forecast = arima_forecast.values
            arima_forecast = np.asarray(arima_forecast).flatten()
            
            # 2. 用ML模型预测残差
            ml_residuals = []
            if hasattr(self.arima_model, 'model') and hasattr(self.arima_model.model, 'endog') and self.ml_model is not None:
                # 使用索引位置而不是日期来避免频率问题
                endog_len = len(self.arima_model.model.endog)
                recent_residuals = np.zeros(3)
                
                if endog_len >= 3:
                    start_idx = max(0, endog_len - 3)
                    end_idx = endog_len - 1
                    try:
                        arima_pred = self.arima_model.predict(start=start_idx, end=end_idx, typ='levels')
                        
                        # 确保endog是1维数组
                        endog = self.arima_model.model.endog[start_idx:end_idx+1]
                        endog = np.asarray(endog).flatten()
                        
                        # 确保arima_pred是1维数组
                        arima_pred = np.asarray(arima_pred).flatten()
                        
                        # 计算残差
                        recent_residuals = endog - arima_pred
                        
                        # 确保残差是1维数组
                        recent_residuals = np.asarray(recent_residuals).flatten()
                        
                        # 如果残差长度不足3，填充0
                        if len(recent_residuals) < 3:
                            recent_residuals = np.pad(recent_residuals, (3 - len(recent_residuals), 0), 'constant')
                        else:
                            recent_residuals = recent_residuals[-3:]
                    except Exception as e:
                        print(f"ARIMA预测残差失败: {str(e)}")
                        recent_residuals = np.zeros(3)
                
                for i in range(steps):
                    # 准备ML模型的特征
                    lag_features = recent_residuals[-3:]
                    time_feature = (endog_len + i) / (endog_len + steps)
                    X_pred = np.array([list(lag_features) + [time_feature]])
                    try:
                        residual_pred = self.ml_model.predict(X_pred)[0]
                        ml_residuals.append(residual_pred)
                        # 更新最近残差
                        recent_residuals = np.append(recent_residuals[1:], residual_pred)
                    except Exception as e:
                        print(f"ML模型预测残差失败: {str(e)}")
                        ml_residuals.append(0)
            else:
                # 没有ARIMA模型或ML模型，残差预测为0
                ml_residuals = [0] * steps
            
            # 3. 合并ARIMA预测和ML残差修正
            # 确保ml_residuals是numpy数组
            ml_residuals = np.array(ml_residuals)
            
            # 确保两者长度一致
            if len(arima_forecast) != len(ml_residuals):
                # 调整长度，以较短的为准
                min_len = min(len(arima_forecast), len(ml_residuals))
                arima_forecast = arima_forecast[:min_len]
                ml_residuals = ml_residuals[:min_len]
            
            final_predictions = arima_forecast + ml_residuals
            return final_predictions
        except Exception as e:
            print(f"混合模型预测失败: {str(e)}")
            # 预测失败时，返回简单的预测
            try:
                # 尝试只使用ARIMA模型预测
                arima_forecast = self.arima_model.forecast(steps=steps)
                if hasattr(arima_forecast, 'values'):
                    arima_forecast = arima_forecast.values
                arima_forecast = np.asarray(arima_forecast).flatten()
                return arima_forecast
            except:
                # 最终兜底，返回0
                return np.zeros(steps)


class HierarchicalForecaster:
    """
    分层预测器：SKU层预测 + 类别层修正
    """
    
    def __init__(self, sku_model_selector):
        self.sku_model_selector = sku_model_selector
        self.sku_models = {}
        self.category_model = None
        self.category_mapping = {}
    
    def fit(self, sku_data, category_mapping):
        """
        训练分层预测模型
        
        Args:
            sku_data: SKU级别数据，格式为{sku_id: demand_data}
            category_mapping: SKU到类别的映射，格式为{sku_id: category_id}
        """
        self.category_mapping = category_mapping
        
        # 1. 训练SKU级别的模型
        for sku_id, demand_data in sku_data.items():
            # 简化处理：使用一维需求数据训练ARIMA模型
            model = ARIMA(demand_data, order=(1,1,1))
            self.sku_models[sku_id] = model.fit()
        
        # 2. 聚合SKU数据到类别级别
        category_data = {}
        for sku_id, demand_data in sku_data.items():
            category_id = category_mapping[sku_id]
            if category_id not in category_data:
                category_data[category_id] = np.zeros_like(demand_data)
            category_data[category_id] += demand_data
        
        # 3. 训练类别级别的模型（使用简单的ARIMA）
        self.category_model = {}
        for category_id, demand_data in category_data.items():
            model = ARIMA(demand_data, order=(1,1,1))
            self.category_model[category_id] = model.fit()
    
    def forecast(self, steps=1):
        """
        进行分层预测
        
        Args:
            steps: 预测步数
        
        Returns:
            forecasts: 预测结果，格式为{sku_id: predictions}
        """
        # 1. SKU级别预测
        sku_forecasts = {}
        for sku_id, model in self.sku_models.items():
            sku_forecasts[sku_id] = model.forecast(steps=steps)
        
        # 2. 类别级别预测
        category_forecasts = {}
        for category_id, model in self.category_model.items():
            category_forecasts[category_id] = model.forecast(steps=steps)
        
        # 3. 聚合SKU预测到类别级别
        category_sku_agg = {}
        for category_id in self.category_model.keys():
            category_sku_agg[category_id] = np.zeros(steps)
        
        for sku_id, forecast in sku_forecasts.items():
            category_id = self.category_mapping[sku_id]
            if hasattr(forecast, 'values'):
                forecast_values = forecast.values
            else:
                forecast_values = np.array(forecast)
            category_sku_agg[category_id] += forecast_values
        
        # 4. 计算修正因子并调整SKU预测
        final_forecasts = {}
        for sku_id, forecast in sku_forecasts.items():
            category_id = self.category_mapping[sku_id]
            
            if hasattr(forecast, 'values'):
                sku_forecast = forecast.values
            else:
                sku_forecast = np.array(forecast)
            
            if hasattr(category_forecasts[category_id], 'values'):
                category_forecast = category_forecasts[category_id].values
            else:
                category_forecast = np.array(category_forecasts[category_id])
            
            # 计算修正因子
            if np.all(category_sku_agg[category_id] == 0):
                correction_factor = 1.0
            else:
                correction_factor = category_forecast / category_sku_agg[category_id]
            
            # 应用修正因子
            final_forecasts[sku_id] = sku_forecast * correction_factor
        
        return final_forecasts


class ForecastModelSelector:
    """
    预测模型选择器，用于根据不同产品的特点选择合适的预测模型
    根据产品ABC分类和需求模式选择最优模型
    """
    
    def __init__(self, use_gpu=False):
        import logging
        # 配置日志
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # 检测GPU可用性
        self.use_gpu = use_gpu
        try:
            import torch
            self.has_cuda = torch.cuda.is_available()
        except ImportError:
            self.has_cuda = False
        
        # 定义所有可用的预测模型
        self.all_models = {
            # 统计模型
            'arima': ARIMA,
            'holt_winters': ExponentialSmoothing,
            'prophet': Prophet,
            
            # 间歇性需求模型
            'croston': Croston,
            'sba': SBA,
            
            # 机器学习模型
            'linear_regression': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'xgb': XGBRegressor(n_estimators=100, random_state=42, 
                              tree_method='gpu_hist' if self.use_gpu else 'auto',
                              device='cuda' if self.use_gpu else 'cpu'),
            'svr': SVR(kernel='rbf'),
            # 'lstm': LSTMModel()  # 需启用tensorflow/keras
            
            # 混合模型
            'hybrid_arima_xgb': HybridModel(ml_model=XGBRegressor(n_estimators=50, random_state=42, 
                                                           tree_method='gpu_hist' if self.use_gpu else 'auto',
                                                           device='cuda' if self.use_gpu else 'cpu')),
            # 模型融合
            'ensemble': ModelEnsemble
        }
        
        # 模型保存路径
        self.model_dir = 'models'
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        
        # 初始化解释器
        try:
            from src.interpretability import ModelInterpreter, BusinessRuleGenerator
            self.interpreter = ModelInterpreter()
            self.rule_generator = BusinessRuleGenerator()
        except ImportError:
            self.interpreter = None
            self.rule_generator = None
        
        # 初始化模型选择记录
        self.model_selections = {}
        
        # 确保解释目录存在
        os.makedirs('interpretations', exist_ok=True)
        os.makedirs('interpretations/figures', exist_ok=True)
        
        # 模型评估配置
        self.evaluation_metrics = {
            'rmse': self._rmse,
            'mae': self._mae,
            'mape': self._mape,
            'smape': self._smape,
            'mase': self._mase
        }
        
        # 模型选择权重配置
        self.model_selection_weights = {
            'rmse': 0.3,
            'mae': 0.2,
            'mape': 0.2,
            'smape': 0.15,
            'mase': 0.15
        }
        
        # 时间序列交叉验证配置
        self.ts_cv_config = {
            'n_splits': 5,
            'test_size': 0.2,
            'min_train_size': 0.3
        }
        
        # 模型缓存，用于提高重复评估的效率
        self.model_cache = {}
        
        # 导入额外的评估指标
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        self.mse = mean_squared_error
        self.mae = mean_absolute_error
    
    def _rmse(self, y_true, y_pred):
        """
        计算均方根误差
        """
        return np.sqrt(self.mse(y_true, y_pred))
    
    def _mae(self, y_true, y_pred):
        """
        计算平均绝对误差
        """
        return self.mae(y_true, y_pred)
    
    def _mape(self, y_true, y_pred):
        """
        计算平均绝对百分比误差
        """
        non_zero_mask = y_true != 0
        if np.any(non_zero_mask):
            return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
        else:
            return 0.0
    
    def _smape(self, y_true, y_pred):
        """
        计算对称平均绝对百分比误差
        """
        denominator = (np.abs(y_true) + np.abs(y_pred))
        non_zero_denominator = denominator != 0
        if np.any(non_zero_denominator):
            return 2 * np.mean(np.abs(y_true[non_zero_denominator] - y_pred[non_zero_denominator]) / denominator[non_zero_denominator]) * 100
        else:
            return 0.0
    
    def _mase(self, y_true, y_pred):
        """
        计算平均绝对标度误差
        """
        # 使用朴素预测（前一天的值）作为基准
        if len(y_true) < 2:
            return 0.0
        
        naive_pred = y_true[:-1]
        naive_true = y_true[1:]
        naive_mae = self.mae(naive_true, naive_pred)
        
        if naive_mae == 0:
            return 0.0
        
        return self.mae(y_true, y_pred) / naive_mae
    
    def classify_abc(self, product_value, total_value):
        """
        ABC分类：根据产品价值占比进行分类
        
        Args:
            product_value: 产品价值
            total_value: 总价值
            
        Returns:
            abc_class: ABC分类结果（A/B/C）
        """
        value_ratio = product_value / total_value
        if value_ratio >= 0.7:
            return 'A'
        elif value_ratio >= 0.2:
            return 'B'
        else:
            return 'C'
    
    def identify_demand_pattern(self, demand_data):
        """
        识别需求模式
        
        Args:
            demand_data: 需求数据序列
            
        Returns:
            pattern: 需求模式（Stable/Seasonal/Promotional/Intermittent）
        """
        # 计算需求特征
        cv = np.std(demand_data) / np.mean(demand_data)  # 变异系数
        
        # 检查是否为零散需求（间歇性需求）
        zero_ratio = np.sum(demand_data == 0) / len(demand_data)
        if zero_ratio > 0.3:
            return 'Intermittent'
        
        # 检查是否有明显的促销特征（需求突变）
        demand_diff = np.diff(demand_data)
        promo_ratio = np.sum(np.abs(demand_diff) > 2 * np.std(demand_diff)) / len(demand_diff)
        if promo_ratio > 0.15:
            return 'Promotional'
        
        # 检查季节性
        # 使用自相关分析检查季节性
        n = len(demand_data)
        if n >= 12:  # 至少需要12个数据点来检测季节性
            autocorr = np.correlate(demand_data - np.mean(demand_data), 
                                  demand_data - np.mean(demand_data), mode='full')
            autocorr = autocorr[n-1:] / (np.var(demand_data) * np.arange(n, 0, -1))
            
            # 检查是否存在季节性（滞后12个月的自相关系数）
            if len(autocorr) >= 12 and autocorr[11] > 0.3:
                return 'Seasonal'
        
        # 稳定需求
        return 'Stable'
    
    def get_recommended_models(self, abc_class, demand_pattern):
        """
        根据ABC分类和需求模式获取推荐的模型列表
        
        Args:
            abc_class: ABC分类结果
            demand_pattern: 需求模式
            
        Returns:
            recommended_models: 推荐的模型名称列表
        """
        if abc_class == 'A':
            if demand_pattern in ['Stable', 'Seasonal']:
                return ['hybrid_arima_xgb', 'prophet', 'arima', 'holt_winters', 'xgb']
            elif demand_pattern == 'Promotional':
                return ['ensemble', 'xgb', 'gradient_boosting', 'random_forest', 'hybrid_arima_xgb']
            else:  # Intermittent
                return ['croston', 'sba', 'arima', 'svr', 'hybrid_arima_xgb']
        elif abc_class == 'B':
            return ['ensemble', 'gradient_boosting', 'random_forest', 'linear_regression', 'hybrid_arima_xgb']
        else:  # C类
            return ['linear_regression', 'svr', 'croston', 'sba', 'arima', 'hybrid_arima_xgb']
    
    def select_best_model(self, X_train, y_train, product_id, model_tag=None):
        """
        根据ABC分类、需求模式和交叉验证结果选择最佳模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            product_id: 产品ID
            model_tag: 模型选择标签，用于指导模型选择
            
        Returns:
            best_model: 最佳模型
            best_model_name: 最佳模型名称
            best_score: 最佳模型得分
        """
        # GPU加速处理：对于大规模数据，尝试使用GPU加速计算
        if self.use_gpu and self.has_cuda:
            # 检查数据规模是否适合GPU加速
            if X_train.shape[0] > 1000 or X_train.shape[1] > 50:
                # 记录GPU加速信息
                print(f"产品 {product_id} 使用GPU加速进行模型训练")
        # 如果提供了模型选择标签，则优先考虑对应的模型类型
        if model_tag:
            print(f"产品 {product_id} 使用模型选择标签: {model_tag}")
        else:
            print(f"产品 {product_id} 未提供模型选择标签，使用默认选择逻辑")
        
        # 计算产品价值（这里简化为总需求）
        product_value = np.sum(y_train)
        total_value = product_value  # 简化处理，实际应使用所有产品的总价值
        abc_class = self.classify_abc(product_value, total_value)
        
        # 识别需求模式
        demand_pattern = self.identify_demand_pattern(y_train)
        
        # 获取推荐模型列表
        recommended_models = self.get_recommended_models(abc_class, demand_pattern)
        
        # 根据model_tag调整推荐模型列表
        if model_tag:
            # 如果model_tag包含特定模型类型，优先考虑
            tag_to_models = {
                'high_variability': ['svr', 'hybrid_arima_xgb'],
                'seasonal': ['holt_winters', 'arima'],
                'intermittent': ['sarimax', 'hybrid_arima_xgb'],
                'stable': ['linear_regression', 'arima'],
                'trending': ['linear_regression', 'gradient_boosting']
            }
            
            # 获取与标签相关的模型
            tag_models = tag_to_models.get(model_tag, [])
            if tag_models:
                # 优先考虑标签相关的模型，但保留所有推荐模型
                prioritized_models = []
                for model in tag_models:
                    if model in recommended_models and model not in prioritized_models:
                        prioritized_models.append(model)
                # 添加剩余的推荐模型
                for model in recommended_models:
                    if model not in prioritized_models:
                        prioritized_models.append(model)
                recommended_models = prioritized_models
                print(f"根据模型标签调整后的推荐模型列表: {recommended_models}")
        
        print(f"产品 {product_id} 分类: ABC-{abc_class}, 需求模式: {demand_pattern}")
        print(f"推荐模型列表: {recommended_models}")
        
        best_score = float('inf')
        best_model = None
        best_model_name = ''
        
        # 对每个推荐模型进行交叉验证
        for model_name in recommended_models:
            try:
                if model_name in ['arima', 'holt_winters', 'prophet', 'croston', 'sba']:
                    # 统计模型和间歇性需求模型需要特殊处理
                    if model_name == 'arima':
                        # 简化处理：使用简单的ARIMA(1,1,1)模型
                        model = ARIMA(y_train, order=(1,1,1))
                        trained_model = model.fit()
                        # 使用历史数据进行回测评估
                        y_pred = trained_model.predict(start=1, end=len(y_train)-1, typ='levels')
                        rmse = np.sqrt(mean_squared_error(y_train[1:], y_pred))
                    elif model_name == 'holt_winters':
                        # 简化处理：使用加法模型
                        model = ExponentialSmoothing(y_train, trend='add', seasonal=None)
                        trained_model = model.fit()
                        # 使用历史数据进行回测评估
                        y_pred = trained_model.predict(start=1, end=len(y_train)-1)
                        rmse = np.sqrt(mean_squared_error(y_train[1:], y_pred))
                    elif model_name == 'prophet':
                        # Prophet模型需要特定的数据格式
                        df = pd.DataFrame({'ds': pd.date_range(start='2020-01-01', periods=len(y_train)), 'y': y_train})
                        model = Prophet()
                        trained_model = model.fit()
                        future = model.make_future_dataframe(periods=0, include_history=True)
                        forecast = trained_model.predict(future)
                        y_pred = forecast['yhat'].values
                        rmse = np.sqrt(mean_squared_error(y_train, y_pred))
                    elif model_name == 'croston':
                        # Croston模型处理间歇性需求
                        model = Croston()
                        trained_model = model.fit(y_train)
                        # 使用历史数据进行回测评估
                        y_pred = trained_model.forecast(steps=len(y_train))
                        rmse = np.sqrt(mean_squared_error(y_train, y_pred))
                    elif model_name == 'sba':
                        # SBA模型处理间歇性需求
                        model = SBA()
                        trained_model = model.fit(y_train)
                        # 使用历史数据进行回测评估
                        y_pred = trained_model.forecast(steps=len(y_train))
                        rmse = np.sqrt(mean_squared_error(y_train, y_pred))
                    
                    current_model = trained_model
                    current_model_name = model_name
                    current_rmse = rmse
                elif model_name == 'hybrid_arima_xgb':
                    # 混合模型特殊处理
                    model = HybridModel(ml_model=XGBRegressor(n_estimators=50, random_state=42))
                    model.fit(y_train)
                    # 使用历史数据进行回测评估
                    # 简化处理：只预测一步，滚动评估
                    y_pred = []
                    for i in range(1, len(y_train)):
                        temp_model = HybridModel(ml_model=XGBRegressor(n_estimators=50, random_state=42))
                        temp_model.fit(y_train[:i])
                        pred = temp_model.forecast(steps=1)[0]
                        y_pred.append(pred)
                    
                    rmse = np.sqrt(mean_squared_error(y_train[1:], y_pred))
                    current_model = model
                    current_model_name = model_name
                    current_rmse = rmse
                else:
                    # 机器学习模型使用交叉验证
                    model = self.all_models[model_name]
                    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
                    mse = -np.mean(scores)
                    rmse = np.sqrt(mse)
                    
                    current_model = model
                    current_model_name = model_name
                    current_rmse = rmse
                
                print(f"产品 {product_id} - {current_model_name} 评估RMSE: {current_rmse:.4f}")
                
                # 选择RMSE最小的模型
                if current_rmse < best_score:
                    best_score = current_rmse
                    best_model = current_model
                    best_model_name = current_model_name
            except Exception as e:
                print(f"模型 {model_name} 训练失败: {e}")
                continue
        
        # 如果没有找到合适的模型，使用默认模型
        if best_model is None:
            best_model_name = 'linear_regression'
            best_model = self.all_models[best_model_name]
            best_model.fit(X_train, y_train)
            best_score = np.sqrt(mean_squared_error(y_train, best_model.predict(X_train)))
        elif best_model_name not in ['arima', 'holt_winters']:
            # 机器学习模型需要拟合
            best_model.fit(X_train, y_train)
        
        # 保存模型
        self.save_model(best_model, best_model_name, product_id, {
            'abc_class': abc_class,
            'demand_pattern': demand_pattern
        })
        
        # 生成解释相关信息
        explanation = None
        business_rules = []
        feature_contribution = {}
        
        try:
            # 生成模型解释（如果支持）
            if self.interpreter:
                # 准备测试数据（使用简单分割）
                split_idx = max(1, len(X_train) - 5)  # 最后5个样本作为测试
                X_test, y_test = X_train[split_idx:], y_train[split_idx:]
                
                # 生成特征名称（如果没有提供）
                feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
                
                # 生成模型解释
                explanation = self.explain_model(best_model, best_model_name, X_train, X_test, y_train, feature_names)
                
                # 生成业务规则
                if explanation and explanation.get('explanation_id'):
                    business_rules = self.generate_business_rules(best_model, best_model_name, X_test, y_test, feature_names)
                
                # 获取特征贡献度
                feature_contribution = self.get_feature_contribution(best_model, best_model_name, X_test, feature_names)
        except Exception as e:
            print(f"生成模型解释或业务规则时出错: {e}")
        
        # 保存模型选择结果
        model_selection = {
            'model_name': best_model_name,
            'abc_class': abc_class,
            'demand_pattern': demand_pattern,
            'model_tag': model_tag,
            'selected_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 添加解释信息
        if explanation:
            model_selection['explanation'] = explanation
        
        if business_rules:
            model_selection['business_rules'] = business_rules
        
        if feature_contribution:
            model_selection['feature_contribution'] = feature_contribution
        
        self.model_selections[product_id] = model_selection
        
        return best_model, best_model_name, best_score
    
    def forecast(self, model, model_name, X=None):
        """
        使用模型进行预测
        
        Args:
            model: 训练好的模型
            model_name: 模型名称
            X: 特征数据（统计模型不需要）
            
        Returns:
            predictions: 预测结果
        """
        try:
            if model_name in ['arima', 'holt_winters', 'prophet', 'croston', 'sba']:
                # 统计模型和间歇性需求模型进行预测
                if model_name == 'arima':
                    # ARIMA预测未来一个点
                    predictions = model.forecast(steps=1)
                elif model_name == 'holt_winters':
                    # Holt-Winters预测未来一个点
                    predictions = model.forecast(steps=1)
                elif model_name == 'prophet':
                    # Prophet预测未来一个点
                    future = model.make_future_dataframe(periods=1, include_history=False)
                    forecast = model.predict(future)
                    predictions = forecast['yhat'].values
                elif model_name in ['croston', 'sba']:
                    # 间歇性需求模型预测
                    predictions = model.forecast(steps=1)
                # 将结果转换为与机器学习模型一致的格式
                return predictions.values if hasattr(predictions, 'values') else predictions
            elif model_name == 'hybrid_arima_xgb':
                # 混合模型预测
                predictions = model.forecast(steps=1)
                return predictions
            elif model_name == 'ensemble':
                # 模型融合预测
                return model.predict(X)
            else:
                # 机器学习模型预测
                return model.predict(X)
        except Exception as e:
            print(f"模型 {model_name} 预测失败: {e}")
            # 返回简单的预测结果
            return np.array([0])
    
    def predict(self, model, model_name, X):
        """
        使用模型对现有数据进行预测（用于误差分析）
        
        Args:
            model: 训练好的模型
            model_name: 模型名称
            X: 特征数据
            
        Returns:
            predictions: 预测结果
        """
        try:
            if model_name in ['arima', 'holt_winters', 'prophet', 'croston', 'sba', 'hybrid_arima_xgb']:
                # 统计模型和间歇性需求模型预测：使用训练好的模型
                predictions = []
                for i in range(len(X)):
                    if model_name in ['arima', 'holt_winters', 'hybrid_arima_xgb']:
                        # 使用训练好的ARIMA、Holt-Winters或混合模型进行预测
                        pred = model.forecast(steps=1)
                    elif model_name == 'prophet':
                        # 使用训练好的Prophet模型进行预测
                        future = model.make_future_dataframe(periods=1, include_history=False)
                        forecast = model.predict(future)
                        pred = forecast['yhat'].values[0]
                    elif model_name in ['croston', 'sba']:
                        # 使用训练好的间歇性需求模型进行预测
                        pred = model.forecast(steps=1)[0]
                    
                    # 处理不同类型的预测结果
                    if hasattr(pred, 'values'):
                        pred_value = pred.values[0] if hasattr(pred.values, '__getitem__') else pred.values
                    elif hasattr(pred, '__getitem__'):
                        pred_value = pred[0]
                    else:
                        pred_value = pred
                    
                    predictions.append(pred_value)
                return np.array(predictions)
            else:
                # 机器学习模型预测
                return model.predict(X)
        except Exception as e:
            print(f"预测时出错: {e}")
            # 返回基准预测
            try:
                if isinstance(X, pd.DataFrame):
                    # 如果是DataFrame，取第一个样本的第一个值
                    base_value = X.iloc[0, 0] if not X.empty else 0
                elif isinstance(X, pd.Series):
                    # 如果是Series，取第一个值
                    base_value = X.iloc[0] if not X.empty else 0
                elif hasattr(X, '__getitem__') and len(X) > 0:
                    # 如果是其他可索引对象，取第一个元素
                    base_value = X[0]
                else:
                    # 否则使用默认值0
                    base_value = 0
                return np.full(len(X), base_value)
            except Exception as fallback_e:
                print(f"基准预测也失败: {fallback_e}")
                # 最后的兜底方案，返回全0数组
                return np.zeros(len(X))
    
    def save_model(self, model, model_name, product_id, metadata=None):
        """
        保存模型到文件
        
        Args:
            model: 训练好的模型
            model_name: 模型名称
            product_id: 产品ID
            metadata: 模型元数据（可选）
        """
        model_path = os.path.join(self.model_dir, f'{product_id}_{model_name}.joblib')
        model_data = {
            'model': model,
            'model_name': model_name,
            'metadata': metadata or {}
        }
        joblib.dump(model_data, model_path)
        print(f"模型已保存到: {model_path}")
    
    def load_model(self, product_id):
        """
        加载产品对应的模型
        
        Args:
            product_id: 产品ID
            
        Returns:
            model: 加载的模型，如果不存在返回None
            model_name: 模型名称
            metadata: 模型元数据
        """
        # 查找产品对应的模型文件
        for file_name in os.listdir(self.model_dir):
            if file_name.startswith(f'{product_id}_'):
                model_path = os.path.join(self.model_dir, file_name)
                model_data = joblib.load(model_path)
                return model_data['model'], model_data['model_name'], model_data['metadata']
        
        print(f"未找到产品 {product_id} 的模型")
        return None, None, {}
    
    def update_model(self, product_id, new_X, new_y):
        """
        使用新数据更新模型
        
        Args:
            product_id: 产品ID
            new_X: 新的特征数据
            new_y: 新的标签数据
            
        Returns:
            updated_model: 更新后的模型
            model_name: 模型名称
        """
        # 加载现有模型
        model, model_name, metadata = self.load_model(product_id)
        
        if model is None:
            print(f"无法更新产品 {product_id} 的模型，模型不存在")
            return None, None
        
        if model_name in ['arima', 'holt_winters']:
            # 统计模型需要重新训练
            combined_y = np.concatenate([model.model.endog, new_y])
            if model_name == 'arima':
                new_model = ARIMA(combined_y, order=(1,1,1))
                updated_model = new_model.fit()
            else:  # holt_winters
                new_model = ExponentialSmoothing(combined_y, trend='add', seasonal=None)
                updated_model = new_model.fit()
        else:
            # 机器学习模型增量训练
            model.fit(new_X, new_y)
            updated_model = model
        
        # 保存更新后的模型
        self.save_model(updated_model, model_name, product_id, metadata)
        
        return updated_model, model_name
    
    def evaluate_model(self, model, model_name, X_test, y_test):
        """
        评估模型性能
        
        Args:
            model: 训练好的模型
            model_name: 模型名称
            X_test: 测试特征
            y_test: 测试标签
            
        Returns:
            metrics: 包含各种评估指标的字典
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        try:
            if model_name in ['arima', 'holt_winters', 'prophet', 'croston', 'sba', 'hybrid_arima_xgb']:
                # 单变量模型评估：使用模型对测试集进行预测
                # 注意：这是简化处理，实际应考虑滚动预测
                predictions = model.forecast(steps=len(y_test))
            else:
                # 机器学习模型评估
                predictions = model.predict(X_test)
            
            # 确保predictions是numpy数组
            if hasattr(predictions, 'values'):
                predictions = predictions.values
            elif isinstance(predictions, list):
                predictions = np.array(predictions)
            
            # 确保预测值和真实值长度一致
            if len(predictions) != len(y_test):
                # 如果预测值长度不足，使用最后一个预测值填充
                if len(predictions) < len(y_test):
                    predictions = np.pad(predictions, (0, len(y_test) - len(predictions)), 'edge')
                else:
                    predictions = predictions[:len(y_test)]
            
            # 计算评估指标
            metrics = {}
            
            # 基本指标
            metrics['mae'] = mean_absolute_error(y_test, predictions)
            metrics['mse'] = mean_squared_error(y_test, predictions)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['r2'] = r2_score(y_test, predictions)
            
            # 百分比指标
            non_zero_mask = y_test != 0
            if np.any(non_zero_mask):
                metrics['mape'] = np.mean(np.abs((y_test[non_zero_mask] - predictions[non_zero_mask]) / y_test[non_zero_mask])) * 100
            else:
                metrics['mape'] = 0.0
            
            # 对称平均绝对百分比误差
            denominator = (np.abs(y_test) + np.abs(predictions))
            non_zero_denominator = denominator != 0
            if np.any(non_zero_denominator):
                metrics['smape'] = 2 * np.mean(np.abs(y_test[non_zero_denominator] - predictions[non_zero_denominator]) / denominator[non_zero_denominator]) * 100
            else:
                metrics['smape'] = 0.0
            
            # 平均绝对标度误差
            if len(y_test) > 1:
                naive_pred = y_test[:-1]
                naive_true = y_test[1:]
                naive_mae = mean_absolute_error(naive_true, naive_pred)
                if naive_mae != 0:
                    metrics['mase'] = metrics['mae'] / naive_mae
                else:
                    metrics['mase'] = 0.0
            else:
                metrics['mase'] = 0.0
            
            # 准确率相关指标（对于需求预测，我们可以定义阈值）
            threshold = 0.1  # 允许10%的误差
            accurate_predictions = np.abs((y_test - predictions) / (y_test + 1e-8)) <= threshold
            metrics['accuracy_10'] = np.mean(accurate_predictions) * 100
            
            # 方向准确率（预测趋势是否正确）
            if len(y_test) > 1:
                actual_changes = y_test[1:] > y_test[:-1]
                predicted_changes = predictions[1:] > predictions[:-1]
                metrics['direction_accuracy'] = np.mean(actual_changes == predicted_changes) * 100
            else:
                metrics['direction_accuracy'] = 0.0
        except Exception as e:
            self.logger.error(f"评估模型时出错: {e}")
            # 使用简单的基准预测（历史平均值）
            predictions = np.full_like(y_test, np.mean(y_test))
            
            # 计算基本指标
            metrics = {
                'mae': mean_absolute_error(y_test, predictions),
                'mse': mean_squared_error(y_test, predictions),
                'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
                'r2': r2_score(y_test, predictions),
                'mape': 0.0,
                'smape': 0.0,
                'mase': 0.0,
                'accuracy_10': 0.0,
                'direction_accuracy': 0.0
            }
        
        return metrics
    
    def explain_model(self, model, model_name, X_train, X_test, y_train, feature_names=None):
        """
        生成模型解释
        
        Args:
            model: 训练好的模型
            model_name: 模型名称
            X_train: 训练特征
            X_test: 测试特征
            y_train: 训练标签
            feature_names: 特征名称列表
            
        Returns:
            explanation_results: 模型解释结果
        """
        if self.interpreter is None:
            return None
        
        # 跳过统计模型的解释（它们的可解释性较低）
        if model_name in ['arima', 'holt_winters', 'prophet', 'croston', 'sba']:
            return {
                "model_type": model_name,
                "message": "该模型类型不支持详细解释",
                "explanation_id": None
            }
        
        try:
            explanation_data, explanation_id, file_path = self.interpreter.generate_model_explanation(
                model, X_train, X_test, y_train, feature_names=feature_names
            )
            
            return {
                "model_type": model_name,
                "explanation_id": explanation_id,
                "explanation_data": explanation_data,
                "file_path": file_path
            }
        except Exception as e:
            print(f"生成模型解释时出错: {e}")
            return {
                "model_type": model_name,
                "message": f"生成解释时出错: {str(e)}",
                "explanation_id": None
            }
    
    def generate_business_rules(self, model, model_name, X, y, feature_names=None, top_n=10):
        """
        生成业务规则
        
        Args:
            model: 训练好的模型
            model_name: 模型名称
            X: 特征数据
            y: 目标变量
            feature_names: 特征名称列表
            top_n: 生成前N个规则
            
        Returns:
            business_rules: 业务规则列表
        """
        if self.rule_generator is None:
            return []
        
        try:
            rules = self.rule_generator.generate_business_rules(
                model, X, y, feature_names=feature_names, top_n=top_n
            )
            
            # 生成规则报告
            rule_report = self.rule_generator.generate_rule_report(rules, model_name)
            
            return {
                "rules": rules,
                "rule_report": rule_report,
                "simplified_rules": self.rule_generator.simplify_rules(rules)
            }
        except Exception as e:
            print(f"生成业务规则时出错: {e}")
            return []
    
    def get_feature_contribution(self, model, model_name, X, feature_names=None, top_n=10):
        """
        获取特征贡献度
        
        Args:
            model: 训练好的模型
            model_name: 模型名称
            X: 特征数据
            feature_names: 特征名称列表
            top_n: 显示前N个特征
            
        Returns:
            feature_contribution: 特征贡献度
        """
        if self.interpreter is None:
            return {}
        
        # 跳过统计模型
        if model_name in ['arima', 'holt_winters', 'prophet', 'croston', 'sba']:
            return {}
        
        try:
            # 获取特征重要性
            feature_importance = self.interpreter.get_feature_importance(model, X, X[:, 0] if len(X.shape) > 1 else X)
            
            # 只返回前N个特征
            top_features = feature_importance.head(top_n).to_dict(orient='records')
            
            return {
                "feature_contribution": top_features,
                "total_features": len(feature_importance)
            }
        except Exception as e:
            print(f"计算特征贡献度时出错: {e}")
            return {}
    
    def visualize_prediction(self, y_true, y_pred, model_name, product_id, explanation_id=None):
        """
        可视化预测结果
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            model_name: 模型名称
            product_id: 产品ID
            explanation_id: 解释ID
            
        Returns:
            plot_path: 可视化图像路径
        """
        plt.figure(figsize=(12, 6))
        
        # 绘制真实值和预测值
        plt.plot(y_true, label='真实需求', marker='o', color='blue')
        plt.plot(y_pred, label=f'{model_name}预测', marker='x', color='red')
        
        plt.title(f'产品 {product_id} 需求预测对比', fontsize=14)
        plt.xlabel('时间', fontsize=12)
        plt.ylabel('需求数量', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 保存图像
        os.makedirs('interpretations/figures', exist_ok=True)
        if explanation_id:
            plot_path = os.path.join('interpretations/figures', f"{explanation_id}_prediction_{product_id}.png")
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_path = os.path.join('interpretations/figures', f"prediction_{product_id}_{timestamp}.png")
        
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        return plot_path
    
    def update_model(self, product_id, X_new, y_new):
        """
        使用新数据更新模型
        
        Args:
            product_id: 产品ID
            X_new: 新的特征数据
            y_new: 新的标签数据
            
        Returns:
            updated_model: 更新后的模型
            model_name: 模型名称
        """
        # 加载现有模型
        model, model_name, metadata = self.load_model(product_id)
        
        if model is None:
            print(f"无法更新产品 {product_id} 的模型，模型不存在")
            return None, None
        
        if model_name in ['arima', 'holt_winters']:
            # 统计模型需要重新训练
            # 确保y_new是1维数组
            y_new_1d = y_new.values.ravel() if hasattr(y_new, 'values') else y_new.ravel()
            
            # 检查model.model.endog的形状并转换为1维
            endog_1d = model.model.endog.ravel()
            
            # 连接数组
            combined_y = np.concatenate([endog_1d, y_new_1d])
            
            if model_name == 'arima':
                new_model = ARIMA(combined_y, order=(1,1,1))
                updated_model = new_model.fit()
            else:  # holt_winters
                new_model = ExponentialSmoothing(combined_y, trend='add', seasonal=None)
                updated_model = new_model.fit()
        else:
            # 机器学习模型增量训练
            model.fit(X_new, y_new)
            updated_model = model
        
        # 保存更新后的模型
        self.save_model(updated_model, model_name, product_id, metadata)
        
        return updated_model, model_name
    
    def get_model_info(self, product_id):
        """
        获取产品模型信息
        
        Args:
            product_id: 产品ID
        """
        model, model_name, metadata = self.load_model(product_id)
        if model is not None:
            return {
                "status": "model_exists",
                "product_id": product_id,
                "model_name": model_name,
                "model_type": type(model).__name__,
                "metadata": metadata
            }
        else:
            return {"status": "model_not_found", "message": "模型不存在"}
    
    def online_learning(self, product_id, X_new, y_new, update_threshold=0.05):
        """
        在线学习 - 支持模型在线更新，适应数据分布变化
        
        Args:
            product_id: 产品ID
            X_new: 新的特征数据
            y_new: 新的目标值
            update_threshold: 性能下降阈值，超过则更新模型
            
        Returns:
            tuple: (updated_model, model_name, updated)
        """
        # 加载当前模型
        model, model_name, metadata = self.load_model(product_id)
        updated = False
        
        if model is None:
            print(f"无法进行在线学习，产品 {product_id} 的模型不存在")
            return None, None, False
        
        try:
            if model_name in ['arima', 'holt_winters', 'prophet', 'croston', 'sba']:
                # 对于统计模型，需要重新训练
                # 确保y_new是1维数组
                y_new_1d = y_new.values.ravel() if hasattr(y_new, 'values') else y_new.ravel()
                
                # 检查model.model.endog的形状并转换为1维
                endog_1d = model.model.endog.ravel()
                
                # 连接数组
                combined_y = np.concatenate([endog_1d, y_new_1d])
                
                if model_name == 'arima':
                    new_model = ARIMA(combined_y, order=(1,1,1))
                    updated_model = new_model.fit()
                elif model_name == 'holt_winters':
                    new_model = ExponentialSmoothing(combined_y, trend='add', seasonal=None)
                    updated_model = new_model.fit()
                elif model_name == 'prophet':
                    # Prophet模型需要特定格式的数据
                    df = pd.DataFrame({'ds': pd.date_range(start='2020-01-01', periods=len(combined_y)), 'y': combined_y})
                    updated_model = Prophet()
                    updated_model.fit(df)
                elif model_name == 'croston':
                    updated_model = Croston()
                    updated_model.fit(combined_y)
                elif model_name == 'sba':
                    updated_model = SBA()
                    updated_model.fit(combined_y)
                updated = True
            elif model_name == 'hybrid_arima_xgb':
                # 混合模型需要重新训练
                updated_model, model_name = self.update_model(product_id, X_new, y_new)
                updated = True
            elif model_name == 'ensemble':
                # 模型融合更新
                if hasattr(model, 'partial_fit'):
                    model.partial_fit(X_new, y_new)
                    updated_model = model
                    updated = True
                else:
                    updated_model, model_name = self.update_model(product_id, X_new, y_new)
                    updated = True
            else:
                # 对于机器学习模型
                if hasattr(model, 'partial_fit'):
                    # 支持增量学习的模型
                    model.partial_fit(X_new, y_new)
                    updated_model = model
                    updated = True
                else:
                    # 不支持增量学习的模型，评估性能后决定是否更新
                    y_pred = model.predict(X_new)
                    current_rmse = np.sqrt(mean_squared_error(y_new, y_pred))
                    
                    # 重新训练模型并比较性能
                    new_model, new_model_name = self.update_model(product_id, X_new, y_new)
                    new_y_pred = new_model.predict(X_new)
                    new_rmse = np.sqrt(mean_squared_error(y_new, new_y_pred))
                    
                    if new_rmse < current_rmse * (1 - update_threshold):
                        updated_model = new_model
                        updated = True
                    else:
                        updated_model = model
            
            # 保存更新后的模型
            if updated:
                self.save_model(updated_model, model_name, product_id, metadata)
                # 更新元数据中的最后更新时间
                metadata['last_updated'] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            
            return updated_model, model_name, updated
        except Exception as e:
            print(f"在线学习失败: {e}")
            return model, model_name, False
    
    def feedback_loop(self, product_id, predictions, actuals, feedback_weight=0.1):
        """
        反馈循环 - 将实际业务结果反馈到模型训练中，持续改进模型性能
        
        Args:
            product_id: 产品ID
            predictions: 模型预测值
            actuals: 实际业务结果
            feedback_weight: 反馈权重，控制反馈对模型的影响程度
            
        Returns:
            tuple: (updated_model, model_name, improvement)
        """
        # 加载当前模型
        model, model_name, metadata = self.load_model(product_id)
        improvement = 0.0
        
        if model is None:
            print(f"无法进行反馈循环，产品 {product_id} 的模型不存在")
            return None, None, 0.0
        
        try:
            # 计算预测误差
            errors = actuals - predictions
            
            # 生成反馈特征
            feedback_features = {
                'prediction_error': np.mean(np.abs(errors)),
                'error_variance': np.var(errors),
                'bias': np.mean(errors),
                'feedback_weight': feedback_weight
            }
            
            # 保存反馈信息到元数据
            if 'feedback_history' not in metadata:
                metadata['feedback_history'] = []
            metadata['feedback_history'].append({
                'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                'errors': errors.tolist(),
                'feedback_features': feedback_features,
                'improvement': improvement
            })
            
            # 简单更新模型（实际应用中可根据反馈特征调整模型）
            updated_model, model_name = self.update_model(product_id, self.X_train, self.y_train) if hasattr(self, 'X_train') else (model, model_name)
            
            # 计算改进幅度（如果有测试数据）
            if hasattr(self, 'X_test') and hasattr(self, 'y_test'):
                y_pred_old = model.predict(self.X_test)
                y_pred_new = updated_model.predict(self.X_test)
                rmse_old = np.sqrt(mean_squared_error(self.y_test, y_pred_old))
                rmse_new = np.sqrt(mean_squared_error(self.y_test, y_pred_new))
                improvement = (rmse_old - rmse_new) / rmse_old * 100 if rmse_old > 0 else 0
            
            # 保存更新后的模型和元数据
            self.save_model(updated_model, model_name, product_id, metadata)
            
            print(f"产品 {product_id} 的模型 {model_name} 反馈循环完成，性能改进: {improvement:.2f}%")
            return updated_model, model_name, improvement
        except Exception as e:
            print(f"反馈循环失败: {e}")
            return model, model_name, 0.0
    
    def schedule_retraining(self, retrain_interval=7, retrain_time='00:00', product_ids=None, X_train=None, y_train=None):
        """
        定期模型重训 - 建立自动化的模型重训机制，确保模型始终保持最佳性能
        
        Args:
            retrain_interval: 重训间隔天数
            retrain_time: 每天重训时间，格式为'HH:MM'
            product_ids: 产品ID列表，若为None则重训所有产品
            X_train: 训练特征数据
            y_train: 训练标签数据
            
        Returns:
            scheduler: 调度器实例或None
        """
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            from apscheduler.triggers.cron import CronTrigger
            
            # 创建调度器
            scheduler = BackgroundScheduler()
            
            # 定义重训任务
            def retrain_models():
                print(f"开始定期模型重训 - {pd.Timestamp.now()}")
                # 遍历指定产品或所有产品
                if product_ids is None:
                    # 获取所有产品ID（从模型文件中推断）
                    product_ids = []
                    for file_name in os.listdir(self.model_dir):
                        if file_name.endswith('.joblib'):
                            product_id = file_name.split('_')[0]
                            if product_id not in product_ids:
                                product_ids.append(product_id)
                
                for product_id in product_ids:
                    print(f"重训产品 {product_id} 的模型...")
                    if X_train is not None and y_train is not None:
                        # 选择最佳模型并训练
                        best_model, best_model_name, best_rmse = self.select_best_model(
                            X_train, y_train, product_id
                        )
                        # 更新元数据
                        metadata = {
                            "model_name": best_model_name,
                            "retrain_time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "rmse": float(best_rmse),
                            "feedback_history": []
                        }
                        # 保存最佳模型
                        self.save_model(best_model, best_model_name, product_id, metadata)
                        print(f"产品 {product_id} 模型重训完成，选择模型: {best_model_name}, RMSE: {best_rmse:.4f}")
                print(f"定期模型重训完成 - {pd.Timestamp.now()}")
            
            # 添加定时任务
            hour, minute = map(int, retrain_time.split(':'))
            scheduler.add_job(
                retrain_models,
                CronTrigger(hour=hour, minute=minute, day_of_week='*'),
                id='model_retraining',
                name='定期模型重训',
                replace_existing=True
            )
            
            # 启动调度器
            scheduler.start()
            print(f"定期模型重训调度器已启动，每天 {retrain_time} 执行")
            
            return scheduler
        except ImportError:
            print("apscheduler 未安装，无法启动定期重训调度器")
            print("请安装apscheduler: pip install apscheduler")
            return None
        except Exception as e:
            print(f"启动定期重训调度器失败: {e}")
            return None
