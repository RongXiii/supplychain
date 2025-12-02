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
    """Croston方法用于处理间歇性需求"""
    
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.demand_level = None
        self.demand_interval = None
        self.last_demand = None
        self.last_interval = None
        self.is_fitted = False
    
    def fit(self, y):
        """训练模型"""
        # 初始化
        demand_mask = y > 0
        if np.sum(demand_mask) < 2:
            # 数据点不足，使用简单平均值
            self.demand_level = np.mean(y)
            self.demand_interval = 1
            self.is_fitted = True
            return self
        
        # 提取需求和间隔
        demand_values = y[demand_mask]
        intervals = np.diff(np.where(demand_mask)[0])
        
        # 初始化平滑值
        self.demand_level = demand_values[0]
        self.demand_interval = intervals[0]
        
        # 平滑处理
        for i in range(1, len(demand_values)):
            self.demand_level = self.alpha * demand_values[i] + (1 - self.alpha) * self.demand_level
            if i < len(intervals):
                self.demand_interval = self.alpha * intervals[i] + (1 - self.alpha) * self.demand_interval
        
        self.is_fitted = True
        return self
    
    def forecast(self, steps=1):
        """预测未来值"""
        if not self.is_fitted:
            raise ValueError("模型未训练")
        
        # Croston预测公式：需求水平 / 平均间隔
        forecast_value = self.demand_level / self.demand_interval
        return np.full(steps, forecast_value)

# 间歇性需求模型 - SBA方法
class SBA:
    """SBA (Syntetos-Boylan Approximation) 方法用于处理间歇性需求"""
    
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.demand_level = None
        self.demand_interval = None
        self.is_fitted = False
    
    def fit(self, y):
        """训练模型"""
        # 初始化
        demand_mask = y > 0
        if np.sum(demand_mask) < 2:
            # 数据点不足，使用简单平均值
            self.demand_level = np.mean(y)
            self.demand_interval = 1
            self.is_fitted = True
            return self
        
        # 提取需求和间隔
        demand_values = y[demand_mask]
        intervals = np.diff(np.where(demand_mask)[0])
        
        # 初始化平滑值
        self.demand_level = demand_values[0]
        self.demand_interval = intervals[0]
        
        # 平滑处理
        for i in range(1, len(demand_values)):
            self.demand_level = self.alpha * demand_values[i] + (1 - self.alpha) * self.demand_level
            if i < len(intervals):
                self.demand_interval = self.alpha * intervals[i] + (1 - self.alpha) * self.demand_interval
        
        self.is_fitted = True
        return self
    
    def forecast(self, steps=1):
        """预测未来值"""
        if not self.is_fitted:
            raise ValueError("模型未训练")
        
        # SBA修正公式
        a = self.alpha
        gamma = (2 - a) / (2 * (1 - a)) if a < 1 else 1
        forecast_value = self.demand_level / (self.demand_interval - gamma)
        return np.full(steps, forecast_value)

# 模型融合类
class ModelEnsemble(BaseEstimator, RegressorMixin):
    """模型融合器，结合多种模型的优势"""
    
    def __init__(self, models=None, weights=None, dynamic_weights=False):
        self.models = models or []
        self.weights = weights or [1/len(models) for _ in models] if models else []
        self.dynamic_weights = dynamic_weights
        self.is_fitted = False
    
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
        
        predictions = np.array([model.predict(X) for model in self.models]).T
        
        if self.dynamic_weights:
            # 动态权重调整：根据模型最近表现调整权重
            # 这里简化处理，使用交叉验证分数作为权重
            weights = np.array(self.weights)
        else:
            weights = np.array(self.weights)
        
        return np.sum(predictions * weights, axis=1)
    
    def update_weights(self, new_weights):
        """更新模型权重"""
        self.weights = new_weights
        
    def update_dynamic_weights(self, X_val, y_val):
        """根据验证集表现动态调整权重"""
        if not self.is_fitted:
            raise ValueError("模型未训练")
        
        # 计算每个模型在验证集上的性能
        performances = []
        for model in self.models:
            y_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            # 使用RMSE的倒数作为权重（性能越好，权重越大）
            performances.append(1 / (rmse + 1e-6))  # 避免除以零
        
        # 归一化权重
        total = sum(performances)
        self.weights = [p / total for p in performances]

# 超参数优化器
class HyperparameterOptimizer:
    """超参数自动优化器"""
    
    def __init__(self, model_name, n_trials=10, use_gpu=False):
        self.model_name = model_name
        self.n_trials = n_trials
        self.use_gpu = use_gpu
    
    def optimize(self, X_train, y_train):
        """优化模型超参数"""
        def objective(trial):
            if self.model_name == 'xgb':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                    'gamma': trial.suggest_float('gamma', 0, 1),
                    'tree_method': 'gpu_hist' if self.use_gpu else 'auto',
                    'device': 'cuda' if self.use_gpu else 'cpu'
                }
                model = XGBRegressor(**params, random_state=42)
            elif self.model_name == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2']),
                }
                model = RandomForestRegressor(**params, random_state=42)
            elif self.model_name == 'gradient_boosting':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                }
                model = GradientBoostingRegressor(**params, random_state=42)
            else:
                # 默认参数
                model = XGBRegressor(random_state=42)
            
            # 交叉验证
            scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
            return -np.mean(scores)  # 返回RMSE的负数，因为optuna最大化目标函数
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials)
        
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
        # 1. 用ARIMA模型捕捉趋势和季节性
        self.arima_model = ARIMA(y_train, order=(1,1,1))
        self.arima_model = self.arima_model.fit()
        
        # 2. 计算ARIMA残差
        arima_pred = self.arima_model.predict(start=0, end=len(y_train)-1, typ='levels')
        residuals = y_train - arima_pred
        
        # 3. 用ML模型预测残差
        # 准备特征：滞后残差和时间特征
        n_lags = 3
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
        
        # 1. ARIMA预测
        arima_forecast = self.arima_model.forecast(steps=steps)
        
        # 2. 用ML模型预测残差
        ml_residuals = []
        if hasattr(self.arima_model, 'model') and hasattr(self.arima_model.model, 'endog'):
            recent_residuals = self.arima_model.model.endog[-3:] - self.arima_model.predict(start=len(self.arima_model.model.endog)-3, end=len(self.arima_model.model.endog)-1, typ='levels')
            
            for i in range(steps):
                # 准备ML模型的特征
                if len(recent_residuals) >= 3:
                    lag_features = recent_residuals[-3:]
                    time_feature = (len(self.arima_model.model.endog) + i) / (len(self.arima_model.model.endog) + steps)
                    X_pred = np.array([list(lag_features) + [time_feature]])
                    residual_pred = self.ml_model.predict(X_pred)[0]
                    ml_residuals.append(residual_pred)
                    # 更新最近残差
                    recent_residuals = np.append(recent_residuals[1:], residual_pred)
                else:
                    ml_residuals.append(0)
        else:
            ml_residuals = [0] * steps
        
        # 3. 合并ARIMA预测和ML残差修正
        if hasattr(arima_forecast, 'values'):
            arima_values = arima_forecast.values
        else:
            arima_values = np.array(arima_forecast)
        
        final_predictions = arima_values + np.array(ml_residuals)
        return final_predictions


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
            if model_name in ['arima', 'holt_winters', 'prophet', 'croston', 'sba']:
                # 统计模型和间歇性需求模型预测：使用训练好的模型
                predictions = []
                for i in range(len(X)):
                    if model_name == 'arima':
                        # 使用训练好的ARIMA模型进行预测
                        pred = model.forecast(steps=1)
                    elif model_name == 'holt_winters':
                        # 使用训练好的Holt-Winters模型进行预测
                        pred = model.forecast(steps=1)
                    elif model_name == 'prophet':
                        # 使用训练好的Prophet模型进行预测
                        future = model.make_future_dataframe(periods=1, include_history=False)
                        forecast = model.predict(future)
                        pred = forecast['yhat'].values[0]
                    elif model_name in ['croston', 'sba']:
                        # 使用训练好的间歇性需求模型进行预测
                        pred = model.forecast(steps=1)[0]
                    predictions.append(pred[0] if hasattr(pred, '__getitem__') else pred)
                return np.array(predictions)
            elif model_name == 'hybrid_arima_xgb':
                # 混合模型预测
                return model.predict(X)
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
        try:
            if model_name in ['arima', 'holt_winters']:
                # 统计模型评估：使用模型对测试集进行预测
                # 注意：这是简化处理，实际应考虑滚动预测
                # 确保y_test至少有2个元素
                if len(y_test) < 2:
                    # 使用简单的基准预测
                    predictions = np.full_like(y_test, y_test[0])
                else:
                    predictions = []
                    for i in range(1, len(y_test)):  # 从i=1开始，确保有足够的数据
                        if model_name == 'arima':
                            # 使用累积数据重新训练模型
                            temp_model = ARIMA(y_test[:i+1], order=(1,1,1))
                            temp_model = temp_model.fit()
                            pred = temp_model.forecast(steps=1)
                        else:  # holt_winters
                            temp_model = ExponentialSmoothing(y_test[:i+1], trend='add', seasonal=None)
                            temp_model = temp_model.fit()
                            pred = temp_model.forecast(steps=1)
                        predictions.append(pred[0] if hasattr(pred, '__getitem__') else pred)
                    
                    # 如果predictions长度不足，使用基准预测填充
                    if len(predictions) < len(y_test):
                        predictions = np.pad(predictions, (len(y_test) - len(predictions), 0), 'constant', constant_values=y_test[0])
                    else:
                        predictions = np.array(predictions)
            else:
                # 机器学习模型评估
                predictions = model.predict(X_test)
            
            metrics = {
                'mae': mean_absolute_error(y_test, predictions),
                'mse': mean_squared_error(y_test, predictions),
                'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
                'mape': np.mean(np.abs((y_test - predictions) / y_test)) * 100
            }
        except Exception as e:
            print(f"评估模型时出错: {e}")
            # 使用简单的基准预测（历史平均值）
            predictions = np.full_like(y_test, np.mean(y_test))
            metrics = {
                'mae': mean_absolute_error(y_test, predictions),
                'mse': mean_squared_error(y_test, predictions),
                'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
                'mape': np.mean(np.abs((y_test - predictions) / y_test)) * 100
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
