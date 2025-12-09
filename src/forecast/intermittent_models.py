import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, RegressorMixin

# 间歇性需求模型基类
class IntermittentDemandModel:
    """间歇性需求模型基类，提取公共功能"""
    
    def __init__(self, trend=True, seasonal=False, seasonal_periods=12):
        """
        初始化间歇性需求模型
        
        Args:
            trend: 是否考虑趋势
            seasonal: 是否考虑季节性
            seasonal_periods: 季节性周期长度
        """
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        
        # 公共属性
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
    
    def _initialize_smoothing_values(self, demand_values, demand_indices, intervals):
        """初始化平滑值"""
        self.demand_level = demand_values[0]
        self.demand_interval = intervals[0] if len(intervals) > 0 else 1
        self.demand_trend = self._detect_trend(demand_values[:min(3, len(demand_values))], 
                                              demand_indices[:min(3, len(demand_values))])
    
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

# 间歇性需求模型 - Croston方法
class Croston(IntermittentDemandModel):
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
        super().__init__(trend, seasonal, seasonal_periods)
        self.alpha = alpha
        self.beta = beta
    
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
        self._initialize_smoothing_values(demand_values, demand_indices, intervals)
        
        # 平滑处理 - Croston特定的实现
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

# 间歇性需求模型 - SBA方法
class SBA(IntermittentDemandModel):
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
        super().__init__(trend, seasonal, seasonal_periods)
        self.alpha = alpha
        self.beta = beta
        self.method = method
    
    def fit(self, y):
        """训练SBA模型，支持自适应平滑参数和趋势检测"""
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
        self._initialize_smoothing_values(demand_values, demand_indices, intervals)
        
        # SBA特定的修正 - 考虑间隔的方差
        if len(intervals) > 1:
            interval_std = np.std(intervals)
            interval_cv = interval_std / (self.demand_interval + 1e-8)
            # SBA修正因子
            sba_factor = 0.5 * interval_cv**2
            self.demand_interval = self.demand_interval * (1 + sba_factor)
        
        # 平滑处理 - SBA特定的实现
        for i in range(1, len(demand_values)):
            # 更新需求水平和趋势
            old_level = self.demand_level
            self.demand_level = self.alpha * demand_values[i] + (1 - self.alpha) * (old_level + self.demand_trend)
            
            if self.trend:
                self.demand_trend = self.beta * (self.demand_level - old_level) + (1 - self.beta) * self.demand_trend
            
            # 更新间隔
            if i < len(intervals):
                self.demand_interval = self.alpha * intervals[i] + (1 - self.alpha) * self.demand_interval
                
                # SBA特定的修正 - 考虑间隔的方差
                if len(intervals[:i+1]) > 1:
                    interval_std = np.std(intervals[:i+1])
                    interval_cv = interval_std / (self.demand_interval + 1e-8)
                    sba_factor = 0.5 * interval_cv**2
                    self.demand_interval = self.demand_interval * (1 + sba_factor)
        
        self.last_demand = demand_values[-1]
        self.last_interval = intervals[-1] if len(intervals) > 0 else 1
        self.last_period = demand_indices[-1]
        self.is_fitted = True
        return self

# 间歇性需求模型 - TSB方法
class TSB(IntermittentDemandModel):
    """TSB (Teunter-Syntetos-Babai) 方法用于处理间歇性需求，是SBA方法的改进版，使用不同的平滑参数分别更新需求水平和间隔"""
    
    def __init__(self, alpha_demand=None, alpha_interval=None, beta=None, trend=True, seasonal=False, seasonal_periods=12):
        """
        初始化TSB模型
        
        Args:
            alpha_demand: 需求水平平滑参数，默认为None（自适应）
            alpha_interval: 需求间隔平滑参数，默认为None（自适应）
            beta: 趋势平滑参数，默认为None（自适应）
            trend: 是否考虑趋势
            seasonal: 是否考虑季节性
            seasonal_periods: 季节性周期长度
        """
        super().__init__(trend, seasonal, seasonal_periods)
        self.alpha_demand = alpha_demand
        self.alpha_interval = alpha_interval
        self.beta = beta
    
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
        
        # 自适应计算alpha_demand（基于需求波动性）
        if self.alpha_demand is None:
            demand_std = np.std(demand_values) if len(demand_values) > 1 else 0
            demand_mean = np.mean(demand_values)
            volatility = demand_std / (demand_mean + 1e-8)
            self.alpha_demand = max(0.05, min(0.3, 0.1 + volatility * 0.2))
        
        # 自适应计算alpha_interval（基于间隔波动性）
        if self.alpha_interval is None:
            interval_std = np.std(intervals) if len(intervals) > 1 else 0
            interval_mean = np.mean(intervals)
            interval_volatility = interval_std / (interval_mean + 1e-8)
            self.alpha_interval = max(0.05, min(0.3, 0.1 + interval_volatility * 0.2))
        
        # 自适应计算beta（基于趋势强度）
        if self.beta is None:
            trend_strength = abs(self._detect_trend(demand_values, demand_indices))
            self.beta = max(0.01, min(0.2, trend_strength * 0.1))
        
        # 检测季节性
        has_seasonality = self._detect_seasonality(y) if self.seasonal else False
        
        if has_seasonality:
            self.seasonal_factors = self._calculate_seasonal_factors(y)
        
        # 初始化平滑值
        self._initialize_smoothing_values(demand_values, demand_indices, intervals)
        
        # 平滑处理 - TSB特定的实现（使用不同的平滑参数更新需求和间隔）
        for i in range(1, len(demand_values)):
            # 更新需求水平和趋势
            old_level = self.demand_level
            self.demand_level = self.alpha_demand * demand_values[i] + (1 - self.alpha_demand) * (old_level + self.demand_trend)
            
            if self.trend:
                self.demand_trend = self.beta * (self.demand_level - old_level) + (1 - self.beta) * self.demand_trend
            
            # 更新间隔（使用不同的平滑参数）
            if i < len(intervals):
                self.demand_interval = self.alpha_interval * intervals[i] + (1 - self.alpha_interval) * self.demand_interval
        
        self.last_demand = demand_values[-1]
        self.last_interval = intervals[-1] if len(intervals) > 0 else 1
        self.last_period = demand_indices[-1]
        self.is_fitted = True
        return self

# 间歇性需求模型 - TSB方法（简化版，用于模型列表）
class TSB_Simple(BaseEstimator, RegressorMixin):
    """简化版TSB模型，用于快速预测"""
    def __init__(self):
        self.model = TSB(alpha_demand=0.1, alpha_interval=0.1, beta=0.05, trend=True, seasonal=False)
    
    def fit(self, X, y):
        self.model.fit(y)
        return self
    
    def predict(self, X):
        steps = len(X)
        return self.model.forecast(steps)
