import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
import joblib
import os
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from xgboost import XGBRegressor
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
from scipy import stats

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
    
    def __init__(self):
        # 定义所有可用的预测模型
        self.all_models = {
            # 统计模型
            'arima': ARIMA,
            'holt_winters': ExponentialSmoothing,
            
            # 机器学习模型
            'linear_regression': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'xgb': XGBRegressor(n_estimators=100, random_state=42),
            'svr': SVR(kernel='rbf'),
            # 'lstm': LSTMModel()  # 需启用tensorflow/keras
            
            # 混合模型
            'hybrid_arima_xgb': HybridModel(ml_model=XGBRegressor(n_estimators=50, random_state=42))
        }
        
        # 模型保存路径
        self.model_dir = 'models'
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
    
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
                return ['hybrid_arima_xgb', 'arima', 'holt_winters', 'xgb']
            elif demand_pattern == 'Promotional':
                return ['xgb', 'gradient_boosting', 'random_forest', 'hybrid_arima_xgb']
            else:  # Intermittent
                return ['arima', 'svr', 'linear_regression', 'hybrid_arima_xgb']
        elif abc_class == 'B':
            return ['gradient_boosting', 'random_forest', 'linear_regression', 'hybrid_arima_xgb']
        else:  # C类
            return ['linear_regression', 'svr', 'arima', 'hybrid_arima_xgb']
    
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
                if model_name in ['arima', 'holt_winters']:
                    # 统计模型需要特殊处理
                    if model_name == 'arima':
                        # 简化处理：使用简单的ARIMA(1,1,1)模型
                        model = ARIMA(y_train, order=(1,1,1))
                        trained_model = model.fit()
                        # 使用历史数据进行回测评估
                        y_pred = trained_model.predict(start=1, end=len(y_train)-1, typ='levels')
                        rmse = np.sqrt(mean_squared_error(y_train[1:], y_pred))
                    else:  # holt_winters
                        # 简化处理：使用加法模型
                        model = ExponentialSmoothing(y_train, trend='add', seasonal=None)
                        trained_model = model.fit()
                        # 使用历史数据进行回测评估
                        y_pred = trained_model.predict(start=1, end=len(y_train)-1)
                        rmse = np.sqrt(mean_squared_error(y_train[1:], y_pred))
                    
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
        if model_name in ['arima', 'holt_winters']:
            # 统计模型进行预测
            if model_name == 'arima':
                # ARIMA预测未来一个点
                predictions = model.forecast(steps=1)
            else:  # holt_winters
                # Holt-Winters预测未来一个点
                predictions = model.forecast(steps=1)
            # 将结果转换为与机器学习模型一致的格式
            return predictions.values if hasattr(predictions, 'values') else [predictions]
        elif model_name == 'hybrid_arima_xgb':
            # 混合模型预测
            predictions = model.forecast(steps=1)
            return predictions
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
            if model_name in ['arima', 'holt_winters']:
                # 统计模型预测：使用训练好的模型
                predictions = []
                for i in range(len(X)):
                    if model_name == 'arima':
                        # 使用训练好的ARIMA模型进行预测
                        pred = model.forecast(steps=1)
                    else:  # holt_winters
                        # 使用训练好的Holt-Winters模型进行预测
                        pred = model.forecast(steps=1)
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
            
        Returns:
            model_info: 模型信息字典
        """
        model, model_name = self.load_model(product_id)
        
        if model is None:
            return None
        
        return {
            'product_id': product_id,
            'model_name': model_name,
            'model_type': type(model).__name__
        }
