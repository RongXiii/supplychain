import os
import sys
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入预测模型相关模块
from src.forecast_models import (
    ForecastModelSelector, Croston, SBA, ModelEnsemble,
    HyperparameterOptimizer
)

# 测试数据生成函数
def generate_test_data(demand_pattern='stable', n_samples=100):
    """
    生成测试数据
    
    Args:
        demand_pattern: 需求模式 ('stable', 'seasonal', 'intermittent')
        n_samples: 样本数量
        
    Returns:
        X: 特征数据
        y: 目标变量
    """
    # 生成时间序列
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
    
    # 基础需求
    base_demand = 100
    
    if demand_pattern == 'stable':
        # 稳定需求
        y = np.ones(n_samples) * base_demand + np.random.normal(0, 5, n_samples)
    elif demand_pattern == 'seasonal':
        # 季节性需求
        t = np.arange(n_samples)
        seasonal = 20 * np.sin(2 * np.pi * t / 7)  # 周季节性
        y = base_demand + seasonal + np.random.normal(0, 5, n_samples)
    elif demand_pattern == 'intermittent':
        # 间歇性需求
        y = np.zeros(n_samples)
        # 随机生成需求发生的时间点
        demand_indices = np.random.choice(n_samples, size=int(n_samples * 0.3), replace=False)
        y[demand_indices] = np.random.poisson(lam=20, size=len(demand_indices))
    else:
        y = np.ones(n_samples) * base_demand + np.random.normal(0, 5, n_samples)
    
    # 生成特征
    X = pd.DataFrame({
        'day_of_week': dates.dayofweek,
        'month': dates.month,
        'quarter': dates.quarter,
        'is_weekend': (dates.dayofweek >= 5).astype(int)
    })
    
    return X.values, y, dates

# 测试1：间歇性需求模型测试
def test_intermittent_demand_models():
    """测试间歇性需求模型"""
    print("\n=== 测试1：间歇性需求模型 ===")
    
    # 生成间歇性需求数据
    X, y, dates = generate_test_data(demand_pattern='intermittent', n_samples=100)
    
    # 分割数据集
    train_size = int(len(y) * 0.8)
    y_train, y_test = y[:train_size], y[train_size:]
    
    # 测试Croston模型
    croston = Croston(alpha=0.1)
    croston.fit(y_train)
    croston_pred = croston.forecast(steps=len(y_test))
    croston_rmse = np.sqrt(mean_squared_error(y_test, croston_pred))
    print(f"Croston模型 RMSE: {croston_rmse:.4f}")
    
    # 测试SBA模型
    sba = SBA(alpha=0.1)
    sba.fit(y_train)
    sba_pred = sba.forecast(steps=len(y_test))
    sba_rmse = np.sqrt(mean_squared_error(y_test, sba_pred))
    print(f"SBA模型 RMSE: {sba_rmse:.4f}")
    
    return {
        'croston_rmse': croston_rmse,
        'sba_rmse': sba_rmse
    }

# 测试2：Prophet模型测试
def test_prophet_model():
    """测试Prophet模型"""
    print("\n=== 测试2：Prophet模型 ===")
    
    # 生成季节性需求数据
    X, y, dates = generate_test_data(demand_pattern='seasonal', n_samples=100)
    
    # 准备Prophet所需的数据格式
    df = pd.DataFrame({'ds': dates, 'y': y})
    
    # 分割数据集
    train_size = int(len(df) * 0.8)
    df_train, df_test = df[:train_size], df[train_size:]
    
    # 测试Prophet模型
    from prophet import Prophet
    prophet = Prophet()
    prophet.fit(df_train)
    
    # 预测
    future = prophet.make_future_dataframe(periods=len(df_test), include_history=False)
    forecast = prophet.predict(future)
    prophet_pred = forecast['yhat'].values
    prophet_rmse = np.sqrt(mean_squared_error(df_test['y'], prophet_pred))
    print(f"Prophet模型 RMSE: {prophet_rmse:.4f}")
    
    return {
        'prophet_rmse': prophet_rmse
    }

# 测试3：模型融合测试
def test_model_ensemble():
    """测试模型融合策略"""
    print("\n=== 测试3：模型融合 ===")
    
    # 生成稳定需求数据
    X, y, dates = generate_test_data(demand_pattern='stable', n_samples=100)
    
    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 创建基础模型
    from xgboost import XGBRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    
    models = [
        XGBRegressor(random_state=42),
        RandomForestRegressor(random_state=42),
        GradientBoostingRegressor(random_state=42)
    ]
    
    # 创建融合模型
    ensemble = ModelEnsemble(models=models, dynamic_weights=False)
    ensemble.fit(X_train, y_train)
    
    # 预测
    ensemble_pred = ensemble.predict(X_test)
    ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
    print(f"模型融合（静态权重） RMSE: {ensemble_rmse:.4f}")
    
    # 测试动态权重调整
    ensemble_dynamic = ModelEnsemble(models=models, dynamic_weights=True)
    ensemble_dynamic.fit(X_train, y_train)
    
    # 更新动态权重
    ensemble_dynamic.update_dynamic_weights(X_test, y_test)
    dynamic_pred = ensemble_dynamic.predict(X_test)
    dynamic_rmse = np.sqrt(mean_squared_error(y_test, dynamic_pred))
    print(f"模型融合（动态权重） RMSE: {dynamic_rmse:.4f}")
    print(f"动态权重: {ensemble_dynamic.weights}")
    
    return {
        'ensemble_rmse': ensemble_rmse,
        'dynamic_ensemble_rmse': dynamic_rmse
    }

# 测试4：超参数自动优化测试
def test_hyperparameter_optimization():
    """测试超参数自动优化"""
    print("\n=== 测试4：超参数自动优化 ===")
    
    # 生成稳定需求数据
    X, y, dates = generate_test_data(demand_pattern='stable', n_samples=100)
    
    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 测试XGBoost超参数优化
    optimizer = HyperparameterOptimizer(model_name='xgb', n_trials=5)
    best_params = optimizer.optimize(X_train, y_train)
    print(f"XGBoost最优参数: {best_params}")
    
    # 使用最优参数训练模型
    from xgboost import XGBRegressor
    model = XGBRegressor(**best_params, random_state=42)
    model.fit(X_train, y_train)
    
    # 预测
    pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    print(f"优化后XGBoost RMSE: {rmse:.4f}")
    
    # 与默认参数对比
    default_model = XGBRegressor(random_state=42)
    default_model.fit(X_train, y_train)
    default_pred = default_model.predict(X_test)
    default_rmse = np.sqrt(mean_squared_error(y_test, default_pred))
    print(f"默认参数XGBoost RMSE: {default_rmse:.4f}")
    
    improvement = ((default_rmse - rmse) / default_rmse) * 100
    print(f"性能提升: {improvement:.2f}%")
    
    return {
        'optimized_rmse': rmse,
        'default_rmse': default_rmse,
        'improvement': improvement
    }

# 测试5：ForecastModelSelector集成测试
def test_forecast_model_selector():
    """测试ForecastModelSelector集成新功能"""
    print("\n=== 测试5：ForecastModelSelector集成测试 ===")
    
    # 创建ForecastModelSelector实例
    selector = ForecastModelSelector()
    
    # 生成测试数据
    X, y, dates = generate_test_data(demand_pattern='seasonal', n_samples=100)
    
    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 测试模型推荐
    abc_class = 'A'
    demand_pattern = 'Seasonal'
    # 使用get_recommended_models方法获取推荐模型列表
    recommended_models = selector.get_recommended_models(abc_class, demand_pattern)
    print(f"推荐模型列表: {recommended_models}")
    recommended_model = recommended_models[0] if recommended_models else 'arima'
    print(f"选择的推荐模型: {recommended_model}")
    
    # 测试模型选择和训练
    selected_model = selector.select_best_model(X_train, y_train, [recommended_model])
    print(f"选择并训练的模型: {selected_model}")
    
    # 测试预测 - 使用训练好的模型进行预测
    # 注意：ForecastModelSelector.forecast方法需要3个参数：model, model_name, X
    forecast_result = selected_model[0].predict(X_test)
    print(f"预测结果示例: {forecast_result[:5]}")
    
    # 测试模型评估
    mse = mean_squared_error(y_test, forecast_result)
    rmse = np.sqrt(mse)
    print(f"模型RMSE: {rmse:.4f}")
    
    # 测试模型更新（使用少量新数据）
    updated = selector.update_model(['hybrid_arima_xgb'], X_train[:10], y_train[:10])
    print(f"模型更新结果: {updated}")
    
    return {
        'recommended_model': recommended_model,
        'forecast_result': forecast_result[:5],
        'rmse': rmse
    }

# 运行所有测试
def run_all_tests():
    """运行所有测试"""
    print("供应链智能补货系统 - 预测模型优化测试")
    print("=" * 50)
    
    results = {
        'intermittent_demand': test_intermittent_demand_models(),
        'prophet': test_prophet_model(),
        'ensemble': test_model_ensemble(),
        'hyperparameter': test_hyperparameter_optimization(),
        'selector': test_forecast_model_selector()
    }
    
    print("\n" + "=" * 50)
    print("测试结果汇总:")
    print(f"1. 间歇性需求模型 - Croston RMSE: {results['intermittent_demand']['croston_rmse']:.4f}")
    print(f"   间歇性需求模型 - SBA RMSE: {results['intermittent_demand']['sba_rmse']:.4f}")
    print(f"2. Prophet模型 RMSE: {results['prophet']['prophet_rmse']:.4f}")
    print(f"3. 模型融合 - 静态权重 RMSE: {results['ensemble']['ensemble_rmse']:.4f}")
    print(f"   模型融合 - 动态权重 RMSE: {results['ensemble']['dynamic_ensemble_rmse']:.4f}")
    print(f"4. 超参数优化 - 优化后 RMSE: {results['hyperparameter']['optimized_rmse']:.4f}")
    print(f"   超参数优化 - 默认参数 RMSE: {results['hyperparameter']['default_rmse']:.4f}")
    print(f"   超参数优化 - 性能提升: {results['hyperparameter']['improvement']:.2f}%")
    print(f"5. 模型选择器推荐模型: {results['selector']['recommended_model']}")
    print("=" * 50)
    
    return results

if __name__ == "__main__":
    # 检查是否安装了必要的依赖
    try:
        import prophet
        import optuna
    except ImportError as e:
        print(f"错误: 缺少必要的依赖包。请安装: {e.name}")
        print("安装命令: pip install prophet optuna")
        sys.exit(1)
    
    run_all_tests()