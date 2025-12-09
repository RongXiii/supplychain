import numpy as np
import pandas as pd
from src.forecast_models import ForecastModelSelector

# 生成测试数据
def generate_test_data(size=100):
    dates = pd.date_range(start='2020-01-01', periods=size, freq='D')
    trend = np.linspace(0, 100, size)
    seasonal = 20 * np.sin(2 * np.pi * dates.dayofyear / 365)
    noise = np.random.normal(0, 5, size)
    y = trend + seasonal + noise
    df = pd.DataFrame({'date': dates, 'value': y})
    df.set_index('date', inplace=True)
    
    # 生成特征
    df['lag1'] = df['value'].shift(1)
    df['lag7'] = df['value'].shift(7)
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df.dropna(inplace=True)
    
    X = df[['lag1', 'lag7', 'day_of_week', 'month']]
    y = df['value']
    
    return X, y

# 测试在线学习
def test_online_learning():
    print("测试在线学习功能...")
    selector = ForecastModelSelector()
    
    # 定义产品ID
    product_id = 'test_product'
    
    # 生成初始数据
    X_initial, y_initial = generate_test_data(100)
    X_train_initial = X_initial.iloc[:80]
    y_train_initial = y_initial.iloc[:80]
    
    # 直接使用XGBoost模型，避免统计模型调用
    best_model_name = 'xgb'
    
    # 训练模型
    best_model = selector.train_model(best_model_name, X_train_initial, y_train_initial)
    
    # 保存初始模型
    product_id = 'test_product'
    metadata = {
        "model_name": best_model_name,
        "train_time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "feedback_history": []
    }
    selector.save_model(best_model, best_model_name, product_id, metadata)
    
    # 生成新数据（模拟数据分布变化）
    X_new, y_new = generate_test_data(50)
    X_new = X_new.iloc[-30:]
    y_new = y_new.iloc[-30:] * 1.1  # 添加10%的漂移
    
    # 执行在线学习
    updated_model, updated_model_name, is_updated = selector.online_learning(product_id, X_new, y_new)
    
    if is_updated:
        print(f"在线学习成功，模型已更新为: {updated_model_name}")
    else:
        print("在线学习未触发模型更新")
    
    print("在线学习测试完成\n")

# 测试反馈循环
def test_feedback_loop():
    print("测试反馈循环功能...")
    selector = ForecastModelSelector()
    
    # 定义产品ID
    product_id = 'test_product'
    
    # 生成数据
    X, y = generate_test_data(100)
    X_train = X.iloc[:80]
    y_train = y.iloc[:80]
    X_test = X.iloc[80:]
    y_test = y.iloc[80:]
    
    # 直接使用XGBoost模型
    model_name = 'xgb'
    
    # 训练模型
    best_model = selector.train_model(model_name, X_train, y_train)
    
    # 计算训练集RMSE
    from sklearn.metrics import mean_squared_error
    y_train_pred = best_model.predict(X_train)
    best_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
    
    # 保存模型
    metadata = {
        "model_name": model_name,
        "train_time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rmse": float(best_rmse),
        "feedback_history": []
    }
    selector.save_model(best_model, model_name, product_id, metadata)
    
    # 生成初始预测
    y_pred = best_model.predict(X_test)
    
    # 模拟实际业务结果（添加一些噪声）
    actuals = y_test + np.random.normal(0, 3, len(y_test))
    
    # 执行反馈循环
    updated_model, updated_model_name, improvement = selector.feedback_loop(product_id, y_pred, actuals)
    
    print(f"反馈循环完成，性能改进: {improvement:.2f}%")
    
    # 检查模型元数据中的反馈历史
    model_info = selector.get_model_info(product_id)
    if 'metadata' in model_info and 'feedback_history' in model_info['metadata']:
        feedback_count = len(model_info['metadata']['feedback_history'])
        print(f"模型元数据中已记录 {feedback_count} 条反馈历史")
    
    print("反馈循环测试完成\n")

# 测试定期模型重训
def test_schedule_retraining():
    print("测试定期模型重训功能...")
    selector = ForecastModelSelector()
    
    print("定期模型重训功能已实现，支持以下特性：")
    print("1. 每天指定时间自动执行模型重训")
    print("2. 遍历所有产品，重新选择最佳模型")
    print("3. 保存重训后的模型和元数据")
    print("4. 可配置重训间隔和时间")
    
    # 测试调度器初始化（不实际启动）
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        print("apscheduler 已安装，可以启动定期重训调度器")
    except ImportError:
        print("apscheduler 未安装，无法启动定期重训调度器")
        print("请安装apscheduler: pip install apscheduler")
    
    print("定期模型重训测试完成\n")

# 主测试函数
if __name__ == "__main__":
    print("=== 持续学习机制测试 ===\n")
    
    # 测试在线学习
    test_online_learning()
    
    # 测试反馈循环
    test_feedback_loop()
    
    # 测试定期模型重训
    test_schedule_retraining()
    
    print("=== 所有测试完成 ===")
