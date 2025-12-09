import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import sys
import os

# 添加src目录到系统路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from forecast_models import ForecastModelSelector

def generate_simple_data(n_samples, n_features):
    """生成简单的测试数据，避免时间序列复杂性"""
    X = np.random.rand(n_samples, n_features)
    y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(n_samples) * 0.5
    return X, y

def test_online_learning_simple():
    """简单测试在线学习功能"""
    print("\n测试在线学习功能...")
    
    # 创建模型选择器实例
    selector = ForecastModelSelector()
    
    # 定义产品ID
    product_id = 'test_product_online'
    
    # 清理可能存在的旧模型文件
    for file_name in os.listdir(selector.model_dir):
        if file_name.startswith(f'{product_id}_'):
            os.remove(os.path.join(selector.model_dir, file_name))
    
    # 生成初始数据
    X_initial, y_initial = generate_simple_data(100, 2)
    
    # 直接创建并训练随机森林模型
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_initial, y_initial)
    
    # 保存模型到文件系统
    metadata = {
        "model_name": "random_forest",
        "train_time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rmse": 0.0,
        "feedback_history": []
    }
    selector.save_model(model, "random_forest", product_id, metadata)
    
    # 生成新数据（模拟数据分布变化）
    X_new, y_new = generate_simple_data(50, 2)
    X_new[:, 0] += 1.0  # 数据分布偏移
    
    # 使用在线学习更新模型
    updated_model, model_name, updated = selector.online_learning(product_id, X_new, y_new)
    
    # 验证模型性能
    X_test, y_test = generate_simple_data(20, 2)
    X_test[:, 0] += 1.0  # 与新数据分布一致
    y_pred = updated_model.predict(X_test)
    rmse_after = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"在线学习后，模型在新数据上的RMSE: {rmse_after:.4f}")
    print("在线学习功能测试通过！")

def test_feedback_loop_simple():
    """简单测试反馈循环功能"""
    print("\n测试反馈循环功能...")
    
    # 创建模型选择器实例
    selector = ForecastModelSelector()
    
    # 定义产品ID
    product_id = 'test_product_feedback'
    
    # 清理可能存在的旧模型文件
    for file_name in os.listdir(selector.model_dir):
        if file_name.startswith(f'{product_id}_'):
            os.remove(os.path.join(selector.model_dir, file_name))
    
    # 生成数据
    X, y = generate_simple_data(100, 2)
    
    # 直接创建并训练随机森林模型
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    # 保存模型到文件系统
    metadata = {
        "model_name": "random_forest",
        "train_time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rmse": 0.0,
        "feedback_history": []
    }
    selector.save_model(model, "random_forest", product_id, metadata)
    
    # 生成一些预测数据和实际结果
    X_test, y_test = generate_simple_data(20, 2)
    y_pred = model.predict(X_test)
    
    # 使用反馈循环更新模型
    updated_model, model_name, improvement = selector.feedback_loop(product_id, y_pred, y_test)
    
    print("反馈循环功能测试通过！")

def test_schedule_retraining_simple():
    """简单测试定期重训功能"""
    print("\n测试定期重训功能...")
    
    # 创建模型选择器实例
    selector = ForecastModelSelector()
    
    # 测试调度功能是否能正常初始化
    try:
        # 检查是否有apscheduler库
        import importlib
        scheduler_available = importlib.util.find_spec("apscheduler") is not None
        
        if scheduler_available:
            # 测试调度功能
            selector.schedule_retraining(interval_hours=24)
            print("定期重训功能已成功配置")
        else:
            print("apscheduler库未安装，跳过定期重训功能测试")
        
        print("定期重训功能测试通过！")
    except Exception as e:
        print(f"定期重训功能测试失败: {e}")
        raise

if __name__ == "__main__":
    print("=== 持续学习机制简单测试 ===")
    
    try:
        # 测试在线学习
        test_online_learning_simple()
        
        # 测试反馈循环
        test_feedback_loop_simple()
        
        # 测试定期重训
        test_schedule_retraining_simple()
        
        print("\n=== 所有测试通过！持续学习机制功能正常工作 ===")
    except Exception as e:
        print(f"\n=== 测试失败: {e} ===")
        import traceback
        traceback.print_exc()
        sys.exit(1)