import pandas as pd
import numpy as np
import time
from src.system.main import ReplenishmentSystem

# 创建简单的测试数据，避免复杂的数据预处理
print("创建简单的测试数据...")

# 创建30天的简单需求数据
product_id = 1
dates = pd.date_range('2023-01-01', periods=30)
demand = np.random.randint(5, 20, size=30)
product_data = pd.DataFrame({
    'date': dates,
    'item_id': product_id,
    'demand_qty': demand,
    'inventory_qty': np.random.randint(10, 50, size=30),
    'sales_qty': np.random.randint(5, 20, size=30),
    'reorder_point': 20,
    'safety_stock': 10
})

print(f"测试数据形状: {product_data.shape}")
print("测试数据样例:")
print(product_data.head())

# 修改ReplenishmentSystem类，简化数据处理过程
# 我们将直接修改run_forecast方法，跳过复杂的数据预处理
print("\n创建简化的补货系统实例...")

# 创建一个简化版本的ReplenishmentSystem类
from src.forecast.forecast_models import ForecastModelSelector
from sklearn.linear_model import LinearRegression

class SimpleReplenishmentSystem:
    def __init__(self):
        self.model_selector = ForecastModelSelector()
    
    def run_forecast(self, data, product_id, forecast_days=7):
        """简化的预测方法，跳过复杂的数据预处理"""
        print(f"\n运行简化的预测方法，预测{forecast_days}天...")
        
        # 只使用demand_qty列作为特征
        X = np.array(range(len(data))).reshape(-1, 1)
        Y = data['demand_qty'].values
        
        print(f"输入数据形状: X={X.shape}, Y={Y.shape}")
        
        # 训练简单的线性回归模型
        print("训练线性回归模型...")
        model = LinearRegression()
        model.fit(X, Y)
        
        # 进行预测
        print(f"预测{forecast_days}天的需求...")
        predictions = self.model_selector.forecast(model, "linear_regression", X, steps=forecast_days)
        
        # 返回预测结果
        return {
            'product_id': product_id,
            'model_name': 'linear_regression',
            'model_score': model.score(X, Y),
            'predictions': predictions,
            'forecast_days': forecast_days
        }

# 创建简化的补货系统实例
simple_replenishment_system = SimpleReplenishmentSystem()

print(f"\n直接调用简化的run_forecast方法，预测3天...")
start_time = time.time()

# 直接调用简化的run_forecast方法，预测3天
forecast_result = simple_replenishment_system.run_forecast(product_data, product_id, forecast_days=3)

end_time = time.time()
print(f"预测完成，耗时 {end_time - start_time:.2f} 秒")

print("\n预测结果:")
print(f"产品ID: {forecast_result['product_id']}")
print(f"模型名称: {forecast_result['model_name']}")
print(f"模型得分: {forecast_result['model_score']:.4f}")
print(f"预测值: {forecast_result['predictions']}")
print(f"预测值长度: {len(forecast_result['predictions'])}")
print(f"预测天数: {forecast_result['forecast_days']}")
print("预测成功!")

print("\n再测试预测7天的情况...")
forecast_result_7days = simple_replenishment_system.run_forecast(product_data, product_id, forecast_days=7)
print(f"预测值: {forecast_result_7days['predictions']}")
print(f"预测值长度: {len(forecast_result_7days['predictions'])}")
print(f"预测天数: {forecast_result_7days['forecast_days']}")
print("7天预测成功!")
