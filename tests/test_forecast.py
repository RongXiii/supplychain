import pandas as pd
import numpy as np
import time
from src.system.main import ReplenishmentSystem
from src.forecast.forecast_models import ForecastModelSelector

# 加载测试数据
inventory_data = pd.read_csv('./data/inventory_daily.csv')

# 获取产品1的数据
product_id = 1
product_data = inventory_data[inventory_data['item_id'] == product_id]

print(f"测试产品 {product_id} 的预测功能...")
print(f"产品数据形状: {product_data.shape}")

# 只使用最近30天的数据来加速测试
product_data = product_data.tail(30)
print(f"使用最近30天数据，形状: {product_data.shape}")

# 创建预测模型选择器实例
model_selector = ForecastModelSelector()

# 直接测试模型选择器的forecast方法
# 使用简单的线性回归模型进行测试
from sklearn.linear_model import LinearRegression

# 准备简单的数据
X = np.array(range(len(product_data))).reshape(-1, 1)
Y = product_data['demand_qty'].values

# 训练简单的线性回归模型
model = LinearRegression()
model.fit(X, Y)

print("\n测试简单线性回归模型的预测:")
forecast_result = model_selector.forecast(model, "linear_regression", X, steps=3)
print(f"预测值: {forecast_result}")
print(f"预测值长度: {len(forecast_result)}")

print("\n简单模型预测测试完成!")
