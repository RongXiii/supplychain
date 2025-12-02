import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录和src目录到Python路径
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, project_root)
sys.path.insert(0, src_path)

# 导入模型类
from forecast.forecast_models import Croston, SBA, ModelEnsemble, ForecastModelSelector
from data.data_processor import DataProcessor
from data.data_warehouse import DataWarehouse
# from forecast.real_time_processor import RealTimeDataProcessor, RealTimeForecastUpdater
# from system.main import ReplenishmentSystem

class TestForecastModels(unittest.TestCase):
    def setUp(self):
        # 准备测试数据
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=60, freq='D')
        self.test_data = pd.DataFrame({
            'date': dates,
            'sales': np.random.randint(0, 100, size=60)
        })
        # 准备用于机器学习模型的特征和标签
        self.X_train = np.arange(60).reshape(-1, 1)
        self.y_train = self.test_data['sales'].values
    
    def test_croston_model(self):
        # 测试Croston间歇性需求模型
        model = Croston()
        model.fit(self.test_data['sales'].values)
        result = model.forecast(7)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), 7)
    
    def test_sba_model(self):
        # 测试SBA间歇性需求模型
        model = SBA()
        model.fit(self.test_data['sales'].values)
        result = model.forecast(7)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), 7)
    
    def test_model_ensemble(self):
        # 测试模型融合
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor
        
        # 创建基础模型
        model1 = LinearRegression()
        model2 = RandomForestRegressor(n_estimators=10, random_state=42)
        
        # 创建模型融合器
        ensemble = ModelEnsemble(models=[model1, model2], weights=[0.5, 0.5])
        ensemble.fit(self.X_train, self.y_train)
        
        # 准备测试数据
        X_test = np.arange(60, 67).reshape(-1, 1)
        result = ensemble.predict(X_test)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), 7)
    
    def test_forecast_model_selector(self):
        # 测试预测模型选择器
        selector = ForecastModelSelector()
        result = selector.select_best_model(self.X_train, self.y_train, product_id='TEST001')
        self.assertIsNotNone(result)
        self.assertIsInstance(result, tuple)

class TestDataProcessor(unittest.TestCase):
    def setUp(self):
        # 准备测试数据 - 只包含数值类型
        np.random.seed(42)
        self.test_data = pd.DataFrame({
            'sales': np.random.randint(0, 100, size=100),
            'inventory': np.random.randint(50, 200, size=100)
        })
        self.processor = DataProcessor()
    
    def test_data_split(self):
        # 测试数据分割
        X_train, X_val, X_test, y_train, y_val, y_test = self.processor.split_data(self.test_data, target_column='sales', test_size=0.2)
        self.assertIsInstance(X_train, np.ndarray)
        self.assertIsInstance(X_val, np.ndarray)
        self.assertIsInstance(X_test, np.ndarray)
        self.assertIsInstance(y_train, pd.Series)
        self.assertIsInstance(y_val, pd.Series)
        self.assertIsInstance(y_test, pd.Series)
        self.assertEqual(len(X_train) + len(X_val) + len(X_test), len(self.test_data))

class TestDataWarehouse(unittest.TestCase):
    def test_data_warehouse_initialization(self):
        # 测试数据仓库初始化
        warehouse = DataWarehouse()
        self.assertIsInstance(warehouse, DataWarehouse)
    
    def test_data_warehouse_operations(self):
        # 测试数据仓库的基本操作
        warehouse = DataWarehouse()
        
        # 模拟一些测试数据
        test_data = pd.DataFrame({
            'sku': ['SKU001', 'SKU002'],
            'date': ['2023-01-01', '2023-01-02'],
            'inventory_on_hand': [100, 200]
        })
        
        # 测试数据更新（模拟插入数据）
        warehouse.update_data('test_table', test_data)
        
        # 测试数据查询
        result = warehouse.get_data('test_table')
        self.assertEqual(len(result), 2)
        
        # 测试数据更新
        update_data = pd.DataFrame({
            'sku': ['SKU001'],
            'date': ['2023-01-03'],
            'inventory_on_hand': [150]
        })
        warehouse.update_data('test_table', update_data)
        result = warehouse.get_data('test_table')
        self.assertEqual(len(result), 3)
        
        # 测试血缘追踪
        lineage = warehouse.get_data_lineage('test_table')
        self.assertIsInstance(lineage, list)



if __name__ == '__main__':
    unittest.main()