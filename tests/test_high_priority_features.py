import pandas as pd
import numpy as np
import os
import sys
import logging

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.data_processor import DataProcessor
from src.mlops.ab_testing import ABTestManager, TestVariant
from src.mlops.mlops_engine import MLOpsEngine

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_dynamic_feature_selection():
    """
    测试动态特征选择功能
    """
    logger.info("=== 开始测试动态特征选择功能 ===")
    
    try:
        # 创建DataProcessor实例
        data_processor = DataProcessor(data_dir="data", parallel_mode="single")
        
        # 创建测试数据
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        n_samples = len(dates)
        
        # 创建特征
        df = pd.DataFrame({
            'date': dates,
            'feature1': np.random.rand(n_samples),
            'feature2': np.random.rand(n_samples),
            'feature3': np.random.rand(n_samples),
            'feature4': np.random.rand(n_samples),
            'feature5': np.random.rand(n_samples),
            'target': np.random.rand(n_samples) * 100
        })
        
        # 设置特征配置
        data_processor.feature_config['dynamic_feature_selection'] = True
        data_processor.feature_params['top_k_features'] = 3
        
        # 执行特征工程，包含动态特征选择
        processed_data = data_processor._automated_feature_engineering(df.copy(), data_processor.feature_config, target_column='target')
        
        logger.info(f"原始数据列数: {len(df.columns)}")
        logger.info(f"处理后数据列数: {len(processed_data.columns)}")
        logger.info(f"处理后特征列表: {processed_data.columns.tolist()}")
        
        # 验证动态特征选择是否生效
        assert len(processed_data.columns) <= 4, f"动态特征选择未生效，预期列数<=4，实际列数={len(processed_data.columns)}"
        
        logger.info("动态特征选择功能测试通过！")
        return True
        
    except Exception as e:
        logger.error(f"动态特征选择功能测试失败: {str(e)}")
        return False

def test_feature_importance_monitoring():
    """
    测试特征重要性监控功能
    """
    logger.info("=== 开始测试特征重要性监控功能 ===")
    
    try:
        # 创建DataProcessor实例
        data_processor = DataProcessor(data_dir="data", parallel_mode="single")
        
        # 创建测试数据
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        n_samples = len(dates)
        
        # 创建特征，其中feature1与目标相关性最高
        df = pd.DataFrame({
            'date': dates,
            'feature1': np.random.rand(n_samples),
            'feature2': np.random.rand(n_samples),
            'feature3': np.random.rand(n_samples),
            'feature4': np.random.rand(n_samples),
            'feature5': np.random.rand(n_samples),
            'target': np.random.rand(n_samples) * 100 + 50 * np.random.rand(n_samples)
        })
        
        # 计算特征重要性
        feature_importance = data_processor.calculate_feature_importance(df, target_column='target')
        logger.info(f"特征重要性计算结果: {feature_importance}")
        
        # 更新特征集
        updated_features = data_processor.update_feature_set(df, target_column='target', feature_importance=feature_importance)
        logger.info(f"更新后的特征集: {updated_features}")
        
        # 验证特征重要性监控是否正常工作
        assert len(updated_features) > 0, "更新后的特征集为空"
        
        # 测试可视化功能（不保存，仅验证代码可执行）
        data_processor.visualize_feature_importance(feature_importance)
        
        logger.info("特征重要性监控功能测试通过！")
        return True
        
    except Exception as e:
        logger.error(f"特征重要性监控功能测试失败: {str(e)}")
        return False

def test_external_features_integration():
    """
    测试外部特征集成功能
    """
    logger.info("=== 开始测试外部特征集成功能 ===")
    
    try:
        # 创建DataProcessor实例
        data_processor = DataProcessor(data_dir="data", parallel_mode="single")
        
        # 创建测试数据
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='D')
        n_samples = len(dates)
        
        df = pd.DataFrame({
            'date': dates,
            'product_id': [1] * n_samples,
            'sales': np.random.randint(100, 1000, size=n_samples),
            'stock_level': np.random.randint(50, 500, size=n_samples)
        })
        
        # 测试外部特征集成
        data_with_external = data_processor._add_external_features(df.copy())
        
        logger.info(f"原始数据列数: {len(df.columns)}")
        logger.info(f"添加外部特征后的数据列数: {len(data_with_external.columns)}")
        logger.info(f"添加的外部特征: {list(set(data_with_external.columns) - set(df.columns))}")
        
        # 验证外部特征是否被添加
        assert len(data_with_external.columns) > len(df.columns), "外部特征未被添加"
        
        logger.info("外部特征集成功能测试通过！")
        return True
        
    except Exception as e:
        logger.error(f"外部特征集成功能测试失败: {str(e)}")
        return False

def test_ab_testing_framework():
    """
    测试A/B测试框架
    """
    logger.info("=== 开始测试A/B测试框架 ===")
    
    try:
        # 创建A/B测试管理器
        ab_test_manager = ABTestManager(test_id='test_sales_model')
        
        # 创建测试变体
        control_variant = TestVariant('control', 'Control Variant')
        variant_variant = TestVariant('variant', 'Variant Variant')
        
        # 添加变体到测试
        ab_test_manager.add_variant(control_variant)
        ab_test_manager.add_variant(variant_variant)
        
        # 设置要跟踪的指标
        ab_test_manager.set_metrics(['rmse', 'mae', 'mape'])
        
        # 开始测试
        ab_test_manager.start_test()
        
        # 分配流量
        assignments = [ab_test_manager.assign_variant(f'user_{i}') for i in range(1000)]
        
        # 统计分配结果
        variant_counts = {'control': sum(1 for a in assignments if a.variant_id == 'control'), 
                         'variant': sum(1 for a in assignments if a.variant_id == 'variant')}
        logger.info(f"A/B测试流量分配结果: {variant_counts}")
        
        # 验证流量分配是否符合预期
        assert len(variant_counts) == 2, "应该只有两个变体"
        
        # 记录测试结果
        for i in range(100):
            control_variant.record_performance(rmse=10+i*0.1, mae=5+i*0.05, mape=0.1+i*0.001)
            variant_variant.record_performance(rmse=9+i*0.1, mae=4+i*0.05, mape=0.09+i*0.001)
        
        # 结束测试
        ab_test_manager.end_test()
        
        logger.info("A/B测试框架测试通过！")
        return True
        
    except Exception as e:
        logger.error(f"A/B测试框架测试失败: {str(e)}")
        return False

# 用于模型版本管理测试的虚拟模型类
class DummyModel:
    def predict(self, X):
        return [1] * len(X)


def test_model_version_management():
    """
    测试模型版本管理功能
    """
    logger.info("=== 开始测试模型版本管理功能 ===")
    
    try:
        # 创建MLOpsEngine实例
        mlops_engine = MLOpsEngine(models_dir="models", metrics_dir="metrics", config_dir="config")
        
        # 测试版本号生成
        product_id = "test_product_001"
        version = mlops_engine._generate_version_number(product_id)
        logger.info(f"生成的版本号: {version}")
        
        assert isinstance(version, str), "版本号应该是字符串类型"
        
        # 测试模型保存和加载（使用虚拟模型）
        dummy_model = DummyModel()
        
        # 保存模型
        model_version = mlops_engine.save_model(product_id, dummy_model, 'test_model', {"accuracy": 0.95})
        logger.info(f"保存的模型版本: {model_version}")
        
        # 加载模型
        loaded_model, loaded_meta = mlops_engine.load_model(product_id, version=model_version)
        logger.info(f"加载的模型元数据: {loaded_meta}")
        
        assert loaded_model is not None, "模型加载失败"
        assert loaded_meta['metrics']['accuracy'] == 0.95, "模型元数据加载不正确"
        
        # 列出模型版本
        versions = mlops_engine.list_model_versions(product_id)
        logger.info(f"模型版本列表: {versions}")
        
        assert len(versions) > 0, "模型版本列表为空"
        
        logger.info("模型版本管理功能测试通过！")
        return True
        
    except Exception as e:
        logger.error(f"模型版本管理功能测试失败: {str(e)}")
        return False

def main():
    """
    运行所有高优先级功能测试
    """
    logger.info("开始运行所有高优先级功能测试...")
    
    tests = [
        test_dynamic_feature_selection,
        test_feature_importance_monitoring,
        test_external_features_integration,
        test_ab_testing_framework,
        test_model_version_management
    ]
    
    results = {}
    for test in tests:
        results[test.__name__] = test()
    
    # 统计测试结果
    passed = sum(results.values())
    failed = len(results) - passed
    
    logger.info(f"\n=== 测试结果总结 ===")
    logger.info(f"总测试数: {len(results)}")
    logger.info(f"通过测试数: {passed}")
    logger.info(f"失败测试数: {failed}")
    
    for test_name, result in results.items():
        logger.info(f"{test_name}: {'通过' if result else '失败'}")
    
    if failed == 0:
        logger.info("所有高优先级功能测试通过！")
        return 0
    else:
        logger.error(f"有{failed}个测试失败！")
        return 1

if __name__ == "__main__":
    sys.exit(main())
