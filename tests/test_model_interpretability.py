#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试模型可解释性功能

该脚本用于验证模型解释器、业务规则生成器和特征贡献分析等可解释性功能是否正常工作。
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from forecast_models import ForecastModelSelector
from interpretability import ModelInterpreter, BusinessRuleGenerator, MILPInterpreter


def test_model_interpreter():
    """
    测试ModelInterpreter类的功能
    """
    print("\n=== 测试ModelInterpreter ===")
    
    try:
        # 创建合成回归数据
        X, y = make_regression(n_samples=100, n_features=5, n_informative=3, random_state=42)
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]
        
        # 初始化解释器
        interpreter = ModelInterpreter()
        
        # 测试随机森林模型
        from sklearn.ensemble import RandomForestRegressor
        rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
        rf_model.fit(X_train, y_train)
        
        print("1. 测试随机森林模型解释...")
        explanation_data, explanation_id, file_path = interpreter.generate_model_explanation(
            rf_model, X_train, X_test, y_train, feature_names=[f'feature_{i}' for i in range(5)]
        )
        print(f"   解释ID: {explanation_id}")
        print(f"   结果文件: {file_path}")
        print(f"   解释数据包含: {list(explanation_data.keys())}")
        
        # 测试XGBoost模型
        from xgboost import XGBRegressor
        xgb_model = XGBRegressor(n_estimators=50, random_state=42)
        xgb_model.fit(X_train, y_train)
        
        print("2. 测试XGBoost模型解释...")
        explanation_data, explanation_id, file_path = interpreter.generate_model_explanation(
            xgb_model, X_train, X_test, y_train, feature_names=[f'feature_{i}' for i in range(5)]
        )
        print(f"   解释ID: {explanation_id}")
        print(f"   结果文件: {file_path}")
        
        # 测试特征重要性
        print("3. 测试特征重要性...")
        feature_importance = interpreter.get_feature_importance(rf_model, X_test, y_test)
        print("   特征重要性:")
        print(feature_importance)
        
        print("✅ ModelInterpreter测试通过")
        return True
    except Exception as e:
        print(f"❌ ModelInterpreter测试失败: {e}")
        return False


def test_business_rule_generator():
    """
    测试BusinessRuleGenerator类的功能
    """
    print("\n=== 测试BusinessRuleGenerator ===")
    
    try:
        # 创建合成回归数据
        X, y = make_regression(n_samples=100, n_features=5, n_informative=3, random_state=42)
        
        # 初始化规则生成器
        rule_generator = BusinessRuleGenerator()
        
        # 测试随机森林模型
        from sklearn.ensemble import RandomForestRegressor
        rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
        rf_model.fit(X, y)
        
        print("1. 生成业务规则...")
        rules = rule_generator.generate_business_rules(
            rf_model, X, y, feature_names=[f'feature_{i}' for i in range(5)], top_n=5
        )
        print(f"   生成规则数量: {len(rules)}")
        for i, rule in enumerate(rules[:3]):
            print(f"   规则{i+1}: {rule}")
        
        # 测试规则简化
        print("2. 测试规则简化...")
        simplified_rules = rule_generator.simplify_rules(rules)
        print(f"   简化后规则数量: {len(simplified_rules)}")
        
        # 测试规则报告生成
        print("3. 测试规则报告生成...")
        rule_report = rule_generator.generate_rule_report(rules, "random_forest")
        print(f"   报告包含: {list(rule_report.keys())}")
        
        print("✅ BusinessRuleGenerator测试通过")
        return True
    except Exception as e:
        print(f"❌ BusinessRuleGenerator测试失败: {e}")
        return False


def test_forecast_model_selector_interpretability():
    """
    测试ForecastModelSelector的可解释性功能集成
    """
    print("\n=== 测试ForecastModelSelector可解释性集成 ===")
    
    try:
        # 创建合成数据
        X, y = make_regression(n_samples=50, n_features=4, n_informative=2, random_state=42)
        
        # 初始化模型选择器
        selector = ForecastModelSelector()
        
        print("1. 测试模型选择与解释生成...")
        best_model, best_model_name, best_score = selector.select_best_model(X, y, "test_product_interpret")
        
        print(f"   最佳模型: {best_model_name}")
        print(f"   最佳分数: {best_score:.4f}")
        
        # 检查是否生成了解释
        if "test_product_interpret" in selector.model_selections:
            model_selection = selector.model_selections["test_product_interpret"]
            print(f"   模型选择记录包含: {list(model_selection.keys())}")
            
            if "explanation" in model_selection:
                print("   ✅ 生成了模型解释")
            
            if "business_rules" in model_selection:
                rules = model_selection["business_rules"]
                print(f"   ✅ 生成了业务规则: {len(rules.get('rules', []))} 条")
            
            if "feature_contribution" in model_selection:
                contribution = model_selection["feature_contribution"]
                print(f"   ✅ 生成了特征贡献度: {len(contribution.get('feature_contribution', []))} 个特征")
        
        print("✅ ForecastModelSelector可解释性集成测试通过")
        return True
    except Exception as e:
        print(f"❌ ForecastModelSelector可解释性集成测试失败: {e}")
        return False


def test_prediction_visualization():
    """
    测试预测可视化功能
    """
    print("\n=== 测试预测可视化 ===")
    
    try:
        # 创建合成数据
        y_true = np.random.randint(10, 100, size=20)
        y_pred = y_true + np.random.randint(-5, 5, size=20)
        
        # 初始化模型选择器
        selector = ForecastModelSelector()
        
        print("1. 测试预测可视化生成...")
        plot_path = selector.visualize_prediction(y_true, y_pred, "test_model", "test_product_vis")
        print(f"   可视化图像保存到: {plot_path}")
        
        # 检查文件是否存在
        if os.path.exists(plot_path):
            print("   ✅ 可视化文件已生成")
        
        print("✅ 预测可视化测试通过")
        return True
    except Exception as e:
        print(f"❌ 预测可视化测试失败: {e}")
        return False


def main():
    """
    主测试函数
    """
    print("🔍 开始测试模型可解释性功能...")
    
    # 运行所有测试
    results = []
    results.append("ModelInterpreter: " + ("✅ 通过" if test_model_interpreter() else "❌ 失败"))
    results.append("BusinessRuleGenerator: " + ("✅ 通过" if test_business_rule_generator() else "❌ 失败"))
    results.append("ForecastModelSelector集成: " + ("✅ 通过" if test_forecast_model_selector_interpretability() else "❌ 失败"))
    results.append("预测可视化: " + ("✅ 通过" if test_prediction_visualization() else "❌ 失败"))
    
    # 清理测试生成的文件
    print("\n🧹 清理测试文件...")
    test_products = ["test_product_interpret", "test_product_vis"]
    model_dir = "models"
    if os.path.exists(model_dir):
        for file in os.listdir(model_dir):
            if any(prod in file for prod in test_products):
                os.remove(os.path.join(model_dir, file))
    
    # 输出测试总结
    print("\n📋 测试总结:")
    for result in results:
        print(f"   {result}")
    
    # 检查是否所有测试都通过
    if all("✅" in result for result in results):
        print("\n🎉 所有测试通过！模型可解释性功能正常工作。")
        return 0
    else:
        print("\n⚠️  部分测试失败，请检查错误信息。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
