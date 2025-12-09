#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
在其他地方引用和使用供应链智能补货系统模型的示例脚本

本脚本展示了两种引用模型的方式：
1. 直接使用joblib加载模型文件
2. 使用项目现有的类和方法加载模型

使用方法：
python use_model_example.py --product_id 1
"""

import os
import sys
import joblib
import json
import argparse

# 设置项目根目录，确保可以导入项目模块
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_root)


def load_model_directly(product_id, model_type='forecast'):
    """
    直接使用joblib加载模型文件
    
    Args:
        product_id: 产品ID
        model_type: 模型类型 ('forecast' 或 'mlops')
        
    Returns:
        model: 加载的模型对象
        model_name: 模型名称
        metadata: 模型元数据
    """
    
    if model_type == 'forecast':
        # 加载预测模型（.joblib格式）
        model_dir = os.path.join(project_root, 'models')
        
        # 查找对应产品的模型文件
        model_files = [f for f in os.listdir(model_dir) 
                      if f.startswith(f'{product_id}_') and f.endswith('.joblib')]
        
        if not model_files:
            print(f"未找到产品 {product_id} 的预测模型文件")
            return None, None, None
        
        # 使用找到的第一个模型文件
        model_path = os.path.join(model_dir, model_files[0])
        print(f"直接加载模型文件: {model_path}")
        
        # 加载模型数据
        model_data = joblib.load(model_path)
        return model_data['model'], model_data['model_name'], model_data['metadata']
    
    elif model_type == 'mlops':
        # 加载MLOps模型（.pkl格式）
        model_dir = os.path.join(project_root, 'models')
        model_path = os.path.join(model_dir, f'model_{product_id}.pkl')
        metadata_path = os.path.join(model_dir, f'model_{product_id}_metadata.json')
        
        if not os.path.exists(model_path) or not os.path.exists(metadata_path):
            print(f"未找到产品 {product_id} 的MLOps模型文件")
            return None, None, None
        
        print(f"直接加载MLOps模型文件: {model_path}")
        
        # 加载模型和元数据
        model = joblib.load(model_path)
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return model, metadata.get('model_name', 'unknown'), metadata
    
    else:
        print(f"不支持的模型类型: {model_type}")
        return None, None, None


def load_model_using_project_class(product_id):
    """
    使用项目现有的ForecastEngine类加载模型
    
    Args:
        product_id: 产品ID
        
    Returns:
        model: 加载的模型对象
        model_name: 模型名称
        metadata: 模型元数据
    """
    try:
        from src.forecast.forecast_models import ForecastEngine
        
        # 创建ForecastEngine实例
        forecast_engine = ForecastEngine(model_dir=os.path.join(project_root, 'models'))
        
        # 使用项目提供的方法加载模型
        print(f"使用ForecastEngine类加载产品 {product_id} 的模型")
        model, model_name, metadata = forecast_engine.load_model(product_id)
        
        return model, model_name, metadata
    
    except ImportError as e:
        print(f"导入项目模块失败: {e}")
        print("请确保项目根目录已添加到Python路径")
        return None, None, None


def make_prediction(model, model_name, input_data):
    """
    使用加载的模型进行预测
    
    Args:
        model: 加载的模型对象
        model_name: 模型名称
        input_data: 输入数据（根据模型类型调整）
        
    Returns:
        prediction: 预测结果
    """
    
    try:
        if model is None:
            print("模型未加载，无法进行预测")
            return None
        
        print(f"使用模型 {model_name} 进行预测")
        
        # 根据模型类型进行预测
        if model_name in ['arima', 'holt_winters']:
            # 统计模型预测
            prediction = model.forecast(steps=1)
            return prediction.values[0] if hasattr(prediction, 'values') else prediction[0]
        
        elif model_name in ['croston', 'sba']:
            # 间歇性需求模型预测
            prediction = model.forecast(steps=1)
            return prediction[0]
        
        elif model_name in ['prophet']:
            # Prophet模型预测
            future = model.make_future_dataframe(periods=1, include_history=False)
            forecast = model.predict(future)
            return forecast['yhat'].values[0]
        
        else:
            # 机器学习模型预测
            # 确保输入数据格式正确（需要根据模型训练时的特征数量调整）
            prediction = model.predict([input_data])
            return prediction[0]
    
    except Exception as e:
        print(f"预测失败: {e}")
        return None


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='供应链智能补货系统模型使用示例')
    parser.add_argument('--product_id', type=int, default=1, help='产品ID')
    parser.add_argument('--model_type', type=str, default='forecast', choices=['forecast', 'mlops'], 
                        help='模型类型')
    args = parser.parse_args()
    
    print("=" * 50)
    print("供应链智能补货系统模型使用示例")
    print("=" * 50)
    
    # 方法1：直接使用joblib加载模型
    print("\n1. 直接使用joblib加载模型：")
    print("-" * 30)
    model1, model_name1, metadata1 = load_model_directly(args.product_id, args.model_type)
    
    if model1:
        print(f"模型加载成功！")
        print(f"模型名称: {model_name1}")
        print(f"模型元数据: {json.dumps(metadata1, indent=2, ensure_ascii=False)}")
        
        # 进行预测示例
        # 注意：实际输入数据需要根据模型训练时的特征进行调整
        if args.model_type == 'forecast' and model_name1 not in ['arima', 'holt_winters', 'prophet']:
            # 假设模型需要10个特征
            sample_input = [100, 20, 15, 30, 40, 50, 60, 70, 80, 90]
            prediction = make_prediction(model1, model_name1, sample_input)
            if prediction:
                print(f"预测结果: {prediction}")
    
    # 方法2：使用项目现有的类和方法加载模型
    print("\n2. 使用项目现有的类和方法加载模型：")
    print("-" * 30)
    model2, model_name2, metadata2 = load_model_using_project_class(args.product_id)
    
    if model2:
        print(f"模型加载成功！")
        print(f"模型名称: {model_name2}")
        print(f"模型元数据: {json.dumps(metadata2, indent=2, ensure_ascii=False)}")
    
    print("\n" + "=" * 50)
    print("模型使用示例结束")
    print("=" * 50)


if __name__ == "__main__":
    main()
