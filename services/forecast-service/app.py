from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import logging
import sys

# 添加src目录到系统路径，以便导入自定义模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + '/src')

from forecast_models import ForecastModelSelector
from mlops_engine import MLOpsEngine

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="Forecast Service",
    description="需求预测服务，支持多种预测模型和模型选择",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 数据目录配置
DATA_DIR = os.getenv("DATA_DIR", "/app/data")
MODELS_DIR = os.getenv("MODELS_DIR", "/app/models")

class ForecastService:
    """
    预测服务类，封装预测相关功能
    """
    
    def __init__(self):
        # 初始化模型选择器
        self.model_selector = ForecastModelSelector()
        # 初始化MLOps引擎
        self.mlops_engine = MLOpsEngine(models_dir=MODELS_DIR)
    
    def preprocess_data(self, product_data):
        """
        预处理产品数据
        """
        # 这里可以根据实际需求扩展预处理逻辑
        # 目前简化处理，直接返回数据
        return product_data
    
    def split_data(self, processed_data):
        """
        分割训练集和测试集
        """
        # 简化实现，实际应该根据时间序列特性分割
        demand_series = processed_data.get('demand_series', [])
        if len(demand_series) < 12:
            return [], [], [], []
        
        # 使用前80%作为训练集，后20%作为测试集
        split_point = int(len(demand_series) * 0.8)
        train_data = demand_series[:split_point]
        test_data = demand_series[split_point:]
        
        # 为简化，返回相同结构
        return train_data, test_data, train_data, test_data
    
    def detect_drift(self, y_train, y_test, product_id):
        """
        检测数据漂移
        """
        if len(y_train) < 10 or len(y_test) < 10:
            return {"drift_detected": False, "confidence": 0.0}
        
        # 使用MLOps引擎检测漂移
        return self.mlops_engine.detect_drift(y_train, y_test, product_id)
    
    def run_forecast(self, product_data):
        """
        运行预测流程
        """
        try:
            product_id = product_data.get('product_id', 'default')
            location_id = product_data.get('location_id', 1)
            demand_series = product_data.get('demand_series', [])
            model_tag = product_data.get('model_tag', 'stable_low_variability')
            
            if not demand_series:
                raise ValueError("需求序列不能为空")
            
            # 预处理数据
            processed_data = self.preprocess_data(product_data)
            
            # 分割训练集和测试集
            X_train, X_test, y_train, y_test = self.split_data(processed_data)
            
            if not X_train or not y_train:
                raise ValueError("训练数据不足")
            
            # 漂移检测
            drift_result = self.detect_drift(y_train, y_test, product_id)
            logger.info(f"产品 {product_id} 漂移检测结果: {drift_result}")
            
            # 选择最佳模型
            best_model, model_name, best_score = self.model_selector.select_best_model(
                X_train, y_train, product_id, model_tag=model_tag
            )
            
            # 评估模型
            test_metrics = self.model_selector.evaluate_model(best_model, model_name, X_test, y_test)
            
            # 生成预测
            if model_name in ['arima', 'holt_winters']:
                # 统计模型
                y_pred = self.model_selector.predict(best_model, model_name, y_test)
            else:
                # 机器学习模型
                y_pred = self.model_selector.predict(best_model, model_name, X_test)
            
            # 计算误差指标
            error_metrics = self.mlops_engine.calculate_error_metrics(y_test, y_pred, product_id)
            
            # 保存模型
            self.mlops_engine.save_model(product_id, best_model, model_name, metrics=error_metrics)
            
            # 进行未来预测
            future_predictions = self.model_selector.forecast(best_model, model_name, X_test)
            
            return {
                'product_id': product_id,
                'model_name': model_name,
                'model_score': best_score,
                'test_metrics': test_metrics,
                'error_metrics': error_metrics,
                'predictions': future_predictions,
                'drift_detection': drift_result
            }
        except Exception as e:
            logger.error(f"预测失败: {e}")
            raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")
    
    def batch_forecast(self, batch_data):
        """
        批量预测
        """
        results = []
        for product_data in batch_data:
            try:
                result = self.run_forecast(product_data)
                results.append({
                    "status": "success",
                    "data": result
                })
            except Exception as e:
                results.append({
                    "status": "error",
                    "product_id": product_data.get('product_id', 'unknown'),
                    "error": str(e)
                })
        return results
    
    def get_model_info(self, product_id):
        """
        获取模型信息
        """
        model_info = self.mlops_engine.get_model_info(product_id)
        if not model_info:
            raise HTTPException(status_code=404, detail=f"模型 {product_id} 不存在")
        return model_info
    
    def update_model(self, product_id, new_data):
        """
        更新模型
        """
        # 简化实现，实际应该重新训练模型
        try:
            # 获取当前模型信息
            current_model_info = self.mlops_engine.get_model_info(product_id)
            if not current_model_info:
                raise ValueError(f"模型 {product_id} 不存在")
            
            # 使用新数据重新预测
            result = self.run_forecast(new_data)
            return {
                "status": "success",
                "message": f"模型 {product_id} 已更新",
                "new_model_info": result
            }
        except Exception as e:
            logger.error(f"更新模型失败: {e}")
            raise HTTPException(status_code=500, detail=f"更新模型失败: {str(e)}")

# 初始化预测服务
forecast_service = ForecastService()

@app.get("/")
def root():
    """服务根路径"""
    return {
        "message": "Forecast Service",
        "version": "1.0.0",
        "endpoints": [
            "/health",
            "/api/forecast/run",
            "/api/forecast/batch",
            "/api/forecast/model-info",
            "/api/forecast/update-model"
        ]
    }

@app.get("/health")
def health_check():
    """
    健康检查
    """
    return {
        "status": "healthy",
        "service": "Forecast Service",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/forecast/run")
def run_forecast(product_data: dict):
    """
    运行单个产品的预测
    """
    return forecast_service.run_forecast(product_data)

@app.post("/api/forecast/batch")
def batch_forecast(batch_data: list):
    """
    批量预测
    """
    return forecast_service.batch_forecast(batch_data)

@app.get("/api/forecast/model-info")
def get_model_info(product_id: str = Query(..., description="产品ID")):
    """
    获取模型信息
    """
    return forecast_service.get_model_info(product_id)

@app.post("/api/forecast/update-model")
def update_model(product_id: str, new_data: dict):
    """
    更新模型
    """
    return forecast_service.update_model(product_id, new_data)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
