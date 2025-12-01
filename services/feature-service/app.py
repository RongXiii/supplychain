from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="Feature Service",
    description="特征计算、特征存储和模型选择标签生成服务",
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
FEATURES_DIR = os.getenv("FEATURES_DIR", "/app/features")

class FeatureStore:
    """
    特征存储类，用于管理和计算SKU×仓库维度的统计特征
    """
    
    def __init__(self, features_dir=FEATURES_DIR):
        self.features_dir = features_dir
        self.sku_location_features = {}
        self.model_selection_tags = {}
        self._load_features()
    
    def _load_features(self):
        """
        从文件加载特征数据
        """
        # 加载SKU×仓库特征
        features_file = os.path.join(self.features_dir, "sku_location_features.json")
        if os.path.exists(features_file):
            with open(features_file, 'r') as f:
                self.sku_location_features = json.load(f)
        
        # 加载模型选择标签
        tags_file = os.path.join(self.features_dir, "model_selection_tags.json")
        if os.path.exists(tags_file):
            with open(tags_file, 'r') as f:
                self.model_selection_tags = json.load(f)
    
    def _save_features(self):
        """
        将特征数据保存到文件
        """
        # 确保目录存在
        os.makedirs(self.features_dir, exist_ok=True)
        
        # 保存SKU×仓库特征
        features_file = os.path.join(self.features_dir, "sku_location_features.json")
        with open(features_file, 'w') as f:
            json.dump(self.sku_location_features, f, indent=2)
        
        # 保存模型选择标签
        tags_file = os.path.join(self.features_dir, "model_selection_tags.json")
        with open(tags_file, 'w') as f:
            json.dump(self.model_selection_tags, f, indent=2)
    
    def calculate_cv(self, demand_series):
        """
        计算变异系数（CV）
        """
        if len(demand_series) == 0:
            return 0
        mean_demand = np.mean(demand_series)
        if mean_demand == 0:
            return 0
        std_demand = np.std(demand_series)
        cv = std_demand / mean_demand
        return round(cv, 3)
    
    def calculate_seasonality_strength(self, demand_series):
        """
        计算季节强度
        """
        if len(demand_series) < 12:
            return 0
        
        # 计算月度平均值（假设数据是月度的）
        monthly_avg = np.mean(np.array(demand_series).reshape(-1, 12), axis=0)
        
        # 计算季节强度
        seasonality_strength = np.std(monthly_avg) / np.mean(monthly_avg)
        return round(seasonality_strength, 3)
    
    def calculate_intermittency_index(self, demand_series):
        """
        计算间歇性指数
        """
        if len(demand_series) == 0:
            return 0
        
        # 计算非零需求的比例
        nonzero_demand_ratio = np.mean([1 if x > 0 else 0 for x in demand_series])
        
        # 计算变异系数
        mean_demand = np.mean(demand_series)
        if mean_demand == 0:
            return 0
        std_demand = np.std(demand_series)
        cv = std_demand / mean_demand
        
        # 计算间歇性指数
        intermittency_index = cv * (1 - nonzero_demand_ratio)
        return round(intermittency_index, 3)
    
    def calculate_demand_frequency(self, demand_series):
        """
        计算需求频率
        """
        if len(demand_series) == 0:
            return 0
        nonzero_demand_count = np.sum([1 if x > 0 else 0 for x in demand_series])
        frequency = nonzero_demand_count / len(demand_series)
        return round(frequency, 3)
    
    def calculate_average_demand(self, demand_series):
        """
        计算平均需求
        """
        if len(demand_series) == 0:
            return 0
        avg_demand = np.mean(demand_series)
        return round(avg_demand, 3)
    
    def calculate_trend_strength(self, demand_series):
        """
        计算趋势强度
        """
        if len(demand_series) < 2:
            return 0
        
        # 使用线性回归计算趋势
        x = np.arange(len(demand_series)).reshape(-1, 1)
        y = np.array(demand_series)
        
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(x, y)
        
        # 计算趋势强度
        trend_strength = np.abs(model.coef_[0]) / np.mean(y)
        return round(trend_strength, 3)
    
    def calculate_max_demand(self, demand_series):
        """
        计算最大需求
        """
        if len(demand_series) == 0:
            return 0
        max_demand = np.max(demand_series)
        return round(max_demand, 3)
    
    def calculate_min_demand(self, demand_series):
        """
        计算最小需求
        """
        if len(demand_series) == 0:
            return 0
        min_demand = np.min(demand_series)
        return round(min_demand, 3)
    
    def calculate_skewness(self, demand_series):
        """
        计算偏度
        """
        if len(demand_series) < 3:
            return 0
        skewness = pd.Series(demand_series).skew()
        return round(skewness, 3)
    
    def calculate_kurtosis(self, demand_series):
        """
        计算峰度
        """
        if len(demand_series) < 4:
            return 0
        kurtosis = pd.Series(demand_series).kurtosis()
        return round(kurtosis, 3)
    
    def calculate_features_for_sku_location(self, demand_series):
        """
        计算SKU×仓库的所有特征
        """
        features = {
            "cv": self.calculate_cv(demand_series),
            "seasonality_strength": self.calculate_seasonality_strength(demand_series),
            "intermittency_index": self.calculate_intermittency_index(demand_series),
            "demand_frequency": self.calculate_demand_frequency(demand_series),
            "average_demand": self.calculate_average_demand(demand_series),
            "trend_strength": self.calculate_trend_strength(demand_series),
            "max_demand": self.calculate_max_demand(demand_series),
            "min_demand": self.calculate_min_demand(demand_series),
            "skewness": self.calculate_skewness(demand_series),
            "kurtosis": self.calculate_kurtosis(demand_series),
            "last_updated": datetime.now().isoformat()
        }
        return features
    
    def determine_model_selection_tag(self, features):
        """
        基于特征值确定模型选择标签
        """
        cv = features.get("cv", 0)
        seasonality_strength = features.get("seasonality_strength", 0)
        intermittency_index = features.get("intermittency_index", 0)
        demand_frequency = features.get("demand_frequency", 0)
        trend_strength = features.get("trend_strength", 0)
        
        # 模型选择逻辑
        if intermittency_index > 0.5:
            return "intermittent"
        elif cv > 0.7:
            if seasonality_strength > 0.3:
                return "high_seasonal_high_variability"
            else:
                return "non_seasonal_high_variability"
        elif cv <= 0.7 and cv > 0.3:
            if seasonality_strength > 0.3:
                return "medium_seasonal_medium_variability"
            else:
                return "non_seasonal_medium_variability"
        else:  # cv <= 0.3
            if seasonality_strength > 0.3:
                return "high_seasonal_low_variability"
            elif trend_strength > 0.1:
                return "trending_low_variability"
            else:
                return "stable_low_variability"
    
    def update_features(self, sku_id, location_id, demand_series):
        """
        更新特定SKU×仓库的特征
        """
        key = f"{sku_id}_{location_id}"
        
        # 计算特征
        features = self.calculate_features_for_sku_location(demand_series)
        
        # 更新特征
        self.sku_location_features[key] = features
        
        # 更新模型选择标签
        tag = self.determine_model_selection_tag(features)
        self.model_selection_tags[key] = tag
        
        # 保存到文件
        self._save_features()
        
        return {
            "features": features,
            "model_selection_tag": tag
        }
    
    def batch_update_features(self, demand_data):
        """
        批量更新特征
        """
        results = []
        
        for sku_id, locations in demand_data.items():
            for location_id, demand_series in locations.items():
                result = self.update_features(sku_id, location_id, demand_series)
                results.append({
                    "sku_id": sku_id,
                    "location_id": location_id,
                    **result
                })
        
        return results
    
    def get_features(self, sku_id, location_id):
        """
        获取特定SKU×仓库的特征
        """
        key = f"{sku_id}_{location_id}"
        return self.sku_location_features.get(key, {})
    
    def get_model_selection_tag(self, sku_id, location_id):
        """
        获取特定SKU×仓库的模型选择标签
        """
        key = f"{sku_id}_{location_id}"
        return self.model_selection_tags.get(key, "stable_low_variability")
    
    def generate_feature_report(self):
        """
        生成特征报告
        """
        report = {
            "total_sku_location_combinations": len(self.sku_location_features),
            "model_selection_tag_distribution": {},
            "average_features": {}
        }
        
        # 计算模型选择标签分布
        for tag in self.model_selection_tags.values():
            if tag not in report["model_selection_tag_distribution"]:
                report["model_selection_tag_distribution"][tag] = 0
            report["model_selection_tag_distribution"][tag] += 1
        
        # 计算平均特征值
        if self.sku_location_features:
            features_list = list(self.sku_location_features.values())
            for feature_name in features_list[0].keys():
                if feature_name != "last_updated":
                    avg_value = np.mean([f.get(feature_name, 0) for f in features_list])
                    report["average_features"][feature_name] = round(avg_value, 3)
        
        return report

# 初始化特征存储
feature_store = FeatureStore()

@app.get("/")
def root():
    """服务根路径"""
    return {
        "message": "Feature Service",
        "version": "1.0.0",
        "endpoints": [
            "/health",
            "/api/features/update",
            "/api/features/batch-update",
            "/api/features/get",
            "/api/features/model-tag",
            "/api/features/report"
        ]
    }

@app.get("/health")
def health_check():
    """
    健康检查
    """
    return {
        "status": "healthy",
        "service": "Feature Service",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/features/update")
def update_features(sku_id: str, location_id: str, demand_series: list):
    """
    更新特定SKU×仓库的特征
    """
    try:
        result = feature_store.update_features(sku_id, location_id, demand_series)
        return result
    except Exception as e:
        logger.error(f"更新特征失败: {e}")
        raise HTTPException(status_code=500, detail=f"更新特征失败: {e}")

@app.post("/api/features/batch-update")
def batch_update_features(demand_data: dict):
    """
    批量更新特征
    """
    try:
        results = feature_store.batch_update_features(demand_data)
        return results
    except Exception as e:
        logger.error(f"批量更新特征失败: {e}")
        raise HTTPException(status_code=500, detail=f"批量更新特征失败: {e}")

@app.get("/api/features/get")
def get_features(sku_id: str = Query(..., description="SKU ID"), location_id: str = Query(..., description="仓库ID")):
    """
    获取特定SKU×仓库的特征
    """
    features = feature_store.get_features(sku_id, location_id)
    return features

@app.get("/api/features/model-tag")
def get_model_selection_tag(sku_id: str = Query(..., description="SKU ID"), location_id: str = Query(..., description="仓库ID")):
    """
    获取特定SKU×仓库的模型选择标签
    """
    tag = feature_store.get_model_selection_tag(sku_id, location_id)
    return {"model_selection_tag": tag}

@app.get("/api/features/report")
def generate_feature_report():
    """
    生成特征报告
    """
    report = feature_store.generate_feature_report()
    return report

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
