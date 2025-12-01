from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import os
from datetime import datetime
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="Data Service",
    description="数据加载、预处理和特征准备服务",
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

class DataProcessor:
    """
    数据处理类，用于处理需求数据、预测数据和实际到货数据
    """
    
    def __init__(self):
        pass
    
    def load_data(self, file_path):
        """
        加载数据文件
        """
        try:
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            raise HTTPException(status_code=500, detail=f"加载数据失败: {e}")
    
    def preprocess_data(self, df, target_column='demand', date_column='date'):
        """
        预处理数据，包括缺失值处理、特征工程等
        """
        # 复制数据以避免修改原始数据
        processed_df = df.copy()
        
        # 处理日期列
        if date_column in processed_df.columns:
            processed_df[date_column] = pd.to_datetime(processed_df[date_column])
            processed_df.set_index(date_column, inplace=True)
        
        # 处理缺失值
        processed_df = processed_df.fillna(0)
        
        # 添加时间特征
        if hasattr(processed_df.index, 'month'):
            processed_df['month'] = processed_df.index.month
            processed_df['quarter'] = processed_df.index.quarter
            processed_df['year'] = processed_df.index.year
        
        return processed_df
    
    def split_data(self, df, test_size=0.2):
        """
        分割训练集和测试集
        """
        # 假设最后一列是目标变量
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        split_idx = int(len(df) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def extract_sku_location_combinations(self, df, sku_column='item_id', location_column='location_id'):
        """
        从数据中提取SKU×仓库组合
        """
        combinations = df[[sku_column, location_column]].drop_duplicates().values.tolist()
        return combinations
    
    def get_demand_series_by_sku_location(self, df, sku_id, location_id, sku_column='item_id', location_column='location_id', demand_column='demand'):
        """
        获取指定SKU×仓库的需求序列
        """
        filtered_df = df[(df[sku_column] == sku_id) & (df[location_column] == location_id)]
        demand_series = filtered_df[demand_column].tolist()
        return demand_series
    
    def prepare_demand_data_for_features(self, demand_df, sku_column='item_id', location_column='location_id', demand_column='demand'):
        """
        准备用于特征更新的需求数据
        """
        demand_data = {}
        
        # 提取所有SKU×仓库组合
        combinations = self.extract_sku_location_combinations(demand_df, sku_column, location_column)
        
        # 为每个组合获取需求序列
        for sku_id, location_id in combinations:
            if sku_id not in demand_data:
                demand_data[sku_id] = {}
            
            demand_series = self.get_demand_series_by_sku_location(
                demand_df, sku_id, location_id, sku_column, location_column, demand_column
            )
            demand_data[sku_id][location_id] = demand_series
        
        return demand_data
    
    def compare_demand(self, actual_demand, predicted_demand):
        """
        比较实际需求和预测需求
        """
        # 计算误差指标
        metrics = {
            'mae': float(np.mean(np.abs(actual_demand - predicted_demand))),
            'mse': float(np.mean((actual_demand - predicted_demand) ** 2)),
            'rmse': float(np.sqrt(np.mean((actual_demand - predicted_demand) ** 2))),
            'mape': float(np.mean(np.abs((actual_demand - predicted_demand) / actual_demand)) * 100)
        }
        
        return metrics

# 初始化数据处理器
processor = DataProcessor()

@app.get("/")
def root():
    """服务根路径"""
    return {
        "message": "Data Service",
        "version": "1.0.0",
        "endpoints": [
            "/health",
            "/api/data/load",
            "/api/data/preprocess",
            "/api/data/combinations",
            "/api/data/demand-series",
            "/api/data/demand-features"
        ]
    }

@app.get("/health")
def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "service": "Data Service",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/data/load")
def load_data(file_name: str = Query(..., description="数据文件名")):
    """
    加载数据文件
    """
    file_path = os.path.join(DATA_DIR, file_name)
    df = processor.load_data(file_path)
    return df.to_dict(orient="records")

@app.post("/api/data/preprocess")
def preprocess_data(data: dict, target_column: str = "demand", date_column: str = "date"):
    """
    预处理数据
    """
    try:
        df = pd.DataFrame(data)
        processed_df = processor.preprocess_data(df, target_column, date_column)
        return processed_df.reset_index().to_dict(orient="records")
    except Exception as e:
        logger.error(f"预处理数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"预处理数据失败: {e}")

@app.post("/api/data/combinations")
def extract_combinations(data: dict, sku_column: str = "item_id", location_column: str = "location_id"):
    """
    提取SKU×仓库组合
    """
    try:
        df = pd.DataFrame(data)
        combinations = processor.extract_sku_location_combinations(df, sku_column, location_column)
        return combinations
    except Exception as e:
        logger.error(f"提取组合失败: {e}")
        raise HTTPException(status_code=500, detail=f"提取组合失败: {e}")

@app.post("/api/data/demand-series")
def get_demand_series(data: dict, sku_id: str, location_id: str, 
                      sku_column: str = "item_id", location_column: str = "location_id", 
                      demand_column: str = "demand"):
    """
    获取指定SKU×仓库的需求序列
    """
    try:
        df = pd.DataFrame(data)
        demand_series = processor.get_demand_series_by_sku_location(
            df, sku_id, location_id, sku_column, location_column, demand_column
        )
        return {"demand_series": demand_series}
    except Exception as e:
        logger.error(f"获取需求序列失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取需求序列失败: {e}")

@app.post("/api/data/demand-features")
def prepare_demand_features(data: dict, sku_column: str = "item_id", 
                           location_column: str = "location_id", demand_column: str = "demand"):
    """
    准备用于特征更新的需求数据
    """
    try:
        df = pd.DataFrame(data)
        demand_data = processor.prepare_demand_data_for_features(
            df, sku_column, location_column, demand_column
        )
        return demand_data
    except Exception as e:
        logger.error(f"准备需求特征数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"准备需求特征数据失败: {e}")

@app.post("/api/data/compare")
def compare_demand(actual_data: list, predicted_data: list):
    """
    比较实际需求和预测需求
    """
    try:
        actual_demand = np.array(actual_data)
        predicted_demand = np.array(predicted_data)
        metrics = processor.compare_demand(actual_demand, predicted_demand)
        return metrics
    except Exception as e:
        logger.error(f"比较需求失败: {e}")
        raise HTTPException(status_code=500, detail=f"比较需求失败: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
