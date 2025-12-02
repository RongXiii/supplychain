from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import json
import os
from datetime import datetime
import numpy as np
import time

# 添加日志管理器和缓存管理器
from logging_manager import get_logger, log_performance
from cache_manager import cache_manager

# 初始化日志记录器
logger = get_logger('api')

# 创建FastAPI应用
app = FastAPI(
    title="供应链智能补货系统API",
    description="为PowerBI等可视化工具提供数据接口",
    version="2.0.0"
)

# 配置CORS，允许PowerBI访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该限制为PowerBI的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 数据目录和指标目录
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
METRICS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "metrics")

# 辅助函数：加载CSV数据，支持缓存
def load_csv_data(filename, cache_expire=3600):
    """加载CSV数据文件，支持缓存"""
    start_time = time.time()
    cache_key = f"csv_data:{filename}"
    
    logger.info(f"Loading CSV data: {filename}")
    
    # 尝试从缓存获取
    cached_data = cache_manager.get(cache_key, data_type='dataframe')
    if cached_data is not None:
        logger.debug(f"Cache hit for CSV data: {filename}")
        log_performance("load_csv_data", time.time() - start_time, filename=filename, cache_hit=True)
        return cached_data
    
    # 缓存不存在，从文件加载
    file_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(file_path):
        logger.error(f"File not found: {filename}")
        raise HTTPException(status_code=404, detail=f"文件不存在: {filename}")
    
    df = pd.read_csv(file_path)
    logger.debug(f"Loaded CSV data: {filename}, rows: {len(df)}")
    
    # 缓存数据
    cache_manager.set(cache_key, df, expire_seconds=cache_expire)
    logger.debug(f"Cached CSV data: {filename}")
    
    log_performance("load_csv_data", time.time() - start_time, filename=filename, cache_hit=False, rows=len(df))
    return df

# 辅助函数：加载指标数据
def load_metrics_data(product_id):
    """加载指定产品的指标数据"""
    start_time = time.time()
    logger.info(f"Loading metrics data for product: {product_id}")
    
    file_path = os.path.join(METRICS_DIR, f"metrics_{product_id}.json")
    if not os.path.exists(file_path):
        logger.error(f"Metrics file not found for product: {product_id}")
        raise HTTPException(status_code=404, detail=f"产品 {product_id} 的指标文件不存在")
    
    with open(file_path, 'r') as f:
        metrics = json.load(f)
    
    logger.debug(f"Loaded metrics data for product: {product_id}, metrics count: {len(metrics)}")
    log_performance("load_metrics_data", time.time() - start_time, product_id=product_id, metrics_count=len(metrics))
    
    return metrics

# 根路径
@app.get("/")
def root():
    return {
        "message": "供应链智能补货系统API",
        "docs": "/docs",
        "endpoints": [
            "/api/models/performance",
            "/api/inventory/levels",
            "/api/purchase/orders",
            "/api/forecast/data",
            "/api/items",
            "/api/suppliers"
        ]
    }

# 获取所有产品列表
@app.get("/api/items")
def get_items():
    """获取所有产品信息"""
    cache_key = "api:items"
    
    # 尝试从缓存获取
    cached_result = cache_manager.get(cache_key, data_type='json')
    if cached_result is not None:
        return cached_result
    
    df = load_csv_data("items.csv")
    result = df.to_dict(orient="records")
    
    # 缓存结果
    cache_manager.set(cache_key, result, expire_seconds=3600)
    
    return result

# 获取供应商列表
@app.get("/api/suppliers")
def get_suppliers():
    """获取所有供应商信息"""
    cache_key = "api:suppliers"
    
    # 尝试从缓存获取
    cached_result = cache_manager.get(cache_key, data_type='json')
    if cached_result is not None:
        return cached_result
    
    df = load_csv_data("suppliers.csv")
    result = df.to_dict(orient="records")
    
    # 缓存结果
    cache_manager.set(cache_key, result, expire_seconds=3600)
    
    return result

# 获取库存水平数据
@app.get("/api/inventory/levels")
def get_inventory_levels(product_id: str = None):
    """获取库存水平数据，可选按产品ID过滤"""
    cache_key = f"api:inventory:levels:{product_id}" if product_id else "api:inventory:levels:all"
    
    # 尝试从缓存获取
    cached_result = cache_manager.get(cache_key, data_type='json')
    if cached_result is not None:
        return cached_result
    
    df = load_csv_data("inventory_daily.csv")
    
    # 转换日期格式
    df['date'] = pd.to_datetime(df['date'])
    
    # 如果指定了产品ID，过滤数据
    if product_id:
        df = df[df['item_id'].astype(str) == product_id]
    
    # 按日期排序
    df = df.sort_values('date')
    
    # 转换为字典列表
    result = df.to_dict(orient="records")
    
    # 将numpy类型转换为Python原生类型
    for item in result:
        for key, value in item.items():
            if isinstance(value, np.integer):
                item[key] = int(value)
            elif isinstance(value, np.floating):
                item[key] = float(value)
    
    # 缓存结果
    cache_manager.set(cache_key, result, expire_seconds=300)  # 5分钟过期
    
    return result

# 获取采购订单数据
@app.get("/api/purchase/orders")
def get_purchase_orders(product_id: str = None, status: str = None):
    """获取采购订单数据，可选按产品ID或状态过滤"""
    cache_key = f"api:purchase:orders:{product_id}:{status}"
    
    # 尝试从缓存获取
    cached_result = cache_manager.get(cache_key, data_type='json')
    if cached_result is not None:
        return cached_result
    
    df = load_csv_data("purchase_orders.csv")
    
    # 转换日期格式
    date_columns = ['order_date', 'expected_delivery_date']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    
    # 过滤数据
    if product_id:
        df = df[df['item_id'].astype(str) == product_id]
    if status:
        df = df[df['status'].str.lower() == status.lower()]
    
    # 转换为字典列表
    result = df.to_dict(orient="records")
    
    # 将numpy类型转换为Python原生类型
    for item in result:
        for key, value in item.items():
            if isinstance(value, np.integer):
                item[key] = int(value)
            elif isinstance(value, np.floating):
                item[key] = float(value)
    
    # 缓存结果
    cache_manager.set(cache_key, result, expire_seconds=300)  # 5分钟过期
    
    return result

# 获取需求预测数据
@app.get("/api/forecast/data")
def get_forecast_data(product_id: str = None):
    """获取需求预测数据，可选按产品ID过滤"""
    cache_key = f"api:forecast:data:{product_id}" if product_id else "api:forecast:data:all"
    
    # 尝试从缓存获取
    cached_result = cache_manager.get(cache_key, data_type='json')
    if cached_result is not None:
        return cached_result
    
    df = load_csv_data("forecast_output.csv")
    
    # 转换日期格式
    df['date'] = pd.to_datetime(df['date'])
    
    # 如果指定了产品ID，过滤数据
    if product_id:
        df = df[df['item_id'].astype(str) == product_id]
    
    # 按日期排序
    df = df.sort_values('date')
    
    # 转换为字典列表
    result = df.to_dict(orient="records")
    
    # 将numpy类型转换为Python原生类型
    for item in result:
        for key, value in item.items():
            if isinstance(value, np.integer):
                item[key] = int(value)
            elif isinstance(value, np.floating):
                item[key] = float(value)
    
    # 缓存结果
    cache_manager.set(cache_key, result, expire_seconds=1800)  # 30分钟过期
    
    return result

# 获取模型性能数据
@app.get("/api/models/performance")
def get_model_performance(product_id: str = None):
    """获取模型性能数据，可选按产品ID过滤"""
    cache_key = f"api:models:performance:{product_id}" if product_id else "api:models:performance:all"
    
    # 尝试从缓存获取
    cached_result = cache_manager.get(cache_key, data_type='json')
    if cached_result is not None:
        return cached_result
    
    result = []
    
    # 确定要处理的产品ID列表
    if product_id:
        product_ids = [product_id]
    else:
        # 获取所有指标文件
        metrics_files = [f for f in os.listdir(METRICS_DIR) if f.startswith("metrics_") and f.endswith(".json")]
        product_ids = [f.split("_")[1].split(".")[0] for f in metrics_files]
    
    # 加载每个产品的指标数据
    for pid in product_ids:
        try:
            metrics = load_metrics_data(pid)
            # 添加产品ID到每个指标项
            for item in metrics:
                item["product_id"] = pid
                # 转换时间戳
                item["timestamp"] = pd.to_datetime(item["timestamp"]).isoformat()
            result.extend(metrics)
        except HTTPException:
            continue
    
    # 缓存结果
    cache_manager.set(cache_key, result, expire_seconds=1800)  # 30分钟过期
    
    return result

# 获取模型平均性能指标
@app.get("/api/models/performance/average")
def get_model_average_performance():
    """获取所有产品的平均模型性能指标"""
    cache_key = "api:models:performance:average"
    
    # 尝试从缓存获取
    cached_result = cache_manager.get(cache_key, data_type='json')
    if cached_result is not None:
        return cached_result
    
    result = []
    
    # 获取所有指标文件
    metrics_files = [f for f in os.listdir(METRICS_DIR) if f.startswith("metrics_") and f.endswith(".json")]
    product_ids = [f.split("_")[1].split(".")[0] for f in metrics_files]
    
    # 计算每个产品的平均性能
    for pid in product_ids:
        try:
            metrics = load_metrics_data(pid)
            df = pd.DataFrame(metrics)
            
            # 计算平均指标
            avg_metrics = {
                "product_id": pid,
                "average_mape": float(df["mape"].mean()),
                "average_smape": float(df["smape"].mean()),
                "average_rmse": float(df["rmse"].mean()),
                "metric_count": len(df),
                "last_updated": pd.to_datetime(df["timestamp"].max()).isoformat()
            }
            
            result.append(avg_metrics)
        except Exception:
            continue
    
    # 缓存结果
    cache_manager.set(cache_key, result, expire_seconds=3600)  # 1小时过期
    
    return result

# 获取最优计划数据
@app.get("/api/optimal/plan")
def get_optimal_plan(product_id: str = None):
    """获取最优补货计划数据，可选按产品ID过滤"""
    cache_key = f"api:optimal:plan:{product_id}" if product_id else "api:optimal:plan:all"
    
    # 尝试从缓存获取
    cached_result = cache_manager.get(cache_key, data_type='json')
    if cached_result is not None:
        return cached_result
    
    df = load_csv_data("optimal_plan.csv")
    
    # 转换日期格式
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    # 如果指定了产品ID，过滤数据
    if product_id:
        df = df[df['item_id'].astype(str) == product_id]
    
    # 转换为字典列表
    result = df.to_dict(orient="records")
    
    # 将numpy类型转换为Python原生类型
    for item in result:
        for key, value in item.items():
            if isinstance(value, np.integer):
                item[key] = int(value)
            elif isinstance(value, np.floating):
                item[key] = float(value)
    
    # 缓存结果
    cache_manager.set(cache_key, result, expire_seconds=1800)  # 30分钟过期
    
    return result

# 启动API服务
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting API server...")
    # 优化UVicorn配置，提高性能
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=4,  # 启动4个工作进程
        loop="uvloop",  # 使用高性能的uvloop事件循环
        http="httptools"  # 使用高性能的httptools HTTP解析器
    )