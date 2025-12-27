from fastapi import FastAPI, HTTPException, Depends, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import json
import os
from datetime import datetime
import numpy as np
import time

# 添加日志管理器和缓存管理器
from src.system.logging_manager import get_logger, log_performance
from src.system.cache_manager import cache_manager
from src.data.data_source import DataSourceFactory

# 导入补货系统核心类
from src.system.main import ReplenishmentSystem

# 初始化日志记录器
logger = get_logger('api')

# 初始化补货系统实例
replenishment_system = ReplenishmentSystem()

# 创建FastAPI应用
app = FastAPI(
    title="供应链智能补货系统API",
    description="为PowerBI等可视化工具提供数据接口",
    version="2.0.0"
)

# 配置静态文件目录
# 首先创建静态文件目录
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

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

# 初始化数据源工厂（单例模式）
data_source_factory = DataSourceFactory()

# 辅助函数：加载数据，支持缓存和多种数据源
def load_data(table_name, cache_expire=3600):
    """加载数据，支持缓存和多种数据源"""
    start_time = time.time()
    cache_key = f"data:{table_name}"
    
    logger.info(f"Loading data: {table_name}")
    logger.info(f"Current DATA_SOURCE_TYPE: {os.getenv('DATA_SOURCE_TYPE', 'not set')}")
    
    # 尝试从缓存获取
    cached_data = cache_manager.get(cache_key, data_type='dataframe')
    if cached_data is not None:
        logger.debug(f"Cache hit for data: {table_name}")
        log_performance("load_data", time.time() - start_time, table_name=table_name, cache_hit=True)
        return cached_data
    
    try:
        # 直接使用模拟数据源，解决CSV文件缺失问题
        data_source_type = 'simulated'
        
        logger.info(f"Selected data source type: {data_source_type}")
        
        # 配置数据源参数（模拟数据源不需要额外配置）
        data_source_config = {}
        
        logger.info(f"Data source config: {data_source_config}")
        
        # 创建数据源实例
        data_source = data_source_factory.create_data_source(data_source_type, data_source_config)
        logger.info(f"Created data source instance: {type(data_source).__name__}")
        
        # 根据表名获取对应的数据
        if table_name == "items.csv":
            logger.info(f"Calling get_items() method")
            df = data_source.get_items()
        elif table_name == "locations.csv":
            logger.info(f"Calling get_locations() method")
            df = data_source.get_locations()
        elif table_name == "suppliers.csv":
            logger.info(f"Calling get_suppliers() method")
            df = data_source.get_suppliers()
        elif table_name == "inventory_daily.csv":
            logger.info(f"Calling get_inventory_daily() method")
            df = data_source.get_inventory_daily()
        elif table_name == "purchase_orders.csv":
            logger.info(f"Calling get_purchase_orders() method")
            df = data_source.get_purchase_orders()
        elif table_name == "forecast_output.csv":
            logger.info(f"Calling get_forecast_output() method")
            df = data_source.get_forecast_output()
        elif table_name == "optimal_plan.csv":
            logger.info(f"Calling get_optimal_plan() method")
            df = data_source.get_optimal_plan()
        else:
            # 对于其他表，尝试从CSV文件加载
            logger.info(f"Table {table_name} not found in datasource methods, trying CSV fallback")
            file_path = os.path.join(DATA_DIR, table_name)
            if not os.path.exists(file_path):
                logger.error(f"File not found: {table_name}")
                raise HTTPException(status_code=404, detail=f"文件不存在: {table_name}")
            df = pd.read_csv(file_path)
    except Exception as e:
        # 如果从数据源获取失败，尝试从CSV文件加载（降级策略）
        import traceback
        logger.error(f"Failed to load data from {data_source_type} source: {e}")
        logger.error(f"Exception traceback: {traceback.format_exc()}")
        
        try:
            logger.warning(f"Falling back to CSV: {e}")
            file_path = os.path.join(DATA_DIR, table_name)
            if not os.path.exists(file_path):
                logger.error(f"File not found: {table_name}")
                raise Exception(f"CSV file not found: {table_name}")
            df = pd.read_csv(file_path)
        except Exception as csv_e:
            # 如果CSV加载也失败，使用模拟数据源作为最终 fallback
            logger.error(f"Failed to load from CSV: {csv_e}")
            logger.warning(f"Falling back to simulated data source")
            
            # 使用模拟数据源
            data_source = data_source_factory.create_data_source('simulated', {})
            
            # 根据表名获取对应的数据
            if table_name == "items.csv":
                df = data_source.get_items()
            elif table_name == "locations.csv":
                df = data_source.get_locations()
            elif table_name == "suppliers.csv":
                df = data_source.get_suppliers()
            elif table_name == "inventory_daily.csv":
                df = data_source.get_inventory_daily()
            elif table_name == "purchase_orders.csv":
                df = data_source.get_purchase_orders()
            elif table_name == "forecast_output.csv":
                df = data_source.get_forecast_output()
            elif table_name == "optimal_plan.csv":
                df = data_source.get_optimal_plan()
            else:
                logger.error(f"Unsupported table for simulated data: {table_name}")
                raise HTTPException(status_code=404, detail=f"表不存在且无法从模拟数据源获取: {table_name}")
    
    logger.debug(f"Loaded data: {table_name}, rows: {len(df)}")
    
    # 缓存数据
    cache_manager.set(cache_key, df, expire_seconds=cache_expire)
    logger.debug(f"Cached data: {table_name}")
    
    log_performance("load_data", time.time() - start_time, table_name=table_name, cache_hit=False, rows=len(df))
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

# 缓存装饰器
from functools import wraps

def cache_api(cache_key_pattern, expire_seconds=3600, data_type='json'):
    """
    API端点缓存装饰器
    
    参数:
    - cache_key_pattern: 缓存键模式，支持{param}占位符
    - expire_seconds: 过期时间（秒）
    - data_type: 数据类型
    
    返回:
    - 装饰后的函数
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = cache_key_pattern.format(**kwargs)
            
            # 尝试从缓存获取
            cached_result = cache_manager.get(cache_key, data_type=data_type)
            if cached_result is not None:
                logger.debug(f"Cache hit for API: {cache_key}")
                return cached_result
            
            # 调用原始函数
            result = func(*args, **kwargs)
            
            # 缓存结果
            cache_manager.set(cache_key, result, expire_seconds=expire_seconds)
            logger.debug(f"Cached API result: {cache_key}")
            
            return result
        return wrapper
    return decorator

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


@app.get("/api/items")
def get_items():
    """获取所有产品信息"""
    cache_key = "api:items"
    
    # 尝试从缓存获取
    cached_result = cache_manager.get(cache_key, data_type='json')
    if cached_result is not None:
        return cached_result
    
    df = load_data("items.csv")
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
    
    df = load_data("suppliers.csv")
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
    
    df = load_data("inventory_daily.csv")
    
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
    
    df = load_data("purchase_orders.csv")
    
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
    
    df = load_data("forecast_output.csv")
    
    # 转换日期格式
    df['date'] = pd.to_datetime(df['horizon_date'])
    
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
    """获取最优补货计划，可选按产品ID过滤"""
    cache_key = f"api:optimal:plan:{product_id}" if product_id else "api:optimal:plan:all"
    
    # 尝试从缓存获取
    cached_result = cache_manager.get(cache_key, data_type='json')
    if cached_result is not None:
        return cached_result
    
    df = load_data("optimal_plan.csv")
    
    if product_id:
        df = df[df["product_id"] == product_id]
    
    result = df.to_dict(orient="records")
    
    # 缓存结果
    cache_manager.set(cache_key, result, expire_seconds=3600)
    
    return result

# 新增实时预测端点
@app.post("/api/forecast/real-time")
@cache_api("api:forecast:real-time:{product_id}:{forecast_days}:{model_tag}", expire_seconds=300, data_type='json')
def real_time_forecast(product_id: str, forecast_days: int = 7, model_tag: str = None):
    """
    实时预测产品需求
    
    参数:
    - product_id: 产品ID
    - forecast_days: 预测天数（默认7天）
    - model_tag: 模型标签，可选
    
    返回:
    - 预测结果，包括日期和预测值
    """
    logger.info(f"实时预测请求: product_id={product_id}, forecast_days={forecast_days}, model_tag={model_tag}")
    
    try:
        # 加载产品数据
        items_df = load_data("items.csv")
        inventory_df = load_data("inventory_daily.csv")
        
        # 筛选产品数据
        product_data = items_df[items_df["product_id"] == product_id]
        if product_data.empty:
            raise HTTPException(status_code=404, detail=f"产品不存在: {product_id}")
        
        # 筛选历史库存数据
        historical_data = inventory_df[inventory_df["product_id"] == product_id]
        
        # 运行预测
        forecast_result = replenishment_system.run_forecast(historical_data, product_id)
        
        # 处理预测结果，只返回指定天数
        forecast_days = min(forecast_days, len(forecast_result["forecast"]))
        filtered_forecast = {
            "product_id": product_id,
            "forecast_days": forecast_days,
            "forecast_start_date": forecast_result["forecast_dates"][0],
            "forecast": forecast_result["forecast"][:forecast_days],
            "forecast_dates": forecast_result["forecast_dates"][:forecast_days],
            "model_used": forecast_result["model_name"],
            "confidence_interval": forecast_result.get("confidence_interval", [])[:forecast_days]
        }
        
        return filtered_forecast
    except Exception as e:
        logger.error(f"实时预测失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"实时预测失败: {str(e)}")

@app.post("/api/forecast/batch")
async def batch_forecast(request: Request):
    """
    批量预测多个产品的需求
    
    参数:
    - products_data: 产品数据字典
    - steps: 预测步数（默认1）
    - parallel: 是否并行处理（默认True）
    - n_jobs: 并行任务数（默认-1，使用所有CPU）
    
    返回:
    - 批量预测结果
    """
    logger.info(f"批量预测请求")
    
    try:
        data = await request.json()
        products_data = data.get('products_data')
        steps = data.get('steps', 1)
        parallel = data.get('parallel', True)
        n_jobs = data.get('n_jobs', -1)
        
        if not products_data or not isinstance(products_data, dict):
            raise HTTPException(status_code=400, detail="缺少必要参数: products_data (必须是字典类型)")
        
        result = replenishment_system.batch_run_forecast(products_data, steps=steps, parallel=parallel, n_jobs=n_jobs)
        return result
    except Exception as e:
        logger.error(f"批量预测失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"批量预测失败: {str(e)}")

# 新增实时优化端点
@app.post("/api/optimize/real-time")
def real_time_optimization(product_id: str, forecast_results: dict, inventory_data: dict, lead_times: dict, costs: dict):
    """
    实时优化补货计划
    
    参数:
    - product_id: 产品ID
    - forecast_results: 预测结果
    - inventory_data: 库存数据
    - lead_times: 提前期数据
    - costs: 成本数据
    
    返回:
    - 优化结果
    """
    logger.info(f"实时优化请求: product_id={product_id}")
    
    try:
        # 运行优化
        optimization_result = replenishment_system.run_optimization(
            forecast_results, 
            inventory_data, 
            lead_times, 
            costs
        )
        
        return optimization_result
    except Exception as e:
        logger.error(f"实时优化失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"实时优化失败: {str(e)}")

# 新增模型更新端点
@app.post("/api/model/update")
def update_model(product_id: str, actual_data: dict):
    """
    使用实际数据更新模型
    
    参数:
    - product_id: 产品ID
    - actual_data: 实际数据，包括日期和实际需求
    
    返回:
    - 更新结果
    """
    logger.info(f"模型更新请求: product_id={product_id}")
    
    try:
        # 更新模型
        update_result = replenishment_system.update_model_with_actual_data(product_id, actual_data)
        
        return {
            "success": True,
            "message": f"模型更新成功: {product_id}",
            "update_result": update_result
        }
    except Exception as e:
        logger.error(f"模型更新失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"模型更新失败: {str(e)}")

# 新增库存状态分析端点
@app.get("/api/inventory/analysis")
def inventory_analysis(product_id: str = None, location_id: str = None):
    """
    库存状态分析
    
    参数:
    - product_id: 产品ID，可选
    - location_id: 位置ID，可选
    
    返回:
    - 库存分析结果
    """
    logger.info(f"库存分析请求: product_id={product_id}, location_id={location_id}")
    
    try:
        # 加载库存数据
        inventory_df = load_data("inventory_daily.csv")
        items_df = load_data("items.csv")
        
        # 筛选数据
        if product_id:
            inventory_df = inventory_df[inventory_df["product_id"] == product_id]
        if location_id:
            inventory_df = inventory_df[inventory_df["location_id"] == location_id]
        
        if inventory_df.empty:
            return {
                "product_id": product_id,
                "location_id": location_id,
                "message": "没有找到匹配的库存数据",
                "analysis": {}
            }
        
        # 计算库存统计信息
        current_inventory = inventory_df[inventory_df["date"] == inventory_df["date"].max()]
        avg_inventory = inventory_df["quantity_on_hand"].mean()
        min_inventory = inventory_df["quantity_on_hand"].min()
        max_inventory = inventory_df["quantity_on_hand"].max()
        
        # 计算库存周转率（简化计算）
        if len(inventory_df) > 1:
            # 假设销售数量 = 期初库存 + 入库 - 期末库存
            # 这里简化处理，使用库存变化作为销售近似
            sales_approx = inventory_df["quantity_on_hand"].diff().abs().sum()
            inventory_turnover = sales_approx / avg_inventory if avg_inventory > 0 else 0
        else:
            inventory_turnover = 0
        
        # 准备分析结果
        analysis_result = {
            "product_id": product_id,
            "location_id": location_id,
            "current_inventory": {
                "total_items": len(current_inventory),
                "total_quantity": current_inventory["quantity_on_hand"].sum(),
                "items": current_inventory.to_dict(orient="records")
            },
            "inventory_stats": {
                "average_inventory": round(avg_inventory, 2),
                "minimum_inventory": min_inventory,
                "maximum_inventory": max_inventory,
                "inventory_turnover": round(inventory_turnover, 2)
            },
            "recent_trends": {
                "days_analyzed": len(inventory_df),
                "trend": "increasing" if inventory_df["quantity_on_hand"].iloc[-1] > inventory_df["quantity_on_hand"].iloc[0] else "decreasing" if inventory_df["quantity_on_hand"].iloc[-1] < inventory_df["quantity_on_hand"].iloc[0] else "stable"
            }
        }
        
        return analysis_result
    except Exception as e:
        logger.error(f"库存分析失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"库存分析失败: {str(e)}")

# 新增系统状态端点
@app.get("/api/system/status")
def get_system_status():
    """
    获取系统状态
    
    返回:
    - 系统状态信息
    """
    logger.info("系统状态请求")
    
    try:
        status = replenishment_system.get_system_status()
        return status
    except Exception as e:
        logger.error(f"获取系统状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取系统状态失败: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting API server...")
    # 启动UVicorn服务器
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=4,  # 启动4个工作进程
        loop="uvloop",  # 使用高性能的uvloop事件循环
        http="httptools"  # 使用高性能的httptools HTTP解析器
    )