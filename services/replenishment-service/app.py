from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import requests
import os
import json
from datetime import datetime
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="Replenishment Service",
    description="库存补货综合服务，整合数据处理、特征计算、需求预测和库存优化",
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

# 服务配置
SERVICES_CONFIG = {
    "data_service": os.getenv("DATA_SERVICE_URL", "http://data-service:8001"),
    "feature_service": os.getenv("FEATURE_SERVICE_URL", "http://feature-service:8002"),
    "forecast_service": os.getenv("FORECAST_SERVICE_URL", "http://forecast-service:8003"),
    "optimization_service": os.getenv("OPTIMIZATION_SERVICE_URL", "http://optimization-service:8004")
}

class ReplenishmentService:
    """
    库存补货综合服务类
    """
    
    def __init__(self):
        self.services = SERVICES_CONFIG
    
    def _call_service(self, service_name, endpoint, method="GET", data=None, params=None):
        """
        调用其他微服务
        """
        service_url = self.services.get(service_name)
        if not service_url:
            raise ValueError(f"服务 {service_name} 未配置")
        
        url = f"{service_url}{endpoint}"
        headers = {"Content-Type": "application/json"}
        
        try:
            if method == "GET":
                response = requests.get(url, params=params, headers=headers, timeout=30)
            elif method == "POST":
                response = requests.post(url, json=data, headers=headers, timeout=30)
            else:
                raise ValueError(f"不支持的HTTP方法: {method}")
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"调用服务 {service_name} 失败: {e}")
            raise HTTPException(status_code=503, detail=f"服务 {service_name} 不可用: {str(e)}")
    
    def run_full_replenishment(self, request_data):
        """
        运行完整的补货流程
        """
        try:
            # 1. 加载和预处理数据
            logger.info("1. 加载和预处理数据...")
            data_result = self._call_service(
                "data_service", "/api/data/load", 
                method="POST",
                data=request_data.get("data_config", {})
            )
            
            # 2. 获取SKU×仓库组合
            logger.info("2. 获取SKU×仓库组合...")
            sku_location_result = self._call_service(
                "data_service", "/api/data/sku-locations",
                method="POST",
                data={"processed_data": data_result.get("processed_data", {})}
            )
            sku_locations = sku_location_result.get("sku_locations", [])
            
            # 3. 对每个SKU×仓库运行预测和优化
            results = []
            for sku_location in sku_locations:
                sku_id = sku_location.get("sku_id")
                location_id = sku_location.get("location_id")
                
                logger.info(f"3. 处理SKU: {sku_id}, 仓库: {location_id}...")
                
                # 3.1 获取需求序列
                demand_result = self._call_service(
                    "data_service", "/api/data/demand-series",
                    method="POST",
                    data={
                        "sku_id": sku_id,
                        "location_id": location_id,
                        "processed_data": data_result.get("processed_data", {})
                    }
                )
                demand_series = demand_result.get("demand_series", [])
                
                # 3.2 更新特征
                logger.info(f"   3.2 更新特征...")
                feature_result = self._call_service(
                    "feature_service", "/api/features/update",
                    method="POST",
                    params={
                        "sku_id": sku_id,
                        "location_id": location_id,
                        "demand_series": demand_series
                    }
                )
                model_tag = feature_result.get("model_selection_tag", "stable_low_variability")
                
                # 3.3 运行预测
                logger.info(f"   3.3 运行预测...")
                forecast_result = self._call_service(
                    "forecast_service", "/api/forecast/run",
                    method="POST",
                    data={
                        "product_id": sku_id,
                        "location_id": location_id,
                        "demand_series": demand_series,
                        "model_tag": model_tag
                    }
                )
                
                # 3.4 运行优化
                logger.info(f"   3.4 运行优化...")
                optimization_data = {
                    "forecast_demands": forecast_result.get("predictions", []),
                    "current_inventory": request_data.get("current_inventory", {}).get(sku_id, 0),
                    "lead_times": request_data.get("lead_times", {}).get(sku_id, 1),
                    "costs": request_data.get("costs", {}),
                    "constraints": request_data.get("constraints", {})
                }
                optimization_result = self._call_service(
                    "optimization_service", "/api/optimization/run",
                    method="POST",
                    data=optimization_data
                )
                
                # 3.5 生成采购订单
                logger.info(f"   3.5 生成采购订单...")
                po_result = self._call_service(
                    "optimization_service", "/api/optimization/generate-orders",
                    method="POST",
                    data={
                        "optimization_results": optimization_result,
                        "current_period": 0
                    }
                )
                
                # 3.6 保存结果
                results.append({
                    "sku_id": sku_id,
                    "location_id": location_id,
                    "forecast_result": forecast_result,
                    "optimization_result": optimization_result,
                    "purchase_orders": po_result.get("purchase_orders", []),
                    "model_tag": model_tag,
                    "features": feature_result.get("features", {})
                })
            
            logger.info("4. 补货流程完成")
            return {
                "status": "success",
                "results": results,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"完整补货流程失败: {e}")
            raise HTTPException(status_code=500, detail=f"完整补货流程失败: {str(e)}")
    
    def get_system_status(self):
        """
        获取系统状态
        """
        statuses = {}
        for service_name in self.services.keys():
            try:
                health_result = self._call_service(service_name, "/health")
                statuses[service_name] = {
                    "status": "healthy",
                    "details": health_result
                }
            except Exception as e:
                statuses[service_name] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
        
        return {
            "status": "healthy" if all(s["status"] == "healthy" for s in statuses.values()) else "degraded",
            "services": statuses,
            "timestamp": datetime.now().isoformat()
        }

# 初始化综合服务
replenishment_service = ReplenishmentService()

@app.get("/")
def root():
    """服务根路径"""
    return {
        "message": "Replenishment Service",
        "version": "1.0.0",
        "endpoints": [
            "/health",
            "/api/replenishment/full",
            "/api/system/status"
        ]
    }

@app.get("/health")
def health_check():
    """
    健康检查
    """
    return {
        "status": "healthy",
        "service": "Replenishment Service",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/replenishment/full")
def run_full_replenishment(request_data: dict):
    """
    运行完整的补货流程
    """
    return replenishment_service.run_full_replenishment(request_data)

@app.get("/api/system/status")
def get_system_status():
    """
    获取系统状态
    """
    return replenishment_service.get_system_status()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
