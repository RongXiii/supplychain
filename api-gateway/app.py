from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import httpx
import uvicorn
import os
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="供应链智能补货系统 API Gateway",
    description="统一管理所有微服务的API请求",
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
SERVICES = {
    "data": os.getenv("DATA_SERVICE_URL", "http://localhost:8001"),
    "features": os.getenv("FEATURES_SERVICE_URL", "http://localhost:8002"),
    "forecast": os.getenv("FORECAST_SERVICE_URL", "http://localhost:8003"),
    "optimization": os.getenv("OPTIMIZATION_SERVICE_URL", "http://localhost:8004"),
    "monitoring": os.getenv("MONITORING_SERVICE_URL", "http://localhost:8005")
}

# 创建HTTP客户端
client = httpx.AsyncClient(timeout=30.0)

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """添加请求ID中间件"""
    request_id = request.headers.get("X-Request-ID", str(datetime.now().timestamp()))
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """日志记录中间件"""
    start_time = datetime.now()
    response = await call_next(request)
    process_time = datetime.now() - start_time
    logger.info(
        f"Request: {request.method} {request.url.path} | Status: {response.status_code} | "
        f"Time: {process_time.total_seconds():.3f}s | Request-ID: {request.state.request_id}"
    )
    return response

async def proxy_request(service_url: str, request: Request):
    """代理请求到指定服务"""
    try:
        # 构建目标URL
        target_url = f"{service_url}{request.url.path}{request.url.query}"
        
        # 复制请求头
        headers = dict(request.headers)
        if "host" in headers:
            del headers["host"]
        
        # 发送请求
        response = await client.request(
            method=request.method,
            url=target_url,
            headers=headers,
            content=await request.body()
        )
        
        # 返回响应
        return JSONResponse(
            content=response.json(),
            status_code=response.status_code,
            headers=dict(response.headers)
        )
    except httpx.RequestError as e:
        logger.error(f"Service unavailable: {service_url} | Error: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")
    except httpx.HTTPStatusError as e:
        logger.error(f"Service error: {service_url} | Status: {e.response.status_code} | Error: {e}")
        raise HTTPException(status_code=e.response.status_code, detail="Service error")

# API网关路由
@app.api_route("/api/data/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_data(request: Request, path: str):
    """代理数据服务请求"""
    return await proxy_request(SERVICES["data"], request)

@app.api_route("/api/features/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_features(request: Request, path: str):
    """代理特征服务请求"""
    return await proxy_request(SERVICES["features"], request)

@app.api_route("/api/forecast/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_forecast(request: Request, path: str):
    """代理预测服务请求"""
    return await proxy_request(SERVICES["forecast"], request)

@app.api_route("/api/optimize/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_optimization(request: Request, path: str):
    """代理优化服务请求"""
    return await proxy_request(SERVICES["optimization"], request)

@app.api_route("/api/monitor/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_monitoring(request: Request, path: str):
    """代理监控服务请求"""
    return await proxy_request(SERVICES["monitoring"], request)

# 根路径
@app.get("/")
async def root():
    """API网关根路径"""
    return {
        "message": "供应链智能补货系统 API Gateway",
        "version": "1.0.0",
        "services": [
            {"name": "Data Service", "url": f"{SERVICES['data']}"},
            {"name": "Feature Service", "url": f"{SERVICES['features']}"},
            {"name": "Forecast Service", "url": f"{SERVICES['forecast']}"},
            {"name": "Optimization Service", "url": f"{SERVICES['optimization']}"},
            {"name": "Monitoring Service", "url": f"{SERVICES['monitoring']}"}
        ]
    }

# 健康检查
@app.get("/health")
async def health_check():
    """健康检查"""
    # 检查各个服务的健康状态
    health_status = {}
    for service_name, service_url in SERVICES.items():
        try:
            response = await client.get(f"{service_url}/health")
            health_status[service_name] = {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "status_code": response.status_code
            }
        except Exception as e:
            health_status[service_name] = {
                "status": "unhealthy",
                "error": str(e)
            }
    
    # 整体状态
    overall_status = "healthy" if all(s["status"] == "healthy" for s in health_status.values()) else "unhealthy"
    
    return {
        "status": overall_status,
        "services": health_status,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
