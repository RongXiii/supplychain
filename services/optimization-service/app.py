from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import os
import json
from datetime import datetime
import logging
import sys

# 添加src目录到系统路径，以便导入自定义模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + '/src')

from milp_optimizer import MILPOptimizer

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="Optimization Service",
    description="库存优化服务，基于MILP模型计算最优订货策略",
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

class OptimizationService:
    """
    优化服务类，封装MILP优化相关功能
    """
    
    def __init__(self):
        # 初始化MILP优化器
        self.milp_optimizer = MILPOptimizer()
    
    def run_optimization(self, optimization_data):
        """
        运行优化计算
        """
        try:
            # 解析输入数据
            forecast_demands = np.array(optimization_data.get('forecast_demands', []))
            current_inventory = np.array(optimization_data.get('current_inventory', []))
            lead_times = optimization_data.get('lead_times', [])
            costs = optimization_data.get('costs', {})
            constraints = optimization_data.get('constraints', {})
            
            # 验证输入数据
            if forecast_demands.size == 0:
                raise ValueError("预测需求不能为空")
            if current_inventory.size == 0:
                raise ValueError("当前库存不能为空")
            if len(lead_times) == 0:
                raise ValueError("提前期不能为空")
            
            # 设置默认成本参数
            default_costs = {
                'ordering_cost': [100] * len(current_inventory),
                'holding_cost': [10] * len(current_inventory),
                'shortage_cost': [100] * len(current_inventory),
                'unit_cost': [100] * len(current_inventory)
            }
            default_costs.update(costs)
            costs = default_costs
            
            # 设置默认约束
            default_constraints = {
                'max_order_quantity': [10000] * len(current_inventory),
                'min_order_quantity': [0] * len(current_inventory),
                'safety_stock_target': [0.1] * len(current_inventory),
                'service_level': 0.95
            }
            default_constraints.update(constraints)
            constraints = default_constraints
            
            # 运行优化
            results = self.milp_optimizer.optimize(forecast_demands, current_inventory, lead_times, costs, constraints)
            
            return results
        except Exception as e:
            logger.error(f"优化失败: {e}")
            raise HTTPException(status_code=500, detail=f"优化失败: {str(e)}")
    
    def generate_purchase_orders(self, optimization_results, current_period=0):
        """
        生成采购订单
        """
        try:
            return self.milp_optimizer.generate_purchase_orders(optimization_results, current_period)
        except Exception as e:
            logger.error(f"生成采购订单失败: {e}")
            raise HTTPException(status_code=500, detail=f"生成采购订单失败: {str(e)}")

# 初始化优化服务
optimization_service = OptimizationService()

@app.get("/")
def root():
    """服务根路径"""
    return {
        "message": "Optimization Service",
        "version": "1.0.0",
        "endpoints": [
            "/health",
            "/api/optimization/run",
            "/api/optimization/generate-orders"
        ]
    }

@app.get("/health")
def health_check():
    """
    健康检查
    """
    return {
        "status": "healthy",
        "service": "Optimization Service",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/optimization/run")
def run_optimization(optimization_data: dict):
    """
    运行优化计算
    """
    return optimization_service.run_optimization(optimization_data)

@app.post("/api/optimization/generate-orders")
def generate_purchase_orders(optimization_results: dict, current_period: int = 0):
    """
    生成采购订单
    """
    return optimization_service.generate_purchase_orders(optimization_results, current_period)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
