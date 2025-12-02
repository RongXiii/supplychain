# 供应链智能补货系统 - 模型使用指南

## 1. 模型概述

供应链智能补货系统是一个完整的供应链库存优化解决方案，结合了高级预测模型、多种补货策略和MILP（混合整数线性规划）优化，用于优化供应链中的库存管理和补货决策。

**核心功能**：
- 智能模型选择与需求预测
- 多种补货策略（ROP、Order-up-to、混合策略）
- 增强的MILP优化（多仓库调拨、数量折扣考虑）
- 自动补单与采购订单生成
- 模型持续优化与监控
- REST API服务与数据可视化

## 2. 安装指南

### 2.1 环境要求
- Python 3.7+
- 足够的内存和磁盘空间（建议至少8GB RAM）

### 2.2 安装步骤

1. **克隆或下载项目**
   ```bash
   # 克隆项目（如果使用Git）
   git clone <项目地址>
   
   # 或直接下载项目压缩包并解压
   ```

2. **创建并激活虚拟环境**
   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate
   
   # Linux/Mac
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **安装依赖包**
   ```bash
   pip install -r requirements.txt
   ```

## 3. 快速开始

### 3.1 运行主程序

主程序演示了系统的完整功能流程：

```bash
python src/system/main.py
```

运行后，系统将执行以下步骤：
1. 生成或加载模拟数据
2. 对每个产品进行智能模型选择
3. 训练最佳预测模型
4. 进行需求预测
5. 运行MILP优化计算最优订货策略
6. 生成采购订单
7. 演示自动补单流程

### 3.2 启动API服务

API服务提供标准REST接口，支持与其他系统集成：

```bash
python start_api.py
```

服务启动后，可访问以下地址：
- API文档：`http://localhost:8000/docs`
- 数据接口：`http://localhost:8000/api/`

### 3.3 运行DEMO
```bash
python demo.py
```

### 3.4 运行测试
```bash
python -m pytest tests/ -v
```

## 4. 主要功能使用

### 4.1 数据处理

#### 4.1.1 加载和预处理数据
```python
from src.data.data_processor import DataProcessor

# 初始化数据处理器
data_processor = DataProcessor()

# 加载数据
df = data_processor.load_data('./data/sample_demand_data.csv')

# 预处理数据（处理缺失值、特征工程等）
processed_data = data_processor.preprocess_data(df)

# 分割训练集、验证集和测试集
X_train, X_val, X_test, y_train, y_val, y_test = data_processor.split_data(processed_data, target_column='demand')
```

#### 4.1.2 比较实际需求与预测
```python
# 比较实际需求与预测需求
comparison_results = data_processor.compare_actual_vs_forecast(actual_df, forecast_df)
```

### 4.2 预测模型使用

#### 4.2.1 智能模型选择
```python
from src.forecast.forecast_models import ForecastModelSelector

# 初始化模型选择器（use_gpu=True可启用GPU加速）
model_selector = ForecastModelSelector(use_gpu=False)

# 为特定产品选择最佳模型
product_id = 1
selected_model = model_selector.select_best_model(product_id, X_train, y_train)
```

#### 4.2.2 模型训练与预测
```python
# 训练模型
model_selector.train_model(product_id, X_train, y_train)

# 进行预测
forecast = model_selector.predict(product_id, X_test)

# 评估模型性能
metrics = model_selector.evaluate_model(product_id, X_test, y_test)
```

#### 4.2.3 模型更新
```python
# 使用新数据更新模型
model_selector.update_model(product_id, X_new, y_new)
```

#### 4.2.4 模型可解释性
```python
from src.forecast.interpretability import ModelInterpreter

# 初始化模型解释器
interpreter = ModelInterpreter()

# 解释模型预测
interpretation = interpreter.interpret_model(model, X_test)
print(f"Model interpretation: {interpretation}")
```

### 4.3 补货策略与MILP优化

#### 4.3.1 自动补货
```python
from src.replenishment.automated_replenishment import AutomatedReplenishment

# 初始化自动补货系统
replenishment_system = AutomatedReplenishment()

# 生成补货建议
replenishment_advice = replenishment_system.generate_replenishment_advice(inventory_data, forecast_data)

# 审批补货建议
approved_orders = replenishment_system.approve_replenishment(replenishment_advice, approval_level=1)

# 生成采购订单
purchase_orders = replenishment_system.generate_purchase_orders(approved_orders)
```

#### 4.3.2 MILP优化
```python
from src.replenishment.milp_optimizer import MILPOptimizer

# 初始化MILP优化器
milp_optimizer = MILPOptimizer()

# 运行优化
optimal_plan = milp_optimizer.optimize(inventory_data, forecast_data, supplier_data)
```

### 4.4 Feature Store使用

#### 4.4.1 特征计算与管理
```python
from src.data.feature_store import FeatureStore

# 初始化Feature Store
feature_store = FeatureStore()

# 计算SKU×仓库特征
sku_location_features = feature_store.calculate_sku_location_features(demand_data)

# 生成模型选择标签
model_tags = feature_store.generate_model_selection_tags(sku_location_features)

# 保存特征
feature_store.save_features()
```

## 5. 配置说明

### 5.1 主要配置文件

| 配置项 | 描述 | 文件位置 |
|--------|------|----------|
| 安全库存参数 | 安全库存计算相关参数 | `config/safety_stock_params_*.json` |
| 模型选择标签 | 用于模型自动选择的标签 | `features/model_selection_tags.json` |
| SKU特征数据 | SKU×仓库维度的统计特征 | `features/sku_location_features.json` |

### 5.2 代码中的配置

#### 5.2.1 预测模型配置
在`forecast_models.py`中可以配置：
- 模型类型和参数
- 模型评估指标
- GPU加速选项

#### 5.2.2 MILP优化配置
在`milp_optimizer.py`中可以配置：
- 求解器选择（SCIP、CBC、GLOP等）
- 成本参数（订购成本、持有成本、缺货成本等）
- 约束条件（最大订货量、最小订货量等）

#### 5.2.3 API服务配置
在`start_api.py`中可以配置：
- 服务端口（默认8000）
- 主机地址（默认0.0.0.0）
- CORS配置

## 6. API使用

### 6.1 主要API端点

| API端点 | 方法 | 描述 |
|---------|------|------|
| `/api/items` | GET | 获取所有产品信息 |
| `/api/suppliers` | GET | 获取所有供应商信息 |
| `/api/inventory/levels` | GET | 获取库存水平数据 |
| `/api/purchase/orders` | GET | 获取采购订单数据 |
| `/api/forecast/data` | GET | 获取需求预测数据 |
| `/api/models/performance` | GET | 获取模型性能数据 |
| `/api/models/performance/average` | GET | 获取模型平均性能指标 |
| `/api/optimal/plan` | GET | 获取最优补货计划数据 |

### 6.2 API使用示例

```bash
# 获取所有产品信息
curl http://localhost:8000/api/items

# 获取库存水平数据
curl http://localhost:8000/api/inventory/levels

# 获取模型性能数据
curl http://localhost:8000/api/models/performance
```

## 7. 性能优化

### 7.1 GPU加速

系统支持XGBoost等模型的GPU加速，启用方法：

```python
# 初始化模型选择器时启用GPU
model_selector = ForecastModelSelector(use_gpu=True)
```

**注意**：需要安装GPU版本的XGBoost和相应的GPU驱动。

### 7.2 并行计算

系统支持多进程并行计算，可在`data_processor.py`和其他模块中配置并行度。

### 7.3 缓存机制

系统使用内存缓存和可选的Redis分布式缓存，提高频繁访问数据的响应速度：

```python
from src.system.cache_manager import CacheManager

# 初始化缓存管理器（distributed=True启用Redis）
cache_manager = CacheManager(distributed=False)  # 仅使用内存缓存
```

## 8. 监控与维护

### 8.1 日志查看

系统日志保存在`logs/`目录下：
- `general.log`：通用日志
- `error.log`：错误日志
- `performance.log`：性能日志

### 8.2 模型监控

使用MLOps引擎监控模型性能和数据漂移：

```python
from src.mlops.mlops_engine import MLOpsEngine

# 初始化MLOps引擎
mlops_engine = MLOpsEngine()

# 监控模型性能
model_performance = mlops_engine.monitor_model_performance()

# 检测数据漂移
drift_results = mlops_engine.detect_data_drift()
```

### 8.3 定期维护任务

- 定期更新模型：使用新数据重新训练模型
- 清理日志文件：避免日志文件过大
- 备份模型和数据：定期备份`models/`和`data/`目录

## 9. 常见问题与故障排除

### 9.1 安装问题

**问题**：安装依赖时出现错误
**解决方案**：
- 确保使用Python 3.7+版本
- 尝试升级pip：`pip install --upgrade pip`
- 逐个安装依赖，定位具体问题：`pip install <package_name>`

### 9.2 运行问题

**问题**：无法连接Redis
**解决方案**：
- 确保Redis服务器已启动
- 检查Redis连接参数（主机、端口、密码）
- 在初始化CacheManager时设置`distributed=False`，使用内存缓存

**问题**：GPU加速无法使用
**解决方案**：
- 检查是否安装了GPU版本的XGBoost：`pip install xgboost-gpu`
- 确保GPU驱动已正确安装
- 检查CUDA版本是否与XGBoost兼容
- 尝试在初始化模型选择器时设置`use_gpu=False`，使用CPU版本

**问题**：MILP优化求解缓慢
**解决方案**：
- 减少优化问题的规模（产品数量、时间范围）
- 调整求解器参数，设置适当的求解时间限制
- 尝试使用不同的求解器

### 9.3 API服务问题

**问题**：API服务无法启动
**解决方案**：
- 检查端口是否被占用：`netstat -ano | findstr :8000`
- 尝试使用不同的端口：修改`start_api.py`中的`PORT`变量
- 检查依赖是否安装完整

**问题**：API返回错误
**解决方案**：
- 查看API文档，确保请求格式正确
- 检查系统日志，查看具体错误信息
- 确保数据文件存在且格式正确

## 10. 扩展与定制

### 10.1 添加新的预测模型

1. 在`forecast_models.py`中添加新模型类
2. 实现`train`和`predict`方法
3. 注册到模型选择器的模型列表中

### 10.2 扩展MILP优化

1. 在`milp_optimizer.py`中修改约束条件
2. 添加新的目标函数项
3. 扩展数据输入格式

### 10.3 定制补货策略

1. 在`automated_replenishment.py`中添加新策略
2. 实现策略逻辑
3. 注册到策略列表中

## 11. 示例脚本

### 11.1 运行性能测试
```bash
python test_performance.py
```

### 11.2 测试Feature Store功能
```bash
python test_feature_store.py
```

### 11.3 测试模型可解释性
```bash
python test_model_interpretability.py
```

### 11.4 测试连续学习功能
```bash
python test_forecast_continuous_learning_simple.py
```

### 11.5 测试A/B测试框架
```bash
python test_ab_testing.py
```

## 12. A/B测试框架

### 12.1 概述
A/B测试框架允许您在生产环境中比较不同模型的性能，以确定最佳模型。

### 12.2 使用示例

```python
from src.mlops.ab_testing.ab_test_manager import ABTestManager

# 初始化A/B测试管理器
ab_test_manager = ABTestManager()

# 创建A/B测试
ab_test = ab_test_manager.create_ab_test(
    name="Model Comparison Test",
    model_a="ARIMA",
    model_b="XGBoost",
    metric="mape"
)

# 运行A/B测试
ab_test_results = ab_test_manager.run_ab_test(ab_test)

# 分析A/B测试结果
analysis = ab_test_manager.analyze_results(ab_test_results)
print(f"A/B Test Results: {analysis}")
```

## 13. 联系方式

如有问题或建议，请联系项目维护人员。

---

**更新日期**：2025-12-01  
**版本**：1.0  
**适用系统**：Windows/Linux/Mac