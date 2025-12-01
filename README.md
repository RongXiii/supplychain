# 供应链智能补货系统

## 项目概述

本项目实现了一个完整的供应链库存优化系统，结合了高级预测模型、多种补货策略和MILP（混合整数线性规划）优化，用于优化供应链中的库存管理和补货决策。系统能够针对不同产品自动选择合适的预测模型，进行需求预测，并利用多种补货策略（ROP、Order-up-to、混合策略）和MILP模型计算最优订货策略，最小化总成本。

## 功能特点

1. **智能模型选择**：针对不同产品自动选择最佳预测模型（ARIMA、Holt-Winters、线性回归、随机森林、梯度提升、支持向量回归等）
2. **需求预测**：基于历史数据进行未来需求预测，支持多种时间序列模型
3. **多种补货策略**：
   - ROP（再订货点）+ 安全库存策略
   - Order-up-to Level策略
   - 混合策略（结合两种策略的优势）
4. **增强的MILP优化**：
   - 支持多仓库库存调拨
   - 考虑数量折扣
   - 优化目标函数，最小化总成本
   - 优先调拨，减少采购
5. **自动补单接口**：
   - 基于多种策略自动生成补货建议
   - 支持多级审批工作流
   - 可配置的补货规则
6. **采购订单生成**：根据优化结果自动生成采购订单
7. **模型持续优化**：支持使用实际数据持续更新和优化预测模型
8. **需求对比分析**：比较实际需求与预测需求，优化AI模型
9. **完整的数据模拟**：生成符合实际业务场景的7个数据表
10. **系统状态监控**：实时查看系统运行状态和模型性能
11. **MLOps引擎**：实现模型监控、漂移检测和自动化更新
12. **REST API服务**：提供标准API接口，支持与Power BI等可视化工具集成
13. **数据仪表盘**：内置数据可视化功能，支持库存水平、采购订单、模型性能等多维度展示
14. **Feature Store功能**：
    - SKU×仓库维度的统计特征（CV、季节强度、间歇性指数、促销标记等）
    - 自动生成模型选择标签
    - 特征持久化和管理
    - 支持批量特征计算和更新
    - 特征重要性分析

## 项目结构

```
supplychain/
├── .venv/                   # 虚拟环境目录
├── config/                  # 配置文件目录
│   └── safety_stock_params_*.json  # 安全库存参数配置
├── data/                    # 数据目录
│   ├── sample_demand_data.csv  # 示例需求数据
│   ├── inventory_daily.csv      # 库存日报表
│   ├── purchase_orders.csv      # 采购订单表
│   ├── suppliers.csv            # 供应商表
│   ├── items.csv                # 产品表
│   ├── locations.csv            # 位置表
│   ├── forecast_output.csv      # 预测输出表
│   └── optimal_plan.csv         # 最优计划表
├── features/                # 特征存储目录
│   ├── model_selection_tags.json  # 模型选择标签
│   └── sku_location_features.json  # SKU×仓库特征数据
├── metrics/                 # 模型指标目录
├── models/                  # 模型保存目录
├── src/                     # 源代码目录
│   ├── __pycache__/         # 编译后的Python文件
│   ├── api.py               # REST API服务模块
│   ├── automated_replenishment.py  # 自动补单模块
│   ├── dashboard.py         # 数据仪表盘模块
│   ├── data_processor.py    # 数据处理模块
│   ├── feature_store.py     # Feature Store模块
│   ├── forecast_models.py   # 预测模型选择模块
│   ├── main.py              # 主程序
│   ├── milp_optimizer.py    # MILP优化模块
│   ├── mlops_engine.py      # MLOps引擎模块
│   └── simulated_data.py    # 模拟数据生成模块
├── README.md                # 项目说明
├── requirements.txt         # 依赖列表
├── start_api.py             # API服务启动脚本
├── test_feature_store.py    # Feature Store测试脚本
└── verify_feature_store.py  # Feature Store验证脚本
```

## 安装指南

1. 克隆或下载项目到本地
2. 安装依赖：
   ```
   pip install -r requirements.txt
   ```
   依赖已包含MILP求解器OR-Tools和API服务所需的FastAPI、uvicorn等组件。

## 使用说明

### 运行主程序

```
python src/main.py
```

程序将执行以下步骤：
1. 生成或加载模拟数据（7个数据表）
2. 对每个产品进行数据预处理和模型选择
3. 训练最佳预测模型
4. 进行需求预测
5. 运行多仓MILP优化（优先调拨，减少采购）
6. 计算EOQ和考虑数量折扣
7. 生成采购订单
8. 演示模型更新功能
9. 执行自动补单（基于多种策略）
10. 演示审批流程
11. 显示最终系统状态

### 启动API服务

```
python start_api.py
```

API服务将启动在 `http://localhost:8000`，提供以下功能：
- API文档：`http://localhost:8000/docs`
- 数据接口：`http://localhost:8000/api/`
- 支持与Power BI等可视化工具集成

### API端点说明

- `/api/items` - 获取所有产品信息
- `/api/suppliers` - 获取所有供应商信息
- `/api/inventory/levels` - 获取库存水平数据
- `/api/purchase/orders` - 获取采购订单数据
- `/api/forecast/data` - 获取需求预测数据
- `/api/models/performance` - 获取模型性能数据
- `/api/models/performance/average` - 获取模型平均性能指标
- `/api/optimal/plan` - 获取最优补货计划数据

### 使用数据仪表盘

在主程序运行结束后，系统会询问是否查看数据仪表盘，输入 `y` 即可显示包含库存水平、采购订单和模型性能等维度的可视化图表。

### 模块说明

#### 数据处理模块（data_processor.py）
- 加载和预处理数据
- 处理缺失值和特征工程
- 分割训练集和测试集
- 比较实际需求和预测需求

#### 预测模型选择模块（forecast_models.py）
- 实现多种预测模型（ARIMA、Holt-Winters、线性回归、随机森林、梯度提升、支持向量回归等）
- 自动选择最佳模型
- 模型训练、预测和评估
- 模型保存和加载
- 模型更新

#### MILP优化模块（milp_optimizer.py）
- 使用OR-Tools创建增强的MILP模型
- 定义目标函数和约束条件
- 支持多仓库库存调拨
- 考虑数量折扣
- 求解模型（默认使用SCIP求解器，支持CBC等多种求解器）
- 提取最优解

#### 自动补单模块（automated_replenishment.py）
- 实现多种补货策略（ROP、Order-up-to、混合策略）
- 生成自动补货建议
- 支持多级审批工作流
- 管理采购订单状态

#### 模拟数据生成模块（simulated_data.py）
- 生成符合实际业务场景的7个数据表
- 支持自定义数据规模和分布
- 生成的数据可直接用于系统测试

#### API服务模块（api.py）
- 提供REST API接口，支持与Power BI等可视化工具集成
- 包含8个核心数据接口
- 支持CORS跨域访问

#### Feature Store模块（feature_store.py）
- SKU×仓库维度特征计算
- 统计特征：CV（变异系数）、季节强度、间歇性指数
- 模型选择标签自动生成
- 支持批量特征计算和更新
- 特征持久化和管理
- 特征重要性分析

### MLOps引擎模块（mlops_engine.py）
- 模型监控和性能评估
- 数据漂移检测
- 模型自动化更新
- 模型版本管理
- 与Feature Store集成，支持特征监控

### 数据仪表盘模块（dashboard.py）
- 库存水平可视化
- 采购订单分析
- 模型性能评估
- 需求预测对比
- 特征分布分析

### 主程序（main.py）
- 系统集成和协调
- 演示系统完整功能
- 管理系统状态
- 集成Feature Store功能

## 配置说明

### 预测模型配置

可以配置以下预测模型参数：
- 模型类型：ARIMA、Holt-Winters、线性回归、随机森林、梯度提升等
- 模型评估指标：MAPE、MAE、RMSE等
- 模型选择策略：自动选择最佳模型

### MILP优化配置

可以配置以下优化参数：
- 求解器选择：SCIP、CBC、GLOP等
- 成本参数：订购成本、持有成本、缺货成本、调拨成本
- 约束条件：
  - 最大订货量
  - 最小订货量
  - 最大库存
  - 供应能力约束
  - 调拨能力约束
  - 服务水平约束

### 补货策略配置

支持以下补货策略：
- ROP（再订货点）策略
- Order-up-to（目标库存）策略
- 混合策略

### 自动补单配置

可以配置以下自动补单参数：
- 审批流程：多级审批工作流
- 通知机制：自动通知相关人员
- 例外处理：异常情况处理规则

### 模拟数据生成配置

可以配置以下模拟数据参数：
- 数据规模：产品数量、时间范围
- 分布参数：需求分布、成本分布
- 业务规则：供应商关系、库存政策

### API服务配置

可以配置以下API服务参数：
- 服务端口：默认8000
- 主机地址：默认0.0.0.0
- CORS配置：允许的源、方法、头部
- 自动重载：开发模式下启用

### 系统配置

可以配置以下系统参数：
- 数据保存路径
- 日志级别
- 并行处理设置
- 模型保存路径

## 数据格式

系统生成并使用以下数据表：

### 特征相关数据

#### 1. SKU×仓库特征数据（sku_location_features.json）
- `item_id`：产品ID
- `location_id`：位置ID
- `cv`：变异系数
- `seasonality_strength`：季节强度
- `intermittency_index`：间歇性指数
- `promotion_flag`：促销标记
- `feature_1`-`feature_5`：其他统计特征

#### 2. 模型选择标签数据（model_selection_tags.json）
- `item_id`：产品ID
- `location_id`：位置ID
- `model_tag`：模型选择标签（如'high_seasonality'、'intermittent'等）

### 核心业务数据表

系统生成并使用以下7个核心业务数据表：

### 1. InventoryDaily（库存日报表）
- `date`：日期
- `item_id`：产品ID
- `location_id`：位置ID
- `on_hand_qty`：当前库存数量
- `on_order_qty`：已订购未到货数量
- `demand_qty`：日需求数量

### 2. PurchaseOrders（采购订单表）
- `order_id`：订单ID
- `item_id`：产品ID
- `supplier_id`：供应商ID
- `order_date`：订单日期
- `order_qty`：订单数量
- `cost_per_unit`：单位成本
- `status`：订单状态
- `due_date`：预计到货日期

### 3. Suppliers（供应商表）
- `supplier_id`：供应商ID
- `supplier_name`：供应商名称
- `lead_time_days`：提前期（天）
- `min_order_qty`：最小订货量
- `price_per_unit`：单位价格

### 4. Items（产品表）
- `item_id`：产品ID
- `item_name`：产品名称
- `category`：产品类别
- `unit_of_measure`：计量单位
- `safety_stock`：安全库存

### 5. Locations（位置表）
- `location_id`：位置ID
- `location_name`：位置名称
- `location_type`：位置类型（仓库/门店）

### 6. ForecastOutput（预测输出表）
- `forecast_id`：预测ID
- `item_id`：产品ID
- `date`：日期
- `forecast_qty`：预测数量
- `model_name`：使用的模型名称
- `confidence_level`：置信水平

### 7. OptimalPlan（最优计划表）
- `plan_id`：计划ID
- `item_id`：产品ID
- `date`：日期
- `optimal_order_qty`：最优订货量
- `safety_stock_level`：安全库存水平
- `reorder_point`：再订货点
- `order_up_to_level`：目标库存水平

示例数据格式见`data/`目录下的各个CSV文件

## 扩展说明

### 模型扩展

- 支持添加新的预测模型
- 在`forecast_models.py`中添加新模型类
- 实现`train`和`predict`方法
- 注册到模型选择器

### 优化扩展

- 支持添加新的约束条件
- 在`milp_optimizer.py`中修改约束条件部分
- 支持自定义目标函数
- 支持多仓库库存调拨扩展

### 数据扩展

- 支持添加新的数据源
- 在`data_processor.py`中添加新的数据加载和处理逻辑
- 在`simulated_data.py`中扩展模拟数据生成

### 策略扩展

- 支持添加新的补货策略
- 在`automated_replenishment.py`中实现新策略

## 技术栈

- Python 3.7+
- NumPy
- Pandas
- scikit-learn
- statsmodels
- OR-Tools (用于MILP优化)
- joblib
- matplotlib
- pytest
- FastAPI (用于REST API服务)
- Uvicorn (ASGI服务器)
- SQLAlchemy (用于数据管理)
- feature-engine (用于特征工程)
- json (用于特征数据持久化)

## 未来改进方向

### 一、预测模型优化
1. 扩展深度学习模型支持（LSTM、Transformer、Prophet等）
2. 引入间歇性需求模型（Croston方法、SBA方法等）
3. 实现模型融合策略，结合多种模型优势
4. 添加超参数自动优化机制
5. 实现模型动态权重调整

### 二、特征工程与Feature Store优化
6. 扩展Feature Store功能，支持更多特征类型
7. 实现特征生命周期管理
8. 添加特征监控和漂移检测
9. 支持特征版本控制
10. 整合外部数据源特征（如天气、节假日、市场趋势等）
11. 实现实时特征计算和更新
12. 添加特征重要性分析和解释功能
13. 优化特征计算性能，支持大规模数据处理

### 三、MILP优化模型优化
14. 实现多目标优化（成本、服务水平、库存周转率等）
15. 开发启发式算法和元启发式算法，提高求解效率
16. 支持更多复杂约束条件（如供应商产能、运输时间窗等）
17. 实现分级优化策略，处理大规模问题
18. 支持多种求解器和优化算法

### 四、数据处理与质量优化
19. 实现智能异常检测和数据清洗
20. 建立数据质量评估和监控机制
21. 支持多源数据集成和标准化
22. 实现数据血缘追踪
23. 优化数据加载和转换性能

### 五、系统性能与扩展性优化
24. 实现并行计算和分布式处理
25. 采用微服务架构，支持模块独立部署
26. 优化API服务性能和响应速度
27. 支持水平扩展，处理大规模数据和请求
28. 实现缓存机制，提高频繁访问数据的响应速度

### 六、监控与可解释性优化
29. 添加更多数据可视化功能和交互式仪表盘
30. 实现模型性能监控和告警机制
31. 添加预测结果解释功能（SHAP、LIME等）
32. 实现供应链决策路径可视化
33. 开发实时监控面板，展示关键指标

### 七、业务价值与落地优化
34. 实现A/B测试框架，评估不同策略效果
35. 开发场景化定制功能，支持不同行业需求
36. 添加供应链风险分析和模拟功能
37. 实现预测性维护和设备故障预警集成
38. 开发ROI分析工具，量化系统价值

### 八、自动化与运维优化
39. 实现更灵活的配置管理
40. 支持实时数据和流式处理
41. 添加自动化测试和CI/CD流程
42. 实现自动化模型更新和重训练
43. 开发运维监控和日志分析系统
44. 支持多环境部署（开发、测试、生产）
45. 实现配置版本控制

### 九、其他改进方向
46. 支持多用户协作和权限管理
47. 实现更复杂的供应链网络优化
48. 添加更多高级分析功能
49. 支持移动端访问和监控
50. 开发API文档自动生成和更新机制

## 许可证

本项目采用MIT许可证。

## 联系方式

如有问题或建议，请联系项目维护人员。
