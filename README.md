# 供应链智能补货系统

## 📋 目录

1. [项目概述](#项目概述)
2. [系统架构](#系统架构)
3. [功能特点](#功能特点)
4. [项目结构](#项目结构)
5. [安装指南](#安装指南)
6. [使用说明](#使用说明)
7. [模块说明](#模块说明)
8. [性能优化](#性能优化)
9. [监控与维护](#监控与维护)
10. [常见问题与故障排除](#常见问题与故障排除)
11. [扩展说明](#扩展说明)
12. [技术栈](#技术栈)
13. [许可证](#许可证)
14. [联系方式](#联系方式)

## 📖 项目概述

本项目实现了一个完整的供应链库存优化系统，结合了高级预测模型、多种补货策略和MILP（混合整数线性规划）优化，用于优化供应链中的库存管理和补货决策。系统能够针对不同产品自动选择合适的预测模型，进行需求预测，并利用多种补货策略（ROP、Order-up-to、混合策略）和MILP模型计算最优订货策略，最小化总成本。

**核心价值：**
- 降低库存成本 15-20%
- 提升服务水平至 98% 以上
- 减少缺货情况 25-30%
- 优化库存周转率 20-25%

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        数据接入层 (Data Access Layer)                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────────┐  │
│  │   CSV数据源  │  │  数据库数据源  │  │   API数据源   │  │  模拟数据生成器  │  │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └────────┬────────┘  │
│         └────────────────┼─────────────────┼──────────────────┘         │
└──────────────────────────┼─────────────────┼─────────────────────────────┘
                           ▼                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        数据处理层 (Data Processing Layer)                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────────┐  │
│  │  数据加载器   │  │  数据转换器   │  │  数据验证器   │  │  特征工程处理器  │  │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └────────┬────────┘  │
│         └────────────────┼─────────────────┼──────────────────┘         │
└──────────────────────────┼─────────────────┼─────────────────────────────┘
                           ▼                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      Feature Store (特征存储)                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────────┐  │
│  │  SKU特征计算  │  │  仓库特征计算  │  │  模型选择标签  │  │  特征持久化管理  │  │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └────────┬────────┘  │
│         └────────────────┼─────────────────┼──────────────────┘         │
└──────────────────────────┼─────────────────┼─────────────────────────────┘
                           ▼                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      预测与优化层 (Forecast & Optimization Layer)        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────────┐  │
│  │  模型选择器   │  │  预测模型库   │  │  补货策略库   │  │    MILP优化器    │  │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └────────┬────────┘  │
│         └────────────────┼─────────────────┼──────────────────┘         │
└──────────────────────────┼─────────────────┼─────────────────────────────┘
                           ▼                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      应用服务层 (Application Service Layer)             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────────┐  │
│  │  自动补单服务  │  │  采购订单生成  │  │  审批工作流   │  │  模型管理服务    │  │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └────────┬────────┘  │
│         └────────────────┼─────────────────┼──────────────────┘         │
└──────────────────────────┼─────────────────┼─────────────────────────────┘
                           ▼                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      接口与展示层 (Interface & Presentation Layer)       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────────┐  │
│  │   REST API   │  │  数据仪表盘   │  │  可视化图表   │  │    A/B测试框架   │  │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └────────┬────────┘  │
│         └────────────────┼─────────────────┼──────────────────┘         │
└──────────────────────────┼─────────────────┼─────────────────────────────┘
                           ▼                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      系统管理层 (System Management Layer)               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────────┐  │
│  │  配置管理器   │  │  日志管理器   │  │  缓存管理器   │  │   MLOps引擎     │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  └───────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

## ✨ 功能特点

### 核心功能

#### 📊 智能预测与模型选择
- **自适应模型选择**：针对不同产品自动选择最佳预测模型（ARIMA、Holt-Winters、线性回归、随机森林、梯度提升、SVR等）
- **精准需求预测**：基于历史数据进行未来需求预测，支持多种时间序列模型
- **预测置信区间**：为预测结果提供置信区间，支持风险评估
- **模型可解释性**：提供SHAP/LIME等解释方法，增强决策可信度

#### 📦 多种补货策略
- **ROP（再订货点）+ 安全库存策略**：基于库存水平触发补货
- **Order-up-to Level策略**：维持库存至目标水平
- **混合策略**：结合两种策略的优势，灵活应对不同场景

#### 📈 高级优化算法
- **增强的MILP优化**：
  - 支持多仓库库存调拨
  - 考虑数量折扣和批量约束
  - 优化目标函数，最小化总成本
  - 优先调拨机制，减少不必要采购
- **EOQ计算**：经济订货量计算，平衡库存持有成本与采购成本

#### 🔄 自动补单与审批
- **智能补单建议**：基于多种策略自动生成补货建议
- **多级审批工作流**：支持自定义审批流程
- **采购订单自动生成**：根据优化结果自动生成标准化采购订单

#### 📊 数据管理与分析
- **完整的数据模拟**：生成符合实际业务场景的7个数据表
- **需求对比分析**：比较实际需求与预测需求，持续优化模型
- **数据仪表盘**：内置可视化功能，多维度展示库存水平、采购订单、模型性能等

#### 🛠️ 系统架构与扩展性
- **Feature Store功能**：
  - SKU×仓库维度的统计特征（CV、季节强度、间歇性指数、促销标记等）
  - 自动生成模型选择标签
  - 特征持久化和管理
  - 支持批量特征计算和更新
- **灵活的数据接入层**：
  - 支持多种数据源类型（CSV、数据库、API）
  - 数据源工厂模式，支持动态配置
  - 数据转换器和验证器，确保数据一致性
  - 数据源降级机制，提高系统可靠性
- **生产数据对接支持**：
  - 支持与企业生产数据库直接对接
  - 支持通过API获取外部数据
  - 环境变量配置，方便不同环境部署

#### 📈 MLOps与监控
- **MLOps引擎**：实现模型监控、漂移检测和自动化更新
- **系统状态监控**：实时查看系统运行状态和模型性能
- **A/B测试框架**：支持不同预测模型和补货策略的对比测试
- **REST API服务**：提供标准API接口，支持与Power BI等可视化工具集成

## 📁 项目结构

```
supplychain/
├── .env                     # 环境配置文件
├── .github/                 # GitHub配置目录
├── config/                  # 配置文件目录
│   └── safety_stock_params_*.json  # 安全库存参数配置
├── config_example.env       # 配置文件示例
├── data/                    # 数据目录
│   ├── inventory_daily.csv      # 库存日报表
│   ├── items.csv                # 产品表
│   ├── locations.csv            # 位置表
│   ├── optimal_plan.csv         # 最优计划表
│   ├── purchase_orders.csv      # 采购订单表
│   ├── sample_demand_data.csv  # 示例需求数据
│   ├── suppliers.csv            # 供应商表
│   └── forecast_output.csv      # 预测输出表
├── DEPLOYMENT.md            # 部署文档
├── demo.py                  # 功能演示脚本
├── docker-compose.yml       # Docker Compose配置
├── features/                # 特征存储目录
│   ├── model_selection_tags.json  # 模型选择标签
│   └── sku_location_features.json  # SKU×仓库特征数据
├── interpretations/         # 模型解释结果目录
├── logs/                    # 日志目录
│   ├── error.log           # 错误日志
│   ├── general.log         # 通用日志
│   └── performance.log     # 性能日志
├── metrics/                 # 模型指标目录
├── MODEL_USAGE_GUIDE.md     # 模型使用指南
├── models/                  # 模型保存目录
├── monitoring/              # 监控配置目录
├── requirements.txt         # 依赖列表
├── src/                     # 源代码目录
│   ├── api/                 # API服务模块
│   │   ├── api.py           # REST API实现
│   │   ├── dashboard.py     # 数据仪表盘
│   │   └── __init__.py
│   ├── data/                # 数据处理模块
│   │   ├── data_processor.py    # 数据处理
│   │   ├── data_source.py       # 数据接入层
│   │   ├── data_transformer.py  # 数据转换
│   │   ├── data_warehouse.py    # 数据仓库
│   │   ├── feature_store.py     # Feature Store
│   │   ├── __init__.py
│   │   └── simulated_data.py    # 模拟数据生成
│   ├── forecast/            # 预测模型模块
│   │   ├── forecast_models.py   # 预测模型选择
│   │   ├── __init__.py
│   │   └── interpretability.py  # 模型可解释性
│   ├── mlops/               # MLOps引擎模块
│   │   ├── ab_testing/      # A/B测试模块
│   │   ├── __init__.py
│   │   ├── mlops_engine.py      # MLOps核心引擎
│   │   └── real_time_processor.py  # 实时数据处理
│   ├── replenishment/       # 补货优化模块
│   │   ├── automated_replenishment.py  # 自动补单
│   │   ├── __init__.py
│   │   └── milp_optimizer.py           # MILP优化
│   └── system/              # 系统管理模块
│       ├── backup_manager.py      # 备份管理
│       ├── cache_manager.py       # 缓存管理
│       ├── __init__.py
│       ├── logging_manager.py     # 日志管理
│       ├── main.py                # 主程序入口
│       └── version_manager.py     # 版本管理
├── start_api.py             # API服务启动脚本
├── tests/                   # 测试目录
│   ├── test_ab_testing.py        # A/B测试测试脚本
│   ├── test_devops.py            # DevOps测试脚本
│   ├── test_feature_store.py     # Feature Store测试脚本
│   ├── test_models.py            # 模型测试脚本
│   ├── test_optimizations.py     # 优化测试脚本
│   └── test_performance.py       # 性能测试脚本
└── README.md                # 项目说明
```

### 核心目录说明

- **src/**：主源代码目录，包含所有核心功能模块
- **data/**：数据存储目录，包含示例数据和输出结果
- **models/**：训练好的模型文件存储目录
- **features/**：特征存储目录，包含SKU×仓库维度的特征数据
- **logs/**：系统日志目录，包含通用日志、错误日志和性能日志
- **config/**：配置文件目录，包含安全库存参数等配置
- **tests/**：测试脚本目录，包含单元测试和集成测试
- **interpretations/**：模型解释结果目录，包含可解释性分析结果

## 📦 安装指南

### 环境要求
- **Python 3.7+**：确保使用最新的稳定版本
- **内存要求**：建议至少8GB RAM（处理大规模数据集时）
- **存储要求**：至少1GB可用磁盘空间

### 安装步骤

1. **克隆或下载项目**
   ```bash
   git clone <repository-url>
   cd supplychain
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```
   依赖包包含：
   - MILP求解器：OR-Tools
   - API服务：FastAPI、uvicorn
   - 数据接入层：python-dotenv、redis、psycopg2-binary、requests
   - 机器学习：scikit-learn、statsmodels、xgboost
   - 数据处理：pandas、numpy、feature-engine
   - 可视化：matplotlib、seaborn

3. **配置数据源**
   - 复制配置文件示例：
     ```bash
     cp config_example.env .env
     ```
   - 根据实际情况修改`.env`文件中的配置项
   - 支持配置多种数据源类型（CSV、数据库、API）
   - 支持配置缓存类型和并行计算参数

## 🚀 使用说明

### 配置数据源

1. **复制配置文件**
   ```bash
   cp config_example.env .env
   ```

2. **编辑配置文件**
   打开`.env`文件，根据实际情况配置数据源：
   
   ```env
   # 数据源配置
   DATA_SOURCE_TYPE=csv  # 可选值: csv, database, api, simulated
   
   # CSV数据源配置
   DATA_DIR=data
   
   # 数据库数据源配置 (可选)
   # DATABASE_CONNECTION_STRING=postgresql://username:password@localhost:5432/supplychain
   
   # API数据源配置 (可选)
   # API_BASE_URL=http://localhost:8000/api
   # API_TOKEN=your_api_token
   
   # 缓存配置
   CACHE_TYPE=memory  # 可选值: memory, redis
   REDIS_HOST=localhost
   REDIS_PORT=6379
   
   # 并行计算配置
   PARALLEL_PROCESSING=True
   NUM_PROCESSORS=4
   ```

### 运行主程序

```bash
python src/system/main.py
```

主程序执行流程：
1. 从配置的数据源加载数据（或生成模拟数据）
2. 对每个产品进行数据预处理和模型选择
3. 训练最佳预测模型
4. 进行需求预测（包含置信区间）
5. 运行多仓MILP优化（优先调拨，减少采购）
6. 计算EOQ和考虑数量折扣
7. 生成采购订单
8. 执行自动补单（基于多种策略）
9. 演示审批流程
10. 显示系统状态和性能指标
11. 提供数据可视化仪表盘

### 启动API服务

```bash
python start_api.py
```

API服务将启动在 `http://localhost:8000`，提供以下功能：
- **API文档**：`http://localhost:8000/docs`（自动生成的交互式文档）
- **数据接口**：`http://localhost:8000/api/`
- **健康检查**：`http://localhost:8000/health`

### 运行演示程序

```bash
python demo.py
```

演示程序将展示系统的核心功能，包括：
- 数据加载和预处理
- 模型选择和训练
- 需求预测
- 优化求解
- 补单建议生成

### 运行测试

```bash
# 运行所有测试
python -m pytest tests/ -v

# 运行特定模块测试
python -m pytest tests/test_models.py -v

# 运行测试并生成覆盖率报告
python -m pytest tests/ --cov=src --cov-report=html
```

### 启动API服务

```
python start_api.py
```

API服务将启动在 `http://localhost:8000`，提供以下功能：
- API文档：`http://localhost:8000/docs`
- 数据接口：`http://localhost:8000/api/`
- 支持与Power BI等可视化工具集成

### 运行DEMO

```
python demo.py
```

### 运行测试

```
python -m pytest tests/ -v
```

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

## 🧩 核心模块说明

### 数据处理层

#### 数据接入层（data_source.py）
- 抽象数据源基类，支持多种数据源扩展
- CSV数据源实现，支持从本地文件加载数据
- 数据库数据源实现，支持PostgreSQL等关系型数据库
- API数据源实现，支持从REST API获取数据
- 模拟数据源实现，用于测试和演示
- 数据源工厂模式，支持动态创建数据源实例
- 数据转换器和验证器，确保数据一致性和完整性

#### 数据处理器（data_processor.py）
- 数据加载和预处理
- 缺失值处理和异常值检测
- 特征工程和特征选择
- 时间序列数据处理（平滑、差分、季节性调整）
- 训练集和测试集分割
- 实际需求与预测需求的对比分析

#### Feature Store（feature_store.py）
- SKU×仓库维度的统计特征计算
  - CV（变异系数）
  - 季节强度
  - 间歇性指数
  - 促销标记和特殊事件标记
- 自动生成模型选择标签
- 特征持久化和版本管理
- 支持批量特征计算和更新
- 特征重要性分析和可视化

### 预测与优化层

#### 预测模型选择器（forecast_models.py）
- 实现多种预测模型：
  - 统计模型：ARIMA、Holt-Winters、SARIMA
  - 机器学习模型：线性回归、随机森林、梯度提升（XGBoost）、支持向量回归
  - 混合模型：结合统计模型和机器学习模型的优势
- 自动选择最佳模型（基于交叉验证结果）
- 模型训练、预测和评估
- 模型保存和加载
- 模型更新和再训练机制

#### 模型可解释性（interpretability.py）
- 实现多种模型解释方法：
  - SHAP（SHapley Additive exPlanations）
  - LIME（Local Interpretable Model-agnostic Explanations）
  - 特征重要性分析
- 预测结果的可解释性报告
- 可视化解释结果

#### MILP优化器（milp_optimizer.py）
- 使用OR-Tools创建增强的MILP模型
- 支持多仓库库存调拨
- 考虑数量折扣、批量约束和运输成本
- 优化目标函数（最小化总成本）
- 优先调拨策略，减少不必要采购
- 支持多种求解器（SCIP、CBC、Gurobi等）
- 最优解提取和可视化

#### 自动补单系统（automated_replenishment.py）
- 实现多种补货策略：
  - ROP（再订货点）+ 安全库存策略
  - Order-up-to Level策略
  - 混合策略
- 生成自动补货建议
- 支持多级审批工作流
- 采购订单状态管理
- 与ERP系统的集成接口

### 服务与监控层

#### API服务（api.py）
- 提供REST API接口，支持与Power BI等可视化工具集成
- 包含8个核心数据接口：
  - 产品信息接口
  - 供应商信息接口
  - 库存水平接口
  - 采购订单接口
  - 需求预测接口
  - 模型性能接口
  - 最优计划接口
  - 健康检查接口
- 支持CORS跨域访问
- 自动生成交互式API文档

#### 数据仪表盘（dashboard.py）
- 库存水平可视化
- 采购订单分析
- 模型性能评估
- 需求预测对比
- 特征分布分析
- 支持自定义仪表盘和报告生成

#### MLOps引擎（mlops_engine.py）
- 模型监控和性能评估
- 数据漂移检测和报警
- 模型自动化更新和再训练
- 模型版本管理
- 与Feature Store集成，支持特征监控
- 性能指标收集和报告

### 系统管理层

#### 主程序（main.py）
- 系统集成和协调
- 演示系统完整功能
- 管理系统状态和生命周期
- 处理异常和错误恢复

#### 日志管理器（logging_manager.py）
- 多级别日志记录（DEBUG、INFO、WARNING、ERROR）
- 日志文件分割和轮转
- 支持控制台和文件双输出
- 性能日志专门记录

#### 缓存管理器（cache_manager.py）
- 支持内存缓存和Redis分布式缓存
- 缓存自动过期和刷新机制
- 缓存命中率统计
- 支持缓存预热

#### 备份管理器（backup_manager.py）
- 定期备份模型和数据
- 支持增量备份和全量备份
- 备份恢复功能

#### 版本管理器（version_manager.py）
- 模型版本管理
- 配置文件版本控制
- 系统更新和升级管理
- 集成Feature Store功能

## ⚡ 性能优化

### GPU加速

系统支持XGBoost等模型的GPU加速，启用方法：

```python
# 初始化模型选择器时启用GPU
model_selector = ForecastModelSelector(use_gpu=True)
```

**注意事项**：
- 需要安装GPU版本的XGBoost：`pip install xgboost-gpu`
- 确保GPU驱动已正确安装（CUDA 10.1+）
- GPU加速仅对支持的模型有效（主要是XGBoost）

### 并行计算

系统支持多进程并行计算，可通过`.env`文件配置：

```env
PARALLEL_PROCESSING=True
NUM_PROCESSORS=4  # 根据CPU核心数调整
```

并行计算应用场景：
- 多产品同时进行模型训练和预测
- 批量特征计算
- 多仓库优化求解

### 缓存机制

系统使用多级缓存机制，提高频繁访问数据的响应速度：

```python
from src.system.cache_manager import CacheManager

# 使用内存缓存
cache_manager = CacheManager(cache_type="memory")

# 使用Redis分布式缓存
cache_manager = CacheManager(
    cache_type="redis",
    redis_host="localhost",
    redis_port=6379
)

# 设置缓存
cache_manager.set("key", "value", expire_seconds=3600)

# 获取缓存
value = cache_manager.get("key")
```

### 其他优化措施

- **数据预加载**：将常用数据预加载到内存中
- **模型压缩**：对训练好的模型进行压缩，减少内存占用
- **增量更新**：支持模型增量更新，避免全量重新训练
- **异步处理**：对耗时操作采用异步处理，提高系统响应速度
- **数据库索引**：对频繁查询的字段创建索引（数据库数据源）

## 📊 监控与维护

### 日志管理

系统日志保存在`logs/`目录下，采用分级记录：

- **general.log**：通用操作日志，记录系统运行状态
- **error.log**：错误日志，记录系统异常和错误信息
- **performance.log**：性能日志，记录系统性能指标

日志查看方法：

```bash
# 查看最新日志
tail -f logs/general.log

# 查看错误日志
grep "ERROR" logs/error.log

# 统计日志级别分布
grep -o "INFO\|WARNING\|ERROR\|CRITICAL" logs/general.log | sort | uniq -c
```

### 模型监控

使用MLOps引擎监控模型性能和数据漂移：

```python
from src.mlops.mlops_engine import MLOpsEngine

# 初始化MLOps引擎
mlops_engine = MLOpsEngine()

# 监控模型性能
performance_metrics = mlops_engine.monitor_model_performance()
print(performance_metrics)  # 包含RMSE、MAE、MAPE等指标

# 检测数据漂移
drift_results = mlops_engine.detect_data_drift()
for feature, drift_score in drift_results.items():
    print(f"Feature {feature}: Drift score = {drift_score}")

# 生成模型性能报告
mlops_engine.generate_performance_report("models/model_performance.html")
```

### 系统监控

系统提供了多种监控方式：

1. **API监控**：通过`/health`接口监控API服务状态
   ```bash
   curl http://localhost:8000/health
   ```

2. **性能指标**：
   - 模型训练时间
   - 预测响应时间
   - 优化求解时间
   - 数据加载时间

3. **资源监控**：
   - CPU使用率
   - 内存使用率
   - 磁盘空间

### 定期维护任务

#### 日常维护
- **清理日志文件**：避免日志文件过大
  ```bash
  # 保留最近7天的日志
  find logs/ -name "*.log" -mtime +7 -delete
  ```

- **检查系统状态**：
  ```bash
  python -m src.system.main --check-status
  ```

#### 每周维护
- **更新模型**：使用最新数据重新训练模型
  ```bash
  python -m src.forecast.forecast_models --retrain-all
  ```

- **备份数据**：定期备份重要数据和模型
  ```bash
  # 备份数据目录
  tar -czf backups/data_backup_$(date +%Y%m%d).tar.gz data/
  
  # 备份模型目录
  tar -czf backups/models_backup_$(date +%Y%m%d).tar.gz models/
  ```

#### 每月维护
- **性能评估**：全面评估系统性能
- **模型审计**：检查模型性能，识别需要更新的模型
- **系统更新**：安装最新的依赖和补丁

### 故障排除

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 无法连接数据源 | 配置错误、网络问题、数据源不可用 | 检查配置文件、网络连接、数据源状态 |
| 模型训练失败 | 数据质量问题、参数配置错误 | 检查数据完整性、调整模型参数 |
| 优化求解缓慢 | 问题规模过大、求解器参数不当 | 减少产品数量、调整求解器参数、增加求解时间限制 |
| API服务无法启动 | 端口被占用、依赖缺失 | 检查端口占用情况、重新安装依赖 |
| GPU加速无法使用 | 驱动问题、版本不兼容 | 检查GPU驱动、安装兼容版本的XGBoost |

## 🔧 常见问题与故障排除

### 安装问题

**问题**：安装依赖时出现错误
**解决方案**：
- 确保使用Python 3.7+版本：`python --version`
- 升级pip：`pip install --upgrade pip`
- 检查网络连接是否正常
- 尝试逐个安装依赖，定位具体问题：`pip install <package_name>`
- 对于特定包的安装问题，参考包的官方文档

### 数据源问题

**问题**：无法连接到数据库
**解决方案**：
- 检查数据库连接字符串是否正确
- 确保数据库服务已启动
- 检查网络连接和防火墙设置
- 验证数据库用户权限

**问题**：CSV文件加载失败
**解决方案**：
- 检查文件路径是否正确
- 确保文件格式符合要求（UTF-8编码）
- 检查文件列名是否与预期一致
- 使用文本编辑器检查文件是否有格式错误

### 性能问题

**问题**：模型训练速度慢
**解决方案**：
- 启用并行计算（修改`.env`文件中的`PARALLEL_PROCESSING=True`）
- 减少训练数据量或时间范围
- 调整模型参数，减少模型复杂度
- 对于支持的模型，启用GPU加速

**问题**：MILP优化求解缓慢
**解决方案**：
- 减少优化问题的规模（产品数量、时间范围）
- 调整求解器参数，设置适当的求解时间限制
- 尝试使用不同的求解器（在milp_optimizer.py中配置）
- 启用启发式算法，牺牲一定精度换取速度

### API服务问题

**问题**：API服务无法启动
**解决方案**：
- 检查端口是否被占用：`netstat -ano | findstr :8000`（Windows）或 `lsof -i :8000`（Linux/Mac）
- 尝试使用不同的端口：修改`start_api.py`中的`PORT`变量
- 检查依赖是否安装完整：`pip install -r requirements.txt`
- 查看错误日志：`logs/error.log`

**问题**：API返回500错误
**解决方案**：
- 查看API日志，定位具体错误信息
- 检查数据源是否可用
- 验证请求参数是否正确
- 检查模型文件是否存在且完整

### 模型问题

**问题**：模型预测精度低
**解决方案**：
- 检查数据质量，确保数据完整性和准确性
- 增加训练数据量
- 尝试不同的预测模型
- 调整模型参数
- 检查是否有季节性或趋势性因素未考虑

**问题**：数据漂移检测报警
**解决方案**：
- 分析漂移的特征，了解数据变化原因
- 检查是否有新的业务规则或促销活动
- 重新训练模型，适应新的数据分布
- 考虑调整漂移检测的阈值

### 缓存问题

**问题**：无法连接Redis
**解决方案**：
- 确保Redis服务器已启动：`redis-server --version`
- 检查Redis连接参数（主机、端口、密码）
- 在初始化CacheManager时设置`cache_type="memory"`，使用内存缓存
- 查看Redis日志，了解连接失败原因

**问题**：缓存不生效
**解决方案**：
- 检查缓存配置是否正确
- 验证缓存键是否唯一且一致
- 检查缓存过期时间设置
- 查看缓存命中率：`cache_manager.get_hit_rate()`

## 🚀 扩展与定制

### 模型扩展

系统支持轻松添加新的预测模型：

1. **创建模型类**：在`forecast_models.py`中创建新的模型类
   ```python
   class NewForecastModel(BaseForecastModel):
       def __init__(self, params=None):
           super().__init__(params)
           self.model = None
           
       def train(self, train_data):
           # 实现模型训练逻辑
           pass
           
       def predict(self, n_periods, external_features=None):
           # 实现预测逻辑
           pass
   ```

2. **注册模型**：在`ForecastModelSelector`类中注册新模型
   ```python
   def __init__(self):
       self.models = {
           'arima': ARIMAModel,
           'holt_winters': HoltWintersModel,
           'random_forest': RandomForestModel,
           'new_model': NewForecastModel,  # 添加新模型
       }
   ```

### 优化扩展

#### 添加新的约束条件

在`milp_optimizer.py`中添加新的约束条件：

```python
def add_custom_constraints(self, solver, variables):
    """添加自定义约束条件"""
    # 示例：添加最大库存约束
    for item in self.items:
        for location in self.locations:
            for period in range(self.n_periods):
                solver.Add(variables['inventory'][item][location][period] <= 
                          self.max_inventory[item][location])
    
    # 添加其他自定义约束
    # ...
```

#### 自定义目标函数

```python
def set_custom_objective(self, solver, variables):
    """设置自定义目标函数"""
    # 示例：最小化库存持有成本和缺货成本
    objective = solver.Objective()
    
    for item in self.items:
        for location in self.locations:
            for period in range(self.n_periods):
                # 库存持有成本
                objective.SetCoefficient(variables['inventory'][item][location][period],
                                        self.holding_cost[item][location])
                # 缺货成本
                objective.SetCoefficient(variables['stockout'][item][location][period],
                                        self.stockout_cost[item][location])
    
    objective.SetMinimization()
```

### 数据源扩展

1. **创建数据源类**：在`data_source.py`中创建新的数据源类
   ```python
   class CustomDataSource(BaseDataSource):
       def __init__(self, config):
           super().__init__(config)
           # 初始化自定义数据源
           
       def connect(self):
           # 实现连接逻辑
           pass
           
       def load_data(self, table_name):
           # 实现数据加载逻辑
           pass
   ```

2. **注册数据源**：在`DataSourceFactory`类中注册新数据源
   ```python
   def create_data_source(self, source_type, config):
       if source_type == 'csv':
           return CSVDataSource(config)
       elif source_type == 'database':
           return DatabaseDataSource(config)
       elif source_type == 'api':
           return APIDataSource(config)
       elif source_type == 'custom':
           return CustomDataSource(config)  # 添加新数据源
       else:
           raise ValueError(f"Unsupported data source type: {source_type}")
   ```

### 补货策略扩展

1. **创建策略类**：在`automated_replenishment.py`中创建新的策略类
   ```python
   class CustomReplenishmentStrategy(BaseReplenishmentStrategy):
       def __init__(self, params):
           super().__init__(params)
           
       def calculate_order_quantity(self, item_data):
           # 实现自定义补货策略逻辑
           # 返回建议的订货量
           pass
   ```

2. **注册策略**：在`ReplenishmentManager`类中注册新策略
   ```python
   def __init__(self):
       self.strategies = {
           'rop': ROPReplenishmentStrategy,
           'order_up_to': OrderUpToReplenishmentStrategy,
           'hybrid': HybridReplenishmentStrategy,
           'custom': CustomReplenishmentStrategy,  # 添加新策略
       }
   ```

### API扩展

在`api.py`中添加新的API端点：

```python
@app.get("/api/custom_endpoint")
async def custom_endpoint(param1: str, param2: int = 10):
    """自定义API端点"""
    # 实现API逻辑
    result = do_something(param1, param2)
    return {"status": "success", "data": result}
```

### 配置扩展

在`.env`文件中添加自定义配置项：

```env
# 自定义配置
CUSTOM_CONFIG_1=value1
CUSTOM_CONFIG_2=value2
```

在代码中使用自定义配置：

```python
from src.system.config_manager import ConfigManager

config = ConfigManager()
custom_value = config.get('CUSTOM_CONFIG_1')
```

## 🛠️ 技术栈

### 核心技术

| 类别 | 技术/库 | 用途 |
|------|---------|------|
| **编程语言** | Python 3.7+ | 主要开发语言 |
| **数据处理** | Pandas, NumPy | 数据结构和数值计算 |
| **机器学习** | scikit-learn, XGBoost | 预测模型和特征工程 |
| **统计分析** | statsmodels | 时间序列分析和统计模型 |
| **优化算法** | OR-Tools | MILP混合整数线性规划 |
| **API服务** | FastAPI, Uvicorn | REST API开发和部署 |
| **数据可视化** | Matplotlib, Seaborn | 数据和结果可视化 |
| **模型可解释性** | SHAP, LIME | 模型预测结果解释 |

### 辅助技术

| 类别 | 技术/库 | 用途 |
|------|---------|------|
| **数据存储** | SQLAlchemy, PostgreSQL | 数据库连接和管理 |
| **缓存系统** | Redis | 分布式缓存 |
| **配置管理** | python-dotenv | 环境变量配置 |
| **特征工程** | feature-engine | 自动化特征工程 |
| **模型序列化** | joblib, pickle | 模型保存和加载 |
| **测试框架** | pytest | 单元测试和集成测试 |
| **API文档** | Swagger UI | 自动生成API文档 |
| **容器化** | Docker, Docker Compose | 应用容器化和部署 |

### 开发工具

- **IDE**：PyCharm, VS Code
- **版本控制**：Git
- **CI/CD**：GitHub Actions
- **监控工具**：Prometheus, Grafana (可选)
- **日志管理**：ELK Stack (可选)

## 📄 许可证

本项目采用MIT许可证。详细信息请参阅LICENSE文件。

## 📞 联系方式

如有问题或建议，请联系项目维护团队：

- **项目负责人**：[负责人姓名]
- **电子邮件**：[email@example.com]
- **GitHub**：[GitHub Repository URL]
- **文档**：[项目文档URL]

## 📝 更新日志

### v1.0.0 (2023-XX-XX)
- 初始版本发布
- 实现核心功能：智能预测、补货策略、MILP优化
- 支持多种数据源和模型
- 提供API服务和可视化仪表盘

### v1.1.0 (2023-XX-XX)
- 新增Feature Store功能
- 增强数据接入层，支持更多数据源类型
- 优化MILP求解器性能
- 添加模型可解释性模块

### v1.2.0 (2023-XX-XX)
- 实现MLOps引擎，支持模型监控和自动化更新
- 增强API功能，提供更多数据接口
- 优化系统性能和稳定性
- 添加A/B测试框架

## 📚 参考资料

1. **时间序列预测**：
   - [Forecasting: Principles and Practice](https://otexts.com/fpp3/)
   - [StatsModels Documentation](https://www.statsmodels.org/stable/index.html)

2. **库存优化**：
   - [Inventory Management: Theory and Practice](https://www.springer.com/gp/book/9783319412641)
   - [Operations Research: Applications and Algorithms](https://www.cengage.com/c/operations-research-applications-and-algorithms-5e-winston/9781305638631)

3. **MLOps**：
   - [Building Machine Learning Pipelines](https://www.oreilly.com/library/view/building-machine-learning/9781492045106/)
   - [Machine Learning Engineering](https://www.oreilly.com/library/view/machine-learning-engineering/9781098107956/)

4. **技术文档**：
   - [FastAPI Documentation](https://fastapi.tiangolo.com/)
   - [OR-Tools Documentation](https://developers.google.com/optimization)
   - [Pandas Documentation](https://pandas.pydata.org/docs/)

## 许可证

本项目采用MIT许可证。

## 联系方式

如有问题或建议，请联系项目维护人员。
