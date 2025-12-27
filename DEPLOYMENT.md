# 供应链库存优化系统 - 部署文档

## 📋 目录

1. [📖 概述](#📖-概述)
2. [📋 前提条件](#📋-前提条件)
3. [🏗️ 架构组件](#🏗️-架构组件)
4. [🚀 部署步骤](#🚀-部署步骤)
5. [🔌 服务访问](#🔌-服务访问)
6. [📊 监控配置](#📊-监控配置)
7. [📡 事件驱动架构](#📡-事件驱动架构)
8. [❗ 常见问题排查](#❗-常见问题排查)
9. [🔧 扩展与维护](#🔧-扩展与维护)

## 📖 概述

本项目采用现代化微服务架构设计，提供完整的供应链库存优化解决方案。系统包含以下核心组件：

- **API网关**：统一服务入口，请求路由，负载均衡，API文档管理
- **数据服务**：多源数据加载、预处理、数据清洗和数据管理
- **特征服务**：特征工程、特征计算和模型选择标签生成
- **预测服务**：智能需求预测、模型选择和模型生命周期管理
- **优化服务**：MILP库存优化、采购订单生成和优化决策支持
- **综合服务**：整合所有服务，提供端到端补货流程支持

## 📊 部署流程图

```
┌─────────────────────────────────────────────────────────────────────┐
│                        部署准备阶段                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐  │
│  │  环境检查    │  │  依赖安装    │  │  代码克隆    │  │  数据准备  │  │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └─────┬─────┘  │
│         └────────────────┼─────────────────┼────────────────┘       │
└──────────────────────────┼─────────────────┼─────────────────────────┘
                           ▼                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        配置阶段                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐  │
│  │  环境变量配置  │  │  数据源配置   │  │  缓存配置    │  │  日志配置  │  │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └─────┬─────┘  │
│         └────────────────┼─────────────────┼────────────────┘       │
└──────────────────────────┼─────────────────┼─────────────────────────┘
                           ▼                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        构建与部署阶段                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐  │
│  │  镜像构建     │  │  容器启动     │  │  网络配置    │  │  健康检查  │  │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └─────┬─────┘  │
│         └────────────────┼─────────────────┼────────────────┘       │
└──────────────────────────┼─────────────────┼─────────────────────────┘
                           ▼                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        验证与维护阶段                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐  │
│  │  API验证     │  │  功能测试     │  │  监控配置    │  │  日志查看  │  │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └─────┬─────┘  │
│         └────────────────┼─────────────────┼────────────────┘       │
└──────────────────────────┼─────────────────┼─────────────────────────┘
                           ▼                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        部署完成                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## 📋 前提条件

在部署前，请确保您的环境满足以下条件：

### 基础环境要求
- **容器化环境**：
  - Docker 20.10.0+（推荐使用最新稳定版本）
  - Docker Compose 1.29.0+ 或 Docker Desktop 2.2.0+（Windows/macOS）
- **硬件资源**：
  - 至少 4GB 可用内存（推荐 8GB+）
  - 至少 20GB 可用磁盘空间
  - CPU：4核心或更高（支持并行计算）
- **开发工具**：
  - Git（可选，用于版本控制和代码克隆）

### 可选依赖服务
- **数据库**：PostgreSQL 12+ 或 MySQL 8+（用于持久化数据存储）
- **缓存服务**：Redis 6.0+（用于提高性能和事件驱动架构）
- **消息队列**：RabbitMQ 3.8+（用于可靠消息传递）
- **API服务**：可访问的外部API服务和有效的API令牌（如果使用API数据源）

## 🏗️ 架构组件

### 1. API网关
- **端口**：8000
- **功能**：统一服务入口，请求路由，负载均衡，API文档管理
- **访问地址**：http://localhost:8000
- **API文档**：http://localhost:8000/docs

### 2. 数据服务层

#### 数据接入服务
- **功能**：支持多种数据源接入和数据格式转换
- **支持的数据源类型**：
  - CSV/Excel文件
  - 关系型数据库（PostgreSQL、MySQL等）
  - REST API服务
  - 模拟数据（用于测试和演示）
- **核心组件**：
  - 数据源工厂：动态创建数据源实例
  - 数据转换器：确保数据格式一致性
  - 数据验证器：验证数据完整性和合法性

#### 数据处理服务
- **端口**：8001
- **功能**：数据加载、预处理、清洗、分割和管理
- **访问地址**：http://localhost:8001

### 3. 特征与预测层

#### 特征服务
- **端口**：8002
- **功能**：特征工程、特征计算、特征存储和模型选择标签生成
- **访问地址**：http://localhost:8002

#### 预测服务
- **端口**：8003
- **功能**：智能需求预测、模型选择、模型训练和模型生命周期管理
- **访问地址**：http://localhost:8003

### 4. 优化与决策层

#### 优化服务
- **端口**：8004
- **功能**：MILP库存优化、采购订单生成和优化决策支持
- **访问地址**：http://localhost:8004

#### 综合服务
- **端口**：8005
- **功能**：整合所有服务，提供端到端补货流程支持
- **访问地址**：http://localhost:8005

### 5. 支持服务层

| 服务名称 | 端口 | 功能 | 访问地址 |
|---------|------|------|----------|
| Redis | 6379 | 缓存和事件驱动架构 | redis://localhost:6379 |
| RabbitMQ | 5672/15672 | 消息队列和管理界面 | http://localhost:15672 |
| Prometheus | 9090 | 监控数据收集 | http://localhost:9090 |
| Grafana | 3000 | 监控数据可视化 | http://localhost:3000 |

## 🚀 部署步骤

### 1. 代码获取（可选）

如果使用Git进行版本控制，可以克隆代码库：

```bash
git clone <repository-url>
cd <project-directory>
```

或者直接下载项目压缩包并解压到目标目录。

### 2. 环境检查与验证

在部署前，执行以下命令验证环境配置：

```bash
# 检查Docker版本
docker --version

# 检查Docker Compose版本
docker compose version  # 或 docker-compose --version (旧版)

# 检查系统资源（Linux/macOS）
free -h && df -h

# 检查系统资源（Windows PowerShell）
systeminfo | findstr /I "Memory Disk"
```

### 3. 配置文件设置

#### 环境变量配置

创建并编辑 `.env` 文件，配置数据源和系统参数：

```bash
# 复制示例配置文件
cp config_example.env .env

# 使用文本编辑器编辑配置
nano .env  # Linux/macOS
notepad .env  # Windows
```

主要配置项说明：

```
# 基础配置
SERVICE_NAME=supplychain-optimization-system
ENVIRONMENT=development  # development, production, testing

# 数据源配置
DATA_SOURCE_TYPE=csv  # csv, database, api, mock

# CSV数据源配置
DATA_DIR=data
DATA_FILE_PATTERN=*.csv

# 数据库数据源配置（需要时取消注释）
# DATABASE_CONNECTION_STRING=postgresql://username:password@localhost:5432/supplychain
# DATABASE_TABLE_PREFIX=supplychain_

# 缓存配置
CACHE_TYPE=memory  # memory, redis
REDIS_URL=redis://localhost:6379/0
CACHE_TTL=3600

# 计算资源配置
PARALLEL_MODE=true
NUM_WORKERS=4
GPU_ENABLED=false
GPU_IDS=0

# 日志与监控配置
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
METRICS_ENABLED=true
```

### 4. 数据准备

根据配置的数据源类型，准备相应的数据：

#### CSV数据源

确保 `data` 目录中包含所需的数据文件：

```bash
# 创建数据目录
mkdir -p data

# 查看数据目录内容
ls -la data/
```

如果没有数据文件，可以生成示例数据：

```bash
python src/system/main.py --generate-data
```

生成的示例数据包括：
- `demand_data.csv`：历史需求数据
- `product_info.csv`：产品信息
- `warehouse_info.csv`：仓库信息

#### 数据库数据源

确保数据库已创建，并且包含所需的表结构：

```bash
# PostgreSQL示例 - 创建数据库
createdb supplychain

# 导入表结构和示例数据
psql -d supplychain -f data/database_schema.sql
psql -d supplychain -f data/sample_data.sql
```

#### API数据源

确保API服务可访问，并验证连接：

```bash
# 测试API连接
curl -H "Authorization: Bearer <your_api_token>" <api_base_url>/health
```

#### 模拟数据源

如果只是测试系统功能，可以使用内置的模拟数据源：

```bash
# 在.env文件中设置
DATA_SOURCE_TYPE=mock
```

### 5. 服务构建与启动

使用 Docker Compose 构建和启动所有服务：

```bash
# 构建并启动所有服务（后台模式）
docker compose up -d --build

# 查看服务启动日志
docker compose logs -f
```

#### 选择性启动服务

如果只需要启动部分服务，可以指定服务名称：

```bash
# 只启动API网关和数据服务
docker compose up -d api-gateway data-service

# 启动预测和优化服务
docker compose up -d forecast-service optimization-service
```

### 6. 服务验证与测试

启动服务后，验证所有服务是否正常工作：

```bash
# 检查所有服务状态
docker compose ps

# 检查API网关健康状态
curl http://localhost:8000/health

# 测试数据API端点
curl http://localhost:8000/api/items

# 测试预测API端点
curl http://localhost:8000/api/forecast

# 访问API文档
open http://localhost:8000/docs  # macOS
start http://localhost:8000/docs  # Windows
```

### 7. 日志管理与故障排查

查看服务日志是排查问题的重要手段：

```bash
# 查看单个服务日志（实时）
docker compose logs -f api-gateway

# 查看所有服务日志（最新100行）
docker compose logs --tail 100

# 查看特定时间范围的日志
docker compose logs --since "2023-01-01T10:00:00" --until "2023-01-01T11:00:00"

# 将日志输出到文件
docker compose logs > deployment.log
```

## 🔌 服务访问

### API 文档

所有服务都提供了 Swagger UI 文档，可以通过以下地址访问：

| 服务名称 | API文档地址 |
|---------|------------|
| API网关 | http://localhost:8000/docs |
| 数据服务 | http://localhost:8001/docs |
| 特征服务 | http://localhost:8002/docs |
| 预测服务 | http://localhost:8003/docs |
| 优化服务 | http://localhost:8004/docs |
| 综合服务 | http://localhost:8005/docs |

### 示例请求

以下是一个完整补货流程的示例请求：

```bash
curl -X POST "http://localhost:8000/api/replenishment/full" \
  -H "Content-Type: application/json" \
  -d '{
    "data_config": {
      "data_file": "demand_data.csv",
      "product_file": "product_info.csv",
      "warehouse_file": "warehouse_info.csv"
    },
    "current_inventory": {
      "1": 100,
      "2": 200,
      "3": 150
    },
    "lead_times": {
      "1": 1,
      "2": 2,
      "3": 1
    },
    "costs": {
      "ordering_cost": [100, 150, 120],
      "holding_cost": [10, 12, 11],
      "shortage_cost": [100, 120, 110],
      "unit_cost": [100, 150, 120]
    },
    "constraints": {
      "max_order_quantity": [1000, 1500, 1200],
      "min_order_quantity": [0, 0, 0],
      "safety_stock_target": [0.1, 0.15, 0.12],
      "service_level": 0.95
    }
  }'
```

## 📊 监控配置

### Prometheus

Prometheus 用于收集所有服务的监控数据，访问地址：http://localhost:9090

**常用查询**：
```
# 服务请求数
sum(rate(http_requests_total[5m])) by (service, endpoint, method, status)

# 服务响应时间
histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (service, endpoint, le))

# 服务错误率
sum(rate(http_requests_total{status="500"}[5m])) / sum(rate(http_requests_total[5m])) by (service, endpoint)
```

### Grafana

Grafana 用于可视化监控数据，访问地址：http://localhost:3000

**默认登录信息**：
- 用户名：admin
- 密码：admin

#### 配置数据源

1. 登录 Grafana
2. 点击左侧菜单的 "Configuration" -> "Data Sources"
3. 点击 "Add data source"
4. 选择 "Prometheus"
5. 在 "HTTP" 部分，设置 "URL" 为 `http://prometheus:9090`
6. 点击 "Save & Test"

#### 导入仪表盘

1. 点击左侧菜单的 "Create" -> "Import"
2. 输入仪表盘 ID 或上传 JSON 文件
3. 选择 Prometheus 数据源
4. 点击 "Import"

## 📡 事件驱动架构

本项目使用 RabbitMQ 和 Redis 实现事件驱动架构：

### RabbitMQ

- 访问地址：http://localhost:15672
- 默认登录信息：admin/admin

### Redis

- 端口：6379
- 可使用 Redis CLI 或 Redis Desktop Manager 访问

### 事件类型

1. **数据更新事件**：当数据发生变化时触发
2. **特征更新事件**：当特征计算完成时触发
3. **预测完成事件**：当预测任务完成时触发
4. **优化完成事件**：当优化任务完成时触发
5. **模型更新事件**：当模型训练完成时触发

## ❗ 常见问题排查

### 1. 服务启动失败

**问题**：部分服务启动失败，日志显示连接超时

**解决方案**：
- 检查 Docker 资源限制，确保有足够的内存和 CPU
- 检查网络配置，确保所有服务在同一网络中
- 尝试重启服务：`docker compose restart <service-name>`

### 2. 数据加载失败

**问题**：数据服务无法加载数据

**解决方案**：
- 检查数据源类型配置是否正确：`DATA_SOURCE_TYPE=csv`
- 对于CSV数据源：
  - 检查数据文件是否存在于配置的 `DATA_DIR` 目录中
  - 检查文件权限，确保 Docker 容器可以访问
  - 检查数据文件格式是否正确
- 对于数据库数据源：
  - 检查数据库连接字符串是否正确
  - 检查数据库服务是否运行
  - 检查数据库用户是否有足够的权限
- 对于API数据源：
  - 检查API基础URL是否正确
  - 检查API令牌是否有效
  - 检查API服务是否可访问

### 3. 预测服务报错

**问题**：预测服务无法运行，日志显示模型选择失败

**解决方案**：
- 检查特征服务是否正常运行
- 检查模型标签是否正确
- 确保有足够的历史数据用于训练

### 4. 优化服务运行缓慢

**问题**：优化服务运行时间过长

**解决方案**：
- 调整 MILP 模型的约束条件，减少变量数量
- 增加优化服务的资源限制
- 考虑使用更强大的求解器

### 5. 数据源切换失败

**问题**：切换数据源类型后，系统无法正常运行

**解决方案**：
- 确保 `.env` 文件中的 `DATA_SOURCE_TYPE` 配置正确
- 确保对应数据源的连接信息配置正确
- 重启所有服务：`docker compose restart`
- 检查日志，确认具体错误信息

### 6. 缓存服务无法连接

**问题**：系统无法连接到Redis缓存服务

**解决方案**：
- 检查Redis服务是否运行
- 检查 `REDIS_URL` 配置是否正确
- 检查网络配置，确保服务可以访问Redis
- 考虑切换到内存缓存：`CACHE_TYPE=memory`

## 🔧 扩展与维护

### 1. 横向扩展

可以通过以下命令扩展特定服务的实例数量：

```bash
docker compose up -d --scale <service-name>=<number-of-instances>
```

**示例：**

```bash
# 扩展预测服务到 3 个实例
docker compose up -d --scale forecast-service=3

# 扩展优化服务到 2 个实例
docker compose up -d --scale optimization-service=2
```

**注意事项：**
- 扩展只适用于无状态服务
- API网关会自动负载均衡到多个实例

### 2. 更新服务

当代码更新后，可以通过以下步骤更新服务：

```bash
# 拉取最新代码
git pull

# 重新构建和启动服务
docker compose up -d --build

# 验证服务是否正常
docker compose ps
curl http://localhost:8000/health
```

### 3. 备份数据

定期备份以下目录：

```bash
# 创建备份目录
mkdir -p backups/$(date +%Y-%m-%d)

# 备份数据目录
cp -r data/ backups/$(date +%Y-%m-%d)/data/

# 备份模型目录
cp -r models/ backups/$(date +%Y-%m-%d)/models/

# 备份特征目录
cp -r features/ backups/$(date +%Y-%m-%d)/features/

# 备份配置文件
cp .env backups/$(date +%Y-%m-%d)/.env
```

## 🎉 部署完成

恭喜！您已成功部署供应链库存优化系统。如果您遇到任何问题，请参考[常见问题排查](#常见问题排查)部分，或联系技术支持团队。

**快速访问链接：**
- API网关：http://localhost:8000
- API文档：http://localhost:8000/docs
- 监控面板：http://localhost:3000
- 消息队列管理：http://localhost:15672

**下一步建议：**
1. 探索API文档，了解系统功能
2. 配置监控告警
3. 进行功能测试
4. 根据业务需求调整配置参数

## 结束

恭喜！您已成功部署供应链库存优化系统。如果您遇到任何问题，请参考[常见问题排查](#常见问题排查)部分，或联系技术支持团队。
