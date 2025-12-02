# 供应链库存优化系统 - 部署文档

## 目录

1. [概述](#概述)
2. [前提条件](#前提条件)
3. [架构组件](#架构组件)
4. [部署步骤](#部署步骤)
5. [服务访问](#服务访问)
6. [监控配置](#监控配置)
7. [事件驱动架构](#事件驱动架构)
8. [常见问题排查](#常见问题排查)
9. [扩展与维护](#扩展与维护)

## 概述

本项目采用微服务架构设计，包含以下核心组件：

- API网关：统一入口，路由请求
- 数据服务：数据加载、预处理和管理
- 特征服务：特征计算和模型选择标签生成
- 预测服务：需求预测和模型管理
- 优化服务：库存优化和采购订单生成
- 综合服务：整合所有服务，提供完整补货流程

## 前提条件

在部署前，请确保您的环境满足以下条件：

- Docker 20.10+ 
- Docker Compose 1.29+ 或 Docker Desktop 2.2.0+ 
- 至少 4GB 可用内存
- 至少 20GB 可用磁盘空间
- Git（可选，用于克隆代码）
- 如果使用数据库数据源：PostgreSQL 12+ 或其他兼容数据库
- 如果使用API数据源：可访问的API服务和有效的API令牌
- 如果使用Redis缓存：Redis 6.0+

## 架构组件

### 1. API网关

- **端口**：8000
- **功能**：统一服务入口，请求路由，负载均衡
- **访问地址**：http://localhost:8000

### 2. 数据接入层

- **功能**：支持多种数据源接入
- **支持的数据源类型**：
  - CSV文件
  - 数据库（PostgreSQL、MySQL等）
  - REST API
  - 模拟数据（用于测试和演示）
- **核心组件**：
  - 数据源工厂：动态创建数据源实例
  - 数据转换器：确保数据格式一致性
  - 数据验证器：验证数据完整性和合法性

### 3. 数据服务

- **端口**：8001
- **功能**：数据加载、预处理、分割和管理
- **访问地址**：http://localhost:8001

### 4. 特征服务

- **端口**：8002
- **功能**：特征计算、特征存储和模型选择标签生成
- **访问地址**：http://localhost:8002

### 5. 预测服务

- **端口**：8003
- **功能**：需求预测、模型选择和模型管理
- **访问地址**：http://localhost:8003

### 6. 优化服务

- **端口**：8004
- **功能**：库存优化、MILP模型求解和采购订单生成
- **访问地址**：http://localhost:8004

### 7. 综合服务

- **端口**：8005
- **功能**：整合所有服务，提供完整的补货流程
- **访问地址**：http://localhost:8005

### 8. 支持服务

- **Redis**：端口 6379，用于事件驱动架构和缓存
- **RabbitMQ**：端口 5672（AMQP）和 15672（管理界面），用于可靠消息传递
- **Prometheus**：端口 9090，用于监控数据收集
- **Grafana**：端口 3000，用于监控数据可视化

## 部署步骤

### 1. 克隆代码（可选）

```bash
git clone <repository-url>
cd <project-directory>
```

### 2. 配置环境变量

创建 `.env` 文件，配置数据源和系统参数：

```bash
cp config_example.env .env
```

编辑 `.env` 文件，根据实际情况配置以下参数：

```
# 数据源配置
DATA_SOURCE_TYPE=csv  # 可选值: csv, database, api

# CSV数据源配置
DATA_DIR=data

# 数据库数据源配置
# DATABASE_CONNECTION_STRING=postgresql://username:password@localhost:5432/supplychain

# API数据源配置
# API_BASE_URL=http://localhost:8000/api
# API_TOKEN=your_api_token

# 缓存配置
# CACHE_TYPE=redis  # 可选值: memory, redis
# REDIS_URL=redis://localhost:6379/0

# 并行计算配置
PARALLEL_MODE=true

# GPU加速配置
GPU_ENABLED=false

# 日志级别
LOG_LEVEL=INFO
```

### 3. 准备数据

根据配置的数据源类型，准备相应的数据：

#### CSV数据源
确保 `data` 目录中包含所需的数据文件。如果没有数据文件，可以运行以下命令生成示例数据：

```bash
python src/system/main.py  # 主程序会自动生成示例数据
```

#### 数据库数据源
确保数据库已创建，并且包含所需的表结构和数据。

#### API数据源
确保API服务可访问，并且有有效的API令牌。

### 4. 构建和启动服务

使用 Docker Compose 构建和启动所有服务：

```bash
# 方法 1：使用 docker-compose 命令（旧版 Docker）
docker-compose up -d

# 方法 2：使用 docker compose 命令（新版 Docker）
docker compose up -d
```

### 5. 数据接入层验证

启动服务后，可以验证数据接入层是否正常工作：

```bash
# 检查数据服务日志，确认数据加载成功
docker compose logs -f data-service

# 测试API端点，验证数据是否可以正常访问
curl http://localhost:8000/api/items
```

### 6. 检查服务状态

运行以下命令检查所有服务的状态：

```bash
# 方法 1
docker-compose ps

# 方法 2
docker compose ps
```

### 7. 查看日志

要查看特定服务的日志，请运行：

```bash
# 方法 1
docker-compose logs -f <service-name>

# 方法 2
docker compose logs -f <service-name>
```

例如，查看 API 网关的日志：

```bash
docker compose logs -f api-gateway
```

查看数据服务日志：

```bash
docker compose logs -f data-service
```

## 服务访问

### API 文档

所有服务都提供了 Swagger UI 文档，可以通过以下地址访问：

- API 网关：http://localhost:8000/docs
- 数据服务：http://localhost:8001/docs
- 特征服务：http://localhost:8002/docs
- 预测服务：http://localhost:8003/docs
- 优化服务：http://localhost:8004/docs
- 综合服务：http://localhost:8005/docs

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

## 监控配置

### Prometheus

Prometheus 用于收集所有服务的监控数据，访问地址：http://localhost:9090

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

## 事件驱动架构

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

## 常见问题排查

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

## 扩展与维护

### 1. 横向扩展

可以通过以下命令扩展特定服务的实例数量：

```bash
docker compose up -d --scale <service-name>=<number-of-instances>
```

例如，扩展预测服务到 3 个实例：

```bash
docker compose up -d --scale forecast-service=3
```

### 2. 更新服务

当代码更新后，可以通过以下步骤更新服务：

1. 拉取最新代码：`git pull`
2. 重新构建和启动服务：`docker compose up -d --build`

### 3. 备份数据

定期备份以下目录：

- `data/`：原始数据和预处理数据
- `models/`：训练好的模型
- `features/`：计算好的特征

### 4. 清理资源

如果需要停止并清理所有资源，可以运行：

```bash
docker compose down -v
```

## 结束

恭喜！您已成功部署供应链库存优化系统。如果您遇到任何问题，请参考[常见问题排查](#常见问题排查)部分，或联系技术支持团队。
