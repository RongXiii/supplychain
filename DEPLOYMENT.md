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

## 架构组件

### 1. API网关

- **端口**：8000
- **功能**：统一服务入口，请求路由，负载均衡
- **访问地址**：http://localhost:8000

### 2. 数据服务

- **端口**：8001
- **功能**：数据加载、预处理、分割和管理
- **访问地址**：http://localhost:8001

### 3. 特征服务

- **端口**：8002
- **功能**：特征计算、特征存储和模型选择标签生成
- **访问地址**：http://localhost:8002

### 4. 预测服务

- **端口**：8003
- **功能**：需求预测、模型选择和模型管理
- **访问地址**：http://localhost:8003

### 5. 优化服务

- **端口**：8004
- **功能**：库存优化、MILP模型求解和采购订单生成
- **访问地址**：http://localhost:8004

### 6. 综合服务

- **端口**：8005
- **功能**：整合所有服务，提供完整的补货流程
- **访问地址**：http://localhost:8005

### 7. 支持服务

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

### 2. 准备数据

确保 `data` 目录中包含所需的数据文件。如果没有数据文件，可以运行以下命令生成示例数据：

```bash
python src/generate_sample_data.py
```

### 3. 构建和启动服务

使用 Docker Compose 构建和启动所有服务：

```bash
# 方法 1：使用 docker-compose 命令（旧版 Docker）
docker-compose up -d

# 方法 2：使用 docker compose 命令（新版 Docker）
docker compose up -d
```

### 4. 检查服务状态

运行以下命令检查所有服务的状态：

```bash
# 方法 1
docker-compose ps

# 方法 2
docker compose ps
```

### 5. 查看日志

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

**问题**：数据服务无法加载数据文件

**解决方案**：
- 检查数据文件是否存在于 `data` 目录中
- 检查文件权限，确保 Docker 容器可以访问
- 检查数据文件格式是否正确

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
