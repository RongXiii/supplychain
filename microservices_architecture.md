# 供应链智能补货系统 - 微服务架构设计

## 1. 架构概述

本设计将原有的单体应用拆分为多个微服务，实现松耦合、高可用、可扩展的架构。采用API网关统一管理外部请求，服务间通过REST API或消息队列通信。

## 2. 微服务拆分方案

### 2.1 服务拆分清单

| 微服务名称 | 功能职责 | 技术栈 | 端口 |
|-----------|---------|-------|------|
| API Gateway | 请求路由、负载均衡、认证授权 | FastAPI + Uvicorn | 8000 |
| Data Service | 数据加载、清洗、转换、预处理 | FastAPI + Pandas | 8001 |
| Feature Service | 特征计算、特征存储、模型选择标签生成 | FastAPI + feature-engine | 8002 |
| Forecast Service | 需求预测、模型管理、模型选择 | FastAPI + scikit-learn + statsmodels | 8003 |
| Optimization Service | MILP优化、补货计划生成、安全库存计算 | FastAPI + OR-Tools | 8004 |
| Monitoring Service | 系统监控、日志管理、性能指标收集 | FastAPI + Prometheus | 8005 |

### 2.2 服务依赖关系

```
┌─────────────────┐
│   API Gateway   │
└─────────┬───────┘
          │
          ▼
┌─────────────────────────────────────────────────────────┐
│                                                         │
▼                                                         ▼
┌───────────────┐                               ┌─────────────────┐
│ Data Service  │                               │ Monitoring      │
└─────────┬─────┘                               │ Service         │
          │                                     └─────────────────┘
          ▼
┌─────────────────┐
│ Feature Service │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Forecast        │
│ Service         │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Optimization    │
│ Service         │
└─────────────────┘
```

## 3. API网关设计

### 3.1 核心功能

- 请求路由：将外部请求路由到相应的微服务
- 负载均衡：支持多种负载均衡策略
- 认证授权：JWT认证、API密钥管理
- 限流熔断：防止系统过载
- 日志记录：统一日志格式和存储
- 监控统计：收集API调用 metrics

### 3.2 路由配置

| 外部API路径 | 内部服务 | 内部路径 |
|------------|---------|----------|
| /api/data/* | Data Service | /* |
| /api/features/* | Feature Service | /* |
| /api/forecast/* | Forecast Service | /* |
| /api/optimize/* | Optimization Service | /* |
| /api/monitor/* | Monitoring Service | /* |

## 4. 容器化部署方案

### 4.1 Docker镜像设计

每个微服务将构建为独立的Docker镜像，使用Docker Compose进行本地开发和测试，使用Kubernetes进行生产部署。

### 4.2 Docker Compose配置

```yaml
version: '3.8'

services:
  api-gateway:
    build: ./api-gateway
    ports:
      - "8000:8000"
    depends_on:
      - data-service
      - feature-service
      - forecast-service
      - optimization-service
      - monitoring-service
    networks:
      - supplychain-network

  data-service:
    build: ./services/data-service
    ports:
      - "8001:8001"
    volumes:
      - ./data:/app/data
    networks:
      - supplychain-network

  feature-service:
    build: ./services/feature-service
    ports:
      - "8002:8002"
    volumes:
      - ./features:/app/features
      - ./data:/app/data
    networks:
      - supplychain-network

  forecast-service:
    build: ./services/forecast-service
    ports:
      - "8003:8003"
    volumes:
      - ./models:/app/models
      - ./data:/app/data
      - ./features:/app/features
    networks:
      - supplychain-network

  optimization-service:
    build: ./services/optimization-service
    ports:
      - "8004:8004"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./features:/app/features
    networks:
      - supplychain-network

  monitoring-service:
    build: ./services/monitoring-service
    ports:
      - "8005:8005"
    volumes:
      - ./logs:/app/logs
    networks:
      - supplychain-network

networks:
  supplychain-network:
    driver: bridge
```

## 5. 事件驱动架构设计

### 5.1 事件类型

| 事件名称 | 事件描述 | 生产者 | 消费者 |
|---------|---------|-------|--------|
| data_updated | 数据更新完成 | Data Service | Feature Service, Forecast Service |
| features_calculated | 特征计算完成 | Feature Service | Forecast Service, Optimization Service |
| forecast_generated | 预测结果生成 | Forecast Service | Optimization Service |
| plan_optimized | 优化计划生成 | Optimization Service | Monitoring Service |
| model_retrained | 模型重新训练 | Forecast Service | Monitoring Service |

### 5.2 消息队列配置

使用Redis或RabbitMQ作为消息队列，实现服务间的异步通信。

## 6. 部署流程

### 6.1 本地开发环境

```bash
# 构建所有镜像
docker-compose build

# 启动所有服务
docker-compose up

# 停止所有服务
docker-compose down
```

### 6.2 生产环境部署

1. 构建Docker镜像并推送到容器 registry
2. 使用Helm或kubectl部署到Kubernetes集群
3. 配置Ingress Controller和Service Mesh
4. 配置监控和告警

## 7. 监控与日志

### 7.1 监控指标

- API请求量、响应时间、错误率
- 服务CPU、内存、磁盘使用率
- 模型预测准确率、MAPE、RMSE
- 优化求解时间、约束满足情况

### 7.2 日志管理

- 统一日志格式：JSON格式，包含时间戳、服务名称、日志级别、请求ID、消息内容
- 日志收集：使用ELK Stack或Prometheus + Grafana
- 日志保留策略：根据业务需求配置

## 8. 安全性设计

### 8.1 API安全

- API密钥认证
- JWT令牌授权
- HTTPS加密传输
- 输入验证和参数校验
- 防止SQL注入和XSS攻击

### 8.2 服务间安全

- 内部服务通信加密
- 服务身份验证
- 访问控制列表（ACL）

## 9. 扩展性设计

### 9.1 水平扩展

- 无状态服务支持水平扩展
- 使用负载均衡器分发请求
- 数据库分片和读写分离

### 9.2 垂直扩展

- 针对CPU密集型服务（如优化服务），增加CPU资源
- 针对内存密集型服务（如数据服务），增加内存资源

## 10. 迁移计划

### 10.1 分阶段迁移

1. 构建API网关，将现有API路由到单体应用
2. 逐步拆分核心功能为微服务
3. 迁移数据存储到分布式数据库
4. 实现事件驱动架构
5. 完全替换单体应用

### 10.2 回滚策略

- 保留单体应用的备份
- 实现灰度发布
- 配置健康检查和自动回滚
- 建立完善的测试机制

## 11. 测试策略

### 11.1 单元测试

- 每个微服务独立进行单元测试
- 测试覆盖率达到80%以上

### 11.2 集成测试

- 测试服务间通信
- 测试API网关路由
- 测试事件驱动流程

### 11.3 端到端测试

- 测试完整的业务流程
- 测试系统在高负载下的性能
- 测试故障恢复能力

## 12. 文档管理

- API文档：使用Swagger/OpenAPI自动生成
- 服务文档：每个服务独立编写README.md
- 架构文档：定期更新架构设计图和说明
- 部署文档：详细的部署步骤和配置说明
