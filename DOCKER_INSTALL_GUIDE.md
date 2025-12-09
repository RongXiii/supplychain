# 🐳 Windows系统Docker安装和配置指南

## 📋 目录

1. [概述](#一概述)
2. [系统要求](#二系统要求)
3. [Docker安装准备](#三docker安装准备)
4. [Docker Desktop安装](#四docker-desktop安装)
5. [Docker高级配置](#五docker高级配置)
6. [安装验证](#六安装验证)
7. [Docker Compose使用](#七docker-compose使用)
8. [供应链系统容器化部署](#八供应链系统容器化部署)
9. [常见问题排查](#九常见问题排查)
10. [Docker最佳实践](#十docker最佳实践)
11. [容器化架构设计](#十一容器化架构设计)
12. [扩展与维护](#十二扩展与维护)

## 一、概述

本指南详细介绍了在Windows系统上安装、配置和使用Docker的完整流程，特别针对供应链智能补货系统的容器化部署需求进行了优化。通过本指南，您将能够：

✅ 安装并配置Docker Desktop
✅ 启用WSL 2以获得最佳性能
✅ 使用Docker Compose部署多容器应用
✅ 解决常见的Docker安装和运行问题
✅ 遵循Docker最佳实践进行容器化部署

## 二、系统要求

在安装Docker之前，请确保您的Windows系统满足以下要求：

### 2.1 操作系统要求

| 操作系统 | 版本要求 | 备注 |
|----------|----------|------|
| Windows 10 | 64位 Pro/Enterprise/Education (Build 15063+) | 家庭版需升级或使用Hyper-V |
| Windows 11 | 64位 Pro/Enterprise/Education | 所有最新版本均支持 |

### 2.2 硬件要求

- **CPU**：至少2核心，推荐4核心以上
- **内存**：至少4GB RAM，推荐8GB以上
- **存储**：至少20GB可用磁盘空间（Docker镜像和容器）
- **虚拟化支持**：Intel VT-x/EPT或AMD-V/RVI技术必须启用

### 2.3 软件要求

- WSL 2（Windows Subsystem for Linux）
- 支持的WSL分发版（如Ubuntu 20.04+）
- 最新的Windows更新补丁

## 三、Docker安装准备

### 3.1 启用虚拟化技术

1. **重启计算机**，进入BIOS设置（通常按F2、F10、Delete或Esc键）
2. 在BIOS中找到**虚拟化选项**（通常在"Advanced"或"Security"菜单中）
3. 启用**Intel VT-x/EPT**或**AMD-V/RVI**
4. 保存设置并退出BIOS

### 3.2 验证虚拟化状态

打开PowerShell（管理员权限），运行以下命令：

```powershell
Get-WmiObject -Class Win32_Processor | Select-Object -Property VirtualizationFirmwareEnabled
```

如果输出为`True`，表示虚拟化已启用。

### 3.3 安装WSL 2

1. **启用WSL功能**：
   ```powershell
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
   ```

2. **启用虚拟机平台**：
   ```powershell
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
   ```

3. **重启计算机**以应用更改

4. **下载并安装WSL 2内核更新包**：
   - [WSL 2内核更新包下载链接](https://wslstorestorage.blob.core.windows.net/wslblob/wsl_update_x64.msi)
   - 运行下载的安装程序并完成安装

5. **设置WSL 2为默认版本**：
   ```powershell
   wsl --set-default-version 2
   ```

### 3.4 安装Linux分发版（可选）

从Microsoft Store安装Ubuntu 22.04 LTS：

```powershell
winget install Canonical.Ubuntu.22.04 LTS
```

安装完成后，首次启动会提示设置用户名和密码。

## 四、Docker Desktop安装

### 4.1 下载Docker Desktop

访问[Docker官网](https://www.docker.com/products/docker-desktop)下载Windows版本的Docker Desktop安装包。

### 4.2 安装Docker Desktop

1. 双击下载的`Docker Desktop Installer.exe`文件
2. 在安装向导中，确保勾选以下选项：
   - ✅ "Install required Windows components for WSL 2"
   - ✅ "Use WSL 2 instead of Hyper-V (recommended)"
3. 点击"OK"开始安装
4. 安装完成后，点击"Close"并重启计算机

### 4.3 启动Docker Desktop

- 重启后，Docker Desktop将自动启动
- 您可以在系统托盘中找到Docker图标
- 首次启动会提示接受许可协议，点击"Accept"

## 安装Docker Desktop

### 步骤1：启用虚拟化和WSL 2

1. **启用虚拟化**：
   - 重启您的计算机
   - 进入BIOS设置（通常按F2、F10、Delete或Esc键）
   - 找到虚拟化选项（通常在"Advanced"或"Security"菜单中）
   - 启用Intel VT-x/EPT或AMD-V/RVI
   - 保存设置并退出BIOS

2. **启用WSL功能**：
   - 以管理员身份打开PowerShell
   - 运行以下命令启用WSL：
     ```powershell
     dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
     ```
   - 运行以下命令启用虚拟机平台：
     ```powershell
     dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
     ```
   - 重启计算机

3. **安装WSL 2内核更新包**：
   - 下载WSL 2内核更新包：[https://wslstorestorage.blob.core.windows.net/wslblob/wsl_update_x64.msi](https://wslstorestorage.blob.core.windows.net/wslblob/wsl_update_x64.msi)
   - 运行下载的安装程序并完成安装

4. **将WSL 2设置为默认版本**：
   - 以管理员身份打开PowerShell
   - 运行以下命令：
     ```powershell
     wsl --set-default-version 2
     ```

### 步骤2：安装Docker Desktop

1. **下载Docker Desktop安装程序**：
   - 访问Docker官网：[https://www.docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop)
   - 点击"Download for Windows"
   - 下载完成后，双击安装程序

2. **安装Docker Desktop**：
   - 按照安装向导的提示进行安装
   - 在"Configuration"页面，确保勾选以下选项：
     - "Install required Windows components for WSL 2"
     - "Use WSL 2 instead of Hyper-V (recommended)"
   - 点击"OK"并完成安装

3. **启动Docker Desktop**：
   - 安装完成后，Docker Desktop将自动启动
   - 您可以在系统托盘中找到Docker图标

## 五、Docker高级配置

### 5.1 登录Docker Hub

1. 点击系统托盘中的Docker图标
2. 选择"Sign in / Create Docker ID"
3. 登录您的Docker Hub账号，或创建一个新账号

### 5.2 资源配置优化

1. 点击系统托盘中的Docker图标 → "Settings"
2. 选择"Resources"选项卡，根据您的硬件配置调整：
   
   | 硬件配置 | CPU核心 | 内存 | 磁盘空间 |
   |----------|---------|------|----------|
   | 基础配置 | 2-4核 | 4-8GB | 20-50GB |
   | 推荐配置 | 4-8核 | 8-16GB | 50-100GB |
   | 高性能配置 | 8核以上 | 16GB以上 | 100GB以上 |

3. 点击"Apply & Restart"保存设置

### 5.3 WSL集成配置

1. 在Docker Desktop设置中，选择"Resources" > "WSL Integration"
2. 勾选"Enable integration with my default WSL distro"
3. 选择要与Docker集成的WSL分发版（如Ubuntu 22.04 LTS）
4. 点击"Apply & Restart"

### 5.4 镜像加速配置

为了解决国内网络访问Docker Hub缓慢的问题，配置镜像加速：

1. 在Docker Desktop设置中，选择"Docker Engine"
2. 添加以下配置：
   ```json
   {
     "registry-mirrors": [
       "https://registry.docker-cn.com",
       "https://docker.mirrors.ustc.edu.cn",
       "https://hub-mirror.c.163.com",
       "https://mirror.baidubce.com"
     ]
   }
   ```
3. 点击"Apply & Restart"

### 5.5 启用BuildKit

BuildKit是Docker的下一代构建引擎，可提高构建速度：

```json
{
  "features": {
    "buildkit": true
  }
}
```

### 5.6 配置Docker代理（可选）

如果您在企业网络环境中，需要配置代理：

1. 在Docker Desktop设置中，选择"Resources" > "Proxies"
2. 选择"Manual proxy configuration"
3. 输入HTTP和HTTPS代理地址
4. 点击"Apply & Restart"

## 六、安装验证

### 6.1 检查Docker版本

打开PowerShell或Windows Terminal，运行以下命令：

```powershell
# 检查Docker版本
docker --version

# 检查Docker Compose版本
docker compose version

# 检查Docker信息（详细）
docker info
```

### 6.2 运行Hello World容器

```powershell
docker run hello-world
```

成功输出示例：
```
Hello from Docker!
This message shows that your installation appears to be working correctly.
```

### 6.3 测试基本Docker操作

```powershell
# 拉取Ubuntu镜像
docker pull ubuntu:22.04

# 运行Ubuntu容器并进入交互式终端
docker run -it --name test-ubuntu ubuntu:22.04 bash

# 在容器内运行命令
ls -la
cat /etc/os-release

# 退出容器
exit

# 删除测试容器
docker rm test-ubuntu
```

### 6.4 测试Docker Compose

1. 创建一个测试目录：
   ```powershell
   mkdir docker-test
   cd docker-test
   ```

2. 创建`docker-compose.yml`文件：
   ```yaml
   version: '3.8'
   services:
     web:
       image: nginx:alpine
       ports:
         - "8080:80"
       volumes:
         - ./html:/usr/share/nginx/html
     redis:
       image: redis:alpine
       volumes:
         - redis-data:/data
   
   volumes:
     redis-data:
   ```

3. 创建`html`目录和测试页面：
   ```powershell
   mkdir html
   echo "<h1>Docker Compose Test</h1>" > html/index.html
   ```

4. 启动服务：
   ```powershell
   docker compose up -d
   ```

5. 访问 `http://localhost:8080` 查看测试页面

6. 查看服务状态：
   ```powershell
   docker compose ps
   docker compose logs
   ```

7. 停止并清理：
   ```powershell
   docker compose down -v
   cd ..
   rmdir /s /q docker-test
   ```

## 七、Docker Compose使用

### 7.1 Docker Compose基础命令

```powershell
# 启动服务（后台运行）
docker compose up -d

# 启动特定服务
docker compose up -d <service-name>

# 查看服务状态
docker compose ps

# 查看服务日志
docker compose logs
# 实时查看日志
docker compose logs -f
# 查看特定服务日志
docker compose logs -f <service-name>

# 停止服务
docker compose stop

# 停止并删除容器、网络、卷
docker compose down
# 包括删除命名卷
docker compose down -v

# 重启服务
docker compose restart

# 查看服务依赖关系
docker compose top

# 进入容器交互式终端
docker compose exec <service-name> bash
```

### 7.2 Docker Compose文件结构

一个标准的`docker-compose.yml`文件包含以下部分：

```yaml
version: '3.8'  # Compose文件版本

services:  # 服务定义
  service1:  # 服务名称
    image: service1:latest  # 镜像名称
    build: ./service1  # 构建上下文
    ports:  # 端口映射
      - "8080:80"
    volumes:  # 卷挂载
      - ./data:/app/data
    environment:  # 环境变量
      - ENV_VAR=value
    depends_on:  # 服务依赖
      - service2
    restart: unless-stopped  # 重启策略

volumes:  # 卷定义
  data-volume:

networks:  # 网络定义
  app-network:
    driver: bridge
```

## 八、供应链系统容器化部署

### 8.1 项目结构

```
supplychain/
├── docker-compose.yml         # 主Compose文件
├── Dockerfile                 # API服务Dockerfile
├── requirements.txt           # Python依赖
├── src/                       # 源代码
├── config/                    # 配置文件
└── data/                      # 数据目录
```

### 8.2 供应链系统Docker Compose配置

```yaml
version: '3.8'

services:
  # API服务
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    environment:
      - REDIS_URL=redis://redis:6379/0
      - DATABASE_URL=postgresql://user:password@postgres:5432/supplychain
      - LOG_LEVEL=info
    depends_on:
      - redis
      - postgres
    restart: unless-stopped

  # Redis缓存
  redis:
    image: redis:7-alpine
    volumes:
      - redis-data:/data
    restart: unless-stopped

  # PostgreSQL数据库
  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=supplychain
    volumes:
      - postgres-data:/var/lib/postgresql/data
    restart: unless-stopped

  # 监控服务（可选）
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./config/prometheus.yml:/etc/prometheus/prometheus.yml
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - grafana-data:/var/lib/grafana
    restart: unless-stopped

volumes:
  redis-data:
  postgres-data:
  grafana-data:

networks:
  default:
    name: supplychain-network
```

### 8.3 构建和部署

```powershell
# 进入项目目录
cd c:\huangrongxi\project\model\supplychain

# 构建镜像
docker compose build

# 启动服务
docker compose up -d

# 查看部署状态
docker compose ps

# 查看API服务日志
docker compose logs -f api
```

### 8.4 访问系统

- API服务：`http://localhost:8000`
- Prometheus监控：`http://localhost:9090`（可选）
- Grafana可视化：`http://localhost:3000`（可选）

### 8.5 数据持久化

供应链系统使用以下卷来持久化数据：
- `postgres-data`：数据库数据
- `redis-data`：缓存数据
- `grafana-data`：监控数据（可选）

## 九、常见问题排查

### 9.1 Docker命令无法识别

**症状**：运行`docker --version`时出现"docker不是内部或外部命令"。

**解决方案**：
1. 确保Docker Desktop已启动
2. 检查环境变量配置：
   ```powershell
   # 检查Docker路径是否在环境变量中
   echo $env:Path | Select-String "Docker"
   
   # 如果不在，手动添加
   [Environment]::SetEnvironmentVariable("Path", $env:Path + ";C:\Program Files\Docker\Docker\resources\bin", [EnvironmentVariableTarget]::Machine)
   ```
3. 重启PowerShell或终端

### 9.2 WSL 2安装失败

**症状**：安装WSL 2时出现错误代码`0x800701bc`。

**解决方案**：
1. 确保Windows版本为Build 19041或更高：
   ```powershell
   winver
   ```
2. 安装最新的Windows更新
3. 重新安装WSL 2内核更新包
4. 执行修复命令：
   ```powershell
   wsl --repair
   ```

### 9.3 Docker容器无法访问

**症状**：无法通过浏览器访问运行中的容器服务。

**解决方案**：
1. 检查容器状态：`docker compose ps`
2. 检查端口映射：`docker compose port <service-name> <container-port>`
3. 检查防火墙设置，确保端口已开放：
   ```powershell
   # 允许端口通过防火墙
   New-NetFirewallRule -DisplayName "Docker API" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow
   ```
4. 检查容器日志：`docker compose logs -f <service-name>`

### 9.4 Docker Desktop启动失败

**症状**：Docker Desktop无法启动，显示错误信息。

**解决方案**：
1. 重启计算机
2. 检查是否有其他虚拟化软件（如VirtualBox）正在运行
3. 重置Docker Desktop：
   - 点击系统托盘中的Docker图标 → "Troubleshoot" → "Reset to factory defaults"
4. 检查WSL状态：
   ```powershell
   wsl --list --verbose
   wsl --shutdown
   ```

### 9.5 镜像拉取缓慢或失败

**症状**：`docker pull`命令执行缓慢或失败。

**解决方案**：
1. 配置国内镜像加速（参考5.4节）
2. 检查网络连接
3. 尝试使用代理
4. 手动下载镜像并导入：
   ```powershell
   # 下载镜像tar文件
   # 使用docker load导入
   docker load -i image.tar
   ```

### 9.6 容器内存不足

**症状**：容器运行时出现"out of memory"错误。

**解决方案**：
1. 增加Docker Desktop的内存分配（参考5.2节）
2. 为容器设置内存限制：
   ```yaml
   services:
     api:
       image: api:latest
       mem_limit: 4g
       mem_reservation: 2g
   ```
3. 优化应用程序内存使用

## 十、Docker最佳实践

### 10.1 镜像构建最佳实践

```dockerfile
# 使用多阶段构建
FROM python:3.11-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY . .

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 10.2 容器运行最佳实践

- **使用非root用户**：
  ```dockerfile
  RUN useradd -m appuser
  USER appuser
  ```

- **合理设置重启策略**：
  ```yaml
  restart: unless-stopped
  ```

- **使用健康检查**：
  ```yaml
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
    interval: 30s
    timeout: 10s
    retries: 3
  ```

### 10.3 数据管理最佳实践

- **使用命名卷**而非绑定挂载：
  ```yaml
  volumes:
    - postgres-data:/var/lib/postgresql/data
  ```

- **定期备份数据卷**：
  ```powershell
  docker run --rm -v postgres-data:/data -v $(pwd):/backup ubuntu tar czf /backup/postgres-backup.tar.gz /data
  ```

### 10.4 安全最佳实践

- 使用官方镜像或可信来源
- 定期更新镜像
- 避免在镜像中存储敏感信息
- 使用Docker Secrets管理敏感数据
- 限制容器权限

## 十一、容器化架构设计

```
┌─────────────────────────────────────────────────────────────────────┐
│                        供应链系统容器化架构                          │
│                                                                     │
│  ┌─────────────┐     ┌─────────────┐     ┌───────────────────────┐  │
│  │  用户界面     │────▶│  API网关     │────▶│  供应链API服务集群    │  │
│  └─────────────┘     └─────────────┘     └───────────────────────┘  │
│                                           ▲                       │  │
│                                           │                       │  │
│  ┌─────────────┐     ┌─────────────┐     ┌───────────────────────┐  │
│  │  Redis缓存    │◀────│  预测模型服务   │◀────│  数据处理服务集群    │  │
│  └─────────────┘     └─────────────┘     └───────────────────────┘  │
│                        ▲                   ▲                         │
│                        │                   │                         │
│  ┌─────────────┐     ┌─────────────┐     ┌───────────────────────┐  │
│  │  PostgreSQL  │     │  监控服务     │     │  配置管理服务        │  │
│  └─────────────┘     └─────────────┘     └───────────────────────┘  │
│                        ▲                   ▲                         │
│                        │                   │                         │
│  ┌─────────────┐     ┌─────────────┐     ┌───────────────────────┐  │
│  │  数据存储卷   │     │  日志服务     │     │  告警服务          │  │
│  └─────────────┘     └─────────────┘     └───────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## 十二、扩展与维护

### 12.1 容器扩展

- **水平扩展服务**：
  ```powershell
  # 扩展API服务到3个实例
  docker compose up -d --scale api=3
  ```

### 12.2 日志管理

- **集中化日志收集**：
  ```yaml
  services:
    api:
      logging:
        driver: "json-file"
        options:
          max-size: "10m"
          max-file: "3"
  ```

### 12.3 定期维护

```powershell
# 清理未使用的资源
docker system prune -a

# 清理旧镜像
docker image prune -a --filter "until=24h"

# 检查容器状态
docker compose ps

# 监控资源使用
docker stats
```

### 12.4 升级服务

```powershell
# 更新镜像
docker compose pull

# 重新启动服务
docker compose up -d

# 验证服务状态
docker compose ps
```

## 十三、总结

恭喜！您已成功完成Windows系统Docker的安装、配置和优化。通过本指南，您学习了：

✅ Docker和WSL 2的完整安装流程
✅ Docker高级配置和性能优化
✅ Docker Compose的使用方法
✅ 供应链系统的容器化部署方案
✅ 常见问题的排查和解决方法
✅ Docker最佳实践和安全配置

现在您可以使用Docker Compose轻松部署和管理供应链智能补货系统了。如果您在使用过程中遇到任何问题，请参考[常见问题排查](#九常见问题排查)部分，或联系技术支持团队。

祝您使用愉快！

---

**版本信息**：
- Docker版本：24.x
- Docker Compose版本：v2.x
- Windows版本：Windows 10/11 Pro

**更新日期**：2023年12月

**文档作者**：供应链系统技术团队
