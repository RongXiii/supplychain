# Windows系统Docker安装和配置指南

## 目录

1. [概述](#概述)
2. [系统要求](#系统要求)
3. [安装Docker Desktop](#安装docker-desktop)
4. [配置Docker](#配置docker)
5. [验证安装](#验证安装)
6. [使用Docker Compose](#使用docker-compose)
7. [常见问题排查](#常见问题排查)
8. [附加配置](#附加配置)

## 概述

本指南将帮助您在Windows系统上安装和配置Docker，以便您可以正常运行Docker命令和Docker Compose来部署供应链库存优化系统。

## 系统要求

在安装Docker之前，请确保您的Windows系统满足以下要求：

- Windows 10 64位：Pro、Enterprise或Education版本（Build 15063或更高）
- Windows 11 64位：Pro、Enterprise或Education版本
- 至少4GB RAM
- 启用虚拟化技术（Intel VT-x/EPT或AMD-V/RVI）
- WSL 2（Windows Subsystem for Linux）支持

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

## 配置Docker

### 步骤1：登录Docker Hub（可选）

1. 点击系统托盘中的Docker图标
2. 选择"Sign in / Create Docker ID"
3. 登录您的Docker Hub账号，或创建一个新账号

### 步骤2：配置资源

1. 点击系统托盘中的Docker图标
2. 选择"Settings"或"Preferences"
3. 在"Resources"选项卡中，您可以调整以下设置：
   - **CPU**：建议分配至少2个CPU核心
   - **Memory**：建议分配至少4GB RAM
   - **Disk image size**：建议至少20GB

### 步骤3：配置WSL集成

1. 在Docker Desktop设置中，选择"Resources" > "WSL Integration"
2. 确保"Enable integration with my default WSL distro"选项已勾选
3. 选择要与Docker集成的WSL分发版
4. 点击"Apply & Restart"

## 验证安装

### 步骤1：检查Docker版本

1. 打开PowerShell或Command Prompt
2. 运行以下命令检查Docker版本：
   ```powershell
   docker --version
   ```
   预期输出：
   ```
   Docker version 20.10.xx, build xxxxxxx
   ```

3. 运行以下命令检查Docker Compose版本：
   ```powershell
   docker compose version
   ```
   预期输出：
   ```
   Docker Compose version v2.xx.x
   ```

### 步骤2：运行Hello World容器

1. 运行以下命令：
   ```powershell
   docker run hello-world
   ```

2. 如果安装成功，您将看到以下输出：
   ```
   Hello from Docker!
   This message shows that your installation appears to be working correctly.
   ```

### 步骤3：测试Docker Compose

1. 创建一个简单的docker-compose.yml文件：
   ```yaml
   version: '3.8'
   services:
     web:
       image: nginx:alpine
       ports:
         - "8080:80"
     redis:
       image: redis:alpine
   ```

2. 运行以下命令启动服务：
   ```powershell
   docker compose up -d
   ```

3. 运行以下命令检查服务状态：
   ```powershell
   docker compose ps
   ```

4. 运行以下命令停止服务：
   ```powershell
   docker compose down
   ```

## 使用Docker Compose

安装完成后，您可以使用Docker Compose来部署供应链库存优化系统：

1. 导航到项目目录：
   ```powershell
   cd c:\huangrongxi\project\model\supplychain
   ```

2. 启动所有服务：
   ```powershell
   docker compose up -d
   ```

3. 检查服务状态：
   ```powershell
   docker compose ps
   ```

4. 查看服务日志：
   ```powershell
   docker compose logs -f <service-name>
   ```

5. 停止所有服务：
   ```powershell
   docker compose down
   ```

## 常见问题排查

### 问题1：Docker命令无法识别

**症状**：运行`docker --version`时出现"docker不是内部或外部命令，也不是可运行的程序或批处理文件"。

**解决方案**：
1. 确保Docker Desktop已启动
2. 检查环境变量是否正确配置：
   - 以管理员身份打开PowerShell
   - 运行以下命令：
     ```powershell
     [Environment]::SetEnvironmentVariable("Path", $env:Path + ";C:\Program Files\Docker\Docker\resources\bin", [EnvironmentVariableTarget]::Machine)
     ```
   - 重启PowerShell

### 问题2：WSL 2安装失败

**症状**：安装WSL 2时出现错误。

**解决方案**：
1. 确保您的Windows版本符合要求（Windows 10 Build 19041或更高）
2. 检查Windows更新，确保系统已安装最新更新
3. 按照步骤1.2重新启用WSL功能

### 问题3：Docker容器无法访问

**症状**：无法通过浏览器访问运行中的Docker容器。

**解决方案**：
1. 检查容器是否正在运行：`docker compose ps`
2. 检查端口映射是否正确：`docker compose port <service-name> <container-port>`
3. 检查防火墙设置，确保端口已开放

### 问题4：Docker Desktop启动失败

**症状**：Docker Desktop无法启动，显示错误信息。

**解决方案**：
1. 重启计算机
2. 检查是否有其他虚拟化软件（如VirtualBox）正在运行，可能与Docker冲突
3. 重置Docker Desktop：
   - 点击系统托盘中的Docker图标
   - 选择"Troubleshoot" > "Reset to factory defaults"

## 附加配置

### 配置镜像加速（可选）

如果您在拉取Docker镜像时遇到网络问题，可以配置镜像加速：

1. 在Docker Desktop设置中，选择"Docker Engine"
2. 添加以下配置：
   ```json
   {
     "registry-mirrors": [
       "https://registry.docker-cn.com",
       "https://docker.mirrors.ustc.edu.cn",
       "https://hub-mirror.c.163.com"
     ]
   }
   ```
3. 点击"Apply & Restart"

### 启用BuildKit（可选）

BuildKit是Docker的下一代构建引擎，可以提高构建速度：

1. 在Docker Desktop设置中，选择"Docker Engine"
2. 添加以下配置：
   ```json
   {
     "features": {
       "buildkit": true
     }
   }
   ```
3. 点击"Apply & Restart"

## 结束

恭喜！您已成功安装和配置Docker。现在您可以使用Docker Compose来部署供应链库存优化系统了。如果您遇到任何问题，请参考[常见问题排查](#常见问题排查)部分，或联系技术支持团队。

祝您使用愉快！
