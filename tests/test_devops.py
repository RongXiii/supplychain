#!/usr/bin/env python3
"""
运维优化功能测试脚本
测试日志管理、备份机制和版本管理功能
"""

import os
import sys
import time
import json
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

# 测试日志管理
print("=== 测试日志管理功能 ===")
from src.logging_manager import get_logger, log_performance

# 创建日志记录器
logger = get_logger('test_devops')
logger.info("开始测试运维优化功能")
logger.debug("调试日志测试")
logger.warning("警告日志测试")

# 测试性能日志
start_time = time.time()
time.sleep(0.1)
log_performance("test_operation", time.time() - start_time, test_param="test_value", data_size=100)

logger.info("日志管理功能测试完成")

# 测试备份机制
print("\n=== 测试备份机制 ===")
from src.backup_manager import BackupManager, backup_all

backup_manager = BackupManager()

# 创建测试数据目录和文件
Path("test_data").mkdir(exist_ok=True)
with open("test_data/test_file.txt", "w") as f:
    f.write("测试备份数据")

# 测试数据备份
print("1. 测试数据备份...")
data_backup = backup_manager.backup_data(["test_data"])
if data_backup:
    print(f"   ✓ 数据备份成功: {data_backup}")
else:
    print("   ✗ 数据备份失败")

# 测试模型备份
print("2. 测试模型备份...")
model_backup = backup_manager.backup_models("models")
if model_backup:
    print(f"   ✓ 模型备份成功: {model_backup}")
else:
    print("   ✗ 模型备份失败")

# 测试获取备份列表
print("3. 测试获取备份列表...")
backup_list = backup_manager.get_backup_list()
print(f"   ✓ 备份列表获取成功，共 {len(backup_list)} 个备份")
for backup in backup_list[:3]:  # 只显示前3个
    print(f"     - {backup}")

# 测试版本管理
print("\n=== 测试版本管理功能 ===")
from src.version_manager import VersionManager, create_version, list_versions

version_manager = VersionManager()

# 测试创建版本
print("1. 测试创建版本...")
version_info = version_manager.create_version(
    version_name="v1.0.0",
    description="测试版本",
    model_paths=["models" if os.path.exists("models") else ""],
    config_paths=["configs" if os.path.exists("configs") else ""]
)
if version_info:
    print(f"   ✓ 版本创建成功: {version_info['version']}")
else:
    print("   ✗ 版本创建失败")

# 测试获取版本列表
print("2. 测试获取版本列表...")
versions = list_versions()
print(f"   ✓ 版本列表获取成功，共 {len(versions)} 个版本")
for version in versions:
    print(f"     - {version['version']}: {version['description']} ({version['timestamp']})")

# 测试获取版本信息
print("3. 测试获取版本信息...")
version_info = version_manager.get_version_info("v1.0.0")
if version_info:
    print(f"   ✓ 版本信息获取成功: {version_info['version']}")
else:
    print("   ✗ 版本信息获取失败")

# 测试设置版本状态
print("4. 测试设置版本状态...")
status = version_manager.set_version_status("v1.0.0", "inactive")
if status:
    print("   ✓ 版本状态设置成功")
else:
    print("   ✗ 版本状态设置失败")

# 测试完整备份功能
print("\n=== 测试完整备份功能 ===")
data_backup, model_backup = backup_all()
print(f"数据备份: {data_backup}")
print(f"模型备份: {model_backup}")

# 清理测试数据
print("\n=== 清理测试数据 ===")
if os.path.exists("test_data"):
    import shutil
    shutil.rmtree("test_data")
    print("✓ 测试数据已清理")

# 测试结果汇总
print("\n=== 测试结果汇总 ===")
print("✓ 日志管理功能测试完成")
print("✓ 备份机制测试完成")
print("✓ 版本管理功能测试完成")
print("\n所有运维优化功能测试完成！")

logger.info("运维优化功能测试完成")
