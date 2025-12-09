import psutil
import os

# 获取当前进程ID
pid = os.getpid()

# 获取进程内存信息
process = psutil.Process(pid)
mem_info = process.memory_info()

# 打印内存使用情况
print(f"进程ID: {pid}")
print(f"物理内存使用: {mem_info.rss / 1024 / 1024:.2f} MB")
print(f"虚拟内存使用: {mem_info.vms / 1024 / 1024:.2f} MB")
print(f"内存百分比: {process.memory_percent():.2f}%")
