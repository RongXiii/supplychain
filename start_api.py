#!/usr/bin/env python3
"""
启动供应链智能补货系统API服务
支持高性能配置和微服务架构
"""

import uvicorn
import sys
import os
import argparse

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='启动供应链智能补货系统API服务')
    parser.add_argument('--workers', type=int, default=4, help='工作进程数量')
    parser.add_argument('--port', type=int, default=8000, help='服务端口')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='服务主机')
    parser.add_argument('--reload', action='store_true', help='开发模式下启用自动重载')
    parser.add_argument('--loop', type=str, default='uvloop', help='事件循环实现')
    parser.add_argument('--http', type=str, default='httptools', help='HTTP解析器')
    parser.add_argument('--log-level', type=str, default='info', help='日志级别')
    
    args = parser.parse_args()
    
    print("=== 供应链智能补货系统API服务 ===")
    print(f"API文档地址: http://{args.host}:{args.port}/docs")
    print(f"PowerBI数据接口: http://{args.host}:{args.port}/api/")
    print(f"工作进程数: {args.workers}")
    print(f"事件循环: {args.loop}")
    print(f"HTTP解析器: {args.http}")
    print("按 Ctrl+C 停止服务")
    print("=" * 70)
    
    # 启动API服务
    uvicorn.run(
        "src.api:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        loop=args.loop,
        http=args.http,
        log_level=args.log_level
    )