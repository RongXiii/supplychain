#!/usr/bin/env python3
"""
缓存管理器，用于缓存频繁访问的数据，提高API响应速度
"""

import json
import os
import time
from typing import Any, Optional
import pandas as pd
import numpy as np

class CacheManager:
    """
    缓存管理器类，支持Redis缓存和内存缓存
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """单例模式实现"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, host='localhost', port=6379, db=0, password=None):
        """
        初始化缓存管理器
        
        Args:
            host: Redis服务器地址
            port: Redis服务器端口
            db: Redis数据库索引
            password: Redis服务器密码
        """
        # 仅在第一次实例化时初始化
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self.host = host
            self.port = port
            self.db = db
            self.password = password
            
            # 内存缓存字典
            self.memory_cache = {}
            
            # 尝试连接Redis
            self.is_redis_available = False
            try:
                import redis
                # 连接到Redis
                self.redis_client = redis.Redis(
                    host=self.host,
                    port=self.port,
                    db=self.db,
                    password=self.password,
                    decode_responses=False  # 保持二进制数据
                )
                
                # 测试连接
                self.redis_client.ping()
                self.is_redis_available = True
                print(f"成功连接到Redis服务器: {host}:{port}，使用Redis作为主要缓存")
            except ImportError:
                print("警告：未安装Redis模块，使用内存缓存")
            except Exception as e:
                print(f"警告：无法连接到Redis，使用内存缓存: {e}")
    
    def is_connected(self):
        """
        检查是否连接到Redis服务器
        
        Returns:
            bool: True if connected, False otherwise
        """
        return self.is_redis_available
    
    def _get_from_memory(self, key: str) -> Optional[Any]:
        """从内存缓存获取数据"""
        if key in self.memory_cache:
            cache_item = self.memory_cache[key]
            # 检查是否过期
            if cache_item['expire_time'] is None or time.time() < cache_item['expire_time']:
                return cache_item['value']
            else:
                # 过期，删除缓存
                del self.memory_cache[key]
        return None
    
    def _set_to_memory(self, key: str, value: Any, expire_seconds: Optional[int] = None) -> bool:
        """设置内存缓存"""
        expire_time = time.time() + expire_seconds if expire_seconds else None
        self.memory_cache[key] = {
            'value': value,
            'expire_time': expire_time
        }
        return True
    
    def set(self, key, value, expire_seconds=3600):
        """
        设置缓存
        
        Args:
            key: 缓存键
            value: 缓存值
            expire_seconds: 过期时间（秒），默认1小时
        """
        # 同时设置到Redis和内存缓存
        success = False
        
        if self.is_redis_available:
            try:
                import redis
                # 根据数据类型进行序列化
                if isinstance(value, pd.DataFrame):
                    serialized_value = pd.util.testing.to_pickle(value)
                elif isinstance(value, np.ndarray):
                    serialized_value = pd.util.testing.to_pickle(value)
                elif isinstance(value, dict) or isinstance(value, list):
                    serialized_value = json.dumps(value).encode('utf-8')
                elif isinstance(value, np.integer):
                    serialized_value = str(int(value)).encode('utf-8')
                elif isinstance(value, np.floating):
                    serialized_value = str(float(value)).encode('utf-8')
                else:
                    serialized_value = str(value).encode('utf-8')
                
                # 设置Redis缓存
                if expire_seconds:
                    self.redis_client.setex(key, expire_seconds, serialized_value)
                else:
                    self.redis_client.set(key, serialized_value)
                
                success = True
            except Exception as e:
                print(f"Redis设置缓存失败: {e}")
        
        # 同时设置到内存缓存
        self._set_to_memory(key, value, expire_seconds)
        
        return success or True
    
    def get(self, key, data_type=None):
        """
        获取缓存
        
        Args:
            key: 缓存键
            data_type: 数据类型，可选值：'json', 'dataframe', 'array', 'list', 'dict'
            
        Returns:
            缓存值，如果不存在则返回None
        """
        # 先从内存缓存获取
        memory_value = self._get_from_memory(key)
        if memory_value is not None:
            return memory_value
        
        # 如果Redis可用，从Redis获取
        if self.is_redis_available:
            try:
                import redis
                # 获取Redis缓存数据
                serialized_value = self.redis_client.get(key)
                if serialized_value is None:
                    return None
                
                # 根据数据类型进行反序列化
                value = None
                if data_type == 'dataframe':
                    value = pd.read_pickle(serialized_value)
                elif data_type == 'array' or data_type == 'ndarray':
                    value = pd.read_pickle(serialized_value)
                elif data_type in ['json', 'list', 'dict']:
                    value = json.loads(serialized_value.decode('utf-8'))
                else:
                    # 尝试自动检测
                    try:
                        # 尝试作为JSON解析
                        value = json.loads(serialized_value.decode('utf-8'))
                    except json.JSONDecodeError:
                        try:
                            # 尝试作为pickle解析（DataFrame或NumPy数组）
                            value = pd.read_pickle(serialized_value)
                        except Exception:
                            # 作为字符串返回
                            value = serialized_value.decode('utf-8')
                
                # 将Redis结果存入内存缓存，提高下次访问速度
                if value is not None:
                    # 获取过期时间
                    ttl = self.redis_client.ttl(key)
                    expire_seconds = ttl if ttl > 0 else None
                    self._set_to_memory(key, value, expire_seconds)
                
                return value
            except Exception as e:
                print(f"Redis获取缓存失败: {e}")
        
        return None
    
    def delete(self, key):
        """
        删除缓存
        
        Args:
            key: 缓存键
        """
        success = False
        
        # 从Redis删除
        if self.is_redis_available:
            try:
                import redis
                self.redis_client.delete(key)
                success = True
            except Exception as e:
                print(f"Redis删除缓存失败: {e}")
        
        # 从内存缓存删除
        if key in self.memory_cache:
            del self.memory_cache[key]
            success = True
        
        return success
    
    def clear(self, pattern='*'):
        """
        清除匹配模式的所有缓存
        
        Args:
            pattern: 匹配模式，默认清除所有缓存
        """
        success = False
        
        # 清除Redis缓存
        if self.is_redis_available:
            try:
                import redis
                # 获取所有匹配的键
                keys = self.redis_client.keys(pattern)
                if keys:
                    self.redis_client.delete(*keys)
                success = True
            except Exception as e:
                print(f"Redis清除缓存失败: {e}")
        
        # 清除内存缓存
        if pattern == '*':
            self.memory_cache.clear()
            success = True
        else:
            # 简单的模式匹配（只支持*）
            import fnmatch
            keys_to_delete = [key for key in self.memory_cache if fnmatch.fnmatch(key, pattern)]
            for key in keys_to_delete:
                del self.memory_cache[key]
            if keys_to_delete:
                success = True
        
        return success
    
    def get_keys(self, pattern='*'):
        """
        获取匹配模式的所有缓存键
        
        Args:
            pattern: 匹配模式，默认获取所有键
            
        Returns:
            list: 匹配的键列表
        """
        keys = []
        
        # 从Redis获取键
        if self.is_redis_available:
            try:
                import redis
                redis_keys = self.redis_client.keys(pattern)
                keys.extend([key.decode('utf-8') for key in redis_keys])
            except Exception as e:
                print(f"Redis获取缓存键失败: {e}")
        
        # 从内存缓存获取键
        import fnmatch
        memory_keys = [key for key in self.memory_cache if fnmatch.fnmatch(key, pattern)]
        # 合并去重
        keys = list(set(keys + memory_keys))
        
        return keys
    
    def exists(self, key):
        """
        检查缓存是否存在
        
        Args:
            key: 缓存键
            
        Returns:
            bool: True if exists, False otherwise
        """
        # 先检查内存缓存
        if self._get_from_memory(key) is not None:
            return True
        
        # 检查Redis缓存
        if self.is_redis_available:
            try:
                import redis
                return self.redis_client.exists(key) > 0
            except Exception as e:
                print(f"Redis检查缓存失败: {e}")
        
        return False

# 单例模式
cache_manager = CacheManager()
