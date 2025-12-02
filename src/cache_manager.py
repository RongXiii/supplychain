#!/usr/bin/env python3
"""
缓存管理器，用于缓存频繁访问的数据，提高API响应速度
支持智能缓存策略、分布式缓存和性能监控
"""

import json
import os
import time
import pickle
from typing import Any, Optional, Dict, Tuple
import pandas as pd
import numpy as np
from functools import wraps

class CacheManager:
    """
    缓存管理器类，支持Redis缓存和内存缓存，实现智能缓存策略
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """单例模式实现"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, host='localhost', port=6379, db=0, password=None, distributed=True):
        """
        初始化缓存管理器
        
        Args:
            host: Redis服务器地址
            port: Redis服务器端口
            db: Redis数据库索引
            password: Redis服务器密码
            distributed: 是否启用分布式缓存
        """
        # 仅在第一次实例化时初始化
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self.host = host
            self.port = port
            self.db = db
            self.password = password
            self.distributed = distributed
            
            # 内存缓存字典
            self.memory_cache = {}
            # 缓存访问统计
            self.cache_stats = {
                'hits': 0,
                'misses': 0,
                'memory_hits': 0,
                'redis_hits': 0
            }
            # 智能缓存策略配置
            self.intelligent_cache = {
                # 缓存访问频率跟踪
                'access_count': {},
                # 缓存大小跟踪
                'cache_sizes': {},
                # 自动过期时间调整
                'dynamic_ttl': {}
            }
            
            # 尝试连接Redis
            self.is_redis_available = False
            self.redis_client = None
            self.redis_pool = None
            
            if self.distributed:
                try:
                    import redis
                    # 创建Redis连接池
                    self.redis_pool = redis.ConnectionPool(
                        host=self.host,
                        port=self.port,
                        db=self.db,
                        password=self.password,
                        max_connections=50,
                        decode_responses=False  # 保持二进制数据
                    )
                    # 连接到Redis
                    self.redis_client = redis.Redis(connection_pool=self.redis_pool)
                    
                    # 测试连接
                    self.redis_client.ping()
                    self.is_redis_available = True
                    print(f"成功连接到Redis服务器: {host}:{port}，使用Redis作为主要缓存")
                except ImportError:
                    print("警告：未安装Redis模块，使用内存缓存")
                except Exception as e:
                    print(f"警告：无法连接到Redis，使用内存缓存: {e}")
            
            # 优化：预分配内存缓存空间
            self.memory_cache = {}
    
    def is_connected(self):
        """
        检查是否连接到Redis服务器
        
        Returns:
            bool: True if connected, False otherwise
        """
        return self.is_redis_available
    
    def get_stats(self) -> Dict[str, int]:
        """
        获取缓存统计信息
        
        Returns:
            Dict[str, int]: 缓存统计数据
        """
        return self.cache_stats.copy()
    
    def _get_from_memory(self, key: str) -> Optional[Any]:
        """从内存缓存获取数据"""
        if key in self.memory_cache:
            cache_item = self.memory_cache[key]
            # 检查是否过期
            if cache_item['expire_time'] is None or time.time() < cache_item['expire_time']:
                # 更新访问次数
                self._update_access_count(key)
                self.cache_stats['memory_hits'] += 1
                return cache_item['value']
            else:
                # 过期，删除缓存
                self._remove_from_memory(key)
        return None
    
    def _set_to_memory(self, key: str, value: Any, expire_seconds: Optional[int] = None) -> bool:
        """设置内存缓存"""
        expire_time = time.time() + expire_seconds if expire_seconds else None
        # 计算缓存大小（近似值）
        cache_size = self._calculate_cache_size(value)
        
        self.memory_cache[key] = {
            'value': value,
            'expire_time': expire_time,
            'access_count': 0,
            'last_access': time.time(),
            'size': cache_size
        }
        return True
    
    def _remove_from_memory(self, key: str) -> bool:
        """从内存缓存中删除数据"""
        if key in self.memory_cache:
            del self.memory_cache[key]
            # 清理相关统计数据
            if key in self.intelligent_cache['access_count']:
                del self.intelligent_cache['access_count'][key]
            if key in self.intelligent_cache['cache_sizes']:
                del self.intelligent_cache['cache_sizes'][key]
            if key in self.intelligent_cache['dynamic_ttl']:
                del self.intelligent_cache['dynamic_ttl'][key]
            return True
        return False
    
    def _calculate_cache_size(self, value: Any) -> int:
        """计算缓存值的大小（字节）"""
        if isinstance(value, (pd.DataFrame, np.ndarray)):
            return value.nbytes
        elif isinstance(value, (dict, list)):
            return len(pickle.dumps(value))
        elif isinstance(value, (str, bytes)):
            return len(value)
        else:
            return len(str(value))
    
    def _update_access_count(self, key: str):
        """更新缓存访问计数"""
        # 更新访问次数
        if key in self.memory_cache:
            self.memory_cache[key]['access_count'] += 1
            self.memory_cache[key]['last_access'] = time.time()
        
        # 全局访问计数
        if key not in self.intelligent_cache['access_count']:
            self.intelligent_cache['access_count'][key] = 0
        self.intelligent_cache['access_count'][key] += 1
    
    def _get_dynamic_ttl(self, key: str, default_ttl: int = 3600) -> int:
        """
        根据访问频率动态调整TTL
        
        Args:
            key: 缓存键
            default_ttl: 默认过期时间
            
        Returns:
            int: 动态调整后的TTL
        """
        if key in self.intelligent_cache['dynamic_ttl']:
            return self.intelligent_cache['dynamic_ttl'][key]
        
        # 根据访问频率调整TTL
        access_count = self.intelligent_cache['access_count'].get(key, 0)
        if access_count > 10:
            # 高频访问，延长TTL
            dynamic_ttl = default_ttl * 2
        elif access_count < 2:
            # 低频访问，缩短TTL
            dynamic_ttl = default_ttl // 2
        else:
            dynamic_ttl = default_ttl
        
        self.intelligent_cache['dynamic_ttl'][key] = dynamic_ttl
        return dynamic_ttl
    
    def _serialize_data(self, value: Any) -> bytes:
        """
        优化的数据序列化
        
        Args:
            value: 要序列化的数据
            
        Returns:
            bytes: 序列化后的数据
        """
        if isinstance(value, pd.DataFrame):
            # 使用更高效的pickle协议
            return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        elif isinstance(value, np.ndarray):
            # 使用numpy内置的序列化
            return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        elif isinstance(value, dict) or isinstance(value, list):
            # 使用更高效的json序列化
            return json.dumps(value, separators=(',', ':')).encode('utf-8')
        elif isinstance(value, np.integer):
            return str(int(value)).encode('utf-8')
        elif isinstance(value, np.floating):
            return str(float(value)).encode('utf-8')
        else:
            return str(value).encode('utf-8')
    
    def _deserialize_data(self, serialized_value: bytes, data_type: str = None) -> Any:
        """
        优化的数据反序列化
        
        Args:
            serialized_value: 序列化后的数据
            data_type: 数据类型
            
        Returns:
            Any: 反序列化后的数据
        """
        if data_type == 'dataframe':
            return pickle.loads(serialized_value)
        elif data_type == 'array' or data_type == 'ndarray':
            return pickle.loads(serialized_value)
        elif data_type in ['json', 'list', 'dict']:
            return json.loads(serialized_value.decode('utf-8'))
        else:
            # 尝试自动检测
            try:
                # 尝试作为JSON解析
                return json.loads(serialized_value.decode('utf-8'))
            except json.JSONDecodeError:
                try:
                    # 尝试作为pickle解析（DataFrame或NumPy数组）
                    return pickle.loads(serialized_value)
                except Exception:
                    # 作为字符串返回
                    return serialized_value.decode('utf-8')
    
    def set(self, key, value, expire_seconds=3600, intelligent=True):
        """
        设置缓存
        
        Args:
            key: 缓存键
            value: 缓存值
            expire_seconds: 过期时间（秒），默认1小时
            intelligent: 是否启用智能缓存策略
            
        Returns:
            bool: 设置是否成功
        """
        # 智能缓存策略：动态调整TTL
        if intelligent:
            expire_seconds = self._get_dynamic_ttl(key, expire_seconds)
        
        # 同时设置到Redis和内存缓存
        success = False
        
        if self.is_redis_available:
            try:
                # 优化序列化
                serialized_value = self._serialize_data(value)
                
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
    
    def get(self, key, data_type=None, increment_access=True):
        """
        获取缓存
        
        Args:
            key: 缓存键
            data_type: 数据类型，可选值：'json', 'dataframe', 'array', 'list', 'dict'
            increment_access: 是否增加访问计数
            
        Returns:
            缓存值，如果不存在则返回None
        """
        # 先从内存缓存获取
        memory_value = self._get_from_memory(key)
        if memory_value is not None:
            self.cache_stats['hits'] += 1
            return memory_value
        
        # 如果Redis可用，从Redis获取
        if self.is_redis_available:
            try:
                # 获取Redis缓存数据
                serialized_value = self.redis_client.get(key)
                if serialized_value is None:
                    self.cache_stats['misses'] += 1
                    return None
                
                # 优化反序列化
                value = self._deserialize_data(serialized_value, data_type)
                
                # 将Redis结果存入内存缓存，提高下次访问速度
                if value is not None:
                    # 获取过期时间
                    ttl = self.redis_client.ttl(key)
                    expire_seconds = ttl if ttl > 0 else None
                    self._set_to_memory(key, value, expire_seconds)
                    self.cache_stats['redis_hits'] += 1
                    self.cache_stats['hits'] += 1
                
                return value
            except Exception as e:
                print(f"Redis获取缓存失败: {e}")
        
        self.cache_stats['misses'] += 1
        return None
    
    def get_with_ttl(self, key, data_type=None):
        """
        获取缓存并返回剩余TTL
        
        Args:
            key: 缓存键
            data_type: 数据类型
            
        Returns:
            Tuple[Any, int]: 缓存值和剩余TTL（-1表示永不过期，-2表示不存在）
        """
        value = self.get(key, data_type)
        if value is None:
            return None, -2
        
        ttl = -1
        if self.is_redis_available:
            ttl = self.redis_client.ttl(key)
        
        return value, ttl
    
    def batch_get(self, keys, data_type=None):
        """
        批量获取缓存
        
        Args:
            keys: 缓存键列表
            data_type: 数据类型
            
        Returns:
            Dict[str, Any]: 缓存键值对
        """
        results = {}
        
        # 先从内存缓存批量获取
        memory_keys = [key for key in keys if key in self.memory_cache]
        for key in memory_keys:
            value = self._get_from_memory(key)
            if value is not None:
                results[key] = value
        
        # 从Redis获取剩余的键
        redis_keys = [key for key in keys if key not in results]
        if self.is_redis_available and redis_keys:
            try:
                import redis
                # 批量获取Redis缓存
                redis_results = self.redis_client.mget(redis_keys)
                for i, key in enumerate(redis_keys):
                    serialized_value = redis_results[i]
                    if serialized_value is not None:
                        value = self._deserialize_data(serialized_value, data_type)
                        results[key] = value
                        # 存入内存缓存
                        ttl = self.redis_client.ttl(key)
                        expire_seconds = ttl if ttl > 0 else None
                        self._set_to_memory(key, value, expire_seconds)
            except Exception as e:
                print(f"Redis批量获取缓存失败: {e}")
        
        return results
    
    def batch_set(self, key_value_pairs, expire_seconds=3600):
        """
        批量设置缓存
        
        Args:
            key_value_pairs: 缓存键值对字典
            expire_seconds: 过期时间
            
        Returns:
            bool: 设置是否成功
        """
        success = False
        
        if self.is_redis_available:
            try:
                import redis
                # 使用管道批量操作
                pipeline = self.redis_client.pipeline()
                for key, value in key_value_pairs.items():
                    serialized_value = self._serialize_data(value)
                    if expire_seconds:
                        pipeline.setex(key, expire_seconds, serialized_value)
                    else:
                        pipeline.set(key, serialized_value)
                pipeline.execute()
                success = True
            except Exception as e:
                print(f"Redis批量设置缓存失败: {e}")
        
        # 同时设置到内存缓存
        for key, value in key_value_pairs.items():
            self._set_to_memory(key, value, expire_seconds)
        
        return success or True
    
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
