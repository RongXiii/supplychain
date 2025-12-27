#!/usr/bin/env python3
"""
缓存管理器，用于缓存频繁访问的数据，提高API响应速度
支持智能缓存策略、分布式缓存和性能监控
"""

import json
import os
import time
import pickle
import zlib
from typing import Any, Optional, Dict, Tuple, List, Callable
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
                'dynamic_ttl': {},
                # 缓存压缩配置
                'compression': {
                    'enabled': True,
                    'threshold': 1024 * 1024,  # 1MB以上数据启用压缩
                    'level': 3  # 压缩级别 1-9
                }
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
        if isinstance(value, np.ndarray):
            return value.nbytes
        elif isinstance(value, pd.DataFrame):
            # 计算DataFrame所有列的nbytes之和
            return value.memory_usage(deep=True).sum()
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
        根据访问频率、时间和数据大小动态调整TTL
        
        Args:
            key: 缓存键
            default_ttl: 默认TTL
            
        Returns:
            int: 调整后的TTL
        """
        access_count = self.intelligent_cache['access_count'].get(key, 0)
        cache_size = self.intelligent_cache['cache_sizes'].get(key, 0)
        
        # 基于访问频率的调整
        frequency_factor = 1.0
        if access_count < 5:
            frequency_factor = 0.5  # 低频访问，缩短TTL
        elif access_count < 10:
            frequency_factor = 1.0  # 正常访问，保持默认TTL
        elif access_count < 20:
            frequency_factor = 2.0  # 高频访问，延长TTL
        else:
            frequency_factor = 3.0  # 超高频访问，大幅延长TTL
        
        # 基于数据大小的调整
        size_factor = 1.0
        if cache_size > 1024 * 1024 * 100:  # 大于100MB
            size_factor = 0.5  # 大数据，缩短TTL
        elif cache_size > 1024 * 1024 * 10:  # 大于10MB
            size_factor = 0.8  # 中大数据，略微缩短TTL
        
        # 基于时间的调整（例如白天访问频繁，延长TTL）
        hour = time.localtime().tm_hour
        time_factor = 1.0
        if 8 <= hour < 20:  # 工作时间
            time_factor = 1.5  # 延长TTL
        else:
            time_factor = 0.7  # 非工作时间，缩短TTL
        
        # 计算最终TTL，确保在合理范围内
        adjusted_ttl = int(default_ttl * frequency_factor * size_factor * time_factor)
        return max(60, min(adjusted_ttl, 86400))  # 限制在1分钟到24小时之间
    
    def _serialize_data(self, value: Any) -> bytes:
        """
        序列化数据，支持多种数据类型
        
        Args:
            value: 要序列化的数据
            
        Returns:
            bytes: 序列化后的数据
        """
        data_type = type(value).__name__
        
        # 准备序列化数据
        serialize_data = {
            'type': data_type,
            'data': value
        }
        
        # 特殊处理不同数据类型
        if data_type in ['DataFrame', 'Series', 'ndarray']:
            # 使用pickle序列化复杂数据类型
            serialized = pickle.dumps(serialize_data)
        elif isinstance(value, (dict, list, tuple, str, int, float, bool, type(None))):
            # 尝试使用JSON序列化简单类型，提高可读性和效率
            try:
                serialized = json.dumps(serialize_data).encode('utf-8')
            except (TypeError, ValueError):
                # 无法JSON序列化的类型回退到pickle
                serialized = pickle.dumps(serialize_data)
        else:
            # 其他类型直接pickle
            serialized = pickle.dumps(serialize_data)
        
        # 压缩大数据
        if len(serialized) > self.intelligent_cache['compression']['threshold'] and self.intelligent_cache['compression']['enabled']:
            compressed = zlib.compress(serialized, level=self.intelligent_cache['compression']['level'])
            # 添加压缩标记
            return b'COMPRESSED:' + compressed
        
        return serialized
    
    def _deserialize_data(self, serialized_value: bytes, data_type: str = None) -> Any:
        """
        反序列化数据，支持多种数据类型
        
        Args:
            serialized_value: 序列化后的数据
            data_type: 可选，预期的数据类型
            
        Returns:
            Any: 反序列化后的数据
        """
        try:
            # 检查是否压缩数据
            if serialized_value.startswith(b'COMPRESSED:'):
                # 解压数据
                compressed_data = serialized_value[11:]  # 移除'COMPRESSED:'前缀
                serialized_value = zlib.decompress(compressed_data)
            
            # 尝试JSON反序列化
            try:
                data = json.loads(serialized_value.decode('utf-8'))
                if isinstance(data, dict) and 'type' in data and 'data' in data:
                    return data['data']
            except (UnicodeDecodeError, json.JSONDecodeError):
                # 尝试pickle反序列化
                data = pickle.loads(serialized_value)
                if isinstance(data, dict) and 'type' in data and 'data' in data:
                    return data['data']
            
            return data
        except Exception as e:
            print(f"反序列化失败: {e}")
            return None
    
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
    
    def warmup(self, warmup_data: List[Tuple[str, Any, int]]):
        """
        缓存预热，预先加载数据到缓存
        
        Args:
            warmup_data: 预热数据列表，格式为 [(key1, value1, ttl1), (key2, value2, ttl2), ...]
            
        Returns:
            bool: 是否成功
        """
        print(f"开始缓存预热，共 {len(warmup_data)} 项数据")
        start_time = time.time()
        
        success = True
        for key, value, ttl in warmup_data:
            try:
                self.set(key, value, ttl)
            except Exception as e:
                print(f"缓存预热失败: {key}, {e}")
                success = False
        
        end_time = time.time()
        print(f"缓存预热完成，耗时 {end_time - start_time:.2f} 秒")
        return success
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        获取详细的缓存统计信息
        
        Returns:
            Dict: 缓存统计信息
        """
        # 计算缓存使用情况
        memory_usage = {
            'keys': len(self.memory_cache),
            'total_size': sum(self.intelligent_cache['cache_sizes'].values()),
            'average_size': sum(self.intelligent_cache['cache_sizes'].values()) / len(self.memory_cache) if self.memory_cache else 0
        }
        
        return {
            'cache_hits': self.cache_stats['hits'],
            'cache_misses': self.cache_stats['misses'],
            'memory_hits': self.cache_stats['memory_hits'],
            'redis_hits': self.cache_stats['redis_hits'],
            'hit_rate': self.cache_stats['hits'] / (self.cache_stats['hits'] + self.cache_stats['misses']) * 100 if (self.cache_stats['hits'] + self.cache_stats['misses']) > 0 else 0,
            'memory_usage': memory_usage,
            'redis_available': self.is_redis_available,
            'access_count': self.intelligent_cache['access_count'],
            'dynamic_ttl': self.intelligent_cache['dynamic_ttl']
        }
    
    def set_compression(self, enabled: bool, threshold: int = 1024 * 1024, level: int = 3):
        """
        设置缓存压缩配置
        
        Args:
            enabled: 是否启用压缩
            threshold: 压缩阈值（字节）
            level: 压缩级别（1-9）
        """
        self.intelligent_cache['compression']['enabled'] = enabled
        self.intelligent_cache['compression']['threshold'] = threshold
        self.intelligent_cache['compression']['level'] = max(1, min(9, level))
    
    def clear_expired(self):
        """
        清理过期的内存缓存
        
        Returns:
            int: 清理的过期缓存数量
        """
        current_time = time.time()
        expired_keys = []
        
        for key, cache_data in self.memory_cache.items():
            if cache_data['expire_time'] and current_time > cache_data['expire_time']:
                expired_keys.append(key)
        
        for key in expired_keys:
            self._remove_from_memory(key)
        
        return len(expired_keys)
    
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
