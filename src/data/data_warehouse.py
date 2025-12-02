import pandas as pd
import numpy as np
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import json

# 获取日志记录器
logger = logging.getLogger(__name__)

class DataWarehouse:
    """
    统一数据仓库，整合所有供应链数据
    """
    
    def __init__(self):
        self.data_store: Dict[str, pd.DataFrame] = {}
        self.metadata: Dict[str, Dict] = {}
        self.data_lineage: List[Dict] = []
        self.last_updated: Dict[str, float] = {}
        self.real_time_sources: List[Any] = []
    
    def register_data_source(self, source_name: str, data_source: Any, priority: int = 1):
        """
        注册数据源
        
        Args:
            source_name: 数据源名称
            data_source: 数据源实例
            priority: 数据源优先级，数值越小优先级越高
        """
        self.metadata[source_name] = {
            'priority': priority,
            'type': type(data_source).__name__,
            'registered_at': datetime.now().isoformat()
        }
        logger.info(f"数据源 {source_name} 已注册，优先级: {priority}")
    
    def load_data(self, table_name: str, source_name: Optional[str] = None):
        """
        从数据源加载数据到数据仓库
        
        Args:
            table_name: 表名
            source_name: 可选，指定数据源名称
            
        Returns:
            pd.DataFrame: 加载的数据
        """
        start_time = time.time()
        
        # 记录数据血缘起始
        lineage_record = {
            'table_name': table_name,
            'start_time': datetime.now().isoformat(),
            'operations': []
        }
        
        # 获取所有数据源，按优先级排序
        sorted_sources = sorted(self.metadata.items(), key=lambda x: x[1]['priority'])
        
        data = None
        used_source = None
        
        # 如果指定了数据源，只使用该数据源
        if source_name:
            if source_name in self.metadata:
                sorted_sources = [(source_name, self.metadata[source_name])]
            else:
                raise ValueError(f"数据源 {source_name} 未注册")
        
        # 尝试从数据源加载数据
        for source_name, source_meta in sorted_sources:
            try:
                # 假设数据源有get_*方法来获取对应表的数据
                get_method = getattr(self, f"_load_{table_name}", None)
                if get_method:
                    data = get_method(source_name)
                else:
                    # 通用方法，假设数据源有对应的get_*方法
                    data_source = self.metadata[source_name].get('instance')
                    if data_source and hasattr(data_source, f'get_{table_name}'):
                        data = getattr(data_source, f'get_{table_name}')()
                
                if data is not None:
                    used_source = source_name
                    break
            except Exception as e:
                logger.warning(f"从数据源 {source_name} 加载 {table_name} 失败: {str(e)}")
                continue
        
        if data is None:
            raise ValueError(f"无法从任何数据源加载 {table_name} 数据")
        
        # 记录数据加载操作
        lineage_record['operations'].append({
            'type': 'load',
            'source': used_source,
            'timestamp': datetime.now().isoformat(),
            'rows_loaded': len(data)
        })
        
        # 数据转换和验证
        try:
            from data_transformer import DataTransformer
            from data_validator import DataValidator
            
            # 转换数据
            transform_method = getattr(DataTransformer, f'transform_{table_name}', None)
            if transform_method:
                data = transform_method(data)
                lineage_record['operations'].append({
                    'type': 'transform',
                    'timestamp': datetime.now().isoformat(),
                    'transformer': f'DataTransformer.transform_{table_name}'
                })
            
            # 验证数据
            validate_method = getattr(DataValidator, f'validate_{table_name}', None)
            if validate_method:
                is_valid, errors = validate_method(data)
                if not is_valid:
                    logger.warning(f"数据 {table_name} 验证失败: {errors}")
                lineage_record['operations'].append({
                    'type': 'validate',
                    'timestamp': datetime.now().isoformat(),
                    'validator': f'DataValidator.validate_{table_name}',
                    'is_valid': is_valid
                })
        except Exception as e:
            logger.warning(f"数据转换或验证失败: {str(e)}")
        
        # 存储数据到数据仓库
        self.data_store[table_name] = data
        self.last_updated[table_name] = time.time()
        
        # 记录数据血缘结束
        lineage_record['end_time'] = datetime.now().isoformat()
        lineage_record['total_time'] = time.time() - start_time
        self.data_lineage.append(lineage_record)
        
        logger.info(f"表 {table_name} 已加载到数据仓库，从数据源 {used_source} 加载了 {len(data)} 行数据")
        return data
    
    def get_data(self, table_name: str, refresh: bool = False, source_name: Optional[str] = None):
        """
        从数据仓库获取数据
        
        Args:
            table_name: 表名
            refresh: 是否刷新数据
            source_name: 可选，指定数据源名称
            
        Returns:
            pd.DataFrame: 请求的数据
        """
        if refresh or table_name not in self.data_store:
            return self.load_data(table_name, source_name)
        return self.data_store[table_name]
    
    def update_data(self, table_name: str, new_data: pd.DataFrame, update_type: str = 'append'):
        """
        更新数据仓库中的数据
        
        Args:
            table_name: 表名
            new_data: 新数据
            update_type: 更新类型，'append' 或 'replace'
        """
        start_time = time.time()
        
        # 记录数据血缘
        lineage_record = {
            'table_name': table_name,
            'start_time': datetime.now().isoformat(),
            'operations': [{
                'type': 'update',
                'update_type': update_type,
                'timestamp': datetime.now().isoformat(),
                'rows_updated': len(new_data)
            }]
        }
        
        if table_name in self.data_store:
            if update_type == 'append':
                self.data_store[table_name] = pd.concat([self.data_store[table_name], new_data], ignore_index=True)
            elif update_type == 'replace':
                self.data_store[table_name] = new_data
        else:
            self.data_store[table_name] = new_data
        
        self.last_updated[table_name] = time.time()
        
        # 完成数据血缘记录
        lineage_record['end_time'] = datetime.now().isoformat()
        lineage_record['total_time'] = time.time() - start_time
        self.data_lineage.append(lineage_record)
        
        logger.info(f"表 {table_name} 已更新，更新类型: {update_type}，更新了 {len(new_data)} 行数据")
    
    def get_data_lineage(self, table_name: Optional[str] = None, start_time: Optional[str] = None, end_time: Optional[str] = None):
        """
        获取数据血缘信息
        
        Args:
            table_name: 可选，表名
            start_time: 可选，开始时间
            end_time: 可选，结束时间
            
        Returns:
            List[Dict]: 数据血缘记录
        """
        lineage = self.data_lineage.copy()
        
        # 过滤条件
        if table_name:
            lineage = [l for l in lineage if l['table_name'] == table_name]
        
        if start_time:
            lineage = [l for l in lineage if l['start_time'] >= start_time]
        
        if end_time:
            lineage = [l for l in lineage if l['end_time'] <= end_time]
        
        return lineage
    
    def add_real_time_data_stream(self, stream_name: str, stream_processor: Any):
        """
        添加实时数据流
        
        Args:
            stream_name: 数据流名称
            stream_processor: 数据流处理器，负责处理实时数据
        """
        self.real_time_sources.append({
            'name': stream_name,
            'processor': stream_processor,
            'added_at': datetime.now().isoformat()
        })
        logger.info(f"实时数据流 {stream_name} 已添加")
    
    def process_real_time_data(self, stream_name: Optional[str] = None):
        """
        处理实时数据
        
        Args:
            stream_name: 可选，指定数据流名称
        """
        for stream in self.real_time_sources:
            if stream_name and stream['name'] != stream_name:
                continue
            
            try:
                processor = stream['processor']
                if hasattr(processor, 'process'):
                    new_data = processor.process()
                    # 假设new_data是一个字典，键为表名，值为数据
                    for table_name, data in new_data.items():
                        self.update_data(table_name, data, update_type='append')
                        logger.info(f"从实时数据流 {stream['name']} 更新了 {table_name} 表，新增 {len(data)} 行数据")
            except Exception as e:
                logger.error(f"处理实时数据流 {stream['name']} 失败: {str(e)}")
    
    def get_data_quality_metrics(self, table_name: str):
        """
        获取数据质量指标
        
        Args:
            table_name: 表名
            
        Returns:
            Dict: 数据质量指标
        """
        if table_name not in self.data_store:
            raise ValueError(f"表 {table_name} 不存在于数据仓库中")
        
        data = self.data_store[table_name]
        
        # 计算基本数据质量指标
        metrics = {
            'table_name': table_name,
            'timestamp': datetime.now().isoformat(),
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'missing_values': data.isnull().sum().sum(),
            'duplicate_rows': data.duplicated().sum(),
            'last_updated': self.last_updated.get(table_name)
        }
        
        # 计算每列的缺失值比例
        metrics['column_missing_ratios'] = {
            col: data[col].isnull().sum() / len(data) for col in data.columns
        }
        
        return metrics
