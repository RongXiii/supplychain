import pandas as pd
import numpy as np
import logging
from typing import Tuple, List, Dict, Any

# 获取日志记录器
logger = logging.getLogger(__name__)

class DataTransformer:
    """
    数据转换器，负责处理不同类型的数据转换
    """
    
    @staticmethod
    def transform_sales_data(data: pd.DataFrame) -> pd.DataFrame:
        """
        转换销售数据
        
        Args:
            data: 原始销售数据
            
        Returns:
            pd.DataFrame: 转换后的销售数据
        """
        try:
            # 确保日期列是datetime类型
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
            elif 'timestamp' in data.columns:
                data['date'] = pd.to_datetime(data['timestamp'])
                data = data.drop(columns=['timestamp'])
            
            # 确保销售数量是数值类型
            if 'sales' in data.columns:
                data['sales'] = pd.to_numeric(data['sales'], errors='coerce')
            
            # 确保SKU列是字符串类型
            if 'sku' in data.columns:
                data['sku'] = data['sku'].astype(str)
            
            # 移除重复行
            data = data.drop_duplicates()
            
            # 按日期排序
            if 'date' in data.columns:
                data = data.sort_values('date')
            
            logger.info(f"销售数据转换完成，处理了 {len(data)} 行数据")
            return data
        except Exception as e:
            logger.error(f"转换销售数据失败: {str(e)}")
            return data
    
    @staticmethod
    def transform_inventory_data(data: pd.DataFrame) -> pd.DataFrame:
        """
        转换库存数据
        
        Args:
            data: 原始库存数据
            
        Returns:
            pd.DataFrame: 转换后的库存数据
        """
        try:
            # 确保日期列是datetime类型
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
            
            # 确保库存数量是数值类型
            if 'inventory_level' in data.columns:
                data['inventory_level'] = pd.to_numeric(data['inventory_level'], errors='coerce')
            
            # 确保SKU列是字符串类型
            if 'sku' in data.columns:
                data['sku'] = data['sku'].astype(str)
            
            # 移除重复行
            data = data.drop_duplicates()
            
            # 按日期排序
            if 'date' in data.columns:
                data = data.sort_values('date')
            
            logger.info(f"库存数据转换完成，处理了 {len(data)} 行数据")
            return data
        except Exception as e:
            logger.error(f"转换库存数据失败: {str(e)}")
            return data
    
    @staticmethod
    def transform_product_data(data: pd.DataFrame) -> pd.DataFrame:
        """
        转换产品数据
        
        Args:
            data: 原始产品数据
            
        Returns:
            pd.DataFrame: 转换后的产品数据
        """
        try:
            # 确保SKU列是字符串类型
            if 'sku' in data.columns:
                data['sku'] = data['sku'].astype(str)
            
            # 确保成本和价格是数值类型
            for col in ['cost', 'price', 'lead_time']:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # 移除重复行
            data = data.drop_duplicates()
            
            logger.info(f"产品数据转换完成，处理了 {len(data)} 行数据")
            return data
        except Exception as e:
            logger.error(f"转换产品数据失败: {str(e)}")
            return data
    
    @staticmethod
    def transform_demand_data(data: pd.DataFrame) -> pd.DataFrame:
        """
        转换需求数据
        
        Args:
            data: 原始需求数据
            
        Returns:
            pd.DataFrame: 转换后的需求数据
        """
        try:
            # 确保日期列是datetime类型
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
            
            # 确保需求数量是数值类型
            if 'demand' in data.columns:
                data['demand'] = pd.to_numeric(data['demand'], errors='coerce')
            
            # 确保SKU列是字符串类型
            if 'sku' in data.columns:
                data['sku'] = data['sku'].astype(str)
            
            # 移除重复行
            data = data.drop_duplicates()
            
            # 按日期排序
            if 'date' in data.columns:
                data = data.sort_values('date')
            
            logger.info(f"需求数据转换完成，处理了 {len(data)} 行数据")
            return data
        except Exception as e:
            logger.error(f"转换需求数据失败: {str(e)}")
            return data


class DataValidator:
    """
    数据验证器，负责验证数据质量
    """
    
    @staticmethod
    def validate_sales_data(data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        验证销售数据
        
        Args:
            data: 销售数据
            
        Returns:
            Tuple[bool, List[str]]: (是否有效, 错误信息列表)
        """
        errors = []
        
        # 检查必要列
        required_columns = ['date', 'sku', 'sales']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            errors.append(f"缺失必要列: {', '.join(missing_columns)}")
        
        # 检查日期列
        if 'date' in data.columns:
            if data['date'].isnull().any():
                errors.append(f"日期列包含 {data['date'].isnull().sum()} 个空值")
        
        # 检查销售列
        if 'sales' in data.columns:
            if data['sales'].isnull().any():
                errors.append(f"销售数量列包含 {data['sales'].isnull().sum()} 个空值")
            if (data['sales'] < 0).any():
                errors.append(f"销售数量列包含 {len(data[data['sales'] < 0])} 个负值")
        
        # 检查SKU列
        if 'sku' in data.columns:
            if data['sku'].isnull().any():
                errors.append(f"SKU列包含 {data['sku'].isnull().sum()} 个空值")
        
        # 检查数据量
        if len(data) == 0:
            errors.append("销售数据为空")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    @staticmethod
    def validate_inventory_data(data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        验证库存数据
        
        Args:
            data: 库存数据
            
        Returns:
            Tuple[bool, List[str]]: (是否有效, 错误信息列表)
        """
        errors = []
        
        # 检查必要列
        required_columns = ['date', 'sku', 'inventory_level']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            errors.append(f"缺失必要列: {', '.join(missing_columns)}")
        
        # 检查日期列
        if 'date' in data.columns:
            if data['date'].isnull().any():
                errors.append(f"日期列包含 {data['date'].isnull().sum()} 个空值")
        
        # 检查库存列
        if 'inventory_level' in data.columns:
            if data['inventory_level'].isnull().any():
                errors.append(f"库存数量列包含 {data['inventory_level'].isnull().sum()} 个空值")
            if (data['inventory_level'] < 0).any():
                errors.append(f"库存数量列包含 {len(data[data['inventory_level'] < 0])} 个负值")
        
        # 检查SKU列
        if 'sku' in data.columns:
            if data['sku'].isnull().any():
                errors.append(f"SKU列包含 {data['sku'].isnull().sum()} 个空值")
        
        # 检查数据量
        if len(data) == 0:
            errors.append("库存数据为空")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    @staticmethod
    def validate_product_data(data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        验证产品数据
        
        Args:
            data: 产品数据
            
        Returns:
            Tuple[bool, List[str]]: (是否有效, 错误信息列表)
        """
        errors = []
        
        # 检查必要列
        required_columns = ['sku', 'cost', 'price']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            errors.append(f"缺失必要列: {', '.join(missing_columns)}")
        
        # 检查SKU列
        if 'sku' in data.columns:
            if data['sku'].isnull().any():
                errors.append(f"SKU列包含 {data['sku'].isnull().sum()} 个空值")
            if data['sku'].duplicated().any():
                errors.append(f"SKU列包含 {data['sku'].duplicated().sum()} 个重复值")
        
        # 检查成本和价格列
        for col in ['cost', 'price']:
            if col in data.columns:
                if data[col].isnull().any():
                    errors.append(f"{col}列包含 {data[col].isnull().sum()} 个空值")
                if (data[col] <= 0).any():
                    errors.append(f"{col}列包含 {len(data[data[col] <= 0])} 个非正值")
        
        # 检查数据量
        if len(data) == 0:
            errors.append("产品数据为空")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    @staticmethod
    def validate_demand_data(data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        验证需求数据
        
        Args:
            data: 需求数据
            
        Returns:
            Tuple[bool, List[str]]: (是否有效, 错误信息列表)
        """
        errors = []
        
        # 检查必要列
        required_columns = ['date', 'sku', 'demand']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            errors.append(f"缺失必要列: {', '.join(missing_columns)}")
        
        # 检查日期列
        if 'date' in data.columns:
            if data['date'].isnull().any():
                errors.append(f"日期列包含 {data['date'].isnull().sum()} 个空值")
        
        # 检查需求列
        if 'demand' in data.columns:
            if data['demand'].isnull().any():
                errors.append(f"需求数量列包含 {data['demand'].isnull().sum()} 个空值")
            if (data['demand'] < 0).any():
                errors.append(f"需求数量列包含 {len(data[data['demand'] < 0])} 个负值")
        
        # 检查SKU列
        if 'sku' in data.columns:
            if data['sku'].isnull().any():
                errors.append(f"SKU列包含 {data['sku'].isnull().sum()} 个空值")
        
        # 检查数据量
        if len(data) == 0:
            errors.append("需求数据为空")
        
        is_valid = len(errors) == 0
        return is_valid, errors
