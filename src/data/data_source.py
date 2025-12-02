import pandas as pd
import numpy as np
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime

# 获取日志记录器
logger = logging.getLogger(__name__)

class DataSource(ABC):
    """
    数据源抽象基类，定义统一的数据获取接口
    """
    
    @abstractmethod
    def get_items(self):
        """
        获取商品信息数据
        
        Returns:
            pandas.DataFrame: 商品信息数据框
        """
        pass
    
    @abstractmethod
    def get_locations(self):
        """
        获取仓库/位置数据
        
        Returns:
            pandas.DataFrame: 仓库/位置数据框
        """
        pass
    
    @abstractmethod
    def get_suppliers(self):
        """
        获取供应商信息数据
        
        Returns:
            pandas.DataFrame: 供应商信息数据框
        """
        pass
    
    @abstractmethod
    def get_inventory_daily(self):
        """
        获取每日库存数据
        
        Returns:
            pandas.DataFrame: 每日库存数据框
        """
        pass
    
    @abstractmethod
    def get_purchase_orders(self):
        """
        获取采购订单数据
        
        Returns:
            pandas.DataFrame: 采购订单数据框
        """
        pass
    
    @abstractmethod
    def get_forecast_output(self):
        """
        获取预测输出数据
        
        Returns:
            pandas.DataFrame: 预测输出数据框
        """
        pass
    
    @abstractmethod
    def get_optimal_plan(self):
        """
        获取MILP优化输出数据
        
        Returns:
            pandas.DataFrame: MILP优化输出数据框
        """
        pass


class CSVDataSource(DataSource):
    """
    CSV文件数据源，从本地CSV文件加载数据
    """
    
    def __init__(self, data_dir='./data', file_mapping=None):
        """
        初始化CSV数据源
        
        Args:
            data_dir: 数据文件目录
            file_mapping: 文件映射配置，格式为{table_name: file_name}
        """
        self.data_dir = data_dir
        self.file_mapping = file_mapping or {
            'items': 'items.csv',
            'locations': 'locations.csv',
            'suppliers': 'suppliers.csv',
            'inventory_daily': 'inventory_daily.csv',
            'purchase_orders': 'purchase_orders.csv',
            'forecast_output': 'forecast_output.csv',
            'optimal_plan': 'optimal_plan.csv'
        }
    
    def _load_csv(self, table_name):
        """
        加载指定表的CSV文件
        
        Args:
            table_name: 表名
            
        Returns:
            pandas.DataFrame: 数据框
        """
        file_name = self.file_mapping.get(table_name)
        if not file_name:
            raise ValueError(f"表名 {table_name} 未在文件映射中配置")
        
        file_path = f"{self.data_dir}/{file_name}"
        logger.info(f"从文件加载数据: {file_path}")
        
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            logger.error(f"加载文件 {file_path} 失败: {e}")
            raise
    
    def get_items(self):
        return self._load_csv('items')
    
    def get_locations(self):
        return self._load_csv('locations')
    
    def get_suppliers(self):
        return self._load_csv('suppliers')
    
    def get_inventory_daily(self):
        return self._load_csv('inventory_daily')
    
    def get_purchase_orders(self):
        return self._load_csv('purchase_orders')
    
    def get_forecast_output(self):
        return self._load_csv('forecast_output')
    
    def get_optimal_plan(self):
        return self._load_csv('optimal_plan')


class DatabaseDataSource(DataSource):
    """
    数据库数据源，从关系型数据库加载数据
    """
    
    def __init__(self, connection_string, query_mapping=None):
        """
        初始化数据库数据源
        
        Args:
            connection_string: 数据库连接字符串
            query_mapping: 查询映射配置，格式为{table_name: query_sql}
        """
        self.connection_string = connection_string
        self.query_mapping = query_mapping or {
            'items': 'SELECT * FROM items',
            'locations': 'SELECT * FROM locations',
            'suppliers': 'SELECT * FROM suppliers',
            'inventory_daily': 'SELECT * FROM inventory_daily',
            'purchase_orders': 'SELECT * FROM purchase_orders',
            'forecast_output': 'SELECT * FROM forecast_output',
            'optimal_plan': 'SELECT * FROM optimal_plan'
        }
    
    def _execute_query(self, table_name):
        """
        执行指定表的查询
        
        Args:
            table_name: 表名
            
        Returns:
            pandas.DataFrame: 数据框
        """
        query = self.query_mapping.get(table_name)
        if not query:
            raise ValueError(f"表名 {table_name} 未在查询映射中配置")
        
        logger.info(f"执行数据库查询: {query}")
        
        try:
            # 使用pandas读取数据库
            return pd.read_sql_query(query, self.connection_string)
        except Exception as e:
            logger.error(f"执行查询失败: {e}")
            raise
    
    def get_items(self):
        return self._execute_query('items')
    
    def get_locations(self):
        return self._execute_query('locations')
    
    def get_suppliers(self):
        return self._execute_query('suppliers')
    
    def get_inventory_daily(self):
        return self._execute_query('inventory_daily')
    
    def get_purchase_orders(self):
        return self._execute_query('purchase_orders')
    
    def get_forecast_output(self):
        return self._execute_query('forecast_output')
    
    def get_optimal_plan(self):
        return self._execute_query('optimal_plan')


class APIDataSource(DataSource):
    """
    API数据源，从REST API加载数据
    """
    
    def __init__(self, base_url, headers=None, endpoint_mapping=None):
        """
        初始化API数据源
        
        Args:
            base_url: API基础URL
            headers: HTTP请求头
            endpoint_mapping: 端点映射配置，格式为{table_name: endpoint}
        """
        self.base_url = base_url
        self.headers = headers or {}
        self.endpoint_mapping = endpoint_mapping or {
            'items': '/api/items',
            'locations': '/api/locations',
            'suppliers': '/api/suppliers',
            'inventory_daily': '/api/inventory/daily',
            'purchase_orders': '/api/purchase/orders',
            'forecast_output': '/api/forecast/output',
            'optimal_plan': '/api/optimization/plan'
        }
    
    def _fetch_api(self, table_name):
        """
        从API获取指定表的数据
        
        Args:
            table_name: 表名
            
        Returns:
            pandas.DataFrame: 数据框
        """
        import requests
        
        endpoint = self.endpoint_mapping.get(table_name)
        if not endpoint:
            raise ValueError(f"表名 {table_name} 未在端点映射中配置")
        
        url = f"{self.base_url}{endpoint}"
        logger.info(f"从API获取数据: {url}")
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"从API获取数据失败: {e}")
            raise
    
    def get_items(self):
        return self._fetch_api('items')
    
    def get_locations(self):
        return self._fetch_api('locations')
    
    def get_suppliers(self):
        return self._fetch_api('suppliers')
    
    def get_inventory_daily(self):
        return self._fetch_api('inventory_daily')
    
    def get_purchase_orders(self):
        return self._fetch_api('purchase_orders')
    
    def get_forecast_output(self):
        return self._fetch_api('forecast_output')
    
    def get_optimal_plan(self):
        return self._fetch_api('optimal_plan')


class SimulatedDataSource(DataSource):
    """
    模拟数据源，从模拟数据生成器获取数据
    保留此类以便于测试和开发
    """
    
    def __init__(self):
        from simulated_data import generate_simulated_data
        self.generate_simulated_data = generate_simulated_data
    
    def _get_all_data(self):
        """
        获取所有模拟数据
        
        Returns:
            dict: 包含所有表数据的字典
        """
        logger.info("生成模拟数据")
        return self.generate_simulated_data()
    
    def get_items(self):
        return self._get_all_data()['items']
    
    def get_locations(self):
        return self._get_all_data()['locations']
    
    def get_suppliers(self):
        return self._get_all_data()['suppliers']
    
    def get_inventory_daily(self):
        return self._get_all_data()['inventory_daily']
    
    def get_purchase_orders(self):
        return self._get_all_data()['purchase_orders']
    
    def get_forecast_output(self):
        return self._get_all_data()['forecast_output']
    
    def get_optimal_plan(self):
        return self._get_all_data()['optimal_plan']


class DataSourceFactory:
    """
    数据源工厂类，用于创建不同类型的数据源实例
    """
    
    @staticmethod
    def create_data_source(source_type, config=None):
        """
        创建数据源实例
        
        Args:
            source_type: 数据源类型，可选值: 'csv', 'database', 'api', 'simulated'
            config: 数据源配置
            
        Returns:
            DataSource: 数据源实例
        """
        config = config or {}
        
        if source_type == 'csv':
            return CSVDataSource(
                data_dir=config.get('data_dir', './data'),
                file_mapping=config.get('file_mapping')
            )
        elif source_type == 'database':
            return DatabaseDataSource(
                connection_string=config.get('connection_string'),
                query_mapping=config.get('query_mapping')
            )
        elif source_type == 'api':
            return APIDataSource(
                base_url=config.get('base_url'),
                headers=config.get('headers'),
                endpoint_mapping=config.get('endpoint_mapping')
            )
        elif source_type == 'simulated':
            return SimulatedDataSource()
        else:
            raise ValueError(f"不支持的数据源类型: {source_type}")


class DataTransformer:
    """
    数据转换器，用于将生产数据转换为系统内部数据格式
    """
    
    @staticmethod
    def transform_items(items_df, mapping=None):
        """
        转换商品信息数据
        
        Args:
            items_df: 原始商品数据框
            mapping: 字段映射配置
            
        Returns:
            pandas.DataFrame: 转换后的商品数据框
        """
        # 默认映射，将生产数据字段映射到系统内部字段
        default_mapping = {
            'item_id': 'item_id',
            'abc_class': 'abc_class',
            'min_order_qty': 'min_order_qty',
            'pack_size': 'pack_size',
            'safety_factor_z': 'safety_factor_z',
            'uom': 'uom',
            'unit_price': 'unit_price',
            'category': 'category',
            'brand': 'brand'
        }
        
        mapping = mapping or default_mapping
        return items_df.rename(columns=mapping)
    
    @staticmethod
    def transform_locations(locations_df, mapping=None):
        """
        转换仓库/位置数据
        
        Args:
            locations_df: 原始仓库/位置数据框
            mapping: 字段映射配置
            
        Returns:
            pandas.DataFrame: 转换后的仓库/位置数据框
        """
        default_mapping = {
            'location_id': 'location_id',
            'capacity_limit': 'capacity_limit',
            'transfer_cost_per_unit': 'transfer_cost_per_unit',
            'region': 'region'
        }
        
        mapping = mapping or default_mapping
        return locations_df.rename(columns=mapping)
    
    @staticmethod
    def transform_suppliers(suppliers_df, mapping=None):
        """
        转换供应商信息数据
        
        Args:
            suppliers_df: 原始供应商数据框
            mapping: 字段映射配置
            
        Returns:
            pandas.DataFrame: 转换后的供应商数据框
        """
        default_mapping = {
            'supplier_id': 'supplier_id',
            'lead_time_days': 'lead_time_days',
            'price_breaks': 'price_breaks',
            'rating': 'rating',
            'type': 'type'
        }
        
        mapping = mapping or default_mapping
        transformed_df = suppliers_df.rename(columns=mapping)
        
        # 确保price_breaks是JSON格式
        if 'price_breaks' in transformed_df.columns:
            def ensure_json(x):
                if isinstance(x, str):
                    try:
                        json.loads(x)
                        return x
                    except:
                        # 如果不是有效的JSON，尝试转换
                        return json.dumps(x)
                return json.dumps(x)
            
            transformed_df['price_breaks'] = transformed_df['price_breaks'].apply(ensure_json)
        
        return transformed_df
    
    @staticmethod
    def transform_inventory_daily(inventory_df, mapping=None):
        """
        转换每日库存数据
        
        Args:
            inventory_df: 原始每日库存数据框
            mapping: 字段映射配置
            
        Returns:
            pandas.DataFrame: 转换后的每日库存数据框
        """
        default_mapping = {
            'date': 'date',
            'item_id': 'item_id',
            'location_id': 'location_id',
            'on_hand_qty': 'on_hand_qty',
            'demand_qty': 'demand_qty',
            'receipts_qty': 'receipts_qty',
            'seasonal_factor': 'seasonal_factor'
        }
        
        mapping = mapping or default_mapping
        transformed_df = inventory_df.rename(columns=mapping)
        
        # 确保日期格式正确
        if 'date' in transformed_df.columns:
            transformed_df['date'] = pd.to_datetime(transformed_df['date']).dt.strftime('%Y-%m-%d')
        
        return transformed_df
    
    @staticmethod
    def transform_purchase_orders(po_df, mapping=None):
        """
        转换采购订单数据
        
        Args:
            po_df: 原始采购订单数据框
            mapping: 字段映射配置
            
        Returns:
            pandas.DataFrame: 转换后的采购订单数据框
        """
        default_mapping = {
            'po_id': 'po_id',
            'item_id': 'item_id',
            'supplier_id': 'supplier_id',
            'order_date': 'order_date',
            'due_date': 'due_date',
            'order_qty': 'order_qty',
            'status': 'status',
            'location_id': 'location_id',
            'unit_price': 'unit_price',
            'priority': 'priority',
            'created_by': 'created_by'
        }
        
        mapping = mapping or default_mapping
        transformed_df = po_df.rename(columns=mapping)
        
        # 确保日期格式正确
        for date_col in ['order_date', 'due_date']:
            if date_col in transformed_df.columns:
                transformed_df[date_col] = pd.to_datetime(transformed_df[date_col]).dt.strftime('%Y-%m-%d')
        
        return transformed_df
    
    @staticmethod
    def transform_forecast_output(forecast_df, mapping=None):
        """
        转换预测输出数据
        
        Args:
            forecast_df: 原始预测输出数据框
            mapping: 字段映射配置
            
        Returns:
            pandas.DataFrame: 转换后的预测输出数据框
        """
        default_mapping = {
            'item_id': 'item_id',
            'location_id': 'location_id',
            'horizon_date': 'horizon_date',
            'yhat': 'yhat',
            'model_used': 'model_used',
            'mape_recent': 'mape_recent',
            'smape_recent': 'smape_recent',
            'confidence_level': 'confidence_level',
            'lower_bound': 'lower_bound',
            'upper_bound': 'upper_bound',
            'seasonal_factor': 'seasonal_factor'
        }
        
        mapping = mapping or default_mapping
        transformed_df = forecast_df.rename(columns=mapping)
        
        # 确保日期格式正确
        if 'horizon_date' in transformed_df.columns:
            transformed_df['horizon_date'] = pd.to_datetime(transformed_df['horizon_date']).dt.strftime('%Y-%m-%d')
        
        return transformed_df
    
    @staticmethod
    def transform_optimal_plan(optimal_df, mapping=None):
        """
        转换MILP优化输出数据
        
        Args:
            optimal_df: 原始MILP优化输出数据框
            mapping: 字段映射配置
            
        Returns:
            pandas.DataFrame: 转换后的MILP优化输出数据框
        """
        default_mapping = {
            'item_id': 'item_id',
            'location_id': 'location_id',
            'supplier_id': 'supplier_id',
            'order_qty': 'order_qty',
            'unit_price': 'unit_price',
            'tier_chosen': 'tier_chosen',
            'due_date': 'due_date',
            'transfer_from': 'transfer_from',
            'transfer_qty': 'transfer_qty',
            'rationale': 'rationale',
            'plan_date': 'plan_date',
            'optimization_goal': 'optimization_goal',
            'confidence_score': 'confidence_score',
            'estimated_savings': 'estimated_savings'
        }
        
        mapping = mapping or default_mapping
        transformed_df = optimal_df.rename(columns=mapping)
        
        # 确保日期格式正确
        for date_col in ['due_date', 'plan_date']:
            if date_col in transformed_df.columns:
                transformed_df[date_col] = pd.to_datetime(transformed_df[date_col]).dt.strftime('%Y-%m-%d')
        
        return transformed_df


class DataValidator:
    """
    数据验证器，用于验证数据的完整性和正确性
    """
    
    @staticmethod
    def validate_items(items_df):
        """
        验证商品信息数据
        
        Args:
            items_df: 商品数据框
            
        Returns:
            bool: 验证是否通过
            list: 错误信息列表
        """
        errors = []
        
        # 检查必需字段
        required_fields = ['item_id', 'abc_class', 'min_order_qty', 'pack_size', 'uom', 'unit_price']
        for field in required_fields:
            if field not in items_df.columns:
                errors.append(f"商品数据缺少必需字段: {field}")
            elif items_df[field].isnull().any():
                errors.append(f"商品数据字段 {field} 包含空值")
        
        # 检查数据类型
        if 'min_order_qty' in items_df.columns:
            if not pd.api.types.is_integer_dtype(items_df['min_order_qty']):
                errors.append("商品数据 min_order_qty 字段应为整数类型")
        
        if 'pack_size' in items_df.columns:
            if not pd.api.types.is_integer_dtype(items_df['pack_size']):
                errors.append("商品数据 pack_size 字段应为整数类型")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_locations(locations_df):
        """
        验证仓库/位置数据
        
        Args:
            locations_df: 仓库/位置数据框
            
        Returns:
            bool: 验证是否通过
            list: 错误信息列表
        """
        errors = []
        
        # 检查必需字段
        required_fields = ['location_id', 'capacity_limit', 'transfer_cost_per_unit']
        for field in required_fields:
            if field not in locations_df.columns:
                errors.append(f"仓库数据缺少必需字段: {field}")
            elif locations_df[field].isnull().any():
                errors.append(f"仓库数据字段 {field} 包含空值")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_suppliers(suppliers_df):
        """
        验证供应商信息数据
        
        Args:
            suppliers_df: 供应商数据框
            
        Returns:
            bool: 验证是否通过
            list: 错误信息列表
        """
        errors = []
        
        # 检查必需字段
        required_fields = ['supplier_id', 'lead_time_days', 'price_breaks']
        for field in required_fields:
            if field not in suppliers_df.columns:
                errors.append(f"供应商数据缺少必需字段: {field}")
            elif suppliers_df[field].isnull().any():
                errors.append(f"供应商数据字段 {field} 包含空值")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_inventory_daily(inventory_df):
        """
        验证每日库存数据
        
        Args:
            inventory_df: 每日库存数据框
            
        Returns:
            bool: 验证是否通过
            list: 错误信息列表
        """
        errors = []
        
        # 检查必需字段
        required_fields = ['date', 'item_id', 'location_id', 'on_hand_qty', 'demand_qty', 'receipts_qty']
        for field in required_fields:
            if field not in inventory_df.columns:
                errors.append(f"库存数据缺少必需字段: {field}")
            elif inventory_df[field].isnull().any():
                errors.append(f"库存数据字段 {field} 包含空值")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_purchase_orders(po_df):
        """
        验证采购订单数据
        
        Args:
            po_df: 采购订单数据框
            
        Returns:
            bool: 验证是否通过
            list: 错误信息列表
        """
        errors = []
        
        # 检查必需字段
        required_fields = ['po_id', 'item_id', 'supplier_id', 'order_date', 'due_date', 'order_qty', 'status']
        for field in required_fields:
            if field not in po_df.columns:
                errors.append(f"采购订单数据缺少必需字段: {field}")
            elif po_df[field].isnull().any():
                errors.append(f"采购订单数据字段 {field} 包含空值")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_forecast_output(forecast_df):
        """
        验证预测输出数据
        
        Args:
            forecast_df: 预测输出数据框
            
        Returns:
            bool: 验证是否通过
            list: 错误信息列表
        """
        errors = []
        
        # 检查必需字段
        required_fields = ['item_id', 'location_id', 'horizon_date', 'yhat', 'model_used']
        for field in required_fields:
            if field not in forecast_df.columns:
                errors.append(f"预测数据缺少必需字段: {field}")
            elif forecast_df[field].isnull().any():
                errors.append(f"预测数据字段 {field} 包含空值")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_optimal_plan(optimal_df):
        """
        验证MILP优化输出数据
        
        Args:
            optimal_df: MILP优化输出数据框
            
        Returns:
            bool: 验证是否通过
            list: 错误信息列表
        """
        errors = []
        
        # 检查必需字段
        required_fields = ['item_id', 'location_id', 'supplier_id', 'order_qty', 'due_date']
        for field in required_fields:
            if field not in optimal_df.columns:
                errors.append(f"优化计划数据缺少必需字段: {field}")
            elif optimal_df[field].isnull().any():
                errors.append(f"优化计划数据字段 {field} 包含空值")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_all(items_df, locations_df, suppliers_df, inventory_df, po_df, forecast_df, optimal_df):
        """
        验证所有数据
        
        Args:
            items_df: 商品数据框
            locations_df: 仓库/位置数据框
            suppliers_df: 供应商数据框
            inventory_df: 每日库存数据框
            po_df: 采购订单数据框
            forecast_df: 预测输出数据框
            optimal_df: MILP优化输出数据框
            
        Returns:
            bool: 验证是否通过
            dict: 各表的错误信息
        """
        all_errors = {}
        
        # 验证各个表
        is_valid, errors = DataValidator.validate_items(items_df)
        if not is_valid:
            all_errors['items'] = errors
        
        is_valid, errors = DataValidator.validate_locations(locations_df)
        if not is_valid:
            all_errors['locations'] = errors
        
        is_valid, errors = DataValidator.validate_suppliers(suppliers_df)
        if not is_valid:
            all_errors['suppliers'] = errors
        
        is_valid, errors = DataValidator.validate_inventory_daily(inventory_df)
        if not is_valid:
            all_errors['inventory_daily'] = errors
        
        is_valid, errors = DataValidator.validate_purchase_orders(po_df)
        if not is_valid:
            all_errors['purchase_orders'] = errors
        
        is_valid, errors = DataValidator.validate_forecast_output(forecast_df)
        if not is_valid:
            all_errors['forecast_output'] = errors
        
        is_valid, errors = DataValidator.validate_optimal_plan(optimal_df)
        if not is_valid:
            all_errors['optimal_plan'] = errors
        
        return len(all_errors) == 0, all_errors
