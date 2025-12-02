import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, Any

# 获取日志记录器
logger = logging.getLogger(__name__)

class RealTimeDataProcessor:
    """
    实时数据流处理器，负责处理和更新实时数据
    """
    
    def __init__(self, data_warehouse):
        self.data_warehouse = data_warehouse
        self.last_processed_time = datetime.now()
    
    def process(self) -> Dict[str, pd.DataFrame]:
        """
        处理实时数据流，生成新的数据
        
        Returns:
            Dict[str, pd.DataFrame]: 新生成的数据，键为表名，值为数据
        """
        current_time = datetime.now()
        time_diff = current_time - self.last_processed_time
        
        # 模拟实时数据生成
        new_data = {
            'inventory_daily': self._generate_realtime_inventory(),
            'purchase_orders': self._generate_realtime_purchase_orders()
        }
        
        self.last_processed_time = current_time
        return new_data
    
    def _generate_realtime_inventory(self) -> pd.DataFrame:
        """
        生成实时库存数据
        
        Returns:
            pd.DataFrame: 实时库存数据
        """
        try:
            # 从数据仓库获取现有库存数据
            inventory_df = self.data_warehouse.get_data('inventory_daily')
            
            # 如果数据为空，返回空DataFrame
            if inventory_df.empty:
                return pd.DataFrame()
            
            # 获取最新日期
            latest_date = pd.to_datetime(inventory_df['date']).max()
            new_date = latest_date + timedelta(days=1)
            
            # 生成新的库存记录
            new_records = []
            
            # 按sku分组，生成新记录
            for sku in inventory_df['sku'].unique():
                # 获取该sku的最新库存记录
                sku_data = inventory_df[inventory_df['sku'] == sku]
                latest_inventory = sku_data.sort_values('date').iloc[-1]
                
                # 模拟库存变化
                new_inventory = latest_inventory['inventory_on_hand'] + np.random.randint(-5, 10)
                new_inventory = max(0, new_inventory)  # 确保库存不为负
                
                # 生成新记录
                new_record = latest_inventory.copy()
                new_record['date'] = new_date.strftime('%Y-%m-%d')
                new_record['inventory_on_hand'] = new_inventory
                new_record['created_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                new_records.append(new_record)
            
            # 创建新的DataFrame
            new_inventory_df = pd.DataFrame(new_records)
            return new_inventory_df
        except Exception as e:
            logger.error(f"生成实时库存数据失败: {str(e)}")
            return pd.DataFrame()
    
    def _generate_realtime_purchase_orders(self) -> pd.DataFrame:
        """
        生成实时采购订单数据
        
        Returns:
            pd.DataFrame: 实时采购订单数据
        """
        try:
            # 从数据仓库获取现有采购订单数据
            po_df = self.data_warehouse.get_data('purchase_orders')
            
            # 如果数据为空，返回空DataFrame
            if po_df.empty:
                return pd.DataFrame()
            
            # 生成新的采购订单记录
            new_records = []
            
            # 获取最新的采购订单ID
            latest_po_id = po_df['purchase_order_id'].max()
            
            # 生成2-5个新的采购订单
            for i in range(np.random.randint(2, 6)):
                # 随机选择一个供应商和sku
                supplier_id = np.random.choice(po_df['supplier_id'].unique())
                sku = np.random.choice(po_df['sku'].unique())
                
                # 生成订单数量
                order_quantity = np.random.randint(10, 100)
                
                # 生成新的采购订单ID
                new_po_id = latest_po_id + i + 1
                
                # 生成新记录
                new_record = {
                    'purchase_order_id': new_po_id,
                    'supplier_id': supplier_id,
                    'sku': sku,
                    'order_quantity': order_quantity,
                    'order_date': datetime.now().strftime('%Y-%m-%d'),
                    'expected_delivery_date': (datetime.now() + timedelta(days=np.random.randint(5, 15))).strftime('%Y-%m-%d'),
                    'status': 'pending',
                    'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                new_records.append(new_record)
            
            # 创建新的DataFrame
            new_po_df = pd.DataFrame(new_records)
            return new_po_df
        except Exception as e:
            logger.error(f"生成实时采购订单数据失败: {str(e)}")
            return pd.DataFrame()


class RealTimeForecastUpdater:
    """
    实时预测更新器，负责根据实时数据更新预测结果
    """
    
    def __init__(self, replenishment_system):
        self.replenishment_system = replenishment_system
    
    def update_forecasts(self):
        """
        根据实时数据更新预测结果
        """
        try:
            # 从数据仓库获取最新的库存和销售数据
            inventory_df = self.replenishment_system.data_warehouse.get_data('inventory_daily')
            
            # 检查数据是否有更新
            if inventory_df.empty:
                return
            
            # 按sku分组，更新每个sku的预测
            for sku in inventory_df['sku'].unique()[:5]:  # 只更新前5个sku作为示例
                # 获取该sku的库存数据
                sku_data = inventory_df[inventory_df['sku'] == sku]
                
                # 如果数据行数不足，跳过
                if len(sku_data) < 30:
                    continue
                
                # 准备历史数据
                historical_data = sku_data[['date', 'inventory_on_hand']].copy()
                historical_data.columns = ['date', 'sales']
                
                # 更新预测
                forecast_result = self.replenishment_system.run_forecast(historical_data, sku)
                
                # 将新的预测结果更新到数据仓库
                if forecast_result is not None:
                    # 获取现有预测数据
                    forecast_df = self.replenishment_system.data_warehouse.get_data('forecast_output')
                    
                    # 创建新的预测记录
                    new_forecast = pd.DataFrame({
                        'date': [datetime.now().strftime('%Y-%m-%d')],
                        'sku': [sku],
                        'forecasted_demand': [forecast_result['forecasted_demand'][-1]],
                        'model_used': [forecast_result['model_used']],
                        'confidence_interval_lower': [forecast_result['confidence_interval_lower'][-1] if 'confidence_interval_lower' in forecast_result else None],
                        'confidence_interval_upper': [forecast_result['confidence_interval_upper'][-1] if 'confidence_interval_upper' in forecast_result else None],
                        'created_at': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
                    })
                    
                    # 更新预测数据到数据仓库
                    self.replenishment_system.data_warehouse.update_data('forecast_output', new_forecast, update_type='append')
                    
                    logger.info(f"已更新sku {sku} 的预测结果")
        except Exception as e:
            logger.error(f"更新实时预测失败: {str(e)}")
