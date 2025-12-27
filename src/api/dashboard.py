import pandas as pd
import numpy as np
import os
import logging
from .visualization.data_adapters import DataAdapter
from .visualization.chart_types import LineChart, BarChart, PieChart
from .visualization.dashboard_builder import DashboardBuilder

logger = logging.getLogger(__name__)

class DataDashboard:
    """旧的数据仪表盘类，用于向后兼容"""
    def __init__(self, data_dir='data', metrics_dir='metrics'):
        """
        初始化数据仪表盘
        
        Args:
            data_dir: 数据目录路径
            metrics_dir: 指标目录路径
        """
        self.data_dir = data_dir
        self.metrics_dir = metrics_dir
        self.dashboard_service = DashboardService(data_dir, metrics_dir)
    
    def load_data(self, filename):
        """
        加载数据文件
        
        Args:
            filename: 文件名
            
        Returns:
            data: DataFrame
        """
        return self.dashboard_service.load_data(filename)
    
    def load_metrics(self, product_id):
        """
        加载指定产品的性能指标
        
        Args:
            product_id: 产品ID
            
        Returns:
            metrics: 指标列表
        """
        return self.dashboard_service.load_metrics(product_id)
    
    def get_demand_forecast_data(self, product_id, actual_data, forecast_data, title=None):
        """
        获取需求预测结果数据（JSON格式）
        
        Args:
            product_id: 产品ID
            actual_data: 实际需求数据
            forecast_data: 预测需求数据
            title: 图表标题
        
        Returns:
            data: JSON格式的需求预测数据
        """
        return self.dashboard_service.get_demand_forecast_chart(product_id, actual_data, forecast_data, title).to_json()
    
    def get_model_performance_data(self, product_id, metrics):
        """
        获取模型性能指标数据（JSON格式）
        
        Args:
            product_id: 产品ID
            metrics: 模型性能指标列表
        
        Returns:
            data: JSON格式的模型性能数据
        """
        chart = self.dashboard_service.get_model_performance_chart(product_id, metrics)
        return chart.to_json() if chart else None
    
    def get_inventory_levels_data(self, inventory_data, product_id=None):
        """
        获取库存水平数据（JSON格式）
        
        Args:
            inventory_data: 库存数据
            product_id: 产品ID（可选，默认为None，显示所有产品）
        
        Returns:
            data: JSON格式的库存水平数据
        """
        return self.dashboard_service.get_inventory_levels_chart(inventory_data, product_id).to_json()
    
    def get_purchase_orders_data(self, order_data):
        """
        获取采购订单情况数据（JSON格式）
        
        Args:
            order_data: 采购订单数据
        
        Returns:
            data: JSON格式的采购订单数据
        """
        return self.dashboard_service.get_purchase_orders_charts(order_data)
    
    def get_safety_stock_data(self, product_ids, safety_stocks):
        """
        获取安全库存数据（JSON格式）
        
        Args:
            product_ids: 产品ID列表
            safety_stocks: 安全库存列表
        
        Returns:
            data: JSON格式的安全库存数据
        """
        return self.dashboard_service.get_safety_stock_chart(product_ids, safety_stocks).to_json()
    
    def _generate_color_palette(self, count):
        """
        生成颜色调色板
        
        Args:
            count: 需要的颜色数量
            
        Returns:
            colors: 颜色列表
        """
        return DashboardService._generate_color_palette(count)
            
    def get_all_data(self, product_id):
        """
        获取所有可视化数据（用于仪表盘）
        
        Args:
            product_id: 产品ID
            
        Returns:
            data: 包含所有可视化数据的JSON对象
        """
        return self.dashboard_service.build_dashboard(product_id)


class DashboardService:
    """新的仪表盘服务类，使用重构后的可视化模块"""
    def __init__(self, data_dir='data', metrics_dir='metrics'):
        """
        初始化仪表盘服务
        
        Args:
            data_dir: 数据目录路径
            metrics_dir: 指标目录路径
        """
        self.data_dir = data_dir
        self.metrics_dir = metrics_dir
    
    def load_data(self, filename):
        """
        加载数据文件
        
        Args:
            filename: 文件名
            
        Returns:
            data: DataFrame
        """
        file_path = os.path.join(self.data_dir, filename)
        if not os.path.exists(file_path):
            logger.warning(f"文件不存在: {file_path}")
            return None
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            logger.error(f"加载数据文件失败: {str(e)}")
            return None
    
    def load_metrics(self, product_id):
        """
        加载指定产品的性能指标
        
        Args:
            product_id: 产品ID
            
        Returns:
            metrics: 指标列表
        """
        file_path = os.path.join(self.metrics_dir, f'metrics_{product_id}.json')
        if not os.path.exists(file_path):
            logger.warning(f"指标文件不存在: {file_path}")
            return None
        try:
            import json
            with open(file_path, 'r') as f:
                metrics = json.load(f)
            return metrics
        except Exception as e:
            logger.error(f"加载指标文件失败: {str(e)}")
            return None
    
    def get_demand_forecast_chart(self, product_id, actual_data, forecast_data, title=None):
        """
        获取需求预测结果图表
        
        Args:
            product_id: 产品ID
            actual_data: 实际需求数据
            forecast_data: 预测需求数据
            title: 图表标题
        
        Returns:
            chart: LineChart实例
        """
        # 确保索引是字符串格式（JSON序列化需要）
        if hasattr(actual_data.index, 'strftime'):
            dates = actual_data.index.strftime('%Y-%m-%d').tolist()
        else:
            dates = actual_data.index.astype(str).tolist()
        
        # 准备数据集
        datasets = [
            {
                'label': '实际需求',
                'data': actual_data.values.tolist(),
                'color': 'blue'
            },
            {
                'label': '预测需求',
                'data': forecast_data.values.tolist(),
                'color': 'red'
            }
        ]
        
        # 创建折线图
        return LineChart(
            title=title if title else f'产品 {product_id} 需求预测对比',
            data={
                'labels': dates,
                'datasets': datasets
            },
            x_axis_label='时间',
            y_axis_label='需求量'
        )
    
    def get_model_performance_chart(self, product_id, metrics):
        """
        获取模型性能指标图表
        
        Args:
            product_id: 产品ID
            metrics: 模型性能指标列表
        
        Returns:
            chart: LineChart实例
        """
        if not metrics:
            return None
        
        # 转换为DataFrame
        df = pd.DataFrame(metrics)
        
        # 提取时间戳作为索引
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.sort_values('timestamp', inplace=True)
        
        # 准备数据
        dates = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist()
        
        # 准备数据集
        datasets = [
            {
                'label': 'MAPE',
                'data': df['mape'].tolist(),
                'color': 'red',
                'unit': '%'
            },
            {
                'label': 'SMAPE',
                'data': df['smape'].tolist(),
                'color': 'green',
                'unit': '%'
            },
            {
                'label': 'RMSE',
                'data': df['rmse'].tolist(),
                'color': 'blue',
                'unit': ''
            }
        ]
        
        # 创建折线图
        return LineChart(
            title=f'产品 {product_id} 模型性能指标',
            data={
                'labels': dates,
                'datasets': datasets
            },
            x_axis_label='时间'
        )
    
    def get_inventory_levels_chart(self, inventory_data, product_id=None):
        """
        获取库存水平图表
        
        Args:
            inventory_data: 库存数据
            product_id: 产品ID（可选，默认为None，显示所有产品）
        
        Returns:
            chart: LineChart实例
        """
        if inventory_data is None or inventory_data.empty:
            return LineChart(
                title='库存水平变化',
                data={'labels': [], 'datasets': []},
                x_axis_label='日期',
                y_axis_label='库存水平'
            )
        
        # 处理日期格式
        inventory_data['date'] = pd.to_datetime(inventory_data['date'])
        inventory_data.sort_values('date', inplace=True)
        
        dates = inventory_data['date'].dt.strftime('%Y-%m-%d').unique().tolist()
        
        if product_id:
            # 只显示指定产品
            product_data = inventory_data[inventory_data['item_id'] == product_id]
            # 使用一致的列名 'quantity_on_hand'
            datasets = [
                {
                    'label': f'产品 {product_id} 库存水平',
                    'data': product_data['quantity_on_hand'].tolist(),
                    'color': 'blue'
                }
            ]
            title = f'产品 {product_id} 库存水平变化'
        else:
            # 显示所有产品
            datasets = []
            colors = self._generate_color_palette(len(inventory_data['item_id'].unique()))
            
            for i, pid in enumerate(inventory_data['item_id'].unique()):
                product_data = inventory_data[inventory_data['item_id'] == pid]
                # 确保数据与日期对齐
                product_dates = product_data['date'].dt.strftime('%Y-%m-%d').tolist()
                product_values = {date: 0 for date in dates}
                
                # 使用一致的列名 'quantity_on_hand'
                for date, value in zip(product_dates, product_data['quantity_on_hand'].tolist()):
                    product_values[date] = value
                
                datasets.append({
                    'label': f'产品 {pid}',
                    'data': [product_values[date] for date in dates],
                    'color': colors[i % len(colors)]
                })
            
            title = '所有产品库存水平变化'
        
        # 创建折线图
        return LineChart(
            title=title,
            data={
                'labels': dates,
                'datasets': datasets
            },
            x_axis_label='日期',
            y_axis_label='库存水平'
        )
    
    def get_purchase_orders_charts(self, order_data):
        """
        获取采购订单情况图表
        
        Args:
            order_data: 采购订单数据
        
        Returns:
            charts: 包含采购订单图表的字典
        """
        if order_data is None or order_data.empty:
            return {
                'by_product': LineChart(title='各产品采购数量', data={'labels': [], 'datasets': []}).to_json(),
                'status_distribution': PieChart(title='订单状态分布', data={'labels': [], 'datasets': []}).to_json()
            }
        
        # 按产品分组的订单数量
        order_by_product = order_data.groupby('item_id')['order_qty'].sum().reset_index()
        
        # 创建产品采购数量条形图
        product_chart = BarChart(
            title='各产品采购数量',
            data={
                'labels': order_by_product['item_id'].astype(str).tolist(),
                'datasets': [
                    {
                        'label': '采购数量',
                        'data': order_by_product['order_qty'].tolist(),
                        'color': 'skyblue'
                    }
                ]
            },
            x_axis_label='产品ID',
            y_axis_label='采购数量'
        )
        
        # 订单状态分布
        order_status = order_data['status'].value_counts().reset_index()
        order_status.columns = ['status', 'count']
        
        # 创建订单状态分布饼图
        status_chart = PieChart(
            title='订单状态分布',
            data={
                'labels': order_status['status'].tolist(),
                'datasets': [
                    {
                        'label': '订单数量',
                        'data': order_status['count'].tolist(),
                        'colors': ['lightgreen', 'lightcoral', 'lightblue']
                    }
                ]
            }
        )
        
        return {
            'by_product': product_chart.to_json(),
            'status_distribution': status_chart.to_json()
        }
    
    def get_safety_stock_chart(self, product_ids, safety_stocks):
        """
        获取安全库存图表
        
        Args:
            product_ids: 产品ID列表
            safety_stocks: 安全库存列表
        
        Returns:
            chart: BarChart实例
        """
        # 创建安全库存条形图
        return BarChart(
            title='各产品安全库存',
            data={
                'labels': [str(pid) for pid in product_ids],
                'datasets': [
                    {
                        'label': '安全库存',
                        'data': safety_stocks,
                        'color': 'orange'
                    }
                ]
            },
            x_axis_label='产品ID',
            y_axis_label='安全库存'
        )
    
    @staticmethod
    def _generate_color_palette(count):
        """
        生成颜色调色板
        
        Args:
            count: 需要的颜色数量
            
        Returns:
            colors: 颜色列表
        """
        colors = [
            '#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6',
            '#1abc9c', '#e67e22', '#34495e', '#95a5a6', '#d35400'
        ]
        
        # 如果需要的颜色数量超过预设，循环使用
        if count <= len(colors):
            return colors[:count]
        else:
            return [colors[i % len(colors)] for i in range(count)]
            
    def build_dashboard(self, product_id):
        """
        构建仪表盘
        
        Args:
            product_id: 产品ID
            
        Returns:
            data: 包含所有可视化数据的JSON对象
        """
        # 加载各种数据
        demand_data = self.load_data('demand_data.csv')
        inventory_data = self.load_data('inventory_data.csv')
        order_data = self.load_data('purchase_orders.csv')
        metrics = self.load_metrics(product_id)
        
        # 创建仪表盘构建器
        dashboard_builder = DashboardBuilder()
        dashboard_builder.set_title(f'产品 {product_id} 供应链仪表盘')
        dashboard_builder.set_description(f'产品 {product_id} 的需求预测、库存水平和订单情况')
        dashboard_builder.set_theme('light')
        
        # 添加需求预测图表
        if demand_data is not None:
            # 假设数据已经按产品和时间排序
            product_demand = demand_data[demand_data['item_id'] == product_id]
            if not product_demand.empty:
                # 分离实际需求和预测需求
                actual_data = product_demand['actual_demand']
                forecast_data = product_demand['forecast_demand']
                chart = self.get_demand_forecast_chart(product_id, actual_data, forecast_data)
                dashboard_builder.add_chart(chart, {'row': 1, 'col': 1, 'span': 2})
        
        # 添加模型性能图表
        if metrics is not None:
            chart = self.get_model_performance_chart(product_id, metrics)
            if chart:
                dashboard_builder.add_chart(chart, {'row': 2, 'col': 1, 'span': 2})
        
        # 添加库存水平图表
        if inventory_data is not None:
            chart = self.get_inventory_levels_chart(inventory_data, product_id)
            dashboard_builder.add_chart(chart, {'row': 3, 'col': 1, 'span': 2})
        
        # 添加采购订单图表
        if order_data is not None:
            # 由于采购订单包含两个图表，我们需要分别处理
            product_chart = None
            status_chart = None
            
            try:
                # 按产品分组的订单数量
                order_by_product = order_data.groupby('item_id')['order_qty'].sum().reset_index()
                
                # 创建产品采购数量条形图
                product_chart = BarChart(
                    title='各产品采购数量',
                    data={
                        'labels': order_by_product['item_id'].astype(str).tolist(),
                        'datasets': [
                            {
                                'label': '采购数量',
                                'data': order_by_product['order_qty'].tolist(),
                                'color': 'skyblue'
                            }
                        ]
                    },
                    x_axis_label='产品ID',
                    y_axis_label='采购数量'
                )
                
                # 订单状态分布
                order_status = order_data['status'].value_counts().reset_index()
                order_status.columns = ['status', 'count']
                
                # 创建订单状态分布饼图
                status_chart = PieChart(
                    title='订单状态分布',
                    data={
                        'labels': order_status['status'].tolist(),
                        'datasets': [
                            {
                                'label': '订单数量',
                                'data': order_status['count'].tolist(),
                                'colors': ['lightgreen', 'lightcoral', 'lightblue']
                            }
                        ]
                    }
                )
            except Exception as e:
                logger.error(f"创建采购订单图表失败: {str(e)}")
            
            # 添加图表到仪表盘
            if product_chart:
                dashboard_builder.add_chart(product_chart, {'row': 4, 'col': 1, 'span': 1})
            if status_chart:
                dashboard_builder.add_chart(status_chart, {'row': 4, 'col': 2, 'span': 1})
        
        # 模拟安全库存数据并添加图表
        if inventory_data is not None:
            all_product_ids = inventory_data['item_id'].unique().tolist()
            safety_stocks = [np.random.randint(50, 200) for _ in all_product_ids]
            chart = self.get_safety_stock_chart(all_product_ids, safety_stocks)
            dashboard_builder.add_chart(chart, {'row': 5, 'col': 1, 'span': 2})
        
        # 构建仪表盘
        dashboard = dashboard_builder.build()
        
        # 添加额外信息
        dashboard['product_id'] = product_id
        dashboard['timestamp'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        
        return dashboard