# 设置matplotlib使用非交互式后端，避免弹出窗口
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

class DataDashboard:
    def __init__(self, data_dir='data', metrics_dir='metrics'):
        """
        初始化数据仪表盘
        
        Args:
            data_dir: 数据目录路径
            metrics_dir: 指标目录路径
        """
        self.data_dir = data_dir
        self.metrics_dir = metrics_dir
        
        # 设置中文显示
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
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
            print(f"文件不存在: {file_path}")
            return None
        return pd.read_csv(file_path)
    
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
            print(f"指标文件不存在: {file_path}")
            return None
        import json
        with open(file_path, 'r') as f:
            metrics = json.load(f)
        return metrics
    
    def visualize_demand_forecast(self, product_id, actual_data, forecast_data, title=None):
        """
        可视化需求预测结果
        
        Args:
            product_id: 产品ID
            actual_data: 实际需求数据
            forecast_data: 预测需求数据
            title: 图表标题
        """
        plt.figure(figsize=(12, 6))
        
        # 绘制实际需求
        plt.plot(actual_data.index, actual_data.values, label='实际需求', color='blue', marker='o')
        
        # 绘制预测需求
        plt.plot(forecast_data.index, forecast_data.values, label='预测需求', color='red', marker='x')
        
        # 设置图表标题和标签
        if title:
            plt.title(title)
        else:
            plt.title(f'产品 {product_id} 需求预测对比')
        plt.xlabel('时间')
        plt.ylabel('需求量')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
    
    def visualize_model_performance(self, product_id, metrics):
        """
        可视化模型性能指标
        
        Args:
            product_id: 产品ID
            metrics: 模型性能指标列表
        """
        if not metrics:
            return
        
        # 转换为DataFrame
        df = pd.DataFrame(metrics)
        
        # 提取时间戳作为索引
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # 创建子图
        fig, axes = plt.subplots(3, 1, figsize=(12, 12))
        fig.suptitle(f'产品 {product_id} 模型性能指标', fontsize=16)
        
        # 绘制MAPE
        axes[0].plot(df.index, df['mape'], label='MAPE', color='red')
        axes[0].set_ylabel('MAPE (%)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 绘制SMAPE
        axes[1].plot(df.index, df['smape'], label='SMAPE', color='green')
        axes[1].set_ylabel('SMAPE (%)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 绘制RMSE
        axes[2].plot(df.index, df['rmse'], label='RMSE', color='blue')
        axes[2].set_ylabel('RMSE')
        axes[2].set_xlabel('时间')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
    
    def visualize_inventory_levels(self, inventory_data, product_id=None):
        """
        可视化库存水平
        
        Args:
            inventory_data: 库存数据
            product_id: 产品ID（可选，默认为None，显示所有产品）
        """
        plt.figure(figsize=(12, 6))
        
        if product_id:
            # 只显示指定产品
            product_data = inventory_data[inventory_data['item_id'] == product_id]
            plt.plot(product_data['date'], product_data['on_hand_qty'], 
                    label=f'产品 {product_id} 库存水平', color='blue', marker='o')
            plt.title(f'产品 {product_id} 库存水平变化')
        else:
            # 显示所有产品
            for pid in inventory_data['item_id'].unique():
                product_data = inventory_data[inventory_data['item_id'] == pid]
                plt.plot(product_data['date'], product_data['on_hand_qty'], 
                        label=f'产品 {pid}', marker='o')
            plt.title('所有产品库存水平变化')
        
        plt.xlabel('日期')
        plt.ylabel('库存水平')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
    
    def visualize_purchase_orders(self, order_data):
        """
        可视化采购订单情况
        
        Args:
            order_data: 采购订单数据
        """
        # 创建子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('采购订单分析', fontsize=16)
        
        # 按产品分组的订单数量
        order_by_product = order_data.groupby('item_id')['order_qty'].sum()
        ax1.bar(order_by_product.index.astype(str), order_by_product.values, color='skyblue')
        ax1.set_title('各产品采购数量')
        ax1.set_xlabel('产品ID')
        ax1.set_ylabel('采购数量')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 订单状态分布
        order_status = order_data['status'].value_counts()
        ax2.pie(order_status.values, labels=order_status.index, autopct='%1.1f%%', startangle=90, 
                colors=['lightgreen', 'lightcoral', 'lightblue'])
        ax2.set_title('订单状态分布')
        
        plt.tight_layout()
    
    def visualize_safety_stock(self, product_ids, safety_stocks):
        """
        可视化安全库存
        
        Args:
            product_ids: 产品ID列表
            safety_stocks: 安全库存列表
        """
        plt.figure(figsize=(12, 6))
        
        plt.bar([str(pid) for pid in product_ids], safety_stocks, color='orange')
        plt.title('各产品安全库存')
        plt.xlabel('产品ID')
        plt.ylabel('安全库存')
        plt.grid(True, alpha=0.3, axis='y')
        
        # 在条形图上添加数值标签
        for i, v in enumerate(safety_stocks):
            plt.text(i, v + 0.5, f'{v:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
    
    def show(self):
        """
        显示所有图表
        """
        plt.show()
    
    def save_figures(self, output_dir='figures'):
        """
        保存所有图表到文件
        
        Args:
            output_dir: 输出目录
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 保存当前所有打开的图表
        for i in range(1, plt.gcf().number + 1):
            plt.figure(i)
            plt.savefig(os.path.join(output_dir, f'figure_{i}.png'), dpi=300, bbox_inches='tight')
        print(f"图表已保存到目录: {output_dir}")