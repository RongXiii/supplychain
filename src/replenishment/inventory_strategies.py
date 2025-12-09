import numpy as np
import pandas as pd
from scipy import stats

class InventoryStrategies:
    """
    高级库存策略管理类，实现ABC分类、动态安全库存等策略
    """
    
    def __init__(self):
        self.abc_classes = {}  # 基本ABC分类
        self.abc_xyz_classes = {}  # ABC-XYZ交叉分类
        self.dynamic_safety_stock_params = {}
        self.min_order_rules = {}
        self.replenishment_strategies = {}  # 产品补货策略
    
    def abc_classification(self, products_data, classification_type='revenue', thresholds=[0.8, 0.95]):
        """
        根据产品的销售额或利润进行ABC分类
        
        Args:
            products_data: 产品数据，包含产品ID、销售额或利润等信息
            classification_type: 分类类型，'revenue'（销售额）或'profit'（利润）
            thresholds: A类和B类的累计百分比阈值，默认[0.8, 0.95]表示A类占80%，B类占15%，C类占5%
            
        Returns:
            abc_classes: 产品ABC分类结果，格式为{product_id: class}
        """
        # 验证输入数据
        if not isinstance(products_data, pd.DataFrame):
            raise ValueError("products_data必须是pandas DataFrame")
        
        required_columns = ['product_id', classification_type]
        if not all(col in products_data.columns for col in required_columns):
            raise ValueError(f"products_data必须包含列: {required_columns}")
        
        # 按销售额或利润降序排序
        sorted_products = products_data.sort_values(by=classification_type, ascending=False)
        
        # 计算累计百分比
        total_value = sorted_products[classification_type].sum()
        if total_value <= 0:
            raise ValueError("分类值总和必须大于0")
        
        sorted_products['cumulative_value'] = sorted_products[classification_type].cumsum()
        sorted_products['cumulative_percentage'] = sorted_products['cumulative_value'] / total_value
        
        # 分配ABC分类
        def assign_class(row):
            if row['cumulative_percentage'] <= thresholds[0]:
                return 'A'
            elif row['cumulative_percentage'] <= thresholds[1]:
                return 'B'
            else:
                return 'C'
        
        sorted_products['abc_class'] = sorted_products.apply(assign_class, axis=1)
        
        # 转换为字典格式
        self.abc_classes = dict(zip(sorted_products['product_id'], sorted_products['abc_class']))
        
        # 输出分类统计
        class_counts = sorted_products['abc_class'].value_counts()
        class_percentage = sorted_products.groupby('abc_class')[classification_type].sum() / total_value * 100
        
        print("ABC分类结果统计:")
        print(f"类别A: {class_counts.get('A', 0)}个产品，占{class_percentage.get('A', 0):.1f}%")
        print(f"类别B: {class_counts.get('B', 0)}个产品，占{class_percentage.get('B', 0):.1f}%")
        print(f"类别C: {class_counts.get('C', 0)}个产品，占{class_percentage.get('C', 0):.1f}%")
        
        return self.abc_classes
    
    def abc_xyz_classification(self, products_data, historical_demand, xyz_thresholds=[0.1, 0.2]):
        """
        ABC-XYZ交叉分类，结合产品价值（ABC）和需求稳定性（XYZ）
        
        Args:
            products_data: 产品数据，包含产品ID、销售额等信息
            historical_demand: 历史需求数据，格式为{product_id: [demand_series]}
            xyz_thresholds: X类和Y类的变异系数阈值，默认[0.1, 0.2]
            
        Returns:
            abc_xyz_classes: ABC-XYZ交叉分类结果，格式为{product_id: 'A-X', 'B-Y'等}
        """
        # 首先执行ABC分类
        self.abc_classification(products_data)
        
        # 计算XYZ分类（基于需求稳定性）
        xyz_classes = {}
        
        for product_id, demands in historical_demand.items():
            if len(demands) < 5:
                xyz_classes[product_id] = 'Y'  # 数据不足，默认Y类
                continue
                
            # 计算变异系数（标准差/均值）
            mean_demand = np.mean(demands)
            if mean_demand <= 0:
                xyz_classes[product_id] = 'Z'  # 需求为0，默认Z类
                continue
                
            std_demand = np.std(demands)
            cv = std_demand / mean_demand
            
            # 分配XYZ分类
            if cv <= xyz_thresholds[0]:
                xyz_class = 'X'  # 稳定需求
            elif cv <= xyz_thresholds[1]:
                xyz_class = 'Y'  # 中等稳定需求
            else:
                xyz_class = 'Z'  # 不稳定需求
                
            xyz_classes[product_id] = xyz_class
        
        # 生成ABC-XYZ交叉分类
        self.abc_xyz_classes = {}
        for product_id, abc_class in self.abc_classes.items():
            xyz_class = xyz_classes.get(product_id, 'Y')
            self.abc_xyz_classes[product_id] = f"{abc_class}-{xyz_class}"
        
        # 输出交叉分类统计
        from collections import Counter
        abc_xyz_counts = Counter(self.abc_xyz_classes.values())
        
        print("\nABC-XYZ交叉分类结果统计:")
        for key, count in sorted(abc_xyz_counts.items()):
            print(f"类别{key}: {count}个产品")
        
        return self.abc_xyz_classes
    
    def calculate_dynamic_safety_stock(self, product_id, historical_demand, lead_time, service_level=0.95, lead_time_std=0):
        """
        计算动态安全库存，考虑需求波动、提前期和ABC分类
        
        Args:
            product_id: 产品ID
            historical_demand: 历史需求数据（时间序列）
            lead_time: 交货提前期（天）
            service_level: 服务水平（默认95%）
            lead_time_std: 提前期标准差（天），默认0表示提前期稳定
            
        Returns:
            safety_stock: 动态安全库存量
            safety_stock_info: 安全库存计算信息
        """
        if not isinstance(historical_demand, (list, np.ndarray)):
            raise ValueError("historical_demand必须是列表或numpy数组")
        
        if len(historical_demand) < 10:
            raise ValueError("至少需要10个历史需求数据点")
        
        # 获取产品ABC分类
        abc_class = self.abc_classes.get(product_id, 'B')  # 默认B类
        
        # 计算需求统计量
        demand_mean = np.mean(historical_demand)
        demand_std = np.std(historical_demand)
        demand_cv = demand_std / demand_mean if demand_mean > 0 else 0  # 变异系数
        
        # 根据服务水平获取Z值
        z_value = {0.90: 1.28, 0.95: 1.65, 0.97: 1.88, 0.98: 2.05, 0.99: 2.33}.get(service_level, 1.65)
        
        # 根据ABC分类调整Z值（A类产品要求更高的服务水平）
        z_adjustment = {'A': 1.1, 'B': 1.0, 'C': 0.9}.get(abc_class, 1.0)
        adjusted_z = z_value * z_adjustment
        
        # 根据需求变异系数调整
        cv_adjustment = min(1 + (demand_cv - 0.5) * 0.5, 2.0)  # 变异系数越高，安全库存越高
        adjusted_z *= cv_adjustment
        
        # 计算提前期需求标准差（考虑提前期波动）
        if lead_time_std > 0:
            # 提前期不稳定时，使用更复杂的公式
            lead_time_demand_std = np.sqrt(
                (lead_time * demand_std ** 2) + 
                (lead_time_std ** 2 * demand_mean ** 2)
            )
        else:
            lead_time_demand_std = demand_std * np.sqrt(lead_time)
        
        # 计算安全库存
        safety_stock = adjusted_z * lead_time_demand_std
        
        # 添加安全库存参数到动态参数字典
        self.dynamic_safety_stock_params[product_id] = {
            'abc_class': abc_class,
            'z_value': adjusted_z,
            'demand_std': demand_std,
            'demand_cv': demand_cv,
            'lead_time': lead_time,
            'lead_time_std': lead_time_std,
            'service_level': service_level
        }
        
        safety_stock_info = {
            'product_id': product_id,
            'abc_class': abc_class,
            'demand_mean': demand_mean,
            'demand_std': demand_std,
            'demand_cv': demand_cv,
            'lead_time': lead_time,
            'lead_time_std': lead_time_std,
            'service_level': service_level,
            'z_value': adjusted_z,
            'safety_stock': safety_stock
        }
        
        return round(safety_stock, 2), safety_stock_info
    
    def set_replenishment_strategy(self, product_id, strategy_type, **params):
        """
        为产品设置补货策略
        
        Args:
            product_id: 产品ID
            strategy_type: 策略类型，如'ROP'（再订货点）、'EOQ'（经济订货量）、'Order-Up-To'（按需订货）
            params: 策略参数
        """
        self.replenishment_strategies[product_id] = {
            'strategy_type': strategy_type,
            'params': params
        }
    
    def get_replenishment_strategy(self, product_id):
        """
        获取产品的补货策略
        
        Args:
            product_id: 产品ID
            
        Returns:
            strategy: 补货策略配置
        """
        return self.replenishment_strategies.get(product_id, {
            'strategy_type': 'EOQ',
            'params': {'safety_stock': 0}
        })
    
    def set_min_order_rules(self, abc_min_order_ratios):
        """
        设置基于ABC分类的最小起订量规则
        
        Args:
            abc_min_order_ratios: 不同ABC分类的最小起订量系数，格式为{'A': ratio, 'B': ratio, 'C': ratio}
            例如: {'A': 0.1, 'B': 0.2, 'C': 0.3} 表示最小起订量为月需求量的10%/20%/30%
        """
        self.min_order_rules = abc_min_order_ratios
        
    def calculate_min_order_quantity(self, product_id, monthly_demand):
        """
        根据ABC分类和月需求量计算最小起订量
        
        Args:
            product_id: 产品ID
            monthly_demand: 月需求量
            
        Returns:
            min_order_qty: 最小起订量
        """
        if not self.min_order_rules:
            raise ValueError("请先调用set_min_order_rules设置最小起订量规则")
        
        # 获取产品ABC分类
        abc_class = self.abc_classes.get(product_id, 'B')  # 默认B类
        
        # 获取最小起订量系数
        ratio = self.min_order_rules.get(abc_class, 0.2)  # 默认20%
        
        # 计算最小起订量
        min_order_qty = monthly_demand * ratio
        
        return max(round(min_order_qty, 2), 1)  # 最小起订量至少为1
    
    def update_abc_classes(self, new_products_data, classification_type='revenue'):
        """
        更新ABC分类
        
        Args:
            new_products_data: 包含最新销售额或利润的产品数据
            classification_type: 分类类型
        """
        return self.abc_classification(new_products_data, classification_type)
    
    def get_abc_class(self, product_id):
        """
        获取产品的ABC分类
        
        Args:
            product_id: 产品ID
            
        Returns:
            abc_class: 产品的ABC分类
        """
        return self.abc_classes.get(product_id, 'B')  # 默认B类
    
    def get_safety_stock_params(self, product_id):
        """
        获取产品的安全库存参数
        
        Args:
            product_id: 产品ID
            
        Returns:
            params: 安全库存参数
        """
        return self.dynamic_safety_stock_params.get(product_id, {})
