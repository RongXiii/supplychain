import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta


def generate_simulated_data():
    """
    生成符合供应链系统需求的模拟数据表
    
    Returns:
        dict: 包含所有模拟数据表的字典
    """
    np.random.seed(42)
    
    # 生成基础数据
    items = generate_items()
    locations = generate_locations()
    suppliers = generate_suppliers()
    inventory_daily = generate_inventory_daily(items, locations)
    purchase_orders = generate_purchase_orders(items, locations, suppliers)
    forecast_output = generate_forecast_output(items, locations)
    optimal_plan = generate_optimal_plan(items, locations, suppliers)
    
    return {
        'items': items,
        'locations': locations,
        'suppliers': suppliers,
        'inventory_daily': inventory_daily,
        'purchase_orders': purchase_orders,
        'forecast_output': forecast_output,
        'optimal_plan': optimal_plan
    }


def generate_items():
    """
    生成商品信息表
    """
    # 生成500个商品
    num_items = 500
    item_ids = list(range(1, num_items + 1))
    
    # 分配ABC分类
    abc_classes = []
    for i in range(num_items):
        if i < 100:  # 20% A类
            abc_classes.append('A')
        elif i < 250:  # 30% B类
            abc_classes.append('B')
        else:  # 50% C类
            abc_classes.append('C')
    
    # 新增：品类和品牌信息
    categories = ['electronics', 'clothing', 'home', 'sports', 'food', 'toys', 'books', 'beauty', 'automotive', 'health']
    brands = [f'Brand_{i}' for i in range(1, 21)]  # 20个品牌
    
    # 生成其他属性
    min_order_qty = [np.random.randint(20, 200) for _ in range(num_items)]
    pack_size = [np.random.choice([5, 10, 15, 20, 25, 30, 50, 100]) for _ in range(num_items)]
    safety_factor_z = [np.random.choice([0.84, 1.04, 1.28, 1.64, 1.96]) for _ in range(num_items)]
    uom = np.random.choice(['EA', 'BOX', 'CASE', 'PK', 'SET'], size=num_items)  # 多种单位
    unit_price = [round(np.random.uniform(1.0, 100.0), 2) for _ in range(num_items)]
    category = np.random.choice(categories, size=num_items)
    brand = np.random.choice(brands, size=num_items)
    
    items_data = {
        'item_id': item_ids,
        'abc_class': abc_classes,
        'min_order_qty': min_order_qty,
        'pack_size': pack_size,
        'safety_factor_z': safety_factor_z,
        'uom': uom,
        'unit_price': unit_price,
        'category': category,
        'brand': brand
    }
    return pd.DataFrame(items_data)


def generate_locations():
    """
    生成仓库/位置表
    """
    # 生成30个仓库
    num_locations = 30
    location_ids = [f'WH{i}' for i in range(1, num_locations + 1)]
    capacity_limit = [np.random.randint(1000, 30000) for _ in range(num_locations)]
    transfer_cost_per_unit = [round(np.random.uniform(0.3, 8.0), 2) for _ in range(num_locations)]
    
    # 新增：区域信息
    regions = ['North', 'South', 'East', 'West', 'Central']
    region = np.random.choice(regions, size=num_locations)
    
    locations_data = {
        'location_id': location_ids,
        'capacity_limit': capacity_limit,
        'transfer_cost_per_unit': transfer_cost_per_unit,
        'region': region
    }
    return pd.DataFrame(locations_data)


def generate_suppliers():
    """
    生成供应商信息表
    """
    # 生成25个供应商
    num_suppliers = 25
    supplier_ids = list(range(1, num_suppliers + 1))
    lead_time_days = [np.random.randint(2, 20) for _ in range(num_suppliers)]
    
    # 新增：供应商评级和类型
    supplier_ratings = [np.random.choice(['Excellent', 'Good', 'Average', 'Poor']) for _ in range(num_suppliers)]
    supplier_types = ['domestic', 'international', 'local']
    
    # 生成价格断点
    price_breaks_list = []
    for _ in range(num_suppliers):
        # 生成3-4个价格等级
        num_tiers = np.random.choice([3, 4])
        tiers = []
        current_min = 0
        
        for i in range(num_tiers):
            if i < num_tiers - 1:
                tier_max = current_min + np.random.randint(100, 800)
                unit_price = round(np.random.uniform(5.0, 80.0), 2) if i == 0 else round(tiers[-1][2] * np.random.uniform(0.8, 0.95), 2)
                tiers.append([current_min, tier_max, unit_price])
                current_min = tier_max + 1
            else:
                # 最后一个等级，无上限
                unit_price = round(tiers[-1][2] * np.random.uniform(0.8, 0.95), 2)
                tiers.append([current_min, None, unit_price])
        
        price_breaks = json.dumps(tiers)
        price_breaks_list.append(price_breaks)
    
    suppliers_data = {
        'supplier_id': supplier_ids,
        'lead_time_days': lead_time_days,
        'price_breaks': price_breaks_list,
        'rating': supplier_ratings,
        'type': np.random.choice(supplier_types, size=num_suppliers)
    }
    return pd.DataFrame(suppliers_data)


def generate_inventory_daily(items, locations):
    """
    生成每日库存数据表
    """
    # 生成过去180天的数据
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    inventory_data = []
    for date in dates:
        for _, item in items.iterrows():
            for _, location in locations.iterrows():
                # 生成随机库存数据，考虑商品属性和季节性
                base_stock = 50 + (item['item_id'] * 1.5)  # 基础库存随商品ID增加
                
                # 模拟季节性波动
                month = date.month
                seasonal_factor = 1.0
                if month in [11, 12, 1]:  # 节假日季节需求增加
                    seasonal_factor = 1.5
                elif month in [3, 4, 5]:  # 春季需求
                    seasonal_factor = 1.2
                
                on_hand_qty = np.random.randint(int(base_stock * 0.3), int(base_stock * 2.5))
                demand_qty = np.random.randint(0, int(base_stock * 0.4 * seasonal_factor))  # 需求受季节性影响
                receipts_qty = np.random.randint(0, int(base_stock * 0.6))  # 收货为基础库存的0-60%
                
                inventory_data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'item_id': item['item_id'],
                    'location_id': location['location_id'],
                    'on_hand_qty': on_hand_qty,
                    'demand_qty': demand_qty,
                    'receipts_qty': receipts_qty,
                    'seasonal_factor': seasonal_factor
                })
    
    return pd.DataFrame(inventory_data)


def generate_purchase_orders(items, locations, suppliers):
    """
    生成采购订单表
    """
    purchase_orders_data = []
    po_id = 1
    
    # 生成过去365天的采购订单
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    # 新增：更多订单状态
    statuses = ['pending', 'approved', 'shipped', 'received', 'cancelled', 'partially_received', 'on_hold']
    status_probs = [0.08, 0.12, 0.15, 0.5, 0.05, 0.07, 0.03]  # received概率最高
    
    for _, item in items.iterrows():
        for _, location in locations.iterrows():
            # 每个商品每个位置生成20个采购订单
            for _ in range(20):
                supplier = suppliers.sample(1).iloc[0]
                order_date = start_date + timedelta(days=np.random.randint(0, 365))
                due_date = order_date + timedelta(days=int(supplier['lead_time_days']))
                
                # 生成订单数量，考虑最小订单量和包装规格
                order_qty = np.random.randint(item['min_order_qty'], item['min_order_qty'] * 6)
                order_qty = ((order_qty + item['pack_size'] - 1) // item['pack_size']) * item['pack_size']
                
                # 随机状态
                status = np.random.choice(statuses, p=status_probs)
                
                # 新增：订单优先级
                priority = np.random.choice(['low', 'medium', 'high'], p=[0.6, 0.3, 0.1])
                
                purchase_orders_data.append({
                    'po_id': po_id,
                    'item_id': item['item_id'],
                    'supplier_id': supplier['supplier_id'],
                    'order_date': order_date.strftime('%Y-%m-%d'),
                    'due_date': due_date.strftime('%Y-%m-%d'),
                    'order_qty': order_qty,
                    'status': status,
                    'location_id': location['location_id'],
                    'unit_price': item['unit_price'],
                    'priority': priority,
                    'created_by': f'user_{np.random.randint(1, 51)}'  # 模拟50个用户
                })
                po_id += 1
    
    return pd.DataFrame(purchase_orders_data)


def generate_forecast_output(items, locations):
    """
    生成预测输出表
    """
    forecast_data = []
    
    # 生成未来90天的预测
    start_date = datetime.now()
    end_date = start_date + timedelta(days=90)
    horizon_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # 新增：更多预测模型
    models = ['holt_winters', 'arima', 'sarimax', 'prophet', 'xgboost', 'lightgbm', 'catboost', 'random_forest', 'linear_regression']
    
    for _, item in items.iterrows():
        for _, location in locations.iterrows():
            for horizon_date in horizon_dates:
                # 生成预测值，考虑商品属性和季节性
                base_demand = 15 + (item['item_id'] * 2.5)  # 基础需求随商品ID增加
                
                # 模拟季节性影响
                month = horizon_date.month
                seasonal_factor = 1.0
                if month in [11, 12, 1]:  # 节假日季节
                    seasonal_factor = 1.6
                elif month in [3, 4, 5]:  # 春季
                    seasonal_factor = 1.3
                
                yhat = base_demand * seasonal_factor + np.random.normal(0, base_demand * 0.25)  # 波动性随基础需求增加
                yhat = max(0, yhat)
                
                # 随机选择模型
                model_used = np.random.choice(models)
                
                # 生成MAPE和SMAPE，考虑商品ABC分类和模型类型
                base_mape = {
                    'A': (2.0, 8.0),
                    'B': (6.0, 13.0),
                    'C': (10.0, 22.0)
                }[item['abc_class']]
                
                # 不同模型有不同的预测精度
                model_precision = {
                    'xgboost': 0.85,
                    'lightgbm': 0.88,
                    'catboost': 0.86,
                    'prophet': 0.9,
                    'sarimax': 0.92,
                    'arima': 0.93,
                    'holt_winters': 0.95,
                    'random_forest': 0.9,
                    'linear_regression': 0.98
                }[model_used]
                
                mape_recent = round(np.random.uniform(base_mape[0] * model_precision, base_mape[1] * model_precision), 2)
                smape_recent = round(np.random.uniform(base_mape[0] * 0.9 * model_precision, base_mape[1] * 0.9 * model_precision), 2)
                
                # 新增：置信区间
                confidence_level = 0.95
                lower_bound = round(yhat * (1 - (mape_recent / 100)), 2)
                upper_bound = round(yhat * (1 + (mape_recent / 100)), 2)
                
                forecast_data.append({
                    'item_id': item['item_id'],
                    'location_id': location['location_id'],
                    'horizon_date': horizon_date.strftime('%Y-%m-%d'),
                    'yhat': round(yhat, 2),
                    'model_used': model_used,
                    'mape_recent': round(mape_recent, 2),
                    'smape_recent': round(smape_recent, 2),
                    'confidence_level': confidence_level,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'seasonal_factor': seasonal_factor
                })
    
    return pd.DataFrame(forecast_data)


def generate_optimal_plan(items, locations, suppliers):
    """
    生成MILP优化输出表
    """
    optimal_plan_data = []
    
    # 为每个商品每个仓库生成多个优化计划（过去15天和未来15天）
    dates = pd.date_range(start=datetime.now() - timedelta(days=15), end=datetime.now() + timedelta(days=15), freq='D')
    
    for date in dates:
        for _, item in items.iterrows():
            for _, location in locations.iterrows():
                # 生成优化计划
                supplier = suppliers.sample(1).iloc[0]
                
                # 生成订单数量
                order_qty = np.random.randint(item['min_order_qty'], item['min_order_qty'] * 5)
                order_qty = ((order_qty + item['pack_size'] - 1) // item['pack_size']) * item['pack_size']
                
                # 确定价格等级
                price_breaks = json.loads(supplier['price_breaks'])
                tier_chosen = 0
                unit_price = item['unit_price']
                for i, (qty_from, qty_to, pb_unit_price) in enumerate(price_breaks):
                    if qty_to is None or (order_qty >= qty_from and order_qty <= qty_to):
                        tier_chosen = i + 1
                        unit_price = pb_unit_price
                        break
                
                # 生成调拨信息
                transfer_from = np.random.choice([loc for loc in locations['location_id'] if loc != location['location_id']])
                transfer_qty = np.random.randint(0, 400) if np.random.random() > 0.4 else 0  # 60%概率有调拨
                
                # 生成due_date
                due_date = date + timedelta(days=int(supplier['lead_time_days']))
                
                # 新增：优化目标
                optimization_goals = ['minimize_cost', 'minimize_stockout', 'balance_inventory', 'maximize_service_level']
                goal = np.random.choice(optimization_goals)
                
                optimal_plan_data.append({
                    'item_id': item['item_id'],
                    'location_id': location['location_id'],
                    'supplier_id': supplier['supplier_id'],
                    'order_qty': order_qty,
                    'unit_price': unit_price,
                    'tier_chosen': tier_chosen,
                    'due_date': due_date.strftime('%Y-%m-%d'),
                    'transfer_from': transfer_from,
                    'transfer_qty': transfer_qty,
                    'rationale': f'Optimized order for {item["abc_class"]} class item based on forecast demand and inventory levels',
                    'plan_date': date.strftime('%Y-%m-%d'),  # 添加计划日期
                    'optimization_goal': goal,
                    'confidence_score': round(np.random.uniform(0.7, 0.99), 2),  # 优化方案置信度
                    'estimated_savings': round(np.random.uniform(0, order_qty * unit_price * 0.2), 2)  # 预计节省成本
                })
    
    return pd.DataFrame(optimal_plan_data)


if __name__ == "__main__":
    # 生成并保存模拟数据
    data = generate_simulated_data()
    
    # 打印数据概览
    for table_name, df in data.items():
        print(f"\n{table_name}:")
        print(df.head())
        print(f"Shape: {df.shape}")
    
    # 保存到CSV文件
    for table_name, df in data.items():
        df.to_csv(f'./data/{table_name}.csv', index=False)
    print("\n模拟数据已保存到data目录")
