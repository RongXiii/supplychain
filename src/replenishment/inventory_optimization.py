import numpy as np
import pandas as pd
import math

class InventoryOptimization:
    """
    库存优化模块，负责处理库存策略、EOQ计算、多仓优化等功能
    """
    
    def __init__(self):
        pass
    
    def calculate_eoq(self, demand, ordering_cost, holding_cost):
        """
        计算经济订货量(EOQ)
        
        Args:
            demand: 年需求量
            ordering_cost: 每次订货成本
            holding_cost: 单位持有成本
            
        Returns:
            eoq: 经济订货量
        """
        if holding_cost <= 0:
            return demand  # 避免除以零
        
        eoq = math.sqrt((2 * demand * ordering_cost) / holding_cost)
        return eoq
    
    def find_optimal_order_qty_with_discount(self, eoq, annual_demand, ordering_cost, 
                                           holding_cost, unit_cost, discount_tiers):
        """
        考虑数量折扣的最优订货量计算
        
        Args:
            eoq: 经济订货量
            annual_demand: 年需求量
            ordering_cost: 每次订货成本
            holding_cost: 单位持有成本（占单位成本的百分比）
            unit_cost: 单位产品成本
            discount_tiers: 折扣阶梯，格式为[(min_qty1, discount_rate1), (min_qty2, discount_rate2), ...]
            
        Returns:
            optimal_qty: 最优订货量
            optimal_cost: 最优总成本
        """
        # 按最小数量排序折扣阶梯
        discount_tiers = sorted(discount_tiers, key=lambda x: x[0])
        
        best_qty = eoq
        best_total_cost = float('inf')
        
        # 检查每个折扣阶梯
        for i, (min_qty, discount_rate) in enumerate(discount_tiers):
            # 计算该折扣下的实际单位成本
            discounted_cost = unit_cost * (1 - discount_rate)
            
            # 计算该折扣下的持有成本（基于平均库存）
            tier_holding_cost = discounted_cost * holding_cost
            
            # 如果EOQ在当前折扣阶梯内
            if eoq >= min_qty:
                qty_candidate = eoq
            else:
                qty_candidate = min_qty
            
            # 计算总成本
            total_cost = (annual_demand / qty_candidate) * ordering_cost + \
                         (qty_candidate / 2) * tier_holding_cost + \
                         annual_demand * discounted_cost
            
            # 更新最优解
            if total_cost < best_total_cost:
                best_total_cost = total_cost
                best_qty = qty_candidate
        
        return best_qty, best_total_cost
    
    def optimize_multi_warehouse_inventory(self, forecast_demands, main_inventory, lead_times, 
                                          costs, constraints, warehouse_inventory, transfer_costs):
        """
        多仓库存优化：优先调拨，减少采购
        
        Args:
            forecast_demands: 预测需求数据
            main_inventory: 主仓库库存数据
            lead_times: 提前期数据
            costs: 成本数据
            constraints: 约束条件
            warehouse_inventory: 各仓库库存数据
            transfer_costs: 调拨成本数据
            
        Returns:
            optimized_inventory: 优化后的库存数据
            transfers: 调拨计划
            gaps: 调拨后各产品的缺口
        """
        # 复制主仓库库存数据
        optimized_inventory = main_inventory.copy()
        transfers = []
        gaps = []
        
        # 检查forecast_demands的类型
        is_dict_format = isinstance(forecast_demands, dict)
        
        # 根据forecast_demands的类型确定产品ID列表
        if is_dict_format:
            product_ids = list(forecast_demands.keys())
        else:
            product_ids = range(len(forecast_demands))
        
        # 为每个产品计算总需求
        for product_id in product_ids:
            # 获取当前产品的预测需求
            if is_dict_format:
                product_demands = forecast_demands[product_id]
            else:
                product_demands = forecast_demands[product_id]
            
            total_demand = sum(product_demands)
            
            # 获取当前库存
            if isinstance(main_inventory, dict):
                current_inventory = main_inventory.get(product_id, 0)
            else:
                current_inventory = main_inventory[product_id]
            
            # 计算缺口
            gap = max(0, total_demand - current_inventory)
            
            # 如果有缺口，尝试从其他仓库调拨
            if gap > 0 and warehouse_inventory and transfer_costs:
                # 遍历所有仓库，寻找可用库存
                for warehouse_id, warehouse_stock in warehouse_inventory.items():
                    if gap <= 0:
                        break
                    
                    # 获取该仓库中当前产品的库存
                    if isinstance(warehouse_stock, dict):
                        warehouse_product_stock = warehouse_stock.get(product_id, 0)
                    else:
                        warehouse_product_stock = warehouse_stock[product_id]
                    
                    # 如果该仓库有库存，计算可调拨数量
                    if warehouse_product_stock > 0:
                        transfer_qty = min(gap, warehouse_product_stock)
                        
                        # 计算调拨成本
                        transfer_cost = transfer_costs.get(warehouse_id, {}).get(product_id, 1) * transfer_qty
                        
                        # 执行调拨
                        transfers.append({
                            'from_warehouse': warehouse_id,
                            'product_id': product_id,
                            'quantity': transfer_qty,
                            'cost': transfer_cost
                        })
                        
                        # 更新库存
                        if isinstance(optimized_inventory, dict):
                            optimized_inventory[product_id] += transfer_qty
                        else:
                            optimized_inventory[product_id] += transfer_qty
                        
                        if isinstance(warehouse_inventory[warehouse_id], dict):
                            warehouse_inventory[warehouse_id][product_id] -= transfer_qty
                        else:
                            warehouse_inventory[warehouse_id][product_id] -= transfer_qty
                        
                        # 更新缺口
                        gap = max(0, total_demand - optimized_inventory[product_id])
            
            gaps.append(gap)
        
        return optimized_inventory, transfers, gaps
    
    def calculate_reorder_point(self, avg_daily_demand, avg_lead_time, safety_stock):
        """
        计算再订货点(ROP)
        
        Args:
            avg_daily_demand: 平均日需求量
            avg_lead_time: 平均提前期（天）
            safety_stock: 安全库存
            
        Returns:
            rop: 再订货点
        """
        rop = avg_daily_demand * avg_lead_time + safety_stock
        return rop

    def calculate_service_level(self, demand, lead_time_demand_std, safety_stock):
        """
        计算服务水平
        
        Args:
            demand: 平均提前期需求
            lead_time_demand_std: 提前期需求标准差
            safety_stock: 安全库存
            
        Returns:
            service_level: 服务水平
        """
        if lead_time_demand_std == 0:
            return 1.0  # 没有需求波动，服务水平100%
        
        # 使用正态分布计算服务水平
        z_value = safety_stock / lead_time_demand_std
        
        # 使用近似的正态分布CDF
        service_level = 0.5 * (1 + erf(z_value / sqrt(2)))
        
        return service_level
