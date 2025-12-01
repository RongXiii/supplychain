import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_processor import DataProcessor
from forecast_models import ForecastModelSelector
from milp_optimizer import MILPOptimizer
from automated_replenishment import AutomatedReplenishment
from simulated_data import generate_simulated_data
from mlops_engine import MLOpsEngine
from feature_store import FeatureStore

import numpy as np
import pandas as pd
import joblib
import json
from datetime import datetime, timedelta
from math import sqrt

class ReplenishmentSystem:
    """
    补货订货策略系统，整合预测和MILP优化
    """
    
    def __init__(self):
        self.data_processor = DataProcessor()
        self.model_selector = ForecastModelSelector()
        self.milp_optimizer = MILPOptimizer()
        self.automated_replenishment = AutomatedReplenishment(self)
        self.mlops_engine = MLOpsEngine()
        self.feature_store = FeatureStore()  # 新增Feature Store实例
        self.models = {}
        
    def calculate_safety_stock(self, lead_time_demand_std, service_level=0.95, historical_data=None, product_id=None):
        """
        计算安全库存，支持参数自适应
        
        Args:
            lead_time_demand_std: 提前期需求标准差
            service_level: 服务水平（默认95%）
            historical_data: 历史需求数据（用于参数自适应）
            product_id: 产品ID（用于参数自适应）
            
        Returns:
            safety_stock: 安全库存量
        """
        # 获取对应服务水平的Z值（正态分布分位数）
        z_value = {0.90: 1.28, 0.95: 1.65, 0.97: 1.88, 0.98: 2.05, 0.99: 2.33}.get(service_level, 1.65)
        
        # 如果提供了历史数据和产品ID，使用MLOps引擎进行参数自适应
        if historical_data is not None and product_id:
            safety_stock_params = {
                'z_value': z_value,
                'std_demand': lead_time_demand_std
            }
            
            # 使用MLOps引擎更新参数
            adaptive_params = self.mlops_engine.adaptive_params_update(
                product_id, historical_data, safety_stock_params
            )
            
            # 使用自适应参数
            z_value = adaptive_params['z_value']
            lead_time_demand_std = adaptive_params['std_demand']
            
            print(f"产品 {product_id} 安全库存参数自适应更新: z_value={z_value:.2f}, std_demand={lead_time_demand_std:.2f}")
        
        safety_stock = z_value * lead_time_demand_std
        return round(safety_stock, 2)
    
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
        rop = (avg_daily_demand * avg_lead_time) + safety_stock
        return round(rop, 2)
    
    def calculate_eoq(self, demand, ordering_cost, holding_cost):
        """
        计算经济订货量(EOQ)
        
        Args:
            demand: 年需求量
            ordering_cost: 每次订货成本
            holding_cost: 单位产品年持有成本
            
        Returns:
            eoq: 经济订货量
        """
        if holding_cost == 0:
            return float('inf')  # 如果持有成本为0，EOQ为无穷大
        eoq = sqrt((2 * demand * ordering_cost) / holding_cost)
        return eoq
    
    def calculate_total_cost(self, order_qty, demand, ordering_cost, holding_cost, unit_cost=0):
        """
        计算总成本：订货成本 + 持有成本 + 采购成本
        
        Args:
            order_qty: 订货量
            demand: 年需求量
            ordering_cost: 每次订货成本
            holding_cost: 单位产品年持有成本
            unit_cost: 单位产品采购成本
            
        Returns:
            total_cost: 总成本
        """
        if order_qty == 0:
            return 0
        ordering_cost_total = (demand / order_qty) * ordering_cost
        holding_cost_total = (order_qty / 2) * holding_cost
        purchase_cost_total = demand * unit_cost
        return ordering_cost_total + holding_cost_total + purchase_cost_total
    
    def find_optimal_order_qty_with_discount(self, eoq, demand, ordering_cost, holding_cost, unit_cost, discount_tiers):
        """
        考虑数量折扣时，计算最优订货量
        
        Args:
            eoq: 无折扣时的经济订货量
            demand: 年需求量
            ordering_cost: 每次订货成本
            holding_cost: 单位产品年持有成本
            unit_cost: 单位产品采购成本
            discount_tiers: 数量折扣阶梯，格式为[(min_qty, discount_rate), ...]
            
        Returns:
            optimal_qty: 最优订货量
            optimal_cost: 最优成本
        """
        # 按数量从小到大排序折扣阶梯
        discount_tiers = sorted(discount_tiers, key=lambda x: x[0])
        
        best_qty = eoq
        best_cost = self.calculate_total_cost(eoq, demand, ordering_cost, holding_cost, unit_cost)
        
        # 考虑每个折扣阶梯
        for min_qty, discount_rate in discount_tiers:
            # 计算该阶梯的单位成本
            discounted_unit_cost = unit_cost * (1 - discount_rate)
            # 计算该阶梯的持有成本（如果持有成本与采购成本相关）
            discounted_holding_cost = holding_cost * (1 - discount_rate) if holding_cost > 0 else 0
            
            # 检查EOQ是否在该阶梯内
            if eoq >= min_qty:
                # 在该阶梯内，重新计算EOQ
                tier_eoq = sqrt((2 * demand * ordering_cost) / discounted_holding_cost)
                if tier_eoq >= min_qty:
                    # 新的EOQ在该阶梯内
                    tier_cost = self.calculate_total_cost(tier_eoq, demand, ordering_cost, discounted_holding_cost, discounted_unit_cost)
                    if tier_cost < best_cost:
                        best_qty = tier_eoq
                        best_cost = tier_cost
            else:
                # EOQ不在该阶梯内，考虑阶梯最小订货量
                tier_cost = self.calculate_total_cost(min_qty, demand, ordering_cost, discounted_holding_cost, discounted_unit_cost)
                if tier_cost < best_cost:
                    best_qty = min_qty
                    best_cost = tier_cost
        
        return best_qty, best_cost
    
    def optimize_multi_warehouse_inventory(self, forecast_demands, inventory_data, lead_times, costs, constraints, warehouse_inventory, transfer_costs):
        """
        多仓库存优化：优先调拨，减少采购
        
        Args:
            forecast_demands: 预测需求
            inventory_data: 主仓库库存数据
            lead_times: 交货提前期
            costs: 成本参数
            constraints: 约束条件
            warehouse_inventory: 其他仓库库存数据，格式为{warehouse_id: {product_id: inventory_level, ...}, ...}
            transfer_costs: 调拨成本，格式为{warehouse_id: {product_id: transfer_cost, ...}, ...}
            
        Returns:
            optimized_inventory: 优化后的库存数据
            transfers: 调拨计划
            gaps: 仍需采购的缺口
        """
        transfers = []
        gaps = []
        
        # 复制主仓库库存数据
        optimized_inventory = inventory_data.copy()
        
        # 计算每个产品的总需求
        total_demand = [sum(demands) for demands in forecast_demands]
        
        # 处理每个产品
        for product_id in range(len(forecast_demands)):
            product_demand = total_demand[product_id]
            current_inventory = optimized_inventory[product_id]
            
            # 计算初始缺口
            gap = max(0, product_demand - current_inventory)
            
            # 如果有缺口，尝试从其他仓库调拨
            if gap > 0:
                for warehouse_id, wh_inventory in warehouse_inventory.items():
                    if product_id in wh_inventory:
                        wh_stock = wh_inventory[product_id]
                        if wh_stock > 0:
                            # 可调拨数量
                            transfer_qty = min(gap, wh_stock)
                            
                            # 计算调拨成本
                            transfer_cost = transfer_costs[warehouse_id][product_id] * transfer_qty
                            
                            # 生成调拨记录
                            transfer = {
                                'from_warehouse': warehouse_id,
                                'to_warehouse': 'main',
                                'product_id': product_id + 1,
                                'quantity': transfer_qty,
                                'cost': transfer_cost
                            }
                            transfers.append(transfer)
                            
                            # 更新库存和缺口
                            optimized_inventory[product_id] += transfer_qty
                            gap = max(0, product_demand - optimized_inventory[product_id])
                            
                            # 更新其他仓库库存
                            warehouse_inventory[warehouse_id][product_id] -= transfer_qty
                            
                            # 如果缺口已满足，停止调拨
                            if gap == 0:
                                break
            
            # 记录最终缺口
            gaps.append(gap)
        
        return optimized_inventory, transfers, gaps
    
    def implement_rop_strategy(self, current_inventory, avg_daily_demand, avg_lead_time, lead_time_demand_std, service_level=0.95, ordering_cost=100, holding_cost=10):
        """
        实现ROP（再订货点）+ 安全库存策略
        
        Args:
            current_inventory: 当前库存量
            avg_daily_demand: 平均日需求量
            avg_lead_time: 平均提前期（天）
            lead_time_demand_std: 提前期需求标准差
            service_level: 服务水平（默认95%）
            ordering_cost: 每次订货成本（用于计算EOQ）
            holding_cost: 单位产品年持有成本（用于计算EOQ）
            
        Returns:
            dict: 包含是否需要补货、再订货点、安全库存、建议订货量等信息
        """
        # 计算安全库存
        safety_stock = self.calculate_safety_stock(lead_time_demand_std, service_level)
        
        # 计算再订货点
        rop = self.calculate_reorder_point(avg_daily_demand, avg_lead_time, safety_stock)
        
        # 计算EOQ作为建议订货量
        annual_demand = avg_daily_demand * 365
        eoq = self.calculate_eoq(annual_demand, ordering_cost, holding_cost)
        
        # 判断是否需要补货
        need_replenishment = current_inventory <= rop
        
        if need_replenishment:
            # 建议订货量为EOQ
            suggested_order_qty = eoq
        else:
            suggested_order_qty = 0
        
        return {
            'need_replenishment': need_replenishment,
            'reorder_point': rop,
            'safety_stock': safety_stock,
            'suggested_order_qty': round(suggested_order_qty, 2),
            'current_inventory': current_inventory,
            'avg_daily_demand': avg_daily_demand,
            'avg_lead_time': avg_lead_time
        }
    
    def implement_order_up_to_strategy(self, current_inventory, on_order_quantity, demand_forecast, lead_time, safety_stock, review_period=1):
        """
        实现Order-up-to Level（补到目标库存）策略
        
        Args:
            current_inventory: 当前库存量
            on_order_quantity: 已订购但未到货的数量
            demand_forecast: 预测需求（考虑提前期和检查周期）
            lead_time: 交货提前期（天）
            safety_stock: 安全库存
            review_period: 库存检查周期（默认1天）
            
        Returns:
            dict: 包含是否需要补货、目标库存水平、建议订货量等信息
        """
        # 计算目标库存水平
        # 目标库存 = 提前期+检查周期的预测需求 + 安全库存
        order_up_to_level = demand_forecast + safety_stock
        
        # 计算当前可用库存（当前库存 + 已订购未到货）
        available_inventory = current_inventory + on_order_quantity
        
        # 计算建议订货量
        suggested_order_qty = max(0, order_up_to_level - available_inventory)
        
        # 判断是否需要补货
        need_replenishment = suggested_order_qty > 0
        
        return {
            'need_replenishment': need_replenishment,
            'order_up_to_level': round(order_up_to_level, 2),
            'suggested_order_qty': round(suggested_order_qty, 2),
            'current_inventory': current_inventory,
            'on_order_quantity': on_order_quantity,
            'available_inventory': round(available_inventory, 2),
            'demand_forecast': demand_forecast,
            'safety_stock': safety_stock
        }
    
    def run_forecast(self, product_data, product_id):
        """
        运行预测流程，包括数据预处理、模型选择和预测
        
        Args:
            product_data: 产品历史数据
            product_id: 产品ID
            
        Returns:
            forecast_result: 预测结果，包括模型信息和预测值
        """
        # 获取产品数据中的位置信息
        location_id = product_data.get('location_id', 1)
        
        # 预处理数据
        processed_data = self.data_processor.preprocess_data(product_data)
        
        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = self.data_processor.split_data(processed_data)
        
        # 漂移检测：比较训练数据和测试数据的分布
        if len(y_train) > 10 and len(y_test) > 10:
            drift_result = self.mlops_engine.detect_drift(y_train, y_test, product_id)
            print(f"产品 {product_id} 漂移检测结果: {drift_result}")
        
        # 更新SKU×仓库的特征
        demand_series = product_data.get('demand_series', y_train + y_test)
        self.update_sku_location_features(product_id, location_id, demand_series)
        
        # 获取模型选择标签并选择模型
        model_tag = self.get_model_selection_tag(product_id, location_id)
        best_model, model_name, best_score = self.model_selector.select_best_model(X_train, y_train, product_id, model_tag=model_tag)
        
        # 在测试集上评估模型
        test_metrics = self.model_selector.evaluate_model(best_model, model_name, X_test, y_test)
        
        # 计算误差指标：MAPE、SMAPE、RMSE
        if model_name in ['arima', 'holt_winters']:
            # 统计模型需要的是单变量时间序列数据，使用y_test
            y_pred = self.model_selector.predict(best_model, model_name, y_test)
        else:
            # 机器学习模型使用特征数据
            y_pred = self.model_selector.predict(best_model, model_name, X_test)
        error_metrics = self.mlops_engine.calculate_error_metrics(y_test, y_pred, product_id)
        
        # 保存模型
        self.models[product_id] = {
            'model': best_model,
            'model_name': model_name,
            'score': best_score,
            'metrics': test_metrics,
            'error_metrics': error_metrics
        }
        
        # 使用MLOps引擎保存模型
        self.mlops_engine.save_model(product_id, best_model, model_name, metrics=error_metrics)
        
        # 进行未来预测
        future_predictions = self.model_selector.forecast(best_model, model_name, X_test)
        
        return {
            'product_id': product_id,
            'model_name': model_name,
            'model_score': best_score,
            'test_metrics': test_metrics,
            'error_metrics': error_metrics,
            'predictions': future_predictions
        }
    
    def run_optimization(self, forecast_results, inventory_data, lead_times, costs, constraints, warehouse_inventory=None, transfer_costs=None, discount_tiers=None):
        """
        运行MILP优化，生成最优订货策略
        
        Args:
            forecast_results: 预测结果
            inventory_data: 库存数据
            lead_times: 交货提前期
            costs: 成本参数
            constraints: 约束条件
            warehouse_inventory: 其他仓库库存数据
            transfer_costs: 调拨成本
            discount_tiers: 数量折扣阶梯
            
        Returns:
            optimization_result: 优化结果，包括调拨计划和采购计划
        """
        # 整理预测需求
        forecast_demands = []
        for result in forecast_results:
            forecast_demands.append(result['predictions'])
        
        # 确保所有产品的预测时间段相同
        num_periods = len(forecast_demands[0])
        for i in range(1, len(forecast_demands)):
            if len(forecast_demands[i]) != num_periods:
                # 填充或截断预测结果，使其长度一致
                if len(forecast_demands[i]) > num_periods:
                    forecast_demands[i] = forecast_demands[i][:num_periods]
                else:
                    forecast_demands[i] = np.pad(forecast_demands[i], (0, num_periods - len(forecast_demands[i])), 'constant')
        
        # 转换为numpy数组
        forecast_demands = np.array(forecast_demands)
        
        # 保存原始库存数据
        original_inventory = inventory_data.copy()
        
        # 复制约束条件并进行调整
        adjusted_constraints = constraints.copy() if constraints else {}
        
        # 多仓库存优化：优先调拨，减少采购
        optimized_inventory = inventory_data.copy()
        transfers = []
        gaps = [0] * len(forecast_demands)
        
        if warehouse_inventory and transfer_costs:
            print("执行多仓库存优化...")
            optimized_inventory, transfers, gaps = self.optimize_multi_warehouse_inventory(
                forecast_demands, inventory_data, lead_times, costs, adjusted_constraints,
                warehouse_inventory, transfer_costs
            )
            print(f"调拨完成，共生成 {len(transfers)} 笔调拨单")
            for transfer in transfers:
                print(f"  从仓库 {transfer['from_warehouse']} 调拨 {transfer['quantity']:.2f} 单位产品 {transfer['product_id']} 到主仓库")
            print(f"调拨后各产品缺口: {gaps}")
        
        # 添加预算约束（示例：每周期预算为10000）
        adjusted_constraints['budget_constraint'] = 10000
        
        # EOQ计算和数量折扣处理
        optimal_order_quantities = []
        for product_id in range(len(forecast_demands)):
            # 计算年需求量
            total_annual_demand = sum(forecast_demands[product_id])
            
            # 计算EOQ
            eoq = self.calculate_eoq(
                demand=total_annual_demand,
                ordering_cost=costs['ordering_cost'][product_id],
                holding_cost=costs['holding_cost'][product_id]
            )
            
            # 考虑数量折扣
            optimal_qty = eoq
            if discount_tiers and product_id in discount_tiers:
                # 假设unit_cost为100（实际应用中应从成本数据获取）
                unit_cost = 100
                optimal_qty, optimal_cost = self.find_optimal_order_qty_with_discount(
                    eoq, total_annual_demand, costs['ordering_cost'][product_id],
                    costs['holding_cost'][product_id], unit_cost, discount_tiers[product_id]
                )
            
            optimal_order_quantities.append(optimal_qty)
            print(f"产品 {product_id+1}: EOQ={eoq:.2f}, 最优订货量={optimal_qty:.2f}")
        
        # 调整MILP约束，考虑调拨后的缺口和最优订货量
        adjusted_constraints = constraints.copy()
        
        # 根据调拨后的缺口调整库存数据
        adjusted_inventory = optimized_inventory.copy()
        
        # 运行MILP优化
        print("执行MILP优化...")
        
        # 将折扣阶梯转换为MILP优化器所需的格式
        if discount_tiers:
            costs['quantity_discounts'] = []
            for product_id in range(len(forecast_demands)):
                if product_id in discount_tiers:
                    discount_list = []
                    for min_qty, discount_rate in discount_tiers[product_id]:
                        discount_list.append({
                            'min_quantity': min_qty,
                            'discount_rate': discount_rate
                        })
                    costs['quantity_discounts'].append(discount_list)
                else:
                    costs['quantity_discounts'].append([])
        
        # 运行MILP优化
        optimization_result = self.milp_optimizer.optimize(
            forecast_demands=forecast_demands,
            current_inventory=adjusted_inventory,
            lead_times=lead_times,
            costs=costs,
            constraints=adjusted_constraints
        )
        
        # 将调拨计划添加到优化结果中
        if optimization_result:
            optimization_result['transfers'] = transfers
            optimization_result['original_inventory'] = original_inventory
            optimization_result['optimized_inventory'] = optimized_inventory
            optimization_result['gaps'] = gaps
            optimization_result['eoq'] = optimal_order_quantities
        
        return optimization_result
    
    def generate_purchase_orders(self, optimization_result):
        """
        生成采购订单
        
        Args:
            optimization_result: 优化结果
            
        Returns:
            purchase_orders: 采购订单列表
        """
        if optimization_result is None:
            return []
        
        return optimization_result.get('purchase_orders', [])
    
    def update_model_with_actual_data(self, product_id, actual_data):
        """
        使用实际数据更新模型，集成MLOps功能：误差分析、漂移检测、模型重训、策略回滚
        
        Args:
            product_id: 产品ID
            actual_data: 实际数据
            
        Returns:
            updated_model: 更新后的模型
        """
        # 检查模型是否存在
        if product_id not in self.models:
            return None
        
        # 预处理实际数据
        processed_data = self.data_processor.preprocess_data(actual_data)
        
        # 准备特征和标签
        X = processed_data.iloc[:, :-1]
        y = processed_data.iloc[:, -1]
        
        # 获取当前模型
        current_model = self.models[product_id]['model']
        current_model_name = self.models[product_id]['model_name']
        
        # 使用当前模型进行预测
        current_predictions = self.model_selector.predict(current_model, current_model_name, X)
        
        # 计算误差指标
        error_metrics = self.mlops_engine.calculate_error_metrics(y, current_predictions, product_id)
        print(f"产品 {product_id} 误差指标: {error_metrics}")
        
        # 漂移检测：比较历史数据和新数据
        if product_id in self.models:
            # 使用模型的训练数据作为基线
            # 这里简化处理，实际应使用保存的训练数据
            baseline_data = y[:len(y)//2]  # 使用一半数据作为基线
            current_data = y[len(y)//2:]   # 使用另一半数据作为当前数据
            drift_result = self.mlops_engine.detect_drift(baseline_data, current_data, product_id)
            print(f"产品 {product_id} 漂移检测结果: {drift_result}")
        
        # 决定是否需要重训模型
        # 获取指标历史记录（这里简化处理，实际应从MLOps引擎获取）
        metrics_history = [error_metrics] if error_metrics else []
        drift_results = drift_result if 'drift_result' in locals() else None
        
        should_retrain, reason = self.mlops_engine.should_retrain_model(product_id, metrics_history, drift_results)
        
        updated_model = None
        if should_retrain:
            print(f"产品 {product_id} 需要重训模型，原因: {reason}")
            
            # 保存当前模型作为回滚点
            self.mlops_engine._save_rollback_point(product_id, {
                'model': current_model,
                'model_name': current_model_name,
                'metrics': error_metrics
            })
            
            # 重训模型
            updated_model, model_name, best_score = self.model_selector.select_best_model(X, y, product_id)
            
            if updated_model is not None:
                # 评估重训后的模型
                y_pred = self.model_selector.predict(updated_model, model_name, X)
                new_error_metrics = self.mlops_engine.calculate_error_metrics(y, y_pred, product_id)
                
                print(f"模型重训完成，新模型名称: {model_name}, 得分: {best_score:.4f}")
                print(f"重训后误差指标: {new_error_metrics}")
                
                # 检查重训后的模型是否更差，如果更差则回滚
                if new_error_metrics and error_metrics:
                    if new_error_metrics['mape'] > error_metrics['mape'] * 1.2:  # 新模型MAPE比旧模型差20%以上
                        print(f"重训后的模型性能下降，执行回滚")
                        # 使用旧模型
                        updated_model = current_model
                        model_name = current_model_name
                    else:
                        # 保存新模型
                        self.mlops_engine.save_model(product_id, updated_model, model_name, new_error_metrics)
                else:
                    # 保存新模型
                    self.mlops_engine.save_model(product_id, updated_model, model_name, new_error_metrics)
        else:
            # 不需要重训，只更新模型
            print(f"产品 {product_id} 不需要重训模型")
            updated_model, model_name = self.model_selector.update_model(product_id, X, y)
        
        if updated_model is not None:
            # 更新模型字典
            self.models[product_id]['model'] = updated_model
            self.models[product_id]['model_name'] = model_name
            self.models[product_id]['metrics'] = error_metrics
        
        # 返回包含模型信息的字典，而不仅仅是模型对象
        return self.models[product_id] if updated_model is not None else None
    
    def compare_demand_forecast(self, actual_demand, forecast_demand, product_id=None):
        """
        比较实际需求和预测需求，使用MLOps引擎的误差分析功能
        
        Args:
            actual_demand: 实际需求
            forecast_demand: 预测需求
            product_id: 产品ID（可选）
            
        Returns:
            comparison_result: 比较结果，包括误差指标
        """
        # 使用MLOps引擎计算误差指标
        error_metrics = self.mlops_engine.calculate_error_metrics(actual_demand, forecast_demand, product_id)
        
        # 获取数据处理器的比较结果
        data_processor_result = self.data_processor.compare_demand(actual_demand, forecast_demand)
        
        # 合并结果
        comparison_result = {
            'mlops_metrics': error_metrics,
            'data_processor_result': data_processor_result
        }
        
        return comparison_result
    
    def get_system_status(self):
        """
        获取系统状态，包括MLOps相关信息
        
        Returns:
            status: 系统状态，包括已训练的模型数量、模型性能、漂移检测结果等
        """
        # 收集每个产品的模型性能报告
        model_performance_reports = {}
        for product_id in self.models:
            report = self.mlops_engine.get_model_performance_report(product_id, time_range='30d')
            if report:
                model_performance_reports[product_id] = report
        
        # 收集漂移检测结果
        drift_results = self.mlops_engine.drift_detection_results
        
        return {
            'trained_models': len(self.models),
            'model_details': {k: v['model_name'] for k, v in self.models.items()},
            'automated_replenishment_status': self.automated_replenishment.get_system_status(),
            'mlops_status': {
                'model_performance_reports': model_performance_reports,
                'drift_detection_results': drift_results,
                'current_policies': self.mlops_engine.current_policies,
                'gray_release_config': self.mlops_engine.gray_release_config
            }
        }
    
    def update_sku_location_features(self, sku_id, location_id, demand_series):
        """
        更新SKU×仓库的特征
        
        Args:
            sku_id: SKU ID
            location_id: 仓库ID
            demand_series: 需求时间序列数据
        """
        self.feature_store.update_features(sku_id, location_id, demand_series)
    
    def get_sku_location_features(self, sku_id, location_id):
        """
        获取SKU×仓库的特征
        
        Args:
            sku_id: SKU ID
            location_id: 仓库ID
            
        Returns:
            features: 特征字典
        """
        return self.feature_store.get_features(sku_id, location_id)
    
    def get_model_selection_tag(self, sku_id, location_id):
        """
        获取模型选择标签
        
        Args:
            sku_id: SKU ID
            location_id: 仓库ID
            
        Returns:
            model_tag: 模型选择标签
        """
        return self.feature_store.get_model_selection_tag(sku_id, location_id)
    
    def batch_update_features(self, demand_data):
        """
        批量更新所有SKU×仓库的特征
        
        Args:
            demand_data: 需求数据，包含多个SKU×仓库的需求序列
        """
        self.feature_store.batch_update_features(demand_data)
    
    def generate_feature_report(self):
        """
        生成特征报告
        
        Returns:
            report: 特征报告
        """
        return self.feature_store.generate_feature_report()
    
    def execute_auto_replenishment(self, strategy='hybrid', requester_role='buyer'):
        """
        执行自动补单
        
        Args:
            strategy: 补货策略（'hybrid', 'rop', 'order_up_to'）
            requester_role: 请求人角色
            
        Returns:
            replenishment_result: 补单结果
        """
        return self.automated_replenishment.execute_auto_replenishment(strategy, requester_role)
    
    def execute_replenishment_strategy(self, strategy='hybrid'):
        """
        执行补货策略，利用系统已有数据
        
        Args:
            strategy: 补货策略 ('rop', 'order_up_to', 'hybrid')
            
        Returns:
            list: 补货建议列表
        """
        # 示例数据，实际应从系统中获取
        # 这里使用一些合理的默认值作为示例
        product_count = len(self.current_inventory)
        
        # 生成示例参数
        avg_daily_demand = [50, 60, 40]  # 平均日需求量
        avg_lead_time = [10, 12, 8]  # 平均提前期（天）
        lead_time_demand_std = [50, 60, 40]  # 提前期需求标准差
        on_order_quantity = [0, 0, 0]  # 已订购未到货数量
        demand_forecast = [500, 720, 320]  # 提前期预测需求
        
        replenishment_suggestions = []
        
        for i in range(product_count):
            product_id = i + 1
            current_inv = self.current_inventory[i]
            
            if strategy == 'rop' or strategy == 'hybrid':
                # 计算安全库存
                safety_stock = self.calculate_safety_stock(lead_time_demand_std[i], service_level=0.95)
                
                # 计算EOQ
                annual_demand = avg_daily_demand[i] * 365
                eoq = self.calculate_eoq(annual_demand, ordering_cost=100, holding_cost=10)
                
                # 计算再订货点
                rop = self.calculate_reorder_point(avg_daily_demand[i], avg_lead_time[i], safety_stock)
                
                # 判断是否需要补货
                need_replenish_rop = current_inv <= rop
                rop_suggestion = eoq if need_replenish_rop else 0
            
            if strategy == 'order_up_to' or strategy == 'hybrid':
                # 计算安全库存
                safety_stock = self.calculate_safety_stock(lead_time_demand_std[i], service_level=0.95)
                
                # 计算目标库存水平
                order_up_to_level = demand_forecast[i] + safety_stock
                
                # 计算可用库存
                available_inventory = current_inv + on_order_quantity[i]
                
                # 计算建议订货量
                oul_suggestion = max(0, order_up_to_level - available_inventory)
            
            # 确定最终建议订货量
            if strategy == 'rop':
                suggested_qty = rop_suggestion
            elif strategy == 'order_up_to':
                suggested_qty = oul_suggestion
            else:  # hybrid
                suggested_qty = max(rop_suggestion, oul_suggestion)
            
            # 生成补货建议
            replenishment_suggestions.append({
                'product_id': product_id,
                'current_inventory': current_inv,
                'suggested_order_qty': suggested_qty,
                'need_replenishment': suggested_qty > 0
            })
        
        return replenishment_suggestions
    
    def process_approval_request(self, order_id, action, approver_role, reason=None):
        """
        处理审批请求
        
        Args:
            order_id: 订单ID
            action: 审批动作（'approve', 'reject'）
            approver_role: 审批人角色
            reason: 拒绝原因（可选）
            
        Returns:
            approval_result: 审批结果
        """
        return self.automated_replenishment.process_approval_request(order_id, action, approver_role, reason)

def generate_sample_data():
    """
    生成示例数据用于测试
    """
    # 生成示例产品数据
    np.random.seed(42)
    
    # 生成3个产品，每个产品24个月的历史数据
    num_products = 3
    num_months = 24
    
    sample_data = {}
    for product_id in range(1, num_products + 1):
        # 生成日期
        dates = pd.date_range(start='2020-01-01', periods=num_months, freq='M')
        
        # 生成需求数据（带趋势和季节性）
        trend = np.linspace(100, 200, num_months)
        seasonality = 50 * np.sin(np.linspace(0, 4 * np.pi, num_months))
        noise = np.random.normal(0, 10, num_months)
        demand = trend + seasonality + noise
        demand = np.maximum(demand, 0)  # 需求不能为负
        
        # 生成其他特征（这里简单使用前几个月的需求作为特征）
        df = pd.DataFrame({'date': dates, 'demand': demand})
        
        # 添加滞后特征
        for i in 1, 2, 3:
            df[f'demand_lag_{i}'] = df['demand'].shift(i)
        
        # 删除包含NaN的行
        df = df.dropna()
        
        sample_data[product_id] = df
    
    return sample_data

def main():
    """
    主函数，演示整个补货订货策略系统的流程
    """
    global pd  # 确保使用全局的pandas模块
    
    # 使用更专业的欢迎信息
    print("=" * 60)
    print("📦 供应链智能补货系统")
    print("=" * 60)
    
    # 初始化系统
    system = ReplenishmentSystem()
    
    # 1. 加载示例数据
    print("\n🚀 第1步：加载示例数据")
    print("-" * 40)
    
    # 生成新的数据表模拟数据
    print("📊 正在生成模拟数据...")
    simulated_tables = generate_simulated_data()
    
    # 展示生成的数据表信息
    print("\n✅ 生成的数据表详情：")
    for table_name, df in simulated_tables.items():
        print(f"  • {table_name}: {df.shape[0]:>4} 行 × {df.shape[1]:>2} 列")
    
    # 2. 准备产品历史数据
    print("\n\n🚀 第2步：准备产品历史数据")
    print("-" * 40)
    
    # 从inventory_daily生成产品历史需求数据
    inventory_df = simulated_tables['inventory_daily']
    
    # 按item_id分组，生成每个产品的历史需求数据
    sample_data = {}
    for item_id in inventory_df['item_id'].unique():
        # 获取该产品的历史需求数据
        item_df = inventory_df[inventory_df['item_id'] == item_id].copy()
        
        # 按日期排序
        item_df['date'] = pd.to_datetime(item_df['date'])
        item_df = item_df.sort_values('date')
        
        # 创建产品数据，包含日期和需求
        product_df = pd.DataFrame({
            'date': item_df['date'],
            'demand': item_df['demand_qty']
        })
        
        # 添加滞后特征
        for i in range(1, 4):
            product_df[f'demand_lag_{i}'] = product_df['demand'].shift(i)
        
        # 删除包含NaN的行
        product_df = product_df.dropna()
        
        sample_data[item_id] = product_df
    
    print(f"✅ 已准备 {len(sample_data)} 个产品的历史需求数据")
    
    # 保存模拟数据到CSV文件
    print("\n💾 正在保存模拟数据到CSV文件...")
    for table_name, df in simulated_tables.items():
        csv_path = f'./data/{table_name}.csv'
        df.to_csv(csv_path, index=False)
        print(f"  • {table_name:<20} 已保存到 {csv_path}")
    
    # 3. 运行预测流程
    print("\n\n🚀 第3步：运行需求预测流程")
    print("-" * 40)
    
    forecast_results = []
    total_products = len(sample_data)
    
    for idx, (product_id, data) in enumerate(sample_data.items(), 1):
        print(f"\n🔄 处理产品 {product_id:<5} ({idx}/{total_products})")
        print(f"  └─ 正在训练预测模型...")
        
        forecast_result = system.run_forecast(data, product_id)
        forecast_results.append(forecast_result)
        
        print(f"  ├─ 最佳模型: {forecast_result['model_name']}")
        print(f"  ├─ 模型得分: {forecast_result['model_score']:.4f}")
        print(f"  └─ 测试集指标: {forecast_result['test_metrics']}")
    
    print(f"\n✅ 预测流程完成，共训练 {len(forecast_results)} 个产品模型")
    
    # 4. 准备MILP优化所需的数据
    print("\n\n🚀 第4步：准备MILP优化数据")
    print("-" * 40)
    
    # 从模拟数据中获取当前库存（使用最新日期的数据）
    inventory_df = simulated_tables['inventory_daily']
    latest_date = inventory_df['date'].max()
    latest_inventory = inventory_df[inventory_df['date'] == latest_date]
    
    # 获取所有唯一产品ID
    unique_item_ids = sorted(latest_inventory['item_id'].unique())
    
    # 动态生成当前库存数据
    current_inventory = []
    for item_id in unique_item_ids:
        # 获取该产品的当前库存
        item_inv = latest_inventory[latest_inventory['item_id'] == item_id]['on_hand_qty'].values[0]
        current_inventory.append(item_inv)
    
    # 动态生成提前期（基于供应商数据，假设每个产品只有一个主要供应商）
    lead_times = []
    suppliers_df = simulated_tables['suppliers']
    purchase_orders_df = simulated_tables['purchase_orders']
    for item_id in unique_item_ids:
        # 查找该产品的主要供应商
        item_orders = purchase_orders_df[purchase_orders_df['item_id'] == item_id]
        if not item_orders.empty:
            # 取最近一次订单的供应商
            main_supplier = item_orders['supplier_id'].iloc[-1]
            # 获取该供应商的提前期
            lead_time = suppliers_df[suppliers_df['supplier_id'] == main_supplier]['lead_time_days'].values[0]
        else:
            # 如果没有订单，使用默认提前期
            lead_time = 1
        lead_times.append(lead_time)
    
    # 动态生成成本数据
    costs = {
        'ordering_cost': [100] * len(unique_item_ids),  # 为每个产品设置相同的订货成本
        'holding_cost': [2.5] * len(unique_item_ids),  # 为每个产品设置相同的持有成本
        'shortage_cost': [12] * len(unique_item_ids)  # 为每个产品设置相同的缺货成本
    }
    
    # 动态生成约束条件
    constraints = {
        'max_order_quantity': [500] * len(unique_item_ids),  # 每个产品的最大订货量
        'min_order_quantity': [50] * len(unique_item_ids),  # 每个产品的最小订货量
        'max_inventory': [1000] * len(unique_item_ids)  # 每个产品的最大库存
    }
    
    # 动态生成多仓库存数据
    warehouse_inventory = {
        'warehouse_1': {},
        'warehouse_2': {}
    }
    
    # 为每个产品在仓库中设置随机库存
    import random
    for i, item_id in enumerate(unique_item_ids):
        # 为每个仓库分配随机库存
        warehouse_inventory['warehouse_1'][i] = random.randint(0, 200)  # 0到200之间的随机库存
        warehouse_inventory['warehouse_2'][i] = random.randint(0, 150)  # 0到150之间的随机库存
    
    # 动态生成调拨成本数据
    transfer_costs = {
        'warehouse_1': {},
        'warehouse_2': {}
    }
    
    for i, item_id in enumerate(unique_item_ids):
        # 为每个产品设置随机调拨成本
        transfer_costs['warehouse_1'][i] = round(random.uniform(0.5, 2.0), 2)  # 0.5到2.0之间的随机成本
        transfer_costs['warehouse_2'][i] = round(random.uniform(0.8, 2.5), 2)  # 0.8到2.5之间的随机成本
    
    # 动态生成数量折扣数据
    discount_tiers = {}
    for i, item_id in enumerate(unique_item_ids):
        # 为每个产品生成随机折扣阶梯
        discount_tiers[i] = [
            (random.randint(50, 100), round(random.uniform(0.02, 0.05), 3)),  # 第一阶梯
            (random.randint(150, 250), round(random.uniform(0.06, 0.10), 3)),  # 第二阶梯
            (random.randint(400, 500), round(random.uniform(0.12, 0.18), 3))   # 第三阶梯
        ]
    
    print("✅ MILP优化数据准备完成")
    
    # 5. 运行MILP优化
    print("\n\n🚀 第5步：运行MILP优化")
    print("-" * 40)
    
    print("🔄 正在求解最优订货方案...")
    optimization_result = system.run_optimization(
        forecast_results, current_inventory, lead_times, costs, constraints,
        warehouse_inventory=warehouse_inventory,
        transfer_costs=transfer_costs,
        discount_tiers=discount_tiers
    )
    
    if optimization_result:
        print("\n🎉 优化完成！")
        print("=" * 40)
        print(f"📊 总优化成本: {optimization_result['total_cost']:.2f}")
        
        # 打印调拨计划
        if 'transfers' in optimization_result and optimization_result['transfers']:
            print("\n🔄 调拨计划:")
            for transfer in optimization_result['transfers']:
                print(f"  • 从 {transfer['from_warehouse']:<12} 调拨 {transfer['quantity']:>6.2f} 单位产品 {transfer['product_id']:<5} 到主仓库，成本: {transfer['cost']:>6.2f}")
        
        # 打印EOQ和最优订货量
        if 'eoq' in optimization_result:
            print("\n📏 EOQ计算结果:")
            for product_id, eoq_val in enumerate(optimization_result['eoq']):
                print(f"  • 产品 {product_id+1:<5}: EOQ = {eoq_val:>8.2f}")
        
        # 打印最优订货量
        print("\n📋 最优订货量:")
        for product_id, order_qtys in optimization_result['order_quantities'].items():
            print(f"  • 产品 {product_id+1:<5}: {order_qtys}")
        
        # 打印价格阶梯选择
        if 'discount_selections' in optimization_result:
            print("\n💲 价格阶梯选择:")
            for product_id, selections in optimization_result['discount_selections'].items():
                print(f"  • 产品 {product_id+1:<5}: {selections}")
        
        # 打印期望到货日期
        if 'expected_arrival_dates' in optimization_result:
            print("\n📅 期望到货日期:")
            for product_id, dates in optimization_result['expected_arrival_dates'].items():
                print(f"  • 产品 {product_id+1:<5}: {dates}")
        
        # 打印OptimalPlan
        if 'optimal_plan' in optimization_result:
            print("\n📈 最优补货计划:")
            for plan in optimization_result['optimal_plan']:
                print(f"  • 产品 {plan['product_id']:<5}, 时期 {plan['period']:<2}: 订货量={plan['optimal_order_qty']:>6.2f}, 价格阶梯={plan['price_tier']:<2}, 期望到货={plan['expected_arrival_date']}")
        
        # 生成采购订单
        purchase_orders = system.generate_purchase_orders(optimization_result)
        
        print("\n📝 生成的采购订单:")
        total_order_cost = 0
        for order in purchase_orders:
            print(f"  • 产品 {order['product_id']:<5}: 订货量 {order['order_quantity']:>6.2f}")
            total_order_cost += order['order_quantity'] * 100  # 假设单位成本为100
        print(f"  • 采购订单总成本 (估算): {total_order_cost:>12.2f}")
    else:
        print("\n⚠️ MILP优化失败，可能是因为没有可用的求解器。")
        print("   请安装GLPK、CBC等求解器后重试。")
        print("   系统仍然可以进行预测和数据分析功能。")
    
    # 6. 显示系统状态
    print("\n\n🚀 第6步：查看系统状态")
    print("-" * 40)
    
    status = system.get_system_status()
    print(f"✅ 已训练模型数量: {status['trained_models']}")
    print(f"📋 模型详情: {status['model_details']}")
    
    # 显示MLOps相关状态
    if 'mlops_status' in status:
        print("\n🔬 MLOps状态:")
        print(f"  • 模型性能报告: 已生成")
        print(f"  • 漂移检测结果: {status['mlops_status']['drift_detection_results']}")
        print(f"  • 当前策略配置: {status['mlops_status']['current_policies']}")
        print(f"  • 灰度上线状态: {status['mlops_status']['gray_release_config']}")
    
    # 7. 演示模型更新和MLOps功能
    print("\n\n🚀 第7步：演示模型更新和MLOps功能")
    print("-" * 40)
    
    # 使用最后一个产品的数据作为示例
    if sample_data:
        product_id = list(sample_data.keys())[-1]
        product_data = sample_data[product_id]
        
        # 取前10行数据作为新数据更新模型
        new_data = product_data.head(10)
        
        print(f"🔄 使用产品 {product_id} 的最新数据更新模型...")
        updated_model = system.update_model_with_actual_data(product_id, new_data)
        
        if updated_model:
            print(f"✅ 模型更新成功")
            print(f"  • 更新后的模型: {updated_model['model_name']}")
            print(f"  • 更新后得分: {updated_model['score']:.4f}")
            
            # 显示误差分析结果
            if 'metrics' in updated_model:
                print(f"  • 误差分析: {updated_model['metrics']}")
            
            # 显示漂移检测结果
            if 'drift_detected' in updated_model:
                print(f"  • 漂移检测: {'⚠️ 检测到漂移' if updated_model['drift_detected'] else '✅ 未检测到漂移'}")
            
            # 显示模型重训结果
            if 'retrained' in updated_model:
                print(f"  • 模型重训: {'✅ 已重训' if updated_model['retrained'] else '⏭️  未重训'}")
    
    # 8. 演示自动补单功能
    print("\n\n🚀 第8步：演示自动补单功能")
    print("-" * 40)
    
    # 设置当前库存，使用较低的值以触发补货
    system.current_inventory = [20, 15, 10]  # 低库存，触发补货
    
    # 执行自动补单（混合策略）
    print("🔄 执行自动补单（混合策略）...")
    auto_replenish_result = system.execute_auto_replenishment(strategy='hybrid', requester_role='buyer')
    
    print("✅ 自动补单完成")
    print(f"  • 总建议数: {auto_replenish_result['total_suggestions']}")
    print(f"  • 生成订单数: {auto_replenish_result['generated_orders']}")
    
    # 查看生成的采购订单
    if auto_replenish_result['orders']:
        print("\n📝 生成的采购订单：")
        for order in auto_replenish_result['orders']:
            print(f"  • 订单ID: {order['order_id']:<10}, 产品: {order['product_id']:<5}, 数量: {order['order_quantity']:>6.2f}, 状态: {order['status']:<10}, 审批状态: {order['approval_status']:<10}")
    
    # 9. 演示审批流程
    print("\n\n🚀 第9步：演示审批流程")
    print("-" * 40)
    
    orders = system.automated_replenishment.get_purchase_orders()
    if orders:
        # 审批第一个订单
        first_order = orders[0]
        print(f"🔄 审批订单ID: {first_order['order_id']}")
        
        # 使用admin角色批准订单
        approval_result = system.process_approval_request(first_order['order_id'], 'approve', 'admin')
        print(f"  • 审批结果: {approval_result['status']} - {approval_result['message']}")
        
        # 拒绝第二个订单（如果有）
        if len(orders) > 1:
            second_order = orders[1]
            print(f"\n🔄 拒绝订单ID: {second_order['order_id']}")
            rejection_result = system.process_approval_request(second_order['order_id'], 'reject', 'admin', reason='库存充足')
            print(f"  • 拒绝结果: {rejection_result['status']} - {rejection_result['message']}")
    
    # 10. 查看最终的采购订单状态
    print("\n\n🚀 第10步：最终采购订单状态")
    print("-" * 40)
    
    final_orders = system.automated_replenishment.get_purchase_orders()
    for order in final_orders:
        print(f"📋 订单ID: {order['order_id']:<10}, 产品: {order['product_id']:<5}, 数量: {order['order_quantity']:>6.2f}, 状态: {order['status']:<10}, 审批状态: {order['approval_status']:<10}")
    
    # 11. 显示完整系统状态
    print("\n\n🚀 第11步：完整系统状态")
    print("-" * 40)
    
    full_status = system.get_system_status()
    print(f"✅ 已训练模型数量: {full_status['trained_models']}")
    print(f"📋 自动补单状态: {full_status['automated_replenishment_status']}")
    
    # 显示完整的MLOps状态
    if 'mlops_status' in full_status:
        print("\n🔬 完整MLOps状态:")
        mlops_status = full_status['mlops_status']
        
        print("\n📊 模型性能报告:")
        for product_id, report in mlops_status['model_performance_reports'].items():
            print(f"  • 产品 {product_id}:")
            print(f"    ├─ MAPE: {report['average_metrics']['mape']:.4f}")
            print(f"    ├─ SMAPE: {report['average_metrics']['smape']:.4f}")
            print(f"    └─ RMSE: {report['average_metrics']['rmse']:.4f}")
        
        print("\n🎯 漂移检测结果:")
        for product_id, drift_result in mlops_status['drift_detection_results'].items():
            # 正确判断漂移状态，检查字典中的drift_detected字段
            drift_detected = drift_result.get('drift_detected', False) if isinstance(drift_result, dict) else False
            status_text = "⚠️  漂移" if drift_detected else "✅ 正常"
            print(f"  • 产品 {product_id}: {status_text}")
            if isinstance(drift_result, dict):
                print(f"    ├─ p值: {drift_result.get('p_value', 'N/A')}")
                print(f"    ├─ 检验统计量: {drift_result.get('test_statistic', 'N/A')}")
                print(f"    └─ 样本量: 基线={drift_result.get('sample_size', {}).get('baseline', 'N/A')}, 当前={drift_result.get('sample_size', {}).get('current', 'N/A')}")
        
        print(f"\n📋 当前策略配置: {mlops_status['current_policies']}")
        print(f"🔄 灰度上线状态: {mlops_status['gray_release_config']}")
    
    # 12. 演示参数自适应功能
    print("\n\n🚀 第12步：演示参数自适应功能")
    print("-" * 40)
    
    # 计算安全库存，使用参数自适应
    for i, product_id in enumerate(list(sample_data.keys())[:2]):
        # 获取历史数据
        product_data = sample_data[product_id]
        
        # 计算提前期需求标准差
        demand_std = product_data['demand'].std()
        lead_time_demand_std = demand_std * np.sqrt(7)  # 假设提前期为7天
        
        # 计算安全库存，使用参数自适应
        safety_stock = system.calculate_safety_stock(
            lead_time_demand_std=lead_time_demand_std,
            service_level=0.95,
            historical_data=product_data['demand'].values,
            product_id=product_id
        )
        print(f"📏 产品 {product_id} 的安全库存: {safety_stock:.2f}")
    
    # 结束信息
    print("\n" + "=" * 60)
    print("🎉 供应链智能补货系统演示完成！")
    print("=" * 60)
    print("📌 系统提供以下功能：")
    print("   • 需求预测和模型自动选择")
    print("   • MILP优化和最优订货量计算")
    print("   • 多仓库调拨和成本优化")
    print("   • 自动补单和审批流程")
    print("   • 模型性能监控和漂移检测")
    print("   • 基于FastAPI的Power BI数据接口")
    print("=" * 60)
    
    # 询问用户是否查看数据仪表盘
    show_dashboard = input("\n是否查看数据仪表盘？(y/n): ")
    if show_dashboard.lower() == 'y':
        from src.dashboard import DataDashboard
        import pandas as pd
        import os
        
        print("正在生成数据仪表盘...")
        dashboard = DataDashboard()
        
        # 加载库存数据（如果存在）
        inventory_file = "inventory_daily.csv"
        inventory_data = dashboard.load_data(inventory_file)
        if inventory_data is not None:
            # 转换日期格式
            inventory_data['date'] = pd.to_datetime(inventory_data['date'])
            # 按产品ID分组，可视化每个产品的库存水平
            for product_id in inventory_data['item_id'].unique()[:3]:  # 只显示前3个产品
                product_inventory = inventory_data[inventory_data['item_id'] == product_id]
                dashboard.visualize_inventory_levels(product_inventory, product_id)
        
        # 加载采购订单数据（如果存在）
        orders_file = "purchase_orders.csv"
        orders_data = dashboard.load_data(orders_file)
        if orders_data is not None:
            dashboard.visualize_purchase_orders(orders_data)
        
        # 可视化模型性能
        for product_id in [1, 2, 3, 4, 5]:
            metrics = dashboard.load_metrics(product_id)
            if metrics:
                dashboard.visualize_model_performance(product_id, metrics)
        
        # 显示图表
        print("正在显示数据仪表盘...")
        dashboard.show()
    else:
        print("已跳过数据仪表盘查看。")

if __name__ == "__main__":
    main()
