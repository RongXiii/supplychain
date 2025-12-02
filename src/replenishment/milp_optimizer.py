from ortools.linear_solver import pywraplp
import numpy as np

class MILPOptimizer:
    """
    MILP模型优化器，使用OR-Tools实现，用于计算最优的订货策略和成本优化
    支持智能求解器选择和约束条件优化
    """
    
    def __init__(self):
        self.solver = None
        self.num_products = 0
        self.num_periods = 0
        self.variables = {}
        self.solver_stats = {}  # 保存求解器性能统计，用于智能选择
    
    def _select_solver(self, problem_size):
        """
        根据问题规模智能选择最合适的求解器
        
        Args:
            problem_size: 问题规模描述，包含产品数量和时间段数量
        
        Returns:
            solver: 选择的OR-Tools求解器实例
        """
        num_products, num_periods = problem_size
        total_variables = num_products * num_periods * 4  # 估计变量数量
        
        # 根据问题规模和历史性能选择求解器
        if total_variables < 1000:
            # 小规模问题，优先使用速度快的求解器
            preferred_solvers = ['CBC', 'SCIP']
        elif total_variables < 10000:
            # 中等规模问题，平衡速度和求解质量
            preferred_solvers = ['SCIP', 'CBC', 'GLPK']
        else:
            # 大规模问题，优先使用求解质量高的求解器
            preferred_solvers = ['SCIP', 'GUROBI', 'CPLEX']
        
        # 尝试创建求解器
        for solver_name in preferred_solvers:
            try:
                solver = pywraplp.Solver.CreateSolver(solver_name)
                if solver:
                    print(f"成功创建求解器: {solver_name}")
                    return solver
            except Exception as e:
                print(f"创建求解器 {solver_name} 失败: {e}")
                continue
        
        # 如果所有求解器都失败，返回None
        print("无法创建任何求解器")
        return None
    
    def create_model(self, forecast_demands, current_inventory, lead_times, costs, constraints):
        """
        创建增强版MILP模型，支持智能求解器选择和约束条件优化
        
        Args:
            forecast_demands: 预测需求 [产品数量 x 时间段]
            current_inventory: 当前库存 [产品数量]
            lead_times: 交货提前期 [产品数量]
            costs: 成本参数，包括订货成本、持有成本、缺货成本等
            constraints: 约束条件，包括最大订货量、最小订货量等
        
        Returns:
            bool: 模型创建是否成功
        """
        # 获取问题规模
        self.num_products = len(forecast_demands)
        self.num_periods = len(forecast_demands[0])
        problem_size = (self.num_products, self.num_periods)
        
        # 智能选择求解器
        self.solver = self._select_solver(problem_size)
        if not self.solver:
            return False
        
        # 获取产品数量和时间段数量
        self.num_products = len(forecast_demands)
        self.num_periods = len(forecast_demands[0])
        
        # 保存参数到实例变量，供extract_results使用
        self.costs = costs
        self.lead_times = lead_times
        
        # 添加调试信息
        print(f"创建模型: 产品数={self.num_products}, 时期数={self.num_periods}")
        print(f"预测需求形状: ({self.num_products}, {self.num_periods})")
        print(f"当前库存: {current_inventory}")
        print(f"提前期: {lead_times}")
        
        # 定义变量
        variables = {}
        
        # 订货量：连续变量
        variables['order_quantity'] = {}
        for p in range(self.num_products):
            for t in range(self.num_periods):
                var_name = f"order_quantity_{p}_{t}"
                variables['order_quantity'][(p, t)] = self.solver.NumVar(0, constraints['max_order_quantity'][p], var_name)
        
        # 库存水平：连续变量
        variables['inventory_level'] = {}
        for p in range(self.num_products):
            for t in range(self.num_periods):
                var_name = f"inventory_level_{p}_{t}"
                variables['inventory_level'][(p, t)] = self.solver.NumVar(0, constraints['max_inventory'][p], var_name)
        
        # 缺货量：连续变量
        variables['shortage_quantity'] = {}
        for p in range(self.num_products):
            for t in range(self.num_periods):
                var_name = f"shortage_quantity_{p}_{t}"
                variables['shortage_quantity'][(p, t)] = self.solver.NumVar(0, self.solver.infinity(), var_name)
        
        # 补货决策变量：二元变量，表示是否在该时期订货（用于固定订货成本）
        variables['order_decision'] = {}
        for p in range(self.num_products):
            for t in range(self.num_periods):
                var_name = f"order_decision_{p}_{t}"
                variables['order_decision'][(p, t)] = self.solver.BoolVar(var_name)
        
        # 安全库存：连续变量
        variables['safety_stock'] = {}
        for p in range(self.num_products):
            for t in range(self.num_periods):
                var_name = f"safety_stock_{p}_{t}"
                variables['safety_stock'][(p, t)] = self.solver.NumVar(0, self.solver.infinity(), var_name)
        
        # 调拨量：连续变量，用于多仓库间的库存调拨
        if 'warehouse_transfer_costs' in costs:
            variables['transfer_quantity'] = {}
            num_warehouses = len(current_inventory[0]) if isinstance(current_inventory[0], list) else 1
            for p in range(self.num_products):
                for w1 in range(num_warehouses):
                    for w2 in range(num_warehouses):
                        if w1 != w2:
                            for t in range(self.num_periods):
                                var_name = f"transfer_quantity_{p}_{w1}_{w2}_{t}"
                                variables['transfer_quantity'][(p, w1, w2, t)] = self.solver.NumVar(0, self.solver.infinity(), var_name)
        
        self.variables = variables
        
        # 定义约束条件
        # 1. 库存平衡约束：期初库存 + 到货量 - 需求 + 缺货量 = 期末库存
        for p in range(self.num_products):
            for t in range(self.num_periods):
                # 期初库存
                if t == 0:
                    initial_inv = current_inventory[p]
                else:
                    initial_inv = variables['inventory_level'][(p, t-1)]
                
                # 计算到货量：考虑提前期
                incoming_order = 0
                if t >= lead_times[p]:
                    arrival_period = t - lead_times[p]
                    incoming_order = variables['order_quantity'][(p, arrival_period)]
                
                # 库存平衡：期初库存 + 到货量 - 需求 + 缺货量 = 期末库存
                self.solver.Add(
                    initial_inv + incoming_order - forecast_demands[p][t] + variables['shortage_quantity'][(p, t)] == 
                    variables['inventory_level'][(p, t)]
                )
        
        # 2. 订货量约束：上下限
        for p in range(self.num_products):
            for t in range(self.num_periods):
                min_order = constraints['min_order_quantity'][p]
                max_order = constraints['max_order_quantity'][p]
                order_qty = variables['order_quantity'][(p, t)]
                order_dec = variables['order_decision'][(p, t)]
                
                # 订货量必须小于等于最大订货量
                self.solver.Add(order_qty <= max_order)
                
                # 如果订货，订货量必须大于等于最小订货量；否则为0
                # 使用大M法实现逻辑约束
                M = max_order  # 大M值，设置为最大订货量
                self.solver.Add(order_qty >= min_order * order_dec)
                self.solver.Add(order_qty <= M * order_dec)
        
        # 3. 最大库存约束
        if 'max_inventory' in constraints:
            for p in range(self.num_products):
                for t in range(self.num_periods):
                    self.solver.Add(
                        variables['inventory_level'][(p, t)] <= constraints['max_inventory'][p]
                    )
        
        # 4. 安全库存约束
        if 'service_level' in constraints:
            for p in range(self.num_products):
                for t in range(self.num_periods):
                    # 基于服务水平的安全库存约束
                    # 期末库存必须大于等于安全库存目标
                    safety_stock_target = constraints.get('safety_stock_target', [0.1] * self.num_products)[p]
                    self.solver.Add(
                        variables['inventory_level'][(p, t)] >= safety_stock_target * forecast_demands[p][t]
                    )
        
        # 5. 最小库存周转率约束
        if 'min_inventory_turnover' in constraints:
            for p in range(self.num_products):
                # 计算总销售额
                total_sales = self.solver.Sum(forecast_demands[p][t] for t in range(self.num_periods))
                # 计算平均库存
                avg_inventory = self.solver.Sum(variables['inventory_level'][(p, t)] for t in range(self.num_periods)) / self.num_periods
                # 库存周转率 = 销售额 / 平均库存 >= 最小库存周转率
                self.solver.Add(total_sales >= constraints['min_inventory_turnover'][p] * avg_inventory)
        
        # 6. 资源约束（如仓库容量、运输能力）
        if 'resource_constraints' in constraints:
            for t in range(self.num_periods):
                # 总订货量不能超过运输能力
                if 'max_shipping_capacity' in constraints['resource_constraints']:
                    self.solver.Add(
                        self.solver.Sum(variables['order_quantity'][(p, t)] for p in range(self.num_products)) <= 
                        constraints['resource_constraints']['max_shipping_capacity']
                    )
        
        # 7. 预算约束
        if 'budget_constraint' in constraints:
            for t in range(self.num_periods):
                # 总成本 = 订货成本 + 运输成本
                total_cost = self.solver.Sum(
                    (costs['ordering_cost'][p] + costs.get('shipping_cost', [0] * self.num_products)[p]) * variables['order_quantity'][(p, t)] + 
                    costs.get('fixed_order_cost', [0] * self.num_products)[p] * variables['order_decision'][(p, t)]
                    for p in range(self.num_products)
                )
                self.solver.Add(total_cost <= constraints['budget_constraint'])
        
        # 8. 多仓调拨约束
        if 'warehouse_transfer_costs' in costs:
            num_warehouses = len(current_inventory[0]) if isinstance(current_inventory[0], list) else 1
            for p in range(self.num_products):
                for t in range(self.num_periods):
                    # 调拨后库存平衡约束
                    if t == 0:
                        initial_inv = current_inventory[p][0] if isinstance(current_inventory[p], list) else current_inventory[p]
                    else:
                        initial_inv = variables['inventory_level'][(p, t-1)]
                    
                    # 计算总调拨量（调入 - 调出）
                    total_transfer_in = 0
                    total_transfer_out = 0
                    for w1 in range(num_warehouses):
                        for w2 in range(num_warehouses):
                            if w1 != w2:
                                transfer_var = variables['transfer_quantity'].get((p, w1, w2, t), 0)
                                if w2 == 0:  # 调入主仓库
                                    total_transfer_in += transfer_var
                                elif w1 == 0:  # 调出主仓库
                                    total_transfer_out += transfer_var
                    
                    # 库存平衡：期初库存 + 调拨净量 = 调整后库存
                    if total_transfer_in > 0 or total_transfer_out > 0:
                        self.solver.Add(
                            initial_inv + total_transfer_in - total_transfer_out == variables['inventory_level'][(p, t)]
                        )
        
        # 定义目标函数：最小化总成本
        objective = self.solver.Objective()
        objective.SetMinimization()
        
        for p in range(self.num_products):
            for t in range(self.num_periods):
                # 1. 订货成本：固定订货成本 + 可变订货成本
                fixed_order_cost = costs.get('fixed_order_cost', [0] * self.num_products)[p]
                variable_order_cost = costs['ordering_cost'][p]
                objective.SetCoefficient(variables['order_decision'][(p, t)], fixed_order_cost)
                objective.SetCoefficient(variables['order_quantity'][(p, t)], variable_order_cost)
                
                # 2. 持有成本：基于库存水平
                holding_cost = costs['holding_cost'][p]
                objective.SetCoefficient(variables['inventory_level'][(p, t)], holding_cost)
                
                # 3. 缺货成本：基于缺货量
                shortage_cost = costs['shortage_cost'][p]
                objective.SetCoefficient(variables['shortage_quantity'][(p, t)], shortage_cost)
                
                # 4. 运输成本：基于订货量
                shipping_cost = costs.get('shipping_cost', [0] * self.num_products)[p]
                objective.SetCoefficient(variables['order_quantity'][(p, t)], shipping_cost)
                
                # 5. 存储成本：基于库存水平
                storage_cost = costs.get('storage_cost', [0] * self.num_products)[p]
                objective.SetCoefficient(variables['inventory_level'][(p, t)], storage_cost)
        
        # 6. 调拨成本：多仓库间的调拨成本
        if 'warehouse_transfer_costs' in costs:
            num_warehouses = len(current_inventory[0]) if isinstance(current_inventory[0], list) else 1
            for p in range(self.num_products):
                for w1 in range(num_warehouses):
                    for w2 in range(num_warehouses):
                        if w1 != w2:
                            for t in range(self.num_periods):
                                transfer_cost = costs['warehouse_transfer_costs'][p][w1][w2]
                                objective.SetCoefficient(variables['transfer_quantity'][(p, w1, w2, t)], transfer_cost)
        
        # 7. 数量折扣：如果订货量达到一定阈值，给予成本优惠
        if 'quantity_discounts' in costs:
            for p in range(self.num_products):
                discounts = costs['quantity_discounts'][p]
                if discounts:
                    # 按最小订货量排序折扣
                    discounts.sort(key=lambda x: x['min_quantity'])
                    num_discounts = len(discounts)
                    
                    for t in range(self.num_periods):
                        order_qty = variables['order_quantity'][(p, t)]
                        # 创建折扣选择变量
                        discount_vars = []
                        for i in range(num_discounts):
                            discount_var = self.solver.BoolVar(f"discount_{p}_{t}_{i}")
                            discount_vars.append(discount_var)
                        
                        # 只能选择一个折扣级别
                        self.solver.Add(sum(discount_vars) <= 1)
                        
                        # 应用折扣约束
                        for i in range(num_discounts):
                            min_qty = discounts[i]['min_quantity']
                            discount_rate = discounts[i]['discount_rate']
                            discount_var = discount_vars[i]
                            
                            # 如果选择该折扣，订货量必须 >= min_qty
                            if i > 0:
                                prev_min_qty = discounts[i-1]['min_quantity']
                                # 对于非第一个折扣，订货量必须在当前折扣的最小量和下一个折扣的最小量之间
                                if i < num_discounts - 1:
                                    next_min_qty = discounts[i+1]['min_quantity']
                                    self.solver.Add(order_qty <= next_min_qty * (1 - discount_vars[i+1]))
                            
                            # 基本约束：如果选择该折扣，订货量必须 >= min_qty
                            self.solver.Add(order_qty >= min_qty * discount_var)
                            
                            # 计算折扣成本节约
                            cost_saving = variable_order_cost * discount_rate
                            # 将折扣成本节约添加到目标函数（作为负成本）
                            # 使用辅助变量来处理乘积
                            aux_var = self.solver.NumVar(0, self.solver.infinity(), f"aux_discount_{p}_{t}_{i}")
                            M = constraints['max_order_quantity'][p]
                            self.solver.Add(aux_var <= M * discount_var)
                            self.solver.Add(aux_var <= order_qty)
                            self.solver.Add(aux_var >= order_qty - M * (1 - discount_var))
                            self.solver.Add(aux_var >= 0)
                            
                            # 将辅助变量添加到目标函数
                            objective.SetCoefficient(aux_var, -cost_saving)
        
        return True
    
    def _set_solver_parameters(self, solver_name):
        """
        根据求解器类型设置最优参数
        
        Args:
            solver_name: 求解器名称
        """
        # 根据求解器类型设置特定参数
        if 'SCIP' in solver_name:
            # SCIP特定参数
            self.solver.SetSolverSpecificParametersAsString(
                'limits/time=300\n'       # 时间限制300秒
                'presolving/maxrounds=50\n'  # 预求解轮次
                'presolving/maxrestarts=10\n'  # 预求解重启次数
                'heuristics/guideddiving/freq=10\n'  # 启发式算法频率
                'branching/relpscost/priority=1000000\n'  # 分支优先级
            )
        elif 'CBC' in solver_name:
            # CBC特定参数
            self.solver.SetSolverSpecificParametersAsString(
                'seconds=300\n'       # 时间限制300秒
                'ratioGap=0.01\n'    # 相对间隙1%
                'maxNodes=1000000\n' # 最大节点数
                'logLevel=0\n'       # 日志级别
            )
        elif 'GUROBI' in solver_name:
            # GUROBI特定参数
            self.solver.SetSolverSpecificParametersAsString(
                'TimeLimit=300\n'     # 时间限制300秒
                'MIPGap=0.001\n'     # 相对间隙0.1%
                'Heuristics=0.5\n'   # 启发式算法强度
                'Presolve=2\n'       # 预求解强度
                'Cuts=2\n'           # 割平面强度
            )
    
    def solve_model(self):
        """
        求解MILP模型，支持智能参数调整和结果分析
        
        Returns:
            results: 求解结果，包括最优订货策略、成本、求解统计信息等
        """
        if not self.solver:
            print("模型未创建，请先调用create_model方法")
            return None
        
        # 获取求解器名称
        solver_name = self.solver.SolverVersion().split()[0] if hasattr(self.solver, 'SolverVersion') else 'Unknown'
        
        # 设置求解器参数
        self._set_solver_parameters(solver_name)
        
        # 记录求解开始时间
        import time
        start_time = time.time()
        
        # 求解模型
        print(f"使用求解器 {self.solver.SolverVersion()}")
        status = self.solver.Solve()
        
        # 记录求解结束时间
        solve_time = time.time() - start_time
        
        # 状态码解释
        status_dict = {
            pywraplp.Solver.OPTIMAL: "找到最优解",
            pywraplp.Solver.FEASIBLE: "找到可行解，但不一定是最优解",
            pywraplp.Solver.INFEASIBLE: "模型不可行，没有满足所有约束的解",
            pywraplp.Solver.UNBOUNDED: "目标函数无界",
            pywraplp.Solver.ABNORMAL: "求解过程异常终止",
            pywraplp.Solver.NOT_SOLVED: "模型尚未求解"
        }
        
        # 准备求解统计信息
        solver_stats = {
            'solver_name': solver_name,
            'solve_time': solve_time,
            'objective_value': self.solver.Objective().Value() if status in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE] else float('inf'),
            'status_code': status,
            'status_message': status_dict.get(status, f"未知状态码: {status}")
        }
        
        # 检查求解结果
        if status in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
            print(f"模型求解成功，{status_dict[status]}，求解时间: {solve_time:.2f}秒")
            results = self.extract_results()
            results['solver_stats'] = solver_stats
            
            # 保存求解器统计信息，用于下次智能选择
            self.solver_stats = solver_stats
            
            return results
        else:
            status_msg = status_dict.get(status, f"未知状态码: {status}")
            print(f"模型求解失败: {status_msg}")
            return {'status': 'failed', 'solver_stats': solver_stats}
    
    def extract_results(self):
        """
        提取求解结果
        
        Returns:
            results: 求解结果字典，包括OptimalPlan所需的所有信息
        """
        if not self.solver:
            return None
        
        results = {
            'total_cost': self.solver.Objective().Value(),
            'status': 'optimal' if self.solver.GetStatus() == pywraplp.Solver.OPTIMAL else 'feasible',
            'order_quantities': {},
            'inventory_levels': {},
            'shortage_quantities': {},
            'discount_selections': {},
            'expected_arrival_dates': {},
            'optimal_plan': [],
            'cost_breakdown': {
                'ordering_cost': 0,
                'holding_cost': 0,
                'shortage_cost': 0,
                'shipping_cost': 0,
                'storage_cost': 0,
                'transfer_cost': 0
            }
        }
        
        # 提取订货量和相关信息
        for p in range(self.num_products):
            results['order_quantities'][p] = []
            results['discount_selections'][p] = []
            results['expected_arrival_dates'][p] = []
            
            for t in range(self.num_periods):
                qty = self.variables['order_quantity'][(p, t)].SolutionValue()
                results['order_quantities'][p].append(qty)
                
                # 确定选择的价格阶梯
                # 这里简化处理，实际应根据折扣变量的解来确定
                selected_tier = 0
                if 'quantity_discounts' in self.costs:  # 假设costs已保存到实例变量
                    discounts = self.costs['quantity_discounts'][p]
                    for tier, discount_info in enumerate(discounts):
                        if qty >= discount_info['min_quantity']:
                            selected_tier = tier + 1  # 从1开始编号
                results['discount_selections'][p].append(selected_tier)
                
                # 计算期望到货日期（假设当前日期为t=0，提前期为lead_time）
                expected_arrival = t + self.lead_times[p]  # 简化处理，实际应考虑具体日期
                results['expected_arrival_dates'][p].append(expected_arrival)
                
                # 构建OptimalPlan记录
                optimal_plan_entry = {
                    'product_id': p + 1,  # 产品ID从1开始
                    'period': t + 1,  # 时间段从1开始
                    'optimal_order_qty': qty,
                    'price_tier': selected_tier,
                    'expected_arrival_date': expected_arrival,
                    'inventory_level': self.variables['inventory_level'][(p, t)].SolutionValue(),
                    'shortage_quantity': self.variables['shortage_quantity'][(p, t)].SolutionValue(),
                    'order_decision': self.variables['order_decision'][(p, t)].SolutionValue()
                }
                results['optimal_plan'].append(optimal_plan_entry)
        
        # 提取库存水平
        for p in range(self.num_products):
            results['inventory_levels'][p] = []
            for t in range(self.num_periods):
                level = self.variables['inventory_level'][(p, t)].SolutionValue()
                results['inventory_levels'][p].append(level)
                
                # 累加持有成本
                if 'holding_cost' in self.costs:
                    holding_cost = self.costs['holding_cost'][p] * level
                    results['cost_breakdown']['holding_cost'] += holding_cost
                
                # 累加存储成本
                if 'storage_cost' in self.costs:
                    storage_cost = self.costs['storage_cost'][p] * level
                    results['cost_breakdown']['storage_cost'] += storage_cost
        
        # 提取缺货量和相关成本
        for p in range(self.num_products):
            results['shortage_quantities'][p] = []
            for t in range(self.num_periods):
                qty = self.variables['shortage_quantity'][(p, t)].SolutionValue()
                results['shortage_quantities'][p].append(qty)
                
                # 累加缺货成本
                if 'shortage_cost' in self.costs:
                    shortage_cost = self.costs['shortage_cost'][p] * qty
                    results['cost_breakdown']['shortage_cost'] += shortage_cost
        
        # 计算订货成本
        for p in range(self.num_products):
            for t in range(self.num_periods):
                order_qty = self.variables['order_quantity'][(p, t)].SolutionValue()
                order_decision = self.variables['order_decision'][(p, t)].SolutionValue()
                
                # 可变订货成本
                if 'ordering_cost' in self.costs:
                    ordering_cost = self.costs['ordering_cost'][p] * order_qty
                    results['cost_breakdown']['ordering_cost'] += ordering_cost
                
                # 固定订货成本
                if 'fixed_order_cost' in self.costs:
                    fixed_order_cost = self.costs['fixed_order_cost'][p] * order_decision
                    results['cost_breakdown']['ordering_cost'] += fixed_order_cost
                
                # 运输成本
                if 'shipping_cost' in self.costs:
                    shipping_cost = self.costs['shipping_cost'][p] * order_qty
                    results['cost_breakdown']['shipping_cost'] += shipping_cost
        
        return results
    
    def generate_purchase_orders(self, results, current_period):
        """
        根据求解结果生成采购订单
        
        Args:
            results: 求解结果
            current_period: 当前时间段
            
        Returns:
            purchase_orders: 采购订单列表
        """
        purchase_orders = []
        
        for product_id, order_quantities in results['order_quantities'].items():
            order_qty = order_quantities[current_period - 1]
            if order_qty > 0.01:  # 允许微小误差
                purchase_order = {
                    'product_id': product_id + 1,
                    'order_quantity': order_qty,
                    'order_period': current_period,
                    'status': 'pending'
                }
                purchase_orders.append(purchase_order)
        
        return purchase_orders
    
    def optimize(self, forecast_demands, current_inventory, lead_times, costs, constraints):
        """
        完整的优化流程：创建模型、求解模型、生成采购订单
        
        Args:
            forecast_demands: 预测需求 [产品数量 x 时间段]
            current_inventory: 当前库存 [产品数量]
            lead_times: 交货提前期 [产品数量]
            costs: 成本参数
            constraints: 约束条件
            
        Returns:
            results: 优化结果，包括最优订货策略、成本、采购订单等
        """
        # 记录优化开始时间
        import time
        start_time = time.time()
        
        # 创建模型
        model_created = self.create_model(forecast_demands, current_inventory, lead_times, costs, constraints)
        if not model_created:
            return None
        
        # 求解模型
        results = self.solve_model()
        
        if results is not None:
            # 生成当前时间段的采购订单
            purchase_orders = self.generate_purchase_orders(results, current_period=1)
            results['purchase_orders'] = purchase_orders
            
            # 计算总优化时间
            total_time = time.time() - start_time
            results['total_optimization_time'] = total_time
            print(f"完整优化流程完成，总耗时: {total_time:.2f}秒")
        
        return results
