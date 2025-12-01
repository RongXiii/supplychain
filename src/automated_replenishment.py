class ApprovalWorkflow:
    """
    审批工作流类，用于处理采购订单的审批流程
    """
    
    def __init__(self):
        self.approval_levels = {
            'low': {'amount_threshold': 1000, 'approver_role': 'buyer'},
            'medium': {'amount_threshold': 5000, 'approver_role': 'manager'},
            'high': {'amount_threshold': float('inf'), 'approver_role': 'admin'}
        }
        self.approval_history = []
    
    def determine_approval_level(self, order_amount):
        """
        根据订单金额确定审批级别
        
        Args:
            order_amount: 订单金额
            
        Returns:
            approval_level: 审批级别
        """
        if order_amount < self.approval_levels['low']['amount_threshold']:
            return 'low'
        elif order_amount < self.approval_levels['medium']['amount_threshold']:
            return 'medium'
        else:
            return 'high'
    
    def get_required_approver(self, order_amount):
        """
        获取所需审批人角色
        
        Args:
            order_amount: 订单金额
            
        Returns:
            approver_role: 审批人角色
        """
        approval_level = self.determine_approval_level(order_amount)
        return self.approval_levels[approval_level]['approver_role']
    
    def submit_for_approval(self, purchase_order, requester_role):
        """
        提交采购订单进行审批
        
        Args:
            purchase_order: 采购订单
            requester_role: 请求人角色
            
        Returns:
            approval_status: 审批状态
        """
        order_amount = purchase_order['order_quantity'] * purchase_order.get('unit_price', 100)  # 简化计算
        required_approver = self.get_required_approver(order_amount)
        
        # 记录审批历史
        approval_record = {
            'order_id': purchase_order['order_id'],
            'order_amount': order_amount,
            'approval_level': self.determine_approval_level(order_amount),
            'required_approver': required_approver,
            'requester_role': requester_role,
            'status': 'pending',
            'submitted_at': '2023-01-01 12:00:00'  # 简化处理，实际应使用当前时间
        }
        self.approval_history.append(approval_record)
        
        # 如果请求人角色高于或等于所需审批人角色，则自动批准
        role_hierarchy = {'buyer': 1, 'manager': 2, 'admin': 3}
        if role_hierarchy.get(requester_role, 0) >= role_hierarchy[required_approver]:
            return self.approve_order(purchase_order['order_id'], requester_role)
        
        return {'status': 'pending', 'message': f'等待{required_approver}审批'}
    
    def approve_order(self, order_id, approver_role):
        """
        批准采购订单
        
        Args:
            order_id: 订单ID
            approver_role: 审批人角色
            
        Returns:
            approval_status: 审批状态
        """
        for record in self.approval_history:
            if record['order_id'] == order_id and record['status'] == 'pending':
                # 检查审批人角色是否符合要求
                if approver_role == record['required_approver'] or \
                   {'buyer': 1, 'manager': 2, 'admin': 3}[approver_role] > \
                   {'buyer': 1, 'manager': 2, 'admin': 3}[record['required_approver']]:
                    record['status'] = 'approved'
                    record['approved_by'] = approver_role
                    record['approved_at'] = '2023-01-01 12:00:00'  # 简化处理
                    return {'status': 'approved', 'message': f'订单已被{approver_role}批准'}
                else:
                    return {'status': 'rejected', 'message': '审批人角色不符合要求'}
        return {'status': 'not_found', 'message': '未找到待审批订单'}
    
    def reject_order(self, order_id, approver_role, reason):
        """
        拒绝采购订单
        
        Args:
            order_id: 订单ID
            approver_role: 审批人角色
            reason: 拒绝原因
            
        Returns:
            approval_status: 审批状态
        """
        for record in self.approval_history:
            if record['order_id'] == order_id and record['status'] == 'pending':
                # 检查审批人角色是否符合要求
                if approver_role == record['required_approver'] or \
                   {'buyer': 1, 'manager': 2, 'admin': 3}[approver_role] > \
                   {'buyer': 1, 'manager': 2, 'admin': 3}[record['required_approver']]:
                    record['status'] = 'rejected'
                    record['rejected_by'] = approver_role
                    record['rejected_at'] = '2023-01-01 12:00:00'  # 简化处理
                    record['rejection_reason'] = reason
                    return {'status': 'rejected', 'message': f'订单已被{approver_role}拒绝，原因：{reason}'}
                else:
                    return {'status': 'error', 'message': '审批人角色不符合要求'}
        return {'status': 'not_found', 'message': '未找到待审批订单'}
    
    def get_approval_history(self, order_id=None):
        """
        获取审批历史
        
        Args:
            order_id: 订单ID（可选）
            
        Returns:
            approval_history: 审批历史
        """
        if order_id:
            return [record for record in self.approval_history if record['order_id'] == order_id]
        return self.approval_history


class AutomatedReplenishment:
    """
    自动补单模块，用于处理自动补货逻辑和审批流程
    """
    
    def __init__(self, replenishment_system):
        self.replenishment_system = replenishment_system
        self.approval_workflow = ApprovalWorkflow()
        self.api_clients = {}
        self.purchase_orders = []
        self.next_order_id = 1
    
    def integrate_external_api(self, api_name, client_config):
        """
        集成外部API
        
        Args:
            api_name: API名称
            client_config: 客户端配置
            
        Returns:
            integration_status: 集成状态
        """
        # 简化实现，实际应初始化API客户端
        self.api_clients[api_name] = client_config
        return {'status': 'success', 'message': f'已集成{api_name} API'}
    
    def execute_auto_replenishment(self, strategy='hybrid', requester_role='buyer'):
        """
        执行自动补单
        
        Args:
            strategy: 补货策略（'hybrid', 'rop', 'order_up_to'）
            requester_role: 请求人角色
            
        Returns:
            replenishment_result: 补单结果
        """
        print(f"执行自动补单，策略：{strategy}")
        
        # 根据策略生成补货建议
        replenishment_suggestions = self.replenishment_system.execute_replenishment_strategy(strategy)
        
        # 生成采购订单并提交审批
        results = []
        for suggestion in replenishment_suggestions:
            if suggestion['need_replenishment']:
                purchase_order = self.generate_purchase_order(suggestion)
                approval_result = self.approval_workflow.submit_for_approval(purchase_order, requester_role)
                purchase_order['approval_status'] = approval_result['status']
                purchase_order['approval_message'] = approval_result['message']
                self.purchase_orders.append(purchase_order)
                results.append(purchase_order)
        
        return {
            'total_suggestions': len(replenishment_suggestions),
            'generated_orders': len(results),
            'orders': results
        }
    
    def generate_purchase_order(self, replenishment_suggestion):
        """
        生成采购订单
        
        Args:
            replenishment_suggestion: 补货建议
            
        Returns:
            purchase_order: 采购订单
        """
        order_id = self.next_order_id
        self.next_order_id += 1
        
        purchase_order = {
            'order_id': order_id,
            'product_id': replenishment_suggestion['product_id'],
            'product_name': replenishment_suggestion.get('product_name', f'产品{replenishment_suggestion["product_id"]}'),
            'order_quantity': replenishment_suggestion['suggested_order_qty'],
            'unit_price': replenishment_suggestion.get('unit_price', 100),  # 简化处理
            'total_amount': replenishment_suggestion['suggested_order_qty'] * replenishment_suggestion.get('unit_price', 100),
            'order_date': '2023-01-01',  # 简化处理
            'expected_delivery_date': '2023-01-15',  # 简化处理
            'status': 'pending_approval',
            'replenishment_strategy': replenishment_suggestion.get('strategy', 'hybrid'),
            'reason': replenishment_suggestion.get('reason', '自动补单'),
            'safety_stock': replenishment_suggestion.get('safety_stock', 0),
            'reorder_point': replenishment_suggestion.get('reorder_point', 0),
            'current_inventory': replenishment_suggestion.get('current_inventory', 0)
        }
        
        return purchase_order
    
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
        if action == 'approve':
            result = self.approval_workflow.approve_order(order_id, approver_role)
        elif action == 'reject':
            result = self.approval_workflow.reject_order(order_id, approver_role, reason)
        else:
            return {'status': 'error', 'message': '无效的审批动作'}
        
        # 更新采购订单状态
        for order in self.purchase_orders:
            if order['order_id'] == order_id:
                order['approval_status'] = result['status']
                order['approval_message'] = result['message']
                if result['status'] == 'approved':
                    order['status'] = 'approved'
                elif result['status'] == 'rejected':
                    order['status'] = 'rejected'
        
        return result
    
    def get_purchase_orders(self, status=None):
        """
        获取采购订单
        
        Args:
            status: 订单状态（可选）
            
        Returns:
            purchase_orders: 采购订单列表
        """
        if status:
            return [order for order in self.purchase_orders if order['status'] == status]
        return self.purchase_orders
    
    def sync_with_erp(self, order_ids):
        """
        与ERP系统同步采购订单
        
        Args:
            order_ids: 订单ID列表
            
        Returns:
            sync_result: 同步结果
        """
        # 简化实现，实际应调用ERP API
        synced_orders = [order for order in self.purchase_orders if order['order_id'] in order_ids]
        return {
            'status': 'success',
            'message': f'已同步{len(synced_orders)}个订单到ERP系统',
            'synced_orders': synced_orders
        }
    
    def get_system_status(self):
        """
        获取自动补单系统状态
        
        Returns:
            system_status: 系统状态
        """
        return {
            'total_orders': len(self.purchase_orders),
            'pending_approval': len([order for order in self.purchase_orders if order['status'] == 'pending_approval']),
            'approved': len([order for order in self.purchase_orders if order['status'] == 'approved']),
            'rejected': len([order for order in self.purchase_orders if order['status'] == 'rejected']),
            'integrated_apis': list(self.api_clients.keys()),
            'approval_history_count': len(self.approval_workflow.get_approval_history())
        }
