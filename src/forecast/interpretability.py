import numpy as np
import pandas as pd
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
import shap
import lime
import lime.lime_tabular
import json
import os
import uuid
from datetime import datetime

class ModelInterpreter:
    """
    模型解释器，集成多种解释工具
    - SHAP
    - LIME
    - Partial Dependence Plots
    - 特征重要性分析
    - 决策路径可视化
    """
    
    def __init__(self, model_dir='interpretations', fig_dir='interpretations/figures'):
        self.model_dir = model_dir
        self.fig_dir = fig_dir
        
        # 创建必要的目录
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.fig_dir):
            os.makedirs(self.fig_dir)
    
    def generate_explanation_id(self):
        """生成唯一的解释ID"""
        return f"explain_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}"
    
    def save_interpretation(self, explanation_data, explanation_id=None):
        """保存解释结果到文件"""
        if explanation_id is None:
            explanation_id = self.generate_explanation_id()
        
        file_path = os.path.join(self.model_dir, f"{explanation_id}.json")
        with open(file_path, 'w') as f:
            json.dump(explanation_data, f, indent=2, default=str)
        
        return explanation_id, file_path
    
    def load_interpretation(self, explanation_id):
        """加载解释结果"""
        file_path = os.path.join(self.model_dir, f"{explanation_id}.json")
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def _is_shap_compatible(self, model):
        """检查模型是否与SHAP兼容
        
        参数:
            model: 机器学习模型对象
            
        返回:
            bool: 模型是否兼容SHAP
        """
        model_type = type(model).__name__
        
        # 检查是否为树模型或线性模型
        tree_models = ['RandomForestRegressor', 'RandomForestClassifier', 
                      'XGBRegressor', 'XGBClassifier', 
                      'LGBMRegressor', 'LGBMClassifier',
                      'GradientBoostingRegressor', 'GradientBoostingClassifier',
                      'DecisionTreeRegressor', 'DecisionTreeClassifier']
        
        linear_models = ['LinearRegression', 'LogisticRegression',
                        'Ridge', 'Lasso', 'ElasticNet']
        
        if model_type in tree_models:
            return True
        elif model_type in linear_models:
            return True
        elif hasattr(model, 'predict'):
            # 对于其他类型的模型，尝试使用KernelExplainer
            return True
        else:
            return False
    
    def get_shap_explanation(self, model, X, feature_names=None, sample_size=100):
        """
        使用SHAP生成模型解释
        
        Args:
            model: 训练好的模型
            X: 特征数据（DataFrame或numpy数组）
            feature_names: 特征名称列表
            sample_size: 用于解释的样本大小
            
        Returns:
            shap_values: SHAP值
            expected_value: 预期值
            shap_explainer: SHAP解释器
        """
        # 转换为DataFrame
        if not isinstance(X, pd.DataFrame):
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=feature_names)
        
        # 采样数据
        if len(X) > sample_size:
            X_sample = X.sample(sample_size, random_state=42)
        else:
            X_sample = X
        
        try:
            # 尝试使用TreeExplainer（适用于树模型）
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
        except Exception:
            try:
                # 尝试使用KernelExplainer（适用于所有模型）
                explainer = shap.KernelExplainer(model.predict, X_sample)
                shap_values = explainer.shap_values(X_sample)
            except Exception:
                # 尝试使用LinearExplainer（适用于线性模型）
                explainer = shap.LinearExplainer(model, X_sample)
                shap_values = explainer.shap_values(X_sample)
        
        # 获取预期值
        expected_value = explainer.expected_value
        
        return shap_values, expected_value, explainer, X_sample
    
    def get_shap_summary_data(self, shap_values, X, feature_names=None):
        """
        获取SHAP摘要数据（JSON格式）
        
        Args:
            shap_values: SHAP值
            X: 特征数据
            feature_names: 特征名称列表
            
        Returns:
            data: JSON格式的SHAP摘要数据
        """
        # 转换为DataFrame
        if not isinstance(X, pd.DataFrame):
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=feature_names)
        else:
            feature_names = X.columns.tolist()
        
        # 计算每个特征的平均SHAP值绝对值（重要性）
        if isinstance(shap_values, list):
            # 多分类情况，取第一类
            shap_values = shap_values[0]
        
        # 计算每个特征的平均SHAP值和绝对值
        mean_shap = np.mean(shap_values, axis=0)
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        
        # 创建特征重要性数据
        shap_summary_data = []
        for i, feature in enumerate(feature_names):
            shap_summary_data.append({
                'feature': feature,
                'mean_shap': float(mean_shap[i]),
                'mean_abs_shap': float(mean_abs_shap[i]),
                'index': i
            })
        
        # 按平均绝对值排序
        shap_summary_data.sort(key=lambda x: x['mean_abs_shap'], reverse=True)
        
        return {
            'summary': shap_summary_data,
            'shap_values': shap_values.tolist(),
            'feature_names': feature_names,
            'sample_size': shap_values.shape[0]
        }
    
    def get_shap_force_data(self, explainer, expected_value, shap_values, X, instance_idx=0):
        """
        获取SHAP力图数据（JSON格式）
        
        Args:
            explainer: SHAP解释器
            expected_value: 预期值
            shap_values: SHAP值
            X: 特征数据
            instance_idx: 要解释的实例索引
            
        Returns:
            data: JSON格式的SHAP力图数据
        """
        # 转换为DataFrame
        if not isinstance(X, pd.DataFrame):
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=feature_names)
        
        # 获取单个实例的数据
        instance = X.iloc[instance_idx]
        
        # 处理SHAP值
        if isinstance(shap_values, list):
            # 多分类情况，取第一类
            instance_shap = shap_values[0][instance_idx]
        else:
            instance_shap = shap_values[instance_idx]
        
        # 计算预测值
        predicted_value = float(expected_value) + np.sum(instance_shap)
        
        # 创建特征贡献数据
        feature_contributions = []
        for i, feature in enumerate(X.columns):
            feature_contributions.append({
                'feature': feature,
                'value': float(instance.iloc[i]),
                'contribution': float(instance_shap[i]),
                'abs_contribution': abs(float(instance_shap[i]))
            })
        
        # 按贡献绝对值排序
        feature_contributions.sort(key=lambda x: x['abs_contribution'], reverse=True)
        
        return {
            'instance_idx': instance_idx,
            'expected_value': float(expected_value) if isinstance(expected_value, (int, float)) else expected_value.tolist(),
            'predicted_value': predicted_value,
            'feature_contributions': feature_contributions,
            'original_features': instance.to_dict()
        }
    
    def get_lime_explanation(self, model, X_train, X_test, feature_names=None, instance_idx=0, num_features=10):
        """
        使用LIME生成模型解释
        
        Args:
            model: 训练好的模型
            X_train: 训练数据
            X_test: 测试数据
            feature_names: 特征名称列表
            instance_idx: 要解释的实例索引
            num_features: 要显示的特征数量
            
        Returns:
            lime_explanation: LIME解释结果
        """
        # 转换为numpy数组
        if isinstance(X_train, pd.DataFrame):
            X_train_np = X_train.values
            feature_names = X_train.columns.tolist() if feature_names is None else feature_names
        else:
            X_train_np = X_train
            feature_names = [f"feature_{i}" for i in range(X_train_np.shape[1])] if feature_names is None else feature_names
        
        if isinstance(X_test, pd.DataFrame):
            X_test_np = X_test.values
        else:
            X_test_np = X_test
        
        # 创建LIME解释器
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train_np, 
            feature_names=feature_names,
            class_names=['demand'],
            mode='regression'
        )
        
        # 生成解释，添加错误处理
        try:
            exp = explainer.explain_instance(
                X_test_np[instance_idx], 
                model.predict, 
                num_features=num_features
            )
            return exp
        except Exception as e:
            print(f"生成LIME解释时出错: {e}")
            # 检查模型的predict方法输出
            try:
                test_pred = model.predict(X_test_np[instance_idx].reshape(1, -1))
                print(f"模型predict测试结果: {test_pred}")
            except Exception as pred_e:
                print(f"模型predict方法测试出错: {pred_e}")
            return None
    
    def get_lime_data(self, lime_explanation, feature_names=None):
        """
        获取LIME解释数据（JSON格式）
        
        Args:
            lime_explanation: LIME解释结果
            feature_names: 特征名称列表
            
        Returns:
            data: JSON格式的LIME解释数据
        """
        if lime_explanation is None:
            return None
        
        # 获取LIME解释的特征列表
        lime_features = lime_explanation.as_list()
        
        # 如果没有提供特征名称，则使用LIME默认的特征名
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(lime_features))]
        
        # 转换为结构化数据
        lime_data = {
            'predicted_value': float(lime_explanation.predict_proba[0] if hasattr(lime_explanation, 'predict_proba') else lime_explanation.predicted_value),
            'prediction_score': float(lime_explanation.score),
            'intercept': float(lime_explanation.intercept[0] if hasattr(lime_explanation.intercept, '__len__') else lime_explanation.intercept),
            'feature_importance': []
        }
        
        # 处理特征重要性
        for feature, weight in lime_features:
            # 解析特征名和值
            if '>' in feature:
                # 连续特征
                parts = feature.split(' > ')
                if len(parts) == 2:
                    feature_name = parts[0]
                    feature_value = float(parts[1])
                else:
                    feature_name = feature
                    feature_value = 0.0
            elif '<=' in feature:
                # 连续特征
                parts = feature.split(' <= ')
                if len(parts) == 2:
                    feature_name = parts[0]
                    feature_value = float(parts[1])
                else:
                    feature_name = feature
                    feature_value = 0.0
            elif '=' in feature:
                # 分类特征
                parts = feature.split(' = ')
                if len(parts) == 2:
                    feature_name = parts[0]
                    feature_value = parts[1]
                else:
                    feature_name = feature
                    feature_value = 'unknown'
            else:
                # 其他情况
                feature_name = feature
                feature_value = 'unknown'
            
            lime_data['feature_importance'].append({
                'feature': feature_name,
                'weight': float(weight),
                'value': feature_value,
                'abs_weight': abs(float(weight))
            })
        
        # 按权重绝对值排序
        lime_data['feature_importance'].sort(key=lambda x: x['abs_weight'], reverse=True)
        
        return lime_data
    
    def get_partial_dependence_data(self, model, X, features, feature_names=None):
        """
        获取部分依赖图(PDP)数据（JSON格式）
        
        Args:
            model: 训练好的模型
            X: 特征矩阵
            features: 特征索引或索引列表
            feature_names: 特征名称列表
            
        Returns:
            data: JSON格式的部分依赖图数据
        """
        try:
            # 转换为DataFrame
            if not isinstance(X, pd.DataFrame):
                if feature_names is None:
                    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
                X = pd.DataFrame(X, columns=feature_names)
            
            # 确保X不为空
            if X.empty:
                raise ValueError("输入数据X不能为空")
            
            # 确保features是列表
            if not isinstance(features, (list, tuple)):
                features = [features]
            
            # 确保特征有效
            if isinstance(features[0], str):
                # 转换特征名称为索引
                feature_indices = [X.columns.get_loc(f) for f in features]
            else:
                feature_indices = features
            
            pdp_data = []
            
            # 生成部分依赖图数据
            display = PartialDependenceDisplay.from_estimator(
                model,
                X,
                feature_indices,
                feature_names=feature_names,
                grid_resolution=100
            )
            
            # 提取数据
            for i, (feature_idx, ax) in enumerate(zip(feature_indices, display.axes_.flatten())):
                # 获取线对象
                line = ax.lines[0] if ax.lines else None
                if line:
                    x_values = line.get_xdata().tolist()
                    y_values = line.get_ydata().tolist()
                    
                    pdp_item = {
                        'feature_index': feature_idx,
                        'feature_name': feature_names[feature_idx],
                        'feature_values': [float(val) for val in x_values],
                        'pdp_values': [float(val) for val in y_values],
                        'min_pdp_value': float(min(y_values)),
                        'max_pdp_value': float(max(y_values))
                    }
                    pdp_data.append(pdp_item)
            
            return pdp_data
        except Exception as e:
            print(f"获取部分依赖图数据时出错: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_feature_importance(self, model, X, y, feature_names=None, method='permutation'):
        """
        获取特征重要性
        
        Args:
            model: 训练好的模型
            X: 特征数据
            y: 目标变量
            feature_names: 特征名称列表
            method: 特征重要性计算方法，可选值为'permutation'（默认）、'coef'、'gain'
            
        Returns:
            feature_importance: 特征重要性数据（JSON格式）
        """
        try:
            # 转换为DataFrame
            if not isinstance(X, pd.DataFrame):
                if feature_names is None:
                    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
                X = pd.DataFrame(X, columns=feature_names)
            else:
                feature_names = X.columns.tolist()
            
            # 不同模型类型和方法的处理逻辑
            if method == 'coef' and hasattr(model, 'coef_'):
                # 使用模型的coef_属性获取特征重要性
                importances = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
            elif method == 'gain' and hasattr(model, 'feature_importances_'):
                # 使用模型的feature_importances_属性获取特征重要性
                importances = model.feature_importances_
            else:
                # 默认使用排列重要性
                result = permutation_importance(
                    model, X, y, n_repeats=30, random_state=42, n_jobs=-1
                )
                importances = result.importances_mean
            
            # 构建特征重要性列表
            feature_importance = []
            for i, imp in enumerate(importances):
                feature_importance.append({
                    'feature_index': i,
                    'feature_name': feature_names[i],
                    'importance': float(imp),
                    'abs_importance': float(abs(imp))
                })
            
            # 按绝对重要性排序
            feature_importance.sort(key=lambda x: x['abs_importance'], reverse=True)
            
            return feature_importance
        except Exception as e:
            print(f"计算特征重要性时出错: {e}")
            return None
    
    def generate_model_explanation(self, model, X_train, X_test, y_train, feature_names=None, 
                                   n_samples=10):
        """
        生成完整的模型解释报告（JSON格式）
        
        Args:
            model: 训练好的模型
            X_train: 训练数据
            X_test: 测试数据
            y_train: 训练目标
            feature_names: 特征名称列表
            n_samples: 用于LIME解释的样本数量
            
        Returns:
            explanation_data: 解释数据字典（JSON格式）
        """
        try:
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
            
            # 转换为numpy数组
            if isinstance(X_train, pd.DataFrame):
                X_train = X_train.values
            if isinstance(X_test, pd.DataFrame):
                X_test = X_test.values
            if isinstance(y_train, pd.Series):
                y_train = y_train.values
            
            explanation_data = {
                'explanation_id': str(uuid.uuid4()),
                'model_type': type(model).__name__,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'feature_names': feature_names,
                'feature_importance': None,
                'shap_data': None,
                'lime_data': None,
                'pdp_data': None,
                'prediction_summary': None
            }
            
            # 1. 计算特征重要性
            feature_importance = self.get_feature_importance(model, X_train, y_train, feature_names)
            explanation_data['feature_importance'] = feature_importance
            
            # 2. 生成SHAP解释
            try:
                # 获取SHAP值和摘要数据
                shap_values, expected_value, explainer, X_sample = self.get_shap_explanation(
                    model, X_train, feature_names=feature_names, sample_size=n_samples
                )
                shap_summary_data = self.get_shap_summary_data(shap_values, X_sample, feature_names)
                explanation_data['shap_data'] = {
                    'summary': shap_summary_data,
                    'expected_value': float(expected_value) if isinstance(expected_value, (int, float)) else expected_value.tolist()
                }
            except Exception as e:
                print(f"生成SHAP解释时出错: {e}")
            
            # 3. 生成LIME解释
            try:
                lime_exp = self.get_lime_explanation(model, X_train, X_test, instance_idx=0, num_features=10)
                if lime_exp:
                    lime_data = self.get_lime_data(lime_exp, feature_names)
                    explanation_data['lime_data'] = lime_data
            except Exception as e:
                print(f"生成LIME解释时出错: {e}")
            
            # 4. 生成部分依赖图数据
            try:
                # 为最重要的几个特征生成部分依赖图数据
                if feature_importance:
                    # 按重要性排序特征
                    top_features = [item['feature_index'] for item in feature_importance[:3]]
                    if top_features:  # 确保有特征可以处理
                        pdp_data = self.get_partial_dependence_data(model, X_train, top_features, feature_names)
                        explanation_data['pdp_data'] = pdp_data
            except Exception as e:
                print(f"生成部分依赖图数据时出错: {e}")
            
            # 5. 生成预测摘要
            if hasattr(model, 'predict'):
                try:
                    y_pred = model.predict(X_test)
                    explanation_data['prediction_summary'] = {
                        'test_samples': X_test.shape[0],
                        'prediction_min': float(np.min(y_pred)),
                        'prediction_max': float(np.max(y_pred)),
                        'prediction_mean': float(np.mean(y_pred)),
                        'prediction_std': float(np.std(y_pred))
                    }
                except Exception as e:
                    print(f"生成预测摘要时出错: {e}")
            
            return explanation_data
        except Exception as e:
            print(f"生成模型解释报告时出错: {e}")
            import traceback
            traceback.print_exc()
            return None

class MILPInterpreter:
    """
    MILP优化器解释器
    - 可视化决策路径
    - 分析约束影响
    - 生成决策规则
    """
    
    def __init__(self, optimizer):
        """
        Args:
            optimizer: MILPOptimizer实例
        """
        self.optimizer = optimizer
    
    def get_constraint_analysis(self):
        """
        分析约束的影响
        
        Returns:
            constraint_analysis: 约束分析结果
        """
        if not self.optimizer.solver:
            raise ValueError("优化器尚未创建模型")
        
        constraint_analysis = {
            "total_constraints": self.optimizer.solver.NumConstraints(),
            "total_variables": self.optimizer.solver.NumVariables(),
            "constraint_types": {},
            "binding_constraints": []
        }
        
        # 分析约束类型（简化版）
        # OR-Tools不直接提供约束类型信息，我们通过变量和约束的关系来分析
        
        return constraint_analysis
    
    def get_decision_path(self):
        """
        获取决策路径
        
        Returns:
            decision_path: 决策路径数据
        """
        if not self.optimizer.variables:
            raise ValueError("优化器尚未创建模型变量")
        
        # 简化版决策路径分析
        decision_path = {
            "variables": {
                "order_quantity": len(self.optimizer.variables.get('order_quantity', {})),
                "inventory_level": len(self.optimizer.variables.get('inventory_level', {})),
                "shortage_quantity": len(self.optimizer.variables.get('shortage_quantity', {})),
                "order_decision": len(self.optimizer.variables.get('order_decision', {})),
                "safety_stock": len(self.optimizer.variables.get('safety_stock', {})),
                "transfer_quantity": len(self.optimizer.variables.get('transfer_quantity', {}))
            },
            "decision_factors": ["forecast_demand", "current_inventory", "lead_time", "costs", "constraints"]
        }
        
        return decision_path
    
    def generate_decision_rules(self, results, top_n=10):
        """
        生成业务决策规则
        
        Args:
            results: 优化结果
            top_n: 生成前N个规则
            
        Returns:
            decision_rules: 业务决策规则
        """
        decision_rules = []
        
        # 1. 基于订货量的规则
        order_quantities = results.get('order_quantity', [])
        if order_quantities:
            # 找出订货量最大的产品和时期
            max_order = np.max(order_quantities)
            max_order_idx = np.unravel_index(np.argmax(order_quantities), order_quantities.shape)
            decision_rules.append({
                "rule_id": 1,
                "priority": "high",
                "condition": f"当产品 {max_order_idx[0]} 在时期 {max_order_idx[1]} 的预测需求较高时",
                "action": f"建议订货 {max_order:.2f} 单位",
                "confidence": 0.95,
                "support": 0.8
            })
        
        # 2. 基于库存水平的规则
        inventory_levels = results.get('inventory_level', [])
        if inventory_levels:
            # 找出库存水平最低的产品和时期
            min_inventory = np.min(inventory_levels)
            min_inventory_idx = np.unravel_index(np.argmin(inventory_levels), inventory_levels.shape)
            decision_rules.append({
                "rule_id": 2,
                "priority": "medium",
                "condition": f"当产品 {min_inventory_idx[0]} 在时期 {min_inventory_idx[1]} 的库存水平较低时",
                "action": f"建议补充库存至安全水平",
                "confidence": 0.85,
                "support": 0.7
            })
        
        # 3. 基于缺货量的规则
        shortage_quantities = results.get('shortage_quantity', [])
        if shortage_quantities:
            # 找出缺货量最大的产品和时期
            max_shortage = np.max(shortage_quantities)
            if max_shortage > 0:
                max_shortage_idx = np.unravel_index(np.argmax(shortage_quantities), shortage_quantities.shape)
                decision_rules.append({
                    "rule_id": 3,
                    "priority": "critical",
                    "condition": f"当产品 {max_shortage_idx[0]} 在时期 {max_shortage_idx[1]} 预计会缺货 {max_shortage:.2f} 单位时",
                    "action": f"建议紧急订货以避免缺货",
                    "confidence": 0.98,
                    "support": 0.9
                })
        
        return decision_rules
    
    def visualize_decision_network(self, results, explanation_id=None):
        """
        可视化决策网络
        
        Args:
            results: 优化结果
            explanation_id: 解释ID
        """
        # 简化版决策网络可视化
        plt.figure(figsize=(14, 10))
        
        # 绘制决策因素
        decision_factors = ["预测需求", "当前库存", "提前期", "成本", "约束条件"]
        decision_actions = ["订货决策", "库存水平", "缺货风险", "调拨决策"]
        
        # 使用网络图表示决策关系
        import networkx as nx
        G = nx.DiGraph()
        
        # 添加节点
        for factor in decision_factors:
            G.add_node(factor, type="factor", color="lightblue", style="filled")
        
        for action in decision_actions:
            G.add_node(action, type="action", color="lightgreen", style="filled")
        
        # 添加边
        for factor in decision_factors:
            for action in decision_actions:
                G.add_edge(factor, action, weight=0.5)
        
        # 设置布局
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        
        # 绘制节点
        node_colors = [G.nodes[n]['color'] for n in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_size=3000, node_color=node_colors, alpha=0.8)
        
        # 绘制边
        nx.draw_networkx_edges(G, pos, width=2, alpha=0.6, edge_color='gray')
        
        # 绘制标签
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
        
        plt.title('MIP优化决策网络', fontsize=16)
        plt.axis('off')
        
        # 保存图像
        if explanation_id:
            fig_path = os.path.join('interpretations/figures', f"{explanation_id}_decision_network.png")
            plt.savefig(fig_path, bbox_inches='tight', dpi=300)
        plt.close()
    
    def generate_milp_explanation(self, results, explanation_id=None):
        """
        生成完整的MIP优化解释报告
        
        Args:
            results: 优化结果
            explanation_id: 解释ID
            
        Returns:
            explanation_data: 解释数据
            explanation_id: 解释ID
            file_path: 保存路径
        """
        if explanation_id is None:
            explanation_id = f"milp_explain_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}"
        
        # 1. 约束分析
        constraint_analysis = self.get_constraint_analysis()
        
        # 2. 决策路径
        decision_path = self.get_decision_path()
        
        # 3. 生成决策规则
        decision_rules = self.generate_decision_rules(results)
        
        # 4. 可视化决策网络
        self.visualize_decision_network(results, explanation_id)
        
        # 5. 关键指标分析
        total_cost = results.get('total_cost', 0)
        order_cost = results.get('order_cost', 0)
        holding_cost = results.get('holding_cost', 0)
        shortage_cost = results.get('shortage_cost', 0)
        
        cost_breakdown = {
            "total_cost": total_cost,
            "order_cost": order_cost,
            "holding_cost": holding_cost,
            "shortage_cost": shortage_cost,
            "cost_percentage": {
                "order_cost": (order_cost / total_cost * 100) if total_cost > 0 else 0,
                "holding_cost": (holding_cost / total_cost * 100) if total_cost > 0 else 0,
                "shortage_cost": (shortage_cost / total_cost * 100) if total_cost > 0 else 0
            }
        }
        
        # 构建解释数据
        explanation_data = {
            "explanation_id": explanation_id,
            "timestamp": datetime.now().isoformat(),
            "optimizer_type": "MILP",
            "constraint_analysis": constraint_analysis,
            "decision_path": decision_path,
            "decision_rules": decision_rules,
            "cost_breakdown": cost_breakdown,
            "results_summary": {
                "total_products": self.optimizer.num_products,
                "total_periods": self.optimizer.num_periods,
                "total_cost": total_cost,
                "avg_cost_per_product": (total_cost / self.optimizer.num_products) if self.optimizer.num_products > 0 else 0
            },
            "figures": {
                "decision_network": f"{explanation_id}_decision_network.png"
            }
        }
        
        # 保存解释结果
        interpreter = ModelInterpreter()
        _, file_path = interpreter.save_interpretation(explanation_data, explanation_id)
        
        return explanation_data, explanation_id, file_path

class BusinessRuleGenerator:
    """
    业务规则生成器
    将模型决策转换为易懂的业务规则
    """
    
    def __init__(self):
        self.rule_templates = {
            "high_demand": "当产品 {product_id} 的{feature}超过{threshold}时，预计需求将增加{expected_increase}%，建议增加订货量{suggested_increase}%",
            "low_inventory": "当产品 {product_id} 的库存水平低于安全库存的{threshold}%时，存在缺货风险{risk_level}%，建议立即订货{suggested_quantity}单位",
            "seasonal_spike": "基于历史数据，产品 {product_id} 在{period}期间需求将季节性增长{growth_rate}%，建议提前备货{suggested_quantity}单位",
            "promotion_impact": "当产品 {product_id} 进行促销活动时，需求预计将增长{promotion_impact}%，建议调整库存水平至{suggested_level}",
            "lead_time_variability": "当供应商 {supplier_id} 的交货提前期波动超过{variability_threshold}%时，建议增加安全库存{suggested_safety_stock}%"
        }
    
    def generate_business_rules(self, model, X, y, feature_names=None, top_n=10):
        """
        从模型中生成业务规则
        
        Args:
            model: 训练好的模型
            X: 特征数据
            y: 目标变量
            feature_names: 特征名称列表
            top_n: 生成前N个规则
            
        Returns:
            business_rules: 业务规则列表
        """
        # 转换为DataFrame
        if not isinstance(X, pd.DataFrame):
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=feature_names)
        
        # 获取特征重要性
        interpreter = ModelInterpreter()
        feature_importance = interpreter.get_feature_importance(model, X, y)
        
        business_rules = []
        
        # 基于特征重要性生成规则
        for i, (_, row) in enumerate(feature_importance.head(top_n).iterrows()):
            feature = row['feature']
            importance = row['importance']
            
            # 根据特征类型生成不同的规则
            if 'demand' in feature.lower():
                rule = {
                    "rule_id": i + 1,
                    "rule_type": "demand_forecast",
                    "condition": f"当 {feature} 增加1单位时",
                    "impact": f"预测需求将变化 {importance:.4f} 单位",
                    "confidence": min(0.99, 0.7 + importance * 10),
                    "priority": "high" if importance > 0.1 else "medium",
                    "feature_importance": importance
                }
            elif 'inventory' in feature.lower():
                rule = {
                    "rule_id": i + 1,
                    "rule_type": "inventory_management",
                    "condition": f"当 {feature} 增加1单位时",
                    "impact": f"预测需求将变化 {importance:.4f} 单位",
                    "confidence": min(0.99, 0.7 + importance * 10),
                    "priority": "medium" if importance > 0.05 else "low",
                    "feature_importance": importance
                }
            elif 'price' in feature.lower() or 'cost' in feature.lower():
                rule = {
                    "rule_id": i + 1,
                    "rule_type": "pricing_impact",
                    "condition": f"当 {feature} 增加1单位时",
                    "impact": f"预测需求将变化 {importance:.4f} 单位",
                    "confidence": min(0.99, 0.7 + importance * 10),
                    "priority": "medium",
                    "feature_importance": importance
                }
            else:
                rule = {
                    "rule_id": i + 1,
                    "rule_type": "general",
                    "condition": f"当 {feature} 增加1单位时",
                    "impact": f"预测需求将变化 {importance:.4f} 单位",
                    "confidence": min(0.99, 0.7 + importance * 10),
                    "priority": "low",
                    "feature_importance": importance
                }
            
            business_rules.append(rule)
        
        return business_rules
    
    def simplify_rules(self, rules, max_rules=5):
        """
        简化业务规则
        
        Args:
            rules: 业务规则列表
            max_rules: 最大规则数量
            
        Returns:
            simplified_rules: 简化后的业务规则
        """
        # 按优先级和置信度排序
        sorted_rules = sorted(
            rules,
            key=lambda x: (x['priority'] == 'high', x['confidence'], x['feature_importance']),
            reverse=True
        )
        
        # 保留前max_rules个规则
        return sorted_rules[:max_rules]
    
    def generate_rule_report(self, rules, model_name="Unknown Model"):
        """
        生成规则报告
        
        Args:
            rules: 业务规则列表
            model_name: 模型名称
            
        Returns:
            rule_report: 规则报告
        """
        report = {
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "total_rules": len(rules),
            "rules_by_priority": {
                "high": len([r for r in rules if r['priority'] == 'high']),
                "medium": len([r for r in rules if r['priority'] == 'medium']),
                "low": len([r for r in rules if r['priority'] == 'low'])
            },
            "rules": rules
        }
        
        return report
