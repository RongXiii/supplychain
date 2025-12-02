import numpy as np
import pandas as pd
# 设置matplotlib使用非交互式后端，避免弹出窗口
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
import shap
import lime
import lime.lime_tabular
import json
import os
from datetime import datetime

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

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
    
    def plot_shap_summary(self, shap_values, X, feature_names=None, explanation_id=None):
        """绘制SHAP摘要图"""
        # SHAP的summary_plot会创建自己的figure，不需要提前创建
        try:
            # 直接使用SHAP的summary_plot，设置show=False避免自动显示
            shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type="bar", show=False)
            
            # 保存图像
            if explanation_id:
                fig_path = os.path.join(self.fig_dir, f"{explanation_id}_shap_summary.png")
                plt.savefig(fig_path, bbox_inches='tight', dpi=300)
        except Exception as e:
            print(f"绘制SHAP摘要图时出错: {e}")
        finally:
            # 确保关闭所有figure，避免内存泄漏
            plt.close('all')
    
    def plot_shap_force(self, explainer, expected_value, shap_values, X, instance_idx=0, explanation_id=None):
        """绘制SHAP力图"""
        try:
            # SHAP的force_plot使用matplotlib=True时会创建自己的figure
            shap.force_plot(expected_value, shap_values[instance_idx], X.iloc[instance_idx], matplotlib=True, show=False)
            
            # 保存图像
            if explanation_id:
                fig_path = os.path.join(self.fig_dir, f"{explanation_id}_shap_force_{instance_idx}.png")
                plt.savefig(fig_path, bbox_inches='tight', dpi=300, pad_inches=0.5)
        except Exception as e:
            print(f"绘制SHAP力图时出错: {e}")
        finally:
            # 确保关闭所有figure
            plt.close('all')
    
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
    
    def plot_lime_explanation(self, lime_explanation, explanation_id=None):
        """绘制LIME解释图"""
        try:
            # as_pyplot_figure()会返回一个Figure对象，我们应该使用这个对象来保存
            fig = lime_explanation.as_pyplot_figure()
            fig.set_size_inches(12, 8)  # 设置图表大小
            
            # 使用返回的Figure对象保存图像
            if explanation_id:
                fig_path = os.path.join(self.fig_dir, f"{explanation_id}_lime.png")
                fig.savefig(fig_path, bbox_inches='tight', dpi=300)
        except Exception as e:
            print(f"绘制LIME解释图时出错: {e}")
        finally:
            # 确保关闭所有figure
            plt.close('all')
    
    def plot_partial_dependence(self, model, X, features, feature_names=None, explanation_id=None):
        """
        绘制部分依赖图
        
        Args:
            model: 训练好的模型
            X: 特征数据
            features: 要分析的特征索引或名称
            feature_names: 特征名称列表
            explanation_id: 解释ID
        """
        try:
            # 转换为DataFrame并进行严格的数据检查
            if not isinstance(X, pd.DataFrame):
                if feature_names is None:
                    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
                X = pd.DataFrame(X, columns=feature_names)
            
            # 确保X不为空且包含有效数据
            if X.empty:
                raise ValueError("输入数据X不能为空")
            
            # 确保features是有效的非空列表
            if not features:
                raise ValueError("特征列表features不能为空")
            
            # 确保特征存在于数据中
            if isinstance(features[0], str):
                # 特征是名称，检查是否都存在
                missing_features = [f for f in features if f not in X.columns]
                if missing_features:
                    raise ValueError(f"以下特征不存在于数据中: {missing_features}")
            else:
                # 特征是索引，检查是否在有效范围内
                max_idx = X.shape[1] - 1
                invalid_indices = [f for f in features if f < 0 or f > max_idx]
                if invalid_indices:
                    raise ValueError(f"以下特征索引无效，有效范围是0到{max_idx}: {invalid_indices}")
            
            # 创建figure和axes（在数据检查通过后）
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # 生成部分依赖图，指定axes
            PartialDependenceDisplay.from_estimator(
                model, 
                X, 
                features, 
                feature_names=feature_names,
                grid_resolution=20,
                ax=ax
            )
            
            # 添加标题和调整布局
            plt.title('Partial Dependence Plots')
            plt.tight_layout()
            
            # 保存图像（使用fig.savefig确保保存正确的figure）
            if explanation_id:
                fig_path = os.path.join(self.fig_dir, f"{explanation_id}_partial_dependence.png")
                fig.savefig(fig_path, bbox_inches='tight', dpi=300)
            
            # 关闭figure
            plt.close(fig)
        except Exception as e:
            print(f"绘制部分依赖图时出错: {e}")
            # 确保关闭所有figure以避免内存泄漏
            plt.close('all')
    
    def get_feature_importance(self, model, X, y, feature_names=None, method='permutation'):
        """
        获取特征重要性
        
        Args:
            model: 训练好的模型
            X: 特征数据
            y: 目标变量
            feature_names: 特征名称列表
            method: 重要性计算方法 ('permutation', 'coef', 'gain')
            
        Returns:
            feature_importance: 特征重要性DataFrame
        """
        # 转换为DataFrame
        if not isinstance(X, pd.DataFrame):
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=feature_names)
        
        # 不同模型类型和方法的处理逻辑
        if hasattr(model, 'feature_importances_') and method in ['gain', 'native', 'built-in']:
            # 使用模型自带的特征重要性
            importances = model.feature_importances_
            std = np.zeros(len(importances))  # 对于树模型，无法直接获取标准误差
        elif hasattr(model, 'coef_') and method in ['coef', 'native', 'built-in']:
            # 线性模型系数
            importances = np.abs(model.coef_)
            if len(importances.shape) > 1:
                importances = np.mean(importances, axis=0)
            std = np.zeros(len(importances))
        else:
            # 使用排列重要性
            result = permutation_importance(
                model, X, y, n_repeats=10, random_state=42, n_jobs=-1
            )
            importances = result.importances_mean
            std = result.importances_std
        
        # 创建DataFrame
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': importances,
            'std': std
        })
        
        # 按重要性排序
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        return feature_importance
    
    def plot_feature_importance(self, feature_importance, explanation_id=None):
        """绘制特征重要性图"""
        plt.figure(figsize=(12, 8))
        
        # 确保feature_importance不为空
        if feature_importance.empty:
            plt.title('特征重要性 - 无可用特征')
            plt.close()
            return None
        
        try:
            # 简化版本：不使用误差线，避免xerr形状不匹配问题
            sns.barplot(
                x='importance', 
                y='feature', 
                data=feature_importance
            )
            
            plt.title('特征重要性', fontsize=16)
            plt.xlabel('重要性', fontsize=14)
            plt.ylabel('特征', fontsize=14)
            plt.tight_layout()
            
            # 保存图像
            fig_path = None
            if explanation_id:
                fig_path = os.path.join(self.fig_dir, f"{explanation_id}_feature_importance.png")
                plt.savefig(fig_path, bbox_inches='tight', dpi=300)
        except Exception as e:
            print(f"绘制特征重要性图时出错: {e}")
            # 进一步简化：仅绘制基本图表
            plt.title('特征重要性 - 绘制失败')
            fig_path = None
        finally:
            plt.close()
        
        return fig_path
    
    def generate_model_explanation(self, model, X_train, X_test, y_train, feature_names=None, 
                                  sample_size=100, num_features=10):
        """
        生成完整的模型解释报告
        
        Args:
            model: 训练好的模型
            X_train: 训练数据
            X_test: 测试数据
            y_train: 训练标签
            feature_names: 特征名称列表
            sample_size: 用于解释的样本大小
            num_features: 要显示的特征数量
            
        Returns:
            explanation_data: 解释数据
            explanation_id: 解释ID
            file_path: 保存路径
        """
        try:
            explanation_id = self.generate_explanation_id()
            
            # 转换为DataFrame并进行数据校验
            if not isinstance(X_train, pd.DataFrame):
                if feature_names is None:
                    feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
                X_train_df = pd.DataFrame(X_train, columns=feature_names)
            else:
                X_train_df = X_train
                feature_names = X_train.columns.tolist()
            
            if not isinstance(X_test, pd.DataFrame):
                X_test_df = pd.DataFrame(X_test, columns=feature_names)
            else:
                X_test_df = X_test
            
            # 确保数据不为空
            if X_train_df.empty or X_test_df.empty:
                raise ValueError("训练或测试数据不能为空")
            
            # 1. 特征重要性
            feature_importance = self.get_feature_importance(model, X_train_df, y_train)
            self.plot_feature_importance(feature_importance, explanation_id)
            
            # 2. SHAP解释
            shap_values, expected_value, explainer, X_sample = self.get_shap_explanation(
                model, X_train_df, feature_names=feature_names, sample_size=sample_size
            )
            self.plot_shap_summary(shap_values, X_sample, feature_names=feature_names, explanation_id=explanation_id)
            
            # 3. 绘制前5个实例的SHAP力图
            for i in range(min(5, len(X_sample))):
                try:
                    self.plot_shap_force(explainer, expected_value, shap_values, X_sample, instance_idx=i, explanation_id=explanation_id)
                except Exception as e:
                    print(f"绘制SHAP力图时出错 (实例 {i}): {e}")
            
            # 4. LIME解释 - 确保测试数据有足够实例
            if len(X_test_df) > 0:
                try:
                    lime_exp = self.get_lime_explanation(model, X_train_df, X_test_df, instance_idx=0, num_features=num_features)
                    self.plot_lime_explanation(lime_exp, explanation_id=explanation_id)
                    # 生成LIME解释的文本表示
                    lime_features = [(feature_names[i], weight) for i, weight in lime_exp.as_list()]
                except Exception as e:
                    print(f"生成LIME解释时出错: {e}")
                    lime_features = []
            else:
                lime_features = []
            
            # 5. 部分依赖图（前3个重要特征）
            try:
                top_features = feature_importance['feature'].head(3).tolist()
                if top_features:  # 确保有特征可以绘制
                    self.plot_partial_dependence(model, X_train_df, top_features, feature_names=feature_names, explanation_id=explanation_id)
            except Exception as e:
                print(f"生成部分依赖图时出错: {e}")
                top_features = []
            
            # 构建解释数据
            explanation_data = {
                "explanation_id": explanation_id,
                "timestamp": datetime.now().isoformat(),
                "model_type": type(model).__name__,
                "feature_names": feature_names,
                "data_shape": {
                    "train": X_train.shape,
                    "test": X_test.shape
                },
                "feature_importance": feature_importance.to_dict(orient='records'),
                "lime_explanation": lime_features,
                "shap_expected_value": float(expected_value) if isinstance(expected_value, (int, float)) else expected_value.tolist(),
                "top_features": top_features,
                "figures": {
                    "feature_importance": f"{explanation_id}_feature_importance.png",
                    "shap_summary": f"{explanation_id}_shap_summary.png",
                    "lime": f"{explanation_id}_lime.png" if lime_features else None,
                    "partial_dependence": f"{explanation_id}_partial_dependence.png" if top_features else None,
                    "shap_force": [f"{explanation_id}_shap_force_{i}.png" for i in range(min(5, len(X_sample)))]
                }
            }
            
            # 保存解释结果
            _, file_path = self.save_interpretation(explanation_data, explanation_id)
            
            return explanation_data, explanation_id, file_path
        except Exception as e:
            print(f"生成模型解释时出错: {e}")
            import traceback
            traceback.print_exc()
            # 返回错误信息
            error_id = f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}"
            return {"error": str(e)}, error_id, None

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
