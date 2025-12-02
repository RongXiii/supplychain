import sys
import os
import numpy as np

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ab_testing import TestVariant, ABTestManager

# 设置随机种子，确保结果可重现
np.random.seed(42)

def test_ab_testing_framework():
    """测试A/B测试框架的核心功能"""
    print("=== 开始测试A/B测试框架 ===")
    
    # 1. 创建A/B测试管理器
    test_manager = ABTestManager(test_id="test_demo_001")
    print(f"✓ 创建测试管理器，测试ID: {test_manager.test_id}")
    
    # 2. 创建测试变体
    # 变体A：基线模型
    variant_a = TestVariant(variant_id="A", name="Baseline Model", sample_size=1000)
    # 变体B：新模型1
    variant_b = TestVariant(variant_id="B", name="New Model 1", sample_size=1000)
    # 变体C：新模型2
    variant_c = TestVariant(variant_id="C", name="New Model 2", sample_size=1000)
    
    # 添加变体会测试管理器
    test_manager.add_variant(variant_a)
    test_manager.add_variant(variant_b)
    test_manager.add_variant(variant_c)
    print(f"✓ 创建并添加了 {len(test_manager.variants)} 个测试变体")
    
    # 3. 设置测试指标
    metrics = ["conversion_rate", "revenue_per_user", "click_through_rate", "average_order_value"]
    test_manager.set_metrics(metrics)
    print(f"✓ 设置了测试指标: {', '.join(metrics)}")
    
    # 4. 开始测试
    test_manager.start_test()
    print(f"✓ 测试开始时间: {test_manager.start_time}")
    
    # 5. 模拟测试数据
    print("\n=== 模拟测试数据 ===")
    
    # 生成模拟数据的参数
    # 变体A（基线）
    params_a = {
        "conversion_rate": (0.10, 0.02),  # 均值, 标准差
        "revenue_per_user": (50, 10),
        "click_through_rate": (0.05, 0.01),
        "average_order_value": (100, 20)
    }
    
    # 变体B（新模型1，性能稍好）
    params_b = {
        "conversion_rate": (0.12, 0.02),
        "revenue_per_user": (55, 11),
        "click_through_rate": (0.06, 0.012),
        "average_order_value": (105, 22)
    }
    
    # 变体C（新模型2，性能最好）
    params_c = {
        "conversion_rate": (0.15, 0.025),
        "revenue_per_user": (60, 12),
        "click_through_rate": (0.08, 0.015),
        "average_order_value": (115, 25)
    }
    
    # 模拟数据生成函数
    def generate_performance_data(params, n_samples=1000):
        """生成模拟性能数据"""
        data = []
        for _ in range(n_samples):
            record = {}
            for metric, (mean, std) in params.items():
                # 生成正态分布数据，确保非负
                value = np.max([0, np.random.normal(mean, std)])
                record[metric] = value
            data.append(record)
        return data
    
    # 为每个变体生成数据
    data_a = generate_performance_data(params_a, 1000)
    data_b = generate_performance_data(params_b, 1000)
    data_c = generate_performance_data(params_c, 1000)
    
    # 记录性能数据
    for record in data_a:
        variant_a.record_performance(**record)
    
    for record in data_b:
        variant_b.record_performance(**record)
    
    for record in data_c:
        variant_c.record_performance(**record)
    
    print(f"✓ 为变体A生成了 {len(data_a)} 条数据")
    print(f"✓ 为变体B生成了 {len(data_b)} 条数据")
    print(f"✓ 为变体C生成了 {len(data_c)} 条数据")
    
    # 6. 结束测试
    test_manager.end_test()
    print(f"\n✓ 测试结束时间: {test_manager.end_time}")
    print(f"✓ 测试持续时间: {test_manager.end_time - test_manager.start_time}")
    
    # 7. 计算变体指标
    for variant in test_manager.variants.values():
        print(f"\n=== 变体 {variant.name} 指标 ===")
        for metric, stats in variant.metrics.items():
            print(f"  {metric}: 均值={stats['mean']:.4f}, 标准差={stats['std']:.4f}, 样本量={stats['count']}")
    
    # 8. 运行统计显著性测试
    print("\n=== 运行统计显著性测试 ===")
    test_manager.run_statistical_tests(alpha=0.05)
    print(f"✓ 完成了 {len(test_manager.test_results)} 次统计比较")
    
    # 9. 显示统计测试结果
    print("\n=== 统计测试结果摘要 ===")
    for key, result in test_manager.test_results.items():
        # 先分割出变体部分和指标部分
        parts = key.split('_vs_')
        variant_a_id = parts[0]
        # 剩余部分包含 variant_b_id 和 metric，例如 "B_conversion_rate"
        variant_b_and_metric = '_vs_'.join(parts[1:])
        # 找到第一个下划线，分割出 variant_b_id 和 metric
        first_underscore_index = variant_b_and_metric.find('_')
        variant_b_id = variant_b_and_metric[:first_underscore_index]
        metric = variant_b_and_metric[first_underscore_index+1:]
        
        variant_a_name = test_manager.variants[variant_a_id].name
        variant_b_name = test_manager.variants[variant_b_id].name
        
        sig_text = "显著" if result['significant'] else "不显著"
        print(f"  {variant_a_name} vs {variant_b_name} ({metric}): p值={result['p_value']:.4f}, {sig_text}")
    
    # 10. 生成可视化
    print("\n=== 生成测试可视化 ===")
    test_manager.generate_visualizations()
    print(f"✓ 可视化已生成，保存在: {test_manager.visualizer.output_dir}")
    
    # 11. 生成测试报告
    print("\n=== 生成测试报告 ===")
    report_path = test_manager.generate_report()
    print(f"✓ 测试报告已生成: {report_path}")
    
    # 12. 获取测试摘要
    test_summary = test_manager.get_test_summary()
    print("\n=== 测试摘要 ===")
    for key, value in test_summary.items():
        print(f"  {key}: {value}")
    
    print("\n=== A/B测试框架测试完成 ===")
    print(f"✓ 测试结果显示：在{len(test_manager.test_results)}次比较中，有{test_summary['num_significant_results']}次结果具有统计显著性")
    
    # 检查测试结果是否符合预期
    # 预期：变体C应该在大多数指标上显著优于变体A和B
    significant_results = [r for r in test_manager.test_results.values() if r['significant']]
    assert len(significant_results) > 0, "应该有显著的测试结果"
    
    print("\n✅ 所有测试通过！A/B测试框架功能正常")
    
    return True

def test_variant_assignment():
    """测试变体分配功能"""
    print("\n=== 测试变体分配功能 ===")
    
    # 创建测试管理器
    test_manager = ABTestManager()
    
    # 创建测试变体
    variant_a = TestVariant(variant_id="A", name="Variant A")
    variant_b = TestVariant(variant_id="B", name="Variant B")
    variant_c = TestVariant(variant_id="C", name="Variant C")
    
    test_manager.add_variant(variant_a)
    test_manager.add_variant(variant_b)
    test_manager.add_variant(variant_c)
    
    # 测试分配稳定性：相同用户ID应该分配到相同变体
    user_ids = ["user_123", "user_456", "user_789", "user_101", "user_202"]
    
    for user_id in user_ids:
        variant1 = test_manager.assign_variant(user_id)
        variant2 = test_manager.assign_variant(user_id)
        assert variant1.variant_id == variant2.variant_id, f"用户 {user_id} 应该分配到相同变体"
        print(f"  用户 {user_id} 稳定分配到变体: {variant1.name}")
    
    # 测试分配均匀性
    assignments = {}
    for i in range(10000):
        user_id = f"user_{i}"
        variant = test_manager.assign_variant(user_id)
        assignments[variant.variant_id] = assignments.get(variant.variant_id, 0) + 1
    
    print(f"\n  10,000次分配结果：")
    for variant_id, count in assignments.items():
        percentage = count / 10000 * 100
        print(f"    变体 {test_manager.variants[variant_id].name}: {count}次 ({percentage:.1f}%)")
    
    # 验证分配是否大致均匀（每个变体分配比例在30%-40%之间）
    for count in assignments.values():
        percentage = count / 10000 * 100
        assert 30 < percentage < 40, f"变体分配应该大致均匀，当前百分比: {percentage:.1f}%"
    
    print("✅ 变体分配功能测试通过！")
    return True

if __name__ == "__main__":
    """运行所有测试"""
    try:
        # 测试A/B测试框架核心功能
        test_ab_testing_framework()
        
        # 测试变体分配功能
        test_variant_assignment()
        
        print("\n🎉 所有A/B测试框架测试都已通过！")
        print("\n📋 测试结果：")
        print("   - 核心功能测试: ✓ 通过")
        print("   - 变体分配测试: ✓ 通过")
        print("   - 统计测试功能: ✓ 通过")
        print("   - 可视化生成: ✓ 通过")
        print("   - 报告生成: ✓ 通过")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
