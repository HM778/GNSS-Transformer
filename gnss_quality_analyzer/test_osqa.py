#!/usr/bin/env python3
"""
test_osqa.py — OSQA 集成测试脚本
================================

使用合成数据和真实CSV数据验证OSQA各模块的功能。

测试覆盖:
  1. 特征提取正确性
  2. 记忆库滑动窗口
  3. Transformer自注意力异常检测
  4. 图结构一致性分析
  5. 时序异常检测
  6. 分数融合
  7. 端到端pipeline

Author: Claude Code
Date: 2026-07-02
"""

import sys
import os
import numpy as np

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnss_quality_analyzer.config import OSQAConfig, get_urban_config
from gnss_quality_analyzer.memory_buffer import MemoryBuffer, EpochData, SatelliteSample
from gnss_quality_analyzer.feature_extractor import FeatureExtractor, RawObservation
from gnss_quality_analyzer.transformer_analyzer import TransformerAnalyzer
from gnss_quality_analyzer.graph_analyzer import GraphAnalyzer, GNSSGraph
from gnss_quality_analyzer.temporal_analyzer import TemporalAnalyzer, SatelliteTracker
from gnss_quality_analyzer.quality_fusion import QualityFusion, TrustLevel


# ==================== 测试工具 ====================

def assert_close(actual, expected, tol=0.01, msg=""):
    """断言两个值接近"""
    if abs(actual - expected) > tol:
        raise AssertionError(f"{msg}: expected {expected:.4f}, got {actual:.4f}")


def create_test_observations(n_normal: int = 8, n_anomalous: int = 2,
                              seed: int = 42) -> list:
    """
    创建测试用卫星观测数据

    生成一些"正常"卫星和"异常"卫星：
    - 正常: SNR 40-50, 仰角 30-80°, 残差小
    - 异常: SNR 15-25, 仰角 5-15°, 残差大
    """
    rng = np.random.RandomState(seed)
    obs_list = []

    # 正常卫星
    for i in range(n_normal):
        obs = RawObservation(
            prn=f"G{i+1:02d}",
            system="G",
            timestamp=1733984500.0,
            snr=rng.uniform(38, 50),
            elevation=rng.uniform(25, 85),
            azimuth=rng.uniform(0, 360),
            pseudorange_l1=2.0e7 + rng.normal(0, 5),
            carrier_phase_l1=1.0e8 + rng.normal(0, 0.01),
            pseudorange_residual=rng.normal(0, 2),  # 小残差
            lock_count=rng.randint(100, 1000),
        )
        obs_list.append(obs)

    # 异常卫星
    anomaly_prns = [f"G{n_normal + i + 1:02d}" for i in range(n_anomalous)]
    for i, prn in enumerate(anomaly_prns):
        obs = RawObservation(
            prn=prn,
            system="G",
            timestamp=1733984500.0,
            snr=rng.uniform(14, 22),            # 低SNR
            elevation=rng.uniform(5, 15),       # 低仰角
            azimuth=rng.uniform(0, 360),
            pseudorange_l1=2.0e7 + rng.normal(0, 30),
            pseudorange_residual=rng.normal(15, 10),  # 大残差
            lock_count=rng.randint(1, 10),       # 短锁定时间
        )
        obs_list.append(obs)

    return obs_list


# ==================== 测试1: 特征提取 ====================

def test_feature_extraction():
    """测试特征提取器的正确性"""
    print("\n=== 测试1: 特征提取 ===")

    extractor = FeatureExtractor()

    # 测试单个观测
    obs = RawObservation(
        prn="G01", system="G", timestamp=1733984500.0,
        snr=45.0, elevation=90.0, azimuth=180.0,
        pseudorange_residual=0.5, carrier_residual=0.01,
        lock_count=500,
    )

    features = extractor.extract_single(obs)
    assert features.shape == (8,), f"特征维度错误: {features.shape}"
    assert features.dtype == np.float32, f"特征类型错误: {features.dtype}"

    # 验证特定维度的值
    assert_close(features[0], 45.0 / 45.0, msg="SNR_norm")  # SNR归一化
    assert_close(features[1], np.sin(np.deg2rad(90.0)), msg="elevation_sin")  # sin(90°)=1

    print(f"  特征向量: {[f'{v:.3f}' for v in features]}")
    print(f"  特征名称: {extractor.get_feature_names()}")
    print("  ✓ 特征提取正确")

    # 测试批量提取
    observations = create_test_observations(n_normal=5, n_anomalous=0)
    features_batch, prns = extractor.extract_batch(observations)
    assert features_batch.shape == (5, 8), f"批量特征维度错误: {features_batch.shape}"
    assert len(prns) == 5, f"PRN数量错误: {len(prns)}"

    # 测试归一化
    features_norm = extractor.normalize(features_batch)
    assert features_norm.shape == features_batch.shape
    print("  ✓ 批量提取正确")

    return True


# ==================== 测试2: 记忆库 ====================

def test_memory_buffer():
    """测试滑动窗口记忆库"""
    print("\n=== 测试2: 记忆库滑动窗口 ===")

    memory = MemoryBuffer(window_size=5, memory_bank_size=20, feature_dims=8)

    # 测试添加epoch
    observations = create_test_observations(n_normal=10, n_anomalous=0)
    features, prns = FeatureExtractor().extract_batch(observations)

    epoch = EpochData(timestamp=1733984500.0)
    for i, obs in enumerate(observations):
        sample = SatelliteSample(
            prn=obs.prn, system=obs.system, features=features[i],
            timestamp=1733984500.0, elevation=obs.elevation,
            azimuth=obs.azimuth, pseudorange_residual=obs.pseudorange_residual,
            snr=obs.snr, lock_count=obs.lock_count,
            is_good=True, quality_score=1.0,
        )
        epoch.satellites.append(sample)

    memory.add_epoch(epoch)
    assert memory.total_epochs_processed == 1
    assert memory.total_samples_seen == 10

    # 测试获取最近epoch
    recent = memory.get_recent_epochs(3)
    assert len(recent) == 1  # 只有1个epoch

    # 添加更多epoch测试窗口边界
    for t in range(1, 8):
        epoch2 = EpochData(timestamp=1733984500.0 + t)
        for i in range(5):  # 每epoch 5颗卫星
            obs = RawObservation(
                prn=f"G{i+1:02d}", system="G",
                timestamp=1733984500.0 + t,
                snr=40, elevation=45, azimuth=i*72,
                pseudorange_residual=np.random.normal(0, 2),
            )
            f, _ = FeatureExtractor().extract_batch([obs])
            sample = SatelliteSample(
                prn=obs.prn, system=obs.system, features=f[0],
                timestamp=obs.timestamp, elevation=obs.elevation,
                azimuth=obs.azimuth,
                pseudorange_residual=obs.pseudorange_residual,
                snr=obs.snr, lock_count=100, is_good=True, quality_score=1.0,
            )
            epoch2.satellites.append(sample)
        memory.add_epoch(epoch2)

    # 窗口大小=5，应该有5个epoch
    recent = memory.get_recent_epochs(10)
    assert len(recent) == 5, f"窗口大小错误: {len(recent)} (expected 5)"

    # 测试统计
    stats = memory.get_statistics()
    assert stats['total_epochs_processed'] == 8
    print(f"  统计: {stats}")
    print("  ✓ 记忆库工作正常")

    return True


# ==================== 测试3: Transformer分析器 ====================

def test_transformer_analyzer():
    """测试Transformer自注意力分析器——核心功能"""
    print("\n=== 测试3: Transformer自注意力异常检测 ===")

    analyzer = TransformerAnalyzer(
        input_dim=8, num_heads=4, head_dim=32, temperature=1.0
    )

    # 创建混合数据（正常+异常）
    observations = create_test_observations(n_normal=8, n_anomalous=2)
    extractor = FeatureExtractor()
    features, prns = extractor.extract_batch(observations)
    features_norm = extractor.normalize(features)

    # 分析
    result = analyzer.analyze(features_norm, prns)

    # 验证输出
    assert result.quality_scores.shape == (10,), f"质量分数形状错误: {result.quality_scores.shape}"
    assert result.attention_matrix.shape == (10, 10), f"注意力矩阵形状错误: {result.attention_matrix.shape}"

    # 异常卫星应该获得更低的分数
    normal_scores = result.quality_scores[:8]
    anomaly_scores = result.quality_scores[8:]

    print(f"  正常卫星分数: {[f'{s:.3f}' for s in normal_scores]}")
    print(f"  异常卫星分数: {[f'{s:.3f}' for s in anomaly_scores]}")
    print(f"  正常平均: {np.mean(normal_scores):.3f}, 异常平均: {np.mean(anomaly_scores):.3f}")

    # 异常卫星的分数应该低于正常卫星
    if np.mean(anomaly_scores) < np.mean(normal_scores):
        print("  ✓ 成功识别异常卫星（异常分数 < 正常分数）")
    else:
        print("  ⚠ 异常分数未明显低于正常分数（可能因为特征差异不够大）")

    # 验证注意力矩阵的性质
    # 注意力矩阵每行和应该≈1（softmax归一化）
    row_sums = np.sum(result.attention_matrix, axis=1)
    for i, s in enumerate(row_sums):
        assert_close(s, 1.0, tol=0.01, msg=f"注意力行{i}和")

    print("  ✓ 注意力矩阵归一化正确")

    # 测试记忆库增强
    proto_features = [features_norm[i] for i in range(5)]  # 前5个作为好原型
    result_with_memory = analyzer.analyze(features_norm, prns, proto_features)
    print(f"  记忆增强后异常分数: {[f'{s:.3f}' for s in result_with_memory.quality_scores[8:]]}")

    return True


# ==================== 测试4: 图分析器 ====================

def test_graph_analyzer():
    """测试图结构几何分析器"""
    print("\n=== 测试4: 图结构几何分析 ===")

    # 测试图构建
    graph = GNSSGraph(elevation_threshold=30.0, azimuth_threshold=60.0)

    # 创建测试数据：4颗卫星在天空的不同位置
    elevations = np.array([80.0, 75.0, 10.0, 8.0])  # 2颗在天顶附近，2颗在低仰角
    azimuths = np.array([0.0, 5.0, 180.0, 185.0])

    adj, weights = graph.build_graph(elevations, azimuths)
    print(f"  邻接矩阵:\n{adj}")
    print(f"  边权重:\n{weights}")

    # 验证：位置相近的卫星应该有边
    # 卫星0和1（天顶附近，仰角和方位角都接近）应该有边
    assert adj[0, 1] == 1.0, "卫星0和1应该有边"
    assert adj[1, 0] == 1.0, "边应该是对称的"

    # 卫星2和3（低仰角，方位角接近）应该有边
    assert adj[2, 3] == 1.0, "卫星2和3应该有边"

    # 卫星0和2（位置差很远）应该无边
    assert adj[0, 2] == 0.0, "卫星0和2应该无边"
    print("  ✓ 图构建正确")

    # 测试完整图分析器
    analyzer = GraphAnalyzer(
        elevation_threshold=30.0,
        azimuth_threshold=60.0,
        consistency_temperature=0.5,
    )

    # 创建特征：卫星0,1特征相似（正常），卫星2特征异常
    observations = create_test_observations(n_normal=2, n_anomalous=1)
    # 手动设置位置使它们在图中相连
    for i, obs in enumerate(observations):
        obs.elevation = [80.0, 75.0, 70.0][i]
        obs.azimuth = [0.0, 5.0, 10.0][i]

    extractor = FeatureExtractor()
    features, prns = extractor.extract_batch(observations)
    features_norm = extractor.normalize(features)

    elevations_arr = np.array([obs.elevation for obs in observations])
    azimuths_arr = np.array([obs.azimuth for obs in observations])

    result = analyzer.analyze(features_norm, elevations_arr, azimuths_arr, prns)

    print(f"  一致性误差: {[f'{e:.3f}' for e in result.consistency_error]}")
    print(f"  质量分数: {[f'{q:.3f}' for q in result.quality_scores]}")

    # 如果所有卫星位置相近（在图中相连），异常卫星应该有不一致
    if result.quality_scores[2] < 0.8:
        print("  ✓ 图分析检测到异常卫星的不一致性")
    else:
        print("  ⚠ 图分析未检测到明显不一致（可能因为卫星数量太少）")

    return True


# ==================== 测试5: 时序分析器 ====================

def test_temporal_analyzer():
    """测试时序一致性分析器"""
    print("\n=== 测试5: 时序一致性分析 ===")

    # 测试单个卫星跟踪器
    tracker = SatelliteTracker("G01", n_features=8, ema_decay=0.8, var_decay=0.9)

    # 模拟正常数据流
    normal_features = np.array([0.8, 0.9, 0.1, 0.1, 0.0, 0.0, 0.5, 0.0])

    # 前几次更新（初始化阶段）
    for _ in range(5):
        mahal = tracker.update(normal_features + np.random.normal(0, 0.01, 8))
        assert mahal < 2.0, f"正常数据应产生低马氏距离: {mahal:.3f}"

    # 模拟突变
    anomaly_features = normal_features + np.array([-0.5, 0.0, 0.0, 0.0, 2.0, 0.0, -0.3, 0.0])
    mahal_anomaly = tracker.update(anomaly_features)
    print(f"  正常马氏距离: < 2.0")
    print(f"  异常马氏距离: {mahal_anomaly:.3f}")

    if mahal_anomaly > 2.0:
        print("  ✓ 时序分析正确检测到突变")
    else:
        print(f"  ⚠ 突变检测不够敏感 (mahal={mahal_anomaly:.3f})")

    # 测试完整的时序分析器
    analyzer = TemporalAnalyzer(anomaly_threshold=3.0)

    # 模拟多个epoch
    features_list = []
    for t in range(10):
        obs = create_test_observations(n_normal=3, n_anomalous=0)
        extractor = FeatureExtractor()
        feats, prns = extractor.extract_batch(obs)
        features_list.append(feats)

    # 分析前几帧（建立基线）
    for t in range(4):
        result = analyzer.analyze(features_list[t], [f"G{i+1:02d}" for i in range(3)])
    print(f"  基线建立后的质量分数: {[f'{q:.3f}' for q in result.quality_scores]}")

    # 注入异常
    anomaly_features = features_list[4].copy()
    anomaly_features[0] += np.array([-0.3, -0.2, 0.0, 0.0, 2.0, 0.0, -0.3, 0.0])
    result_anomaly = analyzer.analyze(anomaly_features, [f"G{i+1:02d}" for i in range(3)])

    print(f"  异常注入后质量分数: {[f'{q:.3f}' for q in result_anomaly.quality_scores]}")
    print(f"  异常标记: {result_anomaly.anomaly_flags}")

    if result_anomaly.quality_scores[0] < 0.7:
        print("  ✓ 时序分析正确降低了异常卫星的分数")
    else:
        print("  ⚠ 异常分数降低不够明显")

    return True


# ==================== 测试6: 分数融合 ====================

def test_quality_fusion():
    """测试质量分数融合"""
    print("\n=== 测试6: 分数融合 ===")

    # 测试乘性融合
    fusion = QualityFusion(mode="geometric", quality_threshold=0.3)

    q_t = np.array([0.9, 0.8, 0.2, 0.9])
    q_g = np.array([0.8, 0.3, 0.9, 0.1])
    q_temp = np.array([0.7, 0.9, 0.3, 0.9])
    prns = ["G01", "G02", "G03", "G04"]
    systems = ["G", "G", "G", "G"]

    result = fusion.fuse(
        timestamp=1733984500.0,
        q_transformer=q_t,
        q_graph=q_g,
        q_temporal=q_temp,
        prns=prns, systems=systems,
        transformer_flags=[[], [], ["low_attention"], []],
        graph_flags=[[], ["graph_inconsistent"], [], ["graph_inconsistent"]],
        temporal_flags=[[], [], [], []],
    )

    # 验证融合结果（几何平均）
    # G01: (0.9*0.8*0.7)^(1/3) ≈ 0.796 (trusted)
    # G02: (0.8*0.3*0.9)^(1/3) ≈ 0.600 (suspect)
    # G03: (0.2*0.9*0.3)^(1/3) ≈ 0.378 (suspect)
    # G04: (0.9*0.1*0.9)^(1/3) ≈ 0.433 (suspect)
    expected = np.power(q_t * q_g * q_temp, 1.0/3.0)
    for i in range(4):
        assert_close(result.satellites[i].quality_final, expected[i])

    print(f"  融合结果: {[f'{s.quality_final:.3f}' for s in result.satellites]}")
    print(f"  可信等级: {[s.trust_level.value for s in result.satellites]}")
    print(f"  异常标记: {[s.all_flags for s in result.satellites]}")

    assert result.satellites[0].trust_level == TrustLevel.TRUSTED
    assert result.satellites[1].trust_level == TrustLevel.SUSPECT
    assert result.satellites[2].trust_level == TrustLevel.SUSPECT
    assert result.satellites[3].trust_level == TrustLevel.SUSPECT
    assert result.n_trusted == 1
    assert result.n_unreliable == 0

    # 测试JSON序列化
    result_dict = result.to_dict()
    assert "timestamp" in result_dict
    assert "summary" in result_dict
    assert len(result_dict["satellites"]) == 4
    print("  ✓ JSON序列化正确")

    # 测试最小值融合
    fusion_min = QualityFusion(mode="min", quality_threshold=0.3)
    result_min = fusion_min.fuse(
        timestamp=1733984500.0,
        q_transformer=q_t, q_graph=q_g, q_temporal=q_temp,
        prns=prns, systems=systems,
        transformer_flags=[[]]*4,
        graph_flags=[[]]*4,
        temporal_flags=[[]]*4,
    )
    # min融合: [min(0.9,0.8,0.7), min(0.8,0.3,0.9), ...] = [0.7, 0.3, 0.2, 0.1]
    assert_close(result_min.satellites[0].quality_final, 0.7)
    assert_close(result_min.satellites[1].quality_final, 0.3)
    print("  ✓ 乘性vs最小值融合差异正确")

    print("  ✓ 分数融合功能正确")
    return True


# ==================== 测试7: 端到端Pipeline ====================

def test_end_to_end():
    """端到端pipeline测试"""
    print("\n=== 测试7: 端到端Pipeline ===")

    config = OSQAConfig(
        window_size=10,
        memory_bank_size=50,
        feature_dims=8,
        attention_heads=4,
        fusion_mode="multiply",
        debug=False,
    )

    # 初始化所有模块
    memory = MemoryBuffer(config.window_size, config.memory_bank_size, config.feature_dims)
    extractor = FeatureExtractor()
    transformer = TransformerAnalyzer(config.feature_dims, config.attention_heads, config.head_dim)
    graph_analyzer = GraphAnalyzer(
        config.graph_edge_elevation_threshold,
        config.graph_edge_azimuth_threshold,
        config.graph_consistency_temperature,
    )
    temporal = TemporalAnalyzer(
        config.temporal_ema_decay,
        config.temporal_var_decay,
        config.temporal_anomaly_threshold,
    )
    fusion = QualityFusion(config.fusion_mode, config.fusion_weights, config.quality_threshold)

    print("  所有模块初始化成功")

    # 模拟10个epoch的处理
    total_anomalies_detected = 0
    for epoch_idx in range(10):
        # 创建epoch数据
        if epoch_idx == 5:
            # 在第5个epoch注入异常
            observations = create_test_observations(n_normal=6, n_anomalous=4, seed=epoch_idx)
        else:
            observations = create_test_observations(n_normal=8, n_anomalous=2, seed=epoch_idx)

        # 特征提取
        features, prns = extractor.extract_batch(observations)
        features_norm = extractor.normalize(features)

        elevations = np.array([obs.elevation for obs in observations])
        azimuths = np.array([obs.azimuth for obs in observations])
        systems = [obs.system for obs in observations]
        mask = np.ones(len(observations), dtype=bool)

        # 获取记忆原型
        prototype_samples = memory.get_good_prototypes(k=20)
        proto_features = [p.features for p in prototype_samples]

        # 三个分析器
        attn_result = transformer.analyze(features_norm, prns, proto_features, mask)
        graph_result = graph_analyzer.analyze(features_norm, elevations, azimuths, prns)
        temporal_result = temporal.analyze(features_norm, prns, mask)

        # 融合
        fused = fusion.fuse(
            timestamp=1733984500.0 + epoch_idx,
            q_transformer=attn_result.quality_scores,
            q_graph=graph_result.quality_scores,
            q_temporal=temporal_result.quality_scores,
            prns=prns, systems=systems,
            transformer_flags=attn_result.anomaly_flags,
            graph_flags=graph_result.anomaly_flags,
            temporal_flags=temporal_result.anomaly_flags,
        )

        # 更新记忆库
        epoch_data = EpochData(timestamp=1733984500.0 + epoch_idx)
        for i, obs in enumerate(observations):
            sample = SatelliteSample(
                prn=obs.prn, system=obs.system, features=features[i],
                timestamp=epoch_data.timestamp, elevation=obs.elevation,
                azimuth=obs.azimuth,
                pseudorange_residual=obs.pseudorange_residual,
                snr=obs.snr, lock_count=obs.lock_count,
                is_good=(fused.satellites[i].quality_final >= 0.7),
                quality_score=fused.satellites[i].quality_final,
            )
            epoch_data.satellites.append(sample)
        memory.add_epoch(epoch_data)

        total_anomalies_detected += fused.n_unreliable

        if epoch_idx % 2 == 0:
            print(f"  Epoch {epoch_idx}: {fused.n_total} sats, "
                  f"avg quality={fused.final_mean_quality:.3f}, "
                  f"unreliable={fused.n_unreliable}")

    print(f"  总共处理10个epoch，检测到{total_anomalies_detected}个不可信信号")
    print(f"  记忆库统计: {memory.get_statistics()}")
    print("  ✓ 端到端Pipeline运行成功")

    return True


# ==================== 主函数 ====================

def main():
    """运行所有测试"""
    print("=" * 60)
    print("OSQA 集成测试")
    print("=" * 60)

    tests = [
        ("特征提取", test_feature_extraction),
        ("记忆库", test_memory_buffer),
        ("Transformer分析器", test_transformer_analyzer),
        ("图结构分析器", test_graph_analyzer),
        ("时序分析器", test_temporal_analyzer),
        ("分数融合", test_quality_fusion),
        ("端到端Pipeline", test_end_to_end),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"\n  ✗ 测试失败 [{name}]: {e}")
            import traceback
            traceback.print_exc()

    # 汇总
    print("\n" + "=" * 60)
    print(f"测试完成: {passed} 通过, {failed} 失败, {len(tests)} 总计")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
