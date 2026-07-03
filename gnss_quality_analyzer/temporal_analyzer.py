"""
temporal_analyzer.py — 时序一致性分析器
=======================================

对每颗卫星独立进行时序分析，检测测量值的异常突变。

=== 原理解释（面向初学者）===

1. 为什么需要时序分析？
   Transformer和图结构分析的是"空间"关系（卫星之间的对比），
   时序分析关注的是"时间"关系（同一颗卫星的历史变化）。

   正常信号随时间缓慢变化（卫星在天空中的运动是平滑的），
   如果某颗卫星的信号突然跳变 → 可能是异常（周跳、遮挡后重新出现等）。

2. EMA（指数移动平均）是什么？
   EMA是对历史值的加权平均，越近的数据权重越大：
     ema[t] = 0.8 * ema[t-1] + 0.2 * value[t]

   这意味着最新的值占20%的权重，历史的EMA占80%。
   EMA代表"预期的正常值"。

3. 马氏距离是什么？
   简单说就是"当前值偏离历史均值多少个标准差"：
     distance = (value - ema)^2 / variance

   如果distance > 阈值（如3.0 = 3个标准差），大概率是异常。
   正态分布下，3个标准差外的概率约为0.3%。

4. 能检测哪些异常？
   - **载波相位跳变(Cycle Slip)**：载波相位突然跳变整数个波长
     → 残差特征突然变化
   - **多路径效应的出现/消失**：车辆驶入/驶出城市峡谷
     → SNR和残差的突变
   - **卫星遮挡后重新出现**：信号短暂丢失后重新跟踪
     → lock_count归零又增长，初始测量不稳定

=== 算法步骤 ===

对每颗卫星维护:
  - ema_features: 8维特征的指数移动平均
  - var_features: 8维特征的指数移动方差
  - anomaly_count: 累计异常次数（连续异常比偶发异常更严重）

对每个新epoch:
  1. 更新EMA和方差
  2. 计算马氏距离
  3. 基于马氏距离计算质量分数
  4. 递减异常计数（连续异常会降低分数）

Author: Claude Code
Date: 2026-07-02
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class TemporalResult:
    """
    时序分析结果
    """
    # 每颗卫星的质量分数
    quality_scores: np.ndarray      # (N,) [0, 1]

    # 详细指标
    mahalanobis_distance: np.ndarray  # (N,) 马氏距离
    feature_jump_magnitude: np.ndarray  # (N,) 特征跳变幅度
    consecutive_anomalies: np.ndarray  # (N,) 连续异常计数

    # EMA状态（用于调试）
    ema_features: np.ndarray        # (N, 8) 当前EMA

    # 标记
    anomaly_flags: List[List[str]]


class SatelliteTracker:
    """
    单颗卫星的状态跟踪器

    维护该卫星的EMA统计和异常历史。
    """

    def __init__(self, prn: str, n_features: int = 8,
                 ema_decay: float = 0.8, var_decay: float = 0.9):
        self.prn = prn
        self.n_features = n_features
        self.ema_decay = ema_decay
        self.var_decay = var_decay

        # EMA状态
        self.ema = np.zeros(n_features)
        self.var = np.ones(n_features) * 0.1  # 初始小方差

        # 异常追踪
        self.consecutive_anomalies: int = 0
        self.total_updates: int = 0

        # 是否已初始化
        self.initialized = False

    def update(self, features: np.ndarray) -> float:
        """
        用新特征更新跟踪器，返回马氏距离

        Args:
            features: (n_features,) 新的特征向量

        Returns:
            马氏距离（标量）
        """
        if not self.initialized:
            # 第一次更新：直接设置EMA，不计算距离
            self.ema = features.copy()
            self.var = np.ones(self.n_features) * 0.01
            self.initialized = True
            self.total_updates = 1
            return 0.0

        # 计算马氏距离（在更新前）
        diff = features - self.ema
        mahalanobis = np.sum((diff ** 2) / (self.var + 1e-8)) / self.n_features

        # 更新EMA
        self.ema = self.ema_decay * self.ema + (1.0 - self.ema_decay) * features

        # 更新方差（使用EMA后的新diff）
        new_diff = features - self.ema
        self.var = self.var_decay * self.var + (1.0 - self.var_decay) * (new_diff ** 2)

        self.total_updates += 1
        return float(mahalanobis)

    def mark_anomaly(self, is_anomaly: bool):
        """
        标记是否为异常（更新异常计数）

        连续异常的累积效应会使质量分数进一步降低。
        """
        if is_anomaly:
            self.consecutive_anomalies += 1
        else:
            # 正常时，异常计数指数衰减
            self.consecutive_anomalies = max(0, self.consecutive_anomalies - 0.5)

    def get_quality_score(self, mahalanobis: float,
                           threshold: float = 3.0) -> float:
        """
        基于马氏距离和连续异常计数计算质量分数

        Args:
            mahalanobis: 当前马氏距离
            threshold: 异常阈值

        Returns:
            质量分数 [0, 1]
        """
        if not self.initialized:
            return 0.5  # 刚初始化，给中间分

        # 基于马氏距离的基础分数
        if mahalanobis <= threshold * 0.5:
            base_score = 1.0  # 非常正常
        elif mahalanobis <= threshold:
            # 在0.5*threshold到threshold之间，线性衰减
            base_score = 1.0 - 0.5 * (mahalanobis - 0.5 * threshold) / (0.5 * threshold)
        elif mahalanobis <= threshold * 2.0:
            # 在threshold到2*threshold之间，快速衰减
            base_score = 0.5 * (1.0 - (mahalanobis - threshold) / threshold)
        else:
            base_score = 0.0  # 严重的异常

        # 连续异常惩罚
        if self.consecutive_anomalies > 3:
            # 连续3次以上异常，大幅降权
            penalty = min(0.5, 0.1 * (self.consecutive_anomalies - 3))
            base_score *= (1.0 - penalty)

        return float(np.clip(base_score, 0.0, 1.0))


class TemporalAnalyzer:
    """
    时序一致性分析器

    并行跟踪所有卫星的时序变化，检测异常突变。
    """

    def __init__(self, ema_decay: float = 0.8, var_decay: float = 0.9,
                 anomaly_threshold: float = 3.0, n_features: int = 8):
        """
        Args:
            ema_decay: 特征EMA的衰减因子
            var_decay: 方差EMA的衰减因子
            anomaly_threshold: 马氏距离的异常阈值（默认3=约3个标准差）
            n_features: 特征维度
        """
        self.ema_decay = ema_decay
        self.var_decay = var_decay
        self.anomaly_threshold = anomaly_threshold
        self.n_features = n_features

        # 每颗卫星的独立跟踪器
        # key: PRN, value: SatelliteTracker
        self.trackers: Dict[str, SatelliteTracker] = {}

        # 统计
        self.total_analyses: int = 0
        self.total_jumps_detected: int = 0

    def analyze(self, features: np.ndarray,
                prns: List[str],
                mask: Optional[np.ndarray] = None
                ) -> TemporalResult:
        """
        分析一帧epoch中各卫星的时序一致性

        Args:
            features: (N, 8) 特征矩阵
            prns: 卫星PRN列表
            mask: (N,) 有效性mask（可选）

        Returns:
            TemporalResult: 时序分析结果
        """
        N = len(prns)

        if mask is None:
            mask = np.ones(N, dtype=bool)

        mahalanobis_dist = np.zeros(N)
        feature_jump = np.zeros(N)
        quality_scores = np.zeros(N)
        ema_features = np.zeros((N, self.n_features))
        anomaly_flags = [[] for _ in range(N)]
        consecutive_anomalies = np.zeros(N)

        for i in range(N):
            if not mask[i]:
                quality_scores[i] = 0.0
                anomaly_flags[i].append("invalid")
                continue

            prn = prns[i]
            feat = features[i]

            # 获取或创建该卫星的跟踪器
            if prn not in self.trackers:
                self.trackers[prn] = SatelliteTracker(
                    prn, self.n_features, self.ema_decay, self.var_decay
                )

            tracker = self.trackers[prn]

            # 更新跟踪器并获取马氏距离
            mahal = tracker.update(feat)
            mahalanobis_dist[i] = mahal

            # 判断是否为异常
            is_anomaly = mahal > self.anomaly_threshold
            tracker.mark_anomaly(is_anomaly)

            # 计算特征跳变幅度
            if tracker.total_updates > 1:
                feature_jump[i] = np.max(np.abs(feat - tracker.ema))
            else:
                feature_jump[i] = 0.0

            # 获取质量分数
            quality_scores[i] = tracker.get_quality_score(mahal, self.anomaly_threshold)

            # 记录EMA
            ema_features[i] = tracker.ema.copy()

            # 连续异常计数
            consecutive_anomalies[i] = tracker.consecutive_anomalies

            # 异常标记
            if is_anomaly:
                anomaly_flags[i].append("temporal_jump")
                self.total_jumps_detected += 1
            if tracker.consecutive_anomalies > 3:
                anomaly_flags[i].append(f"persistent_anomaly({int(tracker.consecutive_anomalies)})")
            if tracker.total_updates <= 3:
                anomaly_flags[i].append("few_samples")

        self.total_analyses += 1

        return TemporalResult(
            quality_scores=quality_scores,
            mahalanobis_distance=mahalanobis_dist,
            feature_jump_magnitude=feature_jump,
            consecutive_anomalies=consecutive_anomalies,
            ema_features=ema_features,
            anomaly_flags=anomaly_flags,
        )

    def reset_satellite(self, prn: str):
        """
        重置某卫星的跟踪器

        当检测到卫星失锁后重新出现时调用，
        因为失锁后EMA不再有效。

        Args:
            prn: 卫星PRN号
        """
        if prn in self.trackers:
            del self.trackers[prn]

    def get_tracker_info(self, prn: str) -> Optional[dict]:
        """
        获取某卫星跟踪器的详细信息

        Args:
            prn: 卫星PRN号

        Returns:
            跟踪器信息字典，若不存在则返回None
        """
        if prn not in self.trackers:
            return None
        tracker = self.trackers[prn]
        return {
            "prn": prn,
            "total_updates": tracker.total_updates,
            "consecutive_anomalies": tracker.consecutive_anomalies,
            "ema_mean": float(np.mean(tracker.ema)),
            "var_mean": float(np.mean(tracker.var)),
        }

    def get_statistics(self) -> dict:
        """获取分析器的统计信息"""
        n_trackers = len(self.trackers)
        total_updates = sum(t.total_updates for t in self.trackers.values())

        return {
            "n_tracked_satellites": n_trackers,
            "total_updates_across_all": total_updates,
            "avg_updates_per_satellite": total_updates / max(n_trackers, 1),
            "total_analyses": self.total_analyses,
            "total_jumps_detected": self.total_jumps_detected,
            "jump_rate": self.total_jumps_detected / max(self.total_analyses, 1),
        }
