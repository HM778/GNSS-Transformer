"""
quality_fusion.py — 多分析器分数融合
====================================

将TransformerAnalyzer、GraphAnalyzer、TemporalAnalyzer的
质量分数融合为最终的信号质量评估。

=== 融合策略说明 ===

1. 几何平均融合 (geometric) — 默认策略（类似trbin的pow(p_rel, 1/6)）
   三个分数的几何平均: q_final = (q_t * q_g * q_temp)^(1/3)
   特点：平衡各分析器的影响，避免乘积过小的问题。
   适合：大多数场景，城市峡谷和开阔天空都适用。

2. 乘性融合 (multiply) — 最保守策略
   直接乘积: q_final = q_transformer * q_graph * q_temporal
   特点：最严格，任一分析器发现问题都会显著降低最终分数。
   适合：对精度要求极高的场景。

3. 最小值融合 (min)
   取最小值: q_final = min(q_transformer, q_graph, q_temporal)
   特点：比乘性更保守（因为最小值通常比乘积更小）。
   适合：高精度定位场景。

4. 加权平均 (weighted)
   加权求和: q_final = w1*q_t + w2*q_g + w3*q_temp
   特点：最平滑，不会因为单个分析器的低分就直接否定一颗卫星。
   适合：开阔天空场景，大多数信号可信。

=== 输出格式 ===

除了数值分数，还提供：
- 异常标记汇总（哪些分析器发现了异常）
- 信号可信度分级（trusted/suspect/unreliable）
- 可视化数据（用于调试和展示）

Author: Claude Code
Date: 2026-07-02
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class TrustLevel(Enum):
    """
    信号可信度等级

    trusted:   质量分 >= 0.7，三个分析器都认为可信
    suspect:   质量分 0.3-0.7，至少一个分析器有疑虑
    unreliable: 质量分 < 0.3，至少一个分析器认为不可信
    """
    TRUSTED = "trusted"
    SUSPECT = "suspect"
    UNRELIABLE = "unreliable"


@dataclass
class SatelliteQuality:
    """
    单颗卫星的完整质量评估结果
    """
    prn: str                        # 卫星PRN号
    system: str                     # 卫星系统

    # 各分析器的质量分数 [0, 1]
    quality_transformer: float
    quality_graph: float
    quality_temporal: float

    # 融合后的最终分数
    quality_final: float

    # 可信度等级
    trust_level: TrustLevel

    # 所有分析器的异常标记汇总
    all_flags: List[str] = field(default_factory=list)

    # 元数据
    snr: float = 0.0
    elevation: float = 0.0
    azimuth: float = 0.0
    pseudorange_residual: float = 0.0

    def to_dict(self) -> dict:
        """转换为字典（便于JSON序列化）"""
        return {
            "prn": self.prn,
            "system": self.system,
            "quality": round(self.quality_final, 4),
            "trust_level": self.trust_level.value,
            "details": {
                "transformer": round(self.quality_transformer, 4),
                "graph": round(self.quality_graph, 4),
                "temporal": round(self.quality_temporal, 4),
            },
            "flags": self.all_flags,
            "snr": round(self.snr, 1),
            "elevation": round(self.elevation, 1),
            "azimuth": round(self.azimuth, 1),
            "pseudorange_residual": round(self.pseudorange_residual, 3),
        }


@dataclass
class FusedResult:
    """
    一个epoch的融合结果
    """
    timestamp: float
    satellites: List[SatelliteQuality]

    # 汇总统计
    n_total: int
    n_trusted: int
    n_suspect: int
    n_unreliable: int

    # 各分析器统计
    transformer_mean_quality: float
    graph_mean_quality: float
    temporal_mean_quality: float
    final_mean_quality: float

    # 融合参数
    fusion_mode: str = "geometric"

    def to_dict(self) -> dict:
        """转换为字典（便于JSON序列化和ROS消息发布）"""
        return {
            "timestamp": self.timestamp,
            "satellites": {s.prn: s.to_dict() for s in self.satellites},
            "summary": {
                "n_total": self.n_total,
                "n_trusted": self.n_trusted,
                "n_suspect": self.n_suspect,
                "n_unreliable": self.n_unreliable,
            },
            "mean_quality": {
                "transformer": round(self.transformer_mean_quality, 4),
                "graph": round(self.graph_mean_quality, 4),
                "temporal": round(self.temporal_mean_quality, 4),
                "final": round(self.final_mean_quality, 4),
            },
            "fusion_mode": self.fusion_mode,
        }

    def get_trusted_prns(self) -> List[str]:
        """获取可信卫星的PRN列表"""
        return [s.prn for s in self.satellites if s.trust_level == TrustLevel.TRUSTED]

    def get_unreliable_prns(self) -> List[str]:
        """获取不可信卫星的PRN列表"""
        return [s.prn for s in self.satellites if s.trust_level == TrustLevel.UNRELIABLE]

    def get_quality_weights(self) -> Dict[str, float]:
        """
        获取可用于gnssfgo因子权重调整的质量权重字典

        key=PRN, value=质量权重 [0, 1]
        gnssfgo可以将这些权重乘到对应卫星的sqrt_info上。
        """
        return {s.prn: s.quality_final for s in self.satellites}


class QualityFusion:
    """
    多分析器质量分数融合器

    将三个分析器的独立评估结果融合为统一的信号质量分数。
    """

    def __init__(self, mode: str = "multiply",
                 weights: Optional[Dict[str, float]] = None,
                 quality_threshold: float = 0.3):
        """
        Args:
            mode: 融合模式 — "multiply" / "min" / "weighted"
            weights: 加权平均模式的权重 {"transformer": 0.4, "graph": 0.35, "temporal": 0.25}
            quality_threshold: 不可信信号的阈值
        """
        self.mode = mode
        self.weights = weights or {
            "transformer": 0.4,
            "graph": 0.35,
            "temporal": 0.25,
        }
        self.quality_threshold = quality_threshold

        # 历史融合统计
        self.fusion_history: List[FusedResult] = []

    def fuse(self, timestamp: float,
             q_transformer: np.ndarray,
             q_graph: np.ndarray,
             q_temporal: np.ndarray,
             prns: List[str],
             systems: List[str],
             transformer_flags: List[List[str]],
             graph_flags: List[List[str]],
             temporal_flags: List[List[str]],
             snr_values: Optional[np.ndarray] = None,
             elevation_values: Optional[np.ndarray] = None,
             azimuth_values: Optional[np.ndarray] = None,
             residual_values: Optional[np.ndarray] = None,
             ) -> FusedResult:
        """
        融合三个分析器的质量分数

        Args:
            timestamp: GPS时间戳
            q_transformer: (N,) Transformer分析器的质量分数
            q_graph: (N,) 图分析器的质量分数
            q_temporal: (N,) 时序分析器的质量分数
            prns: 卫星PRN列表
            systems: 卫星系统列表
            transformer_flags: Transformer的异常标记
            graph_flags: 图分析的异常标记
            temporal_flags: 时序分析的异常标记
            snr_values: (N,) SNR值（可选）
            elevation_values: (N,) 仰角（可选）
            azimuth_values: (N,) 方位角（可选）
            residual_values: (N,) 伪距残差（可选）

        Returns:
            FusedResult: 融合结果
        """
        N = len(prns)

        # 融合策略
        # 确保所有分数都在[eps, 1]范围内（避免log(0)和除零）
        eps = 1e-6
        qt = np.clip(q_transformer, eps, 1.0)
        qg = np.clip(q_graph, eps, 1.0)
        qtmp = np.clip(q_temporal, eps, 1.0)

        if self.mode == "geometric":
            # 几何平均: (q1 * q2 * q3)^(1/3)
            # 类似trbin的pow(p_rel, 1/6)方法，平衡各分析器的影响
            q_final = np.power(qt * qg * qtmp, 1.0 / 3.0)
        elif self.mode == "multiply":
            # 直接乘积: 最保守，任一低分都会大幅降低总分
            q_final = q_transformer * q_graph * q_temporal
        elif self.mode == "min":
            q_final = np.minimum(np.minimum(q_transformer, q_graph), q_temporal)
        elif self.mode == "weighted":
            q_final = (self.weights["transformer"] * q_transformer +
                       self.weights["graph"] * q_graph +
                       self.weights["temporal"] * q_temporal)
        else:
            raise ValueError(f"Unknown fusion mode: {self.mode}")

        # 构建每颗卫星的完整质量评估
        satellites = []
        for i in range(N):
            # 汇总所有异常标记
            all_flags = []
            all_flags.extend(transformer_flags[i])
            all_flags.extend(graph_flags[i])
            all_flags.extend(temporal_flags[i])

            # 移除重复标记
            all_flags = list(dict.fromkeys(all_flags))

            # 判断可信度等级
            if q_final[i] >= 0.7:
                trust_level = TrustLevel.TRUSTED
            elif q_final[i] >= self.quality_threshold:
                trust_level = TrustLevel.SUSPECT
            else:
                trust_level = TrustLevel.UNRELIABLE

            sat = SatelliteQuality(
                prn=prns[i],
                system=systems[i] if i < len(systems) else "?",
                quality_transformer=float(q_transformer[i]),
                quality_graph=float(q_graph[i]),
                quality_temporal=float(q_temporal[i]),
                quality_final=float(q_final[i]),
                trust_level=trust_level,
                all_flags=all_flags,
                snr=float(snr_values[i]) if snr_values is not None else 0.0,
                elevation=float(elevation_values[i]) if elevation_values is not None else 0.0,
                azimuth=float(azimuth_values[i]) if azimuth_values is not None else 0.0,
                pseudorange_residual=float(residual_values[i]) if residual_values is not None else 0.0,
            )
            satellites.append(sat)

        # 汇总统计
        n_trusted = sum(1 for s in satellites if s.trust_level == TrustLevel.TRUSTED)
        n_suspect = sum(1 for s in satellites if s.trust_level == TrustLevel.SUSPECT)
        n_unreliable = sum(1 for s in satellites if s.trust_level == TrustLevel.UNRELIABLE)

        result = FusedResult(
            timestamp=timestamp,
            satellites=satellites,
            n_total=N,
            n_trusted=n_trusted,
            n_suspect=n_suspect,
            n_unreliable=n_unreliable,
            transformer_mean_quality=float(np.mean(q_transformer)),
            graph_mean_quality=float(np.mean(q_graph)),
            temporal_mean_quality=float(np.mean(q_temporal)),
            final_mean_quality=float(np.mean(q_final)),
            fusion_mode=self.mode,
        )

        self.fusion_history.append(result)

        # 只保留最近1000个结果
        if len(self.fusion_history) > 1000:
            self.fusion_history = self.fusion_history[-1000:]

        return result

    def get_recent_results(self, n: int = 10) -> List[FusedResult]:
        """获取最近n个融合结果"""
        return self.fusion_history[-n:]

    def get_long_term_stats(self) -> dict:
        """
        获取长期统计信息

        可用于评估各分析器的贡献度，指导融合权重的调整。
        """
        if not self.fusion_history:
            return {}

        n_epochs = len(self.fusion_history)
        t_qualities = [r.transformer_mean_quality for r in self.fusion_history]
        g_qualities = [r.graph_mean_quality for r in self.fusion_history]
        tmp_qualities = [r.temporal_mean_quality for r in self.fusion_history]
        f_qualities = [r.final_mean_quality for r in self.fusion_history]

        # 各分析器分数的标准差（标准差越大说明该分析器区分度越高）
        t_std = float(np.std(t_qualities))
        g_std = float(np.std(g_qualities))
        tmp_std = float(np.std(tmp_qualities))

        # 各分析器标记的异常比例
        total_anomalies = sum(
            sum(1 for s in r.satellites if s.all_flags) for r in self.fusion_history
        )
        total_satellites = sum(r.n_total for r in self.fusion_history)

        return {
            "n_epochs_analyzed": n_epochs,
            "avg_quality": {
                "transformer": float(np.mean(t_qualities)),
                "graph": float(np.mean(g_qualities)),
                "temporal": float(np.mean(tmp_qualities)),
                "final": float(np.mean(f_qualities)),
            },
            "quality_std": {
                "transformer": t_std,
                "graph": g_std,
                "temporal": tmp_std,
            },
            "anomaly_rate": total_anomalies / max(total_satellites, 1),
        }
