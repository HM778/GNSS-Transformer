"""
memory_buffer.py — 滑动窗口记忆库
=================================

这是OSQA的"记忆"模块，类似于trbin的200-epoch历史窗口，
但存储的是连续特征向量（而非离散桶统计）。

类比理解：
  trbin方案: 把信号质量分成"好/中/差"这样的桶来统计
  OSQA方案: 直接记住历史上"好信号"的特征是什么样子，
            新信号来了就和记忆中的"好信号"对比，相似就信任，不相似就怀疑

核心数据结构：
  1. good_samples: 存储历史上被确认为"好"的特征向量（残差<0.05m的测量）
  2. epoch_buffer: 环形缓冲区，存储最近N个epoch的完整数据
  3. per_sat_stats: 每颗卫星的运行统计（在线更新的均值和标准差）

Author: Claude Code
Date: 2026-07-02
"""

import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class SatelliteSample:
    """
    单个卫星在一次epoch中的采样数据

    存储特征向量和元数据，用于记忆库的检索和对比。
    """
    # 卫星标识
    prn: str          # 卫星PRN号，如 'G05', 'C108'
    system: str       # 卫星系统: 'G'(GPS), 'R'(GLONASS), 'E'(Galileo), 'C'(BeiDou)

    # 特征向量（8维numpy数组）
    features: np.ndarray

    # 元数据
    timestamp: float        # GPS时间（秒）
    elevation: float        # 仰角（度）
    azimuth: float          # 方位角（度）
    pseudorange_residual: float   # 伪距残差（米）
    snr: float              # 信噪比（dB-Hz）
    lock_count: int         # 连续锁定计数

    # 质量标签（由后续优化结果反馈）
    is_good: bool = True    # 是否被确认为"好"样本
    quality_score: float = 1.0  # 最终质量分数

    def get_feature_dict(self) -> Dict[str, float]:
        """将特征向量转换为可读的字典"""
        feature_names = [
            "snr_norm", "elev_sin", "azim_cos", "azim_sin",
            "psr_residual", "cp_residual", "lock_count_norm", "elev_rate"
        ]
        return {name: float(self.features[i]) for i, name in enumerate(feature_names)}


@dataclass
class EpochData:
    """
    单个epoch的完整数据

    包含该epoch的所有卫星观测，以及元数据。
    """
    timestamp: float                    # GPS时间（秒）
    satellites: List[SatelliteSample] = field(default_factory=list)
    receiver_ecef: Optional[np.ndarray] = None  # 接收机ECEF坐标（3,）
    receiver_lla: Optional[np.ndarray] = None   # 接收机经纬高（3,）

    @property
    def n_satellites(self) -> int:
        return len(self.satellites)


class RunningStats:
    """
    在线运行统计 — 使用指数移动平均(EMA)更新均值和方差

    为什么用EMA而非累积统计？
    - 信号环境是时变的（开阔地→城市峡谷），旧的统计会过时
    - EMA自动给近期数据更高权重，适应环境变化
    - 计算效率高，只需O(1)更新
    """

    def __init__(self, n_dims: int, decay: float = 0.9):
        """
        Args:
            n_dims: 特征维度
            decay: EMA衰减因子，越大越保守（变化慢）
        """
        self.n_dims = n_dims
        self.decay = decay
        self.mean = np.zeros(n_dims)
        self.var = np.ones(n_dims)  # 初始方差=1（而非0，避免除零）
        self.n_updates = 0

    def update(self, x: np.ndarray):
        """
        用新样本更新统计量

        使用EMA公式:
          mean_new = decay * mean_old + (1-decay) * x
          var_new  = decay * var_old  + (1-decay) * (x - mean_new)^2

        Args:
            x: 特征向量 (n_dims,) numpy数组
        """
        if self.n_updates == 0:
            # 第一个样本：直接设置
            self.mean = x.copy()
            self.var = np.ones(self.n_dims) * 0.01  # 小方差初始化
        else:
            # EMA更新
            self.mean = self.decay * self.mean + (1.0 - self.decay) * x
            diff = x - self.mean
            self.var = self.decay * self.var + (1.0 - self.decay) * (diff ** 2)

        self.n_updates += 1

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """
        对特征进行z-score归一化

        z = (x - mean) / sqrt(var + epsilon)

        Args:
            x: 原始特征向量

        Returns:
            归一化后的特征向量
        """
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)

    def get_std(self) -> np.ndarray:
        """获取标准差"""
        return np.sqrt(self.var)


class MemoryBuffer:
    """
    滑动窗口记忆库

    维护两类记忆：
    1. epoch_buffer: 最近N个epoch的完整数据（环形缓冲区）
    2. good_samples: 历史上被确认为"好"的特征样本（双端队列，FIFO）

    记忆库的作用：
    - 为新信号提供"正常基准"：对比记忆中的好样本判断新信号是否正常
    - 支持时间相关的分析：获取历史epoch用于时序分析
    - 支持在线统计更新：维护每颗卫星和全局的特征统计
    """

    def __init__(self, window_size: int = 50, memory_bank_size: int = 200,
                 feature_dims: int = 8, ema_decay: float = 0.9):
        """
        Args:
            window_size: 滑动窗口大小（epochs）
            memory_bank_size: 好样本记忆库最大容量
            feature_dims: 特征维度
            ema_decay: EMA衰减因子
        """
        self.window_size = window_size
        self.memory_bank_size = memory_bank_size
        self.feature_dims = feature_dims
        self.ema_decay = ema_decay

        # 环形缓冲区：存储最近的epoch数据
        self.epoch_buffer: deque = deque(maxlen=window_size)

        # 好样本记忆库：存储被确认为"好"的SatelliteSample
        self.good_samples: deque = deque(maxlen=memory_bank_size)

        # 全局特征统计
        self.global_stats = RunningStats(feature_dims, ema_decay)

        # 每颗卫星的独立统计
        # key: PRN (如'G05'), value: RunningStats
        self.per_satellite_stats: Dict[str, RunningStats] = {}

        # 每颗卫星的时序特征历史（用于时序分析）
        # key: PRN, value: deque of (timestamp, features)
        self.per_satellite_history: Dict[str, deque] = {}

        # 统计计数器
        self.total_epochs_processed: int = 0
        self.total_samples_seen: int = 0

    def add_epoch(self, epoch: EpochData):
        """
        添加一个新epoch的数据到记忆库

        这个方法是OSQA数据流的入口。每收到一个新epoch就调用一次。

        处理流程：
        1. 将epoch加入环形缓冲区
        2. 对每个卫星样本更新统计量
        3. 将"好"样本加入记忆库

        Args:
            epoch: 包含所有卫星观测的EpochData
        """
        # 1. 加入环形缓冲区
        self.epoch_buffer.append(epoch)
        self.total_epochs_processed += 1

        # 2. 处理每个卫星
        for sat in epoch.satellites:
            self.total_samples_seen += 1

            # 更新全局统计
            self.global_stats.update(sat.features)

            # 更新该卫星的独立统计
            if sat.prn not in self.per_satellite_stats:
                self.per_satellite_stats[sat.prn] = RunningStats(
                    self.feature_dims, self.ema_decay
                )
                self.per_satellite_history[sat.prn] = deque(maxlen=self.window_size)

            self.per_satellite_stats[sat.prn].update(sat.features)
            self.per_satellite_history[sat.prn].append(
                (sat.timestamp, sat.features.copy())
            )

            # 3. 将"好"样本加入记忆库
            if sat.is_good:
                self.good_samples.append(sat)

    def get_recent_epochs(self, n: int) -> List[EpochData]:
        """
        获取最近n个epoch的数据

        Args:
            n: 要获取的epoch数量

        Returns:
            最近n个EpochData列表（按时间顺序）
        """
        if n >= len(self.epoch_buffer):
            return list(self.epoch_buffer)
        return list(self.epoch_buffer)[-n:]

    def get_latest_epoch(self) -> Optional[EpochData]:
        """获取最新的epoch"""
        if len(self.epoch_buffer) == 0:
            return None
        return self.epoch_buffer[-1]

    def get_good_prototypes(self, k: int = 50) -> List[SatelliteSample]:
        """
        获取k个最典型的"好"样本（原型）

        用于Transformer注意力计算中的记忆增强：
        新信号除了和其他卫星对比，还要和这些"好原型"对比。

        选择策略：从记忆库中均匀采样（确保多样性）

        Args:
            k: 要获取的原型数量

        Returns:
            k个SatelliteSample列表
        """
        if len(self.good_samples) <= k:
            return list(self.good_samples)

        # 均匀采样
        indices = np.linspace(0, len(self.good_samples) - 1, k, dtype=int)
        return [self.good_samples[i] for i in indices]

    def get_satellite_history(self, prn: str) -> List[Tuple[float, np.ndarray]]:
        """
        获取某卫星的时序历史

        Args:
            prn: 卫星PRN号

        Returns:
            (timestamp, features) 列表，按时间排序
        """
        if prn not in self.per_satellite_history:
            return []
        return list(self.per_satellite_history[prn])

    def get_satellite_stats(self, prn: str) -> Optional[RunningStats]:
        """
        获取某卫星的运行统计

        Args:
            prn: 卫星PRN号

        Returns:
            该卫星的RunningStats，若从未出现则返回None
        """
        return self.per_satellite_stats.get(prn, None)

    def normalize_features(self, features: np.ndarray, prn: Optional[str] = None) -> np.ndarray:
        """
        对特征向量进行归一化

        优先使用该卫星的独立统计，若不可用则用全局统计。

        Args:
            features: 原始特征向量 (n_dims,)
            prn: 可选的卫星PRN号

        Returns:
            归一化后的特征向量
        """
        if prn and prn in self.per_satellite_stats:
            return self.per_satellite_stats[prn].normalize(features)
        return self.global_stats.normalize(features)

    def mark_satellite_quality(self, prn: str, timestamp: float, quality: float,
                                residual: float):
        """
        根据质量分数和残差标记样本的"好/坏"

        这个方法在收到FGO反馈后调用，用于更新记忆库中样本的标签。

        Args:
            prn: 卫星PRN号
            timestamp: 时间戳
            quality: 质量分数 [0, 1]
            residual: FGO优化后的残差（米）
        """
        # 在epoch_buffer中找到对应样本并更新
        for epoch in self.epoch_buffer:
            if abs(epoch.timestamp - timestamp) < 0.01:  # 10ms容差
                for sat in epoch.satellites:
                    if sat.prn == prn:
                        sat.quality_score = quality
                        sat.is_good = (residual < 0.05)  # 和trbin相同的5cm阈值
                        # 如果之前被标记为坏但现在确认为好，加入记忆库
                        if sat.is_good and sat not in self.good_samples:
                            self.good_samples.append(sat)
                        return

    def get_statistics(self) -> dict:
        """
        获取记忆库的统计摘要

        Returns:
            包含各类统计信息的字典
        """
        return {
            "total_epochs_processed": self.total_epochs_processed,
            "total_samples_seen": self.total_samples_seen,
            "current_buffer_size": len(self.epoch_buffer),
            "good_samples_count": len(self.good_samples),
            "tracked_satellites": len(self.per_satellite_stats),
            "global_feature_mean": self.global_stats.mean.tolist(),
            "global_feature_std": self.global_stats.get_std().tolist(),
        }

    def clear(self):
        """清空所有记忆（重置）"""
        self.epoch_buffer.clear()
        self.good_samples.clear()
        self.per_satellite_stats.clear()
        self.per_satellite_history.clear()
        self.global_stats = RunningStats(self.feature_dims, self.ema_decay)
        self.total_epochs_processed = 0
        self.total_samples_seen = 0
