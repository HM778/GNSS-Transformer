"""
feature_extractor.py — 特征提取器
================================

从原始GNSS观测数据中提取8维归一化特征向量。
这是OSQA的"感知层"，负责将原始的SNR、仰角、方位角等数据
转化为Transformer可以处理的连续特征向量。

特征设计原则：
1. 连续性：所有特征都是连续值，避免离散化造成的信息损失
2. 物理意义：每个特征都有明确的物理含义
3. 几何不变性：角度的sin/cos编码避免方位角和仰角的不连续性
4. 归一化：特征值量纲统一，便于注意力计算

8维特征详情：
  [0] SNR_norm:           归一化信噪比（反映信号强度）
  [1] elevation_sin:      sin(仰角)（反映信号穿过大气层的厚度）
  [2] azimuth_cos:        cos(方位角)（卫星在天空中的东西方向位置）
  [3] azimuth_sin:        sin(方位角)（卫星在天空中的南北方向位置）
  [4] pseudorange_residual: 伪距残差（SPP解算后，反映测量误差的大小）
  [5] carrier_residual:  载波相位残差（若有双频数据，反映更精细的测量误差）
  [6] lock_count_norm:   归一化连续锁定计数（反映跟踪稳定性）
  [7] elevation_rate:    仰角变化率（反映卫星相对运动速度）

为什么用sin/cos编码角度？
  问题：方位角350°和10°相差只有20°，但直接相减得到340°
  解决：用(sin, cos)对编码，两个角度之间的真实差异可以通过向量距离计算
  例如：(sin(350°), cos(350°)) ≈ (-0.17, 0.98)
       (sin(10°), cos(10°))   ≈ (0.17, 0.98)
       两者接近，正确反映了它们之间的真实关系

Author: Claude Code
Date: 2026-07-02
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class RawObservation:
    """
    单颗卫星的单频原始观测数据

    这是从ublox接收机输出中提取的原始观测量。
    在使用前需要进行各种校正（电离层、对流层、卫星钟差等）。
    """
    prn: str                # 卫星PRN号，如 'G05'
    system: str             # 卫星系统: 'G'/'R'/'E'/'C'
    timestamp: float        # GPS时间（秒）

    # 信号特征
    snr: float              # 信噪比 (dB-Hz)，范围通常15-55
    elevation: float        # 仰角（度），0(地平线)-90(天顶)
    azimuth: float          # 方位角（度），0-360

    # 观测量（L1/L2）
    pseudorange_l1: float = 0.0   # L1伪距（米）
    pseudorange_l2: float = 0.0   # L2伪距（米，若不可用则为0）
    carrier_phase_l1: float = 0.0  # L1载波相位（周）
    carrier_phase_l2: float = 0.0  # L2载波相位（周，若不可用则为0）
    doppler_l1: float = 0.0        # L1多普勒（Hz）

    # 跟踪信息
    lock_count: int = 0     # 连续锁定计数（越大表示跟踪越稳定）
    lli_flags: int = 0      # 失锁指示器（非0表示可能有周跳）

    # 校正后的残差（由SPP解算后填入）
    pseudorange_residual: float = 0.0
    carrier_residual: float = 0.0

    # 接收机位置（用于后续计算）
    receiver_ecef: Optional[np.ndarray] = None  # (3,) ECEF坐标
    receiver_lla: Optional[np.ndarray] = None   # (3,) 经纬高

    @property
    def has_dual_frequency(self) -> bool:
        """是否拥有双频数据"""
        return self.pseudorange_l2 > 0 and self.carrier_phase_l2 > 0

    @property
    def has_carrier_phase(self) -> bool:
        """L1载波相位是否可用"""
        return self.carrier_phase_l1 > 0 and self.lock_count > 0

    @property
    def is_cycle_slip_suspected(self) -> bool:
        """是否有周跳嫌疑（基于LLI标志）"""
        return self.lli_flags != 0


class FeatureExtractor:
    """
    特征提取器

    将原始GNSS观测转换为标准化的8维特征向量。
    支持在线归一化（使用运行时统计而非预设参数）。

    使用示例:
        extractor = FeatureExtractor()
        features = extractor.extract_single(obs)
        # features 是一个 (8,) numpy数组
    """

    def __init__(self, epsilon: float = 1e-8):
        """
        Args:
            epsilon: 防止除零的小值
        """
        self.epsilon = epsilon

        # 特征统计（由MemoryBuffer提供，用于在线归一化）
        self.feature_means: Optional[np.ndarray] = None  # (8,)
        self.feature_stds: Optional[np.ndarray] = None   # (8,)

        # 统计计数器
        self.extraction_count: int = 0

    def set_statistics(self, means: np.ndarray, stds: np.ndarray):
        """
        设置归一化统计参数

        Args:
            means: 特征均值 (8,)
            stds: 特征标准差 (8,)
        """
        self.feature_means = means
        self.feature_stds = stds

    def extract_single(self, obs: RawObservation, prev_elevation: Optional[float] = None,
                       prev_timestamp: Optional[float] = None) -> np.ndarray:
        """
        从单个原始观测中提取特征向量

        Args:
            obs: 原始观测
            prev_elevation: 该卫星上一epoch的仰角（用于计算变化率）
            prev_timestamp: 该卫星上一epoch的时间戳（用于计算变化率）

        Returns:
            8维特征向量 numpy数组
        """
        features = np.zeros(8, dtype=np.float32)

        # [0] SNR归一化
        # SNR典型范围15-55 dB-Hz，使用45作为参考值归一化到约[0, 1.2]
        features[0] = obs.snr / 45.0

        # [1] 仰角sin编码
        # sin(仰角)在[0°, 90°]范围内是[0, 1]的单调函数
        elev_rad = np.deg2rad(obs.elevation)
        features[1] = np.sin(elev_rad)

        # [2][3] 方位角cos/sin编码
        azim_rad = np.deg2rad(obs.azimuth)
        features[2] = np.cos(azim_rad)
        features[3] = np.sin(azim_rad)

        # [4] 伪距残差（已校正）
        # 残差通常在[-20, 20]米范围内，除以20做初步缩放
        features[4] = obs.pseudorange_residual / 20.0

        # [5] 载波相位残差（若可用）
        # 载波相位精度约1-2mm，残差通常很小
        if obs.has_carrier_phase:
            features[5] = obs.carrier_residual / 0.1  # 缩放到米量级
        else:
            features[5] = 0.0  # 不可用时置零

        # [6] 锁定计数归一化
        # lock_count典型范围0-1000+，使用log归一化避免极端值
        if obs.lock_count > 0:
            features[6] = min(np.log2(obs.lock_count + 1) / 10.0, 1.0)
        else:
            features[6] = 0.0

        # [7] 仰角变化率
        if prev_elevation is not None and prev_timestamp is not None:
            dt = obs.timestamp - prev_timestamp
            if dt > 0.01:  # 至少10ms间隔
                elev_rate = (obs.elevation - prev_elevation) / dt
                # 典型范围[-0.02, 0.02] deg/s，缩放到[-1, 1]
                features[7] = np.clip(elev_rate / 0.02, -1.0, 1.0)
            else:
                features[7] = 0.0
        else:
            features[7] = 0.0

        self.extraction_count += 1
        return features

    def extract_batch(self, observations: List[RawObservation],
                      satellite_histories: Optional[Dict[str, List[Tuple[float, float]]]] = None
                      ) -> Tuple[np.ndarray, List[str]]:
        """
        从一批原始观测中提取特征矩阵

        Args:
            observations: 原始观测列表（一个epoch的所有卫星）
            satellite_histories: 每颗卫星的历史(时间, 仰角)数据，用于计算变化率
                                key: PRN, value: [(timestamp, elevation), ...]

        Returns:
            features: (N, 8) 特征矩阵，N=卫星数量
            prns: 卫星PRN列表（保持与features的行对应）
        """
        N = len(observations)
        features = np.zeros((N, 8), dtype=np.float32)
        prns = []

        for i, obs in enumerate(observations):
            prns.append(obs.prn)

            # 查找该卫星的前一epoch数据（用于仰角变化率）
            prev_elev = None
            prev_time = None
            if satellite_histories and obs.prn in satellite_histories:
                history = satellite_histories[obs.prn]
                if len(history) >= 1:
                    prev_time, prev_elev = history[-1]

            features[i] = self.extract_single(obs, prev_elev, prev_time)

        return features, prns

    def normalize(self, features: np.ndarray) -> np.ndarray:
        """
        对特征矩阵进行z-score归一化

        使用运行时统计（由MemoryBuffer提供），实现自适应归一化。

        Args:
            features: (N, 8) 原始特征矩阵

        Returns:
            归一化后的特征矩阵
        """
        if self.feature_means is not None and self.feature_stds is not None:
            return (features - self.feature_means) / (self.feature_stds + self.epsilon)
        # 如果没有统计信息，使用简单的min-max启发式归一化
        return self._heuristic_normalize(features)

    def _heuristic_normalize(self, features: np.ndarray) -> np.ndarray:
        """
        启发式归一化（后备方案，无统计信息时使用）

        使用预设的典型范围和偏移量进行大致归一化。
        """
        # 各维度的典型中心和尺度
        centers = np.array([0.6, 0.5, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0])
        scales = np.array([0.3, 0.5, 1.0, 1.0, 0.5, 0.5, 0.3, 0.5])

        return (features - centers) / (scales + self.epsilon)

    def extract_from_precomputed(self, sat_data: dict,
                                 prev_elevation: Optional[float] = None,
                                 dt: Optional[float] = None) -> np.ndarray:
        """
        从 gnssfgo 预计算数据直接构造 8 维特征向量

        与 extract_single() 的区别:
        - 不需要 RawObservation 对象，直接使用 dict
        - 所有值来自 gnssfgo 预计算，不重复计算
        - 跳过 SPP、星历解析等 gnssfgo 已完成的工作

        Args:
            sat_data: gnssfgo 导出的单颗卫星数据字典 (JSON 解析后的 dict)
                      包含: snr_l1, elevation, azimuth, psr_residual_l1,
                      tr_dd_cp_residual, lock_count_l1 等字段
            prev_elevation: 该卫星上一 epoch 的仰角（用于计算变化率），可选
            dt: 时间间隔（秒），用于计算变化率，可选

        Returns:
            8 维特征向量 (np.float32)
        """
        features = np.zeros(8, dtype=np.float32)

        # [0] SNR_norm: 使用 gnssfgo 导出的 L1 SNR
        snr = float(sat_data.get('snr_l1', 0))
        features[0] = snr / 45.0

        # [1] elevation_sin: 使用 gnssfgo 计算好的仰角（度）
        elev = float(sat_data.get('elevation', 0))
        features[1] = np.sin(np.deg2rad(elev))

        # [2][3] azimuth_cos/sin: 使用 gnssfgo 计算好的方位角（度）
        azim = float(sat_data.get('azimuth', 0))
        azim_rad = np.deg2rad(azim)
        features[2] = np.cos(azim_rad)
        features[3] = np.sin(azim_rad)

        # [4] pseudorange_residual: SPP 伪距残差 (已由 gnssfgo 计算)
        psr_res = float(sat_data.get('psr_residual_l1', 0))
        features[4] = psr_res / 20.0

        # [5] carrier_residual: 使用 TR DD 载波残差 (gnssfgo 因子残差)
        # 或伪距因子后验残差
        cp_res = float(sat_data.get('tr_dd_cp_residual', 0))
        if cp_res == 0.0:
            cp_res = float(sat_data.get('psr_factor_residual', 0))
        features[5] = cp_res / 0.1

        # [6] lock_count_norm: 使用 gnssfgo 统计的 L1 锁定计数
        lock = float(sat_data.get('lock_count_l1', 0))
        if lock > 0:
            features[6] = min(np.log2(lock + 1) / 10.0, 1.0)
        else:
            features[6] = 0.0

        # [7] elevation_rate: 仰角变化率（需要历史数据）
        if prev_elevation is not None and dt is not None and dt > 0.01:
            elev_rate = (elev - prev_elevation) / dt
            features[7] = np.clip(elev_rate / 0.02, -1.0, 1.0)
        else:
            features[7] = 0.0

        self.extraction_count += 1
        return features

    def extract_batch_from_precomputed(self, satellite_data_list: list,
                                       satellite_histories: Optional[Dict[str, List[tuple]]] = None
                                       ) -> Tuple[np.ndarray, List[str]]:
        """
        从 gnssfgo 预计算数据列表中批量提取特征矩阵

        Args:
            satellite_data_list: dict 列表，每个 dict 是 gnssfgo 导出的一颗卫星数据
            satellite_histories: 卫星历史数据字典，key=PRN, value=[(timestamp, elevation), ...]

        Returns:
            features: (N, 8) 特征矩阵
            prns: 卫星 PRN 列表
        """
        N = len(satellite_data_list)
        features = np.zeros((N, 8), dtype=np.float32)
        prns = []

        for i, sat_data in enumerate(satellite_data_list):
            prn = sat_data.get('prn', f'?_{i}')
            prns.append(prn)

            # 查找历史数据用于仰角变化率
            prev_elev = None
            dt = None
            if satellite_histories and prn in satellite_histories:
                history = satellite_histories[prn]
                if len(history) >= 1:
                    prev_time, prev_elev = history[-1]
                    dt = sat_data.get('timestamp', 0.0) - prev_time if prev_time else None

            features[i] = self.extract_from_precomputed(sat_data, prev_elev, dt)

        return features, prns

    def get_feature_names(self) -> List[str]:
        """获取特征名称列表（用于可视化和调试）"""
        return [
            "SNR_norm",
            "elevation_sin",
            "azimuth_cos",
            "azimuth_sin",
            "pseudorange_residual",
            "carrier_residual",
            "lock_count_norm",
            "elevation_rate",
        ]

    def get_feature_descriptions(self) -> Dict[str, str]:
        """获取特征说明（中文）"""
        return {
            "SNR_norm": "归一化信噪比 — 反映信号强度，低SNR(如<30)信号可能受多路径影响",
            "elevation_sin": "sin(仰角) — 反映信号穿过大气层的厚度，低仰角信号经过更厚的大气",
            "azimuth_cos": "cos(方位角) — 卫星在天空东西方向的位置",
            "azimuth_sin": "sin(方位角) — 卫星在天空南北方向的位置",
            "pseudorange_residual": "归一化伪距残差 — SPP解算后的残差，反映测量误差大小",
            "carrier_residual": "归一化载波残差 — 比伪距残差精度高100倍，能检测微小异常",
            "lock_count_norm": "归一化锁定计数 — 反映跟踪的稳定性，刚锁定的信号不稳定",
            "elevation_rate": "仰角变化率 — 反映卫星相对运动速度，接近天顶时变化慢",
        }
