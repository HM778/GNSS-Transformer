"""
graph_analyzer.py — 图结构几何分析器
====================================

利用卫星在天空中的空间分布构建图（Graph），通过消息传递检测
几何不一致的异常信号。

=== 原理解释（面向初学者）===

1. 什么是"图"(Graph)？
   图由"节点"(Node)和"边"(Edge)组成。
   - 节点 = 每颗卫星
   - 边 = 卫星之间的几何关系（比如"在天空中相邻"）

   类比：卫星就像分布在天空中的城市，在天空中相近的卫星之间
   有"高速公路"（边）相连。

2. 图结构的直觉：
   假设天空被分成几个区域（东南、西南、天顶等）。
   同一个区域内的卫星经历相似的大气延迟和多路径环境。
   因此：
   - 如果某卫星的测量质量很差，但同区域其他卫星都很好 → 可能是多路径
   - 如果整个区域的卫星质量都差 → 可能是该方向确实有建筑物遮挡

   图结构能区分这两种情况。

3. 图卷积(GCN)消息传递的直觉：
   每颗卫星"询问"它的邻居：
   "邻居们，你们的信号质量怎么样？"

   邻居通过加权平均回答（距离越近权重越大）。

   然后这颗卫星对比自己的特征和邻居的聚合特征：
   - 如果一致 → 质量分高
   - 如果不一致 → 质量分低

4. 为什么有预测功能？
   图结构记录了卫星之间的几何关系。
   当接收机移动时，可以根据卫星的运动规律（仰角/方位角的变化）
   预测下一时刻每颗卫星的预期特征。
   预测值和实际值差异大的卫星 → 可能发生了异常（如周跳）。

=== 算法步骤 ===

1. 图构建: 基于仰角差和方位角差构建邻接矩阵
2. GCN消息传递: 2层图卷积聚合邻居特征
3. 一致性检查: 原始特征 vs 邻居聚合特征的差异
4. 轨迹预测: 一阶外推 + 图约束平滑

Author: Claude Code
Date: 2026-07-02
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class GraphResult:
    """
    图分析结果
    """
    # 每颗卫星的质量分数
    quality_scores: np.ndarray      # (N,) [0, 1]

    # 详细指标
    consistency_error: np.ndarray   # (N,) 与邻居的一致性误差
    neighbor_consensus: np.ndarray  # (N,) 邻居的平均质量（加权）
    predicted_features: np.ndarray  # (N, 8) 预测的下一时刻特征

    # 图结构
    adjacency_matrix: np.ndarray    # (N, N) 邻接矩阵
    edge_weights: np.ndarray        # (N, N) 边权重矩阵

    # 标记
    anomaly_flags: List[List[str]]


class GNSSGraph:
    """
    GNSS卫星空间图

    根据卫星的仰角和方位角构建图结构。
    两颗卫星如果在天空中位置相近，就用边连接起来。
    """

    def __init__(self, elevation_threshold: float = 30.0,
                 azimuth_threshold: float = 60.0):
        """
        Args:
            elevation_threshold: 仰角差阈值（度），小于此值认为在"同一高度层"
            azimuth_threshold: 方位角差阈值（度），小于此值认为在"同一方向"
        """
        self.elevation_threshold = elevation_threshold
        self.azimuth_threshold = azimuth_threshold

    def build_graph(self, elevations: np.ndarray, azimuths: np.ndarray
                    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        根据卫星的仰角和方位角构建图

        连接条件：两颗卫星的仰角差 < threshold_elev AND 方位角差 < threshold_azm

        边权重基于角距离（angular distance），越近权重越大。

        角距离公式（球面余弦定律的简化）：
          cos(d) = sin(e1)*sin(e2) + cos(e1)*cos(e2)*cos(a1-a2)
          angular_distance = arccos(cos(d))

        Args:
            elevations: (N,) 仰角数组（度）
            azimuths: (N,) 方位角数组（度）

        Returns:
            adjacency: (N, N) 邻接矩阵（0/1，无自环）
            weights: (N, N) 边权重矩阵（余弦角距离）
        """
        N = len(elevations)

        # 转换为弧度
        elev_rad = np.deg2rad(elevations)
        azim_rad = np.deg2rad(azimuths)

        adjacency = np.zeros((N, N), dtype=np.float32)
        weights = np.zeros((N, N), dtype=np.float32)

        for i in range(N):
            for j in range(i + 1, N):
                # 计算仰角差和方位角差
                elev_diff = abs(elevations[i] - elevations[j])
                # 方位角差需要考虑环绕（350°和10°只差20°）
                azim_diff = abs(azimuths[i] - azimuths[j])
                if azim_diff > 180:
                    azim_diff = 360 - azim_diff

                # 检查是否满足连接条件
                if elev_diff < self.elevation_threshold and azim_diff < self.azimuth_threshold:
                    adjacency[i, j] = 1.0
                    adjacency[j, i] = 1.0

                    # 计算角距离作为权重
                    cos_d = (np.sin(elev_rad[i]) * np.sin(elev_rad[j]) +
                             np.cos(elev_rad[i]) * np.cos(elev_rad[j]) *
                             np.cos(azim_rad[i] - azim_rad[j]))
                    cos_d = np.clip(cos_d, -1.0, 1.0)
                    angular_dist = np.arccos(cos_d)  # [0, pi]

                    # 角距离越小，权重越大
                    # 使用余弦相似度作为权重
                    weight = float(cos_d)
                    weights[i, j] = weight
                    weights[j, i] = weight

        return adjacency, weights

    def get_neighbors(self, i: int, adjacency: np.ndarray) -> np.ndarray:
        """
        获取节点i的所有邻居索引

        Args:
            i: 节点索引
            adjacency: 邻接矩阵

        Returns:
            邻居索引数组
        """
        return np.where(adjacency[i] > 0)[0]


class PositionalEncoder:
    """
    卫星位置编码器

    将仰角和方位角编码为高维向量，帮助图模型理解卫星的绝对位置。

    使用sin/cos频率编码（类似Transformer的位置编码）：
      pos_encoding[k] = sin(angle * 2*pi * freq_k) 或 cos(angle * 2*pi * freq_k)

    为什么需要位置编码？
    - 图结构只编码了相对关系（谁和谁相邻），没有绝对位置信息
    - 位置编码告诉模型"这颗卫星在天顶附近"或"这颗卫星在低仰角区域"
    - 不同位置的卫星有不同的信号特征（低仰角SNR通常更低等）
    """

    def __init__(self, encoding_dim: int = 16):
        """
        Args:
            encoding_dim: 编码维度（必须是偶数）
        """
        self.encoding_dim = encoding_dim

        # 不同频率的正弦/余弦基
        self.frequencies = np.array([
            2.0 ** i for i in range(encoding_dim // 2)
        ], dtype=np.float32)

    def encode(self, elevations: np.ndarray, azimuths: np.ndarray) -> np.ndarray:
        """
        对仰角和方位角进行位置编码

        Args:
            elevations: (N,) 仰角数组（度），归一化到[0, 1]
            azimuths: (N,) 方位角数组（度），归一化到[0, 1]

        Returns:
            (N, encoding_dim) 位置编码矩阵
        """
        N = len(elevations)

        # 归一化到[0, 1]
        elev_norm = elevations / 90.0  # 仰角: [0, 90] → [0, 1]
        azim_norm = azimuths / 360.0   # 方位角: [0, 360] → [0, 1]

        encoding = np.zeros((N, self.encoding_dim), dtype=np.float32)

        for i in range(N):
            half_dim = self.encoding_dim // 2
            # 前一半：仰角编码
            for k in range(half_dim // 2):
                freq = self.frequencies[k]
                encoding[i, 2 * k] = np.sin(elev_norm[i] * 2.0 * np.pi * freq)
                encoding[i, 2 * k + 1] = np.cos(elev_norm[i] * 2.0 * np.pi * freq)
            # 后一半：方位角编码
            for k in range(half_dim // 2):
                freq = self.frequencies[k]
                offset = half_dim
                encoding[i, offset + 2 * k] = np.sin(azim_norm[i] * 2.0 * np.pi * freq)
                encoding[i, offset + 2 * k + 1] = np.cos(azim_norm[i] * 2.0 * np.pi * freq)

        return encoding


class GraphAnalyzer:
    """
    图结构几何分析器

    使用图卷积(GCN)消息传递来检查卫星信号质量的空间一致性。
    """

    def __init__(self, elevation_threshold: float = 30.0,
                 azimuth_threshold: float = 60.0,
                 consistency_temperature: float = 0.5,
                 num_layers: int = 2,
                 position_encoding_dim: int = 16):
        """
        Args:
            elevation_threshold: 图边构建的仰角阈值（度）
            azimuth_threshold: 图边构建的方位角阈值（度）
            consistency_temperature: 一致性检查的温度参数
            num_layers: GCN层数
            position_encoding_dim: 位置编码维度
        """
        self.elevation_threshold = elevation_threshold
        self.azimuth_threshold = azimuth_threshold
        self.consistency_temperature = consistency_temperature
        self.num_layers = num_layers

        # 子模块
        self.graph_builder = GNSSGraph(elevation_threshold, azimuth_threshold)
        self.position_encoder = PositionalEncoder(position_encoding_dim)

        # 用于预测的历史数据
        self.prev_features: Optional[np.ndarray] = None  # (N_prev, 8)
        self.prev_elevations: Optional[np.ndarray] = None
        self.prev_azimuths: Optional[np.ndarray] = None
        self.prev_prns: List[str] = []

    def analyze(self, features: np.ndarray,
                elevations: np.ndarray,
                azimuths: np.ndarray,
                prns: List[str],
                pseudorange_residuals: Optional[np.ndarray] = None
                ) -> GraphResult:
        """
        对一帧epoch进行图结构几何分析

        Args:
            features: (N, 8) 特征矩阵
            elevations: (N,) 仰角（度）
            azimuths: (N,) 方位角（度）
            prns: 卫星PRN列表
            pseudorange_residuals: (N,) 伪距残差（可选，用于更好的异常判断）

        Returns:
            GraphResult: 图分析结果
        """
        N = len(elevations)

        if N < 3:
            # 卫星太少，无法构建有意义的图
            return GraphResult(
                quality_scores=np.ones(N) * 0.5,
                consistency_error=np.zeros(N),
                neighbor_consensus=np.ones(N) * 0.5,
                predicted_features=features.copy(),
                adjacency_matrix=np.eye(N),
                edge_weights=np.eye(N),
                anomaly_flags=[["insufficient_satellites"] for _ in range(N)],
            )

        # ========== 步骤1: 构建几何图 ==========
        adjacency, edge_weights = self.graph_builder.build_graph(elevations, azimuths)

        # ========== 步骤2: 位置编码 ==========
        pos_encoding = self.position_encoder.encode(elevations, azimuths)

        # ========== 步骤3: GCN消息传递 ==========
        # 将特征和位置编码拼接
        combined_features = np.concatenate([features, pos_encoding], axis=1)

        # 多层图卷积
        hidden = combined_features.copy()
        for _ in range(self.num_layers):
            hidden = self._gcn_layer(hidden, adjacency, edge_weights)

        # ========== 步骤4: 一致性检查 ==========
        # 对比原始特征和GCN聚合后的特征
        original = combined_features[:, :features.shape[1]]  # 取特征部分
        aggregated = hidden[:, :features.shape[1]]

        consistency_error = np.zeros(N)
        neighbor_consensus = np.zeros(N)
        quality_scores = np.zeros(N)
        anomaly_flags = [[] for _ in range(N)]

        for i in range(N):
            # 一致性误差：原始特征和邻居聚合特征的L2距离
            diff = original[i] - aggregated[i]
            consistency_error[i] = np.sqrt(np.mean(diff ** 2))

            # 邻居共识：该卫星邻居的平均特征与整体平均的差异
            neighbors = self.graph_builder.get_neighbors(i, adjacency)
            if len(neighbors) > 0:
                neighbor_mean = np.mean(original[neighbors], axis=0)
                global_mean = np.mean(original, axis=0)
                neighbor_consensus[i] = 1.0 - np.sqrt(np.mean((neighbor_mean - global_mean) ** 2))
            else:
                neighbor_consensus[i] = 0.5  # 无邻居时给中间值

            # 质量分数：一致性误差的指数衰减
            quality_scores[i] = np.exp(-consistency_error[i] / self.consistency_temperature)

            # 异常标记
            if consistency_error[i] > 2.0 * self.consistency_temperature:
                anomaly_flags[i].append("graph_inconsistent")
            if len(neighbors) == 0:
                anomaly_flags[i].append("no_graph_neighbors")

        # ========== 步骤5: 轨迹预测 ==========
        predicted_features = self._predict_next_features(features, elevations, azimuths,
                                                          adjacency, edge_weights, prns)

        # 更新历史
        self.prev_features = features.copy()
        self.prev_elevations = elevations.copy()
        self.prev_azimuths = azimuths.copy()
        self.prev_prns = prns.copy()

        return GraphResult(
            quality_scores=quality_scores,
            consistency_error=consistency_error,
            neighbor_consensus=neighbor_consensus,
            predicted_features=predicted_features,
            adjacency_matrix=adjacency,
            edge_weights=edge_weights,
            anomaly_flags=anomaly_flags,
        )

    def _gcn_layer(self, features: np.ndarray, adjacency: np.ndarray,
                   edge_weights: np.ndarray) -> np.ndarray:
        """
        图卷积层（简化实现，无需训练）

        GCN核心公式:
          H_new = sigma(D^(-1/2) * A * D^(-1/2) * H * W)

        简化版（使用边权重的加权平均）:
          H_new[i] = sum(w_ij * H[j]) / sum(w_ij)  for j in neighbors(i)

        Args:
            features: (N, D) 节点特征
            adjacency: (N, N) 邻接矩阵
            edge_weights: (N, N) 边权重矩阵

        Returns:
            (N, D) 更新后的节点特征
        """
        N, D = features.shape
        output = np.zeros_like(features)

        for i in range(N):
            neighbors = np.where(adjacency[i] > 0)[0]

            if len(neighbors) == 0:
                # 无邻居：保留自身特征
                output[i] = features[i]
                continue

            # 加权平均（包括自身）
            weights = edge_weights[i, neighbors].copy()
            # 自环权重（自身特征的重要性）
            self_weight = 0.5
            total_weight = self_weight + np.sum(weights)

            output[i] = (self_weight * features[i] +
                         np.sum(weights[:, np.newaxis] * features[neighbors], axis=0)) / total_weight

        return output

    def _predict_next_features(self, features: np.ndarray,
                                elevations: np.ndarray,
                                azimuths: np.ndarray,
                                adjacency: np.ndarray,
                                edge_weights: np.ndarray,
                                prns: List[str]) -> np.ndarray:
        """
        预测下一时刻每颗卫星的特征

        使用一阶外推 + 图约束平滑：
          1. raw_prediction[i] = features[t] + (features[t] - features[t-1])
             （假设趋势继续）
          2. smoothed[i] = 0.7 * raw_prediction[i] + 0.3 * mean(neighbor_predictions)
             （图约束平滑 — 邻居的预测应该相似）

        预测值可以用于下一epoch与实际值的比较，差异大的可能是异常。

        Args:
            features: (N, 8) 当前特征
            elevations: (N,) 仰角
            azimuths: (N,) 方位角
            adjacency: (N, N) 邻接矩阵
            edge_weights: (N, N) 边权重
            prns: 卫星PRN列表

        Returns:
            (N, 8) 预测的下一时刻特征
        """
        N = features.shape[0]

        # 一阶外推
        if self.prev_features is not None and len(self.prev_prns) > 0:
            raw_prediction = np.zeros_like(features)

            for i, prn in enumerate(prns):
                if prn in self.prev_prns:
                    prev_idx = self.prev_prns.index(prn)
                    # features[t+1] ≈ features[t] + (features[t] - features[t-1])
                    delta = features[i] - self.prev_features[prev_idx]
                    raw_prediction[i] = features[i] + delta
                else:
                    # 新卫星：无法外推，假设不变
                    raw_prediction[i] = features[i]
        else:
            raw_prediction = features.copy()

        # 图约束平滑
        smoothed = np.zeros_like(raw_prediction)
        for i in range(N):
            neighbors = np.where(adjacency[i] > 0)[0]

            if len(neighbors) == 0:
                smoothed[i] = raw_prediction[i]
            else:
                neighbor_pred = np.mean(raw_prediction[neighbors], axis=0)
                # 70%自己的预测 + 30%邻居的平均预测
                smoothed[i] = 0.7 * raw_prediction[i] + 0.3 * neighbor_pred

        return smoothed

    def get_graph_statistics(self, result: GraphResult) -> dict:
        """
        获取图结构的统计信息

        Args:
            result: 图分析结果

        Returns:
            统计信息字典
        """
        adj = result.adjacency_matrix
        N = adj.shape[0]
        n_edges = int(np.sum(adj) / 2)  # 无向图，除以2

        # 计算每个节点的度（连接数）
        degrees = np.sum(adj, axis=1)

        return {
            "n_nodes": N,
            "n_edges": n_edges,
            "avg_degree": float(np.mean(degrees)),
            "max_degree": int(np.max(degrees)),
            "isolated_nodes": int(np.sum(degrees == 0)),
            "avg_consistency_error": float(np.mean(result.consistency_error)),
        }
