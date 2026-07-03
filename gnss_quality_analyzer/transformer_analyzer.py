"""
transformer_analyzer.py — Transformer自注意力信号质量分析器
============================================================

这是OSQA的核心创新模块。它使用Transformer的自注意力机制来检测异常信号，
关键特点是：**不需要预训练** — 注意力权重完全由特征相似度决定。

=== 原理解释（面向初学者）===

1. 什么是"自注意力"(Self-Attention)？
   想象一个会议，每个人（卫星）都可以看到房间里的所有人。
   每颗卫星问三个问题：
   - Query(查询): "谁和我类似？"（我想关注什么样的信号）
   - Key(键):   "我是什么？"（我的特征是什么）
   - Value(值): "我的信息是？"（我有什么信息可以分享）

   注意力分数 = Query和Key的相似度（点积）
   最终输出 = 对所有卫星的Value做加权平均，权重=注意力分数

   结果：相似的卫星互相给予高注意力，异常的卫星被"孤立"。

2. 为什么不需要预训练？
   传统的Transformer中，Q/K/V是通过学习的线性层（nn.Linear）生成的。
   但在OSQA中，我们直接使用特征本身作为Q和K：
   - Q = K = 归一化后的特征向量
   - 这意味着注意力完全基于特征的数值相似度
   - 如果两颗卫星的SNR、仰角、残差都相似，它们的注意力自然就高
   - 如果某卫星SNR很低、残差很大，它和其他卫星的相似度就低 → 被孤立 → 被标记为异常

   这本质上是"基于密度的异常检测"，只是用注意力机制优雅地实现了。

3. 多头注意力是什么？
   就像请了多个专家从不同角度看问题：
   - 专家1：主要看SNR和仰角的相似度
   - 专家2：主要看残差和锁定计数的相似度
   - 专家3：主要看几何位置(方位角)的相似度
   - 专家4：看所有特征的全局相似度

   每个专家使用不同的随机投影矩阵（固定，不学习），从而捕获不同的相似模式。
   最终的判断是多个专家意见的综合。

4. 记忆库增强：
   除了卫星之间的比较，每颗卫星还要和一个"好信号记忆库"比较。
   记忆库存放了历史被确认为好的信号特征。
   如果一颗卫星的特征和记忆中所有好信号都不像 → 很可能是异常信号。

=== 算法伪代码 ===

对于每一帧epoch（N颗卫星）：
  1. 输入: features (N, 8) — N颗卫星的8维特征
  2. 归一化: 用运行时统计做z-score归一化
  3. 多头注意力:
     for each head h:
       Q_h = features @ W_q[h]    # 随机但冻结的投影矩阵
       K_h = features @ W_k[h]
       attention_h = softmax(Q_h @ K_h^T / sqrt(d_k))  # (N, N)矩阵

     平均所有头的注意力: attention = mean(attention_h over heads)
  4. 异常分析:
     for each 卫星i:
       attended_by[i] = mean(attention[:, i])  # 其他卫星对i的关注度
       entropy[i] = -sum(attention[i,:] * log(attention[i,:]))  # i的关注分布熵
  5. 记忆库对比:
     for each 卫星i:
       memory_sim[i] = max(cosine_similarity(features[i], prototype) for prototype in memory)
  6. 质量分数:
     quality[i] = sigmoid(attended_by[i] * (1-entropy[i]/max_entropy) * memory_sim[i])

Author: Claude Code
Date: 2026-07-02
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class AttentionResult:
    """
    注意力分析结果

    包含每个卫星的详细注意力指标，便于理解和调试。
    """
    # 每颗卫星的质量分数
    quality_scores: np.ndarray      # (N,) [0, 1]

    # 详细的注意力指标
    attended_by_others: np.ndarray  # (N,) 被其他卫星关注的平均程度
    attention_entropy: np.ndarray   # (N,) 注意力分布熵
    memory_similarity: np.ndarray   # (N,) 与记忆库好原型的最大相似度

    # 注意力矩阵（用于可视化）
    attention_matrix: np.ndarray    # (N, N) 平均注意力权重

    # 标记
    anomaly_flags: List[List[str]]  # 每个卫星的异常标记列表

    # 特征向量（用于后续分析）
    features_normalized: np.ndarray # (N, 8) 归一化后的特征


class MultiHeadProjection:
    """
    多头随机投影

    为每个注意力头生成随机但固定的投影矩阵。
    投影矩阵在初始化时随机生成（Xavier初始化），之后保持不变。

    为什么用随机投影而非学习的投影？
    - 随机投影保留了特征空间中的距离关系（Johnson-Lindenstrauss引理）
    - 不同的随机投影提供了不同的"视角"来比较卫星相似度
    - 避免了预训练的需求
    """

    def __init__(self, input_dim: int, num_heads: int, head_dim: int,
                 seed: int = 42):
        """
        Args:
            input_dim: 输入特征维度（8）
            num_heads: 注意力头数（4）
            head_dim: 每个头的维度（32）
            seed: 随机种子（保证可复现）
        """
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        # 为每个头生成Q和K的投影矩阵
        rng = np.random.RandomState(seed)
        self.W_q = []  # Query投影矩阵列表
        self.W_k = []  # Key投影矩阵列表

        for h in range(num_heads):
            # Xavier初始化：权重 ~ Uniform(-sqrt(6/(in+out)), sqrt(6/(in+out)))
            limit = np.sqrt(6.0 / (input_dim + head_dim))
            Wq = rng.uniform(-limit, limit, (input_dim, head_dim))
            Wk = rng.uniform(-limit, limit, (input_dim, head_dim))
            self.W_q.append(Wq)
            self.W_k.append(Wk)

    def project(self, features: np.ndarray, head_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        对特征进行投影

        Args:
            features: (N, input_dim) 特征矩阵
            head_idx: 头索引

        Returns:
            Q: (N, head_dim)
            K: (N, head_dim)
        """
        Q = features @ self.W_q[head_idx]
        K = features @ self.W_k[head_idx]
        return Q, K


class TransformerAnalyzer:
    """
    Transformer自注意力信号质量分析器

    这是OSQA的三个核心分析器之一，使用注意力机制检测异常信号。

    核心指标说明：
    - attended_by_others: 其他卫星对这颗卫星的平均关注度
      高 → 这颗卫星的特征和大多数卫星相似 → 可信
      低 → 这颗卫星与众不同 → 可疑

    - attention_entropy: 这颗卫星对其他卫星的关注分布熵
      低 → 它只关注少数几颗卫星 → 偏好明确 → 正常
      高 → 它对所有卫星都平均关注 → 没有明确偏好 → 可能自己特征异常

    - memory_similarity: 与记忆中"好信号"的相似度
      高 → 和历史好信号很像 → 可信
      低 → 和历史好信号都不像 → 可疑
    """

    def __init__(self, input_dim: int = 8, num_heads: int = 4,
                 head_dim: int = 32, temperature: float = 1.0,
                 dropout: float = 0.1, seed: int = 42):
        """
        Args:
            input_dim: 输入特征维度
            num_heads: 注意力头数
            head_dim: 每个头的维度
            temperature: 注意力温度（越高分布越平滑）
            dropout: 注意力dropout率
            seed: 随机种子
        """
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.temperature = temperature
        self.dropout = dropout

        # 初始化多头投影
        self.projection = MultiHeadProjection(
            input_dim, num_heads, head_dim, seed
        )

        # 统计
        self.analysis_count: int = 0
        self.anomaly_count: int = 0

    def analyze(self, features: np.ndarray,
                prns: List[str],
                memory_prototypes: Optional[List[np.ndarray]] = None,
                mask: Optional[np.ndarray] = None
                ) -> AttentionResult:
        """
        分析一帧epoch的卫星信号质量

        这是主要的对外接口。输入一帧epoch的所有卫星特征，
        输出每颗卫星的质量分数和详细分析结果。

        Args:
            features: (N, 8) 特征矩阵，N颗卫星
            prns: 卫星PRN列表，长度N
            memory_prototypes: 记忆库中的"好原型"特征列表
            mask: (N,) bool数组，True=有效（可选）

        Returns:
            AttentionResult: 包含所有分析结果
        """
        N = features.shape[0]
        if N < 2:
            # 只有1颗卫星时，无法做注意力分析，给默认高质量分
            return self._single_satellite_result(features, prns)

        # 如果无mask则默认所有卫星都有效
        if mask is None:
            mask = np.ones(N, dtype=bool)

        # ========== 步骤1: 计算多头注意力 ==========
        all_attention = np.zeros((N, N))

        for h in range(self.num_heads):
            Q, K = self.projection.project(features, h)

            # 计算注意力分数: scores = Q @ K^T / sqrt(d_k)
            scale = np.sqrt(self.head_dim)
            scores = (Q @ K.T) / (scale * self.temperature)

            # 对无效卫星的分数设为负无穷（softmax后为零）
            scores_masked = scores.copy()
            for i in range(N):
                if not mask[i]:
                    scores_masked[:, i] = -1e9  # 不关注无效卫星
                    scores_masked[i, :] = -1e9  # 无效卫星也不关注别人

            # Softmax归一化
            attention_h = self._stable_softmax(scores_masked, axis=1)
            all_attention += attention_h

        # 平均所有头的注意力
        avg_attention = all_attention / self.num_heads

        # ========== 步骤2: 计算注意力指标 ==========
        attended_by_others = np.zeros(N)
        attention_entropy = np.zeros(N)

        for i in range(N):
            if not mask[i]:
                continue

            # 被关注度: 其他所有卫星对i的平均注意力
            # 排除自注意力（卫星对自己的关注）
            others_attention = np.delete(avg_attention[:, i], i)
            attended_by_others[i] = np.mean(others_attention)

            # 关注分布熵: 卫星i对其他卫星的注意力分布
            # 排除自注意力
            own_attention = np.delete(avg_attention[i, :], i)
            attention_entropy[i] = self._compute_entropy(own_attention)

        # 归一化熵到[0, 1]（最大熵 = log(N-1)）
        if N > 2:
            max_entropy = np.log(N - 1)
            attention_entropy = attention_entropy / (max_entropy + 1e-8)

        # ========== 步骤3: 记忆库对比 ==========
        memory_similarity = np.ones(N)  # 默认全1（无记忆时不影响分数）

        if memory_prototypes and len(memory_prototypes) > 0:
            proto_matrix = np.array(memory_prototypes)  # (M, 8)
            for i in range(N):
                if not mask[i]:
                    memory_similarity[i] = 0.0
                    continue
                # 计算与所有原型的余弦相似度，取最大值
                sims = self._cosine_similarity(features[i:i+1], proto_matrix)
                memory_similarity[i] = np.max(sims) if len(sims) > 0 else 1.0

        # ========== 步骤4: 计算质量分数 ==========
        quality_scores = np.zeros(N)
        anomaly_flags = [[] for _ in range(N)]

        for i in range(N):
            if not mask[i]:
                quality_scores[i] = 0.0
                anomaly_flags[i].append("invalid")
                continue

            # 综合三个指标计算质量分数
            # 被关注度高 → 好
            # 熵低（偏好明确）→ 好
            # 记忆相似度高 → 好
            #
            # 被关注度缩放：attended_by_others ≈ 1/N（均匀注意力时）
            # 所以将其乘以N/2来校准，使"均匀被关注"的卫星得分约0.5
            # 被关注度高于均匀水平 → 得分>0.5, 低于均匀水平 → 得分<0.5
            score_attention = np.clip(attended_by_others[i] * N / 2.0, 0.0, 1.0)
            score_entropy = 1.0 - attention_entropy[i]
            score_memory = np.clip(memory_similarity[i], 0.0, 1.0)

            # 几何平均融合（类似trbin的pow(p_rel, 1/6)）
            # 优点：不会因为一个指标稍低就过度惩罚
            quality_scores[i] = np.power(
                score_attention * score_entropy * score_memory, 1.0 / 3.0
            )

            # 异常标记
            if score_attention < 0.3:
                anomaly_flags[i].append("low_attention")
            if attention_entropy[i] > 0.7:
                anomaly_flags[i].append("high_entropy")
            if score_memory < 0.3:
                anomaly_flags[i].append("low_memory_similarity")

        # ========== 统计 ==========
        self.analysis_count += 1
        self.anomaly_count += sum(1 for flags in anomaly_flags if len(flags) > 0)

        return AttentionResult(
            quality_scores=quality_scores,
            attended_by_others=attended_by_others,
            attention_entropy=attention_entropy,
            memory_similarity=memory_similarity,
            attention_matrix=avg_attention,
            anomaly_flags=anomaly_flags,
            features_normalized=features,
        )

    def _single_satellite_result(self, features: np.ndarray,
                                  prns: List[str]) -> AttentionResult:
        """处理只有1颗卫星的特殊情况"""
        N = 1
        return AttentionResult(
            quality_scores=np.array([0.5]),  # 单卫星给中间分
            attended_by_others=np.array([0.5]),
            attention_entropy=np.array([0.0]),
            memory_similarity=np.array([0.5]),
            attention_matrix=np.array([[1.0]]),
            anomaly_flags=[["single_satellite"]],
            features_normalized=features,
        )

    @staticmethod
    def _stable_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """
        数值稳定的softmax实现

        减去最大值防止指数溢出。
        """
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / (np.sum(exp_x, axis=axis, keepdims=True) + 1e-8)

    @staticmethod
    def _compute_entropy(probs: np.ndarray) -> float:
        """
        计算概率分布的熵

        H = -sum(p_i * log(p_i))

        熵高 → 分布均匀（无偏好）
        熵低 → 分布集中（有明确偏好）
        """
        # 过滤掉极小值（避免log(0)）
        probs = probs[probs > 1e-10]
        if len(probs) == 0:
            return 0.0
        return float(-np.sum(probs * np.log(probs)))

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        计算余弦相似度

        sim(a, b) = a·b / (|a| * |b|)

        Args:
            a: (1, D) 或 (D,) 查询向量
            b: (M, D) 参考向量组

        Returns:
            (M,) 相似度数组，范围[-1, 1]
        """
        a = a.reshape(1, -1)
        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)
        return ((a @ b.T) / (a_norm @ b_norm.T + 1e-8)).flatten()

    def get_attention_visualization(self, result: AttentionResult,
                                     prns: List[str]) -> Dict:
        """
        生成注意力可视化的数据结构

        Args:
            result: 分析结果
            prns: 卫星PRN列表

        Returns:
            包含可视化数据的字典（可用于生成热力图等）
        """
        N = len(prns)
        return {
            "prns": prns,
            "attention_matrix": result.attention_matrix.tolist(),
            "quality_scores": result.quality_scores.tolist(),
            "attended_by_others": result.attended_by_others.tolist(),
            "attention_entropy": result.attention_entropy.tolist(),
            "memory_similarity": result.memory_similarity.tolist(),
            "anomaly_flags": result.anomaly_flags,
            "n_satellites": N,
            "n_anomalies": sum(1 for f in result.anomaly_flags if len(f) > 0),
        }

    def get_statistics(self) -> dict:
        """获取分析器统计信息"""
        return {
            "analysis_count": self.analysis_count,
            "anomaly_count": self.anomaly_count,
            "anomaly_rate": (self.anomaly_count / max(self.analysis_count, 1)),
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "temperature": self.temperature,
        }
