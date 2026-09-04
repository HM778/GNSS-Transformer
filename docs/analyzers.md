# OSQA 三个核心分析器详解

本文档解释 `GNSS-Transformer/gnss_quality_analyzer/` 中三个分析器的处理逻辑。

---

## 0. 总览

OSQA 把每颗卫星看成一个 8 维特征向量，三个分析器从不同角度给这颗卫星打分，最后通过 `QualityFusion` 融合成 `quality_final`。

| 分析器 | 文件 | 视角 | 核心思想 |
|---|---|---|---|
| TransformerAnalyzer | `transformer_analyzer.py` | 同伴相似度 | 一颗卫星如果和其他卫星"长得不一样"，就可能是异常 |
| GraphAnalyzer | `graph_analyzer.py` | 空间几何一致性 | 天空中相邻的卫星应该经历相似环境，离群者异常 |
| TemporalAnalyzer | `temporal_analyzer.py` | 自身历史一致性 | 同一颗卫星的特征随时间应该平滑变化，突变者异常 |

三个分数在 `quality_fusion.py` 中按加权（默认）或几何平均融合。此外，从 gnssfgo 反馈回来的 `carrier_residual`（`dop_cp` 残差）会作为**硬惩罚**直接修正最终分数。

---

## 1. TransformerAnalyzer：自注意力相似度

### 1.1 输入输出

```
输入：
  features          (N, 8)   当前 epoch 所有卫星的 8 维特征
  prns              (N,)     PRN 列表
  memory_prototypes (M, 8)   记忆库中"好信号"的原型特征（可选）
  mask              (N,)     是否有效

输出：
  quality_scores     (N,)    [0, 1] 分数
  attended_by_others (N,)    其他卫星对 i 的平均关注度
  attention_entropy  (N,)    i 对其他卫星关注分布的熵
  memory_similarity  (N,)    i 与好原型最大余弦相似度
  attention_matrix   (N, N)  注意力权重矩阵
  anomaly_flags      (N, []) 异常标记
```

### 1.2 核心流程

**步骤 1：多头随机投影**

代码里并没有用 PyTorch 训练 Transformer，而是用手写的 numpy 实现。

- 初始化 4 个注意力头，每个头有随机的 `W_q` 和 `W_k` 投影矩阵。
- 这些矩阵是 **固定冻结** 的，不会在线学习。
- 对第 `h` 个头：
  ```
  Q_h = features @ W_q[h]
  K_h = features @ W_k[h]
  ```

> 为什么随机投影可行？随机投影大致保留特征之间的距离关系（Johnson-Lindenstrauss 思想），不同头提供不同视角。

**步骤 2：计算注意力**

```
scores_h = (Q_h @ K_h^T（转置）) / sqrt(head_dim) / temperature
attention_h = softmax(scores_h, axis=1)
```

对所有头求平均，得到 `attention_matrix`。

**步骤 3：被关注度（attended_by_others）**

对卫星 i：
```
attended_by_others[i] = mean(attention_matrix[:, i])  # 去掉自己对角线
```

- 如果 i 和大家都很像，别人都会关注它 → 值高 → 可信。
- 如果 i 和别人都不一样，没人关注它 → 值低 → 可疑。

**步骤 4：注意力熵（attention_entropy）**

卫星 i 对其他卫星的关注分布越均匀，熵越高。注意：健康卫星彼此相似，关注自然均匀，熵本来就高。所以 **熵不再参与打分**，只把"显著比同伴更均匀"作为 `high_entropy` 标记保留。

**步骤 5：记忆库对比**

如果提供了 `memory_prototypes`：
```
memory_similarity[i] = max(cosine_similarity(features[i], prototype))
```

- 和历史好信号越像，分越高。
- 记忆库由 `MemoryBuffer` 维护。

**步骤 6：质量分数**

```
ratio[i] = attended_by_others[i] * N   # 均匀时 ≈ 1
score_attention = clip(ratio, 0, 1)
score_memory    = clip(memory_similarity, 0, 1)
quality[i] = sqrt(score_attention * score_memory)
```

### 1.3 异常标记

- `low_attention`：被关注度 < 0.3
- `high_entropy`：熵显著高于同伴中位数（> 0.2）
- `low_memory_similarity`：记忆相似度 < 0.3

### 1.4 关键说明

这个 Transformer **不学习任何参数**。投影矩阵随机且冻结，它本质上是一个"基于注意力机制的密度估计器"：找出被同伴孤立的卫星。

---

## 2. GraphAnalyzer：图结构几何一致性

### 2.1 输入输出

```
输入：
  features   (N, 8)   8 维特征
  elevations (N,)     仰角（度）
  azimuths   (N,)     方位角（度）
  prns       (N,)     PRN 列表

输出：
  quality_scores      (N,)    [0, 1] 分数
  consistency_error   (N,)    与邻居聚合后的差异
  neighbor_consensus  (N,)    邻居平均特征和全局平均的差异
  predicted_features  (N, 8)   预测的下一时刻特征
  adjacency_matrix    (N, N)   邻接矩阵
  edge_weights        (N, N)   边权重
  anomaly_flags       (N, [])  异常标记
```

### 2.2 核心流程

**步骤 1：建图**

两颗卫星 i 和 j 之间有边，当且仅当：
```
|elevation_i - elevation_j| < 30°
且
|azimuth_i - azimuth_j| < 60°   （方位角考虑 0°/360° 环绕）
```

边权重用球面角距离余弦：
```
cos(d) = sin(e1)sin(e2) + cos(e1)cos(e2)cos(a1 - a2)
weight = cos(d)
```

越近权重越大。

**步骤 2：位置编码**

把仰角、方位角用 sin/cos 多频编码成 16 维向量，拼到 8 维特征后面，得到 `combined_features (N, 24)`。

**步骤 3：GCN 消息传递**

做 2 层简化图卷积：
```
output[i] = (0.5 * features[i] + sum(w_ij * features[j])) / total_weight
```

每颗卫星的特征被替换成自己和邻居的加权平均。

**步骤 4：一致性检查**

对比原始特征 `original` 和 GCN 聚合后的特征 `aggregated`：
```
consistency_error[i] = sqrt(mean((original[i] - aggregated[i])^2))
```

- 如果 i 和邻居一致，聚合后变化不大，误差小。
- 如果 i 和邻居不一致（比如多路径只影响这一颗），聚合后变化大，误差大。

**步骤 5：质量分数（相对校准）**

旧实现用固定温度 `exp(-err / T)`，但 z-score 特征空间里健康卫星也有"正常地板误差"，导致系统性低分。

新实现改为以本历元非零误差的中位数 `err_ref` 为基准：
```
excess = max(0, (consistency_error[i] - err_ref) / err_ref)
quality[i] = exp(-0.5 * excess)
```

- 误差不超过同伴 → 满分 1.0
- 误差 3 倍于同伴 → 约 0.37
- 误差 5 倍于同伴 → 约 0.14

**步骤 6：轨迹预测**

用上一历元特征做一阶外推：
```
prediction[t+1] = features[t] + (features[t] - features[t-1])
```

再用图约束平滑（70% 自己 + 30% 邻居平均），得到 `predicted_features`。下一历元会和实际值对比，用于时序分析。

### 2.3 异常标记

- `graph_inconsistent`：一致性误差 > 3 倍同伴中位数
- `no_graph_neighbors`：没有邻居（孤立卫星）

### 2.4 关键说明

GCN 也没有可学习参数。它回答的问题是："天空中相邻的卫星应该相似，谁和邻居不一致？"

---

## 3. TemporalAnalyzer：时序一致性

### 3.1 输入输出

```
输入：
  features (N, 8)   8 维特征
  prns     (N,)     PRN 列表
  mask     (N,)     是否有效（可选）

输出：
  quality_scores         (N,)    [0, 1] 分数
  mahalanobis_distance   (N,)    马氏距离
  feature_jump_magnitude (N,)    特征跳变幅度
  consecutive_anomalies  (N,)    连续异常计数
  ema_features           (N, 8)   当前 EMA
  anomaly_flags          (N, [])  异常标记
```

### 3.2 核心流程

对每颗卫星维护一个独立的 `SatelliteTracker`：

```python
self.trackers[prn] = SatelliteTracker(prn, n_features=8)
```

**SatelliteTracker 内部状态：**
- `ema`：8 维特征的指数移动平均
- `var`：8 维特征的指数移动方差
- `consecutive_anomalies`：连续异常计数
- `total_updates`：更新次数

**步骤 1：更新 EMA 和方差**

第一次见某颗卫星：直接设 `ema = features`，`var = 0.01`。

之后：
```
diff = features - ema
mahalanobis = sum(diff^2 / (var + eps)) / 8

ema = 0.8 * ema + 0.2 * features
var = 0.9 * var + 0.1 * (features - ema)^2
```

`mahalanobis` 就是当前特征偏离历史均值多少个"综合标准差"。

**步骤 2：预热期保护**

前 3 次更新是预热期，只积累统计不评分，避免初始方差太小时把正常波动误判为异常。

**步骤 3：判断异常**

```
is_anomaly = mahalanobis > anomaly_threshold
```

默认 `anomaly_threshold = 3.0`，约 3 个标准差。

**步骤 4：连续异常惩罚**

```
if is_anomaly: consecutive_anomalies += 1
else:          consecutive_anomalies = max(0, consecutive_anomalies - 0.5)
```

连续异常次数越多，惩罚越大。

**步骤 5：质量分数**

```
if mahalanobis <= 1.5:        score = 1.0
elif mahalanobis <= 3.0:      score 线性降到 0.5
elif mahalanobis <= 6.0:      score 线性降到 0.0
else:                         score = 0.0

if consecutive_anomalies > 3:
    score *= (1 - penalty)
```

### 3.3 异常标记

- `temporal_jump`：当前历元马氏距离超过阈值
- `persistent_anomaly(k)`：连续 k 次异常
- `few_samples`：总更新次数 <= 3（预热期）

### 3.4 关键说明

TemporalAnalyzer 是在线学习最重的部分。它确实随着运行时间积累经验：
- EMA 会越来越稳定
- 方差估计会越来越准
- 对每颗卫星的"正常行为"会越来越了解

但前提是输入特征 `features` 本身要能反映异常。如果 `carrier_residual` 长期为 0 或特征被污染，时序分析也会失效。

---

## 4. 三者的融合

`QualityFusion.fuse()` 默认使用 `weighted` 模式：

```
q_final = 0.45 * q_transformer + 0.40 * q_graph + 0.15 * q_temporal
```

之后再根据 gnssfgo 反馈的 `carrier_residual`（`dop_cp` 残差，单位：周）做硬惩罚：

```python
penalty = clip((abs(carrier_residual) - 5.0) / 95.0, 0.0, 0.9)
q_final = q_final * (1.0 - penalty)
```

| dop_cp 残差 | 惩罚 | 效果 |
|---|---|---|
| 1 周 | 0 | 几乎无影响 |
| 20 周 | ~0.16 | 分数降低 16% |
| 100 周 | 0.9 | 分数只剩 10% |
| 200+ 周 | 0.9 并加 `extreme_carrier_residual` 标记 | 直接 unreliable |

---

## 5. 为什么叫"学习"却没有训练

三个分析器都没有反向传播和训练：
- Transformer 的投影矩阵随机冻结
- GCN 没有可学习权重
- Temporal 的 EMA/方差是在线统计量

"学习"体现在：
1. **TemporalAnalyzer 的 EMA**：每颗卫星的历史均值/方差随时间更新。
2. **MemoryBuffer 的好原型**：被确认为好的样本会加入记忆库，供 Transformer 对比。

因此，系统的"经验"主要来自**时序统计**和**记忆库原型**。如果这两个机制没利用好 gnssfgo 的反馈残差，就会出现"运行很久也不见好"的现象。

---

## 6. 调试建议

想看每颗卫星三个分析器分别给了多少分，可以查看 `osqa_output.jsonl`：

```json
{
  "satellites": {
    "G01": {
      "details": {
        "transformer": 0.82,
        "graph": 0.91,
        "temporal": 0.95
      },
      "carrier_residual": -1.2,
      "flags": []
    }
  }
}
```

- `transformer` 低 → 看 `flags` 是 `low_attention` 还是 `low_memory_similarity`
- `graph` 低 → 看是否 `graph_inconsistent` 或 `no_graph_neighbors`
- `temporal` 低 → 看是否 `temporal_jump` 或 `persistent_anomaly`
- `carrier_residual` 绝对值大 → 会被强制降级
