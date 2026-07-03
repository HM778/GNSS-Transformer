# OSQA — GNSS在线信号质量分析器 详细解释

> 面向初学者：本文档用通俗语言解释Transformer自注意力、图结构等概念，以及它们如何用于GNSS信号质量分析。

---

## 目录

1. [问题背景：为什么加了四系统和L1L2反而没提升？](#1-问题背景)
2. [核心思路：用Transformer和图结构做信号"裁判"](#2-核心思路)
3. [Transformer自注意力 - 通俗解释](#3-transformer自注意力)
4. [图结构(GNN) - 通俗解释](#4-图结构gnn)
5. [时序分析 - 通俗解释](#5-时序分析)
6. [三个分析器如何协作](#6-三个分析器协作)
7. [为什么不需要预训练？](#7-为什么不需要预训练)
8. [如何使用OSQA](#8-如何使用osqa)
9. [与gnssfgo的集成细节](#9-与gnssfgo的集成)
10. [调优建议](#10-调优建议)
11. [常见问题](#11-常见问题)

---

## 1. 问题背景

### 为什么加了四系统和L1L2频段数据，定位效果反而没有明显提升？

想象一个场景：你有一个裁判团队（因子图优化器 gnssfgo），依靠多个"证人"（卫星信号）来判断一个位置。加入四系统和L1L2频段相当于：
- 之前：只有GPS的L1证人（约8-12个）
- 现在：GPS/GLO/GAL/BDS全部证人 + 每个证人提供两个版本(L1/L2)的证词（约25-40个）

**问题在于**：更多证人不等于更准确。如果新增的证人中有"说谎者"（被多路径/NLOS污染的不可信信号），它们会误导裁判。

### 现有trbin方案的局限

trbin使用"桶式统计"来评估信号质量。它把信号的SNR、仰角等特征分成几个离散的桶（比如SNR分为<35, 35-40, 40-45, >45四个桶），然后统计每个桶中"好"信号的比例。

这种方法的局限：
1. **信息损失**：SNR=34.9和SNR=35.1被分到不同桶，但实际上几乎一样
2. **忽略交互**：SNR低+仰角高（可能是大气异常）vs SNR低+仰角低（通常正常），桶式方法无法区分
3. **无卫星间对比**：不知道"相邻卫星表现如何"，无法利用空间一致性

---

## 2. 核心思路

OSQA用三个互补的分析视角来评估信号质量：

```
          卫星之间互相对比         天空中的几何关系       自己的历史表现
               ↓                       ↓                    ↓
        TransformerAnalyzer      GraphAnalyzer       TemporalAnalyzer
         "你和大家像不像？"     "你和邻居一致吗？"    "你和自己历史比，突变了没？"
```

**类比理解**：
- Transformer：把每颗卫星和所有其他卫星比较，就像把一张考卷和班里所有其他考卷比较，看谁答案和大家不一样
- Graph：只看相邻的卫星（天空中相近方向的卫星），就像同桌之间互相对答案
- Temporal：看这颗卫星自己之前的测量，就像跟踪一个学生的成绩变化，突然变差就警惕

---

## 3. Transformer自注意力 — 通俗解释

### 3.1 什么是"注意力"？

**生活中注意力的例子**：
你在一个嘈杂的餐厅里和朋友聊天。你的大脑自动"关注"朋友的声音，同时"忽略"其他桌的谈话声。这就是注意力机制——自动决定哪些信息重要、哪些不重要。

**数学上**：
注意力 = 计算一个"相似度分数" → 根据分数决定关注多少

### 3.2 什么是"自注意力"(Self-Attention)？

自注意力就是：一组元素**互相关注**。

在OSQA中，每颗卫星都"看到"所有其他卫星，然后自己决定"我该关注谁"。

**具体过程**（以3颗卫星为例）：

```
步骤1: 每颗卫星提取特征向量（8维数字，描述SNR、仰角等）

  卫星G05: [0.9, 0.85, 0.1, 0.2, 0.01, 0.0, 0.8, 0.0]   ← 正常信号
  卫星G13: [0.8, 0.80, 0.2, 0.3, 0.02, 0.0, 0.7, 0.0]   ← 正常信号
  卫星G29: [0.3, 0.10, 0.1, 0.1, 0.50, 0.0, 0.1, 0.0]   ← 可疑! SNR低，残差大

步骤2: 计算相似度

  每颗卫星问："我们有多相似？"
  G05和G13很相似（都是正常信号） → 相似度高
  G05和G29不相似（一个正常一个异常） → 相似度低
  G13和G29不相似 → 相似度低

步骤3: 分配注意力

  注意力矩阵（softmax归一化后）:
           G05   G13   G29
  G05  [ 0.48  0.47  0.05 ]   ← G05关注G13很多，关注G29很少
  G13  [ 0.47  0.48  0.05 ]   ← G13也是
  G29  [ 0.33  0.33  0.34 ]   ← G29对谁都差不多（它和谁都不像）

步骤4: 异常检测

  看"被关注度"：其他人对这颗卫星的平均关注
  - G05: (0.47+0.33)/2 = 0.40  ← 被高度关注 → 正常
  - G13: (0.48+0.33)/2 = 0.40  ← 被高度关注 → 正常
  - G29: (0.05+0.05)/2 = 0.05  ← 几乎不被关注 → 异常!
```

### 3.3 多头注意力是什么？

就像请了多个专家从不同角度看问题：

- **专家1**：主要看SNR和仰角的相似度
- **专家2**：主要看残差和锁定计数的相似度
- **专家3**：主要看几何位置(方位角)的相似度
- **专家4**：看所有特征的全局相似度

每个专家使用不同的随机投影矩阵（让每个专家有不同的"视角"），最终综合所有专家的意见。

### 3.4 记忆库增强

除了卫星之间的互相对比，还和"历史好样本"对比。

记忆库中存储了200个历史被确认为"好"的信号特征。每颗新信号不仅要和其他当前卫星比较，还要和历史好样本比较。

```
如果卫星G29的特征和历史好样本都不像 → 很可能是异常
如果卫星G05的特征和历史好样本很像 → 进一步确认可信
```

---

## 4. 图结构(GNN) — 通俗解释

### 4.1 什么是图(Graph)？

图由**节点**(Node)和**边**(Edge)组成。

在OSQA中：
- **节点** = 每颗卫星
- **边** = 两颗卫星在天空中位置相近（仰角差<30° 且 方位角差<60°）

```
天空中的卫星分布（俯视图）:

        北 (0°)
          │
     C108 ●
          │    ● G05
          │
西 ──────┼────── 东
  (270°) │       (90°)
          │  ● G13
          │     ● G29
          │
        南 (180°)

在这个图中：
- G05和G13距离近（都在东南方向）→ 有边连接
- C108在北方，和其他卫星都远 → 孤立节点
```

### 4.2 图卷积(GCN)消息传递

每颗卫星"询问"它的邻居：

```
卫星G05问邻居G13："你的特征是什么？"
G13回答："我的SNR=38, 仰角=45°, 残差=2.3m"

G05对比：
  我的特征: SNR=40, 仰角=50°, 残差=2.1m
  邻居的特征: SNR=38, 仰角=45°, 残差=2.3m
  差异很小 → 一致 → 可信
```

但如果：
```
卫星G29的邻居G13报告："残差=2.3m"
而G29自己的残差=25m！
差异巨大 → 不一致 → G29可能有异常
```

### 4.3 为什么图结构有预测能力？

卫星在天空中的运动是平滑的。如果我们知道某颗卫星之前的位置和速度，就可以预测它下一时刻的位置。图结构可以利用相邻卫星的信息来改善这个预测：

```
预测下一时刻G05的仰角:
  直接外推: elev[t+1] ≈ elev[t] + (elev[t] - elev[t-1])
  图约束: 邻居G13的仰角也以相似速率变化
  平滑预测: 0.7 * 直接外推 + 0.3 * 邻居的平均预测
```

---

## 5. 时序分析 — 通俗解释

### 5.1 核心思想

正常的GNSS信号随时间**缓慢平滑变化**。如果突然出现跳变，通常是异常：

```
正常信号（SNR随时间变化）:
  45.2 → 45.0 → 44.8 → 44.5 → 44.3 → 44.0  (平滑下降)

异常信号（发生遮挡后重新出现）:
  44.0 → 43.8 → [信号丢失] → 28.1 → 33.5 → 37.2  (突然跳变!)
```

### 5.2 EMA（指数移动平均）

EMA是对历史值的加权平均，最近的值权重最大：

```
EMA[今天] = 0.8 × EMA[昨天] + 0.2 × 今天的实际值
```

这代表"预期的正常值"。如果今天的实际值偏离EMA太远 → 异常。

### 5.3 能检测的异常类型

1. **载波相位跳变(Cycle Slip)**：载波相位突然跳变整数个波长，残差特征骤变
2. **多路径效应出现**：车辆驶入高楼区，SNR突然下降
3. **卫星遮挡后重跟踪**：lock_count归零又增长，初始测量不稳定
4. **电离层闪烁**：信号强度和相位快速波动

---

## 6. 三个分析器如何协作

最终的信号质量分数由三个分析器融合得到：

```
quality_final = (q_transformer × q_graph × q_temporal)^(1/3)
```

这是**几何平均**，意味着：
- 如果三个分析器都给出高分 → 最终分数高（可信）
- 如果任何一个分析器给出低分 → 最终分数被拉低
- 但不会像直接乘法那样过度惩罚

**举例**：
- 卫星G05：三个分数[0.9, 0.85, 0.95] → 几何平均=0.90 → **可信**
- 卫星G13：三个分数[0.8, 0.3, 0.9] → 几何平均=0.60 → **可疑**（图分析发现问题）
- 卫星G29：三个分数[0.2, 0.15, 0.4] → 几何平均=0.23 → **不可信**

---

## 7. 为什么不需要预训练？

这是OSQA最核心的设计特点。传统的深度学习模型需要大量标注数据训练，但OSQA可以开机即用。

### 传统Transformer（需要训练）：
```
输入 → [学习的QKV投影] → 注意力 → [学习的前馈网络] → 输出
        ↑ 需要训练         ↑ 需要训练
```

### OSQA的Transformer（不需要训练）：
```
输入 → [随机但固定的投影] → 注意力 → [直接计算异常指标] → 输出
        ↑ 初始化后不变        ↑ 纯数学计算，无需训练
```

**关键区别**：

1. **QKV投影**：使用Xavier初始化的随机投影（不学习）
   - 随机投影保留了特征空间的距离关系（数学上保证）
   - 不同的随机投影提供了不同的"视角"

2. **注意力计算**：基于特征相似度（不是学到的语义相似度）
   - 如果两颗卫星的特征数值相似 → 注意力自然高
   - 没有任何需要学习的参数

3. **异常判断**：基于统计指标（不是学到的分类器）
   - "被关注度"低于阈值 → 异常
   - 不需要学习"什么算异常"

**类比**：就像用尺子测量身高，你不需要"训练"尺子。相似度计算就是OSQA的"尺子"——直接测量特征之间的距离。

### 可演进的在线学习
虽然不需要预训练，但OSQA可以在运行中逐步改进：
- 用FGO优化后的残差作为反馈信号
- 统计哪些特征模式产生了好的结果
- 微调投影矩阵和阈值

---

## 8. 如何使用OSQA

### 8.1 安装依赖

```bash
pip install numpy
# ROS模式还需要:
# pip install rospkg (如果使用ROS)
```

OSQA只依赖numpy，不依赖PyTorch！这使得它可以轻量化部署。

### 8.2 命令行使用

```bash
# 1. 离线测试（CSV回放）
cd GNSS-Transformer
python gnss_quality_analyzer/run_analyzer.py \
    --csv gnss_transformer_training_data.csv \
    --output results.csv \
    --debug

# 2. 城市峡谷场景（更严格的异常检测）
python gnss_quality_analyzer/run_analyzer.py \
    --csv data.csv \
    --urban \
    --output results.csv

# 3. 开阔天空场景（更宽松）
python gnss_quality_analyzer/run_analyzer.py \
    --csv data.csv \
    --open-sky

# 4. 使用自定义配置
python gnss_quality_analyzer/run_analyzer.py \
    --config my_config.json \
    --csv data.csv
```

### 8.3 Python API

```python
from gnss_quality_analyzer import (
    OSQAConfig, MemoryBuffer, FeatureExtractor,
    TransformerAnalyzer, GraphAnalyzer, TemporalAnalyzer,
    QualityFusion
)

# 创建配置
config = OSQAConfig(
    window_size=50,
    fusion_mode="geometric",
    quality_threshold=0.3,
)

# 初始化模块
memory = MemoryBuffer(config.window_size, config.memory_bank_size)
extractor = FeatureExtractor()
transformer = TransformerAnalyzer(config.feature_dims, config.attention_heads)
graph_analyzer = GraphAnalyzer()
temporal = TemporalAnalyzer()
fusion = QualityFusion(config.fusion_mode)

# 处理每个epoch
for raw_data in data_source:
    features, prns = extractor.extract_batch(raw_data.observations)
    features_norm = extractor.normalize(features)

    attn_result = transformer.analyze(features_norm, prns)
    graph_result = graph_analyzer.analyze(features_norm, elevations, azimuths, prns)
    temporal_result = temporal.analyze(features_norm, prns)

    fused = fusion.fuse(
        timestamp, attn_result.quality_scores,
        graph_result.quality_scores, temporal_result.quality_scores,
        prns, systems, ...
    )

    # fused.get_quality_weights() → {prn: quality_score}
    # 将这些权重传递给gnssfgo
```

### 8.4 运行测试

```bash
cd GNSS-Transformer
python gnss_quality_analyzer/test_osqa.py
```

输出会显示每个模块的测试结果，包括异常检测的演示。

---

## 9. 与gnssfgo的集成

### 9.1 集成架构

```
┌──────────┐  ROS Topics   ┌──────────┐  ROS Topic     ┌──────────┐
│ u-blox   │──────────────►│  OSQA    │──────────────►│ gnssfgo  │
│ receiver │               │ 节点     │                │ (trbin)  │
└──────────┘               └──────────┘                └──────────┘
                            信号质量分析              使用质量分数
                            输出质量分数              调整因子权重
```

### 9.2 gnssfgo端需要添加的代码

在 `trbin_factorgraph.hpp` 中：

```cpp
// 1. 添加订阅者
ros::Subscriber quality_sub = nh.subscribe("/osqa/signal_quality", 10,
    &TRBinning::qualityCallback, this);

// 2. 存储最新质量分数
std::map<std::string, double> latest_quality_scores;
std::mutex quality_mutex;

// 3. 回调函数
void qualityCallback(const std_msgs::String::ConstPtr& msg) {
    // 解析JSON获取每颗卫星的质量分数
    // 存储到 latest_quality_scores
}

// 4. 在 addPsrFactors() 中使用:
for (each satellite) {
    double quality = 1.0;  // 默认完全信任
    auto it = latest_quality_scores.find(prn);
    if (it != latest_quality_scores.end()) {
        quality = it->second;
    }
    // 将质量分数应用到信息矩阵
    sqrt_info *= quality;  // 不可信信号的信息量降低
}
```

### 9.3 ROS消息格式

OSQA发布到 `/osqa/signal_quality` 的JSON格式：

```json
{
  "timestamp": 1733984500.0,
  "satellites": {
    "G05": {
      "prn": "G05",
      "system": "G",
      "quality": 0.92,
      "trust_level": "trusted",
      "details": {
        "transformer": 0.89,
        "graph": 0.94,
        "temporal": 0.95
      },
      "flags": [],
      "snr": 44.5,
      "elevation": 65.2,
      "azimuth": 120.3,
      "pseudorange_residual": 0.85
    },
    "G13": {
      "quality": 0.15,
      "trust_level": "unreliable",
      "flags": ["low_attention", "graph_inconsistent"],
      ...
    }
  },
  "summary": {
    "n_total": 25,
    "n_trusted": 22,
    "n_suspect": 2,
    "n_unreliable": 1
  },
  "fusion_mode": "geometric"
}
```

---

## 10. 调优建议

### 10.1 关键参数

| 参数 | 默认值 | 效果 | 何时调整 |
|------|--------|------|---------|
| `window_size` | 50 | 更大→统计更稳定，响应更慢 | 静态场景增大，动态场景减小 |
| `attention_temperature` | 1.0 | 更小→异常检测更敏感 | 城市峡谷减小(0.6-0.8) |
| `graph_consistency_temperature` | 0.5 | 更小→一致性要求更高 | 开阔天空增大(0.8-1.0) |
| `temporal_anomaly_threshold` | 3.0 | 更小→检测更多跳变 | 高动态场景减小(2.0) |
| `quality_threshold` | 0.3 | 低于此值排除信号 | 精度要求高时降低(0.2) |
| `fusion_mode` | geometric | 融合策略 | 高风险场景用multiply |

### 10.2 场景预设

```python
# 城市峡谷（保守）
config = get_urban_config()
# → 更严格的异常检测，宁可少用信号也不放过不可信信号

# 开阔天空（宽松）
config = get_open_sky_config()
# → 更宽松的异常检测，相信大多数信号是好的

# 默认（平衡）
config = get_default_config()
# → 适合大多数场景
```

### 10.3 调试技巧

启用debug模式查看详细输出：
```bash
python run_analyzer.py --csv data.csv --debug
```

会打印每个epoch的：
- 可信/可疑/不可信卫星数量
- 被标记为不可信的卫星及其原因
- 每颗卫星的具体标记（是Transformer、Graph还是Temporal发现了问题）

---

## 11. 常见问题

### Q: OSQA会增加多少计算延迟？
A: 对于25颗卫星的典型epoch，计算时间约2-5ms（纯numpy，无GPU）。远低于gnssfgo的Ceres优化时间（通常20-50ms）。

### Q: 如果没有足够的历史数据（刚启动），OSQA表现如何？
A: 启动阶段（前几个epoch）：
- Transformer：卫星间对比立即可用（不需要历史）
- Graph：图结构分析立即可用（不需要历史）
- Temporal：前3-5个epoch在建立基线，给中间分(0.5)
- 综合：启动后5-10秒即可全功能运行

### Q: 如果所有卫星信号都差（比如在隧道里），OSQA会怎么做？
A: OSQA检测的是**相对异常**（一颗卫星和其他卫星的差异），而非绝对质量。如果所有卫星都差：
- Transformer不会标记异常（大家都差，没有"异常"的）
- Graph会根据邻居一致性判断（同一区域的卫星都差→可能是该方向有遮挡）
- 融合后整体质量分数会偏低，但不会错误地只排除个别卫星

### Q: 能否用实时数据在线微调模型？
A: 可以。设置 `online_finetune=True` 后，OSQA会使用FGO返回的残差作为弱监督信号，逐步调整Transformer的投影矩阵。需要先积累约100个epoch的数据。

### Q: 和trbin的桶式自适应加权如何共存？
A: 两种方式可以互补：
1. **串行**：OSQA先过滤明显不可信的信号 → trbin再对剩余信号做桶式加权
2. **并行**：两个系统的输出取更保守的一个（min(trbin_weight, osqa_quality)）
3. **替代**：用OSQA完全替代trbin的桶式系统（如果OSQA表现更好）

---

## 附录：模块速查

| 模块 | 文件 | 核心职责 |
|------|------|---------|
| 配置 | `config.py` | 所有可调参数 |
| 记忆库 | `memory_buffer.py` | 滑动窗口+好样本存储 |
| 特征提取 | `feature_extractor.py` | 原始观测→8维特征向量 |
| Transformer | `transformer_analyzer.py` | 自注意力异常检测 |
| 图分析 | `graph_analyzer.py` | 几何一致性检查+预测 |
| 时序分析 | `temporal_analyzer.py` | 时序突变检测 |
| 分数融合 | `quality_fusion.py` | 三分析器分数融合 |
| 通信桥梁 | `gnssfgo_bridge.py` | ROS/文件/管道通信 |
| 主入口 | `run_analyzer.py` | 完整pipeline协调 |
| 测试 | `test_osqa.py` | 集成测试套件 |
