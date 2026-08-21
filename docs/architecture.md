# GNSS-Transformer / OSQA 工程架构说明

本文档说明 `GNSS-Transformer/gnss_quality_analyzer` 目录下各模块的职责、调用关系和数据流，帮助理解整个工程质量分析（OSQA）部分的逻辑。

---

## 1. 工程定位

`GNSS-Transformer` 在这里只运行 **OSQA（Online Satellite Quality Analyzer）** 部分：

- **输入**：`gnssfgo` 写入 `osqa_input.jsonl` 的逐历元卫星观测信息。
- **处理**：对每颗卫星提取特征，并用三种分析器评估其可信度。
- **输出**：把每颗卫星的综合置信度写回 `osqa_output.jsonl`，供 `gnssfgo` 读取并用于调整因子权重。

> 本仓库中 `scripts/`、`results/`、`README.md` 等还保留了原始 Transformer 训练/推理的入口，但在与 `gnssfgo` 联合运行时，实际只用到 `gnss_quality_analyzer/` 包。

---

## 2. 顶层调用关系

```text
run.sh
  └── gnome-terminal
        └── python3 run_analyzer.py --gnssfgo-input ... --gnssfgo-output ... --permissive --vis --layout compact
              │
              ├── 配置加载 ──► config.py (OSQAConfig)
              │
              ├── 数据输入 ──► gnssfgo_bridge.py (GNSSFGOBridge)
              │                    │
              │                    ├── 读取 osqa_input.jsonl
              │                    │       └── gnssfgo_data_reader.py ──► GnssEpoch / SatelliteInfo
              │                    │
              │                    └── 写入 osqa_output.jsonl
              │                            └── quality_fusion.py ──► FusedResult.to_dict()
              │
              ├── 特征提取 ──► feature_extractor.py (FeatureExtractor)
              │                    └── 将 SatelliteInfo 转为特征向量
              │
              ├── 三种分析器 ──┬─► transformer_analyzer.py (TransformerAnalyzer)
              │                ├─► graph_analyzer.py (GraphAnalyzer)
              │                └─► temporal_analyzer.py (TemporalAnalyzer)
              │
              ├── 质量融合 ──► quality_fusion.py (QualityFusion)
              │                    └── 融合三种分数，生成 FusedResult
              │
              ├── 记忆/历史 ──► memory_buffer.py (MemoryBuffer)
              │                    └── 保存最近若干历元特征，供 Temporal/Graph 使用
              │
              └── 可视化 ──► visualizer.py / visualize_output.py (Visualizer)
                            └── 天空图 + 置信度柱状图
```

---

## 3. 各文件作用

### 3.1 入口与协调

| 文件 | 核心类/函数 | 作用 |
|------|------------|------|
| `run_analyzer.py` | `OSQAAnalyzer`, `main()` | 程序入口。解析命令行参数，加载配置，实例化桥接、分析器、融合器、可视化器，并启动主循环。 |
| `config.py` | `OSQAConfig` | 配置数据中心。包含默认、urban、open-sky、permissive 四种预设，可被 JSON 文件覆盖。 |

### 3.2 与 gnssfgo 的交互

| 文件 | 核心类/函数 | 作用 |
|------|------------|------|
| `gnssfgo_bridge.py` | `GNSSFGOBridge` | 与 `gnssfgo` 进程通信。负责轮询/监听 `osqa_input.jsonl`，把分析结果写入 `osqa_output.jsonl`。 |
| `gnssfgo_data_reader.py` | `read_gnssfgo_input()`, `GnssEpoch` | 解析 `gnssfgo` 输出的 JSONL 格式，转换为内部数据结构。 |

### 3.3 核心分析算法

| 文件 | 核心类/函数 | 作用 |
|------|------------|------|
| `feature_extractor.py` | `FeatureExtractor`, `RawObservation` | 把每颗卫星的原始观测（SNR、仰角、方位角、伪距残差、载波相位等）归一化为固定维度的特征向量。 |
| `transformer_analyzer.py` | `TransformerAnalyzer` | 用自注意力机制评估卫星之间的相似度。被其他卫星“关注”少的卫星视为异常。 |
| `graph_analyzer.py` | `GraphAnalyzer` | 根据卫星在天空中的方位角/仰角建图，让相邻卫星互相投票，检测局部异常。 |
| `temporal_analyzer.py` | `TemporalAnalyzer` | 维护每颗卫星的历史特征 EMA 与协方差，用马氏距离检测突发异常和周跳。 |
| `quality_fusion.py` | `QualityFusion`, `FusedResult` | 把三种分析器的分数按配置方式（几何平均 / 加权平均 / 最小值等）融合为最终置信度，并划分 `trusted`/`suspect`/`unreliable`。 |

### 3.4 辅助模块

| 文件 | 核心类/函数 | 作用 |
|------|------------|------|
| `memory_buffer.py` | `MemoryBuffer`, `EpochData`, `SatelliteSample` | 滑动窗口缓存最近 N 个历元的特征数据，供 Graph 和 Temporal 分析器使用。 |
| `visualizer.py` | `Visualizer` | 兼容层，重新导出 `visualize_output.Visualizer`，保持旧导入路径可用。 |
| `visualize_output.py` | `Visualizer`, `_result_from_json()`, `main()` | 实时可视化模块，同时也是 `Visualizer` 类的实际存放地。支持 `full` 四子图和 `compact` 双子图两种布局。 |
| `test_osqa.py` | 测试函数 | 单元测试与离线验证脚本。 |

---

## 4. 单次历元的数据流

```text
osqa_input.jsonl (由 gnssfgo 写入)
        │
        ▼
GNSSFGOBridge 读取新行
        │
        ▼
gnssfgo_data_reader.py 解析为 GnssEpoch
        │
        ├──► FeatureExtractor 提取每颗卫星特征
        │           │
        │           ▼
        │    MemoryBuffer 存入当前历元
        │           │
        │           ▼
        ├──── TransformerAnalyzer ──► transformer_score
        │
        ├──── GraphAnalyzer ────────► graph_score
        │
        └──── TemporalAnalyzer ─────► temporal_score
                    │
                    ▼
            QualityFusion 融合
                    │
                    ▼
            FusedResult (每颗卫星 final_score + trust_level)
                    │
                    ├──► GNSSFGOBridge 写入 osqa_output.jsonl
                    │           (gnssfgo 读取后调整因子权重)
                    │
                    └──► Visualizer 更新窗口
```

---

## 5. 与 gnssfgo 的闭环

```text
        ┌─────────────────────────────────────────────────────────────┐
        │                         gnssfgo                              │
        │  1. 收集卫星观测                                              │
        │  2. 写入 osqa_input.jsonl                                     │
        │  3. 等待/读取 osqa_output.jsonl                                │
        │  4. 用 quality_score 调整伪距/载波相位因子权重                  │
        │  5. 执行因子图优化并输出定位结果                                │
        └─────────────────────────────────────────────────────────────┘
                            ▲                          │
                            │                          │
                   osqa_output.jsonl           osqa_input.jsonl
                            │                          │
                            │                          ▼
        ┌─────────────────────────────────────────────────────────────┐
        │              GNSS-Transformer / OSQA                         │
        │  run_analyzer.py → 三种分析器 → 融合 → 输出 quality_score     │
        └─────────────────────────────────────────────────────────────┘
```

---

## 6. 关键参数入口

| 参数/配置项 | 文件 | 说明 |
|------------|------|------|
| `--permissive` | `run_analyzer.py` | 使用宽松学习策略，降低正常卫星被误判的概率。 |
| `--layout compact/full` | `run_analyzer.py` → `Visualizer` | 可视化布局。 |
| `quality_threshold` | `config.py` | 低于此分数视为 `unreliable`。 |
| `fusion_mode` | `config.py` | 三种分析器分数的融合方式：`weighted` / `geometric` / `min` / `multiply`。 |
| `attention_temperature` / `graph_consistency_temperature` | `config.py` | 控制 Transformer 和 Graph 分析器的敏感程度，越高越宽松。 |
| `temporal_anomaly_threshold` | `config.py` | 时序马氏距离异常阈值，越高越宽松。 |

---

## 7. 常见调试路径

- **OSQA 没输出 / `osqa_output.jsonl` 为空**：先检查 `gnssfgo` 是否以 `-DENABLE_TRANSFORMER_BRIDGE=ON` 编译，以及 `run.sh` 中 `ENABLE_OSQA` 是否为 `true`。
- **所有卫星都是 suspect/unreliable**：使用 `--permissive` 或调低 `quality_threshold`、提高 `attention_temperature` / `graph_consistency_temperature`。
- **可视化窗口白屏/不刷新**：检查是否有 GUI 后端（`MPLBACKEND=TkAgg`），或在 `visualize_output.py` 中确认字体/布局设置。
