"""
config.py — OSQA 全局配置
=========================
定义在线信号质量分析器的所有可配置参数。
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict

@dataclass
class OSQAConfig:
    """
    OSQA配置类

    使用dataclass可以方便地从JSON/YAML文件加载和保存配置。
    所有参数都有合理的默认值，开机即可使用。
    """

    # ==================== 滑动窗口配置 ====================
    # 滑动窗口大小（epochs），窗口越大，历史信息越多，但内存消耗也越大
    window_size: int = 50

    # 记忆库最大样本数 — 存储"好"特征向量的最大数量
    # 类似trbin的history_epoch_window（200），但存储连续向量而非离散桶
    memory_bank_size: int = 200

    # ==================== 特征提取配置 ====================
    # 特征维度（见feature_extractor.py中的详细说明）
    # 8维: SNR, sin(仰角), cos(方位角), sin(方位角),伪距残差, 载波残差, 锁定计数, 仰角变化率
    feature_dims: int = 8

    # 特征归一化时使用的epsilon值（防止除零）
    epsilon: float = 1e-8

    # 指数移动平均(EMA)的衰减因子
    # 0.9表示新值的权重为0.1，旧EMA的权重为0.9
    # 值越大，统计量变化越慢（越保守）
    ema_decay: float = 0.9

    # ==================== Transformer 注意力配置 ====================
    # 多头注意力的头数
    # 每个头使用不同的随机投影，从不同角度评估卫星间的相似度
    # 4个头是效率和效果的平衡点
    attention_heads: int = 4

    # 每个注意力头的特征维度
    # 总特征维度 = attention_heads * head_dim
    head_dim: int = 32

    # 注意力dropout比例（正则化，防止过拟合）
    attention_dropout: float = 0.1

    # 用于计算异常分数的温度参数
    # 温度越高，注意力分布越平滑（异常检测越宽松）
    # 温度越低，注意力越集中（异常检测越严格）
    attention_temperature: float = 1.0

    # 被关注度阈值 — 低于此值的卫星被视为"孤立"（可能是异常）
    # 范围[0, 1]，推荐0.3-0.5
    isolation_threshold: float = 0.4

    # ==================== 图结构分析配置 ====================
    # 图边构建的角度阈值（度）
    # 两颗卫星如果仰角差小于此值 AND 方位角差小于2倍此值，则连接
    # 这模拟了"天空中的相邻区域"
    graph_edge_elevation_threshold: float = 30.0
    graph_edge_azimuth_threshold: float = 60.0

    # 图一致性分析的温度参数
    # 温度越高，对不一致的容忍度越高
    graph_consistency_temperature: float = 0.5

    # GCN消息传递的层数（1层=只看直接邻居，2层=看邻居的邻居）
    gcn_num_layers: int = 2

    # 位置编码维度（用于将卫星的仰角/方位角编码为高维向量）
    position_encoding_dim: int = 16

    # ==================== 时序分析配置 ====================
    # 特征EMA的衰减因子（与时序跟踪相关）
    temporal_ema_decay: float = 0.8

    # 波动性的衰减因子
    temporal_var_decay: float = 0.9

    # 马氏距离的异常阈值（超过此值视为突变）
    # 3.0 ≈ 3个标准差（对于正态分布，3sigma外概率~0.3%）
    temporal_anomaly_threshold: float = 3.0

    # ==================== 融合配置 ====================
    # 融合模式: 'geometric' (几何平均, 默认), 'multiply' (直接乘), 'min' (最小值), 'weighted' (加权平均)
    fusion_mode: str = "geometric"

    # 加权平均模式下的权重（若启用）
    fusion_weights: Dict[str, float] = field(default_factory=lambda: {
        "transformer": 0.4,
        "graph": 0.35,
        "temporal": 0.25,
    })

    # 质量分数阈值 — 低于此值的信号被视为不可信
    quality_threshold: float = 0.3

    # ==================== 系统配置 ====================
    # ROS topic名称
    ros_topic_input_meas: str = "/ublox_driver/range_meas"
    ros_topic_input_ephem: str = "/ublox_driver/ephem"
    ros_topic_input_glo_ephem: str = "/ublox_driver/glo_ephem"
    ros_topic_input_lla: str = "/ublox_driver/receiver_lla"
    ros_topic_output: str = "/osqa/signal_quality"

    # 处理频率（Hz），0表示每个epoch都处理
    processing_rate: float = 1.0

    # 是否启用调试输出（详细日志）
    debug: bool = False

    # 是否启用可视化（注意力矩阵热力图等）
    visualization: bool = False

    # ==================== gnssfgo 文件交换配置 ====================
    # gnssfgo 写入 JSONL 输入文件且 OSQA 写入输出的目录/路径
    # gnssfgo JSONL 模式下的输入文件（gnssfgo → OSQA）
    gnssfgo_input_path: str = "./osqa_input.jsonl"
    # gnssfgo JSONL 模式下的输出文件（OSQA → gnssfgo）
    gnssfgo_output_path: str = "./osqa_output.jsonl"
    # gnssfgo JSONL 模式下读取新数据的轮询间隔（秒）
    gnssfgo_poll_interval_s: float = 0.1

    # ==================== 在线学习配置 ====================
    # 是否启用在线微调（使用FGO残差反馈调整投影矩阵）
    online_finetune: bool = False

    # 在线微调的学习率
    finetune_learning_rate: float = 1e-4

    # 启动在线微调前需要累积的最小epoch数
    finetune_min_epochs: int = 100

    def to_dict(self) -> dict:
        """将配置序列化为字典（便于JSON保存）"""
        import dataclasses
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "OSQAConfig":
        """从字典加载配置"""
        return cls(**config_dict)

    @classmethod
    def from_json(cls, path: str) -> "OSQAConfig":
        """从JSON文件加载配置"""
        import json
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))

    def to_json(self, path: str):
        """保存配置到JSON文件"""
        import json
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# 预设配置：不同使用场景
def get_urban_config() -> OSQAConfig:
    """
    城市峡谷场景的预设配置

    城市环境中多路径严重，需要更严格的异常检测：
    - 注意力温度更低（更敏感）
    - 质量阈值更高
    - 图一致性温度更低（对不一致更敏感）
    """
    return OSQAConfig(
        attention_temperature=0.8,
        graph_consistency_temperature=0.3,
        quality_threshold=0.35,
        temporal_anomaly_threshold=2.5,
    )


def get_open_sky_config() -> OSQAConfig:
    """
    开阔天空场景的预设配置

    开阔环境中信号普遍较好，可以降低检测灵敏度：
    - 注意力温度更高（更宽松）
    - 质量阈值更低
    - 窗口可以更大
    """
    return OSQAConfig(
        window_size=100,
        attention_temperature=1.2,
        graph_consistency_temperature=0.8,
        quality_threshold=0.2,
        temporal_anomaly_threshold=4.0,
    )


def get_default_config() -> OSQAConfig:
    """获取默认配置"""
    return OSQAConfig()
