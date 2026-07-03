"""
GNSS在线信号质量分析器 (Online Signal Quality Analyzer, OSQA)
=============================================================

使用Transformer自注意力 + 图结构几何分析 + 时序一致性检测，
实现无需预训练的在线GNSS信号质量评估。

三个核心分析模块：
- TransformerAnalyzer: 基于特征相似度的自注意力异常检测
- GraphAnalyzer: 基于卫星空间分布的图结构一致性检查
- TemporalAnalyzer: 基于时间序列的突变检测

Author: Claude Code
Date: 2026-07-02
"""

__version__ = "0.1.0"
__author__ = "Claude Code"

from .config import OSQAConfig
from .memory_buffer import MemoryBuffer
from .feature_extractor import FeatureExtractor
from .transformer_analyzer import TransformerAnalyzer
from .graph_analyzer import GraphAnalyzer
from .temporal_analyzer import TemporalAnalyzer
from .quality_fusion import QualityFusion
from .gnssfgo_bridge import GNSSFGOBridge

__all__ = [
    "OSQAConfig",
    "MemoryBuffer",
    "FeatureExtractor",
    "TransformerAnalyzer",
    "GraphAnalyzer",
    "TemporalAnalyzer",
    "QualityFusion",
    "GNSSFGOBridge",
]
