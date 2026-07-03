#!/usr/bin/env python3
"""
run_analyzer.py — OSQA主入口程序
================================

启动在线信号质量分析器，与gnssfgo同步运行。

使用方式:
  1. 离线测试（CSV回放）:
     python run_analyzer.py --csv gnss_transformer_training_data.csv --output results.csv

  2. ROS在线模式:
     python run_analyzer.py --ros

  3. 混合模式（CSV输入 + ROS输出）:
     python run_analyzer.py --csv data.csv --ros

命令行参数:
  --csv PATH       CSV数据文件路径（离线模式）
  --ros            启用ROS模式
  --config PATH    配置文件路径（JSON格式）
  --output PATH    输出文件路径
  --urban          使用城市峡谷预设配置
  --open-sky       使用开阔天空预设配置
  --debug          启用调试模式
  --vis            启用可视化输出

Author: Claude Code
Date: 2026-07-02
"""

import sys
import os
import time
import argparse
import json
import signal
import numpy as np
from typing import Optional

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnss_quality_analyzer.config import (
    OSQAConfig, get_urban_config, get_open_sky_config, get_default_config
)
from gnss_quality_analyzer.memory_buffer import MemoryBuffer, EpochData, SatelliteSample
from gnss_quality_analyzer.feature_extractor import FeatureExtractor, RawObservation
from gnss_quality_analyzer.transformer_analyzer import TransformerAnalyzer
from gnss_quality_analyzer.graph_analyzer import GraphAnalyzer
from gnss_quality_analyzer.temporal_analyzer import TemporalAnalyzer
from gnss_quality_analyzer.quality_fusion import QualityFusion
from gnss_quality_analyzer.gnssfgo_bridge import GNSSFGOBridge


class OSQAAnalyzer:
    """
    OSQA 在线信号质量分析器 — 顶层协调器

    协调MemoryBuffer、三个分析器和融合器的工作流。
    封装了整个pipeline：数据输入 → 特征提取 → 分析 → 融合 → 输出。
    """

    def __init__(self, config: OSQAConfig):
        self.config = config

        # 初始化各模块
        self.memory = MemoryBuffer(
            window_size=config.window_size,
            memory_bank_size=config.memory_bank_size,
            feature_dims=config.feature_dims,
            ema_decay=config.ema_decay,
        )

        self.feature_extractor = FeatureExtractor(epsilon=config.epsilon)

        self.transformer_analyzer = TransformerAnalyzer(
            input_dim=config.feature_dims,
            num_heads=config.attention_heads,
            head_dim=config.head_dim,
            temperature=config.attention_temperature,
            dropout=config.attention_dropout,
        )

        self.graph_analyzer = GraphAnalyzer(
            elevation_threshold=config.graph_edge_elevation_threshold,
            azimuth_threshold=config.graph_edge_azimuth_threshold,
            consistency_temperature=config.graph_consistency_temperature,
            num_layers=config.gcn_num_layers,
            position_encoding_dim=config.position_encoding_dim,
        )

        self.temporal_analyzer = TemporalAnalyzer(
            ema_decay=config.temporal_ema_decay,
            var_decay=config.temporal_var_decay,
            anomaly_threshold=config.temporal_anomaly_threshold,
            n_features=config.feature_dims,
        )

        self.fusion = QualityFusion(
            mode=config.fusion_mode,
            weights=config.fusion_weights,
            quality_threshold=config.quality_threshold,
        )

        self.bridge = GNSSFGOBridge(config)

        # 运行状态
        self.running = False
        self.epoch_count = 0
        self.start_time: Optional[float] = None

    def setup(self, csv_path: Optional[str] = None,
              ros_mode: bool = False,
              output_path: Optional[str] = None):
        """设置通信桥梁"""
        self.bridge.setup(csv_path=csv_path, ros_mode=ros_mode, output_path=output_path)

    def start(self):
        """启动分析器"""
        self.bridge.start()
        self.running = True
        self.start_time = time.time()
        print(f"[OSQA] Started. Config: window={self.config.window_size}, "
              f"fusion={self.config.fusion_mode}, threshold={self.config.quality_threshold}")

    def stop(self):
        """停止分析器"""
        self.running = False
        self.bridge.stop()
        elapsed = time.time() - (self.start_time or time.time())
        print(f"[OSQA] Stopped. Processed {self.epoch_count} epochs in {elapsed:.1f}s "
              f"({self.epoch_count / max(elapsed, 0.01):.1f} Hz)")

        # 打印最终统计
        stats = self.get_statistics()
        print(f"[OSQA] Statistics:")
        print(f"  Memory: {json.dumps(stats['memory'], indent=2)}")
        print(f"  Transformer: {json.dumps(stats['transformer'], indent=2)}")
        print(f"  Graph: {json.dumps(stats['graph'], indent=2)}")
        print(f"  Temporal: {json.dumps(stats['temporal'], indent=2)}")
        print(f"  Fusion: {json.dumps(stats['fusion'], indent=2)}")

    def process_epoch(self, raw_epoch: dict) -> Optional[dict]:
        """
        处理一个epoch的原始数据

        这是主处理循环的核心。每个epoch调用一次。

        Args:
            raw_epoch: 来自DataProvider的原始数据字典

        Returns:
            融合后的质量评估结果字典，若处理失败返回None
        """
        try:
            # ========== 步骤1: 解析原始数据 ==========
            observations = self._parse_raw_epoch(raw_epoch)
            if len(observations) == 0:
                return None

            # ========== 步骤2: 特征提取 ==========
            timestamp = raw_epoch['timestamp']
            features, prns = self.feature_extractor.extract_batch(observations)

            # 设置归一化统计（使用记忆库的统计）
            if self.memory.global_stats.n_updates > 10:
                self.feature_extractor.set_statistics(
                    self.memory.global_stats.mean,
                    self.memory.global_stats.get_std(),
                )

            # 归一化
            features_norm = self.feature_extractor.normalize(features)

            # 提取辅助信息
            elevations = np.array([obs.elevation for obs in observations])
            azimuths = np.array([obs.azimuth for obs in observations])
            snrs = np.array([obs.snr for obs in observations])
            residuals = np.array([obs.pseudorange_residual for obs in observations])
            systems = [obs.system for obs in observations]
            mask = np.ones(len(observations), dtype=bool)

            # ========== 步骤3: Transformer分析 ==========
            # 获取记忆库中的好原型
            prototypes = self.memory.get_good_prototypes(k=50)
            proto_features = [p.features for p in prototypes] if prototypes else None

            attn_result = self.transformer_analyzer.analyze(
                features_norm, prns, proto_features, mask
            )

            # ========== 步骤4: 图结构分析 ==========
            graph_result = self.graph_analyzer.analyze(
                features_norm, elevations, azimuths, prns, residuals
            )

            # ========== 步骤5: 时序分析 ==========
            temporal_result = self.temporal_analyzer.analyze(
                features_norm, prns, mask
            )

            # ========== 步骤6: 分数融合 ==========
            fused = self.fusion.fuse(
                timestamp=timestamp,
                q_transformer=attn_result.quality_scores,
                q_graph=graph_result.quality_scores,
                q_temporal=temporal_result.quality_scores,
                prns=prns,
                systems=systems,
                transformer_flags=attn_result.anomaly_flags,
                graph_flags=graph_result.anomaly_flags,
                temporal_flags=temporal_result.anomaly_flags,
                snr_values=snrs,
                elevation_values=elevations,
                azimuth_values=azimuths,
                residual_values=residuals,
            )

            # ========== 步骤7: 更新记忆库 ==========
            epoch_data = self._build_epoch_data(
                timestamp, observations, features, fused
            )
            self.memory.add_epoch(epoch_data)

            # ========== 步骤8: 发布结果 ==========
            self.bridge.publish_quality(fused)

            self.epoch_count += 1

            # 定期打印状态
            if self.config.debug and self.epoch_count % 10 == 0:
                self._print_status(fused)

            return fused.to_dict()

        except Exception as e:
            print(f"[OSQA] Error processing epoch: {e}")
            if self.config.debug:
                import traceback
                traceback.print_exc()
            return None

    def run(self):
        """
        主处理循环

        持续从数据源获取epoch并处理，直到数据结束或收到停止信号。
        """
        self.start()

        while self.running:
            # 获取下一个epoch
            raw_epoch = self.bridge.receive_epoch(timeout=1.0)
            if raw_epoch is None:
                # 检查是否应该退出
                if self.epoch_count > 0:
                    print("[OSQA] No more data, stopping...")
                break

            # 处理
            self.process_epoch(raw_epoch)

        self.stop()

    def _parse_raw_epoch(self, raw_epoch: dict) -> list:
        """
        解析原始epoch数据为RawObservation列表

        支持CSV数据格式。
        """
        observations = []
        timestamp = raw_epoch.get('timestamp', 0.0)

        for sat_data in raw_epoch.get('satellites', []):
            sys_code = sat_data.get('sys', '0')
            system_map = {'1': 'G', '2': 'R', '3': 'E', '32': 'C'}
            system = system_map.get(str(sys_code), '?')

            obs = RawObservation(
                prn=sat_data.get('prn', '?'),
                system=system,
                timestamp=timestamp,
                snr=float(sat_data.get('snr', 0)),
                elevation=float(sat_data.get('elevation', 0)),
                azimuth=float(sat_data.get('azimuth', 0)),
                pseudorange_l1=float(sat_data.get('pseudorange', 0)),
                pseudorange_residual=float(sat_data.get('psr_residual', 0)),
            )
            observations.append(obs)

        return observations

    def _build_epoch_data(self, timestamp: float,
                           observations: list,
                           features: np.ndarray,
                           fused_result) -> EpochData:
        """构建EpochData对象"""
        epoch = EpochData(timestamp=timestamp)

        for i, obs in enumerate(observations):
            # 查找该卫星的融合质量分数
            sat_quality = None
            for sq in fused_result.satellites:
                if sq.prn == obs.prn:
                    sat_quality = sq
                    break

            sample = SatelliteSample(
                prn=obs.prn,
                system=obs.system,
                features=features[i],
                timestamp=timestamp,
                elevation=obs.elevation,
                azimuth=obs.azimuth,
                pseudorange_residual=obs.pseudorange_residual,
                snr=obs.snr,
                lock_count=0,
                is_good=(sat_quality.quality_final >= 0.7 if sat_quality else True),
                quality_score=(sat_quality.quality_final if sat_quality else 0.5),
            )
            epoch.satellites.append(sample)

        return epoch

    def _print_status(self, fused_result):
        """打印当前epoch的状态"""
        trusted = fused_result.n_trusted
        suspect = fused_result.n_suspect
        unreliable = fused_result.n_unreliable
        total = fused_result.n_total

        print(f"[Epoch {self.epoch_count}] "
              f"Satellites: {total} total | "
              f"✓{trusted} trusted | "
              f"?{suspect} suspect | "
              f"✗{unreliable} unreliable | "
              f"Avg quality: {fused_result.final_mean_quality:.3f}")

        if unreliable > 0 and self.config.debug:
            for sat in fused_result.satellites:
                if sat.trust_level.value == "unreliable":
                    print(f"  ✗ {sat.prn}: q={sat.quality_final:.3f} "
                          f"flags={sat.all_flags} "
                          f"SNR={sat.snr:.0f} elev={sat.elevation:.0f}°")

    def get_statistics(self) -> dict:
        """获取所有模块的统计信息"""
        return {
            "epoch_count": self.epoch_count,
            "memory": self.memory.get_statistics(),
            "transformer": self.transformer_analyzer.get_statistics(),
            "graph": self.graph_analyzer.get_graph_statistics(
                # 使用最近的图结果（如果有的话）
                self.graph_analyzer.prev_features is not None
            ),
            "temporal": self.temporal_analyzer.get_statistics(),
            "fusion": self.fusion.get_long_term_stats(),
        }


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="OSQA - GNSS在线信号质量分析器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_analyzer.py --csv data.csv
  python run_analyzer.py --csv data.csv --urban --output results.csv
  python run_analyzer.py --ros --open-sky
  python run_analyzer.py --csv data.csv --debug
        """
    )

    parser.add_argument('--csv', type=str, help='CSV数据文件路径（离线回放模式）')
    parser.add_argument('--ros', action='store_true', help='启用ROS模式')
    parser.add_argument('--config', type=str, help='配置文件路径（JSON）')
    parser.add_argument('--output', type=str, help='输出文件路径（CSV格式）')
    parser.add_argument('--urban', action='store_true', help='使用城市峡谷预设配置')
    parser.add_argument('--open-sky', action='store_true', help='使用开阔天空预设配置')
    parser.add_argument('--debug', action='store_true', help='启用调试输出')
    parser.add_argument('--vis', action='store_true', help='启用可视化输出')

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 加载配置
    if args.config:
        config = OSQAConfig.from_json(args.config)
    elif args.urban:
        config = get_urban_config()
    elif args.open_sky:
        config = get_open_sky_config()
    else:
        config = get_default_config()

    if args.debug:
        config.debug = True
    if args.vis:
        config.visualization = True

    # 创建分析器
    analyzer = OSQAAnalyzer(config)

    # 设置通信
    analyzer.setup(
        csv_path=args.csv,
        ros_mode=args.ros,
        output_path=args.output,
    )

    # 注册信号处理（支持Ctrl+C优雅退出）
    def signal_handler(sig, frame):
        print("\n[OSQA] Received stop signal...")
        analyzer.running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 运行
    analyzer.run()


if __name__ == "__main__":
    main()
