"""
gnssfgo_bridge.py — 与gnssfgo的通信接口
========================================

负责OSQA和gnssfgo之间的数据交换。

支持两种模式：
1. ROS模式（推荐）：通过ROS topic通信，OSQA作为独立节点运行
2. 文件模式：通过JSON文件通信（简单但延迟高，用于测试和调试）
3. 标准输入模式：通过管道通信（用于非ROS环境）

=== ROS模式详解 ===

OSQA订阅以下ROS topics（与gnssfgo相同的输入）：
  - /ublox_driver/range_meas  (GnssMeasMsg): 原始观测数据
  - /ublox_driver/ephem       (GnssEphemMsg): 广播星历
  - /ublox_driver/glo_ephem   (GnssGloEphemMsg): GLONASS星历
  - /ublox_driver/receiver_lla (NavSatFix): 接收机位置

OSQA发布以下topic：
  - /osqa/signal_quality      (String/JSON): 信号质量评估结果

gnssfgo需要订阅 /osqa/signal_quality 并使用质量分数调整因子权重。

=== gnssfgo端修改指南 ===

在 trbin_factorgraph.hpp 的 addPsrFactors() 和 addTRDDCPFactors() 中：
  1. 从 /osqa/signal_quality 获取当前epoch的质量分数
  2. 对每颗卫星，将 sqrt_info *= quality_score
  3. 可选：完全跳过 quality_score < threshold 的卫星

Author: Claude Code
Date: 2026-07-02
"""

import json
import time
import threading
import numpy as np
from typing import Dict, List, Optional, Callable
from abc import ABC, abstractmethod


class GNSSDataProvider(ABC):
    """
    抽象数据提供者接口

    定义OSQA获取GNSS数据的方式。不同的实现对应不同的数据源。
    """

    @abstractmethod
    def start(self):
        """启动数据流"""
        pass

    @abstractmethod
    def stop(self):
        """停止数据流"""
        pass

    @abstractmethod
    def get_next_epoch(self, timeout: float = 1.0) -> Optional[dict]:
        """
        获取下一个epoch的数据

        Returns:
            包含原始观测数据的字典，若超时则返回None
        """
        pass


class FileDataProvider(GNSSDataProvider):
    """
    文件模式数据提供者

    从CSV文件读取数据，用于离线测试和调试。
    支持的CSV格式与gnss_transformer_training_data.csv相同。

    列格式：
      timestamp, week, tow, prn, sys, snr, azimuth, elevation,
      pseudorange, doppler, psr_residual, spp_x, spp_y, spp_z,
      gt_lat, gt_lon, gt_h
    """

    def __init__(self, csv_path: str, rate_hz: float = 1.0):
        """
        Args:
            csv_path: CSV文件路径
            rate_hz: 回放速率（Hz），1.0=实时，2.0=2倍速
        """
        self.csv_path = csv_path
        self.rate_hz = rate_hz
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._epochs: List[dict] = []
        self._current_idx = 0
        self._lock = threading.Lock()

    def start(self):
        """加载CSV并启动回放"""
        import csv

        # 按timestamp分组epoch
        epochs_map: Dict[float, List[dict]] = {}
        with open(self.csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts = float(row.get('timestamp', 0))
                if ts not in epochs_map:
                    epochs_map[ts] = []
                epochs_map[ts].append(row)

        # 转换为epoch字典列表
        for ts in sorted(epochs_map.keys()):
            epoch = {
                'timestamp': ts,
                'satellites': []
            }
            for row in epochs_map[ts]:
                sat = {
                    'prn': row.get('prn', ''),
                    'sys': str(row.get('sys', '0')),
                    'snr': float(row.get('snr', 0)),
                    'azimuth': float(row.get('azimuth', 0)),
                    'elevation': float(row.get('elevation', 0)),
                    'pseudorange': float(row.get('pseudorange', 0)),
                    'doppler': float(row.get('doppler', 0)),
                    'psr_residual': float(row.get('psr_residual', 0)),
                }
                epoch['satellites'].append(sat)
            self._epochs.append(epoch)

        self._running = True
        self._current_idx = 0
        print(f"[FileDataProvider] Loaded {len(self._epochs)} epochs from {self.csv_path}")

        # 启动回放线程
        self._thread = threading.Thread(target=self._replay_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """停止回放"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _replay_loop(self):
        """回放循环：按rate_hz速率输出epoch"""
        if not self._epochs:
            return

        start_time = time.time()
        data_start_ts = self._epochs[0]['timestamp']

        while self._running and self._current_idx < len(self._epochs):
            epoch = self._epochs[self._current_idx]

            # 计算应该等待的时间
            elapsed_data = epoch['timestamp'] - data_start_ts
            elapsed_real = time.time() - start_time
            sleep_time = (elapsed_data / self.rate_hz) - elapsed_real

            if sleep_time > 0:
                time.sleep(min(sleep_time, 1.0))  # 最多等1秒

            self._current_idx += 1

    def get_next_epoch(self, timeout: float = 1.0) -> Optional[dict]:
        """获取下一个epoch"""
        with self._lock:
            if self._current_idx < len(self._epochs):
                return self._epochs[self._current_idx]
        return None


# ==================== ROS模式（如果ROS可用） ====================

try:
    import rospy
    from sensor_msgs.msg import NavSatFix
    HAS_ROS = True
except ImportError:
    HAS_ROS = False


class ROSDataProvider(GNSSDataProvider):
    """
    ROS模式数据提供者

    订阅gnssfgo相同的ROS topics，实时获取GNSS观测数据。
    需要ROS环境（rospy可用）。
    """

    def __init__(self):
        if not HAS_ROS:
            raise ImportError(
                "rospy not available. Install ROS or use FileDataProvider."
            )

        self._latest_epoch: Optional[dict] = None
        self._lock = threading.Lock()
        self._new_data = threading.Event()

        # Subscribers（在start中初始化）
        self._meas_sub = None
        self._ephem_sub = None
        self._glo_ephem_sub = None
        self._lla_sub = None

        # 缓冲的最新数据
        self._latest_meas = None
        self._latest_ephem = None
        self._latest_glo_ephem = None
        self._latest_lla = None

    def start(self):
        """初始化ROS subscribers"""
        if not HAS_ROS:
            return

        # 注意：rospy.init_node应该在外部调用（通常由run_analyzer.py调用）
        self._meas_sub = rospy.Subscriber(
            "/ublox_driver/range_meas",
            rospy.AnyMsg,  # 使用AnyMsg避免直接依赖gnss_comm消息
            self._meas_callback,
            queue_size=10,
        )
        self._ephem_sub = rospy.Subscriber(
            "/ublox_driver/ephem",
            rospy.AnyMsg,
            self._ephem_callback,
            queue_size=50,
        )
        self._glo_ephem_sub = rospy.Subscriber(
            "/ublox_driver/glo_ephem",
            rospy.AnyMsg,
            self._glo_ephem_callback,
            queue_size=50,
        )
        self._lla_sub = rospy.Subscriber(
            "/ublox_driver/receiver_lla",
            NavSatFix,
            self._lla_callback,
            queue_size=10,
        )

        print("[ROSDataProvider] Subscribed to GNSS topics")

    def stop(self):
        """取消订阅"""
        if self._meas_sub:
            self._meas_sub.unregister()
        if self._ephem_sub:
            self._ephem_sub.unregister()
        if self._glo_ephem_sub:
            self._glo_ephem_sub.unregister()
        if self._lla_sub:
            self._lla_sub.unregister()

    def _meas_callback(self, msg):
        """原始观测数据回调"""
        with self._lock:
            self._latest_meas = msg
            self._try_assemble_epoch()

    def _ephem_callback(self, msg):
        """星历回调"""
        with self._lock:
            self._latest_ephem = msg

    def _glo_ephem_callback(self, msg):
        """GLONASS星历回调"""
        with self._lock:
            self._latest_glo_ephem = msg

    def _lla_callback(self, msg):
        """接收机位置回调"""
        with self._lock:
            self._latest_lla = msg

    def _try_assemble_epoch(self):
        """尝试从最新数据组装epoch（简化版，仅做结构占位）"""
        # 完整的实现需要解析GnssMeasMsg中的观测数据
        # 这里提供接口结构，具体解析需要gnss_comm Python绑定
        self._new_data.set()

    def get_next_epoch(self, timeout: float = 1.0) -> Optional[dict]:
        """等待并获取下一个epoch"""
        if self._new_data.wait(timeout=timeout):
            self._new_data.clear()
            # 简化返回（需要完整实现）
            return {'timestamp': time.time(), 'satellites': []}
        return None


class GNSSFGODataProvider(GNSSDataProvider):
    """
    gnssfgo JSONL 文件数据提供者

    从 gnssfgo 导出的 JSONL 文件读取预计算数据。
    与 FileDataProvider 不同：
    - 输入格式为 JSONL（每行一个 epoch JSON 对象）
    - 包含 gnssfgo 预计算的卫星位置、残差、校正等数据
    - 轮询文件尾部新行以支持流式在线处理
    - 不需要 CSV 解析、星历处理或大气校正

    gnssfgo 以 append-only 方式写入 osqa_input.jsonl，
    OSQA 读取新行并处理每个 epoch。
    """

    def __init__(self, input_path: str, output_path: str):
        """
        Args:
            input_path: gnssfgo → OSQA JSONL 输入文件路径
            output_path: OSQA → gnssfgo JSONL 输出文件路径
        """
        self.input_path = input_path
        self.output_path = output_path
        self._running = False
        self._reader = None  # 延迟导入

    def start(self):
        """启动数据读取器"""
        from gnss_quality_analyzer.gnssfgo_data_reader import GNSSFGODataReader
        self._reader = GNSSFGODataReader(self.input_path, self.output_path)
        self._reader.start()
        self._running = True
        print(f"[GNSSFGODataProvider] Polling {self.input_path} for new epochs...")

    def stop(self):
        """停止数据读取"""
        self._running = False
        if self._reader:
            self._reader.stop()

    def get_next_epoch(self, timeout: float = 1.0) -> Optional[dict]:
        """
        获取下一个 epoch 的预计算数据

        Args:
            timeout: 最大等待时间（秒）

        Returns:
            epoch 数据字典，包含 'timestamp', 'satellites' (list of dict) 等字段
            超时返回 None
        """
        if not self._running or not self._reader:
            return None
        return self._reader.read_next_epoch(timeout)

    def write_quality(self, fused_result) -> bool:
        """
        将质量评分写回 gnssfgo 输出文件

        Args:
            fused_result: FusedResult 对象

        Returns:
            成功返回 True
        """
        if not self._reader:
            return False
        return self._reader.write_quality_result(fused_result)


class QualityPublisher:
    """
    质量分数发布者

    将融合后的质量分数发布出去，供gnssfgo使用。
    """

    def __init__(self, ros_topic: str = "/osqa/signal_quality",
                 file_path: Optional[str] = None):
        """
        Args:
            ros_topic: ROS topic名称（ROS模式）
            file_path: 输出文件路径（文件模式，可选）
        """
        self.ros_topic = ros_topic
        self.file_path = file_path

        # ROS publisher
        self._ros_pub = None
        if HAS_ROS:
            try:
                # 尝试初始化publisher（假设rospy已初始化）
                self._ros_pub = rospy.Publisher(
                    ros_topic, rospy.String, queue_size=10
                )
                print(f"[QualityPublisher] Publishing to {ros_topic}")
            except Exception:
                pass

        # 文件输出
        self._file_handle = None
        if file_path:
            self._file_handle = open(file_path, 'w')
            self._file_handle.write('["timestamp","prn","quality","trust_level","flags"]\n')

        # 统计
        self.publish_count = 0

    def publish(self, fused_result) -> bool:
        """
        发布质量分数

        Args:
            fused_result: FusedResult对象

        Returns:
            是否发布成功
        """
        self.publish_count += 1

        # 转换为JSON
        result_dict = fused_result.to_dict()

        # ROS发布
        if self._ros_pub:
            try:
                json_str = json.dumps(result_dict)
                self._ros_pub.publish(json_str)
            except Exception as e:
                print(f"[QualityPublisher] ROS publish error: {e}")
                return False

        # 文件输出
        if self._file_handle:
            for sat in fused_result.satellites:
                flags_str = ';'.join(sat.all_flags) if sat.all_flags else 'none'
                self._file_handle.write(
                    f'{fused_result.timestamp},{sat.prn},{sat.quality_final:.4f},'
                    f'{sat.trust_level.value},{flags_str}\n'
                )
            self._file_handle.flush()

        return True

    def close(self):
        """关闭资源"""
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None

    def __del__(self):
        self.close()


class GNSSFGOBridge:
    """
    gnssfgo通信桥梁

    封装了OSQA与gnssfgo之间的所有数据交换逻辑。
    根据配置自动选择ROS模式或文件模式。
    """

    def __init__(self, config=None):
        """
        Args:
            config: OSQAConfig对象
        """
        self.config = config

        # 数据提供者（根据环境自动选择）
        self.data_provider: Optional[GNSSDataProvider] = None

        # 质量发布者
        self.quality_publisher: Optional[QualityPublisher] = None

    def setup(self, csv_path: Optional[str] = None,
              ros_mode: bool = True,
              output_path: Optional[str] = None,
              gnssfgo_input_path: Optional[str] = None,
              gnssfgo_output_path: Optional[str] = None):
        """
        设置通信桥梁

        Args:
            csv_path: CSV文件路径（文件模式时需要）
            ros_mode: 是否使用ROS模式
            output_path: 输出文件路径（可选，CSV/ROS模式）
            gnssfgo_input_path: gnssfgo JSONL 输入文件路径（gnssfgo JSONL模式）
            gnssfgo_output_path: gnssfgo JSONL 输出文件路径（gnssfgo JSONL模式）
        """
        # 选择数据提供者
        if gnssfgo_input_path and gnssfgo_output_path:
            self.data_provider = GNSSFGODataProvider(
                gnssfgo_input_path, gnssfgo_output_path
            )
            print(f"[GNSSFGOBridge] Using gnssfgo JSONL mode: {gnssfgo_input_path} -> {gnssfgo_output_path}")
        elif ros_mode and HAS_ROS:
            self.data_provider = ROSDataProvider()
            print("[GNSSFGOBridge] Using ROS mode")
        elif csv_path:
            self.data_provider = FileDataProvider(csv_path)
            print(f"[GNSSFGOBridge] Using file mode: {csv_path}")
        else:
            raise ValueError("One of gnssfgo_input_path, ROS mode, or csv_path must be specified")

        # 创建质量发布者
        ros_topic = "/osqa/signal_quality"
        if self.config:
            ros_topic = self.config.ros_topic_output

        self.quality_publisher = QualityPublisher(
            ros_topic=ros_topic,
            file_path=output_path,
        )

    def start(self):
        """启动数据流"""
        if self.data_provider:
            self.data_provider.start()

    def stop(self):
        """停止"""
        if self.data_provider:
            self.data_provider.stop()
        if self.quality_publisher:
            self.quality_publisher.close()

    def receive_epoch(self, timeout: float = 1.0) -> Optional[dict]:
        """接收一个epoch的数据"""
        if self.data_provider:
            return self.data_provider.get_next_epoch(timeout)
        return None

    def publish_quality(self, fused_result) -> bool:
        """发布质量分数"""
        # gnssfgo JSONL 模式: 使用 data_provider 的 write_quality
        if isinstance(self.data_provider, GNSSFGODataProvider):
            return self.data_provider.write_quality(fused_result)
        if self.quality_publisher:
            return self.quality_publisher.publish(fused_result)
        return False
