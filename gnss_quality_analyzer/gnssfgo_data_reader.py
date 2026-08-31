"""
gnssfgo_data_reader.py — gnssfgo JSONL 数据读写器
=================================================

从 gnssfgo 导出的 JSONL 格式文件读取已预处理的 GNSS 观测数据，
跳过所有 gnssfgo 已完成的计算（星历匹配、卫星位置、大气校正等）。

同时提供将 OSQA 质量评分写回 JSONL 格式文件的功能，供 gnssfgo 读取。

=== 使用示例 ===

    reader = GNSSFGODataReader("osqa_input.jsonl", "osqa_output.jsonl")
    reader.start()

    while True:
        epoch = reader.read_next_epoch(timeout=1.0)
        if epoch is None:
            break
        # epoch 包含可直接用于特征提取的预计算数据
        process(epoch)

    reader.write_quality_result(fused_result)
    reader.stop()

=== JSONL 格式说明 ===

每行一个 JSON 对象，代表一个 epoch。
gnssfgo 以 append-only 方式写入新行到 osqa_input.jsonl，
OSQA 从尾部读取新行。

"""

import os
import json
import time
import threading
from typing import Dict, List, Optional


class GNSSFGODataReader:
    """
    从 gnssfgo 导出的 JSONL 文件读取预计算数据

    工作方式:
    - 轮询 osqa_input.jsonl 文件的新行
    - 每行是一个完整的 epoch JSON 对象
    - 跟踪已读行数，只返回新行
    - 所有值由 gnssfgo 预计算，不需要额外的星历/大气校正处理
    """

    def __init__(self, input_path: str, output_path: str):
        """
        Args:
            input_path: gnssfgo → OSQA 输入 JSONL 文件路径
            output_path: OSQA → gnssfgo 输出 JSONL 文件路径
        """
        self.input_path = input_path
        self.output_path = output_path
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._read_line_count = 0
        self._pending_epochs: List[dict] = []
        self._last_poll_time = 0.0

    def start(self):
        """启动轮询线程"""
        self._running = True
        # 确保输出文件存在（清空旧内容，gnssfgo 读取最后一行）
        if os.path.exists(self.output_path):
            os.remove(self.output_path)

        # 重置输入文件读取位置（从头开始读，跳过已处理的行）
        self._read_line_count = 0

        print(f"[GNSSFGODataReader] Started. Input: {self.input_path}, Output: {self.output_path}")

    def stop(self):
        """停止轮询"""
        self._running = False

    def read_next_epoch(self, timeout: float = 1.0) -> Optional[dict]:
        """
        读取下一个可用的 epoch 数据

        阻塞等待最多 timeout 秒。返回 None 表示超时无新数据。

        Returns:
            epoch 字典，包含 'timestamp', 'time_frame', 'receiver_ecef',
            'receiver_enu', 'satellites' (list of dict) 等字段
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if self._pending_epochs:
                    return self._pending_epochs.pop(0)

            # 轮询文件新行
            self._poll_new_lines()
            time.sleep(0.05)  # 50ms 轮询间隔

        return None

    def _poll_new_lines(self):
        """读取文件尾部新增的行"""
        if not os.path.exists(self.input_path):
            return

        try:
            with open(self.input_path, 'r') as f:
                lines = f.readlines()

            # 只处理新行
            new_lines = lines[self._read_line_count:]
            if not new_lines:
                return

            for line in new_lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    epoch_data = json.loads(line)
                    with self._lock:
                        self._pending_epochs.append(epoch_data)
                except json.JSONDecodeError as e:
                    print(f"[GNSSFGODataReader] JSON parse error: {e}")

            self._read_line_count = len(lines)
            self._last_poll_time = time.time()
        except Exception as e:
            print(f"[GNSSFGODataReader] File read error: {e}")

    def write_quality_result(self, fused_result) -> bool:
        """
        将 OSQA 质量评分写回 gnssfgo 可读取的 JSONL 格式

        Args:
            fused_result: FusedResult 对象（来自 quality_fusion.py）

        Returns:
            写入成功返回 True
        """
        # 构建 gnssfgo 可读的 JSON 输出格式
        # time_frame 必须写出: gnssfgo 端用 readLatestQualityScores(path, time_frame)
        # 按历元匹配评分, 缺失该字段会导致匹配永远失败、评分被静默丢弃
        # (time_frame = gps 时间 × 10, 与 gnssfgo 内部时间帧约定一致)
        output = {
            "timestamp": fused_result.timestamp,
            "time_frame": round(fused_result.timestamp * 10.0, 3),
            "n_total": fused_result.n_total,
            "n_trusted": fused_result.n_trusted,
            "n_suspect": fused_result.n_suspect,
            "n_unreliable": fused_result.n_unreliable,
            "mean_quality": round(fused_result.final_mean_quality, 4),
            "satellites": {}
        }

        for sat in fused_result.satellites:
            # 优先使用 gnssfgo 导出并直通的正确 sat_id;
            # 仅在无 sat_id 的场景 (CSV 回放等) 回退到 PRN 反推
            sat_id = int(getattr(sat, "sat_id", 0) or 0)
            if sat_id <= 0:
                sat_id = self._prn_to_sat_id(sat.prn)
            output["satellites"][str(sat_id)] = {
                "prn": sat.prn,
                "system": sat.system,
                "sat_id": sat_id,
                "quality": round(sat.quality_final, 4),
                "trust_level": sat.trust_level.value,
                "details": {
                    "transformer": round(sat.quality_transformer, 4),
                    "graph": round(sat.quality_graph, 4),
                    "temporal": round(sat.quality_temporal, 4),
                },
                "flags": sat.all_flags if sat.all_flags else [],
                "snr": round(sat.snr, 1),
                "elevation": round(sat.elevation, 1),
                "azimuth": round(sat.azimuth, 1),
            }

        try:
            with open(self.output_path, 'a') as f:
                f.write(json.dumps(output) + '\n')
            return True
        except Exception as e:
            print(f"[GNSSFGODataReader] Write error: {e}")
            return False

    @staticmethod
    def _prn_to_sat_id(prn: str) -> int:
        """
        将 PRN 字符串转换为 gnss_comm 的 sat_id 整数 (回退路径)

        gnssfgo JSONL 模式下卫星的 sat_id 已随输入直通 (RawObservation.sat_id),
        本方法仅用于无 sat_id 的 CSV 回放等场景。

        与 gnss_comm 的 sat_no() 对齐 (src/gnss_comm/src/gnss_utility.cpp),
        前提是 PRN 字符串为系统内真实 PRN (gnssfgo 导出侧已修复, 不再使用 sat % 100):
          GPS     1-32
          GLONASS 33-59   (32 + prn)
          Galileo 60-97   (59 + prn)
          BeiDou  98-160  (97 + prn)
        例如: 'G05' → 5, 'R06' → 38, 'E07' → 66, 'C22' → 119。
        """
        if not prn or len(prn) < 2:
            return 0
        sys_char = prn[0].upper()
        try:
            num = int(prn[1:])
        except ValueError:
            return 0
        if sys_char == 'G':
            return num       # GPS: 1-32
        elif sys_char == 'R':
            return 32 + num  # GLONASS: 33-59
        elif sys_char == 'E':
            return 59 + num  # Galileo: 60-97
        elif sys_char == 'C':
            return 97 + num  # BeiDou: 98-160
        return 0
