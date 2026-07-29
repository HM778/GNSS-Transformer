"""
visualize_output.py — OSQA output file visualizer
==================================================

Reads the JSONL file written by OSQA (OSQA_OUTPUT_PATH) and replays the
per-satellite quality scores in real time.

This file also hosts the Visualizer class, which is reused by run_analyzer.py
via the compatibility wrapper in visualizer.py.

Usage:
    python visualize_output.py --output <path_to_osqa_output.jsonl> [--realtime]

If --realtime is given, epochs are replayed according to their original
timestamps; otherwise the file is played back as fast as possible.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from typing import List, Optional

import numpy as np

# Allow importing from the gnss_quality_analyzer package
# package_root is the parent directory of this file (i.e. GNSS-Transformer)
package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if package_root not in sys.path:
    sys.path.insert(0, package_root)

from gnss_quality_analyzer.quality_fusion import FusedResult, TrustLevel


# Compatibility: register this module under the legacy name
# "gnss_quality_analyzer.visualizer" so that importers such as
#   from gnss_quality_analyzer.visualizer import Visualizer
# still work when visualizer.py has been deleted.
import sys
if __name__ != 'gnss_quality_analyzer.visualizer':
    sys.modules['gnss_quality_analyzer.visualizer'] = sys.modules[__name__]


# =============================================================================
# Visualizer class (merged here so this file is self-contained)
# =============================================================================
class Visualizer:
    """
    OSQA real-time visualizer.

    Supports two layouts:
    - "full"  : 2x2 grid with sky plot, per-analyzer bar chart,
                quality history curves, and statistics panel.
    - "compact": 1x2 grid with sky plot and per-satellite final-confidence bar chart.
    """

    def __init__(
        self,
        history_length: int = 100,
        update_interval_ms: int = 200,
        dpi: int = 100,
        figsize: tuple = (14, 8),
        layout: str = "full",
    ):
        self.history_length = history_length
        self.update_interval_ms = update_interval_ms
        self.layout = layout.lower()
        self._last_update_time = 0.0
        self._available = False

        # History buffers
        self._timestamps: deque[float] = deque(maxlen=history_length)
        self._mean_q_transformer: deque[float] = deque(maxlen=history_length)
        self._mean_q_graph: deque[float] = deque(maxlen=history_length)
        self._mean_q_temporal: deque[float] = deque(maxlen=history_length)
        self._mean_q_final: deque[float] = deque(maxlen=history_length)
        self._n_trusted: deque[int] = deque(maxlen=history_length)
        self._n_suspect: deque[int] = deque(maxlen=history_length)
        self._n_unreliable: deque[int] = deque(maxlen=history_length)

        try:
            import matplotlib

            try:
                matplotlib.use("TkAgg")
            except Exception:
                pass

            import matplotlib.pyplot as plt

            plt.rcParams.update({
                "font.family": "DejaVu Sans",
                "axes.unicode_minus": False,
            })

            self.plt = plt
            self._setup_figure(dpi, figsize)
            self._available = True
        except Exception as e:  # pragma: no cover
            print(f"[OSQA Visualizer] matplotlib initialization failed, visualization disabled: {e}")
            self.plt = None

    # ------------------------------------------------------------------
    # Figure setup
    # ------------------------------------------------------------------
    def _setup_figure(self, dpi: int, figsize: tuple) -> None:
        if self.layout == "compact":
            figsize = (12, 6)
        self.fig = self.plt.figure(figsize=figsize, dpi=dpi)
        self.fig.canvas.manager.set_window_title("OSQA Satellite Quality Monitor")
        self.plt.ion()
        self.plt.show(block=False)

        if self.layout == "compact":
            self.ax_sky = self.fig.add_subplot(1, 2, 1, projection="polar")
            self.ax_bar = self.fig.add_subplot(1, 2, 2)
            self.ax_history = None
            self.ax_stats = None
        else:
            self.ax_sky = self.fig.add_subplot(2, 2, 1, projection="polar")
            self.ax_bar = self.fig.add_subplot(2, 2, 2)
            self.ax_history = self.fig.add_subplot(2, 2, 3)
            self.ax_stats = self.fig.add_subplot(2, 2, 4)

        self._init_sky_plot()
        self._init_bar_plot()
        if self.ax_history is not None:
            self._init_history_plot()
        if self.ax_stats is not None:
            self._init_stats_plot()

        self.fig.tight_layout(pad=2.0)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _init_sky_plot(self) -> None:
        ax = self.ax_sky
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_rlim(0, 90)
        ax.set_rticks([15, 30, 45, 60, 75, 90])
        ax.set_yticklabels(["75", "60", "45", "30", "15", "0"])
        ax.set_title("Sky Plot (color = final quality)", pad=20)
        self._sky_scatter = ax.scatter([], [], c=[], cmap="RdYlGn", vmin=0, vmax=1, s=120)
        self._sky_texts: List = []

    def _init_bar_plot(self) -> None:
        ax = self.ax_bar
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Quality")
        ax.set_title("Per-Analyzer Scores")
        ax.axhline(0.7, color="green", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.axhline(0.3, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
        self._bar_bars: List = []
        self._bar_legend = None

    def _init_history_plot(self) -> None:
        ax = self.ax_history
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean Quality")
        ax.set_title("Quality History")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        (self._line_t,) = ax.plot([], [], "-", color="#3498db", label="Transformer", linewidth=1.2)
        (self._line_g,) = ax.plot([], [], "-", color="#e67e22", label="Graph", linewidth=1.2)
        (self._line_tmp,) = ax.plot([], [], "-", color="#9b59b6", label="Temporal", linewidth=1.2)
        (self._line_f,) = ax.plot([], [], "-", color="#2ecc71", label="Final", linewidth=2.0)
        ax.legend(loc="lower left")

    def _init_stats_plot(self) -> None:
        ax = self.ax_stats
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title("Statistics")
        self._stats_text = ax.text(
            0.05, 0.95, "", transform=ax.transAxes,
            fontsize=11, verticalalignment="top", fontfamily="monospace",
        )

    # ------------------------------------------------------------------
    # Public update interface
    # ------------------------------------------------------------------
    def update(self, result: FusedResult) -> None:
        """Refresh the window with the latest FusedResult."""
        if not self._available or result is None:
            print(f"[OSQA Visualizer] update skipped: available={self._available}, result={result}")
            return

        now = time.time()
        if (now - self._last_update_time) * 1000 < self.update_interval_ms:
            return
        self._last_update_time = now

        print(f"[OSQA Visualizer] updating epoch {result.timestamp:.2f}, "
              f"sats={result.n_total}, mean_q={result.final_mean_quality:.3f}")

        self._append_history(result)
        self._update_sky_plot(result)
        self._update_bar_plot(result)
        if self.ax_history is not None:
            self._update_history_plot()
        if self.ax_stats is not None:
            self._update_stats_plot(result)

        try:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            self.plt.pause(0.001)
        except Exception as e:  # pragma: no cover
            print(f"[OSQA Visualizer] refresh failed: {e}")

    def close(self) -> None:
        """Close the visualization window."""
        if self._available and self.plt is not None:
            try:
                self.plt.close(self.fig)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Internal update logic
    # ------------------------------------------------------------------
    def _append_history(self, result: FusedResult) -> None:
        self._timestamps.append(result.timestamp)
        self._mean_q_transformer.append(result.transformer_mean_quality)
        self._mean_q_graph.append(result.graph_mean_quality)
        self._mean_q_temporal.append(result.temporal_mean_quality)
        self._mean_q_final.append(result.final_mean_quality)
        self._n_trusted.append(result.n_trusted)
        self._n_suspect.append(result.n_suspect)
        self._n_unreliable.append(result.n_unreliable)

    def _update_sky_plot(self, result: FusedResult) -> None:
        ax = self.ax_sky
        sats = result.satellites

        for txt in self._sky_texts:
            txt.remove()
        self._sky_texts.clear()

        if not sats:
            self._sky_scatter.set_offsets(np.empty((0, 2)))
            self._sky_scatter.set_array(np.array([]))
            return

        azimuths = np.array([s.azimuth for s in sats])
        elevations = np.array([s.elevation for s in sats])
        qualities = np.array([s.quality_final for s in sats])
        snrs = np.array([max(s.snr, 1.0) for s in sats])

        theta = np.deg2rad(azimuths % 360.0)
        r = 90.0 - np.clip(elevations, 0.0, 90.0)
        sizes = np.clip(80 + snrs * 3, 60, 300)

        offsets = np.column_stack([theta, r])
        self._sky_scatter.set_offsets(offsets)
        self._sky_scatter.set_array(qualities)
        self._sky_scatter.set_sizes(sizes)

        for s, th, rd in zip(sats, theta, r):
            txt = ax.annotate(
                s.prn,
                xy=(th, rd),
                xytext=(th + 0.05, rd + 4),
                fontsize=7,
                color="black",
            )
            self._sky_texts.append(txt)

    def _update_bar_plot(self, result: FusedResult) -> None:
        ax = self.ax_bar
        sats = result.satellites

        for bar_group in self._bar_bars:
            for rect in bar_group:
                rect.remove()
        self._bar_bars.clear()

        if not sats:
            ax.set_xticks([])
            if self._bar_legend is not None:
                self._bar_legend.remove()
                self._bar_legend = None
            return

        prns = [s.prn for s in sats]
        x = np.arange(len(prns))

        if self.layout == "compact":
            q_f = [s.quality_final for s in sats]
            colors = ["#2ecc71" if s.trust_level == TrustLevel.TRUSTED else
                      "#f1c40f" if s.trust_level == TrustLevel.SUSPECT else
                      "#e74c3c" for s in sats]
            bars_f = ax.bar(x, q_f, 0.5, color=colors, edgecolor="black", linewidth=0.5)
            self._bar_bars = [bars_f]

            ax.set_xticks(x)
            ax.set_xticklabels(prns, rotation=45, ha="right", fontsize=9)
            ax.set_xlim(-0.6, len(prns) - 0.4)
            ax.set_ylabel("Final Confidence")
            ax.set_title("Per-Satellite Final Confidence")

            if self._bar_legend is not None:
                self._bar_legend.remove()
            from matplotlib.patches import Patch
            legend_handles = [
                Patch(facecolor="#2ecc71", edgecolor="black", label="Trusted (>=0.7)"),
                Patch(facecolor="#f1c40f", edgecolor="black", label="Suspect (0.3-0.7)"),
                Patch(facecolor="#e74c3c", edgecolor="black", label="Unreliable (<0.3)"),
            ]
            self._bar_legend = ax.legend(handles=legend_handles, loc="upper right")
        else:
            width = 0.2
            q_t = [s.quality_transformer for s in sats]
            q_g = [s.quality_graph for s in sats]
            q_tmp = [s.quality_temporal for s in sats]
            q_f = [s.quality_final for s in sats]

            bars_t = ax.bar(x - 1.5 * width, q_t, width, label="Transformer", color="#3498db")
            bars_g = ax.bar(x - 0.5 * width, q_g, width, label="Graph", color="#e67e22")
            bars_tmp = ax.bar(x + 0.5 * width, q_tmp, width, label="Temporal", color="#9b59b6")
            bars_f = ax.bar(x + 1.5 * width, q_f, width, label="Final", color="#2ecc71")

            self._bar_bars = [bars_t, bars_g, bars_tmp, bars_f]

            ax.set_xticks(x)
            ax.set_xticklabels(prns, rotation=45, ha="right", fontsize=8)
            ax.set_xlim(-0.6, len(prns) - 0.4)
            ax.set_ylabel("Quality")
            ax.set_title("Per-Analyzer Scores")

            if self._bar_legend is not None:
                self._bar_legend.remove()
            self._bar_legend = ax.legend(handles=[bars_t, bars_g, bars_tmp, bars_f], loc="upper right")

    def _update_history_plot(self) -> None:
        if not self._timestamps:
            return
        x = np.arange(len(self._timestamps))
        self._line_t.set_data(x, list(self._mean_q_transformer))
        self._line_g.set_data(x, list(self._mean_q_graph))
        self._line_tmp.set_data(x, list(self._mean_q_temporal))
        self._line_f.set_data(x, list(self._mean_q_final))
        self.ax_history.set_xlim(0, max(len(x), 10))

    def _update_stats_plot(self, result: FusedResult) -> None:
        lines = [
            f"Epoch: {result.timestamp:.2f}",
            f"Total: {result.n_total}  |  Trusted: {result.n_trusted}  |  Suspect: {result.n_suspect}  |  Unreliable: {result.n_unreliable}",
            "",
            "Mean quality:",
            f"  Transformer: {result.transformer_mean_quality:.3f}",
            f"  Graph:       {result.graph_mean_quality:.3f}",
            f"  Temporal:    {result.temporal_mean_quality:.3f}",
            f"  Final:       {result.final_mean_quality:.3f}",
            "",
            "Top unreliable:",
        ]

        unreliable = [s for s in result.satellites if s.trust_level == TrustLevel.UNRELIABLE]
        unreliable.sort(key=lambda s: s.quality_final)
        for s in unreliable[:5]:
            lines.append(f"  {s.prn}: {s.quality_final:.3f}  flags={','.join(s.all_flags) or 'none'}")

        self._stats_text.set_text("\n".join(lines))


# =============================================================================
# JSONL parsing and replay logic
# =============================================================================
def _parse_trust_level(value: str) -> TrustLevel:
    """Map string trust level to enum."""
    try:
        return TrustLevel(value.lower())
    except ValueError:
        return TrustLevel.SUSPECT


def _result_from_json(data: dict) -> FusedResult:
    """Reconstruct a FusedResult from the JSONL line written by OSQA."""
    from gnss_quality_analyzer.quality_fusion import SatelliteQuality

    satellites = []
    sat_dict = data.get("satellites", {})

    for sat_id, sat_info in sat_dict.items():
        details = sat_info.get("details", {})
        satellites.append(
            SatelliteQuality(
                prn=sat_info.get("prn", str(sat_id)),
                system=sat_info.get("system", "?"),
                quality_transformer=float(details.get("transformer", sat_info.get("quality", 1.0))),
                quality_graph=float(details.get("graph", sat_info.get("quality", 1.0))),
                quality_temporal=float(details.get("temporal", sat_info.get("quality", 1.0))),
                quality_final=float(sat_info.get("quality", 1.0)),
                trust_level=_parse_trust_level(sat_info.get("trust_level", "trusted")),
                all_flags=sat_info.get("flags", []),
                snr=float(sat_info.get("snr", 0.0)),
                elevation=float(sat_info.get("elevation", 0.0)),
                azimuth=float(sat_info.get("azimuth", 0.0)),
            )
        )

    mean_quality = data.get("mean_quality", 1.0)
    if isinstance(mean_quality, dict):
        transformer_mean = float(mean_quality.get("transformer", 1.0))
        graph_mean = float(mean_quality.get("graph", 1.0))
        temporal_mean = float(mean_quality.get("temporal", 1.0))
        final_mean = float(mean_quality.get("final", 1.0))
    else:
        final_mean = float(mean_quality)
        transformer_mean = float(np.mean([s.quality_transformer for s in satellites])) if satellites else final_mean
        graph_mean = float(np.mean([s.quality_graph for s in satellites])) if satellites else final_mean
        temporal_mean = float(np.mean([s.quality_temporal for s in satellites])) if satellites else final_mean

    return FusedResult(
        timestamp=float(data.get("timestamp", 0.0)),
        satellites=satellites,
        n_total=int(data.get("n_total", len(satellites))),
        n_trusted=int(data.get("n_trusted", 0)),
        n_suspect=int(data.get("n_suspect", 0)),
        n_unreliable=int(data.get("n_unreliable", 0)),
        transformer_mean_quality=transformer_mean,
        graph_mean_quality=graph_mean,
        temporal_mean_quality=temporal_mean,
        final_mean_quality=final_mean,
    )


def _read_existing_lines(path: str) -> list:
    """Read all complete JSON lines currently in the file."""
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Visualize OSQA quality output JSONL file."
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the OSQA output JSONL file (e.g. osqa_output.jsonl).",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Replay epochs according to their timestamps; otherwise play as fast as possible.",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=1.0,
        help="Replay speed multiplier when --realtime is used (default: 1.0).",
    )
    parser.add_argument(
        "--update-interval",
        type=int,
        default=200,
        help="Visualizer refresh interval in milliseconds (default: 200).",
    )
    args = parser.parse_args()

    output_path = args.output
    print(f"[OSQA Visualize Output] Reading {output_path}")

    visualizer = Visualizer(update_interval_ms=args.update_interval, layout="compact")

    read_line_count = 0
    last_timestamp: Optional[float] = None
    playback_start = time.time()

    try:
        while True:
            lines = _read_existing_lines(output_path)
            new_lines = lines[read_line_count:]

            if not new_lines:
                time.sleep(0.1)
                continue

            for line in new_lines:
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[OSQA Visualize Output] JSON parse error: {e}")
                    continue

                result = _result_from_json(data)

                if args.realtime and last_timestamp is not None:
                    delta_data = result.timestamp - last_timestamp
                    delta_real = (time.time() - playback_start) * args.rate
                    sleep_time = delta_data / args.rate - delta_real
                    if sleep_time > 0:
                        time.sleep(min(sleep_time, 1.0))

                visualizer.update(result)
                print(
                    f"[OSQA Visualize Output] epoch {result.timestamp:.2f}, "
                    f"sats={result.n_total}, mean_q={result.final_mean_quality:.3f}"
                )

                last_timestamp = result.timestamp

            read_line_count = len(lines)

    except KeyboardInterrupt:
        print("\n[OSQA Visualize Output] Stopped by user.")
    finally:
        visualizer.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
