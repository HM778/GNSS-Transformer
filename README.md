# 🛰️ GNSS-Transformer: Transformer Models for GNSS Positioning

## 📋 Project Overview

GNSS-Transformer is a new workspace for building Transformer-based deep learning models for GNSS positioning. It features a **C++/Python co-project** architecture:

- **C++ (ROS catkin package)**: High-performance GNSS data collection via `ublox_driver` + `gnss_comm`. Extracts ML-ready features (SNR, Azimuth, Elevation, Pseudorange Residuals) in real-time and exports to CSV.
- **Python**: Data loading, Transformer model training, evaluation, and inference using PyTorch.

### Architecture

```
ublox_driver (ROS) → gnss_comm topics → GNSSDataCollector (C++) → CSV
                                                                    ↓
                                               PyTorch Dataset → Transformer Model → Position Corrections
```

### Key Features
- **ROS-Integrated C++ Collector**: Subscribes to `/ublox_driver/range_meas`, `/ublox_driver/ephem`, `/ublox_driver/receiver_lla` topics
- **gnss_comm Powered**: Uses `gnss_comm::psr_pos` SPP solver, `gnss_comm::eph2pos` satellite positions, `gnss_comm::sat_azel` azimuth/elevation
- **CSV Pipeline**: C++ saves `[timestamp, prn, snr, az, el, residual, spp_xyz, gt_lat/lon/h]` CSV consumed by Python
- **Transformer Model**: Custom PyTorch Transformer for sequence-based GNSS residual correction
- **pyubx2 Support**: Python-side UBX file parsing for offline dataset construction

## 📁 Directory Structure

```
GNSS-Transformer/
├── README.md
├── requirements.txt              # Python dependencies (torch, pyubx2, etc.)
├── cpp/                          # C++ ROS catkin package
│   ├── package.xml
│   ├── CMakeLists.txt            # ROS catkin build, depends on gnss_comm
│   ├── include/gnss_transformer/
│   │   ├── gnss_types.hpp        # Core types (GpsTime, SatelliteFeature, EpochData)
│   │   ├── gnss_parser.hpp       # ROS data collector (ublox_driver subscriber)
│   │   └── gnss_engine.hpp       # SPP + geometry engine (wraps gnss_comm)
│   ├── src/
│   │   ├── main.cpp              # ROS node entry point
│   │   ├── gnss_parser.cpp       # Topic callbacks, feature extraction
│   │   └── gnss_engine.cpp       # SPP solver, coord transforms, CSV export
│   └── launch/
│       └── collect_data.launch   # ROS launch file for data collection
├── python/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── gnss_dataset.py       # PyTorch Dataset from CSV
│   │   └── gnss_parser.py        # pyubx2-based UBX/RINEX parser
│   ├── models/
│   │   ├── __init__.py
│   │   ├── transformer.py        # Transformer model architecture
│   │   └── layers.py             # Custom layers
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py            # Training loop
│   │   ├── losses.py             # Loss functions
│   │   ├── train.py              # Training entry point
│   │   ├── evaluate.py           # Evaluation script
│   │   └── inference.py          # Inference script
│   └── utils/
│       ├── __init__.py
│       ├── coord.py              # Coordinate transformations
│       └── metrics.py            # Evaluation metrics
├── config/
│   └── train_config.json         # Training hyperparameters
├── scripts/
│   ├── prepare_data.sh           # ROS → CSV data collection wrapper
│   ├── run_training.sh           # Python training script
│   └── run_inference.sh          # Python inference script
└── results/
    └── .gitkeep
```

## 🚀 Getting Started

### Prerequisites

- **ROS** (Melodic/Noetic) with catkin workspace
- **gnss_comm** package (ROS catkin package for GNSS data structures)
- **ublox_driver** (ROS driver for u-blox GNSS receivers) publishing:
  - `/ublox_driver/range_meas` (gnss_comm::GnssMeasMsg)
  - `/ublox_driver/ephem` (gnss_comm::GnssEphemMsg)
  - `/ublox_driver/receiver_lla` (sensor_msgs::NavSatFix)
- Python 3.8+ with PyTorch

### C++ ROS Node — Data Collection

1. Symlink or copy `cpp/` into your ROS catkin workspace `src/` directory:
```bash
ln -s /path/to/GNSS-Transformer/cpp ~/catkin_ws/src/gnss_transformer
```

2. Build with catkin:
```bash
cd ~/catkin_ws
catkin_make
# or: catkin build gnss_transformer
```

3. Run data collection (with ublox_driver running):
```bash
roslaunch gnss_transformer collect_data.launch \
    output_path:=/tmp/training_data.csv \
    duration:=60.0
```

4. The node subscribes to ublox_driver topics, extracts:
   - **Per-satellite**: PRN, SNR, Azimuth, Elevation, Pseudorange, Doppler, Pseudorange Residual
   - **Per-epoch**: SPP position (ECEF), ground truth (LLA)
   - Outputs CSV consumed by the Python training pipeline

### Python Environment Setup

```bash
# Create conda environment
conda create --name gnss-transformer python=3.10
conda activate gnss-transformer

# Install dependencies
pip install -r requirements.txt
```

### Offline Data Parsing (Python + pyubx2)

```bash
# Parse UBX binary files for offline dataset construction
python -c "
from python.data.gnss_parser import parse_ubx_to_csv
parse_ubx_to_csv('path/to/data.ubx', 'output.csv')
"

# Or parse RINEX observation files
python -c "
from python.data.gnss_parser import parse_rinex_obs
parse_rinex_obs('path/to/obs.rnx', 'path/to/nav.rnx', 'output.csv')
"
```

### Training

```bash
./scripts/run_training.sh
```

Or manually:
```bash
python -m python.training.train \
    --config config/train_config.json \
    --data /tmp/training_data.csv \
    --output ./results
```

## 📊 Data Format

### CSV Columns (C++ output → Python input)

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | float | GPS time of week (seconds) |
| `week` | int | GPS week number |
| `tow` | float | Time of week (seconds) |
| `prn` | int | Satellite PRN number |
| `sys` | int | GNSS system ID |
| `snr` | float | Signal-to-Noise Ratio (dB-Hz) |
| `azimuth` | float | Azimuth angle (radians) |
| `elevation` | float | Elevation angle (radians) |
| `pseudorange` | float | Pseudorange (meters) |
| `doppler` | float | Doppler shift (Hz) |
| `psr_residual` | float | Pseudorange residual (meters) — **ML target** |
| `spp_x/y/z` | float | SPP position in ECEF (meters) |
| `gt_lat/lon/h` | float | Ground truth position (deg, m) |

## 🧠 Model Architecture

The Transformer model processes satellite observations as variable-length sets:

1. **Input**: `[SNR, Azimuth, Elevation]` per satellite (3 features)
2. **Target**: Pseudorange residual per satellite
3. **Output**: Corrected position (ECEF) or per-satellite residual predictions

Key components:
- Multi-head self-attention for inter-satellite correlation
- Learned positional encoding
- Set-to-sequence architecture (handles variable satellite counts)
- Regression head for residual prediction

## 📝 License

MIT