# LiDAR Converter

A Python library for automatically converting raw LiDAR sensor data (PCAP format) from various manufacturers into standardized LAS/LAZ formats. The system detects which sensor produced the data, selects the appropriate SDK or library, and performs conversion to ensure compatibility with major geospatial and point cloud tools.

## Features

- 🔍 **Automatic Vendor Detection**: Multi-method detection using UDP ports, packet structure, magic bytes, and companion files
- 🔄 **Multi-Vendor Support**: Supports Ouster, Velodyne, and Livox sensors with unified conversion pipeline
- 📦 **Multiple Output Formats**: Converts to LAS, LAZ, PCD, BIN (KITTI), and CSV formats
- ⚡ **Optimized Processing**: Fast conversion with configurable scan limits and streaming PCAP processing
- 🛠️ **Easy Integration**: Simple Python API and comprehensive CLI for automation
- 🏥 **Health Monitoring**: Built-in health checks and SDK validation
- 📊 **Batch Processing**: Convert multiple files efficiently with progress tracking

## Supported Sensors

- ✅ **Ouster**: OS-0, OS-1, OS-2, OS-Dome series (16/32/64/128 channels)
- ✅ **Velodyne**: VLP-16, VLP-32C, HDL-32E, HDL-64E, VLS-128
- ✅ **Livox**: Avia, Horizon, Tele-15, Mid-40/70 (All formats supported)
- ✅ **Hesai**: PandarXT-32, PandarXT-16, Pandar64, Pandar40P, PandarQT (All formats supported)
- 🚧 **RoboSense**: RS-LiDAR-M1, RS-Ruby, RS-Helios (planned)

## Supported Formats

### Input Formats

| Format | Ouster | Velodyne | Livox | Hesai | Notes |
|--------|--------|----------|-------|-------|-------|
| PCAP   | ✅     | ✅       | ✅    | ✅    | Requires metadata JSON for Ouster |
| CSV    | ❌     | ❌       | ✅    | ❌    | Livox Viewer export format |
| LVX    | ❌     | ❌       | ✅    | ❌    | Livox proprietary format |
| LVX2   | ❌     | ❌       | ✅    | ❌    | Livox proprietary format v2 |

### Output Formats

| Format | Ouster | Velodyne | Livox | Hesai | Description | Best For |
|--------|--------|----------|-------|-------|-------------|----------|
| **LAS** | ✅ | ✅ | ✅ | ✅ | ASPRS LASer format (uncompressed) | GIS, surveying, general use |
| **LAZ** | ✅ | ✅ | ✅ | ✅ | Compressed LAS format | Storage optimization, archival |
| **PCD** | ✅ | ✅ | ✅ | ✅ | Point Cloud Data (PCL format) | Robotics, ROS, PCL tools |
| **BIN** | ✅ | ✅ | ✅ | ✅ | Binary format (KITTI standard) | Machine learning, autonomous driving |
| **CSV** | ✅ | ✅ | ✅ | ✅ | Comma-separated values | Data analysis, spreadsheets, custom processing |

**Legend:**
- ✅ Fully supported
- 🚧 In progress
- ❌ Not supported

### Format Details

#### LAS (ASPRS Standard)
- Industry-standard format for LiDAR data exchange
- Includes point coordinates (X, Y, Z) and intensity
- Compatible with CloudCompare, PDAL, QGIS, ArcGIS
- File size: ~30-40 MB per million points

#### LAZ (Compressed LAS)
- LASzip-compressed LAS format
- 60-80% smaller than LAS files
- Lossless compression
- Automatic compression using laspy 2.0+ or external laszip tool
- File size: ~10-15 MB per million points

#### PCD (Point Cloud Library)
- Native format for Point Cloud Library (PCL)
- ASCII format for maximum compatibility
- Includes point coordinates and intensity
- Ideal for robotics and ROS applications
- File size: ~60-80 MB per million points

#### BIN (KITTI Format)
- Binary format used by KITTI dataset
- Raw float32 values: [x, y, z, intensity] per point
- Little-endian byte order
- Optimized for machine learning pipelines
- File size: ~16 MB per million points (16 bytes/point)

#### CSV (Comma-Separated Values)
- Human-readable text format
- Header row: x, y, z, intensity
- Easy to import into Excel, pandas, MATLAB
- Good for data analysis and visualization
- File size: ~60-80 MB per million points

## Installation

[![PyPI version](https://img.shields.io/pypi/v/lidar-converter.svg)](https://pypi.org/project/lidar-converter/)
[![Python versions](https://img.shields.io/pypi/pyversions/lidar-converter.svg)](https://pypi.org/project/lidar-converter/)

### Prerequisites

- Python 3.8+
- Microsoft Visual C++ Redistributable 2015-2022 (x64) for Windows

### SDK Notes

| Vendor | SDK | Status |
|--------|-----|--------|
| Ouster | `ouster-sdk` (PyPI) | Full SDK — accurate point cloud extraction |
| Velodyne | `velodyne-decoder` (PyPI) | Full SDK — supports VLP-16, VLP-32C, HDL-32E, HDL-64E, VLS-128 |
| Hesai | C++ SDK (Linux only, bundled in `Wrappers/hesai_sdk/`) | dpkt-based parsing on Windows |
| Livox | No Python SDK available | dpkt-based parsing for PCAP; native parser for LVX/LVX2 |


### Install from PyPI (Recommended)

```bash
pip install lidar-converter
```

Verify installation:
```bash
lidar-converter health
```

### Install from Source (Alternative)

1. Clone the repository:
```bash
git clone https://github.com/Param-Patel-o5/lidar-multi-convert.git
cd lidar-multi-convert
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies (editable install recommended for development):
```bash
pip install -r requirements.txt
pip install -e .
```

4. Verify installation:
```bash
lidar-converter health
# or: python -m Lidar_Converter.cli health
```

### SDK vs PCAP parsing

| Vendor | Typical PCAP path | Notes |
|--------|-------------------|--------|
| Ouster | **ouster-sdk** (Python, on PyPI) | Use the JSON metadata file next to the PCAP (same basename) or pass it with `-c`. |
| Velodyne | **dpkt** + Python parsers | No Velodyne Python SDK on PyPI; PCAP decoding is in-repo. Optional VeloView desktop tools are separate. |
| Livox | **dpkt** (PCAP) / CSV or LVX elsewhere | Optional **openpylivox** is not reliably available via `pip` here; PCAP still works with **dpkt**. |
| Hesai | **dpkt** unless native lib built | **`Lidar_Converter/Wrappers/hesai_sdk/`** may contain the vendor C++ SDK sources; the wrapper uses native libs only if built (e.g. `PandarGeneralSDK.dll` under `build/`). Otherwise **dpkt** is used. |

GitHub source repo: **[github.com/Param-Patel-o5/lidar-multi-convert](https://github.com/Param-Patel-o5/lidar-multi-convert)** (PyPI package name remains `lidar-converter`).

Run `lidar-converter --log-level INFO health` once to see how each wrapper validated on your machine.

## Quick Start

### Automatic Conversion (Any Supported Vendor)

```python
from Lidar_Converter.converter import LiDARConverter

# Initialize converter
converter = LiDARConverter()

# Automatic vendor detection and conversion to LAS
result = converter.convert(
    input_path="data.pcap",
    output_path="output.las",
    max_scans=1000  # Optional: limit for faster processing
)

if result["success"]:
    print(f"Converted {result['points_converted']} points from {result['vendor']}")
else:
    print(f"Error: {result['message']}")
```

### Converting to Different Output Formats

**Note**: Output format is automatically detected from the file extension! No need to specify `output_format` parameter.

```python
from Lidar_Converter.converter import LiDARConverter

converter = LiDARConverter()

# Format is auto-detected from file extension
# Convert to LAZ (compressed LAS)
result = converter.convert(
    input_path="data.pcap",
    output_path="output.laz"  # .laz extension → LAZ format
)

# Convert to PCD (Point Cloud Library format)
result = converter.convert(
    input_path="data.pcap",
    output_path="output.pcd",  # .pcd extension → PCD format
    preserve_intensity=True  # Include intensity values
)

# Convert to BIN (KITTI format for ML)
result = converter.convert(
    input_path="data.pcap",
    output_path="output.bin"  # .bin extension → BIN format
)

# Convert to CSV (for data analysis)
result = converter.convert(
    input_path="data.pcap",
    output_path="output.csv",  # .csv extension → CSV format
    preserve_intensity=True
)
```

### Format-Specific Use Cases

```python
# For GIS and surveying (compressed storage)
converter.convert("scan.pcap", "survey_data.laz")  # Auto-detects LAZ format

# For robotics and ROS applications
converter.convert("scan.pcap", "robot_scan.pcd")  # Auto-detects PCD format

# For machine learning training (KITTI format)
converter.convert("scan.pcap", "training_data.bin")  # Auto-detects BIN format

# For data analysis in Python/Excel
converter.convert("scan.pcap", "analysis.csv")  # Auto-detects CSV format
```

### Command Line Usage

```bash
# Check system health
python Lidar_Converter/cli.py health

# Detect vendor automatically
python Lidar_Converter/cli.py detect data.pcap

# Convert - format auto-detected from file extension (no --format flag needed!)
python Lidar_Converter/cli.py convert data.pcap -o output.las --max-scans 1000
python Lidar_Converter/cli.py convert data.pcap -o output.laz --max-scans 1000
python Lidar_Converter/cli.py convert data.pcap -o output.pcd --max-scans 1000
python Lidar_Converter/cli.py convert data.pcap -o output.bin --max-scans 1000
python Lidar_Converter/cli.py convert data.pcap -o output.csv --max-scans 1000

# Or explicitly specify format (optional)
python Lidar_Converter/cli.py convert data.pcap -o output.laz --format laz

# Batch convert multiple files (preserves format from output path)
python Lidar_Converter/cli.py batch ./data_dir -o ./output_dir

# Convert with validation
python Lidar_Converter/cli.py convert data.pcap -o output.las --validate
```

## Project Structure

```
lidar-converter/   # PyPI name; clone: lidar-multi-convert
├── pyproject.toml           # Package metadata and tool config
├── requirements.txt         # Runtime deps (mirrors pyproject)
├── README.md
├── TESTING_GUIDE.md         # Manual CLI checks, SDK notes, Windows/Rich notes
├── Lidar_Converter/         # Main package
│   ├── __init__.py
│   ├── cli.py               # Command line interface
│   ├── converter.py         # Main conversion orchestrator
│   ├── detector.py          # Multi-method vendor detection
│   ├── CLI_README.md        # CLI reference (accurate for current argparse CLI)
│   └── Wrappers/            # Vendor-specific wrappers
│       ├── base_wrapper.py
│       ├── ouster_wrapper.py
│       ├── velodyne_wrapper.py
│       ├── livox_wrapper.py
│       ├── hesai_wrapper.py
│       └── README.md
└── .gitignore
```

## Performance

The converter is optimized for fast processing with configurable scan limits:

### Processing Speed (with --max-scans 5-10)
- **Ouster**: ~0.5s for 517K points
- **Velodyne**: ~0.05s for 866 points  
- **Livox**: ~3.5s for 500K points
- **Hesai**: ~0.04s for 103 points

### Scan Limiting
Use `--max-scans` to process only a portion of the file for faster testing:

```bash
# Process only first 10 scans (fast preview)
python Lidar_Converter/cli.py convert data.pcap -o output.las --max-scans 10

# Process full file (omit --max-scans)
python Lidar_Converter/cli.py convert data.pcap -o output.las
```

**Note**: Livox uses point-based limiting internally (1 scan ≈ 100,000 points) for optimal performance.

## Development

See **`TESTING_GUIDE.md`** for manual `detect` / `convert` command examples (all output formats), SDK notes, and Windows console tips.

Quick checks:

```bash
lidar-converter --output-format json health
lidar-converter --output-format json detect sample.pcap
lidar-converter convert sample.pcap -o out.las --max-scans 100
lidar-converter test sample.pcap
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Ouster SDK](https://github.com/ouster-lidar/ouster-sdk) for LiDAR data processing
- [LASpy](https://github.com/laspy/laspy) for LAS file handling
- [CloudCompare](https://www.cloudcompare.org/) for point cloud visualization

## Vendor Detection Methods

The system uses multiple detection methods with weighted confidence scoring:

- **UDP Port Detection** (35% weight): Analyzes destination ports (Ouster: 7502/7503, Velodyne/Hesai: 2368/2369, Livox: 57000)
- **Packet Structure** (30% weight): Checks magic bytes in UDP payload (Ouster: 0x0001, Velodyne: 0xFFEE, Hesai: 0xEEFF)
- **Magic Bytes** (30% weight): File header signatures
- **Companion Files** (25% weight): Required metadata files (e.g., Ouster JSON)
- **Packet Size** (20% weight): UDP payload size patterns (Velodyne: 1206 bytes, Hesai: 861 bytes)
- **File Extension** (5% weight): File extension hints

Minimum confidence threshold: 14% for positive detection.

## Roadmap

- [x] ~~Add Velodyne sensor support~~
- [x] ~~Add Livox sensor support (Avia, Horizon)~~
- [x] ~~Implement LAZ compression~~
- [x] ~~Add PCD, BIN, and CSV output formats~~
- [x] ~~Add Hesai sensor support (PandarXT, Pandar64, Pandar40P)~~
- [x] ~~Enable LAZ compression for all vendors (lazrs)~~
- [ ] Add RoboSense sensor support (RS-LiDAR-M1, RS-Ruby, RS-Helios)
- [ ] Add binary PCD format support
- [ ] Add E57 and PLY format support
- [ ] Create Docker container
- [ ] Add CI/CD pipeline (optional)
- [ ] Performance optimizations (parallel processing)
- [ ] Web interface for conversion
