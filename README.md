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
- ✅ **Livox**: Avia, Horizon, Tele-15, Mid-40/70 (LAS output only, other formats in progress)
- 🚧 **Hesai**: PandarXT, Pandar64, Pandar40P (planned)
- 🚧 **RIEGL**: VUX series, miniVUX (planned)

## Supported Formats

### Input Formats

| Format | Ouster | Velodyne | Livox | Notes |
|--------|--------|----------|-------|-------|
| PCAP   | ✅     | ✅       | ✅    | Requires metadata JSON for Ouster |
| CSV    | ❌     | ❌       | ✅    | Livox Viewer export format |
| LVX    | ❌     | ❌       | ✅    | Livox proprietary format |
| LVX2   | ❌     | ❌       | ✅    | Livox proprietary format v2 |

### Output Formats

| Format | Ouster | Velodyne | Livox | Description | Best For |
|--------|--------|----------|-------|-------------|----------|
| **LAS** | ✅ | ✅ | ✅ | ASPRS LASer format (uncompressed) | GIS, surveying, general use |
| **LAZ** | ✅ | ✅ | 🚧 | Compressed LAS format | Storage optimization, archival |
| **PCD** | ✅ | ✅ | 🚧 | Point Cloud Data (PCL format) | Robotics, ROS, PCL tools |
| **BIN** | ✅ | ✅ | 🚧 | Binary format (KITTI standard) | Machine learning, autonomous driving |
| **CSV** | ✅ | ✅ | 🚧 | Comma-separated values | Data analysis, spreadsheets, custom processing |

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

### Prerequisites

- Python 3.8+
- Microsoft Visual C++ Redistributable 2015-2022 (x64) for Windows

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Param-Patel-o5/lidar-converter.git
cd lidar-converter
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Verify installation:
```bash
python Lidar_Converter/cli.py health
```

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

```python
from Lidar_Converter.converter import LiDARConverter

converter = LiDARConverter()

# Convert to LAZ (compressed LAS)
result = converter.convert(
    input_path="data.pcap",
    output_path="output.laz",
    output_format="laz"
)

# Convert to PCD (Point Cloud Library format)
result = converter.convert(
    input_path="data.pcap",
    output_path="output.pcd",
    output_format="pcd",
    preserve_intensity=True  # Include intensity values
)

# Convert to BIN (KITTI format for ML)
result = converter.convert(
    input_path="data.pcap",
    output_path="output.bin",
    output_format="bin"
)

# Convert to CSV (for data analysis)
result = converter.convert(
    input_path="data.pcap",
    output_path="output.csv",
    output_format="csv",
    preserve_intensity=True
)
```

### Format-Specific Use Cases

```python
# For GIS and surveying (compressed storage)
converter.convert("scan.pcap", "survey_data.laz", output_format="laz")

# For robotics and ROS applications
converter.convert("scan.pcap", "robot_scan.pcd", output_format="pcd")

# For machine learning training (KITTI format)
converter.convert("scan.pcap", "training_data.bin", output_format="bin")

# For data analysis in Python/Excel
converter.convert("scan.pcap", "analysis.csv", output_format="csv")
```

### Command Line Usage

```bash
# Check system health
python Lidar_Converter/cli.py health

# Detect vendor automatically
python Lidar_Converter/cli.py detect data.pcap

# Convert to LAS (default format)
python Lidar_Converter/cli.py convert data.pcap -o output.las --max-scans 1000

# Convert to different formats
python Lidar_Converter/cli.py convert data.pcap -o output.laz --format laz
python Lidar_Converter/cli.py convert data.pcap -o output.pcd --format pcd
python Lidar_Converter/cli.py convert data.pcap -o output.bin --format bin
python Lidar_Converter/cli.py convert data.pcap -o output.csv --format csv

# Batch convert multiple files (preserves format from output path)
python Lidar_Converter/cli.py batch ./data_dir -o ./output_dir

# Convert with validation
python Lidar_Converter/cli.py convert data.pcap -o output.las --validate
```

## Project Structure

```
lidar-converter/
├── Lidar_Converter/          # Main package
│   ├── __init__.py
│   ├── cli.py               # Command line interface
│   ├── converter.py         # Main conversion orchestrator
│   ├── detector.py          # Multi-method vendor detection
│   ├── utils.py             # Utility functions
│   ├── Wrappers/            # Vendor-specific wrappers
│   │   ├── __init__.py
│   │   ├── base_wrapper.py  # Abstract base class
│   │   ├── ouster_wrapper.py    # Ouster SDK wrapper
│   │   ├── velodyne_wrapper.py  # Velodyne wrapper (dpkt-based)
│   │   └── README.md        # Wrapper documentation
│   ├── CLI_README.md        # CLI usage guide
│   ├── TESTING_GUIDE.md     # Testing instructions
│   └── pyproject.toml       # Package configuration
├── requirements.txt         # Python dependencies
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

## Development

### Running Tests

```bash
python -m pytest tests/
```

### Testing

See `Lidar_Converter/TESTING_GUIDE.md` for comprehensive testing instructions.

Quick test commands:

```bash
# Test system health
python Lidar_Converter/cli.py health

# Test vendor detection
python Lidar_Converter/cli.py detect sample.pcap

# Test conversion with limited scans (fast)
python Lidar_Converter/cli.py convert sample.pcap -o test.las --max-scans 100

# Run full pipeline test
python Lidar_Converter/cli.py test sample.pcap
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

- **UDP Port Detection** (35% weight): Analyzes destination ports (Ouster: 7502/7503, Velodyne: 2368/2369)
- **Packet Structure** (30% weight): Checks magic bytes in UDP payload (Ouster: 0x0001, Velodyne: 0xFFEE)
- **Magic Bytes** (30% weight): File header signatures
- **Companion Files** (25% weight): Required metadata files (e.g., Ouster JSON)
- **Packet Size** (20% weight): UDP payload size patterns
- **File Extension** (5% weight): File extension hints

Minimum confidence threshold: 14% for positive detection.

## Roadmap

- [x] ~~Add Velodyne sensor support~~
- [x] ~~Add Livox sensor support (Avia, Horizon)~~
- [x] ~~Implement LAZ compression~~
- [x] ~~Add PCD, BIN, and CSV output formats~~
- [ ] Complete Livox multi-format support (PCD, BIN, CSV, LAZ)
- [ ] Add Hesai sensor support (PandarXT, Pandar64)
- [ ] Add RIEGL sensor support (VUX series)
- [ ] Add binary PCD format support
- [ ] Add E57 and PLY format support
- [ ] Create Docker container
- [ ] Add CI/CD pipeline
- [ ] Performance optimizations (parallel processing)
- [ ] Web interface for conversion
