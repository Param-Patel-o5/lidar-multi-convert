# LiDAR Converter

A Python library for automatically converting raw LiDAR sensor data (PCAP format) from various manufacturers into standardized LAS/LAZ formats. The system detects which sensor produced the data, selects the appropriate SDK or library, and performs conversion to ensure compatibility with major geospatial and point cloud tools.

## Features

- 🔍 **Automatic Vendor Detection**: Multi-method detection using UDP ports, packet structure, magic bytes, and companion files
- 🔄 **Multi-Vendor Support**: Supports Ouster and Velodyne sensors with unified conversion pipeline
- 📦 **Standardized Output**: Converts to LAS/LAZ formats compatible with CloudCompare, PDAL, and other tools
- ⚡ **Optimized Processing**: Fast conversion with configurable scan limits and streaming PCAP processing
- 🛠️ **Easy Integration**: Simple Python API and comprehensive CLI for automation
- 🏥 **Health Monitoring**: Built-in health checks and SDK validation
- 📊 **Batch Processing**: Convert multiple files efficiently with progress tracking

## Supported Sensors

- ✅ **Ouster**: OS-0, OS-1, OS-2, OS-Dome series (16/32/64/128 channels)
- ✅ **Velodyne**: VLP-16, VLP-32C, HDL-32E, HDL-64E, VLS-128
- 🚧 **Hesai**: PandarXT, Pandar64, Pandar40P (planned)
- 🚧 **Livox**: Avia, Horizon, Tele-15 (planned)
- 🚧 **RIEGL**: VUX series, miniVUX (planned)

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

# Automatic vendor detection and conversion
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

### Command Line Usage

```bash
# Check system health
python Lidar_Converter/cli.py health

# Detect vendor automatically
python Lidar_Converter/cli.py detect data.pcap

# Convert with automatic vendor detection
python Lidar_Converter/cli.py convert data.pcap -o output.las --max-scans 1000

# Batch convert multiple files
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
- [ ] Add Hesai sensor support (PandarXT, Pandar64)
- [ ] Add Livox sensor support (Avia, Horizon)
- [ ] Add RIEGL sensor support (VUX series)
- [ ] Implement LAZ compression
- [ ] Add PCD and other output formats
- [ ] Create Docker container
- [ ] Add CI/CD pipeline
- [ ] Performance optimizations
- [ ] Web interface for conversion
