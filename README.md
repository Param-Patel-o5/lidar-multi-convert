# LiDAR Converter

A Python library for automatically converting raw LiDAR sensor data (PCAP format) from various manufacturers into standardized LAS/LAZ formats. The system detects which sensor produced the data, selects the appropriate SDK or library, and performs conversion to ensure compatibility with major geospatial and point cloud tools.

## Features

- 🔍 **Automatic Sensor Detection**: Identifies LiDAR sensor manufacturer from PCAP data
- 🔄 **Multi-Manufacturer Support**: Currently supports Ouster sensors, with plans for Velodyne and others
- 📦 **Standardized Output**: Converts to LAS/LAZ formats compatible with CloudCompare, PDAL, and other tools
- ⚡ **Optimized Processing**: Fast conversion with configurable scan limits
- 🛠️ **Easy Integration**: Simple Python API for integration into larger projects

## Supported Sensors

- ✅ **Ouster**: OS-0, OS-1, OS-2 series sensors
- 🚧 **Velodyne**: VLP-16, VLP-32, HDL-32E, HDL-64E (planned)
- 🚧 **Livox**: Avia, Horizon, Tele-15 (planned)

## Installation

### Prerequisites

- Python 3.8+
- Microsoft Visual C++ Redistributable 2015-2022 (x64) for Windows

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/lidar-converter.git
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

## Quick Start

### Convert Ouster PCAP to LAS

```python
from lidar_converter import LidarConverter

# Initialize converter
converter = LidarConverter()

# Convert PCAP file
converter.convert_pcap_to_las(
    pcap_path="data/sample.pcap",
    json_path="data/sample.json",
    output_path="output.las",
    max_scans=1000
)
```

### Command Line Usage

```bash
# Convert with default settings
python -m lidar_converter.cli convert data/sample.pcap data/sample.json

# Convert with custom parameters
python -m lidar_converter.cli convert data/sample.pcap data/sample.json --output result.las --max-scans 500
```

## Project Structure

```
lidar_converter/
├── lidar_converter/          # Main package
│   ├── __init__.py
│   ├── cli.py               # Command line interface
│   ├── converters.py        # Core conversion logic
│   ├── detector.py          # Sensor detection
│   ├── utils.py             # Utility functions
│   └── wrappers/            # Manufacturer-specific wrappers
│       ├── ouster.py        # Ouster SDK wrapper
│       └── velodyne.py      # Velodyne SDK wrapper (planned)
├── tests/                   # Test files
├── examples/                # Example scripts
├── requirements.txt         # Python dependencies
├── pyproject.toml          # Project configuration
└── README.md               # This file
```

## Development

### Running Tests

```bash
python -m pytest tests/
```

### SDK Testing

The `SDK _testing/` directory contains scripts for testing individual manufacturer SDKs:

```bash
# Test Ouster SDK
cd "SDK _testing"
python for_ouster.py
python pcap_to_las.py --max-scans 100 --output test.las
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

## Roadmap

- [ ] Add Velodyne sensor support
- [ ] Add Livox sensor support
- [ ] Implement LAZ compression
- [ ] Add batch processing capabilities
- [ ] Create Docker container
- [ ] Add CI/CD pipeline
- [ ] Improve error handling and logging
- [ ] Add comprehensive documentation
