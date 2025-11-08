# Changelog

All notable changes to the LiDAR Converter project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Multiple Output Format Support**: Comprehensive format conversion capabilities
  - **PCD Format**: Point Cloud Library (PCL) format support for Ouster and Velodyne
    - ASCII format for maximum compatibility
    - Includes X, Y, Z coordinates and intensity values
    - Ideal for robotics and ROS applications
  - **BIN Format**: KITTI binary format support for Ouster and Velodyne
    - Raw float32 binary format (little-endian)
    - 16 bytes per point (x, y, z, intensity)
    - Optimized for machine learning pipelines
  - **CSV Format**: Comma-separated values format for Ouster and Velodyne
    - Human-readable text format with headers
    - Easy import into Excel, pandas, MATLAB
    - Configurable intensity column inclusion
  - **LAZ Compression**: Compressed LAS format support for Ouster and Velodyne
    - Automatic compression using laspy 2.0+ built-in support
    - Fallback to external laszip command-line tool
    - 60-80% file size reduction with lossless compression

- **Livox Wrapper Implementation**: Complete support for Livox LiDAR sensors
  - Avia, Horizon, Tele-15, Mid-40/70 sensor models
  - Multiple input format support (PCAP, CSV, LVX, LVX2)
  - LAS output format fully implemented
  - Multi-format output support in progress (LAZ, PCD, BIN, CSV)

- **Refactored Conversion Pipeline**: Modular architecture for format flexibility
  - Extracted point cloud extraction from format-specific writing
  - Shared format conversion methods in BaseVendorWrapper
  - Consistent error handling across all formats
  - Format-agnostic intermediate representation (numpy arrays)

### Changed
- **BaseVendorWrapper**: Added shared helper methods for format conversion
  - `_points_to_pcd()`: Convert numpy array to PCD format
  - `_points_to_bin()`: Convert numpy array to BIN format
  - `_points_to_csv()`: Convert numpy array to CSV format
  - `_compress_las_to_laz()`: Compress LAS to LAZ with multiple fallback methods

- **OusterWrapper**: Refactored for multi-format support
  - Separated point extraction from LAS writing
  - Added routing logic for all output formats
  - Maintains backward compatibility with existing LAS conversion

- **VelodyneWrapper**: Refactored for multi-format support
  - Separated point extraction from LAS writing
  - Added routing logic for all output formats
  - Maintains backward compatibility with existing LAS conversion

### Fixed
- LAZ format now properly compresses files instead of returning uncompressed LAS
- Improved error messages for missing dependencies (laspy, laszip)
- Enhanced file I/O error handling with specific error reporting

## [0.2.0] - 2024-11-05

### Added
- **Velodyne Wrapper Implementation**: Complete support for Velodyne LiDAR sensors
  - VLP-16, VLP-32C, HDL-32E, HDL-64E, VLS-128 sensor models
  - PCAP parsing using dpkt library (no vendor SDK required)
  - Velodyne packet structure parsing (1206 bytes, 0xFFEE magic bytes)
  - Polar to Cartesian coordinate conversion
  - LAS file output with intensity preservation

- **Multi-Vendor Architecture**: Unified wrapper system
  - `BaseVendorWrapper` abstract base class defining common interface
  - `OusterWrapper` for Ouster sensors (Python SDK integration)
  - `VelodyneWrapper` for Velodyne sensors (dpkt-based parsing)
  - Wrapper registry system in `converter.py`

- **Enhanced Vendor Detection**: Multi-method detection system
  - UDP port detection (Ouster: 7502/7503, Velodyne: 2368/2369)
  - Packet structure analysis (magic bytes in UDP payload)
  - File header magic bytes detection
  - Companion file validation (Ouster JSON metadata)
  - Packet size pattern matching
  - Weighted confidence scoring (0-100%)

- **Comprehensive CLI**: Full command-line interface
  - `convert` - Single file conversion with automatic vendor detection
  - `batch` - Batch processing of multiple files
  - `detect` - Vendor detection without conversion
  - `health` - System health check and SDK validation
  - `test` - End-to-end pipeline testing
  - Rich output formatting with tables and progress bars
  - JSON output support for automation
  - Configurable logging and validation

- **Documentation**: Complete documentation suite
  - `CLI_README.md` - Comprehensive CLI usage guide
  - `TESTING_GUIDE.md` - Testing instructions and examples
  - `Wrappers/README.md` - Wrapper architecture documentation
  - Updated main `README.md` with current features

### Changed
- **Project Structure**: Reorganized for better modularity
  - Moved from single-file to multi-module architecture
  - Separated concerns: detection, conversion, CLI, wrappers
  - Added proper package structure with `__init__.py` files

- **Detection Algorithm**: Improved from simple pattern matching to multi-method scoring
  - Increased accuracy with weighted confidence calculation
  - Support for multiple detection methods per vendor
  - Configurable confidence thresholds

- **Error Handling**: Enhanced error reporting and recovery
  - Structured error dictionaries with detailed messages
  - Graceful degradation when SDKs unavailable
  - Comprehensive input validation

### Fixed
- **Memory Management**: Streaming PCAP processing to handle large files
- **Performance**: Configurable scan limits for faster testing
- **Compatibility**: Support for Python 3.8+ across platforms

## [0.1.0] - 2024-10-15

### Added
- **Initial Implementation**: Basic Ouster LiDAR conversion
  - Ouster PCAP to LAS conversion
  - Basic vendor detection
  - Simple CLI interface
  - Ouster SDK integration

- **Core Features**:
  - PCAP file parsing
  - LAS file generation
  - Point cloud coordinate conversion
  - Basic error handling

### Infrastructure
- Python package structure
- Requirements management
- Git repository setup
- Basic documentation

---

## Development Notes

### Version Numbering
- **Major** (X.0.0): Breaking changes to public API
- **Minor** (0.X.0): New features, backward compatible
- **Patch** (0.0.X): Bug fixes, backward compatible

### Supported Vendors Status
- ✅ **Ouster**: Full support (OS-0, OS-1, OS-2, OS-Dome series)
  - Output formats: LAS, LAZ, PCD, BIN, CSV
- ✅ **Velodyne**: Full support (VLP-16, VLP-32C, HDL-32E, HDL-64E, VLS-128)
  - Output formats: LAS, LAZ, PCD, BIN, CSV
- ✅ **Livox**: Partial support (Avia, Horizon, Tele-15, Mid-40/70)
  - Output formats: LAS (complete), LAZ/PCD/BIN/CSV (in progress)
- 🚧 **Hesai**: Planned (PandarXT, Pandar64, Pandar40P)
- 🚧 **RIEGL**: Planned (VUX series, miniVUX)

### Dependencies
- **Core**: numpy, scipy, laspy, dpkt
- **CLI**: click, rich, tqdm
- **Vendor SDKs**: ouster-sdk (optional), others as available
- **Development**: pytest, black, flake8, mypy