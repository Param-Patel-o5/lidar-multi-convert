# Changelog

All notable changes to the LiDAR Converter project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- ✅ **Velodyne**: Full support (VLP-16, VLP-32C, HDL-32E, HDL-64E, VLS-128)
- 🚧 **Hesai**: Planned (PandarXT, Pandar64, Pandar40P)
- 🚧 **Livox**: Planned (Avia, Horizon, Tele-15)
- 🚧 **RIEGL**: Planned (VUX series, miniVUX)

### Dependencies
- **Core**: numpy, scipy, laspy, dpkt
- **CLI**: click, rich, tqdm
- **Vendor SDKs**: ouster-sdk (optional), others as available
- **Development**: pytest, black, flake8, mypy