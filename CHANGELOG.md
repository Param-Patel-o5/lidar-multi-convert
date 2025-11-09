# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Hesai LiDAR Support** - Full support for Hesai sensors (PandarXT-32, PandarXT-16, Pandar64, Pandar40P, PandarQT)
  - Automatic vendor detection with 48% confidence scoring
  - PCAP packet parsing based on official Hesai documentation
  - Support for all output formats (LAS, LAZ, PCD, BIN, CSV)
  - Dynamic packet structure parsing for flexibility across models
  - Reads channel count, block count, and distance unit from packet headers
  - Adaptive block size calculation for different Hesai models
- **LAZ Compression** - Real compression now working for all vendors
  - Added `lazrs>=0.8.0` dependency for LAZ compression
  - Compression ratio: 60-80% smaller than uncompressed LAS
  - Verified compression for Ouster, Velodyne, Livox, and Hesai
  - Typical compression: 3-15 bytes/point (vs 28-30 for uncompressed)

### Changed
- Updated README.md with Hesai support information
- Updated vendor detection to include Hesai magic bytes (0xEEFF)
- Updated UDP port detection to recognize Hesai port (2368)
- Improved packet size detection for Hesai (861 bytes)
- Enhanced documentation with Hesai packet structure details

### Fixed
- LAZ compression now works properly (was creating uncompressed files before)
- Hesai packet parsing now correctly reads data from byte 12 (not byte 42)
- Fixed block size calculation to use header information
- Improved point filtering to remove invalid/noise points

## [0.2.0] - 2025-11-10

### Added
- Livox LiDAR support (Avia, Horizon, Tele-15, Mid-40/70, HAP)
- Multiple output format support (PCD, BIN, CSV) in addition to LAS/LAZ
- Batch conversion capability
- Health check command
- Comprehensive CLI with rich output formatting

### Changed
- Improved vendor detection with multi-method approach
- Enhanced error handling and logging
- Optimized performance with configurable scan limits

## [0.1.0] - 2025-11-01

### Added
- Initial release
- Ouster LiDAR support (OS-0, OS-1, OS-2, OS-Dome series)
- Velodyne LiDAR support (VLP-16, VLP-32C, HDL-32E, HDL-64E, VLS-128)
- Automatic vendor detection
- LAS/LAZ output format support
- Command-line interface
- Python API

[Unreleased]: https://github.com/Param-Patel-o5/lidar-converter/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Param-Patel-o5/lidar-converter/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Param-Patel-o5/lidar-converter/releases/tag/v0.1.0
