# LiDAR Converter - Command Line Interface Guide

A comprehensive guide to using the LiDAR Converter CLI tool for converting LiDAR data between formats.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Commands](#commands)
- [Usage Examples](#usage-examples)
- [Supported Formats](#supported-formats)
- [Advanced Options](#advanced-options)
- [Troubleshooting](#troubleshooting)

---

## Installation

```bash
pip install lidar-converter
```

Verify installation:
```bash
lidar-converter --version
lidar-converter health
```

---

## Quick Start

### Basic Conversion

```bash
# Convert a single file
lidar-converter convert input.pcap output.las

# Convert with auto-detection
lidar-converter convert data.pcap output.pcd --vendor auto

# Batch convert a directory
lidar-converter convert input_folder/ output_folder/ --format las
```

### Check System Health

```bash
lidar-converter health
```

This displays:
- Python version compatibility
- Installed dependencies
- Vendor SDK status
- Supported format capabilities

---

## Commands

### 1. `convert` - Convert LiDAR Data

Convert LiDAR files between different formats.

**Syntax:**
```bash
lidar-converter convert <input> <output> [OPTIONS]
```

**Arguments:**
- `input` - Input file or directory path
- `output` - Output file or directory path

**Options:**
- `--vendor` - Specify vendor (ouster, velodyne, livox, hesai, auto)
- `--format` - Output format (las, laz, pcd, ply, bin, csv)
- `--compression` - Enable compression for supported formats
- `--batch` - Enable batch processing for directories
- `--recursive` - Process subdirectories recursively
- `--overwrite` - Overwrite existing output files
- `--verbose` - Enable detailed logging

**Examples:**
```bash
# Single file conversion
lidar-converter convert scan.pcap scan.las --vendor ouster

# Batch conversion with compression
lidar-converter convert input/ output/ --format laz --batch

# Auto-detect vendor
lidar-converter convert data.pcap output.pcd --vendor auto --verbose
```

---

### 2. `detect` - Detect LiDAR Vendor

Identify the vendor and format of LiDAR data files.

**Syntax:**
```bash
lidar-converter detect <input> [OPTIONS]
```

**Arguments:**
- `input` - Input file or directory path

**Options:**
- `--detailed` - Show detailed detection information
- `--json` - Output results in JSON format
- `--batch` - Detect multiple files in a directory

**Examples:**
```bash
# Detect single file
lidar-converter detect scan.pcap

# Detailed detection
lidar-converter detect scan.pcap --detailed

# Batch detection with JSON output
lidar-converter detect data_folder/ --batch --json
```

---

### 3. `health` - System Health Check

Check system dependencies and capabilities.

**Syntax:**
```bash
lidar-converter health [OPTIONS]
```

**Options:**
- `--json` - Output in JSON format
- `--verbose` - Show detailed dependency information

**Example:**
```bash
lidar-converter health --verbose
```

---

### 4. `info` - File Information

Display detailed information about a LiDAR file.

**Syntax:**
```bash
lidar-converter info <input> [OPTIONS]
```

**Arguments:**
- `input` - Input file path

**Options:**
- `--json` - Output in JSON format
- `--stats` - Include statistical analysis

**Examples:**
```bash
# Basic file info
lidar-converter info scan.las

# Detailed statistics
lidar-converter info scan.las --stats --json
```

---

## Usage Examples

### Example 1: Convert Ouster PCAP to LAS

```bash
lidar-converter convert ouster_scan.pcap output.las --vendor ouster --verbose
```

### Example 2: Batch Convert Directory

```bash
lidar-converter convert ./raw_data/ ./converted/ \
  --format las \
  --batch \
  --recursive \
  --compression
```

### Example 3: Auto-Detect and Convert

```bash
lidar-converter detect unknown_file.pcap --detailed
lidar-converter convert unknown_file.pcap output.pcd --vendor auto
```

### Example 4: Convert with Compression

```bash
lidar-converter convert large_scan.las compressed.laz --compression
```

### Example 5: Pipeline Processing

```bash
# Detect vendor
VENDOR=$(lidar-converter detect scan.pcap --json | jq -r '.vendor')

# Convert based on detection
lidar-converter convert scan.pcap output.las --vendor $VENDOR
```

---

## Supported Formats

### Input Formats

| Format | Extension | Vendors | Description |
|--------|-----------|---------|-------------|
| PCAP | `.pcap` | Ouster, Velodyne, Livox, Hesai | Network packet capture |
| LAS | `.las` | All | ASPRS LiDAR format |
| LAZ | `.laz` | All | Compressed LAS |
| PCD | `.pcd` | All | Point Cloud Data |
| PLY | `.ply` | All | Polygon File Format |
| BIN | `.bin` | All | Binary point cloud |
| CSV | `.csv` | All | Comma-separated values |

### Output Formats

| Format | Extension | Compression | Features |
|--------|-----------|-------------|----------|
| LAS | `.las` | No | Industry standard, full metadata |
| LAZ | `.laz` | Yes | Compressed LAS, smaller files |
| PCD | `.pcd` | Optional | ASCII or binary, Open3D compatible |
| PLY | `.ply` | No | Mesh support, color data |
| BIN | `.bin` | No | Raw binary, fastest I/O |
| CSV | `.csv` | No | Human-readable, spreadsheet compatible |

---

## Advanced Options

### Environment Variables

```bash
# Set default vendor
export LIDAR_DEFAULT_VENDOR=ouster

# Set output format
export LIDAR_DEFAULT_FORMAT=las

# Enable debug logging
export LIDAR_DEBUG=1
```

### Configuration File

Create `~/.lidar-converter/config.yaml`:

```yaml
default_vendor: ouster
default_format: las
compression: true
overwrite: false
verbose: true
```

### Performance Tuning

```bash
# Process large files with memory optimization
lidar-converter convert huge.pcap output.las --chunk-size 1000000

# Parallel processing
lidar-converter convert input/ output/ --batch --workers 4
```

---

## Troubleshooting

### Common Issues

**1. "Vendor SDK not found"**
```bash
# Check health status
lidar-converter health

# Install missing dependencies
pip install ouster-sdk openpylivox
```

**2. "File format not recognized"**
```bash
# Use detect command first
lidar-converter detect unknown_file.pcap --detailed

# Try auto-detection
lidar-converter convert unknown_file.pcap output.las --vendor auto
```

**3. "Out of memory error"**
```bash
# Use chunked processing
lidar-converter convert large.pcap output.las --chunk-size 500000

# Convert to compressed format
lidar-converter convert large.las output.laz --compression
```

**4. "Permission denied"**
```bash
# Check file permissions
ls -la input_file.pcap

# Run with appropriate permissions
sudo lidar-converter convert input.pcap output.las
```

### Debug Mode

Enable verbose logging for troubleshooting:

```bash
lidar-converter convert input.pcap output.las --verbose --vendor auto
```

### Getting Help

```bash
# General help
lidar-converter --help

# Command-specific help
lidar-converter convert --help
lidar-converter detect --help
lidar-converter health --help
```

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | File not found |
| 3 | Invalid format |
| 4 | Vendor detection failed |
| 5 | Conversion failed |
| 6 | Permission denied |

---

## Best Practices

1. **Always check health before processing**
   ```bash
   lidar-converter health
   ```

2. **Use vendor detection for unknown files**
   ```bash
   lidar-converter detect input.pcap --detailed
   ```

3. **Enable compression for large datasets**
   ```bash
   lidar-converter convert input.las output.laz --compression
   ```

4. **Use batch mode for multiple files**
   ```bash
   lidar-converter convert input/ output/ --batch --recursive
   ```

5. **Verify output with info command**
   ```bash
   lidar-converter info output.las --stats
   ```

---

## Additional Resources

- **GitHub Repository**: https://github.com/Param-Patel-o5/lidar-multi-convert
- **Documentation**: See main README.md
- **Issue Tracker**: Report bugs on GitHub Issues
- **PyPI Package**: https://pypi.org/project/lidar-converter/

---

## Version Information

Check your installed version:
```bash
lidar-converter --version
```

Update to latest version:
```bash
pip install --upgrade lidar-converter
```

---

**Last Updated**: November 2024  
**CLI Version**: 0.2.0
