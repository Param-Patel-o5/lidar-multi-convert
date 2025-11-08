# LiDAR Converter CLI Documentation

Complete guide to using the command-line interface for the LiDAR conversion pipeline.

## Installation

After installing the package, the CLI is available as:

```bash
lidar-convert --help
```

Or directly via Python:

```bash
python -m Lidar_Converter.cli --help
```

## Commands

### 1. Convert Single File

Convert a single LiDAR file to a standardized format.

```bash
lidar-convert convert <input_file> [options]
```

**Options:**
- `-o, --output <path>` - Output file path (auto-generated if not provided)
- `-f, --format <format>` - Output format: `las`, `laz`, `pcd`, `bin`, `csv` (auto-detected from file extension if not specified)
- `--max-scans <number>` - Limit number of scans to process (for faster testing)
- `--sensor-model <model>` - Sensor model identifier (e.g., "OS1-64", "VLP-16")
- `-c, --calibration <file>` - Path to calibration/metadata file
- `--validate` - Validate output file after conversion
- `--no-intensity` - Don't preserve intensity values

**Examples:**
```bash
# Basic conversion - format auto-detected from extension
lidar-convert convert data.pcap -o output.las

# Convert to different formats (auto-detected)
lidar-convert convert data.pcap -o output.laz  # LAZ format
lidar-convert convert data.pcap -o output.pcd  # PCD format
lidar-convert convert data.pcap -o output.bin  # BIN format
lidar-convert convert data.pcap -o output.csv  # CSV format

# Fast preview with scan limit
lidar-convert convert data.pcap -o output.las --max-scans 10

# Convert with validation
lidar-convert convert data.pcap -o output.laz --validate

# Explicitly specify format (optional)
lidar-convert convert data.pcap -o output.laz --format laz
```

### 2. Batch Conversion

Convert multiple files in a directory.

```bash
lidar-convert batch <input_dir> [options]
```

**Options:**
- `-o, --output-dir <dir>` - Output directory (default: `converted_output/`)
- `-f, --format <format>` - Output format for all files (default: `las`)
- `-p, --pattern <pattern>` - File pattern to match (default: `*.pcap`)
- `-r, --recursive` - Search subdirectories
- `--validate` - Validate all output files
- `--no-intensity` - Don't preserve intensity values

**Examples:**
```bash
# Convert all PCAP files in directory
lidar-convert batch ./lidar_data -o ./converted

# Convert with recursive search
lidar-convert batch ./lidar_data -r -o ./converted

# Convert only Velodyne files
lidar-convert batch ./lidar_data -p "velodyne_*.pcap" -o ./converted

# Convert mixed vendor files (automatic detection)
lidar-convert batch ./mixed_lidar_data -o ./converted
```

### 3. Detect Vendor

Detect the vendor/manufacturer of a LiDAR file without converting.

```bash
lidar-convert detect <input_file>
```

**Examples:**
```bash
# Detect vendor
lidar-convert detect data.pcap

# JSON output for scripting
lidar-convert detect data.pcap --output-format json

# Verbose output
lidar-convert detect data.pcap --verbose
```

### 4. Health Check

Check the status of all registered vendor wrappers and their SDKs.

```bash
lidar-convert health
```

**Output shows:**
- Overall system status (ok/degraded/error)
- Per-vendor status
- SDK availability and versions
- Supported output formats

**Example:**
```bash
lidar-convert health
```

### 5. Test Pipeline

Run an end-to-end test (detection → conversion → validation) on a file.

```bash
lidar-convert test <input_file>
```

**Examples:**
```bash
# Test pipeline
lidar-convert test data.pcap

# JSON output
lidar-convert test data.pcap --output-format json
```

## Output Formats

The CLI supports automatic format detection from file extensions:

### Supported Formats

| Format | Extension | Vendor Support | Description |
|--------|-----------|----------------|-------------|
| **LAS** | `.las` | Ouster, Velodyne, Livox | ASPRS standard, uncompressed |
| **LAZ** | `.laz` | Ouster, Velodyne, Livox | Compressed LAS (60-80% smaller) |
| **PCD** | `.pcd` | Ouster, Velodyne, Livox | Point Cloud Library format |
| **BIN** | `.bin` | Ouster, Velodyne, Livox | KITTI binary format |
| **CSV** | `.csv` | Ouster, Velodyne, Livox | Human-readable text format |

### Format Auto-Detection

Simply specify the desired extension in the output filename:

```bash
# No --format flag needed!
lidar-convert convert data.pcap -o output.las  # → LAS format
lidar-convert convert data.pcap -o output.laz  # → LAZ format
lidar-convert convert data.pcap -o output.pcd  # → PCD format
lidar-convert convert data.pcap -o output.bin  # → BIN format
lidar-convert convert data.pcap -o output.csv  # → CSV format
```

### Format-Specific Notes

- **LAZ**: Automatic compression using laspy 2.0+ or external laszip tool. Falls back to uncompressed LAS if compression unavailable.
- **PCD**: ASCII format for maximum compatibility with PCL and ROS.
- **BIN**: Little-endian float32 format (16 bytes per point: x, y, z, intensity).
- **CSV**: Includes header row with column names.

## Performance Tips

### Scan Limiting

Use `--max-scans` to process only a portion of large files:

```bash
# Fast preview (10 scans)
lidar-convert convert large_file.pcap -o preview.las --max-scans 10

# Medium sample (100 scans)
lidar-convert convert large_file.pcap -o sample.las --max-scans 100

# Full file (omit --max-scans)
lidar-convert convert large_file.pcap -o full.las
```

### Performance by Vendor

| Vendor | 10 Scans | Points | Time |
|--------|----------|--------|------|
| Ouster | ~500K | 517,860 | ~0.5s |
| Velodyne | ~900 | 866 | ~0.05s |
| Livox | ~1M | 1,000,000 | ~11s |

**Note**: Livox uses point-based limiting (1 scan ≈ 100,000 points) for optimal performance.

## Global Options

Available for all commands:

- `--log-level <level>` - Set logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR`
- `--log-file <file>` - Save logs to file
- `--config <file>` - Path to configuration file
- `--output-format <format>` - Output format: `default`, `json`, `verbose`, `quiet`
- `--verbose, -v` - Show detailed output
- `--quiet, -q` - Minimal output (just success/failure)

## Output Formats

### Default (Human-Readable)
Pretty tables and formatted text with colored output (requires `rich` library).

### JSON
Machine-readable JSON output for scripting and automation:
```bash
lidar-convert convert data.pcap --output-format json
```

### Verbose
Detailed output with all metadata:
```bash
lidar-convert convert data.pcap --verbose
```

### Quiet
Minimal output - just SUCCESS or FAILED:
```bash
lidar-convert convert data.pcap --quiet
```

## Configuration

### Configuration File

Create a JSON configuration file at `~/.lidar_converter.json` or `.lidar_converter.json`:

```json
{
  "output_format": "las",
  "output_dir": "./converted",
  "log_level": "INFO",
  "validate_output": true,
  "preserve_intensity": true
}
```

### Environment Variables

Override defaults using environment variables:

- `LIDAR_OUTPUT_FORMAT` - Default output format
- `LIDAR_OUTPUT_DIR` - Default output directory
- `LIDAR_LOG_LEVEL` - Logging level

### Priority Order

Configuration priority (highest to lowest):
1. Command-line arguments
2. Environment variables
3. Configuration file
4. Hardcoded defaults

## Examples

### Complete Workflow

```bash
# 1. Check system health
lidar-convert health

# 2. Detect vendor of a file
lidar-convert detect data.pcap

# 3. Convert single file
lidar-convert convert data.pcap -o output.las --validate

# 4. Batch convert all files
lidar-convert batch ./lidar_data -o ./converted --validate

# 5. Test pipeline
lidar-convert test data.pcap
```

### Advanced Usage

```bash
# Convert with custom sensor model and calibration
lidar-convert convert data.pcap \
  --sensor-model "OS1-64" \
  --calibration metadata.json \
  -o output.las \
  --validate

# Batch convert with recursive search and custom pattern
lidar-convert batch ./data \
  -r \
  -p "*.pcap" \
  -o ./output \
  -f laz \
  --validate

# Verbose output for debugging
lidar-convert convert data.pcap \
  --verbose \
  --log-level DEBUG \
  --log-file conversion.log

# JSON output for automation
lidar-convert convert data.pcap \
  --output-format json \
  | jq '.points_converted'
```

## Exit Codes

- `0` - Success
- `1` - Error (conversion failed, file not found, etc.)
- `2` - Invalid arguments
- `130` - User interrupt (Ctrl+C)

## Troubleshooting

### SDK Not Found

If a vendor's SDK is not installed:

```bash
# Check status
lidar-convert health

# Install required SDK (example for Ouster)
pip install ouster-sdk
```

### Permission Errors

Ensure output directory is writable:

```bash
# Check permissions
ls -la output_dir/

# Create output directory if needed
mkdir -p output_dir
```

### Debug Mode

Enable debug logging to see detailed information:

```bash
lidar-convert convert data.pcap \
  --log-level DEBUG \
  --log-file debug.log \
  --verbose
```

## Integration with Scripts

The CLI can be integrated into scripts:

```bash
#!/bin/bash
# Convert all files and check results
for file in *.pcap; do
  result=$(lidar-convert convert "$file" --output-format json)
  success=$(echo "$result" | jq -r '.success')
  
  if [ "$success" = "true" ]; then
    echo "✓ $file converted successfully"
  else
    echo "✗ $file failed: $(echo "$result" | jq -r '.message')"
  fi
done
```

## Support

For issues and questions, see:
- GitHub: https://github.com/Param-Patel-o5/lidar-converter
- Documentation: See `README.md` for project overview

