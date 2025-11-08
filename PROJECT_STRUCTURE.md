# LiDAR Converter - Project Structure

## Clean Project Layout

```
lidar-converter/
├── Lidar_Converter/              # Main Python package
│   ├── Wrappers/                 # Vendor-specific wrappers
│   │   ├── __init__.py
│   │   ├── base_wrapper.py       # Abstract base class
│   │   ├── ouster_wrapper.py     # Ouster implementation ✅
│   │   ├── velodyne_wrapper.py   # Velodyne implementation ✅
│   │   ├── livox_wrapper.py      # Livox implementation ✅
│   │   └── README.md             # Wrapper documentation
│   ├── SDK_Testing/              # SDK testing scripts
│   │   ├── generic_lidar_tester.py
│   │   ├── livox_sdk_tester.py
│   │   ├── pcap_to_las.py
│   │   └── velodyne_sdk_tester.py
│   ├── Sample_Data/              # Test data (gitignored)
│   │   ├── ouster/
│   │   ├── velodyne/
│   │   ├── livox/
│   │   └── hesai/
│   ├── Output/                   # Conversion outputs (gitignored)
│   ├── __init__.py               # Package initialization
│   ├── cli.py                    # Command-line interface
│   ├── converter.py              # Main conversion orchestrator
│   ├── detector.py               # Multi-method vendor detection
│   ├── utils.py                  # Utility functions
│   ├── pyproject.toml            # Package configuration
│   ├── CLI_README.md             # CLI documentation
│   └── TESTING_GUIDE.md          # Testing instructions
├── CloudCompare/                 # Visualization tool (gitignored)
├── venv/                         # Virtual environment (gitignored)
├── .gitignore                    # Git ignore rules
├── CHANGELOG.md                  # Version history
├── README.md                     # Main documentation
├── requirements.txt              # Python dependencies
├── example_usage.py              # Usage examples
└── verify_installation.py        # Installation verification

```

## Core Components

### Main Package (Lidar_Converter/)

**Core Modules:**
- `cli.py` - Command-line interface with rich output
- `converter.py` - Main orchestrator (detection → wrapper → conversion)
- `detector.py` - Multi-method vendor detection with confidence scoring
- `utils.py` - Shared utility functions

**Vendor Wrappers (Wrappers/):**
- `base_wrapper.py` - Abstract interface for all vendors
- `ouster_wrapper.py` - Ouster SDK integration
- `velodyne_wrapper.py` - Velodyne dpkt-based parsing
- `livox_wrapper.py` - Livox multi-format support (PCAP, LVX, CSV)

**Testing & Development:**
- `SDK_Testing/` - Individual SDK testing scripts
- `Sample_Data/` - Test data files (not in git)
- `Output/` - Conversion outputs (not in git)

### Documentation

**User Documentation:**
- `README.md` - Project overview, installation, quick start
- `CLI_README.md` - Complete CLI usage guide
- `TESTING_GUIDE.md` - Testing instructions and examples
- `CHANGELOG.md` - Version history and changes

**Developer Documentation:**
- `Wrappers/README.md` - Wrapper architecture and patterns
- `example_usage.py` - Code examples
- `verify_installation.py` - Installation verification

## File Organization Principles

### What's Included in Git:
✅ All source code (.py files)
✅ Documentation (.md files)
✅ Configuration (pyproject.toml, requirements.txt, .gitignore)
✅ Example and utility scripts

### What's Excluded from Git:
❌ Virtual environments (venv/)
❌ Data files (Sample_Data/, Output/)
❌ IDE configuration (.kiro/)
❌ Binary tools (CloudCompare/)
❌ Python cache (__pycache__/)
❌ Reference documents (PDFs, CSVs)

## Key Features by Component

### Detector (detector.py)
- Multi-method detection with weighted scoring
- UDP port analysis (35% weight)
- Packet structure detection (30% weight)
- Magic bytes detection (30% weight)
- Companion file validation (25% weight)
- Packet size patterns (20% weight)
- File extension hints (5% weight)

### Converter (converter.py)
- Automatic vendor detection
- Wrapper registry and instantiation
- Batch processing support
- Health monitoring
- Pipeline testing

### CLI (cli.py)
- `convert` - Single file conversion
- `batch` - Multiple file processing
- `detect` - Vendor identification
- `health` - System status check
- `test` - Pipeline validation

### Wrappers
- **Ouster**: Official SDK (ouster-sdk)
- **Velodyne**: dpkt PCAP parsing
- **Livox**: dpkt PCAP + LVX/LVX2 + CSV parsing

## Dependencies

### Core:
- numpy, scipy - Numerical computing
- laspy - LAS file handling
- dpkt - PCAP parsing

### CLI:
- click - Command framework
- rich - Terminal output
- tqdm - Progress bars

### Vendor SDKs:
- ouster-sdk - Ouster sensors (required)
- openpylivox - Livox sensors (optional)
- pandas - CSV parsing (optional)

### Development:
- pytest - Testing
- black - Code formatting
- flake8 - Linting
- mypy - Type checking

## Usage Patterns

### Quick Start:
```bash
python Lidar_Converter/cli.py health
python Lidar_Converter/cli.py detect file.pcap
python Lidar_Converter/cli.py convert file.pcap -o output.las
```

### Python API:
```python
from Lidar_Converter.converter import LiDARConverter

converter = LiDARConverter()
result = converter.convert("file.pcap", "output.las")
```

## Maintenance Notes

- Keep SDK_Testing/ for development reference
- Update CHANGELOG.md for each release
- Run verify_installation.py after dependency changes
- Test with all three vendors before releases