# lidar-converter — command-line interface

The installed console script is **`lidar-converter`** (defined in `pyproject.toml`).  
Global options (`--log-level`, `--config`, `--output-format`, etc.) must appear **before** the subcommand.

## Install

```bash
pip install lidar-converter
# or from source:
pip install -e .
```

Verify:

```bash
lidar-converter --help
lidar-converter --log-level ERROR --output-format json health
```

## Subcommands

| Command | Purpose |
|---------|---------|
| `convert` | One input file → one output file (format from `-o` extension or `-f`) |
| `batch` | Many files in a directory (shared output format) |
| `detect` | Vendor detection only |
| `health` | SDK / wrapper availability |
| `test` | Run pipeline test on one file (detect + wrapper test conversion) |

There is **no** `info` subcommand and **no** `--version` flag in the current CLI (use `pip show lidar-converter` for the package version).

## `convert`

**Syntax:**

```bash
lidar-converter convert INPUT_FILE -o OUTPUT_FILE [options]
```

**Important:** output path is **` -o ` / `--output`**, not a second positional argument.

**Examples:**

```bash
# LAS (extension chooses format)
lidar-converter convert data.pcap -o out.las --max-scans 10

# LAZ, PCD, BIN, CSV
lidar-converter convert data.pcap -o out.laz --max-scans 10
lidar-converter convert data.pcap -o out.pcd --max-scans 10
lidar-converter convert data.pcap -o out.bin --max-scans 10
lidar-converter convert data.pcap -o out.csv --max-scans 10

# Explicit format (optional)
lidar-converter convert data.pcap -o out.file --format laz

# Ouster: metadata JSON beside PCAP (same stem) or pass explicitly
lidar-converter convert scan.pcap -o out.las -c scan.json --max-scans 10
```

**Common options:**

- `-o`, `--output` — output file path  
- `-f`, `--format` — `las`, `laz`, `pcd`, `bin`, `csv` (optional if extension is clear)  
- `--max-scans` — limit scans for faster tests (default in CLI: 1000)  
- `-c`, `--calibration` — calibration / metadata file  
- `--validate` — validate output after conversion  
- `--no-intensity` — drop intensity  

## `batch`

```bash
lidar-converter batch INPUT_DIR -o OUTPUT_DIR [--pattern "*.pcap"] [-r] [--max-scans 10]
```

Output filenames are derived from input stems; format comes from `-f` or config.

## `detect`

```bash
lidar-converter --output-format json detect file.pcap
```

## `health`

```bash
lidar-converter --output-format json health
```

Use JSON on Windows if the Rich tables fail due to console encoding (see `TESTING_GUIDE.md`).

## `test`

```bash
lidar-converter test file.pcap
```

## Configuration file

Optional JSON (merged with defaults), searched in order:

1. Path from `--config`  
2. `~/.lidar_converter.json`  
3. `./.lidar_converter.json`  

Environment overrides: `LIDAR_OUTPUT_FORMAT`, `LIDAR_OUTPUT_DIR`, `LIDAR_LOG_LEVEL`.

## Further reading

- Repository root: `README.md`, `TESTING_GUIDE.md`  
- Source on GitHub: [Param-Patel-o5/lidar-multi-convert](https://github.com/Param-Patel-o5/lidar-multi-convert) (PyPI: `lidar-converter`)  
- Wrappers: `Lidar_Converter/Wrappers/README.md`
