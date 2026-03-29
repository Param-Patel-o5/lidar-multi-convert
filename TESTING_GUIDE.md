# Testing and validation

## Source repository

Canonical GitHub project: **[Param-Patel-o5/lidar-multi-convert](https://github.com/Param-Patel-o5/lidar-multi-convert)**  
PyPI package name remains **`lidar-converter`** (install: `pip install lidar-converter`).

## Health check

Global flags go **before** the subcommand:

```bash
lidar-converter --log-level ERROR --output-format json health
```

On Windows, if Rich tables error on Unicode, keep `--output-format json`.

## Vendor SDKs vs what `pip` can install

| Vendor | Shipped Python path | `pip install` vendor SDK? |
|--------|---------------------|---------------------------|
| **Ouster** | **`ouster-sdk`** | Yes — already a dependency (`ouster-sdk`). |
| **Velodyne** | **dpkt** + in-repo parsers | No — there is no official Velodyne Python SDK on PyPI; the wrapper uses **dpkt** (optional VeloView CLI if you install the desktop app separately). |
| **Livox** | **dpkt** for PCAP (primary) | **openpylivox** / **openpylivox-pkg** are **not** published on PyPI under those names in a way that `pip install` can resolve here; PCAP conversion still works via **dpkt**. Optional: install **Livox Viewer** on Windows if you want the wrapper to detect `livox_viewer`. |
| **Hesai** | **dpkt** fallback unless native lib is built | **No** `hesai-sdk` wheel on PyPI for this project. The repo may contain **`Lidar_Converter/Wrappers/hesai_sdk/`** (C++ SDK sources); the wrapper looks for a **built** `PandarGeneralSDK.dll` / `.so` under `hesai_sdk/build/`. If not built, conversion uses **dpkt** (same as today). |

Commands attempted (for reference):

```text
pip install openpylivox openpylivox-pkg   # No matching distribution (typical)
pip index versions hesai-sdk            # Not found on PyPI
```

To use the native Hesai SDK you must compile the vendor C++ SDK and place the library in `hesai_sdk/build/` as documented in Hesai’s SDK package.

## Manual CLI checks (sample PCAPs)

Adjust paths to your machine. Examples use **`--max-scans 100`** (raise toward `1000` or omit for full files).  
Ouster needs metadata: use the JSON next to the PCAP or `-c` to the JSON path.

### Velodyne (VLP-16 sample)

```bash
python -m Lidar_Converter.cli --output-format json detect "Lidar_Converter/Sample_Data/Velodyne/CarLoop_Velodyne-VLP16.pcap"

python -m Lidar_Converter.cli convert "Lidar_Converter/Sample_Data/Velodyne/CarLoop_Velodyne-VLP16.pcap" -o manual_convert_out/velodyne/out.las  --max-scans 100
python -m Lidar_Converter.cli convert "Lidar_Converter/Sample_Data/Velodyne/CarLoop_Velodyne-VLP16.pcap" -o manual_convert_out/velodyne/out.laz --max-scans 100
python -m Lidar_Converter.cli convert "Lidar_Converter/Sample_Data/Velodyne/CarLoop_Velodyne-VLP16.pcap" -o manual_convert_out/velodyne/out.pcd --max-scans 100
python -m Lidar_Converter.cli convert "Lidar_Converter/Sample_Data/Velodyne/CarLoop_Velodyne-VLP16.pcap" -o manual_convert_out/velodyne/out.bin --max-scans 100
python -m Lidar_Converter.cli convert "Lidar_Converter/Sample_Data/Velodyne/CarLoop_Velodyne-VLP16.pcap" -o manual_convert_out/velodyne/out.csv --max-scans 100
```

### Hesai

```bash
python -m Lidar_Converter.cli --output-format json detect "Lidar_Converter/Sample_Data/hesai/hesai_BusyRoad.pcap"

python -m Lidar_Converter.cli convert "Lidar_Converter/Sample_Data/hesai/hesai_BusyRoad.pcap" -o manual_convert_out/hesai/out.las  --max-scans 100
python -m Lidar_Converter.cli convert "Lidar_Converter/Sample_Data/hesai/hesai_BusyRoad.pcap" -o manual_convert_out/hesai/out.laz --max-scans 100
python -m Lidar_Converter.cli convert "Lidar_Converter/Sample_Data/hesai/hesai_BusyRoad.pcap" -o manual_convert_out/hesai/out.pcd --max-scans 100
python -m Lidar_Converter.cli convert "Lidar_Converter/Sample_Data/hesai/hesai_BusyRoad.pcap" -o manual_convert_out/hesai/out.bin --max-scans 100
python -m Lidar_Converter.cli convert "Lidar_Converter/Sample_Data/hesai/hesai_BusyRoad.pcap" -o manual_convert_out/hesai/out.csv --max-scans 100
```

### Livox

```bash
python -m Lidar_Converter.cli --output-format json detect "Lidar_Converter/Sample_Data/Livox/StaticCarIntersection_Livox-HAP.pcap"

python -m Lidar_Converter.cli convert "Lidar_Converter/Sample_Data/Livox/StaticCarIntersection_Livox-HAP.pcap" -o manual_convert_out/livox/out.las  --max-scans 100
python -m Lidar_Converter.cli convert "Lidar_Converter/Sample_Data/Livox/StaticCarIntersection_Livox-HAP.pcap" -o manual_convert_out/livox/out.laz --max-scans 100
python -m Lidar_Converter.cli convert "Lidar_Converter/Sample_Data/Livox/StaticCarIntersection_Livox-HAP.pcap" -o manual_convert_out/livox/out.pcd --max-scans 100
python -m Lidar_Converter.cli convert "Lidar_Converter/Sample_Data/Livox/StaticCarIntersection_Livox-HAP.pcap" -o manual_convert_out/livox/out.bin --max-scans 100
python -m Lidar_Converter.cli convert "Lidar_Converter/Sample_Data/Livox/StaticCarIntersection_Livox-HAP.pcap" -o manual_convert_out/livox/out.csv --max-scans 100
```

### Ouster (metadata JSON required)

```bash
python -m Lidar_Converter.cli --output-format json detect "Lidar_Converter/Sample_Data/ouster/Urban_Drive_.pcap"

python -m Lidar_Converter.cli convert "Lidar_Converter/Sample_Data/ouster/Urban_Drive_.pcap" -o manual_convert_out/ouster/out.las  -c "Lidar_Converter/Sample_Data/ouster/Urban_Drive_.json" --max-scans 100
python -m Lidar_Converter.cli convert "Lidar_Converter/Sample_Data/ouster/Urban_Drive_.pcap" -o manual_convert_out/ouster/out.laz -c "Lidar_Converter/Sample_Data/ouster/Urban_Drive_.json" --max-scans 100
python -m Lidar_Converter.cli convert "Lidar_Converter/Sample_Data/ouster/Urban_Drive_.pcap" -o manual_convert_out/ouster/out.pcd -c "Lidar_Converter/Sample_Data/ouster/Urban_Drive_.json" --max-scans 100
python -m Lidar_Converter.cli convert "Lidar_Converter/Sample_Data/ouster/Urban_Drive_.pcap" -o manual_convert_out/ouster/out.bin -c "Lidar_Converter/Sample_Data/ouster/Urban_Drive_.json" --max-scans 100
python -m Lidar_Converter.cli convert "Lidar_Converter/Sample_Data/ouster/Urban_Drive_.pcap" -o manual_convert_out/ouster/out.csv -c "Lidar_Converter/Sample_Data/ouster/Urban_Drive_.json" --max-scans 100
```

Use `--output-format json` on `convert` to print `points_converted`, `conversion_time`, and `detection_confidence` for metrics.

## Ignored output folders

Add local conversion output dirs (e.g. `manual_convert_out/`) to `.gitignore` so large files are not committed.
