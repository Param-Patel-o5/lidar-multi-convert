#!/usr/bin/env python3
"""
Command-line interface for LiDAR conversion pipeline.

This module provides a user-friendly CLI for converting LiDAR files
across multiple vendors to standardized formats.

Usage:
    lidar-convert convert <input_file> -o <output_file>
    lidar-convert batch <input_dir> -o <output_dir>
    lidar-convert detect <input_file>
    lidar-convert health
    lidar-convert test <input_file>
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime

try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.panel import Panel
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from .converter import LiDARConverter
from .detector import VendorDetector

# Initialize console for rich output
if RICH_AVAILABLE:
    console = Console()
else:
    console = None

logger = logging.getLogger(__name__)


def setup_logger(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Configure logging for the CLI.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path to save logs
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    handlers = [logging.StreamHandler(sys.stderr)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from file, environment variables, or defaults.
    
    Priority: CLI args > env vars > config file > defaults
    
    Args:
        config_path: Optional path to config file
        
    Returns:
        dict: Configuration dictionary
    """
    config = {
        "output_format": "las",
        "output_dir": None,
        "log_level": "INFO",
        "validate_output": True,
        "preserve_intensity": True
    }
    
    # Load from config file
    if config_path:
        config_file = Path(config_path)
    else:
        # Try default locations
        config_file = Path.home() / ".lidar_converter.json"
        if not config_file.exists():
            config_file = Path(".lidar_converter.json")
    
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                file_config = json.load(f)
                config.update(file_config)
        except Exception as e:
            logger.warning(f"Failed to load config file: {e}")
    
    # Override with environment variables
    config["output_format"] = os.environ.get("LIDAR_OUTPUT_FORMAT", config["output_format"])
    config["output_dir"] = os.environ.get("LIDAR_OUTPUT_DIR", config["output_dir"])
    config["log_level"] = os.environ.get("LIDAR_LOG_LEVEL", config["log_level"])
    
    return config


def format_result(result: Dict[str, Any], output_format: str = "default", verbose: bool = False) -> str:
    """
    Format conversion result for display.
    
    Args:
        result: Result dictionary from converter
        output_format: Output format ("default", "json", "verbose", "quiet")
        verbose: Whether to show detailed information
        
    Returns:
        str: Formatted result string
    """
    if output_format == "json":
        return json.dumps(result, indent=2, default=str)
    
    if output_format == "quiet":
        return "SUCCESS" if result.get("success") else "FAILED"
    
    # Default/verbose formatting
    if RICH_AVAILABLE:
        return format_result_rich(result, verbose)
    else:
        return format_result_plain(result, verbose)


def format_result_rich(result: Dict[str, Any], verbose: bool) -> str:
    """Format result using rich library for pretty output."""
    if result.get("success"):
        status = "[green]✓ SUCCESS[/green]"
    else:
        status = "[red]✗ FAILED[/red]"
    
    # Main info table
    table = Table(title="Conversion Result", box=box.ROUNDED)
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Status", status)
    table.add_row("Vendor", result.get("vendor", "Unknown"))
    table.add_row("Input File", result.get("input_file", "N/A"))
    table.add_row("Output File", result.get("output_file", "N/A"))
    table.add_row("Format", result.get("output_format", "N/A").upper())
    
    if result.get("success"):
        table.add_row("Points Converted", f"{result.get('points_converted', 0):,}")
        table.add_row("Conversion Time", f"{result.get('conversion_time', 0):.2f}s")
        table.add_row("Detection Time", f"{result.get('detection_time', 0):.3f}s")
        table.add_row("Confidence", f"{result.get('detection_confidence', 0):.2%}")
    
    message = result.get("message", "")
    
    # Warnings and errors
    warnings = result.get("warnings", [])
    errors = result.get("errors", [])
    
    output_parts = []
    
    # Display main table
    console.print(table)
    
    # Display message
    if message:
        if result.get("success"):
            console.print(f"\n[green]{message}[/green]")
        else:
            console.print(f"\n[red]{message}[/red]")
    
    # Display errors
    if errors:
        error_text = "\n".join(f"  • {e}" for e in errors)
        console.print(Panel(error_text, title="[red]Errors[/red]", border_style="red"))
    
    # Display warnings
    if warnings:
        warning_text = "\n".join(f"  • {w}" for w in warnings)
        console.print(Panel(warning_text, title="[yellow]Warnings[/yellow]", border_style="yellow"))
    
    # Verbose details
    if verbose and result.get("success"):
        verbose_info = {
            "Vendor": result.get("vendor"),
            "Detection Confidence": f"{result.get('detection_confidence', 0):.4f}",
            "Total Time": f"{result.get('conversion_time', 0):.4f}s",
            "Points": result.get("points_converted", 0)
        }
        verbose_table = Table(title="Verbose Details", box=box.SIMPLE)
        verbose_table.add_column("Key", style="cyan")
        verbose_table.add_column("Value", style="white")
        for key, value in verbose_info.items():
            verbose_table.add_row(key, str(value))
        console.print(verbose_table)
    
    return ""  # Rich prints directly, return empty string


def format_result_plain(result: Dict[str, Any], verbose: bool) -> str:
    """Format result as plain text (fallback when rich is not available)."""
    lines = []
    
    status = "SUCCESS" if result.get("success") else "FAILED"
    lines.append(f"\n{'='*60}")
    lines.append(f"Conversion Result: {status}")
    lines.append(f"{'='*60}")
    
    lines.append(f"Vendor: {result.get('vendor', 'Unknown')}")
    lines.append(f"Input File: {result.get('input_file', 'N/A')}")
    lines.append(f"Output File: {result.get('output_file', 'N/A')}")
    lines.append(f"Format: {result.get('output_format', 'N/A').upper()}")
    
    if result.get("success"):
        lines.append(f"Points Converted: {result.get('points_converted', 0):,}")
        lines.append(f"Conversion Time: {result.get('conversion_time', 0):.2f}s")
        lines.append(f"Detection Confidence: {result.get('detection_confidence', 0):.2%}")
    
    message = result.get("message", "")
    if message:
        lines.append(f"\nMessage: {message}")
    
    errors = result.get("errors", [])
    if errors:
        lines.append(f"\nErrors:")
        for error in errors:
            lines.append(f"  • {error}")
    
    warnings = result.get("warnings", [])
    if warnings:
        lines.append(f"\nWarnings:")
        for warning in warnings:
            lines.append(f"  • {warning}")
    
    lines.append(f"{'='*60}\n")
    
    return "\n".join(lines)


def cmd_convert(args, config: Dict[str, Any]) -> int:
    """
    Handle convert command.
    
    Args:
        args: Parsed arguments
        config: Configuration dictionary
        
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    input_file = args.input_file
    output_file = args.output or config.get("output_dir")
    
    # Infer format from output file extension if not explicitly specified
    if args.format:
        output_format = args.format
    elif output_file and Path(output_file).suffix:
        # Extract format from file extension
        output_format = Path(output_file).suffix.lstrip('.').lower()
    else:
        output_format = config.get("output_format", "las")
    
    # Build kwargs from args
    kwargs = {}
    if args.sensor_model:
        kwargs["sensor_model"] = args.sensor_model
    if args.calibration:
        kwargs["calibration_file"] = args.calibration
    if args.validate:
        kwargs["validate_output"] = True
    if args.no_intensity:
        kwargs["preserve_intensity"] = False
    if hasattr(args, 'max_scans') and args.max_scans:
        kwargs["max_scans"] = args.max_scans
    
    # Show progress
    if RICH_AVAILABLE:
        console.print(f"[cyan]Converting:[/cyan] {input_file}")
        console.print(f"[cyan]Output:[/cyan] {output_file or 'auto-generated'}")
        console.print(f"[cyan]Format:[/cyan] {output_format.upper()}\n")
    
    # Initialize converter
    converter = LiDARConverter()
    
    # Perform conversion
    result = converter.convert(
        input_file,
        output_file,
        output_format,
        **kwargs
    )
    
    # Format and display result
    output = format_result(result, args.output_format or "default", args.verbose)
    if output:
        print(output)
    
    # Exit with appropriate code
    return 0 if result.get("success") else 1


def cmd_batch(args, config: Dict[str, Any]) -> int:
    """
    Handle batch command.
    
    Args:
        args: Parsed arguments
        config: Configuration dictionary
        
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    input_dir = Path(args.input_dir)
    output_dir = args.output_dir or config.get("output_dir") or "converted_output"
    output_format = args.format or config.get("output_format", "las")
    pattern = args.pattern or "*.pcap"
    
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}", file=sys.stderr)
        return 1
    
    # Find files
    files = []
    if args.recursive:
        files = list(input_dir.rglob(pattern))
    else:
        files = list(input_dir.glob(pattern))
    
    if not files:
        print(f"No files found matching pattern '{pattern}' in {input_dir}", file=sys.stderr)
        return 1
    
    if RICH_AVAILABLE:
        console.print(f"[cyan]Found {len(files)} files to convert[/cyan]")
        console.print(f"[cyan]Output directory:[/cyan] {output_dir}")
        console.print(f"[cyan]Format:[/cyan] {output_format.upper()}\n")
    
    # Initialize converter
    converter = LiDARConverter()
    
    # Progress callback
    def progress_callback(current: int, total: int, file_path: str):
        if TQDM_AVAILABLE:
            # tqdm will be handled by convert_batch
            pass
        elif RICH_AVAILABLE:
            console.print(f"[{current}/{total}] Processing: {Path(file_path).name}")
    
    # Build kwargs
    kwargs = {}
    if args.validate:
        kwargs["validate_output"] = True
    if args.no_intensity:
        kwargs["preserve_intensity"] = False
    if hasattr(args, 'max_scans') and args.max_scans:
        kwargs["max_scans"] = args.max_scans
    
    # Convert batch
    results = converter.convert_batch(
        [str(f) for f in files],
        output_dir,
        output_format,
        progress_callback=progress_callback if not TQDM_AVAILABLE else None,
        **kwargs
    )
    
    # Summary
    successful = sum(1 for r in results if r.get("success"))
    failed = len(results) - successful
    
    if RICH_AVAILABLE:
        table = Table(title="Batch Conversion Summary", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        table.add_row("Total Files", str(len(results)))
        table.add_row("Successful", f"[green]{successful}[/green]")
        table.add_row("Failed", f"[red]{failed}[/red]" if failed > 0 else "0")
        
        total_points = sum(r.get("points_converted", 0) for r in results if r.get("success"))
        total_time = sum(r.get("conversion_time", 0) for r in results)
        table.add_row("Total Points", f"{total_points:,}")
        table.add_row("Total Time", f"{total_time:.2f}s")
        
        console.print("\n")
        console.print(table)
        
        # Show failed files
        if failed > 0:
            failed_table = Table(title="Failed Files", box=box.SIMPLE, border_style="red")
            failed_table.add_column("File", style="red")
            failed_table.add_column("Error", style="white")
            
            for r in results:
                if not r.get("success"):
                    failed_table.add_row(
                        Path(r.get("input_file", "")).name,
                        r.get("message", "Unknown error")
                    )
            
            console.print("\n")
            console.print(failed_table)
    else:
        print(f"\n{'='*60}")
        print(f"Batch Conversion Summary")
        print(f"{'='*60}")
        print(f"Total Files: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        
        if failed > 0:
            print("\nFailed Files:")
            for r in results:
                if not r.get("success"):
                    print(f"  • {Path(r.get('input_file', '')).name}: {r.get('message', 'Unknown error')}")
    
    return 0 if failed == 0 else 1


def cmd_detect(args, config: Dict[str, Any]) -> int:
    """
    Handle detect command.
    
    Args:
        args: Parsed arguments
        config: Configuration dictionary
        
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    input_file = args.input_file
    
    detector = VendorDetector()
    result = detector.detect_vendor(input_file)
    
    if args.output_format == "json":
        print(json.dumps(result, indent=2, default=str))
        return 0 if result.get("success") else 1
    
    # Format detection result
    if RICH_AVAILABLE:
        table = Table(title="Detection Result", box=box.ROUNDED)
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="white")
        
        status = "[green]✓ DETECTED[/green]" if result.get("success") else "[red]✗ NOT DETECTED[/red]"
        table.add_row("Status", status)
        table.add_row("File", result.get("file_path", "N/A"))
        table.add_row("File Size", f"{result.get('file_size', 0):,} bytes")
        
        if result.get("success"):
            table.add_row("Vendor", result.get("vendor_name", "Unknown"))
            table.add_row("Confidence", f"{result.get('confidence', 0):.2%}")
            table.add_row("File Signature", result.get("file_signature", "N/A"))
            
            metadata = result.get("metadata", {})
            if metadata:
                methods = metadata.get("detection_methods", [])
                table.add_row("Detection Methods", ", ".join(methods) if methods else "N/A")
        else:
            table.add_row("Error", result.get("error", "Unknown error"))
        
        console.print(table)
        
        if args.verbose and result.get("metadata"):
            verbose_table = Table(title="Metadata", box=box.SIMPLE)
            verbose_table.add_column("Key", style="cyan")
            verbose_table.add_column("Value", style="white")
            
            metadata = result.get("metadata", {})
            for key, value in metadata.items():
                if key != "detection_methods":  # Already shown
                    verbose_table.add_row(str(key), str(value))
            
            if len(verbose_table.rows) > 0:
                console.print("\n")
                console.print(verbose_table)
    else:
        print(f"\n{'='*60}")
        print(f"Detection Result")
        print(f"{'='*60}")
        print(f"Status: {'DETECTED' if result.get('success') else 'NOT DETECTED'}")
        print(f"File: {result.get('file_path', 'N/A')}")
        
        if result.get("success"):
            print(f"Vendor: {result.get('vendor_name', 'Unknown')}")
            print(f"Confidence: {result.get('confidence', 0):.2%}")
            print(f"File Signature: {result.get('file_signature', 'N/A')}")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
        print(f"{'='*60}\n")
    
    return 0 if result.get("success") else 1


def cmd_health(args, config: Dict[str, Any]) -> int:
    """
    Handle health check command.
    
    Args:
        args: Parsed arguments
        config: Configuration dictionary
        
    Returns:
        int: Exit code (0 if all healthy, 1 otherwise)
    """
    converter = LiDARConverter()
    health = converter.health_check()
    
    if args.output_format == "json":
        print(json.dumps(health, indent=2, default=str))
        return 0 if health.get("status") == "ok" else 1
    
    # Format health check
    if RICH_AVAILABLE:
        status_color = {
            "ok": "green",
            "degraded": "yellow",
            "error": "red"
        }.get(health.get("status"), "white")
        
        console.print(f"\n[bold]Overall Status:[/bold] [{status_color}]{health.get('status', 'unknown').upper()}[/{status_color}]\n")
        
        table = Table(title="Vendor Status", box=box.ROUNDED)
        table.add_column("Vendor", style="cyan")
        table.add_column("Available", style="white")
        table.add_column("Status", style="white")
        table.add_column("SDK Version", style="white")
        table.add_column("Supported Formats", style="white")
        
        for vendor, info in health.get("vendors", {}).items():
            available = "✓" if info.get("available") else "✗"
            available_style = "green" if info.get("available") else "red"
            
            table.add_row(
                vendor,
                f"[{available_style}]{available}[/{available_style}]",
                info.get("status", "unknown"),
                info.get("sdk_version", "N/A") or "N/A",
                ", ".join(info.get("supported_formats", []))
            )
        
        console.print(table)
    else:
        print(f"\n{'='*60}")
        print(f"Health Check")
        print(f"{'='*60}")
        print(f"Overall Status: {health.get('status', 'unknown').upper()}\n")
        
        print("Vendor Status:")
        for vendor, info in health.get("vendors", {}).items():
            available = "✓" if info.get("available") else "✗"
            print(f"  {vendor}: {available}")
            print(f"    Status: {info.get('status', 'unknown')}")
            print(f"    SDK Version: {info.get('sdk_version', 'N/A')}")
            print(f"    Formats: {', '.join(info.get('supported_formats', []))}")
        print(f"{'='*60}\n")
    
    return 0 if health.get("status") == "ok" else 1


def cmd_test(args, config: Dict[str, Any]) -> int:
    """
    Handle test command.
    
    Args:
        args: Parsed arguments
        config: Configuration dictionary
        
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    input_file = args.input_file
    
    if RICH_AVAILABLE:
        console.print(f"[cyan]Testing pipeline with:[/cyan] {input_file}\n")
    
    converter = LiDARConverter()
    result = converter.test_pipeline(input_file)
    
    if args.output_format == "json":
        print(json.dumps(result, indent=2, default=str))
        return 0 if result.get("success") else 1
    
    # Format test result
    if RICH_AVAILABLE:
        status = "[green]✓ PASSED[/green]" if result.get("success") else "[red]✗ FAILED[/red]"
        
        table = Table(title="Pipeline Test Result", box=box.ROUNDED)
        table.add_column("Stage", style="cyan")
        table.add_column("Status", style="white")
        
        # Detection stage
        detection = result.get("detection", {})
        det_status = "[green]✓[/green]" if detection.get("success") else "[red]✗[/red]"
        table.add_row("Detection", f"{det_status} {detection.get('vendor_name', 'N/A')}")
        
        # Conversion stage
        conversion = result.get("conversion", {})
        conv_status = "[green]✓[/green]" if conversion.get("success") else "[red]✗[/red]"
        table.add_row("Conversion", conv_status)
        
        # Validation stage
        val_status = "[green]✓[/green]" if result.get("validation") else "[red]✗[/red]"
        table.add_row("Validation", val_status)
        
        table.add_row("Overall", status)
        
        console.print(table)
        console.print(f"\n[white]{result.get('message', '')}[/white]")
    else:
        print(f"\n{'='*60}")
        print(f"Pipeline Test Result")
        print(f"{'='*60}")
        print(f"Overall: {'PASSED' if result.get('success') else 'FAILED'}")
        print(f"Detection: {'✓' if result.get('detection', {}).get('success') else '✗'}")
        print(f"Conversion: {'✓' if result.get('conversion', {}).get('success') else '✗'}")
        print(f"Validation: {'✓' if result.get('validation') else '✗'}")
        print(f"Message: {result.get('message', '')}")
        print(f"{'='*60}\n")
    
    return 0 if result.get("success") else 1


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="lidar-convert",
        description="LiDAR conversion pipeline - Convert LiDAR files across multiple vendors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single file
  lidar-convert convert data.pcap -o output.las
  
  # Batch convert all pcap files
  lidar-convert batch ./lidar_data -o ./converted --pattern "*.pcap"
  
  # Detect vendor
  lidar-convert detect data.pcap
  
  # Health check
  lidar-convert health
  
  # Test pipeline
  lidar-convert test data.pcap
  
For more information, see: https://github.com/Param-Patel-o5/lidar-converter
        """
    )
    
    # Global arguments
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="Save logs to file"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output-format",
        choices=["default", "json", "verbose", "quiet"],
        default="default",
        help="Output format"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimal output"
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Convert command
    convert_parser = subparsers.add_parser("convert", help="Convert a single LiDAR file")
    convert_parser.add_argument("input_file", type=str, help="Input LiDAR file")
    convert_parser.add_argument("-o", "--output", type=str, help="Output file path")
    convert_parser.add_argument("-f", "--format", type=str, default=None,
                               choices=["las", "laz", "pcd", "bin", "csv"],
                               help="Output format (auto-detected from output file extension if not specified)")
    convert_parser.add_argument("--sensor-model", type=str, help="Sensor model identifier")
    convert_parser.add_argument("-c", "--calibration", type=str, help="Path to calibration/metadata file")
    convert_parser.add_argument("--validate", action="store_true", help="Validate output file")
    convert_parser.add_argument("--no-intensity", action="store_true", help="Don't preserve intensity values")
    convert_parser.add_argument("--max-scans", type=int, default=1000, 
                               help="Maximum number of scans to process (default: 1000, set lower for faster conversion)")
    
    # Batch command
    batch_parser = subparsers.add_parser("batch", help="Batch convert multiple files")
    batch_parser.add_argument("input_dir", type=str, help="Input directory")
    batch_parser.add_argument("-o", "--output-dir", type=str, help="Output directory")
    batch_parser.add_argument("-f", "--format", type=str, default=None,
                            choices=["las", "laz", "pcd", "bin", "csv"],
                            help="Output format (auto-detected from output file extension if not specified)")
    batch_parser.add_argument("-p", "--pattern", type=str, default="*.pcap",
                             help="File pattern to match")
    batch_parser.add_argument("-r", "--recursive", action="store_true",
                             help="Search subdirectories")
    batch_parser.add_argument("--validate", action="store_true", help="Validate output files")
    batch_parser.add_argument("--no-intensity", action="store_true", help="Don't preserve intensity values")
    batch_parser.add_argument("--max-scans", type=int, default=1000,
                             help="Maximum number of scans to process per file (default: 1000)")
    
    # Detect command
    detect_parser = subparsers.add_parser("detect", help="Detect vendor of a LiDAR file")
    detect_parser.add_argument("input_file", type=str, help="Input LiDAR file")
    
    # Health command
    health_parser = subparsers.add_parser("health", help="Check health status of all vendors")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test entire pipeline on a file")
    test_parser.add_argument("input_file", type=str, help="Input test file")
    
    return parser.parse_args()


def main() -> int:
    """Main entry point for CLI."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config if args else None)
    
    # Setup logging
    log_level = args.log_level if args else config.get("log_level", "INFO")
    if args and args.quiet:
        log_level = "ERROR"
    setup_logger(log_level, args.log_file if args else None)
    
    # Handle quiet mode
    if args and args.quiet:
        args.output_format = "quiet"
    
    # Route to appropriate command handler
    if not args or not args.command:
        print("Error: No command specified. Use --help for usage information.", file=sys.stderr)
        return 2
    
    try:
        if args.command == "convert":
            return cmd_convert(args, config)
        elif args.command == "batch":
            return cmd_batch(args, config)
        elif args.command == "detect":
            return cmd_detect(args, config)
        elif args.command == "health":
            return cmd_health(args, config)
        elif args.command == "test":
            return cmd_test(args, config)
        else:
            print(f"Error: Unknown command: {args.command}", file=sys.stderr)
            return 2
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.", file=sys.stderr)
        return 130
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

