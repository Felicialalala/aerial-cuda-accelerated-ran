# DLC Test Bench Runner

Automated script for running DLC (Data Link Control) test bench within the Aerial container environment.

## Prerequisites

**IMPORTANT: This test bench only runs on aarch64 (ARM64/Grace) systems.**

The script will automatically detect the architecture and fail if not running on aarch64.

## Quick Start

### Running from within Aerial Container (aarch64)

```bash
# Full test run with all patterns
cd $cuBB_SDK/cuPHY-CP/tests/dlc_test_bench
./run_dlc_tests.sh

# OR explicitly with bash
bash ./run_dlc_tests.sh

# Quick iteration (skip build and config)
./run_dlc_tests.sh --skip-build --skip-config
```

**Important**: Do NOT run with `sh` command (use `bash` or `./` instead)

### Running from Host (Outside Container)

```bash
# The script will automatically use run_aerial.sh if not in container
cd aerial_sdk/cuPHY-CP/tests/dlc_test_bench
./run_dlc_tests.sh
```

## Common Usage Patterns

### Development Workflow

```bash
# Initial run (full build and test)
./run_dlc_tests.sh

# Iteration on code (skip build, fast execution)
./run_dlc_tests.sh --skip-build --skip-config --pattern 79

# After code changes (rebuild only)
./run_dlc_tests.sh --build-only

# Test specific pattern after rebuild
./run_dlc_tests.sh --skip-config --pattern 79
```

### Debugging

```bash
# Run with verbose output
./run_dlc_tests.sh --pattern 79 -v

# Capture packets for Wireshark analysis
./run_dlc_tests.sh --pattern 79 --enable-pcap --pcap-file debug_pattern79.pcap

# Run without verification (faster)
./run_dlc_tests.sh --pattern 79 --no-verify
```

### Performance Analysis

```bash
# Run benchmarks
./run_dlc_tests.sh --benchmark --pattern 79

# Run benchmarks with packet capture
./run_dlc_tests.sh --benchmark --enable-pcap --pattern 79
```

## Test Pattern Configuration

Test patterns are read from CSV files in this directory:

- `test_patterns_mmimo.csv` - mMIMO (massive MIMO) test patterns
- `test_patterns_4t4r.csv` - 4T4R test patterns

### CSV Format

```csv
# Comments start with #
# One pattern number per line
# Commas are optional and will be stripped

79
59
69
```

### Editing Patterns

1. Edit the appropriate CSV file
2. Add or remove pattern numbers
3. Run the script (it will automatically read the updated file)

```bash
# Edit patterns
vim test_patterns_mmimo.csv

# Run updated patterns
./run_dlc_tests.sh
```

## Command Line Options

### Build Options

| Option | Description |
|--------|-------------|
| `--skip-build` | Skip build step, use existing binaries |
| `--build-only` | Only build, don't run tests |
| `--build-type TYPE` | Build type: `Release` (default) or `Debug` |
| `--build-dir DIR` | Custom build directory (default: `build.aarch64`) |
| `--toolchain TC` | CMake toolchain: `grace-cross` (default for aarch64), `devkit`, `native`, etc. |

### Test Options

| Option | Description |
|--------|-------------|
| `--skip-config` | Skip YAML configuration step |
| `--mode MODE` | Test mode: `mmimo` (default) or `4t4r` |
| `--pattern NUM` | Run specific pattern only |

### Execution Options

| Option | Description |
|--------|-------------|
| `--benchmark` | Run benchmarks instead of unit tests |
| `--enable-pcap` | Enable PCAP packet capture |
| `--pcap-file NAME` | PCAP filename (default: cplane_packets.pcap) |
| `--no-verify` | Disable C-Plane packet verification |
| `-v, --verbose` | Enable verbose output |

### Other Options

| Option | Description |
|--------|-------------|
| `-h, --help` | Show help message |

## Directory Structure

```
dlc_test_bench/
├── run_dlc_tests.sh          # Main runner script
├── test_patterns_mmimo.csv   # mMIMO test patterns
├── test_patterns_4t4r.csv    # 4T4R test patterns
├── dlc_test_bench.cpp        # Test implementation
├── sendCplaneUnitTest.cpp    # Unit test implementation
├── pcap_writer.cpp           # PCAP capture utility
└── README_run_dlc_tests.md   # This file
```

## Important Notes

### Architecture Requirement

**This test bench ONLY runs on aarch64 (ARM64/Grace) architecture.**

The script will automatically:
- Detect the system architecture with `uname -m`
- Exit with an error if not running on aarch64
- Use the appropriate build directory (`build.aarch64`)
- Use the correct toolchain (`grace-cross` by default)

### Execution Directory

**The dlc_test_bench executable must run from the aerial_sdk root directory** due to hardcoded relative paths in the code:
- `./cuPHY/nvlog/config/nvlog_config.yaml`
- `./cuPHY-CP/testMAC/testMAC/test_mac_config.yaml`

The script automatically handles this by changing to `$AERIAL_SDK_ROOT` before execution. Log files are still saved in the dlc_test_bench directory.

### Build System

The script uses the existing `testBenches/phase4_test_scripts/build_aerial_sdk.sh` script for building, which:
- Automatically selects the correct toolchain for aarch64
- Creates architecture-specific build directories
- Supports incremental builds
- Integrates with the mission-mode testing infrastructure

### Test Configuration

The script uses `dlc_test_config.sh` helper script to configure test parameters without requiring `setup1_DU.sh` and `setup2_RU.sh`. It:
- Creates a minimal `test_config_summary.sh` with required variables
- Calls `test_config.sh` with DLC-specific parameters:
  - Pattern number (from CSV or command line)
  - 1 cell configuration
  - `-o 4` (MODCOMP compression method)
  - 1 port configuration
- Bypasses setup dependency checks

## Output Files

### Log Files

Test execution logs are saved in the test bench directory:
- `dlc_test_<pattern>.log` - Per-pattern execution log (in dlc_test_bench directory)
- `build.log` - Build output log (in dlc_test_bench directory)
- `nvlog_out.log` - NVLog output (in aerial_sdk root directory)

### PCAP Files

When `--enable-pcap` is used, packet captures are saved to:
- `/tmp/<pcap_file_name>` (default: `/tmp/cplane_packets.pcap`)

To analyze in Wireshark:
```bash
# Copy from container to host
docker cp <container_id>:/tmp/cplane_packets.pcap .

# Open in Wireshark
wireshark cplane_packets.pcap
```

## Workflow Examples

### Initial Setup and Test

```bash
# 1. Enter container
./cuPHY-CP/container/run_aerial.sh

# 2. Navigate to test directory
cd $cuBB_SDK/cuPHY-CP/tests/dlc_test_bench

# 3. Run full test suite
./run_dlc_tests.sh

# Output:
# ========================================================================
# DLC Test Bench Runner
# ========================================================================
# Mode: mmimo
# Build directory: /workspace/aerial_sdk/build.x86_64
# Skip build: 0
# Skip config: 0
# ...
# ========================================================================
# Test Summary
# ========================================================================
# Total patterns: 3
# Passed: 3
# All patterns passed!
```

### Iterative Development

```bash
# 1. Make code changes to sendCplaneUnitTest.cpp
vim sendCplaneUnitTest.cpp

# 2. Rebuild
./run_dlc_tests.sh --build-only

# 3. Test specific pattern
./run_dlc_tests.sh --skip-config --pattern 79

# 4. Debug with PCAP if needed
./run_dlc_tests.sh --skip-build --skip-config --pattern 79 --enable-pcap
```

### Adding New Test Patterns

```bash
# 1. Edit pattern file
echo "99" >> test_patterns_mmimo.csv

# 2. Run tests (will automatically include new pattern)
./run_dlc_tests.sh --skip-build
```

## Troubleshooting

### Shell Error (sh vs bash)

```
Error: This script requires bash, but it's running under a different shell.

Please run it with one of these commands:
  bash ./run_dlc_tests.sh
  ./run_dlc_tests.sh

Do NOT use: sh ./run_dlc_tests.sh
```

**Problem**: Running with `sh` instead of `bash`

**Solution**: The script uses bash-specific features. Run it with:
```bash
# Method 1: Direct execution (uses shebang)
./run_dlc_tests.sh

# Method 2: Explicit bash
bash ./run_dlc_tests.sh

# Do NOT use:
# sh ./run_dlc_tests.sh  ❌ WRONG
```

### Architecture Error

```
[ERROR] This test bench only runs on aarch64 (ARM64) architecture
[ERROR] Current architecture: x86_64
[ERROR] Please run this on an ARM64/Grace system
```

**Solution**: This test bench is designed for ARM64/Grace systems only. You must run it on an aarch64 machine.

### Build Fails

```bash
# Clean and rebuild
rm -rf $cuBB_SDK/build.aarch64
./run_dlc_tests.sh

# Check build log
cat $cuBB_SDK/cuPHY-CP/tests/dlc_test_bench/build.log
```

### Executable Not Found

```bash
# Verify build directory (should be build.aarch64 on ARM systems)
./run_dlc_tests.sh --build-dir $cuBB_SDK/build.aarch64

# Or rebuild
./run_dlc_tests.sh
```

### Pattern File Not Found

```bash
# Check CSV file exists
ls test_patterns_mmimo.csv

# Create if missing
cat > test_patterns_mmimo.csv << EOF
# Test patterns
79
59
EOF
```

### Configuration Issues

```bash
# Skip configuration if not needed
./run_dlc_tests.sh --skip-config

# Or check testMAC config exists
ls $cuBB_SDK/cuPHY-CP/testMAC/testMAC/test_mac_config.yaml
```

## Integration with Existing Scripts

This script is designed to work standalone within the DU container, without requiring:
- `setup1_du.sh` / `setup2_ru.sh`
- RU emulator running on separate node
- Test vector copying

For full mission-mode testing with RU emulator, use the existing phase4 test scripts in `testBenches/phase4_test_scripts/`.

## Environment Variables

The script respects the following environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `cuBB_SDK` | Aerial SDK root directory | Auto-detected |
| `BUILD_DIR` | Build directory | `$cuBB_SDK/build.aarch64` |
| `ARCH` | System architecture | Auto-detected (`uname -m`) |

## Notes

- **This test bench ONLY runs on aarch64 (ARM64/Grace) systems**
- The script automatically detects if running inside the Aerial container
- Build artifacts are placed in `build.aarch64` (architecture-specific)
- Uses the existing `build_aerial_sdk.sh` script from phase4 test infrastructure
- Configuration step is simplified for DU-only execution
- All test vectors should be available in the standard locations
- The script automatically selects the correct toolchain for aarch64 (`grace-cross`)

## Support

For issues or questions:
1. Check the main Aerial SDK README
2. Verify all prerequisites are met
3. Check container logs for detailed error messages

