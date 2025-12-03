# Running Phase-4 Tests

The scripts in `$cuBB_SDK/testBenches/phase4_test_scripts` can be used to run phase-4 tests (also known as cuBB tests).

## Quick Start with Test Configuration Parser

The recommended way to run tests is using the test configuration parser, which automatically generates all required parameters from a test case string.

### Prerequisites

1. **Reserve and log into a pair of lockable systems** (DU and RU nodes)
2. **Ensure codebase is on shared NFS** - All nodes must have access to the same `$cuBB_SDK` directory via shared network filesystem
3. **Deploy containers:**
   - DU node: Either two containers (DU1 for main operations, DU2 for testMAC) OR one container with two sessions
   - RU node: One container (RU)
4. **Prepare test information:**
   - Test case string (e.g., `F08_6C_79_MODCOMP_STT480000_EH_1P`)
   - Host configuration (e.g., `CG1_R750`, `CG1_CG1`, or `GL4_R750`)

### Test Configuration and Setup

| Container | Command | Description |
|-----------|---------|-------------|
| DU1 | `$cuBB_SDK/testBenches/phase4_test_scripts/parse_test_config_params.sh "F08_6C_79_MODCOMP_STT480000_EH_1P" "CG1_R750" test_params.sh` | Generate parameter file from test case string |
| DU1 | `source test_params.sh` | Load parameters into environment |
| RU | `source test_params.sh` | Load parameters into environment |

### Build and Configure

| Container | Command | Description |
|-----------|---------|-------------|
| DU1* | `$cuBB_SDK/testBenches/phase4_test_scripts/copy_test_files.sh $COPY_TEST_FILES_PARAMS` | Copy test vectors |
| DU1 | `$cuBB_SDK/testBenches/phase4_test_scripts/build_aerial_sdk.sh $BUILD_AERIAL_PARAMS` | Build Aerial SDK |
| DU1 | `$cuBB_SDK/testBenches/phase4_test_scripts/setup1_DU.sh $SETUP1_DU_PARAMS` | Configure DU setup |
| RU | `$cuBB_SDK/testBenches/phase4_test_scripts/setup2_RU.sh $SETUP2_RU_PARAMS` | Configure RU setup |
| DU1 | `$cuBB_SDK/testBenches/phase4_test_scripts/test_config.sh $TEST_CONFIG_PARAMS` | Configure test parameters |

*Note: `copy_test_files.sh` may need to be run outside the container if the container does not have access to the test vector directory.

### Run Test

| Container | Command | Description |
|-----------|---------|-------------|
| RU | `$cuBB_SDK/testBenches/phase4_test_scripts/run1_RU.sh $RUN1_RU_PARAMS` | Start RU emulator |
| DU1 | `$cuBB_SDK/testBenches/phase4_test_scripts/run2_cuPHYcontroller.sh $RUN2_CUPHYCONTROLLER_PARAMS` | Start cuPHY controller |
| DU2 | `source test_params.sh`<br>`$cuBB_SDK/testBenches/phase4_test_scripts/run3_testMAC.sh $RUN3_TESTMAC_PARAMS` | Start testMAC (separate container on DU) |

## Test Case String Format

Format: `F08_<cells>C_<pattern>_<modifiers>`

- `F08` - Required prefix for performance test cases
- `<cells>C` - Number of cells (e.g., `6C`, `20C`)
- `<pattern>` - Pattern number (e.g., `79`, `59c`)
- `<modifiers>` - Optional modifiers in any order:
  - `BFP9` or `BFP14` - BFP compression with specified bits
  - `STT<value>` - Schedule total time (e.g., `STT480000`)
  - `1P` or `2P` - Number of ports
  - `EH` - Enable early HARQ
  - `GC` - Enable green context
  - `WC<value>` - Work cancel mode (e.g., `WC2`)
  - `PMU<value>` - PMU metrics mode (e.g., `PMU3`)
  - `NS<value>` - Number of slots (e.g., `NS30000`)
  - `NICD` - Enable NIC timing logs
  - `RUWT` - Enable RU C-plane worker tracing logs
  - `NOPOST` - Enable reduced logging mode (disables detailed tracing and processing time logs)

### Host Configuration

Format: `<DU_HOST>_<RU_HOST>`

Valid combinations:
- `CG1_R750` - Grace (CG1) DU with x86 (R750) RU
- `CG1_CG1` - Grace DU with Grace RU
- `GL4_R750` - Grace+L4 (GL4) DU with x86 RU


## Parser Examples

### Basic 6-cell MODCOMP test with Early HARQ:
```bash
$cuBB_SDK/testBenches/phase4_test_scripts/parse_test_config_params.sh "F08_6C_79_MODCOMP_STT480000_EH_1P" "CG1_R750" test_params.sh
```

### 20-cell test with BFP14, Early HARQ, Green Context, Dual port, 30000 total slots in the run, with NIC-level debug:
```bash
$cuBB_SDK/testBenches/phase4_test_scripts/parse_test_config_params.sh "F08_20C_59c_BFP14_EH_GC_NS30000_NICD_2P" "GL4_R750" test_params.sh
```

### With comprehensive debugging (NIC timings + RU C-plane worker tracing):
```bash
$cuBB_SDK/testBenches/phase4_test_scripts/parse_test_config_params.sh "F08_6C_79_MODCOMP_STT480000_EH_NICD_RUWT_1P" "CG1_R750" test_params.sh
```

### With custom build directories:

By default, the parsing scripts will determine which preset configuration to run the test with and the build artifacts will be in build.$PRESET.$(uname -m).
However, if desired, the --custom-build-dir flag can be used, which will tell all of the scripts to use $CUSTOM.$(uname -m) instead.

```bash
$cuBB_SDK/testBenches/phase4_test_scripts/parse_test_config_params.sh "F08_6C_79_MODCOMP_STT480000_EH_1P" "CG1_R750" test_params.sh \
    --custom-build-dir custom
```

## Environment Variables Set by Parser

- `COPY_TEST_FILES_PARAMS` - Parameters for copy_test_files.sh
- `BUILD_AERIAL_PARAMS` - Parameters for build_aerial_sdk.sh
- `SETUP1_DU_PARAMS` - Parameters for setup1_DU.sh
- `SETUP2_RU_PARAMS` - Parameters for setup2_RU.sh
- `TEST_CONFIG_PARAMS` - Parameters for test_config.sh
- `RUN1_RU_PARAMS` - Parameters for run1_RU.sh
- `RUN2_CUPHYCONTROLLER_PARAMS` - Parameters for run2_cuPHYcontroller.sh
- `RUN3_TESTMAC_PARAMS` - Parameters for run3_testMAC.sh

## Antenna Configuration Patterns

Test patterns are designed for different antenna configurations:

- **4T4R (4 Transmit / 4 Receive):**
  - Pattern `59c` - Peak traffic load
  - Pattern `60c` - Average traffic load

- **64T64R (64 Transmit / 64 Receive):**
  - Pattern `79` - Requires MUMIMO configuration

MUMIMO configuration should be used when running 64T64R patterns. All MUMIMO patterns are defined in the `mimo_patterns` variable in `valid_perf_patterns.sh`.

## Manual Script Usage

All scripts support `-h` or `--help` options for detailed usage information. You can also run them manually without the parser:

```bash
# Example manual usage
$cuBB_SDK/testBenches/phase4_test_scripts/copy_test_files.sh 79 --max_cells 6
$cuBB_SDK/testBenches/phase4_test_scripts/setup1_DU.sh --mumimo 1
$cuBB_SDK/testBenches/phase4_test_scripts/setup2_RU.sh
$cuBB_SDK/testBenches/phase4_test_scripts/test_config.sh 79 --compression=4 --num-cells=6 --num-slots=600000
$cuBB_SDK/testBenches/phase4_test_scripts/run1_RU.sh
$cuBB_SDK/testBenches/phase4_test_scripts/run2_cuPHYcontroller.sh
$cuBB_SDK/testBenches/phase4_test_scripts/run3_testMAC.sh
```

## Multi-L2 test

Multi-L2 test means running two L2/testMAC instances with one cuphycontroller instance.
Below instruction use F08 8C 60 case as example.

### Configure

```bash
$cuBB_SDK/testBenches/phase4_test_scripts/setup1_DU.sh --ml2  # Enable Multi-L2 by "--ml2"
$cuBB_SDK/testBenches/phase4_test_scripts/setup2_RU.sh
$cuBB_SDK/testBenches/phase4_test_scripts/test_config.sh 60 --num-cells=8
```

The test_config.sh command automatically configure as below for Multi-L2 test.

(1) Enable Multi-L2 by setting `nvipc_config_file` in `l2_adapter_config_XXX.yaml`.

```yaml
nvipc_config_file: nvipc_multi_instances.yaml
```

(2) Configure nvipc_multi_instances.yaml to assign cell `0~3` for the first instance and cell `4~7` for the second instance, set the second instance prefix to nvipc1.

```yaml
transport:
- transport_id: 0
  phy_cells: [0, 1, 2, 3]
  shm_config:
    prefix: nvipc

- transport_id: 1
  phy_cells: [4, 5, 6, 7]
  shm_config:
    prefix: nvipc1
```

(3) Create a test_mac_config_1.yaml for the secondary testMAC instance and configure necessary values in it.

```yaml
# Copy test_mac_config.yaml and change below values to test_mac_config_1.yaml

transport:
  shm_config: {prefix: nvipc1}
log_name: testmac1.log
oam_server_addr: 0.0.0.0:50053

# Assign CPU cores for the second test_mac instance
low_priority_core: 30
sched_thread_config: {name: mac_sched, cpu_affinity: 31, sched_priority: 96}
recv_thread_config: {name: mac_recv, cpu_affinity: 32, sched_priority: 95}
builder_thread_config: {name: fapi_builder, cpu_affinity: 33, sched_priority: 95}
worker_cores: [34, 35, 36, 37, 38, 39]
```

### Run Multi-L2 test

```bash
$cuBB_SDK/testBenches/phase4_test_scripts/run1_RU.sh                 # Run ru_emulator
$cuBB_SDK/testBenches/phase4_test_scripts/run2_cuPHYcontroller.sh    # Run cuphycontroller_scf
$cuBB_SDK/testBenches/phase4_test_scripts/run3_testMAC.sh --ml2 0    # Run the first instance of test_mac
$cuBB_SDK/testBenches/phase4_test_scripts/run3_testMAC.sh --ml2 1    # Run the second instance of test_mac
```

The --ml2 0/1 is translated to below arguments for test_mac:

```bash
# First instance: enable cell 0~3 by cell_mask=0x0F, use default config file test_mac_config.yaml
--ml2 0  =>  --cells 0x0F
# Second instance: enable cell 4~7 by cell_mask=0xF0, explicitly select config file test_mac_config_1.yaml
--ml2 1  =>  --cells 0xF0 --config test_mac_config_1.yaml
```

## Additional Notes

- Configuration files can be automatically generated for nrSim test cases. See [Generic steps to run nrSim test cases](https://docs.nvidia.com/aerial/cuda-accelerated-ran/latest/aerial_cubb/cubb_quickstart/running_cubb-end-to-end.html#generic-steps-to-run-nrsim-test-cases-using-phase4-scripts)
- The `yq` tool must be installed for test_config.sh to work outside the container
- Build directories may differ based on system architecture (aarch64 or x86_64)
