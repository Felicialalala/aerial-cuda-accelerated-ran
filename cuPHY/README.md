# cuPHY Software Development Kit (SDK)

## Overview

The cuPHY SDK is the physical-layer (L1) GPU-accelerated library within the Aerial SDK.
This README focuses on how the cuPHY component is organized and how to use its examples and tests.
Repository-wide prerequisites (toolchains, CUDA, container usage, etc.) are documented in the top-level `README.md` of the Aerial SDK.

## Directory Layout

- `src/` – Core cuPHY library implementation.
  - `src/cuphy/` – Main CUDA/C++ implementations and public APIs for 5G NR physical-layer functions and pipelines.
  - `src/cuphy_channels/` – Channel level aggregations used by `cuPHY-CP` and examples and tests.
  - `src/cuphy_hdf5/` – HDF5-based utilities for reading and writing test vectors, configuration data, and other artifacts used by cuPHY non-realtime logging and to feed reference data for examples.
- `examples/` – Reference implementations and sample applications that demonstrate how to build complete PHY pipelines and key components using cuPHY.
  Typical examples include:
  - `pdsch_tx` – PDSCH transmit processing pipeline.
  - `pusch_rx_multi_pipe` – Multi-cell PUSCH receive processing pipeline.
  - `srs_rx_pipeline` – SRS-based channel estimation and beamforming pipelines.
  - `prach_receiver_multi_cell` – PRACH receiver.
- `test/` – Unit and component tests for cuPHY kernels and higher-level primitives
  (e.g., rate matching, modulation/demodulation, error correction, HDF5 IO, etc.).
  This directory also includes helper scripts such as `cuphy_unit_test.sh` to orchestrate running collections of tests.
- `nvlog/` – Structured logging utility used throughout cuPHY (and other parts of the SDK when integrated).
  It is tailored to the performance requirements of the library (low overhead, high throughput logging).
- `docs/` – Doxygen configuration (`docs/Doxyfile.in`) and related documentation assets.
- `cmake/` – CMake utilities and toolchain files used when building cuPHY as a standalone component.
- `util/` – Supporting utilities (e.g., MATLAB integration, performance collection scripts, LDPC helper data).

## Building cuPHY Examples

The cuPHY SDK includes several example applications that demonstrate the usage of different cuPHY components.
These examples are typically located in the `examples/` directory and their CMake targets are defined in `CMakeLists.txt`.

Assuming you have already configured a build directory for the Aerial SDK/cuPHY (see the top-level `README.md` for details),
you can build individual cuPHY examples from that build directory.

To build a specific example target (e.g., `pdsch_tx`):

```shell
cmake --build . --target pdsch_tx
```

To build multiple specific example targets in a single command:

```shell
cmake --build . --target pdsch_tx pusch_rx_multi_pipe
```

To build a curated set of key cuPHY examples (as defined in the main `CMakeLists.txt` via the `cuphy_examples` target):

```shell
cmake --build . --target cuphy_examples
```

This is convenient for building a collection of demonstrator applications and pipeline reference implementations at once.

## Running cuPHY Tests

The `test/cuphy_unit_test.sh` script in the cuPHY source tree provides a convenient wrapper for running
subsets of unit tests and collecting summaries; see the script itself for details and options.

## Documentation

The cuPHY build system can generate API documentation using Doxygen when `BUILD_DOCS` is enabled (this is `ON` by default in `CMakeLists.txt`).
When enabled, the `docs_doxygen` target is created and invoked as part of the build, producing HTML documentation in the build tree under `docs/`.

To (re)generate the documentation explicitly from your build directory:

```shell
cmake --build . --target docs_doxygen
```

For more information on toolchains, global build options, and container usage, please refer to the top-level Aerial SDK `README.md`.


