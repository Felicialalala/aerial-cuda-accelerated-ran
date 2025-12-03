# TensorRT-based Channel Estimator Testing Framework

This directory contains scripts for testing and validating the EnhancedFusedChannelEstimator model across multiple formats (PyTorch, ONNX, TensorRT) and integration scenarios. The framework provides a comprehensive test suite for verifying the model's performance both in isolation and as part of the PUSCH Rx pipeline.

## Prerequisites

- CUDA-compatible GPU
- CUDA Toolkit (11.0 or later)
- PyTorch (1.10 or later)
- ONNX Runtime (with CUDA support)
- TensorRT (8.0 or later)
- CuPy

Ensure the environment is properly set up with PYTHONPATH including the pyaerial directory:

```bash
export PYTHONPATH=/opt/nvidia/cuBB:$PYTHONPATH
```

## Algorithm Framework Support

The framework provides built-in support for various input/output tensor types:

- NumPy arrays
- PyTorch tensors
- CuPy arrays (GPU-accelerated arrays)

The `PyTorchAlgorithm` class in `algorithm_base/ml_algorithm.py` automatically handles conversion between these tensor types, allowing seamless operation with both CPU and GPU data.

## Test Flow

The test framework follows this sequence of operations:

1. Run PyTorch model to generate reference outputs
2. Export the model to ONNX format and verify outputs match PyTorch
3. Export the ONNX model to TensorRT engine and verify outputs match ONNX
4. Generate YAML configuration for the TensorRT engine
5. Test the TensorRT-based channel estimator in standalone mode
6. Test the TensorRT-based channel estimator in the PUSCH Rx pipeline

This flow is fully automated in the `test-in-cicd.sh` script.

## Tensor Dimensions

To ensure compatibility with the ground truth configurations, the following dimensions are used:

- Input shape: [batch, subcarriers, layers, rx_antennas, symbols, 2]
  - batch: 1
  - subcarriers: 1638
  - layers: 4
  - rx_antennas: 4
  - symbols: 2
  - last dim: 2 (real & imaginary parts)

- Output shape: [batch, rx_antennas, layers, subcarriers_out, symbols, 2]
  - batch: 1
  - rx_antennas: 4
  - layers: 4
  - subcarriers_out: 3276 (2x input subcarriers)
  - symbols: 2
  - last dim: 2 (real & imaginary parts)

The last dimension (size 2) represents the real and imaginary parts for complex-valued data.

## CI/CD Integration

The comprehensive testing pipeline is automated through the `test-in-cicd.sh` script:

```bash
# Run the full CI/CD test flow
./test-in-cicd.sh
```

You can customize the test parameters by setting environment variables:

```bash
# Customize test parameters
NUM_RES=1638 LAYERS=4 NUM_RX_ANT=4 SYMBOLS=2 PRECISION=fp16 ./test-in-cicd.sh
```

Available environment variables:
- `NUM_RES`: Number of resource elements/subcarriers (default: 1638)
- `COMB_SIZE`: Comb size parameter (default: 2) 
- `NUM_PRBS`: Number of physical resource blocks (default: 137, calculated as NUM_RES/12)
- `NUM_RX_ANT`: Number of receive antennas (default: 4)
- `BATCH_SIZE`: Batch size (default: 1)
- `LAYERS`: Number of layers (default: 4)
- `SYMBOLS`: Number of symbols (default: 2)
- `PRECISION`: Precision for TensorRT export (default: fp16)

## Testing Scripts

The framework consists of the following key scripts:

### 1. Run PyTorch Model

`run_pytorch_model.py` - Runs the PyTorch version of EnhancedFusedChannelEstimator and generates reference outputs.

### 2. Export to ONNX

`export_to_onnx.py` - Exports the model to ONNX format and verifies that outputs match the PyTorch reference.

### 3. Export to TensorRT

`export_to_trt.py` - Converts the ONNX model to a TensorRT engine with timestamp in the filename.

### 4. Generate YAML Configuration

`generate_yaml_for_engine.py` - Creates a YAML configuration file for the TensorRT engine to be used by the channel estimator.

### 5. Test TensorRT Standalone

`test_trt_standalone.py` - Tests the TensorRT-based channel estimator in standalone mode.

### 6. Test TensorRT in PUSCH RX Pipeline

`test_trt_pusch_rx.py` - Tests the TensorRT-based channel estimator in the full PUSCH RX pipeline.

## Output Files

The test scripts generate output files in the `cicd_output` directory:

- PyTorch model outputs: `pytorch_output.npy`, `pytorch_output_input.npy`
- ONNX model: `model_YYYYMMDD_HHMMSS.onnx` and outputs: `onnx_output.npy`
- TensorRT engine: `enhanced_fused_channel_estimator_YYYYMMDD_HHMMSS.engine`
- YAML configuration: `chest_trt.yaml`
- Standalone test output: `standalone_output.npy`
- PUSCH RX test results in the output directory

## Known Issues and Troubleshooting

### Expected Warnings and Errors

- **CRC check failures**: During the PUSCH RX pipeline test, CRC check failures are expected and do not indicate an issue with the TensorRT channel estimator. This is due to the synthetic test data and is normal behavior.

- **ONNX output differences**: Some numerical differences between PyTorch and ONNX outputs are expected due to operator implementations. As long as the test completes successfully, these differences are acceptable.

- **CUDA device reset warning**: The warning "CUDA device reset failed: module 'cupy_backends.cuda.api.runtime' has no attribute 'deviceReset'" occurs during the PUSCH RX pipeline test and can be safely ignored. This is a known limitation in the interaction between CuPy and CUDA when attempting to reset the device.

### General Troubleshooting

If you encounter issues with TensorRT execution, check:

1. Engine file exists and is readable
2. YAML file has the correct path to the engine file (absolute path is recommended)
3. Input dimensions match the expected dimensions in the model
4. CUDA device is available and has sufficient memory

For ONNX runtime errors, verify:

1. ONNX model is valid (can be checked with `onnx.checker.check_model`)
2. Input shapes match the model's expected input

For PyTorch model errors:

1. Ensure all model parameters and input dimensions are correct
2. Check CUDA device availability and memory

If unexpected errors occur with tensor handling in the algorithm base classes, ensure you're using the latest version of the `ml_algorithm.py` file which supports multiple tensor types including NumPy, PyTorch, and CuPY arrays.
