# Channel Estimator Model-to-Engine Framework

This framework provides a complete workflow for exporting PyTorch-based channel estimator models (specifically the EnhancedFusedChannelEstimator) to ONNX and TensorRT engines, generating YAML configurations for integration with the Aerial/cuBB stack, and validating the exported models through comprehensive testing.

## Directory Structure

```
pyaerial/
├── src/aerial/model_to_engine/       # Framework core implementation
│   ├── algorithm_base/               # Base classes for algorithm implementations
│   ├── exporters/                    # ONNX and TensorRT exporters
│   │   ├── onnx_exporter.py
│   │   └── tensorrt_exporter.py
│   ├── model/                        # PyTorch model implementations
│   │   └── enhanced_channel_estimator.py
│   ├── diagrams/                     # UML diagrams and documentation
│   └── to-remove/                    # Deprecated code (pending removal)
└── tests/model_to_engine_tests/      # Test scripts and CI/CD pipeline
    ├── cicd_output/                  # Output directory for CI/CD artifacts
    ├── run_pytorch_model.py          # Step 1: Run PyTorch model and save outputs
    ├── export_to_onnx.py             # Step 2: Export to ONNX and validate
    ├── export_to_trt.py              # Step 3: Export to TensorRT and validate
    ├── generate_yaml_for_engine.py   # Step 4: Generate YAML config for engine
    ├── test_trt_standalone.py        # Step 5: Test TRT engine in standalone mode
    ├── test_trt_pusch_rx.py          # Step 6: Test TRT engine in PUSCH RX pipeline
    └── test-in-cicd.sh               # Main CI/CD script running all steps
```

## Workflow Overview

The end-to-end workflow is orchestrated by the `test-in-cicd.sh` script, which runs the following steps:

1. **PyTorch Model Validation**: Run the EnhancedFusedChannelEstimator with random test data and save inputs/outputs
2. **ONNX Export and Validation**: Export the model to ONNX format and validate numerical equivalence
3. **TensorRT Engine Generation**: Convert ONNX to optimized TensorRT engine (FP16/TF32) using TensorRT API
4. **YAML Configuration Generation**: Generate a `chest_trt.yaml` file for the cuBB stack
5. **Standalone Engine Testing**: Test the TensorRT engine independently
6. **Full PUSCH RX Integration**: Test the engine as part of the complete PUSCH RX pipeline

Each step logs progress and validates against previous steps to ensure numerical equivalence throughout the pipeline.

## Usage

### Running the Complete CI/CD Pipeline

```bash
# Activate the venv
source /opt/nvidia/cuBB/.venv/bin/activate

# Run the complete pipeline
cd pyaerial/tests/model_to_engine_tests
./test-in-cicd.sh
```

The script accepts environment variables to configure tensor dimensions:
- `NUM_RES`: Number of resource elements (default: 1638)
- `COMB_SIZE`: Comb size (default: 2)
- `NUM_PRBS`: Number of PRBs (default: 137)
- `NUM_RX_ANT`: Number of receive antennas (default: 4)
- `BATCH_SIZE`: Batch size (default: 1)
- `LAYERS`: Number of layers (default: 4)
- `SYMBOLS`: Number of symbols (default: 2)
- `PRECISION`: Engine precision (default: fp16)

### Individual Component Usage

Each step can be run independently:

#### 1. Run PyTorch Model
```bash
python run_pytorch_model.py --num_res 1638 --comb_size 2 --do_fft --batch 1 --layers 4 \
  --rx_antennas 4 --symbols 2 --output output/pytorch_output.npy
```

#### 2. Export to ONNX
```bash
python export_to_onnx.py --num_res 1638 --comb_size 2 --do_fft \
  --input output/pytorch_output_input.npy --pytorch_output output/pytorch_output.npy \
  --onnx_path output/model_YYYYMMDD_HHMMSS.onnx --onnx_output output/onnx_output.npy
```

#### 3. Export to TensorRT
```bash
python export_to_trt.py --num_res 1638 --comb_size 2 --do_fft --output_dir output \
  --onnx_path output/model_YYYYMMDD_HHMMSS.onnx --precision fp16 \
  --engine_filename enhanced_fused_channel_estimator_YYYYMMDD_HHMMSS.engine --use_api_direct
```

#### 4. Generate YAML Configuration
```bash
python generate_yaml_for_engine.py --engine output/enhanced_fused_channel_estimator_YYYYMMDD_HHMMSS.engine \
  --yaml output/chest_trt.yaml
```

#### 5. Test in Standalone Mode
```bash
python test_trt_standalone.py --yaml output/chest_trt.yaml --num_prbs 137 --num_rx_ant 4 \
  --output output/standalone_output.npy
```

#### 6. Test in PUSCH RX Pipeline
```bash
python test_trt_pusch_rx.py --yaml output/chest_trt.yaml --num_prbs 137 --num_rx_ant 4 \
  --output output
```

## Implementation Architecture

The framework follows a modular design with specialized components:

- **ONNX Exporter**: Handles PyTorch to ONNX conversion with dynamic input shapes
- **TensorRT Exporter**: Converts ONNX to TensorRT engines using the TensorRT API
- **YAML Generator**: Creates configuration files compatible with the Aerial/cuBB stack
- **Standalone Tester**: Validates TensorRT engines in isolation
- **PUSCH RX Tester**: Validates TensorRT engines within the full PUSCH RX pipeline

## Success Criteria

The CI/CD pipeline is considered successful when:

1. Every step in the workflow completes without errors
2. Numerical outputs between PyTorch, ONNX, and TensorRT match within tolerance
3. The generated TensorRT engine works in both standalone and full-pipeline tests
4. The YAML configuration file is correctly generated

## Dependencies

- Python 3.8+
- PyTorch 2.0+
- ONNX and ONNX Runtime
- TensorRT 8.5+
- CUDA 12.0+
- NumPy
