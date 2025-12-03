# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This module defines the TensorRTExporter class, responsible for converting
ONNX models to optimized TensorRT engines. It supports conversion using
the direct TensorRT Python API. It handles precision settings (FP16, TF32),
ONNX parsing, optimization profiles for dynamic shapes, and includes
engine benchmarking using trtexec.
"""

import os

import time
import subprocess
from typing import Dict, Any, Optional
import logging

from ..algorithm_base import MLAlgorithm
from .onnx_exporter import ONNXExporter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TensorRTExporter:
    """Exporter for TensorRT format."""

    def __init__(self, algorithm: MLAlgorithm):
        """
        Initialize TensorRT exporter.

        Args:
            algorithm: The ML algorithm to export
        """
        self.algorithm = algorithm

    def export(self, output_path: str,
               precision: str = "fp16",
               onnx_path: Optional[str] = None,
               **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Export the model to TensorRT format.

        Args:
            output_path: Path to save the TensorRT engine
            precision: Precision mode ("fp32", "fp16", "tf32", "int8")
            onnx_path: Path to an existing ONNX model, if available
            **kwargs: Additional parameters for TensorRT conversion

        Returns:
            Dictionary with export results
        """
        if not output_path.endswith('.engine'):
            output_path += '.engine'

        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Export to ONNX first if needed
        if onnx_path is None:
            onnx_path = os.path.splitext(output_path)[0] + '.onnx'
            onnx_exporter = ONNXExporter(self.algorithm)
            # onnx_result is unused, so just call export
            onnx_exporter.export(onnx_path, **kwargs.get('onnx', {}))
            logger.info(f"Exported ONNX model to {onnx_path}")

        # Start timing
        start_time = time.time()

        # Use direct TensorRT API
        result = self._convert_with_trt_api(
            onnx_path, output_path, precision
        )

        export_time = time.time() - start_time

        # Add common export info to result
        result.update({
            "format": "tensorrt",
            "path": output_path,
            "export_time": export_time,
            "precision": precision,
        })

        # Benchmarking
        benchmark_results = self._benchmark_engine(output_path)
        if benchmark_results:
            result["benchmarks"] = benchmark_results

        return result

    def _convert_with_trt_api(self, onnx_path: str,
                              output_path: str,
                              precision: str) -> Dict[str, Any]:
        """
        Convert ONNX model to TensorRT engine using TensorRT API directly.

        Args:
            onnx_path: Path to the ONNX model
            output_path: Path to save the TensorRT engine
            precision: Precision mode

        Returns:
            Dictionary with conversion results
        """
        try:
            import tensorrt as trt  # type: ignore

            logger.info(f"Converting {onnx_path} to TensorRT engine with "
                        f"{precision} precision using TensorRT API")
            logger.info(f"TensorRT version: {trt.__version__}")

            # Create TensorRT logger
            trt_logger = trt.Logger(trt.Logger.INFO)

            # Create builder and network
            builder = trt.Builder(trt_logger)
            network_flags = (
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            network = builder.create_network(network_flags)

            # Parse ONNX model
            parser = trt.OnnxParser(network, trt_logger)
            with open(onnx_path, 'rb') as model_file:
                if not parser.parse(model_file.read()):
                    for error in range(parser.num_errors):
                        logger.error("ONNX parsing error: "
                                     f"{parser.get_error(error)}")
                    raise RuntimeError("Failed to parse ONNX model")

            # Create builder config
            builder_config = builder.create_builder_config()

            # Set precision flags
            builder_config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)

            if precision == "bf16":
                builder_config.set_flag(trt.BuilderFlag.BF16)
            if precision in ["fp16", "mixed", "int8"]:
                builder_config.set_flag(trt.BuilderFlag.FP16)
            if precision in ["tf32", "mixed"]:
                builder_config.set_flag(trt.BuilderFlag.TF32)

            # Create optimization profile for dynamic dimensions
            profile = builder.create_optimization_profile()

            # Set input shape constraints based on algorithm specs
            input_specs = self.algorithm.get_input_specs()
            for spec in input_specs:
                if spec.is_dynamic:
                    # Create shape for min, opt, max dimensions
                    min_shape = []
                    opt_shape = []
                    max_shape = []

                    for i, dim in enumerate(spec.shape):
                        if isinstance(dim, str):
                            # Dynamic dimension
                            if i in spec.dynamic_axes:
                                min_val = 1  # Default min
                                opt_val = 1  # Default optimal
                                max_val = 4  # Default max
                            else:
                                # Constant dimension with a string name
                                min_val = opt_val = max_val = 1
                        else:
                            # Fixed dimension
                            min_val = opt_val = max_val = dim

                        min_shape.append(min_val)
                        opt_shape.append(opt_val)
                        max_shape.append(max_val)

                    profile.set_shape(
                        spec.name, min_shape, opt_shape, max_shape
                    )

            # Add profile to config
            builder_config.add_optimization_profile(profile)

            # Set workspace size (3 GB)
            builder_config.set_memory_pool_limit(
                trt.MemoryPoolType.WORKSPACE, 3 << 30
            )

            # Build and serialize engine
            serialized_engine = builder.build_serialized_network(
                network, builder_config
            )
            if serialized_engine is None:
                raise RuntimeError("Failed to build TensorRT engine")

            # Save engine to file
            with open(output_path, 'wb') as f:
                f.write(serialized_engine)

            # Verify the engine was saved
            if os.path.exists(output_path):
                logger.info(f"Engine file saved to: {output_path}")
                logger.info(f"Engine file size: "
                            f"{os.path.getsize(output_path)} bytes")
            else:
                logger.error(f"Failed to save engine to {output_path}")
                return {"status": "error", "message": "Failed to save engine"}

            return {"status": "success",
                    "message": "Conversion successful using TensorRT API"}

        except Exception as e:
            logger.error(f"Error in TensorRT API conversion: {str(e)}")
            return {"status": "error", "message": str(e)}

    def _benchmark_engine(self, engine_path: str) -> Dict[str, Any]:  # noqa: C901, E501
        """
        Benchmark TensorRT engine using trtexec.

        Args:
            engine_path: Path to the TensorRT engine

        Returns:
            Dictionary with benchmark results
        """
        if not os.path.exists(engine_path):
            logger.error(f"Engine file not found at: {engine_path}")
            return {"status": "error", "message": "Engine file not found"}

        try:
            # Get input shapes from algorithm specs
            input_specs = self.algorithm.get_input_specs()
            input_shapes = {}

            for spec in input_specs:
                shape = []
                for dim in spec.shape:
                    if isinstance(dim, str):
                        # Use 1 for dynamic dimensions in benchmark
                        shape.append(1)
                    else:
                        shape.append(dim)

                # Format shape for trtexec (e.g., "input:1x3x224x224")
                input_shapes[spec.name] = "x".join(map(str, shape))

            # Construct trtexec command
            shapes_param = " ".join(
                [f"--shapes={name}:{shape}"
                 for name, shape in input_shapes.items()]
            )

            command = (
                f"trtexec --useSpinWait --useCudaGraph "
                f"--loadEngine={engine_path} "
                f"{shapes_param} "
                f"--iterations=100 --warmUp=20"
            )

            logger.info(f"Running benchmark: {command}")

            # Execute command
            result = subprocess.run(
                command, shell=True, text=True, capture_output=True
            )

            # Process results
            if result.returncode == 0:
                # Extract performance metrics
                metrics = {
                    "gpu_compute_time": "",
                    "perfs_per_second": "",
                    "min_latency": "",
                    "max_latency": "",
                    "mean_latency": ""
                }

                for line in result.stdout.split('\n'):
                    if "GPU Compute Time" in line:
                        parts = line.split("=")
                        if len(parts) > 1:
                            metrics["gpu_compute_time"] = parts[1].strip()

                    elif "Throughput" in line:
                        parts = line.split("=")
                        if len(parts) > 1:
                            # Extract only the numeric part
                            throughput_str = parts[1].strip().split()[0]
                            metrics["perfs_per_second"] = throughput_str

                    elif "Latency" in line:
                        if "min" in line:
                            parts = line.split("=")
                            if len(parts) > 1:
                                metrics["min_latency"] = parts[1].strip()
                        elif "max" in line:
                            parts = line.split("=")
                            if len(parts) > 1:
                                metrics["max_latency"] = parts[1].strip()
                        elif "mean" in line:
                            parts = line.split("=")
                            if len(parts) > 1:
                                metrics["mean_latency"] = parts[1].strip()

                return metrics
            else:
                logger.error(f"trtexec failed with error: {result.stderr}")
                return {"status": "error",
                        "message": "trtexec benchmark failed"}

        except Exception as e:
            logger.error(f"Error in benchmarking: {str(e)}")
            return {"status": "error", "message": str(e)}
