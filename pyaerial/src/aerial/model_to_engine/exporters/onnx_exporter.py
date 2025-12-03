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
This module provides the ONNXExporter class, responsible for converting
PyTorch models (specifically those derived from MLAlgorithm) into the ONNX
(Open Neural Network Exchange) format. It handles dynamic axes, input/output
specifications, and includes verification using onnxruntime if available.
"""

import os
import torch
import onnx
import numpy as np
import time
from typing import Dict, Any, List, Optional

from ..algorithm_base import MLAlgorithm


class ONNXExporter:
    """Exporter for ONNX format."""

    def __init__(self, algorithm: MLAlgorithm):
        """
        Initialize ONNX exporter.

        Args:
            algorithm: The ML algorithm to export
        """
        self.algorithm = algorithm

    def export(self, output_path: str,  # noqa: C901
               opset_version: int = 12,
               dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None
               ) -> Dict[str, Any]:
        """
        Export the model to ONNX format.

        Args:
            output_path: Path to save the ONNX model
            opset_version: ONNX opset version
            dynamic_axes: Dictionary of dynamic axes for ONNX export

        Returns:
            Dictionary with export results
        """
        if not output_path.endswith('.onnx'):
            output_path += '.onnx'

        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Get model and example inputs
        model = self.algorithm.get_model()
        model.eval()  # Set to evaluation mode

        # Get example input for tracing
        example_inputs = self.algorithm.get_example_inputs()

        # Get input/output names from specs
        input_specs = self.algorithm.get_input_specs()
        output_specs = self.algorithm.get_output_specs()

        input_names = [spec.name for spec in input_specs]
        output_names = [spec.name for spec in output_specs]

        # Prepare inputs in the right format
        example_inputs_list = []
        for name in input_names:
            if name in example_inputs:
                inp = example_inputs[name]
                # Convert numpy to torch if needed
                if isinstance(inp, np.ndarray):
                    if np.iscomplexobj(inp):
                        # Convert complex to real/imag pairs
                        inp_real = torch.tensor(
                            np.real(inp), dtype=torch.float32
                        )
                        inp_imag = torch.tensor(
                            np.imag(inp), dtype=torch.float32
                        )
                        inp_tensor = torch.stack([inp_real, inp_imag], dim=-1)
                    else:
                        inp_tensor = torch.tensor(inp, dtype=torch.float32)
                    example_inputs_list.append(inp_tensor)
                else:
                    # Already a torch tensor
                    example_inputs_list.append(inp)

        # Prepare dynamic axes if not provided
        if dynamic_axes is None:
            dynamic_axes_dict = {}  # type: Dict[str, Dict[int, str]]
            # Add dynamic axes from input specs
            for spec in input_specs:
                if spec.is_dynamic:
                    dynamic_axes_dict[spec.name] = {}  # Initialize empty dict
                    for name, idx in spec.dynamic_axes.items():
                        dynamic_axes_dict[spec.name][idx] = name
            # Add dynamic axes from output specs
            for spec in output_specs:
                if spec.is_dynamic:
                    dynamic_axes_dict[spec.name] = {}  # Initialize empty dict
                    for name, idx in spec.dynamic_axes.items():
                        dynamic_axes_dict[spec.name][idx] = name

            dynamic_axes = dynamic_axes_dict

        # Export to ONNX
        start_time = time.time()

        # Use a dummy forward method that handles both dict and tensor outputs
        class ExportWrapper(torch.nn.Module):
            """Wrapper class for PyTorch model export to ONNX.

            This wrapper handles both dictionary and tensor outputs from the model,
            making the export process consistent regardless of the model's
            output format.
            """
            def __init__(self, algorithm: MLAlgorithm):
                """Initialize the wrapper.

                Args:
                    algorithm: The ML algorithm to wrap for ONNX export
                """
                super().__init__()
                self.algorithm = algorithm
                self.model = algorithm.get_model()

            def forward(self, *args: torch.Tensor) -> tuple:  # type: ignore
                """Forward pass for the wrapped model.

                Args:
                    *args: Input tensors to pass to the model

                Returns:
                    Tuple of output tensors in order specified by output_names
                """
                with torch.no_grad():
                    outputs = self.model(*args)

                    # If outputs is a dictionary, extract values in order
                    if isinstance(outputs, dict):
                        outputs_list = tuple(
                            outputs[output_name]
                            for output_name in output_names
                        )
                        return outputs_list
                    # If single output, wrap in tuple
                    elif not isinstance(outputs, tuple):
                        return (outputs,)
                    # Already a tuple
                    return outputs

        wrapper = ExportWrapper(self.algorithm)

        torch.onnx.export(
            wrapper,
            tuple(example_inputs_list),
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
            verbose=False
        )
        export_time = time.time() - start_time

        # Verify the exported model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)

        # Verify model via onnxruntime if available
        try:
            # Keep import inside try block as onnxruntime is optional
            import onnxruntime as ort
            verification_results = self._verify_onnx_model(
                output_path, example_inputs_list, model, ort
            )
        except ImportError:
            verification_results = {
                "status": "onnxruntime not available for verification"
            }

        return {
            "format": "onnx",
            "path": output_path,
            "export_time": export_time,
            "opset_version": opset_version,
            "input_names": input_names,
            "output_names": output_names,
            "dynamic_axes": dynamic_axes,
            "verification_results": verification_results
        }

    def _verify_onnx_model(self, onnx_path: str,
                           input_data: List[torch.Tensor],
                           pytorch_model: torch.nn.Module,
                           ort_module: Any) -> Dict[str, Any]:
        """
        Verify ONNX model produces the same outputs as the PyTorch model.

        Args:
            onnx_path: Path to the ONNX model
            input_data: Input data for verification
            pytorch_model: Original PyTorch model for reference output
            ort_module: The imported onnxruntime module

        Returns:
            Dictionary with verification results
        """

        # Get reference output from PyTorch model
        with torch.no_grad():
            reference_outputs = pytorch_model(*input_data)

        # Create ONNX runtime session
        ort_session = ort_module.InferenceSession(
            onnx_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )

        # Prepare inputs
        ort_inputs = {}
        input_specs = self.algorithm.get_input_specs()
        for i, name in enumerate([spec.name for spec in input_specs]):
            ort_inputs[name] = input_data[i].cpu().numpy()

        # Run ONNX model
        ort_outputs = ort_session.run(None, ort_inputs)

        # Compare outputs
        verification_results = {}
        output_specs = self.algorithm.get_output_specs()
        output_names = [spec.name for spec in output_specs]

        if isinstance(reference_outputs, tuple):
            # Handle tuple output (order matters)
            for i, (ref, ort_out) in enumerate(
                    zip(reference_outputs, ort_outputs)):
                name = output_names[i] if i < len(output_names) \
                    else f"output_{i}"
                error = np.abs(ref.cpu().numpy() - ort_out).mean()
                verification_results[name] = {
                    "mean_error": float(error),
                    "valid": error < 1e-5
                }
        elif isinstance(reference_outputs, dict):
            # Handle dictionary output (use names from spec)
            for i, name in enumerate(output_names):
                ref = reference_outputs[name]
                ort_out = ort_outputs[i]
                if isinstance(ref, torch.Tensor):
                    ref_np = ref.cpu().numpy()
                else:
                    ref_np = ref
                error = np.abs(ref_np - ort_out).mean()
                verification_results[name] = {
                    "mean_error": float(error),
                    "valid": error < 1e-5
                }
        else:
            # Handle single tensor output
            name = output_names[0] if output_names else "output"
            if isinstance(reference_outputs, torch.Tensor):
                ref_np = reference_outputs.cpu().numpy()
            else:
                ref_np = reference_outputs
            error = np.abs(ref_np - ort_outputs[0]).mean()
            verification_results[name] = {
                "mean_error": float(error),
                "valid": error < 1e-5
            }

        return verification_results
