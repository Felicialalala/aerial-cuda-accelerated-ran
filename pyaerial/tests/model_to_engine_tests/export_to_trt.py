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

import os
import argparse
from aerial.model_to_engine.model.enhanced_channel_estimator import (
    EnhancedFusedChannelEstimator
)
from aerial.model_to_engine.exporters.tensorrt_exporter import TensorRTExporter


def main():
    parser = argparse.ArgumentParser(
        description="Export EnhancedFusedChannelEstimator to TensorRT"
    )
    parser.add_argument('--num_res', type=int, default=612,
                        help='Number of resource elements')
    parser.add_argument('--comb_size', type=int, default=2,
                        help='Comb size (2 or 4)')
    parser.add_argument('--do_fft', action='store_true', default=True,
                        help='Use FFT in the model')
    parser.add_argument('--output_dir', type=str,
                        default='test_outputs/trt_export',
                        help='Directory to save all outputs')
    parser.add_argument('--onnx_path', type=str, default=None,
                        help='Path to existing ONNX model (optional)')
    parser.add_argument('--precision', type=str, default='fp16',
                        choices=['fp32', 'fp16', 'tf32', 'int8', 'mixed'],
                        help='Precision for TensorRT engine')
    parser.add_argument('--use_api_direct', action='store_true',
                        help='Use TensorRT API directly instead of Polygraphy')
    parser.add_argument(
        '--engine_filename',
        type=str,
        default='enhanced_fused_channel_estimator.engine',
        help='Filename for the TensorRT engine'
    )
    args = parser.parse_args()

    # Ensure output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    # Create model
    print(f"Creating EnhancedFusedChannelEstimator model with num_res="
          f"{args.num_res}, comb_size={args.comb_size}, do_fft={args.do_fft}")
    model = EnhancedFusedChannelEstimator(
        num_res=args.num_res,
        comb_size=args.comb_size,
        do_fft=args.do_fft,
        reshape=True
    )
    model.eval()

    # Use TensorRTExporter to export the model
    engine_path = os.path.join(args.output_dir, args.engine_filename)

    exporter = TensorRTExporter(model)
    result = exporter.export(
        output_path=str(engine_path),  # Cast to str to fix type error
        precision=args.precision,
        onnx_path=args.onnx_path,
        use_polygraphy=not args.use_api_direct
    )

    # Print export results
    print("\n--- TensorRT Export Results ---")
    print(f"Format: {result.get('format', 'unknown')}")
    print(f"Path: {result.get('path', 'unknown')}")
    print(f"Export time: {result.get('export_time', 0):.2f} seconds")
    print(f"Precision: {result.get('precision', 'unknown')}")
    print(f"Status: {result.get('status', 'unknown')}")

    if result.get('status') == 'success':
        print("\nExport successful! Engine saved to:")
        print(f"  {engine_path}")

        if 'benchmarks' in result and result['benchmarks']:
            print("\n--- Benchmark Results ---")
            benchmarks = result['benchmarks']
            for key, value in benchmarks.items():
                if value is not None:
                    print(f"{key}: {value}")
    else:
        print(f"\nExport failed: {result.get('message', 'unknown error')}")

    return 0


if __name__ == "__main__":
    main()
