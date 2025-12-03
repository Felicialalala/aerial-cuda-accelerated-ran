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
import numpy as np
import torch
import argparse
from aerial.model_to_engine.model.enhanced_channel_estimator import EnhancedFusedChannelEstimator


def main():
    parser = argparse.ArgumentParser(description="Run PyTorch EnhancedFusedChannelEstimator model")
    parser.add_argument('--num_res', type=int, default=612, help='Number of resource elements')
    parser.add_argument('--comb_size', type=int, default=2, help='Comb size (2 or 4)')
    parser.add_argument('--do_fft', action='store_true', default=True, help='Use FFT in the model')
    parser.add_argument('--output', type=str, default='pytorch_output.npy', help='Output file path')
    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='Input numpy file (if not provided, random data will be generated)'
    )
    parser.add_argument('--batch', type=int, default=1, help='Batch size')
    parser.add_argument('--layers', type=int, default=1, help='Number of layers')
    parser.add_argument('--rx_antennas', type=int, default=4, help='Number of rx antennas')
    parser.add_argument('--symbols', type=int, default=2, help='Number of symbols')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on (cuda or cpu)')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Instantiate model
    print(
        f"Creating EnhancedFusedChannelEstimator model "
        f"(num_res={args.num_res}, comb_size={args.comb_size}, "
        f"do_fft={args.do_fft})"
    )
    model = EnhancedFusedChannelEstimator(
        num_res=args.num_res,
        comb_size=args.comb_size,
        do_fft=args.do_fft,
        reshape=True
    )

    # Move model to device
    device = torch.device(args.device)
    model.to(device)
    model.eval()

    # Prepare input data
    if args.input and os.path.exists(args.input):
        print(f"Loading input data from {args.input}")
        input_data = np.load(args.input)
        input_tensor = torch.tensor(input_data, device=device)
    else:
        print("Generating random input data")
        input_shape = (args.batch, args.num_res, args.layers, args.rx_antennas, args.symbols, 2)
        input_data = np.random.randn(*input_shape).astype(np.float32)
        # Save the input data for later use
        input_file = os.path.splitext(args.output)[0] + '_input.npy'
        np.save(input_file, input_data)
        print(f"Random input data saved to {input_file}")
        input_tensor = torch.tensor(input_data, device=device)

    print(f"Input shape: {input_tensor.shape}")

    # Run model inference
    print("Running model inference...")
    with torch.no_grad():
        output = model(input_tensor)

    # Convert output to numpy and save
    output_np = output["zout"].cpu().numpy()
    print(f"Output shape: {output_np.shape}")

    # Save the output for comparison
    np.save(args.output, output_np)
    print(f"Output saved to {args.output}")

    # Print some statistics
    print(f"Input mean: {input_data.mean():.6f}, std: {input_data.std():.6f}")
    print(f"Output mean: {output_np.mean():.6f}, std: {output_np.std():.6f}")

    return 0


if __name__ == "__main__":
    main()
