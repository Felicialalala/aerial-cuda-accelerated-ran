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
import numpy as np
import cupy as cp

from aerial.phy5g.algorithms import ChannelEstimator
from aerial.phy5g.config import PuschConfig, PuschUeConfig


def safe_to_numpy(array):
    """
    Safely convert a CuPy array to NumPy array. If already NumPy, return as is.

    Args:
        array: Either a CuPy or NumPy array

    Returns:
        NumPy array
    """
    if isinstance(array, cp.ndarray):
        return array.get()
    return array


def safe_to_numpy_with_order(array, order='C'):
    """
    Safely convert a CuPy array to NumPy array with specified order.
    If already NumPy, return as is.

    Args:
        array: Either a CuPy or NumPy array
        order: Memory order ('C' or 'F')

    Returns:
        NumPy array
    """
    if isinstance(array, cp.ndarray):
        return array.get(order=order)
    return array


def create_dummy_pusch_config(num_prbs=51, num_symbols=14, dmrs_positions=None):
    """
    Create a dummy PUSCH configuration for testing.

    Args:
        num_prbs: Number of PRBs
        num_symbols: Number of symbols
        dmrs_positions: List indicating DMRS positions (1 for DMRS, 0 for data)

    Returns:
        PuschConfig object with test parameters
    """
    if dmrs_positions is None:
        # Default: DMRS on symbols 2 and 11 (index starting from 0)
        dmrs_positions = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]

    # Create PUSCH UE config
    ue_config = PuschUeConfig(
        rnti=1,
        layers=1,
        dmrs_ports=2,  # DMRS port 0 used (bitmap: 0010)
        data_scid=0,
        scid=0,
        mcs_index=20,  # Mid-range MCS
        mcs_table=0    # Table 1
    )

    # Create main PUSCH config
    pusch_config = PuschConfig(
        ue_configs=[ue_config],
        start_prb=0,
        num_prbs=num_prbs,
        start_sym=0,
        num_symbols=num_symbols,
        dmrs_syms=dmrs_positions,
        dmrs_add_ln_pos=0,
        dmrs_max_len=1,
        dmrs_scrm_id=0,
        num_dmrs_cdm_grps_no_data=2
    )

    return pusch_config


def create_random_rx_slot(num_prbs, num_symbols, num_rx_ant, h2d=True):
    """
    Create random received slot data for testing.

    Args:
        num_prbs: Number of PRBs
        num_symbols: Number of symbols
        num_rx_ant: Number of receive antennas
        h2d: Whether to transfer data to GPU (CuPy)

    Returns:
        Array with random RX slot data
    """
    # Generate random complex data
    # Shape: [num_symbols, num_prbs*12, num_rx_ant]
    num_subcarriers = num_prbs * 12
    rx_slot_np = np.random.randn(num_symbols, num_subcarriers, num_rx_ant) + \
        1j * np.random.randn(num_symbols, num_subcarriers, num_rx_ant)
    rx_slot_np = rx_slot_np.astype(np.complex64)

    if h2d:
        # Transfer to GPU
        rx_slot = cp.array(rx_slot_np, order='F')
    else:
        rx_slot = rx_slot_np

    return rx_slot


def main():
    parser = argparse.ArgumentParser(
        description="Test TensorRT-based channel estimator in standalone mode"
    )
    parser.add_argument('--yaml', type=str, required=True,
                        help='Path to chest_trt.yaml with TensorRT '
                             'engine configuration')
    parser.add_argument('--num_prbs', type=int, default=51,
                        help='Number of PRBs (resource blocks)')
    parser.add_argument('--num_rx_ant', type=int, default=4,
                        help='Number of receive antennas')
    parser.add_argument('--h2d', action='store_true', default=True,
                        help='Run test with data on GPU (CuPy)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file to save channel estimates')
    args = parser.parse_args()

    # Verify YAML file exists
    if not os.path.exists(args.yaml):
        print(f"Error: YAML file not found at {args.yaml}")
        return 1

    print(f"Testing TensorRT-based channel estimator with YAML: {args.yaml}")

    # Create dummy PUSCH configuration
    num_symbols = 14
    # DMRS on symbols 2 and 11
    dmrs_positions = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    pusch_config = create_dummy_pusch_config(
        num_prbs=args.num_prbs,
        num_symbols=num_symbols,
        dmrs_positions=dmrs_positions
    )

    # Count DMRS symbols for verification
    dmrs_syms_count = sum(dmrs_positions)
    print(f"Number of DMRS symbols: {dmrs_syms_count}")

    # Create random RX slot data
    rx_slot = create_random_rx_slot(
        num_prbs=args.num_prbs,
        num_symbols=num_symbols,
        num_rx_ant=args.num_rx_ant,
        h2d=args.h2d
    )
    print(f"RX slot shape: {rx_slot.shape}")

    # Create ChannelEstimator with TensorRT configuration
    print("Creating ChannelEstimator with TensorRT configuration")
    channel_estimator = ChannelEstimator(
        num_rx_ant=args.num_rx_ant,
        ch_est_algo=3,  # LS channel estimation
        chest_factory_settings_filename=args.yaml
    )

    # Estimate channel
    print("Running channel estimation...")
    slot = 0  # Slot number doesn't matter for this test
    ch_est = channel_estimator.estimate(
        rx_slot=rx_slot,
        slot=slot,
        pusch_configs=[pusch_config]
    )

    # Get results and verify
    if args.h2d:
        # Transfer results back to CPU if needed
        ch_est_cpu = []
        for est in ch_est:
            ch_est_cpu.append(safe_to_numpy_with_order(est, order='F'))
    else:
        ch_est_cpu = ch_est

    # Print output information
    for ue_grp_idx, est in enumerate(ch_est_cpu):
        print(f"UE group {ue_grp_idx} channel estimate shape: {est.shape}")
        print(f"Mean magnitude: {np.abs(est).mean():.6f}")
        print(f"Max magnitude: {np.abs(est).max():.6f}")

    # Save output if requested
    if args.output:
        # Convert to numpy if not already
        ch_est_cpu = [safe_to_numpy(est) for est in ch_est_cpu]

        # Save the first UE group's channel estimate
        np.save(args.output, ch_est_cpu[0])
        print(f"Channel estimate saved to {args.output}")

    print("Test completed successfully!")
    return 0


if __name__ == "__main__":
    main()
