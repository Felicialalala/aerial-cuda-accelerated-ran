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

"""Tests for trt_engine.py."""
import itertools
import pathlib
import pytest
import os

import cupy as cp
import numpy as np
from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.trt import engine_from_bytes
from polygraphy.backend.trt.runner import TrtRunner

from aerial.phy5g.algorithms import TrtEngine
from aerial.phy5g.algorithms import TrtTensorPrms

# Test cases.
model_configs = [
    {
        'name': 'LLRNet-fixed-batch',
        'max_batch_size': 42588,
        'batch_size': 42588,
        'model_file': os.path.join(os.environ.get('HOME'), 'models/llrnet.trt'),
        'inputs': [TrtTensorPrms('input', (2,), np.float32)],
        'outputs': [TrtTensorPrms('dense_1', (8,), np.float32)]
    },
    {
        'name': 'LLRNet-dynamic-batch',
        'max_batch_size': 42588,
        'batch_size': 12345,
        'model_file': os.path.join(os.environ.get('HOME'), 'models/llrnet.trt'),
        'inputs': [TrtTensorPrms('input', (2,), np.float32)],
        'outputs': [TrtTensorPrms('dense_1', (8,), np.float32)]
    },
    {
        'name': 'NeuralRx',
        'model_file': os.path.join(os.environ.get('HOME'), 'models/neural_rx.trt'),
        'max_batch_size': 1,
        'batch_size': 1,
        'inputs': [TrtTensorPrms('rx_slot_real', (3276, 12, 4), np.float32),
                   TrtTensorPrms('rx_slot_imag', (3276, 12, 4), np.float32),
                   TrtTensorPrms('h_hat_real', (4914, 1, 4), np.float32),
                   TrtTensorPrms('h_hat_imag', (4914, 1, 4), np.float32),
                   TrtTensorPrms('active_dmrs_ports', (1,), np.float32),
                   TrtTensorPrms('dmrs_ofdm_pos', (3,), np.int32),
                   TrtTensorPrms('dmrs_subcarrier_pos', (6,), np.int32)],
        'outputs': [TrtTensorPrms('output_1', (2, 1, 3276, 12), np.float32),
                    TrtTensorPrms('output_2', (1, 3276, 12, 8), np.float32)]
    }
]

all_cases = list(itertools.product(model_configs, [True, False]))


@pytest.mark.parametrize(
    "model_config, h2d",
    all_cases,
    ids=[f"{config['name']} - cuPy: {h2d}" for config, h2d in all_cases]
)
def test_trt_engine(cuda_stream, model_config, h2d):
    """Test TrtEngine."""
    model_path = str(pathlib.Path(model_config["model_file"]).resolve())

    trt_engine = TrtEngine(
        trt_model_file=model_path,
        max_batch_size=model_config["max_batch_size"],
        input_tensors=model_config["inputs"],
        output_tensors=model_config["outputs"],
        cuda_stream=cuda_stream
    )

    ref_engine = engine_from_bytes(bytes_from_path(model_path))

    # Run multiple times to check that all the buffers get set correctly.
    num_batches = 2
    for _ in range(num_batches):

        # Generate random input.
        feed_dict = {}
        ref_feed_dict = {}
        for input_prms in model_config["inputs"]:
            shape = (model_config["batch_size"],) + input_prms.dims
            input_tensor = np.random.randn(*shape).astype(input_prms.data_type)
            if h2d:
                feed_dict[input_prms.name] = cp.array(input_tensor, order='F')
            else:
                feed_dict[input_prms.name] = input_tensor
            ref_feed_dict[input_prms.name] = input_tensor

        output_tensors = trt_engine.run(feed_dict)

        # Reference output using the polygraphy package.
        with TrtRunner(ref_engine) as ref_trt_runner:
            ref_output = ref_trt_runner.infer(ref_feed_dict)

            for name, tensor in output_tensors.items():
                if h2d:
                    tensor = tensor.get(order='F')
                assert np.allclose(ref_output[name], tensor)
