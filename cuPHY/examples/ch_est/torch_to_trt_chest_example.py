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

""" End to end channel estimation 

Purpose of this script:
    - To take a model working in PyAerial and compile it into a TrT Engine that 
    can run in cuPHY

Before using this script: 
    - Ensure the model to be compiled acceps inputs from the PyAerial
    Least Squares Channel Estimator (algo=3) and outputs an estimate with the
    same shape as the PyAerial MMSE Estimator (algo=1). 
    - INPUT:  [batch, subcarriers, layers, rx antennas, symbols, 2]
    - OUTPUT: [batch, rx antennas, layers, subcarriers, symbols, 2]
    - (the output should have 2x more subcarriers than the input)

Instructions to use this script: 
    - Copy the model definition in Pytorch or Tensorflow to this script
      We show an example with a Pytorch model. To use Tensorflow, see notes.
    - Load the trained model weights
    - Select the type of inputs desired:
        - Test Vectors used for cuPHY validation (<use_tvs> = True)
        - Channel Generated with Sionna and estimated with PyAerial (<use_tvs> = False)
        - (optionally, if any other input & output can be saved and used with 
           the model, as long as the dimensions match the input and output)
    - Provide the model as an argument to the check_results() function
      Note: this function should not be modified because it emulates the steps
      in cuPHY. 
    - Compare the MSE of the model provided with the MSE of LS. This should give
      an indication of whether the model is working properly. The results for 
      the example model (depending on the input type) are the following:
        - using TVs, the MSEs obtained for LS and the model are:
            LS= -7.6 dB ; ML: -14.1 dB (tv = 7201)
            LS= -7.6 dB ; ML: -13.4 dB (tv = 7217)
        - using Sionna channels
            LS= -20.0 dB ; ML = -24.8 dB
    - Later in the script the model should be exported to ONNX and evaluated again.
      Results should match the previous numbers.
    - Lastly, the model is compiled to TrT using polygraphy. The results should
      again match the ones previously obtained

Notes:
    - This script can be executed end-to-end without modification and the 
    provided values should appear in the respective steps
    - Any pre or post processing applied to the input data should be included
      inside the model. This limits the operations allowed in the model: 
      - Supported operations when exporting to ONNX (PyTorch):
        https://pytorch.org/docs/stable/onnx_torchscript.html
      - Supported operations when exporting to ONNX (Tensorflow):
        https://github.com/onnx/tensorflow-onnx/blob/main/README.md
        Note further that "The current implementation only works for graphs that do not contain any
        control flow or embedding related ops." as described in
        tensorflow github: tensorflow/python/framework/convert_to_constants.py#L1149
      - Supported operations when compiling to TRT:
        https://github.com/onnx/onnx-tensorrt/blob/main/docs/operators.md
      - If the model cannot be adjusted to work without a given forbidden 
        operation, then a workaround worth considering is to use a plugin:
        https://github.com/Alexey-Kamenev/tensorrt-dft-plugins
    - The batch dimension is used to stack the estimations of multiple users
    - To use a Tensorflow model instead of Pytorch, the main difference 
    comes in exporting the model to ONNX. This can be done with tf2onnx:
    https://github.com/onnx/tensorflow-onnx (available in PyPI)

MD5 sums:
    - channel_estimator.onnx: 64af9b805c9c7e7d831a93dbb4a646ad (repeatable!)
    - channel_estimator_fp16-True_tf32-True.engine: 2170dd84c2e64470b3f221ca6a310ef3 (not repeatable)

Dependencies information:
    - mamba install numpy pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
    - pip install ipykernel polygraphy onnx nvidia-pyindex nvidia-tensorrt
    - Ensure that the container version of TRT matches Python's version:
    pip install tensorrt==10.3 onnx2torch
"""

#%% Imports

import os
import h5py
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def db(v):
    return 10 * np.log10(v)

def read_h5_mat(mat, s, order='C'):
    a = mat[s][()]
    b = np.array([item[0] + 1j*item[1] for item in a.flatten()], order=order)
    return b.reshape(a.shape)

# Net selection params
SNR = 20.
num_prbs = 273
batch_size = 32
num_res = num_prbs * 12

folder = '' # change to '' to send to Jakub!


#%% Torch models

dev = torch.device(f'cuda')
torch.set_default_device(dev)

class ResidualBlock(nn.Module):
    def __init__(self, num_conv_channels, num_res, dilation):
        super(ResidualBlock, self).__init__()
        self._norm1 = nn.LayerNorm((num_conv_channels, num_res))
        self._conv1 = nn.Conv1d(in_channels=num_conv_channels,
                                out_channels=num_conv_channels,
                                kernel_size=3,
                                padding=dilation,
                                dilation=dilation,
                                bias=False)
        self._norm2 = nn.LayerNorm((num_conv_channels, num_res))
        self._conv2 = nn.Conv1d(in_channels=num_conv_channels,
                                out_channels=num_conv_channels,
                                kernel_size=3,
                                padding=dilation,
                                dilation=dilation,
                                bias=False)

    def forward(self, inputs):
        z = self._conv1(inputs)
        z = self._norm1(z)
        z = F.relu(z)
        z = self._conv2(z)
        z = self._norm2(z)
        z = z + inputs # skip connection
        z = F.relu(z)

        return z


class ChannelEstimator(nn.Module):
    def __init__(self, num_res: int, num_conv_channels: int = 32):
        """
        - num_res = number of subcarriers with reference signals
        - num_conv_channels = number of internal convolutional channels 
          (impacts model complexity, not input/output size)
        """
        super(ChannelEstimator, self).__init__()
        self.num_res = num_res

        # Input convolution
        self.input_conv = nn.Sequential(
                nn.Conv1d(in_channels=2, out_channels=num_conv_channels, 
                          kernel_size=3, padding=1, bias=False),
                nn.ReLU())

        # Residual blocks
        self.res_block_1 = ResidualBlock(num_conv_channels, num_res, dilation=1)
        self.res_block_2 = ResidualBlock(num_conv_channels, num_res, dilation=3)

        # Output convolution        
        self.output_conv = nn.Conv1d(in_channels=num_conv_channels,
                                     out_channels=2,
                                     kernel_size=3,
                                     padding=1,
                                     bias=False)
        

    def forward(self, z):
        # z: [batch size, num subcarriers, 2 (re & im)]
        z = z.permute(0, 2, 1)

        # z: [batch size, 2, num_res]
        z = self.input_conv(z)
        # z: [batch size, num_conv_channels, num_res]

        # Residual blocks
        z = self.res_block_1(z)
        z = self.res_block_2(z)
        # z: [batch size, No. conv. channels, num_res]

        z = self.output_conv(z)
        # z: [batch size, 2 * freq_interp_factor, num_res]

        z = z.permute(0, 2, 1)
        # z:  [batch size, num_res, 2]

        return z 


class FusedChannelEstimator(nn.Module):
    def __init__(self, num_res: int):
        super(FusedChannelEstimator, self).__init__()
        self.num_res = num_res
        self.m1 = ChannelEstimator(num_res).to(dev)
        self.m2 = ChannelEstimator(num_res).to(dev)

    def forward(self, z):
        
        # Input: [batch, subcarriers, layers, rx antennas, symbol, 2 (real & imag)]
        n_batch, n_layers, n_ant, n_symb = z.shape[0], z.shape[2], z.shape[3], z.shape[4]
        
        # Swapaxes - Needed to make the LS have the same order from MMSE
        # (LS iterates over antennas first, MMSE iterates over subcarriers first)
        z = z.swapaxes(1,3)

        # Model-dependent reshaping
        z = z.swapaxes(3,4).reshape((-1, self.num_res, 2))

        # z: [batch size, num subcarriers, 2 (re & im)]
        scale_factors = torch.sqrt(torch.sum(z**2, dim=-1)).max(axis=1)[0][..., None, None]
        z /= scale_factors

        z1 = self.m1(z) # z:  [batch size, num_res, 2]
        z2 = self.m2(z) # z:  [batch size, num_res, 2]
        
        zout = torch.stack((z1, z2), dim=2).reshape((z1.shape[0], -1, 2))
        # z:  [batch size, 2*num_res, 2 (re & im)]

        zout *= scale_factors
        
        zout = zout.reshape((n_batch, n_ant, n_layers, n_symb, 2*self.num_res, 2)).swapaxes(4,3)
        # Output: [batch, rx antennas, layers, DMRS subcarriers, symbol , 2 (real & imag)]

        return zout
    

class ComplexMSELoss(nn.Module):
    def __init__(self):
        super(ComplexMSELoss, self).__init__()
    
    def forward(self, output: torch.complex64, target: torch.complex64):
        diff = output - target
        den = torch.linalg.norm((target * torch.conj(target)).mean())
        return torch.linalg.norm((diff * torch.conj(diff)).mean()) / den


#%% Test

# Load model weights
model_path = folder + f'model_SNR={SNR:.1f}_3.path'
model_fused = FusedChannelEstimator(num_res//2)
model_fused.load_state_dict(torch.load(model_path, weights_only=True))

criterion = ComplexMSELoss()

# Load channels

tv = 7217  # [7201, 7217, 7218, 7219] -> 1,2,3,4 dmrs symbols
use_TVs = True # True:  Loads LS and MMSE estimates from TVs
               # False: Loads LS and ground truth channels from generated 
               #        with Sionna (in ChEst notebook)

if use_TVs:
    with h5py.File(folder + f'TVnr_{tv}_PUSCH_gNB_CUPHY_s0p0.h5', 'r+') as tv_mat:
        # [TV 7201] LS: (4, 1, 1638)    & MMSE: (3276, 1, 4)
        # [TV 7217] LS: (2, 4, 1, 1638) & MMSE: (2, 3276, 1, 4)
        # [TV 7218] LS: (3, 4, 1, 1638) & MMSE: (3, 3276, 1, 4)
        # [TV 7219] LS: (4, 4, 1, 1638) & MMSE: (4, 3276, 1, 4)

        ls_chs_old   = read_h5_mat(tv_mat, 'reference_ChEst_w_delay_est_H_LS_est_save0', order='C')
        mmse_chs_old = read_h5_mat(tv_mat, 'reference_H_est0', order='C')

    if len(ls_chs_old.shape) == 3: # add symbol dimension
        ls_chs_old, mmse_chs_old = ls_chs_old[None,...], mmse_chs_old[None,...]
    
    if len(ls_chs_old.shape) == 4: # add batch dimension
        ls_chs_old, mmse_chs_old = ls_chs_old[None,...], mmse_chs_old[None,...]

    n_batch, n_symb, n_ant, n_layers = tuple(ls_chs_old.shape[:-1])
    
    # Copy data in the right order
    ls_chs = np.zeros((n_batch, num_res//2, n_layers,   n_ant, n_symb), dtype=np.complex64)
    new_mmse = np.zeros((n_batch, n_ant, n_layers,   num_res, n_symb), dtype=np.complex64)
    n_res = num_res//2
    for b_i in range(n_batch):
        for s_i in range(n_symb):
            for l_i in range(n_layers):
                for a_i in range(n_ant):
                    for sub_i in range(n_res):
                        ls_chs[b_i, sub_i, l_i, a_i, s_i] = ls_chs_old[b_i, s_i, a_i, l_i, sub_i]
                        new_mmse[b_i, a_i, l_i, sub_i, s_i] = mmse_chs_old[b_i, s_i, sub_i, l_i, a_i]
                        new_mmse[b_i, a_i, l_i, sub_i + n_res, s_i] = mmse_chs_old[b_i, s_i, sub_i + n_res, l_i, a_i]

    true_chs_ls = new_mmse[..., ::2, :].swapaxes(1, 3)
    out_chs = new_mmse
    
else:
    
    ls_chs = np.load(folder + 'ls_chs.npy')[..., ::2] # Shape: (32, 273 * 6)
    true_chs = np.load(folder + 'true_chs.npy')       # Shape: (32, 273 * 12)

    # Concat several so enough exist in the batch dimensions to allow reshaping 
    # and match any configs
    ls_chs_b = np.concatenate((ls_chs, ls_chs, ls_chs, ls_chs), axis=0)
    true_chs_b = np.concatenate((true_chs, true_chs, true_chs, true_chs), axis=0)

    # Example configs:
    n_symb = 2   # num of dmrs symbols in 1 slot
    n_ant = 4    # number of rx antennas
    n_layers = 4 # number of layers
    n_dim_needed = n_layers * n_ant * n_symb
    assert n_dim_needed <= ls_chs_b.shape[0] # If this breaks, stack more channels. 

    ls_chs_t, true_chs_t = ls_chs_b[:n_dim_needed, :], true_chs_b[:n_dim_needed, :]

    n_batch = 1
    ls_chs = np.zeros((n_batch, num_res//2, n_layers,   n_ant, n_symb), dtype=np.complex64)
    true_chs_ls = np.zeros((n_batch, num_res//2, n_layers,   n_ant, n_symb), dtype=np.complex64)
    out_chs = np.zeros((n_batch, n_ant, n_layers,   num_res, n_symb), dtype=np.complex64)
    
    # Copy data in the right order
    n_res = num_res//2
    for s_i in range(n_symb):
        for l_i in range(n_layers):
            for a_i in range(n_ant):
                b_i = s_i + l_i * n_symb + a_i * n_symb * n_layers
                
                for sub_i in range(n_res):
                    ls_chs[0, sub_i, l_i, a_i, s_i] = ls_chs_t[b_i, sub_i]
                    true_chs_ls[0, sub_i, l_i, a_i, s_i] = true_chs_t[b_i, 2*sub_i]

                    out_chs[0, a_i, l_i, sub_i, s_i] = true_chs_t[b_i, sub_i]
                    out_chs[0, a_i, l_i, sub_i + n_res, s_i] = true_chs_t[b_i, sub_i + n_res]


def check_results(model=model_fused, ls=ls_chs, out=out_chs, ls_true = true_chs_ls):
    h, h_ls, h_ls_gt = torch.tensor(out), torch.tensor(ls), torch.tensor(ls_true)

    # (Prefer stack instead of view_as_real to force a tensor copy, and decouple from ls)
    inputs_r = torch.stack((h_ls.real, h_ls.imag), dim=-1).type(torch.cuda.FloatTensor) # complex -> real

    outputs_r = model(inputs_r) # model with pre & post processing (batching)

    h_hat = torch.view_as_complex(outputs_r) # real -> complex

    ml_loss = criterion(h_hat.to(dev), h).item()
    ls_loss = criterion(h_ls, h_ls_gt).item()

    # MSE loss versus MMSE if use_TVs else versus ground truth
    print(f'Avg. LS MSE loss: {db(ls_loss):.1f} dB')
    print(f'Avg. ML MSE loss: {db(ml_loss):.1f} dB')
    
    return inputs_r, outputs_r


model_fused.eval()
with torch.no_grad():
    inputs, outputs = check_results()

# Write outputs (in case they are necessary for comparing with cuPHY dumps)
filename = "ml_results.h5"
with h5py.File(filename, "w") as data_file:
    data_file.create_dataset("ml_est", data=outputs.cpu().numpy())

print(f'Outputs written to "ml_est" inside: {filename}')


#%% Export to ONNX & Evaluate
import onnx
import onnx2torch

# DO NOT CHANGE THE CONSTANTS NAMES (they match what cuPHY expects)
MODEL_INPUT_NAME = 'z'
MODEL_OUTPUT_NAME = 'zout'
ONNX_MODEL_PATH = "channel_estimator.onnx"

torch.onnx.export(
    model_fused,     # model to export
    (inputs,),       # inputs of the model,
    ONNX_MODEL_PATH, # filename of the ONNX model
    input_names=[MODEL_INPUT_NAME],        # Rename inputs for the ONNX model
    output_names=[MODEL_OUTPUT_NAME],
)

# Load ONNX model
onnx_model = onnx.load(ONNX_MODEL_PATH)

# Convert ONNX to Torch
torch_model = onnx2torch.convert(onnx_model).to(dev)

# Evaluate
with torch.no_grad():
    _, _ = check_results(torch_model)


#%% Compile to TRT & Evaluate

from polygraphy.backend.trt import (
    CreateConfig,
    EngineFromNetwork,
    NetworkFromOnnxPath,
    SaveEngine,
    TrtRunner,
)

model_fused.eval()
with torch.no_grad():
    inputs, outputs = check_results()

def build_trt_engine(config, output_file):
    build_engine = EngineFromNetwork(
            NetworkFromOnnxPath(ONNX_MODEL_PATH), config=CreateConfig(**config)
    )

    build_engine = SaveEngine(build_engine, path=output_file)

    def f(z):
        with TrtRunner(build_engine) as runner:
            res = runner.infer(feed_dict={MODEL_INPUT_NAME: z})
            return res[MODEL_OUTPUT_NAME]

    return f

configs = [
    {"fp16": True, "tf32": True},
    # {"fp16": False, "tf32": True},
    # {"fp16": False, "tf32": False},
    # {"bf16": True, "tf32": True},
]

for config in configs:
    suffix = "_".join(f"{k}-{v}" for k, v in config.items())
    output_file = f"channel_estimator_{suffix}_NEW.engine"
    print(f"building TRT engine with config: {config}")
    trt_model = build_trt_engine(config, output_file=output_file)
    check_results(trt_model)
    os.system(
        # graph mode OFF
        # f"trtexec --useSpinWait --loadEngine={output_file} 2>&1 | grep \"\\[I\\] GPU Compute Time\""
        # graph mode ON
        f"trtexec --useSpinWait --useCudaGraph --loadEngine={output_file} 2>&1 | grep \"\\[I\\] GPU Compute Time\"" # +PERF!
    )

# %%
