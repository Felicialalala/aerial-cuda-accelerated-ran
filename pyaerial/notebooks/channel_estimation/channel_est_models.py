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

"Models for PUSCH DMRS-based channel estimation with flexible comb size"
import torch
import torch.nn as nn
import torch.nn.functional as F


# Torch Utils
def roll_in_half(x):
    els_in_last_dim = x.shape[-1]
    return torch.roll(x, int(els_in_last_dim / 2.), dims=-1)


def unroll_in_half(x):
    els_in_last_dim = x.shape[-1]
    return torch.roll(x, -int(els_in_last_dim / 2.), dims=-1)


def sfft(x, roll=True):
    y = torch.fft.fft(x)
    if roll:
        y = roll_in_half(y)
    return y


def isfft(x, roll=True):
    if roll:
        x = unroll_in_half(x)
    return torch.fft.ifft(x)


class ResidualBlock(nn.Module):
    def __init__(self, num_conv_channels, num_res, dilation):
        super(ResidualBlock, self).__init__()
        self._conv1 = nn.Conv1d(in_channels=num_conv_channels,
                                out_channels=num_conv_channels,
                                kernel_size=3,
                                padding=dilation,
                                dilation=dilation,
                                bias=False)
        self._norm1 = nn.LayerNorm((num_conv_channels, num_res))
        self._conv2 = nn.Conv1d(in_channels=num_conv_channels,
                                out_channels=num_conv_channels,
                                kernel_size=3,
                                padding=dilation,
                                dilation=dilation,
                                bias=False)
        self._norm2 = nn.LayerNorm((num_conv_channels, num_res))

    def forward(self, inputs):
        z = self._conv1(inputs)
        z = self._norm1(z)
        z = F.relu(z)
        z = self._conv2(z)
        z = self._norm2(z)
        z = z + inputs  # skip connection
        z = F.relu(z)

        return z


class ChannelEstimator(nn.Module):
    def __init__(self, num_res: int, do_fft: bool = True, num_conv_channels: int = 32):
        """
        Performs 1:1 estimation. E.g. in DMRS, two of these models are used,
        respectively, to compute the even and the odd subcarriers
        - num_res = number of subcarriers with reference signals
        - num_conv_channels = number of internal convolutional channels
          (impacts model complexity, not input/output size)
        """
        super(ChannelEstimator, self).__init__()
        self.num_res = num_res
        self.do_fft = do_fft

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

        if self.do_fft:
            z = torch.view_as_real(sfft(torch.view_as_complex(z)))

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

        if self.do_fft:
            z = torch.view_as_real(isfft(torch.view_as_complex(z.contiguous())))

        return z


class FusedChannelEstimator(nn.Module):
    """
    Class that uses one or more 1-to-1 channel estimators
    """
    def __init__(self, num_res: int, comb_size: int = 2, do_fft: bool = True):
        super(FusedChannelEstimator, self).__init__()
        self.num_res = num_res  # number of input subcarriers
        self.comb_size = comb_size

        if self.comb_size not in [2, 4]:
            raise Exception('Comb size not supported. Choose 2 or 4.')

        self.m1 = ChannelEstimator(num_res, do_fft)
        self.m2 = ChannelEstimator(num_res, do_fft)
        if self.comb_size == 4:
            self.m3 = ChannelEstimator(num_res, do_fft)
            self.m4 = ChannelEstimator(num_res, do_fft)

    def forward(self, z):

        # Input: [batch, subcarriers, layers, rx antennas, symbol, 2 (real & imag)]
        n_batch, n_layers, n_ant, n_symb = z.shape[0], z.shape[2], z.shape[3], z.shape[4]

        # Swapaxes - Needed to make the LS have the same order from MMSE
        # (LS iterates over antennas first, MMSE iterates over subcarriers first)
        z = z.swapaxes(1, 3)

        # Model-dependent reshaping
        z = z.swapaxes(3, 4).reshape((-1, self.num_res, 2))

        # z: [batch size, num subcarriers, 2 (re & im)]
        scale_factors = torch.sqrt(torch.sum(torch.abs(z)**2,
                                             dim=-1)).max(axis=1)[0][..., None, None]
        z /= scale_factors

        z1 = self.m1(z)  # z:  [batch size, num_res, 2]
        z2 = self.m2(z)
        if self.comb_size == 2:
            z_to_stack = (z1, z2)
        else:  # comb_size == 4
            z3 = self.m3(z)
            z4 = self.m4(z)
            z_to_stack = (z1, z2, z3, z4)

        # Interleave subcarriers
        zout = torch.stack(z_to_stack, dim=2).reshape((z1.shape[0], -1, 2))
        # zout:  [batch size, comb_size*num_res, 2 (re & im)]

        zout *= scale_factors

        zout = zout.reshape((n_batch, n_ant, n_layers, n_symb,
                             self.comb_size * self.num_res, 2)).swapaxes(4, 3)
        # Output: [batch, rx antennas, layers, DMRS subcarriers, symbol , 2 (real & imag)]

        return zout


class ComplexMSELoss(nn.Module):
    def __init__(self):
        super(ComplexMSELoss, self).__init__()

    def forward(self, output: complex, target: complex):
        diff = output - target
        den = torch.linalg.norm((target * torch.conj(target)).mean())
        return torch.linalg.norm((diff * torch.conj(diff)).mean()) / den
