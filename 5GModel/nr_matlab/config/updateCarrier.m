% SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
% SPDX-License-Identifier: Apache-2.0
%
% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
%
% http://www.apache.org/licenses/LICENSE-2.0
%
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.

function carrier = updateCarrier(carrier)
% function SysPar = updateSysPar(SysPar)
%
% This function derives all the other parameters based on the pre-set
% configurations
%
% Input:    SysPar: structure with all simulation configurations
%
% Output:   SysPar: structure with all simulation configurations
%

global SimCtrl

carrier.T_c = 1/(480e3*4096);       % NR basic sampling interval
carrier.T_s = 1/(15e3*2048);        % LTE 20M sampling interval
carrier.k_const = 64;               % T_s/T_c
carrier.N_sc_RB = 12;               % number of subcarrier per RB
carrier.N_subframe = 10;            % number of subframe per frame
carrier.T_subframe = 1e-3;          % duration of a subframe

% read from input SysPar
mu = carrier.mu;
start_mu = carrier.N_grid_start_mu;
k_const = carrier.k_const;
N_sc_RB = carrier.N_sc_RB;
N_RB = carrier.N_grid_size_mu;
T_subframe = carrier.T_subframe;
CpType = carrier.CpType;
carrierFreq = carrier.carrierFreq;

% derive frame structure parameters
[N_symb_slot, N_slot_frame_mu, N_slot_subframe_mu] ...
    = readFrameStructureTable(mu, CpType);

delta_f = 15e3*2^mu;        % data channel subcarrier spacing (Hz)
N_sc = N_sc_RB*N_RB;        % # of used subcarriers for data channel
Nfft = 2^ceil(log2(N_sc));  % FFT size for data channel
f_samp = delta_f*Nfft;      % sampling rate
T_samp = 1/f_samp;          % sampling interval
T_slot = T_subframe/N_slot_subframe_mu; % duration of a slot
N_samp_slot = T_slot*f_samp;            % # of samples in a slot
N_samp_subframe = T_subframe*f_samp;            % # of samples in a slot
N_u_mu = 2048*k_const*2^(-mu);          % # of samples in unit of T_c

% write to output SysPar
carrier.N_symb_slot = N_symb_slot;
carrier.N_slot_frame_mu = N_slot_frame_mu;
carrier.N_slot_subframe_mu = N_slot_subframe_mu;
carrier.N_symb_subframe_mu = N_symb_slot*N_slot_subframe_mu;
carrier.N_u_mu = N_u_mu;
carrier.delta_f = delta_f;
carrier.N_sc = N_sc;
carrier.Nfft = Nfft;
carrier.f_samp = f_samp;
carrier.T_samp = T_samp;
carrier.N_samp_slot = N_samp_slot;
carrier.N_samp_subframe = N_samp_subframe;

if carrierFreq <= 6
    carrier.FR = 1;
else
    carrier.FR = 2;
end
carrier.BW = ceil((N_RB*delta_f*N_sc_RB)/1e6/5)*5;

carrier.mu0 = mu; % force m0 = mu
carrier.N_grid_size_mu0 = N_RB;       % grid size in PRB
carrier.N_grid_start_mu0 = start_mu;       % ???

carrier.N_BWP_start = carrier.N_grid_start_mu;            % inital UL BW part in PRB
carrier.N_BWP_size = carrier.N_grid_size_mu; 

N_grid_start_mu = carrier.N_grid_start_mu;
N_grid_size_mu = carrier.N_grid_size_mu;
N_grid_start_mu0 = carrier.N_grid_start_mu0;
N_grid_size_mu0 = carrier.N_grid_size_mu0;
mu0 = carrier.mu0;

carrier.k0_mu = (N_grid_start_mu + N_grid_size_mu/2)*N_sc_RB - ...
    (N_grid_start_mu0 + N_grid_size_mu0/2)*N_sc_RB*2^(mu0 - mu);

if isfield(carrier, 'Nant_gNB')
    if SimCtrl.enable_dynamic_BF == 0
        carrier.N_FhPort_DL = carrier.Nant_gNB;
        carrier.N_FhPort_UL = carrier.Nant_gNB;
    end
end

return


