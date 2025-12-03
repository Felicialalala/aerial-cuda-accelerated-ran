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

function [Y,H_true,TV,PAR] = TV_generation_main(H)

%inputs:
%H --> TF channel for a large number of UEs. Dim: Nf x Nt x L_BS x 12

%outputs:
%Y --> signal received by BS. Dim: Nf x Nt x L_BS
%H_true --> interpolation target. Dim: Nf_a x Nt x L_BS x num_UE
%TV --> test vector parameters

%%
%PARAMETERS

%modulation parameters:
PAR.mod.Nf = size(H,1); %number of subcarriers in subframe
PAR.mod.Nt = size(H,2); %number of OFDM symbols in subframe
PAR.mod.df = 15*10^3; %subcarrier spacing
PAR.mod.dt = 1 / PAR.mod.df; %OFDM symbol duration
PAR.mod.L_BS = size(H,3); %number of antennas at the BS

%cluster parameters:
PAR.cluster.num_PRB = 8; %number of PRBs per cluster
PAR.cluster.num_clusters = PAR.mod.Nf / (PAR.cluster.num_PRB*12); %total number of clusters
PAR.cluster.Nf_a = PAR.cluster.num_PRB*12; %number of subcarriers in each allocation

%pilot parameters:
PAR.pilot.Nf_p = PAR.cluster.num_PRB*4; %number of frequency pilot per allocation
PAR.pilot.Nt_p = 4; %number of time pilots per allocation
PAR.pilot.mux = 12; %number of pilots muxed in same allocation

%simulation parameters:
PAR.sim.num_UE = PAR.pilot.mux*PAR.cluster.num_clusters; %total number of UEs
PAR.sim.N0 = 10^(-1.5); %noise variance

%MMSE parameters:
PAR.MMSE.delay_spread = 6*10^(-6); %apriori delay spread (s)
PAR.MMSE.Doppler_spread = 222; %apriori Doppler spread (Hz)
PAR.MMSE.N0 = 10^(-1.5); %apriori noise variance (linear)

%visualize
PAR.visualize = 1;

%%
%PILOTS

%build basic pilot grids:
PAR = build_pilot_grid(PAR);

%build basic pilot signals:
PAR = build_pilot_signals(PAR);

%%
%INDICES

%allocation indices:
IND = build_allocation_indices(PAR);

%DMRS allocations:
IND = build_DMRS_indices(IND,PAR);

%%
%RECEIVED SIGNAL

[Y,S,OCC_F,OCC_T] = compute_received_signal(H,PAR,IND);

%%
%FILTERS

W_freq = compute_1D_freq_interpolator(PAR);
W_time = compute_1D_time_interpolator(PAR);

[W_freq_collection,W_time_collection] = choose_UE_interp_filters(W_freq,W_time,PAR);


%%
%TRUTH

H_true = extract_true_channel(H,IND,PAR);

%%
%PARAMETERS

TV = extract_TV_parameters(W_freq_collection,W_time_collection,IND,PAR);
