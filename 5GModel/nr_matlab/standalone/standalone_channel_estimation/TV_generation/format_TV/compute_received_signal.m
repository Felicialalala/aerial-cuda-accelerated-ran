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

function [Y,V,OCC_F,OCC_T] = compute_received_signal(H,PAR,IND)

%compute the signal received by BS

%inputs:
%H --> TF channel for a large number of UEs. Dim: Nf x Nt x L_BS x 12

%outputs:
%Y --> TF signal received by BS. Dim: Nf x Nt x L_BS

%%
%PARAMETERS

%modulation parameters:
Nf = PAR.mod.Nf; %number of subcarriers in subframe
Nt = PAR.mod.Nt; %number of OFDM symbols in subframe
L_BS = PAR.mod.L_BS; %number of antennas at the BS

%cluster parameters:
num_clusters = PAR.cluster.num_clusters; %total number of clusters
Nf_a = PAR.cluster.Nf_a; %number of subcarriers in each allocation

%pilot parameters:
P = PAR.pilot.P;

%allocation indices:
A_index = IND.allocation.A_index;% indices of each UEs subcarrier allocations. 

%DMRS indices:
DMRS_index_time = IND.DMRS.DMRS_index_time; %location of pilot symbols allocated to each UE. Dim: num_UE x Nt_p
DMRS_index_freq = IND.DMRS.DMRS_index_freq; %location of pilot subcarriers allocated to each UE. Dim: num_UE x Nf_p

%simulation parameters:
N0 = PAR.sim.N0; %noise variance
num_UE = PAR.sim.num_UE;

%%
%START

Y = zeros(Nf,Nt,L_BS);
ue = 0;

V = zeros(num_UE,1);
OCC_T = zeros(num_UE,1);
OCC_F = zeros(num_UE,1);


for c = 1 : num_clusters
    little_count = 0;
    
    for s = 1 : 3
        for OCC_t = 1 : 2
            for OCC_f = 1 : 2
                ue = ue + 1;
                little_count = little_count + 1;
                
                %generate rnd QPSK signal:
                S = (-ones(Nf,Nt)).^(round(rand(Nf,Nt)))+ 1i*(-ones(Nf,Nt)).^(round(rand(Nf,Nt)));
                S = zeros(size(S));
                
                %embed pilot signal:
                S(DMRS_index_freq(ue,:),DMRS_index_time(ue,:)) = P(:,:,OCC_f,OCC_t);
                
                %extract allocation:
                S = S(A_index(ue,:),:);
                
                %apply channel:
                for ant = 1 : L_BS
                    Y(A_index(ue,:),:,ant) = Y(A_index(ue,:),:,ant) + H(A_index(ue,:),:,ant,little_count) .* S;
                end
                
                V(ue) = s;
                OCC_T(ue) = OCC_t;
                OCC_F(ue) = OCC_f;
            end
        end
    end
end

Y = Y + sqrt(N0 / 2) * (randn(Nf,Nt,L_BS) + 1i*randn(Nf,Nt,L_BS));


              
                
                
                


