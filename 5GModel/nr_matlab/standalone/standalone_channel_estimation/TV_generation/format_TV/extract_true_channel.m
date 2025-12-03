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

function H_true = extract_true_channel(H,IND,PAR)

%function extracts the true channel for each UE (i.e. what we want the
%output of interpolation to be)

%inputs:
%H --> TF channel for a large number of UEs. Dim: Nf x Nt x L_BS x 12

%outputs:
%H_true --> extracted TF channel. Dim: Nf_a x Nt x L_BS x num_UE

%%
%PARAMETERS

%modulation parameters:
Nt = PAR.mod.Nt; %number of OFDM symbols in subframe
L_BS = PAR.mod.L_BS; %number of antennas at the BS

%cluster parameters:
num_clusters = PAR.cluster.num_clusters; %total number of clusters
Nf_a = PAR.cluster.Nf_a; %number of subcarriers in each allocation

%simulation parameters:
num_UE = PAR.sim.num_UE; %total number of UEs

%allocation indices:
A_index = IND.allocation.A_index;

%%
%START

H_true = zeros(Nf_a,Nt,L_BS,num_UE);

ue = 0;
for c = 1 : num_clusters
    little_count = 0;
    
    for s = 1 : 3
        for OCC_t = 1 : 2
            for OCC_f = 1 : 2
                ue = ue + 1;
                little_count = little_count + 1;
                H_true(:,:,:,ue) = H(A_index(ue,:),:,:,little_count);
            end
        end
    end
end


                
                




