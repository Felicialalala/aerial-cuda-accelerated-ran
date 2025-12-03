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

function IND = build_allocation_indices(PAR)

%computes the allocation sizes for each UE and their indices

%outputs:
%IND.allocation.A_size --> size (in subcarriers) of each UEs
%allocation. Dim: num_UE x 1

%IND.allocation.A_index --> indices of each UEs subcarrier allocations. 
%Dim: num_UE x Nf_a. 



%%
%PARAMETERS

%modulation parameters:
Nf = PAR.mod.Nf; %number of subcarriers in subframe

%cluster parameters:
Nf_a = PAR.cluster.Nf_a; %number of frequency pilots per allocation
num_clusters = PAR.cluster.num_clusters; %total number of clusters

%pilot parameters:
mux = PAR.pilot.mux; %number of pilots muxed in same allocation

%simulation parameters:
num_UE = PAR.sim.num_UE; %total number of UEs

%%
%ALLOCATION SIZE

%gives the number of subcarriers in each UEs allocation:
A_size = ones(num_UE,1)*Nf_a;

%%
%ALLOCATION INDICES

A_index = zeros(num_UE,Nf_a);

count = 0;

for c = 1 : num_clusters
    c_index = (c - 1)*Nf_a + 1 : c*Nf_a;
    
    for m = 1 : mux
        count = count + 1;
        A_index(count, : ) = c_index;
    end
end

%%
%WRAP

IND.allocation.A_size = A_size;
IND.allocation.A_index = A_index;















