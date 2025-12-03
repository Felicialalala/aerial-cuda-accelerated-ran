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

function TannerPar = load_Tanner(BGN,i_LS,Zc)

%function loads user's Tanner graph. 

%follows 38.212 section 5.2.2

%inputs:
%BGN                           --> base graph number
%i_LS                          --> lifting set index
%Zc                            --> lifting size

%outputs:
%TannerPar.nC                  --> number of check nodes
%TannerPar.nV_sym              --> number of systematic (data) variable nodes
%TannerPar.nNeighbors          --> For each check node, gives number connected variable nodes. Dim: numC x 1
%TannerPar.NeighborIdx         --> For each check node, lists indicies of connected variable nodes. Dim: numC x 1 (cell)
%TannerPar.NeighborShift       --> For each check node, lists expansion cyclic shift used by each connected variable node. Dim: numC x 1 (cell)

%%
%LOAD TABLE

if BGN == 1
    load('Tanner_BG1.mat'); %tables 5.3.2-2
    nC = 46;
    nV_sym = 22;
else
    load('Tanner_BG2.mat'); %tables 5.3.2-3
    nC = 42;
    nV_sym = 10;
end

%%
%PERMUATIONS

%select permutations based on lifting set

switch i_LS
    case 1 
        NeighborShift = NeighborPermutations_LS1;
    case 2
        NeighborShift = NeighborPermutations_LS2;
    case 3
        NeighborShift = NeighborPermutations_LS3;
    case 4
        NeighborShift = NeighborPermutations_LS4;
    case 5
        NeighborShift = NeighborPermutations_LS5;
    case 6
        NeighborShift = NeighborPermutations_LS6;
    case 7 
        NeighborShift = NeighborPermutations_LS7;
    case 8
        NeighborShift = NeighborPermutations_LS8;
end

% modify permutations based on lifting size
for c = 1 : nC
    for i = 1 : numNeighbors(c)
        NeighborShift{c}(i) = mod(NeighborShift{c}(i),Zc);
    end
end


%%
%WRAP

TannerPar.nV_sym = nV_sym;
TannerPar.nNeighbors = numNeighbors;
TannerPar.NeighborIdx = NeighborIndicies;
TannerPar.NeighborShift = NeighborShift;
       
