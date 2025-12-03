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

function [TbCodedCbs,CleanCW] = LDPC_encode(TbCbs, C, K, F, BGN, i_LS, Zc, pdschTable)

%function applies NR LDPC encoder

%inputs:
%TbCbs --> transport block segmented into code blocks. Dim: K x C

%outputs:
%TbCodedCbs --> fully encoded code blocks with puncturing
%CleanCW    --> fully encoded codeblocks without puncturing

%PARAMATERS

%coding paramaters:
% Zc = coding.Zc;      %lifting size
% C = coding.C;        %number of codeblocks
% K = coding.K;  %number of systematic bits per codeblock
% F = coding.F;  %number of filler bits per codeblock

%tanner graph paramaters:
TannerPar = load_Tanner(BGN, i_LS, Zc, pdschTable);
nV = TannerPar.nV;       %number of variable nodes
nV_sym = TannerPar.nV_sym; %number of systematic variable nodes

%SETUP

%convert bits to lifting format:
TbCbs = reshape(TbCbs,Zc,nV_sym,C);

%set filler bits to zero:
TbCbs(TbCbs == -1) = 0;

%embed systematic bits into codeblocks:
TbCodedCbs = zeros(Zc,nV,C);
TbCodedCbs(:,1 : nV_sym,:) = TbCbs;

%ENCODE BLOCKS

for c = 1 : C
    
    %first, compute core pairity bits:
    TbCodedCbs(:,nV_sym + 1 : nV_sym + 4, c) = ...
        compute_core_pairity(TbCodedCbs(:,:,c),TannerPar,BGN, Zc, i_LS);
    
    %next, compute extended pairity bits:
    TbCodedCbs(:,nV_sym + 5 : end, c) = ...
        compute_ext_pairity(TbCodedCbs(:,:,c),TannerPar,Zc);
    
end

%convert to binary/reshape:
TbCodedCbs = mod(TbCodedCbs,2);
TbCodedCbs = reshape(TbCodedCbs, Zc*nV,C);

%FILLER/PUNCHER

%save the "clean" CWs (filler bits = 0 and no puncturing)
CleanCW = TbCodedCbs;

%set filler bits back to -1 (indicates they are not to be transmitted)
TbCodedCbs(K - F + 1 : K,:) = -1;

%puncture first 2*Zc bits:
TbCodedCbs(1 : 2*Zc,:) = [];

return
