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

function TbCodedCbs = LDPC_encode_main(TbCbs,PuschCfg)

%function applies NR LDPC encoder

%inputs:
%TbCbs --> transport block segmented into code blocks. Dim: (nV_sym*Zc) x C

%outputs:
%TbCodedCbs --> fully encoded code blocks with puncturing. Dim: ((nV - 2)*Zc) x C


%%
%PARAMATERS

%coding paramaters:
Zc = PuschCfg.coding.Zc;                % lifting size
C = PuschCfg.coding.C;                  % number of codeblocks
nV_parity = PuschCfg.coding.nV_parity;  % number of desired parity nodes
BGN = PuschCfg.coding.BGN;              % 1 or 2. Indicates which base graph used
i_LS = PuschCfg.coding.i_LS;            % lifting set index

%tanner graph paramaters:
TannerPar = load_Tanner(BGN,i_LS,Zc);
nV_sym = TannerPar.nV_sym;             % number of systematic (data) variable nodes
nV = nV_sym + nV_parity;               % total number of variable nodes (data + parity)

%%
%SETUP

%convert bits to lifting format:
TbCbs = reshape(TbCbs,Zc,nV_sym,C);

%embed systematic bits into codeblocks:
TbCodedCbs = zeros(Zc,nV,C);
TbCodedCbs(:,1 : nV_sym,:) = TbCbs;

%%
%ENCODE BLOCKS

%loop over different codewords:
for c = 1 : C
    
    %first, compute core pairty checks:
    coreChecks = compute_core_checks(TbCodedCbs(:,:,c),Zc,TannerPar);
    
    %second, compute core paity bits:
    TbCodedCbs(:,nV_sym + 1 : nV_sym + 4, c) = ...
        compute_core_pairity(coreChecks,BGN,i_LS,Zc);
    
    %third, compute extended pairity bits:
    TbCodedCbs(:,nV_sym + 5 : end, c) = ...
        compute_ext_pairity(TbCodedCbs(:,:,c),nV_parity,Zc,TannerPar);
    
end


%%
%RESHAPE/PUNTURE

%reshape
TbCodedCbs = reshape(TbCodedCbs, Zc*nV,C);

%puncture first 2*Zc bits:
%TbCodedCbs(1 : 2*Zc,:) = [];





    
    
    

















