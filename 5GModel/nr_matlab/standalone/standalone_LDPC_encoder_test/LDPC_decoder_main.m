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

function TbCbs_est = LDPC_decoder_main(LLR,PuschCfg)

%function applies LDPC decoder to recieved LLRs

%inputs:
%LLR        --> recieved LLRs
%TbCbs_est  --> estimates of transmited data bits

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
nV_sym = TannerPar.nV_sym;              % number of systematic (data) variable nodes
nV = nV_sym + nV_parity;                % total number of variable nodes (data + parity)
nC = nV_parity;                         % number of pairty check equations

%%
%SETUP

%add punctured bits:
%LLR = [zeros(2*Zc,C) ; LLR];
%add punctured bits when initial bits not punctured
LLR = LLR;

%lifting reshape:
LLR = reshape(LLR,Zc,nV,C);

%%
%START

TbCbs_est = zeros(Zc,nV,C);

for c = 1 : C
    TbCbs_est(:,:,c) = MSA_layering(LLR(:,:,c),nC,Zc,10,TannerPar);
end

%%
%OUTPUT

if BGN == 1
    TbCbs_est = TbCbs_est(:,1:22,:);
    TbCbs_est = reshape(TbCbs_est,22*Zc,C);
else
    TbCbs_est = TbCbs_est(:,1:10,:);
    TbCbs_est = reshape(TbCbs_est,10*Zc,C);
end

    




