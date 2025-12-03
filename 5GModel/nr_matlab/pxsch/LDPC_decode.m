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

function [TbCbs_est,numItr,badItrCnt] = LDPC_decode(LLR,nV_parity,puschTable, ...
    Zc, C, BGN, i_LS, K_prime, CRCstr, maxNumItr_CBs)

%function applies LDPC decoder to recieved LLRs

%inputs:
%LLR        --> recieved LLRs
%nV_parity  --> number of parity nodes
%maxItr     --> maximium number of iterations
%alpha      --> normalization constant

%outputs:
%TbCbs_est  --> ldpc estimate of transmited data bits
%numItr     --> number of iterations required. Dim: C x 1

%%
%PARAMATERS

%coding paramaters:
% Zc = PuschCfg.coding.Zc;                % lifting size
% C = PuschCfg.coding.C;                  % number of codeblocks
% BGN = PuschCfg.coding.BGN;              % 1 or 2. Indicates which base graph used
% i_LS = PuschCfg.coding.i_LS;            % lifting set index

global SimCtrl;
LDPCpar.earlyTerm = SimCtrl.alg.LDPC_enableEarlyTerm;      % option to enable early termination, 0 by default (disable)
LDPCpar.earlyTermAlg = SimCtrl.alg.LDPC_earlyTermAlg; % option of the early termination alg
LDPCpar.NBF_num_consecutive_itr = SimCtrl.alg.LDPC_earlyTerm_NBF_num_consecutive_itr;
LDPCpar.SAFEET_badItrThres = SimCtrl.alg.LDPC_earlyTerm_SAFEET_badItrThres;
LDPCpar.numCWs = C;
LDPCpar.K_prime = K_prime;
LDPCpar.puschTable = puschTable;
LDPCpar.BGN = BGN;
LDPCpar.CRCstr = CRCstr;
if length(maxNumItr_CBs)==1
    maxNumItr_CBs = maxNumItr_CBs*ones(1,C);
end

%tanner graph paramaters:
TannerPar = load_Tanner(BGN, i_LS, Zc, puschTable);
nV_sym = TannerPar.nV_sym;              % number of systematic (data) variable nodes
nV = nV_sym + nV_parity;                % total number of variable nodes (data + parity)
nC = nV_parity;                         % number of pairty check equations

%%
%SETUP


% select normalization constant:
LDPCpar.alpha = set_LDPC_normalization(nV_parity, BGN);

%add punctured bits:
LLR = [zeros(2*Zc,C) ; LLR(1 : Zc*(nV - 2),:)];

%lifting reshape:
LLR = reshape(LLR,Zc,nV,C);

%%
%START

TbCbs_est = zeros(Zc,nV,C);
numItr = zeros(C,1);
badItrCnt = zeros(C,1);

for c = 1 : C
    LDPCpar.maxItr = maxNumItr_CBs(c); 
    [TbCbs_est(:,:,c),numItr(c),badItrCnt(c)] = MSA_layering(LLR(:,:,c),nC,Zc,LDPCpar,TannerPar);
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

end


%%
%
function alpha = set_LDPC_normalization(nV_parity, BGN)

    g_min_sum_norm_BG1_Z384 = ...
    [   0.0
        0.0
        0.0
        0.0
        0.79
        0.77
        0.75
        0.73
        0.75
        0.70
        0.67
        0.68
        0.67
        0.67
        0.68
        0.66
        0.65
        0.66
        0.64
        0.65
        0.65
        0.65
        0.65
        0.66
        0.66
        0.66
        0.66
        0.66
        0.66
        0.67
        0.66
        0.65
        0.64
        0.63
        0.63
        0.63
        0.63
        0.63
        0.62
        0.63
        0.63
        0.64
        0.63
        0.63
        0.63
        0.62
        0.63];
    
    
g_min_sum_norm_BG2_Z384 = ...
[   0.0
    0.0
    0.0
    0.0
    0.86
    0.84
    0.80
    0.77
    0.75
    0.75
    0.74
    0.74
    0.74
    0.73
    0.73
    0.73
    0.73
    0.72
    0.70
    0.71
    0.71
    0.71
    0.71
    0.70
    0.69
    0.70
    0.70
    0.70
    0.70
    0.70
    0.70
    0.70
    0.70
    0.68
    0.67
    0.67
    0.68
    0.69
    0.69
    0.69
    0.69
    0.69
    0.69];


    
    switch(BGN)
        case 1
            alpha = g_min_sum_norm_BG1_Z384(nV_parity + 1);
        case 2
            alpha = g_min_sum_norm_BG2_Z384(nV_parity + 1);
        otherwise
            error("unsupported BGN. Most be 1 or 2");
    end
end

