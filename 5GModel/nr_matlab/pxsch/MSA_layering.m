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

function [TbCbs_est,itr,badItrCnt] = MSA_layering(LLR,nC,Zc,LDPCpar,TannerPar)

%function performs min-sum layering on recieved LLRs

%inputs:
% nC     --> number of active check nodes
% nV     --> number of active variable nodes
% Zc     --> lifting size
% LLR    --> recieved LLRs. Dim: Zc x nV
% maxItr --> max number of iterations
% alpha  --> normalization constant

%outputs:
% TbCbs_est --> hard estimate of transmited bits. Dim: Zc x nV
% itr       --> number of ldpc iterations performed

%%
%PARAMATERS

maxItr = LDPCpar.maxItr;        % maximium number of LDPC iterations
alpha = LDPCpar.alpha;          % normalization constant
earlyTerm = LDPCpar.earlyTerm;  % option to enable early termination
earlyTermAlg = LDPCpar.earlyTermAlg;        % early termination algorithm, 'PCE', 'CRC'
numCWs = LDPCpar.numCWs;
K_prime = LDPCpar.K_prime; % required for the CRC based early termination for each CW
puschTable = LDPCpar.puschTable; % need to load CRC table for the CRC based early termination for each CW
BGN = LDPCpar.BGN; % needed for the CRC based early termination for each CW
if numCWs > 1
    CRCstr = '24B';
else
    CRCstr = LDPCpar.CRCstr; % needed for the CRC based early termination when there is only one CW
end
LLR_len = numel(LLR);
APP_history = zeros(LLR_len,maxItr);
LLR_history = zeros(LLR_len,maxItr);

%%
%SETUP

APP = LLR;
C2V = zeros(Zc,nC,19);

%%
%START

% full PCE, CRC, NBF_*, core PCE, core PCE or nBadItr > *
nPceFail = zeros(maxItr, 1);
badItrCnt = 0;
if maxItr > 0
    for itr = 1 : maxItr

        for c = 1 : nC

            V2C = compute_V2C(c,APP,C2V,Zc,alpha,TannerPar);
            cC2V = compute_cC2V(c,V2C,Zc,TannerPar);
            C2V = update_C2V(c,cC2V,C2V,V2C,TannerPar);
            APP = update_APP(c,APP,C2V,V2C,alpha,TannerPar);

        end
        APP_history(:,itr) = (sign(reshape(APP,[],1))<0);
        LLR_history(:,itr) = reshape(APP,[],1);

        % check early termination for given algorithm                
        if earlyTerm
            if strcmp(earlyTermAlg,'PCE') % all nC * Zc parity equations must pass
                p = syndrome_check(APP,nC,Zc,TannerPar);
            elseif strcmp(earlyTermAlg, 'CRC') % CRC after hard decision must pass
                % hard decision
                tmp_Cbs_est = APP;
                tmp_Cbs_est(APP > 0) = 0;
                tmp_Cbs_est(APP <= 0) = 1;
                if BGN == 1
                    Cbs_est = tmp_Cbs_est(:,1:22);
                    Cbs_est = reshape(Cbs_est,22*Zc,1);
                else
                    Cbs_est = tmp_Cbs_est(:,1:10);
                    Cbs_est = reshape(Cbs_est,10*Zc,1);
                end
                [~, cbErr] = CRC_decode(Cbs_est(1:K_prime), CRCstr, puschTable);
                p = ~cbErr;
            elseif strcmp(earlyTermAlg, 'NBF')     % no bit flipping for all LLR for consecutive NBF_num_consecutive_itr
                APP_history(:,itr) = (sign(reshape(APP,[],1))<0);
                p = 0;
                if itr >= LDPCpar.NBF_num_consecutive_itr
                    tmp = sum(APP_history(:,itr-LDPCpar.NBF_num_consecutive_itr+1:itr),2);
                    if sum(mod(tmp,LDPCpar.NBF_num_consecutive_itr)) == 0
                        p = 1;
                    end
                end
            elseif strcmp(earlyTermAlg, 'CPCE') % core min(4,nC) * Zc parity equations must pass, using min(4,nC) just in case nC < 4 for customized LDPC
                p = syndrome_check(APP,min(4,nC),Zc,TannerPar);
            elseif strcmp(earlyTermAlg, 'SAFEET') % either min(4,nC) * Zc parity equations pass or too many badItr, badItr means core parity equation violations increases after iteration
                [p, nPceFail(itr)] = syndrome_check(APP,min(4,nC),Zc,TannerPar);
                if(itr > 1 && nPceFail(itr) > nPceFail(itr - 1))
                    badItrCnt = badItrCnt + 1;
                end
                p = (p || (badItrCnt > LDPCpar.SAFEET_badItrThres));
            else
                error('Undefined LDPC early termination algorithm!')
            end
            if p == 1
                break
            end
        end

    end

else
    itr = 0;
end


%%
%HARD

TbCbs_est = APP;
if (0) % Choose wisely
    % Sign function.  All-zero IQ tensor, or rvIdx != 0 / 3 (no systematic bits)
    % results in All-Zero codeword and passing CRC.  Don't pick this one.
    TbCbs_est(APP >= 0) = 0;
    TbCbs_est(APP < 0) = 1
else
    TbCbs_est(APP > 0) = 0;
    TbCbs_est(APP <= 0) = 1;
end
