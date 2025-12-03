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

function [Hest, dbg] = srs_ChEst_main(y,W,cuphySRSpar)

%function call SRS channel estiamtion kernels

%inputs:
% y --> signal recieved by gNB on SRS symbols. Dim: Nf x nSym x L_gNB
% W --> ChEst filters. Dim: nPrbPerThreadBlock/2 x 3*nPrbPerThreadBlock x 4. (freq out) x (freq in) x (comb offset)

%outputs:
% Hest --> uplink channel estimates. Dim: 136 x 4 x L_gNB x nUe

%%
%PARAMATERS

Lgnb               = cuphySRSpar.Lgnb;        % number of gnb antennas
nUe                = cuphySRSpar.nUe;         % number of user transmitting srs
nPrb               = cuphySRSpar.nPrb;        % number of srs prbs 
nSym               = cuphySRSpar.nSym;        % number of srs symbols
N_zc               = cuphySRSpar.N_zc;        % zc prime
q                  = cuphySRSpar.q;           % zc sequence number
delaySpread        = cuphySRSpar.delaySpread; % delay spread assumed by kernel
nPrbPerThreadBlock = cuphySRSpar.nPrbPerThreadBlock;


%%
%SETUP

nFreqBlocks = nPrb / nPrbPerThreadBlock;
 
%%
%START

Hest = zeros(nPrb / 2, 4, Lgnb, nUe);

s_in = zeros(nPrb*12/4,1); % nPrb * Nsc / nComb
s_in_phase = zeros(nPrb*12/4,1); % nPrb * Nsc / nComb
yk_perm = zeros(nPrbPerThreadBlock*12/4,nFreqBlocks,4,4); % nPrbPerThreadBlock * Nsc / nComb, nFreqBlocks, nCycShifts, nCombs
Hest_preUnshift = zeros(2,nFreqBlocks,4,4); % nSrsChEst, nFreqBlocks, nCycShifts, nCombs
s_out = zeros(nFreqBlocks*2,4); % nSrsChEst, nFreqBlocks, nCycShifts, nCombs
Hest_postUnshift = zeros(2,nFreqBlocks,4,4); % nSrsChEst, nFreqBlocks, nCycShifts, nCombs

for symIdx = 0 : (nSym - 1)
    for gnbAntIdx = 0 : (Lgnb - 1)
        for freqBlockIdx = 0 : (nFreqBlocks - 1)
            
             if freqBlockIdx == 40 && gnbAntIdx == 17 && symIdx == 1
                 %keyboard
             end

            [Hest, s_in_f, s_in_phase_f, yk_perm_f, Hest_preUnshift_f, s_out_f, Hest_postUnshift_f] = srs_ChEst_kernel(freqBlockIdx,gnbAntIdx,symIdx,y,Hest,W,nPrbPerThreadBlock,N_zc,q,delaySpread);
            
            % Capture debug information for specific SRS symbol and BS antenna
            if (symIdx == 1) && (gnbAntIdx == 17)
                s_in((freqBlockIdx*12) + (1:12)) = s_in_f;
                s_in_phase((freqBlockIdx*12) + (1:12)) = s_in_phase_f;
                yk_perm(:,freqBlockIdx+1,:,:) = yk_perm_f;
                Hest_preUnshift(:,freqBlockIdx+1,:,:) = Hest_preUnshift_f;
                s_out(2*freqBlockIdx+ (1:2),:) = s_out_f;
                Hest_postUnshift(:,freqBlockIdx+1,:,:) = Hest_postUnshift_f;
            end
        end
    end
end

dbg.s_in = s_in;
dbg.s_in_phase = s_in_phase;
dbg.yk_perm = yk_perm;
dbg.Hest_preUnshift = Hest_preUnshift;
dbg.s_out = s_out;
dbg.Hest_postUnshift = Hest_postUnshift;

