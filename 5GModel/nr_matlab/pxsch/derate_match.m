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

function [derateCbs,nV_parity,derateCbsIndices,derateCbsIndicesSizes] = derate_match(LLR_demap, BGN, C, qam, ...
    K, F, K_prime, Zc, nl, N_data, rvIdx, Nref,G)

%Function undos rate matching. Takes LLRs of recieved bits and organizes
%them into LLRs of codeblocks.

%follows TS 38.212 section 5.4.2

%inputs:
%LLR_demap    --> demaped and descrambled LLRs

%outputs:
%deRateCbs    --> derate matched codeblocks
%nV_parity    --> number of parity nodes

%%
%PARAMATERS

%coding paramaters:
% BGN = PuschCfg.coding.BGN;         % 1 or 2. Indicates which base graph used
% C  = PuschCfg.coding.C;            % number of codeblocks
% qam = PuschCfg.coding.qam;         % bits per qam
% K = PuschCfg.coding.K;             % number of systematic bits per codeblock
% F = PuschCfg.coding.F;             % number of filler bits per codeblock
% K_prime = PuschCfg.coding.K_prime; %number of CRC encoded bits per codeblock (no filler)
% Zc = PuschCfg.coding.Zc;           % lifting size
%
% %allocation paramaters:
% nl = PuschCfg.mimo.nl;             % number of layers transmited by user
% N_data = PuschCfg.alloc.N_data;    % number of TF data resources in allocation

%%
%SIZE

%number of bits to be transmitted:
% G = N_data * qam * nl;

%derive number of rate matched bits per codeblock:
E = zeros(C,1);

for r = 0 : (C - 1)
    if r <= (C - mod( G / (nl * qam) , C) - 1)
        E(r + 1) = nl * qam * floor( G / (C * nl * qam) );
    else
        E(r + 1) = nl * qam * ceil( G / (C * nl * qam) );
    end
end


% apply LBMR
if BGN == 1
    N = Zc*66;
else
    N = Zc*50;
end


if Nref > 0
    N_cb = min(N, Nref);
else
    N_cb = N;
end


%%
%START

derateCbs = zeros(N,C);
derateCbsIndices = zeros(N,C);
derateCbsIndicesSizes = zeros(C,1);
current_bit = 0;

if BGN == 1
    switch rvIdx
        case 0
            k0 = 0;
        case 1
            k0 = floor(17*N_cb/(66*Zc))*Zc;
        case 2
            k0 = floor(33*N_cb/(66*Zc))*Zc;
        case 3
            k0 = floor(56*N_cb/(66*Zc))*Zc;
        otherwise
            error('rv is not supported...\n');
    end
elseif BGN == 2
    switch rvIdx
        case 0
            k0 = 0;
        case 1
            k0 = floor(13*N_cb/(50*Zc))*Zc;
        case 2
            k0 = floor(25*N_cb/(50*Zc))*Zc;
        case 3
            k0 = floor(43*N_cb/(50*Zc))*Zc;
        otherwise
            error('rv is not supported...\n');
    end
else
    error('BGN is not supported...\n');
end

%number of variable nodes:
if(BGN == 1)
    maxLLRPerCb    = Zc * 66;
    maxParityNodes = 46;
else
    maxLLRPerCb    = Zc * 50;
    maxParityNodes = 42;
end
nLLRPerCb    = min(floor(G / C) + k0, maxLLRPerCb);
nSymLLRPerCb = K - F - 2*Zc;
nParLLRPerCb = nLLRPerCb - nSymLLRPerCb;
nV_parity    = max(4, min(ceil(nParLLRPerCb / Zc), maxParityNodes));


for c = 1 : C

    %recover llrs corresponding to this codeblock:
    LLR_c = LLR_demap(current_bit + 1 : sum(E(1 : c)));

    %de-interleave:
    LLR_c = reshape(LLR_c, qam, E(c) / qam);
    LLR_c = LLR_c';
    LLR_c = LLR_c(:);

    %undo bit-selection:
    k = 0;
    j = 0;

    %display(['Zc=',num2str(Zc),' BGN=',num2str(BGN),' K=',num2str(K),' K_prime=',num2str(K_prime)]);
    %if (c == 1) || (c == C)
    %    global SimCtrl;
    %    display(['    {',num2str(E(c)),', ',num2str(K-2*Zc),', ',num2str(F),', ',num2str(k0),', ',num2str(N_cb),', ',num2str(Zc),'}, // ',SimCtrl.genTV.TVname,' c=',num2str(c)]);
    %end

    while k < E(c)
        %avoid filler bits:
        idx = mod(k0+j,N_cb) + 1;
        if ( idx <= (K_prime - 2*Zc) ) || ( idx > (K - 2*Zc) )
            cur_llr = derateCbs(idx,c) + LLR_c(k + 1);
            % clamp LLR as in cuPHY rate_matching.cu
            if (cur_llr > 10000)
                cur_llr = 10000;
            elseif (cur_llr < -10000)
                cur_llr = -10000;
            end
            derateCbs(idx,c) = cur_llr;
            derateCbsIndices(k+1,c) = idx;
            %line = sprintf('c,outIdx,inIdx,llr,%05d,%05d,%05d,%f',c,idx,k,derateCbs(idx,c));
            %display([' ',line]);
            k = k + 1;
        end
        j = j + 1;
    end

    derateCbsIndicesSizes(c) = E(c);

    %update current bit:
    current_bit = current_bit + E(c);
end

%add filler bits:
derateCbs(K_prime - 2*Zc + 1 : (K - 2*Zc),:) = 10000;
%for c = 1:size(derateCbs,2)
%    for idx = K_prime - 2*Zc + 1 : (K - 2*Zc)
%        line = sprintf('c,outIdx,inIdx,llr,%05d,%05d,%05d,%f',c,idx,99998,derateCbs(idx,c));
%        display([' ',line]);
%    end
%end

% threshold LLRs to match cuPHY
derateCbs(derateCbs > 10000)  = 10000;
derateCbs(derateCbs < -10000) = -10000;

% Adjust for Matlab to C indexing, includes padding used in C
derateCbsIndices = derateCbsIndices + 2*Zc - 1;



end
