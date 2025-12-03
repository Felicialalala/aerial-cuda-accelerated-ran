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

function TbRateMatCbs = rate_match(TbCodedCbs, C, qam, nl, N_data_used, rvIdx, BGN, Zc, Nref,G)

%function performs rate matching: 
%1.) Selects which bits to transmit from fully encoded blocks.
%2.) Interleaves the selected bits

%follows TS 38.212 section 5.4.2

%inputs:
%TbCodedCbs  --> fully coded codeblocks

%outputs:
%TbRateMatCbs --> rate matched codeblocks

%PARAMATERS

%coding paramaters:
% C  = coding.C;            %number of codeblocks
% qam = coding.qam;         %bits per qam

%allocation paramaters:
% nl = alloc.nl;             %number of layers transmited by user
% N_data = alloc.N_data;   %number of TF data resources in allocation

%SIZE

%number of bits to be transmitted:
% G = N_data_used * qam * nl;

%derive number of rate matched bits per codeblock:
E = zeros(C,1);

for r = 0 : (C - 1) 
    if r <= (C - mod( G / (nl * qam) , C) - 1)
        E(r + 1) = nl * qam * floor( G / (C * nl * qam) );
    else
        E(r + 1) = nl * qam * ceil( G / (C * nl * qam) );
    end
end

%number of bits in fully coded blocks:
N = size(TbCodedCbs,1);
% apply LBMR
if Nref > 0
    N_cb = min(N, Nref);
else
    N_cb = N;
end        

%SELECT

%select bits to be transmited:
TbRateMatCbs = [];

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


for c = 1 : C
    
    %selct bits for code block c:
    TbRateMatCbs_c = zeros(E(c),1);
    
    k = 0;
    j = 0;
    
    while k < E(c)
        %avoid filler bits:
        if TbCodedCbs( mod(k0+j,N_cb) + 1 , c ) ~= -1
            TbRateMatCbs_c(k + 1) = TbCodedCbs(mod(k0+j,N_cb) + 1 , c);
            k = k + 1;
        end
        j = j + 1;
    end
    
    %bit interleaving:
    TbRateMatCbs_c = reshape(TbRateMatCbs_c, E(c) / qam, qam);
    TbRateMatCbs_c = TbRateMatCbs_c';
    TbRateMatCbs_c = TbRateMatCbs_c(:);
    
    %embed:
    TbRateMatCbs = [TbRateMatCbs ; TbRateMatCbs_c];    
    
end

return
