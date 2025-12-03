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

function [Hest, s_in, s_in_phase, yk_perm, Hest_preUnshift, s_out, Hest_postUnshift] = srs_ChEst_kernel(freqBlockIdx,gnbAntIdx,symIdx,y,Hest,W,nPrbPerThreadBlock,N_zc,q,delaySpread)

% performs channel estimation for single frequency block/gnb antenna using
% srs from a single symbol

%input indices:
% freqBlockIdx  --> index of estimated frequency block
% gnbAntIdx     --> index of estimated gnb antenna
% symIdx        --> index of srs symbol

%input arrays:
% y      --> recieved signal. Dim: Nf x nSym x L_gNB
% Hest   --> channel estimate. Dim: 136 x 4 x L_gNB x nUe
% W      --> ChEst filters. Dim: 4 x 24 x 4. (freq out) x (freq in) x (comb offset)

%input paramaters:
% nPrbPerThreadBlock --> number of prb in frequency block
% N_zc               --> zc prime
% q                  --> zc sequence number
% delaySpread        --> delay spread assumed by kernel

%%
%INDICES

prb0 = freqBlockIdx*nPrbPerThreadBlock; % starting prb of block
idx0 = 3*prb0;                          % starting sequence index

Nf_in  = nPrbPerThreadBlock * 3;
Nf_out = nPrbPerThreadBlock / 2;

%%
%LOAD

% here we load data the kernel needs

idx_freq = 12*prb0 + (0 : (12*nPrbPerThreadBlock-1));
yk = y(idx_freq + 1, symIdx + 1, gnbAntIdx + 1);

%%
%INPUT SEQUENCE

% here we compute the input sequence: removes ZC and centers delay

%indicies:
idx = idx0 + (0 : (3*nPrbPerThreadBlock - 1));
idx = idx.';
idx_zc = mod(idx,N_zc);


% compute phase:
phase = (0.4*delaySpread*30*10^3*4)*idx + ...
    q / (2*N_zc) * idx_zc.*(idx_zc + 1);

% input sequence:
s_in = exp(2*pi*1i*phase);
 
s_in_phase = 2*pi*1i*phase;

%%
%OUTPUT SEQUENCE

% here we compute the output sequence: used to undue delay shift.
s_out = zeros(Nf_out,4); % (freq out) x (comb offset)

for combOffset = 0 : 3 
    
    % compute phase:
    idx = 12*prb0 + (12 : 24 : (12*nPrbPerThreadBlock - 1));
    idx = idx - combOffset;
    idx = idx.';
    phase = -0.4*delaySpread*30*10^3*idx;

    % output sequence:
    s_out(:,combOffset+1) = exp(2*pi*1i*phase);
    
end

%%
%PROCESS INPUT

% seperate grids:
yk = reshape(yk,4,3*nPrbPerThreadBlock).'; % now: 3*nPrbPerThreadBlock x 4 

% remove zc and center delay:
for combOffset = 0 : 3
    yk(:,combOffset + 1) = s_in .* yk(:,combOffset + 1);
end

% permute cyclic shifts:
p = [1 + 0i   1 + 0i   1 + 0i   1 + 0i
    1 + 0i   0 - 1i  -1 + 0i   0 + 1i
    1 + 0i  -1 + 0i   1 + 0i  -1 + 0i
    1 + 0i   0 + 1i  -1 + 0i   0 - 1i];

yk_perm = zeros(Nf_in,4,4); %(freq in) x (cyclic shift) x (comb offset)

for cs = 0 : 3
    for combOffset = 0 : 3
        yk_perm(:,cs+1,combOffset+1) = repmat(p(:,cs+1),3*nPrbPerThreadBlock/4,1) .* yk(:,combOffset + 1);
    end
end

%% 
%FILTER

Hestk = zeros(Nf_out,4,4); % (freq out) x (cyclic shift) x (comb offset)

for cs = 0 : 3
    for combOffset = 0 : 3
        Hestk(:,cs+1,combOffset+1) = W(:,:,combOffset+1) * yk_perm(:,cs+1,combOffset+1);
    end
end

Hest_preUnshift = Hestk;

%%
%PROCESS OUTPUT

for cs = 0 : 3
    for combOffset = 0 : 3
        Hestk(:,cs+1,combOffset+1) = s_out(:,combOffset+1) .* Hestk(:,cs+1,combOffset+1);
    end
end

Hest_postUnshift = Hestk;

%%
%STORE

freq_idx = 0 : (nPrbPerThreadBlock/2 - 1);
freq_idx = prb0/2 + freq_idx;

for combOffset = 0 : 3
    ue_idx = 4*symIdx + combOffset;
    Hest(freq_idx+1,:,gnbAntIdx+1,ue_idx+1) = Hestk(:,:,combOffset+1);
end











        

















 


