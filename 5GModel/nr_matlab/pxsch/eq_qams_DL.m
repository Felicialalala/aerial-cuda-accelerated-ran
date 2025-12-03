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

function Qams_eq = eq_qams_DL(xTF, nl, portIdx, n_scid,...
    nPrb, BWPStart, startPrb,N_data,symIdx_data, symIdx_dmrs,...
    numDmrsCdmGrpsNoData, Xtf_remap, eq_coef, resourceAlloc, rbBitmap)

%function embeds a user's layer-mapped qam symbols into the slot

%inputs:
% Xtf      --> time-frequency slot. Dim: Nf x Nt x L_UE
% Qams     --> users qam symbols. Dim: N_data * nl
% lm_flag  --> flag indicating if layer mapping should be performed

%outputs:
% Xtf      --> time-frequency slot w/h embeded qams. Dim: Nf x Nt x L_UE

%PARAMATERS

%mimo paramaters:
% nl = alloc.nl;                    % number of layers transmited by user
% portIdx = alloc.portIdx;          % user's antenna ports (matlab 1 indexing). Dim: nl x 1
% n_scid = alloc.SCID;            % 0 or 1. User's dmrs scrambling id

%allocation paramaters:
% nPrb = alloc.nPrb;               % number of prbs in allocation
% BWPStart = alloc.BWPstart;       % BWP starting location
% startPrb = alloc.startPrb;       % starting prb of allocation
% Nf_data = alloc.Nf_data;         % number of data subcarriers in allocation
% Nt_data = alloc.Nt_data;         % number of data symbols in allocation
% N_data = alloc.N_data;           % number of data TF resources in allocation
% symIdx_data = alloc.symIdx_data; % indicies of data symbols. Dim: Nt_data x 1 

%SHAPE SYMBOLS

% layer mapping:
% Qams = reshape(Qams,nl,N_data);      % now dim: nl x N_data
% Qams = permute(Qams, [2,1]);

idxQam = 0;
antIdx = portIdx + n_scid*8;
Qams_eq = [];
eq_out = [];

if numDmrsCdmGrpsNoData == 1
    symIdx_data = [min(symIdx_data(1), symIdx_dmrs(1)) : max(symIdx_data(end), symIdx_dmrs(end))];
    dmrs_remap = zeros(size(xTF, 1, 2));
    dmrs_remap(1:2:end, symIdx_dmrs) = 1;
    noData_remap = Xtf_remap + dmrs_remap;
else
    noData_remap = Xtf_remap;
end

for idx = 1:length(symIdx_data)
    idxSym = symIdx_data(idx);
    if resourceAlloc == 0
        rbBitmapSize = size(rbBitmap,1);
        reBitmap = logical(reshape(repmat([zeros(BWPStart,1);rbBitmap(1:rbBitmapSize-BWPStart)],[1,12]).',[],1));
        eq_out{idx} = permute( pagemtimes( permute(xTF(reBitmap, idxSym,:),[2,3,1]), permute(eq_coef(:,idx,:,:), [3,4,1,2]) ), [3,2,1]);
        freqIdx_data = find(reBitmap==1);
    else
        for idxRe = 1:nPrb*12
            eq_RE = squeeze(eq_coef(idxRe, idx, :, :));
            xTF_RE = squeeze(xTF(idxRe + 12 * (startPrb - 1), idxSym, :));
            eq_out{idx}(idxRe, :) = xTF_RE.' * eq_RE;
        end    
        freqIdx_data = 12 * (startPrb - 1) + [1:nPrb*12];
    end
    eq_out{idx}(find(noData_remap(freqIdx_data, idxSym)), :) = [];
    nREinSym = size(eq_out{idx}, 1);
    Qams_eq(idxQam+1 : idxQam+nREinSym, :) = eq_out{idx};
    idxQam = idxQam + nREinSym;
end

Qams_eq = Qams_eq.';
Qams_eq = Qams_eq(:);

return
