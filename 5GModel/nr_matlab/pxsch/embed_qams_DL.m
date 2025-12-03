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

function xTF = embed_qams_DL(xTF,Qams,nl, portIdx, n_scid, nlAbove16, ...
    nPrb, BWPStart, startPrb,N_data,symIdx_data, symIdx_dmrs,...
    Xtf_remap, numDmrsCdmGrpsNoData, enablePrcdBf, PM_W, resourceAlloc, rbBitmap)

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
% startPrb = alloc.startPrb;       % starting prb of allocation
% BWPStart = alloc.BWPstart;       % BWP starting location
% Nf_data = alloc.Nf_data;         % number of data subcarriers in allocation
% Nt_data = alloc.Nt_data;         % number of data symbols in allocation
% N_data = alloc.N_data;           % number of data TF resources in allocation
% symIdx_data = alloc.symIdx_data; % indicies of data symbols. Dim: Nt_data x 1 

%SHAPE SYMBOLS

% layer mapping:
Qams = reshape(Qams,nl,N_data);      % now dim: nl x N_data
Qams = permute(Qams, [2,1]);

idxQam = 0;
antIdx = portIdx + n_scid*8 + nlAbove16*16;

symAll = [min(symIdx_data(1), symIdx_dmrs(1)) : max(symIdx_data(end), symIdx_dmrs(end))];
if resourceAlloc == 0
    reBitmap = reshape(repmat(rbBitmap,[1,12]).',[],1);
    freqIdx_data = find(reBitmap==1);
    startPrb = BWPStart+1; % because the first bit in the rb bitmap is referenced to BWPStart
else
    freqIdx_data = 12 * (startPrb - 1) + 1 : 12 * (startPrb + nPrb - 1);
end
for idxSym = symAll
    if resourceAlloc == 0
        freqIdx = find(Xtf_remap(freqIdx_data, idxSym) == 0);
        freqIdx = freqIdx_data(freqIdx);
    else
        freqIdx = find(Xtf_remap(freqIdx_data, idxSym) == 0);  
    end
    if ismember(idxSym, symIdx_dmrs) 
        if (numDmrsCdmGrpsNoData == 1)
            freqIdx = freqIdx(2:2:end);
        else
            continue;
        end
    end
    nREinSym = length(freqIdx);
    if enablePrcdBf
        qamPrcd = Qams(idxQam + (1:nREinSym), :) * PM_W.';
        [nQam, nAnt] = size(qamPrcd);
        qamPrcd = reshape(qamPrcd, nQam, 1, nAnt);
        xTF(freqIdx + 12 * (startPrb - 1), idxSym, 1:nAnt) = ...
            xTF(freqIdx + 12 * (startPrb - 1), idxSym, 1:nAnt) + qamPrcd;
    else
        % Even when precoding is disabled, QAM values should be added to existing xTF values
        % as it is possible to have some UEs with precoding enabled and some without even in the same cell.
        [nRE, nAnt] = size(Qams(idxQam + (1:nREinSym), :));
        xTF(freqIdx + 12 * (startPrb - 1), idxSym, antIdx) = ...
            xTF(freqIdx + 12 * (startPrb - 1), idxSym, antIdx) + reshape(Qams(idxQam + (1:nREinSym), :), nRE, 1, nAnt);
    end
    idxQam = idxQam + nREinSym;
end
            
% for idx = 1:length(symIdx_data)
%     idxSym = symIdx_data(idx);
%     freqIdx_data = 12 * (startPrb - 1) + 1 : 12 * (startPrb + nPrb - 1);
%     freqIdx = find(Xtf_remap(freqIdx_data, idxSym) == 0);
%     nREinSym = length(freqIdx);
%     if enablePrcdBf
%         qamPrcd = Qams(idxQam + (1:nREinSym), :) * PM_W.';
%         [nQam, nAnt] = size(qamPrcd);
%         qamPrcd = reshape(qamPrcd, nQam, 1, nAnt);
%         xTF(freqIdx + 12 * (startPrb - 1), idxSym, 1:nAnt) = ...
%             xTF(freqIdx + 12 * (startPrb - 1), idxSym, 1:nAnt) + qamPrcd;
%     else
%         % Even when precoding is disabled, QAM values should be added to existing xTF values
%         % as it is possible to have some UEs with precoding enabled and some without even in the same cell.
%         [nRE, nAnt] = size(Qams(idxQam + (1:nREinSym), :));
%         xTF(freqIdx + 12 * (startPrb - 1), idxSym, antIdx) = ...
%             xTF(freqIdx + 12 * (startPrb - 1), idxSym, antIdx) + reshape(Qams(idxQam + (1:nREinSym), :), nRE, 1, nAnt);
%     end
%     idxQam = idxQam + nREinSym;
% end

return
