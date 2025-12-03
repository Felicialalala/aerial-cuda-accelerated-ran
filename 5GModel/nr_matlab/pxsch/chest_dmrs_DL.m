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

function dmrs_ch = chest_dmrs_DL(Xtf, r_dmrs, nl, portIdx, n_scid, symIdx_dmrs, ...
    Nt_dmrs, energy, nPrb, BWPStart, startPrb, pdschTable, refPoint, AdditionalPosition, resourceAlloc, rbBitmap)

%function embed users dmrs into tf slot.

%inputs:
% Xtf --> slot tf signal. Dim: Nf x Nt x L_UE

%outputs:
% Xtf --> slot tf signal w/h embedded dmrs

%%
%PARAMATERS

%gnb paramaters:

%mimo paramaters:
%mimo paramaters:
% nl = alloc.nl;                    % number of layers transmited by user
% portIdx = alloc.portIdx;          % user's antenna ports (matlab 1 indexing). Dim: nl x 1
% n_scid = alloc.SCID;            % 0 or 1. User's dmrs scrambling id

%dmrs paramaters:
% symIdx_dmrs = dmrs.symIdx_dmrs;  % Indicies of dmrs symbols (matlab 1 indexing). Dim: Nt_dmrs x 1
% Nt_dmrs = dmrs.Nt_dmrs;          % number of dmrs symbols 
% energy = dmrs.energy;            % dmrs energy

% %allocation paramaters:
% nPrb = alloc.nPrb;               % number of prbs in allocation
% startPrb = alloc.startPrb;       % starting prb of allocation

%%
%SETUP

% load('type1_dmrs_table.mat');

fOCC_table = pdschTable.fOCC_table;
grid_table = pdschTable.grid_table;
tOCC_table = pdschTable.tOCC_table;

%extract dmrs scrambling sequence:
if resourceAlloc == 0
    len_r_dmrs = length(r_dmrs);
    numOutputPrbs = size(Xtf,1)/12;
    tmp_scramIdx = reshape([1:len_r_dmrs], 6, []);
    rbBitmap = [zeros(BWPStart,1);rbBitmap(1:numOutputPrbs-BWPStart)];
    nPrb = sum(rbBitmap);
    if refPoint == 0
        scramIdx = reshape(tmp_scramIdx(:,logical(rbBitmap)), 1, []);
    else
        scramIdx = reshape(tmp_scramIdx(:,logical(rbBitmap(BWPStart+1:end))), 1, []);
    end    
else
    if refPoint == 0
        scramIdx = (startPrb - 1)*6 + 1 : (startPrb + nPrb - 1)*6;
    else
        scramIdx = (startPrb - 1 - BWPStart)*6 + [1 : nPrb*6];
    end
end

r = r_dmrs(scramIdx,symIdx_dmrs,n_scid+1);

nDmrs = length(symIdx_dmrs);

%build dmrs freq indicies:
if resourceAlloc == 0
    reBitmap_dmrs = reshape(repmat(rbBitmap,[1,12]).',[],1);
    reBitmap_dmrs(2:2:end) = 0;
    freqIdx_dmrs = find(reBitmap_dmrs==1);
else
    freqIdx_dmrs = 0 : 2 : (nPrb*12 - 2);
    freqIdx_dmrs = 12*(startPrb - 1) + freqIdx_dmrs;
    freqIdx_dmrs = freqIdx_dmrs + 1;
end

%build fOCC:
fOCC = ones(6*nPrb,1);
fOCC(mod(1 : 6*nPrb,2) == 0) = -1;
fOCC = repmat(fOCC,1,Nt_dmrs);

%build tOCC:
tOCC = ones(1,Nt_dmrs);
tOCC(mod(1 : Nt_dmrs,2) == 0) = -1;
tOCC = repmat(tOCC,6*nPrb,1);

%%
%START

nAnt = size(Xtf, 3);

dmrs_ch_preRE = [];
for idxAnt = 1:nAnt
    for i = 1 : nl
        %initialize:
        r_layer = r;
        
        %apply fOCC:
        if fOCC_table(portIdx(i))
            r_layer = fOCC .* r_layer;
        end
        
        %apply tOCC:
        if tOCC_table(portIdx(i))
            r_layer = tOCC .* r_layer;
        end
        
        %grid offset:
        delta = grid_table(portIdx(i));
        
        % descramble
        dmrs_ch_perRE(:, :, i, idxAnt) = ...
            Xtf(freqIdx_dmrs + delta,symIdx_dmrs, idxAnt) .* conj(r_layer)/sqrt(energy);
    end
end

maxLen = Nt_dmrs/(AdditionalPosition+1);

for idxAnt = 1:nAnt
    for i = 1 : nl
        for idxPos = 1:AdditionalPosition+1
            temp1 = squeeze(dmrs_ch_perRE(:, (idxPos-1)*maxLen+1:idxPos*maxLen, i, idxAnt));
            temp2 = mean(temp1, 2);
            temp3 = reshape(temp2, 6, length(temp2)/6); % average over 6 REs (one PRB)
            temp4 = mean(temp3, 1);
            temp5 = repmat(temp4, 12, 1); % apply to all 12 REs in one PRB
            dmrs_ch(:, idxPos, i ,idxAnt) = temp5(:);
        end
    end
end

return
