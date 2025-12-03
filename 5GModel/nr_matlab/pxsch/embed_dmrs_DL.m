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

function Xtf = embed_dmrs_DL(Xtf, r_dmrs, nl, portIdx, n_scid, nlAbove16, symIdx_dmrs, ...
    Nt_dmrs, energy, nPrb, BWPStart, startPrb, pdschTable, refPoint, enablePrcdBf, PM_W, resourceAlloc, rbBitmap)

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
    
    %embed:
    if enablePrcdBf
        if nDmrs == 1
            qamPrcd = sqrt(energy) * r_layer * PM_W(:, i).';
        else
            qamPrcd = [];
            for idx = 1:nDmrs
                qamPrcd = [qamPrcd, sqrt(energy) * r_layer(:, idx) * PM_W(:, i).'];
            end
        end
        nQam = size(qamPrcd, 1);
        nAnt = length(PM_W(:, i));
%         qamPrcd = reshape(qamPrcd, nQam, nDmrs, nAnt);
        qamPrcd = reshape(qamPrcd, nQam, nAnt, nDmrs);
        qamPrcd = permute(qamPrcd, [1,3,2]);

        Xtf(freqIdx_dmrs + delta,symIdx_dmrs,1:nAnt) = ...
            Xtf(freqIdx_dmrs + delta,symIdx_dmrs,1:nAnt) + qamPrcd;
    else
        % Even when precoding is disabled, DMRS values should be added to existing xTF values
        % as it is possible to have some UEs with precoding enabled and some without even in the same cell.
        Xtf(freqIdx_dmrs + delta,symIdx_dmrs,portIdx(i) + 8*n_scid + nlAbove16*16) = ...
            Xtf(freqIdx_dmrs + delta,symIdx_dmrs,portIdx(i) + 8*n_scid + nlAbove16*16) + sqrt(energy) * r_layer;
        %     Xtf(freqIdx_dmrs + delta,symIdx_dmrs,portIdx(i)) = sqrt(energy) * r_layer;
    end
end

return
