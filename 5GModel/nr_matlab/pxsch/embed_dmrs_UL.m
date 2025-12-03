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

function Xtf = embed_dmrs_UL(Xtf, r_dmrs, nl, portIdx, n_scid, symIdx_dmrs, ...
    Nt_dmrs, energy, nPrb, startPrb, pdschTable, enableTfPrcd,...
    enablePrcdBf, PM_W, scrSeq)

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
global SimCtrl;

grid_table = pdschTable.grid_table;
nDmrs = length(symIdx_dmrs);

%build dmrs freq indicies:
freqIdx_dmrs = 0 : 2 : (nPrb*12 - 2);
freqIdx_dmrs = 12*(startPrb - 1) + freqIdx_dmrs;
freqIdx_dmrs = freqIdx_dmrs + 1;
scramIdx = (startPrb - 1)*6 + 1 : (startPrb + nPrb - 1)*6;
    
if enableTfPrcd == 0
    
    fOCC_table = pdschTable.fOCC_table;
    tOCC_table = pdschTable.tOCC_table;
    
    %extract dmrs scrambling sequence:
    r = r_dmrs(scramIdx,symIdx_dmrs,n_scid+1);
    
    if nargin > 16
        scr = scrSeq(scramIdx,symIdx_dmrs,n_scid+1);
    end
        
    %build fOCC:
    fOCC = ones(6*nPrb,1);
    fOCC(mod(1 : 6*nPrb,2) == 0) = -1;
    fOCC = repmat(fOCC,1,Nt_dmrs);
    
    %build tOCC:
    tOCC = ones(1,Nt_dmrs);
    tOCC(mod(1 : Nt_dmrs,2) == 0) = -1;
    tOCC = repmat(tOCC,6*nPrb,1);

else
    r = r_dmrs(scramIdx,symIdx_dmrs);
end

%%
%START

for i = 1 : nl
   
    if enableTfPrcd == 0
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
        
    else
        r_layer = r;
    end
    
    %grid offset:
    delta = grid_table(portIdx(i));
    
%     antIdx = portIdx(i) + 8*n_scid;
%     [~,~,nAnt] = size(Xtf);
%     antIdx = mod(antIdx-1, nAnt)+1;
    antIdx = i;
    
    %embed:
    if enablePrcdBf
        qamPrcd = sqrt(energy) * r_layer * PM_W(:, i).';
        nQam = size(qamPrcd, 1);
        nAnt = length(PM_W(:, i));
        qamPrcd = reshape(qamPrcd, nQam, nDmrs, nAnt);

        Xtf(freqIdx_dmrs + delta,symIdx_dmrs,:) = ...
            Xtf(freqIdx_dmrs + delta,symIdx_dmrs,:) + qamPrcd;
    else
        if SimCtrl.normalize_pusch_tx_power_over_layers
            Xtf(freqIdx_dmrs + delta,symIdx_dmrs,antIdx) = sqrt(energy/nl) * r_layer;
        else
            Xtf(freqIdx_dmrs + delta,symIdx_dmrs,antIdx) = sqrt(energy) * r_layer;
        end
    %     Xtf(freqIdx_dmrs + delta,symIdx_dmrs,portIdx(i) + 8*n_scid) = sqrt(energy) * r_layer;
    %     Xtf(freqIdx_dmrs + delta,symIdx_dmrs,portIdx(i)) = sqrt(energy) * r_layer;
    end        
end

return
