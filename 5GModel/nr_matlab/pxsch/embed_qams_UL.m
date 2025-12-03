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

function xTF = embed_qams_UL(xTF,Qams,lm_flag, nl, portIdx, n_scid,...
    nPrb, startPrb,Nf_data,Nt_data,N_data,symIdx_data, symIdx_dmrs, ...
    numDmrsCdmGrpsNoData, isDataPresent, enableTfPrcd, enablePrcdBf, PM_W)

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
% Nf_data = alloc.Nf_data;         % number of data subcarriers in allocation
% Nt_data = alloc.Nt_data;         % number of data symbols in allocation
% N_data = alloc.N_data;           % number of data TF resources in allocation
% symIdx_data = alloc.symIdx_data; % indicies of data symbols. Dim: Nt_data x 1 

%SHAPE SYMBOLS
global SimCtrl;

% layer mapping:
if lm_flag
    Qams = reshape(Qams,nl,N_data);      % now dim: nl x N_data
    Qams = Qams(:);
else
    Qams = reshape(Qams,N_data,nl);
end

% frequency first mapping:
% Qams = reshape(Qams,Nf_data,Nt_data,nl); % now dim: Nf_data x Nt_data x nl

%INDICIES

% transmit antennas:
% antIdx = portIdx + n_scid*8;
% [~,~,nAnt] = size(xTF);
% antIdx = mod(antIdx-1, nAnt)+1;
antIdx = 1:nl;

% frequency indicies:
freqIdx_data = 12 * (startPrb - 1) + 1 : 12 * (startPrb + nPrb - 1);


%EMBED
if enablePrcdBf
    [~, ~, nAnt] = size(xTF);
    nQams = length(Qams(:));     
    Qams = reshape(Qams, nQams/nl, nl) * PM_W.';
    QamsIdx = 1;
    symAll = [min(symIdx_data(1), symIdx_dmrs(1)) : max(symIdx_data(end), symIdx_dmrs(end))];
    for symIdx = symAll
        if ismember(symIdx, symIdx_dmrs) && (numDmrsCdmGrpsNoData == 1)
            thisQams = Qams(QamsIdx: QamsIdx + Nf_data/2 -1, :); 
            if enableTfPrcd == 1
                thisQams = sqrt(2/Nf_data)*fft(thisQams);
            end
            xTF(freqIdx_data(2:2:end), symIdx, :) = ...
                reshape(thisQams, Nf_data/2, 1, nAnt);
            QamsIdx = QamsIdx + Nf_data/2;
        elseif ~ismember(symIdx, symIdx_dmrs)
            thisQams = Qams(QamsIdx: QamsIdx + Nf_data -1, :); 
            if enableTfPrcd == 1
                thisQams = sqrt(1/Nf_data)*fft(thisQams);
            end
            xTF(freqIdx_data, symIdx, :) = ...
                reshape(thisQams, Nf_data, 1, nAnt);
            QamsIdx = QamsIdx + Nf_data;
        end
    end     
else
    QamsIdx = 1;
    symAll = [min(symIdx_data(1), symIdx_dmrs(1)) : max(symIdx_data(end), symIdx_dmrs(end))];
    for symIdx = symAll
        if ismember(symIdx, symIdx_dmrs) && (numDmrsCdmGrpsNoData == 1)
            if SimCtrl.normalize_pusch_tx_power_over_layers
                thisQams = 1/sqrt(nl)*Qams(QamsIdx: QamsIdx + Nf_data/2 -1, :);
            else
                thisQams = Qams(QamsIdx: QamsIdx + Nf_data/2 -1, :);
            end
            if enableTfPrcd == 1
                thisQams = sqrt(2/Nf_data)*fft(thisQams);
            end
            xTF(freqIdx_data(2:2:end), symIdx, antIdx) = ...
                reshape(thisQams, Nf_data/2, 1, nl);
            QamsIdx = QamsIdx + Nf_data/2;            
        elseif ~ismember(symIdx, symIdx_dmrs)
            if SimCtrl.normalize_pusch_tx_power_over_layers
                thisQams = 1/sqrt(nl)*Qams(QamsIdx: QamsIdx + Nf_data -1, :);
            else
                thisQams = Qams(QamsIdx: QamsIdx + Nf_data -1, :);
            end
            if enableTfPrcd == 1
                thisQams = sqrt(1/Nf_data)*fft(thisQams);
            end
            xTF(freqIdx_data, symIdx, antIdx) = ...
                reshape(thisQams, Nf_data, 1, nl);
            QamsIdx = QamsIdx + Nf_data;
        end
    end
end

%EMBED
% if enablePrcdBf
%      [~, ~, nAnt] = size(xTF);   
%      Qams = reshape(Qams, Nf_data*Nt_data, nl) * PM_W.';
%      Qams = reshape(Qams, Nf_data, Nt_data, nAnt);    
%      xTF(freqIdx_data, symIdx_data, :) = Qams;
% else
%     xTF(freqIdx_data, symIdx_data, antIdx) = Qams;
%     % xTF(freqIdx_data, symIdx_data, portIdx) = Qams;
% end

return
