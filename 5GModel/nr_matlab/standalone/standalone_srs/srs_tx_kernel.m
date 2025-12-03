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

function s_tx = srs_tx_kernel(s_tx,combPar)

% function computes and embeds srs for a single comb.

%inputs:
% s_tx    --> srs transmitted by all users. Dim: 4 x Nf x nSym x nRes
% combPdu --> pdu for current comb

%output:
% s_tx   --> updated srs signal

%%
%PARAMATERS

 
%time/frequency allocation:
startPrb = combPar.startPrb;   % start srs prb. 0-272
nPrb =  combPar.nPrb;          % number of srs prbs. 0-272
startSym = combPar.startSym;   % starting srs symbol. 0-5. (Within all srs symbols! Not slot)
nSym = combPar.nSym;           % number of srs symbols. 1, 2, or 4

%comb structure:
combSize = combPar.combSize;      % scomb spacing. 2 or 4
nScPrb = combPar.nScPrb;          % number of srs subcarriers per Prb. 6 or 3
combOffset = combPar.combOffset;  % offset of comb from 0th subcarrier or 0th srs prb
                                       % 0->1 for combSize 2. 0->3 for combSize 4.
                  
%sequence:
N_zc = combPar.N_zc;                % Zadoff-Chu prime
q = combPar.q;                      % Zadoff-Chu sequence number
cyclicShift = combPar.cyclicShift;  % cyclic shift offset. 0-7 for combSize 2. 0-11 for combSize 4.
numAntPorts = combPar.numAntPorts;  % number of antenna ports muxed in comb. 1,2, or 4. 

%storage:
portMapping = combPar.portMapping; % mapping from comb antenna ports to srs antenna ports.
resIdx = combPar.resIdx;           % srs resourse index comb is from.


%%
%GENERATE ZC 

n_srs = nPrb*nScPrb;

idx = 0 : (n_srs - 1);
idx = mod(idx,N_zc).';

ZC = exp(-1i*pi*q*idx.*(idx+1) / N_zc);


%%
%GENERATE SRS SEQUENCE

s = zeros(n_srs,numAntPorts);

idx = 0 : (n_srs - 1);
idx = idx';

if combSize == 2
    cs_max = 8;
else
    cs_max = 12;
end

for p = 0 : (numAntPorts - 1)
    cs = mod(cyclicShift + cs_max*p/numAntPorts,cs_max);
    alpha = 2*pi*cs/cs_max;
    
    s(:,p+1) = exp(1i*alpha*idx).*ZC;
end

%%
%EMBED SRS SEQUENCE

% freq indicies:
freqIdx = 0 : combSize : (12*nPrb-1);
freqIdx = 12*startPrb + freqIdx + combOffset;

% time indicies:
timeIdx = startSym : (startSym + nSym - 1);

% repeate signal:
s = permute(s,[2 1]);          % now: numAntPorts x Nf_srs
s = repmat(s,1,1,nSym);        % now: numAntPorts x Nf_srs x nSym

% embed:
s_tx(portMapping + 1,freqIdx + 1,timeIdx + 1, resIdx + 1) = s;


end















