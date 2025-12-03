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

% set test config
udIqWidth = {1, 2, 3, 4};
reMask = {[1 1 1 1 1 1 1 1 1 1 1 1], ...
          [1 1 0 1 1 1 1 1 1 0 1 1; 0 0 1 0 0 0 0 0 0 1 0 0], ...
          [1 1 1 1 1 1 0 0 0 0 0 0], ...
          [1 0 1 0 1 0 1 0 1 0 1 0]};
pcScaler = {1, ...
            [0.8, 1.1], ...
            1.1, ...
            1.5};
csf = {1,...
       [1, 0], ...
       1, ...
       1};

% load qam constant
qamstr = {'QPSK', '16QAM', '64QAM', '256QAM'};
qamScaler = [2/sqrt(2), 4/sqrt(10), 8/sqrt(42), 16/sqrt(170)]/sqrt(2);
qamTable = loadTable;

% init
inputBits = [];
Qams = [];
iqSamp_in = [];
modCompSamp = [];
iqSamp_out = [];
err = [];

for idxTest = 1:4
        
    % init random seed
    rng(idxTest*3);
    
    thisUdIqWidth = udIqWidth{idxTest};
    thisQamstr = qamstr{thisUdIqWidth};
    thisReMask = reMask{idxTest};
    thisPcScaler = pcScaler{idxTest};
    thisCsf = csf{idxTest};
    
    nMask = size(thisReMask, 1);
    iqSamp_in{idxTest} = zeros(12, 1);
    thisScaler = [];
    
    for idxMask = 1:nMask
        % generate modulated and scaled symbols
        usedRe = find(thisReMask(idxMask, :));
        nRe = length(usedRe);
        if idxMask == 1
            inputBits = round(rand(1, 2 * thisUdIqWidth * nRe));
            Qams = modulate_bits(inputBits, thisQamstr, qamTable);
            thisScaler(idxMask) = thisPcScaler(idxMask) * qamScaler(thisUdIqWidth);
        elseif idxMask == 2 % for CSI-RS QPSK
            inputBits = round(rand(1, 2 * 1 * nRe));
            Qams = modulate_bits(inputBits, 'QPSK', qamTable);
            thisScaler(idxMask) = thisPcScaler(idxMask) * qamScaler(1);
        end        
        iqSamp_in{idxTest}(usedRe) = thisPcScaler(idxMask) * Qams;
    end
    
    % apply modulation compression
    modCompSamp{idxTest} = modComp(iqSamp_in{idxTest}, thisUdIqWidth, thisReMask, thisScaler, thisCsf);
    
    % apply modulation decompression
    iqSamp_out{idxTest} = modDecomp(modCompSamp{idxTest}, thisUdIqWidth, thisReMask, thisScaler, thisCsf);
    
    % compare iqSamp_in and iqSamp_out
    err(idxTest) = sum(abs(iqSamp_in{idxTest} -  iqSamp_out{idxTest}).^2);
    
end

err

return
    
    
    
    