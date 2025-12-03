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

function pdu_cell = assign_pdu

% function assigns pdu for all srs resourse (currently only supports single
% user).

%outputs:
% pdu_cell --> Cell containing pdu for each user. Dim: nRes x 1

%%
%PARAMATERS


pdu_cell = cell(8,1);

count = 0;
for sym = 0 : 1
    for combOffset = 0 : 3
        count = count + 1;
        pdu = [];
        
        pdu.numAntPorts = 2;           % nAnt ports in SRS resourse.
                                       % 0 = 1 port
                                       % 1 = 2 ports
                                       % 2 = 4 ports

        pdu.numSymbols = 0;            % nSyms in SRS resourse.
                                       % 0 = 1 symbol
                                       % 1 = 2 symbols
                                       % 2 = 4 symbols

        pdu.numRepetitions = 0;        % number of time SRS repeated in slot
                                       % 0 = 1 time
                                       % 1 = 2 times
                                       % 2 = 4 times


        pdu.timeStartPosition = 12+sym;  % starting symbol of SRS resourse. 
                                         % value: 0->13

        pdu.sequenceId = 32;           % srs sequence ID. 
                                       % value: 0->1023.

        pdu.configIndex = 63;          % SRS bandwidth configuration index
                                       % value: 0->63

        pdu.bandwidthIndex = 0;        % SRS bandwidth index
                                       % value: 0->3

        pdu.combSize = 1;              % Transmission comb size
                                       % 0 = comb size 2
                                       % 1 = comb size 4

        pdu.combOffset = combOffset;   % Transmission comb offset
                                       % value: 0->1 (combSize = 0)
                                       % value: 0->3 (combSize = 1)

        pdu.cyclicShift = 0;           % cyclic shift offset. 0-11.
                                       % value: 0->7 (combSize = 0)
                                       % value: 0->11 (combSize = 1)

        pdu.frequencyPosition = 0;     % frequency domain position
                                       % value: 0->67

        pdu.frequencyShift = 0;        % frequency domain shift
                                       % value: 0->268
                                       
        pdu_cell{count} = pdu;
        
    end      
end
