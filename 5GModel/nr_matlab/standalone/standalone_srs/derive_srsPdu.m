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

function srsPar = derive_srsPdu(combPdu,symIdx)


% Function uses 3gpp/FAPI srs paramaters to derive "natural" paramaters
% (number of prbs, ZC sequence number, etc...)

%%
%OUTPUTS 

%all outputs stored in srsPar structure:

%time/frequency allocation:
% startPrb --> starting srs prb. 0-272
% nPrb     --> number of srs prbs.0-272.
% startSym --> starting srs symbol. 0-5. (Within all srs symbols! Not slot)
% nSym     --> number of srs symbols. 1, 2, or 4


%comb structure:
% combSize    --> comb spacing. 2 or 4
% nScPrb      --> number of srs subcarriers per Prb. 6 or 3
% combOffset  --> offset of comb from 0th subcarrier or 0th srs prb
                    % 0->1 for combSize 2. 0->3 for combSize 4.
                                     
%sequence:
% N_zc         --> Zadoff-Chu prime
% q            --> Zadoff-Chu sequence number
% cyclicShift  --> cyclic shift offset. 0-7 for combSize 2. 0-11 for combSize 4.
% numAntPorts  --> number of antenna ports muxed in comb. 1,2, or 4. 


%%
%INPUTS

%slot paramaters:
% symIdx --> indicies of all srs symbols within slot. 0-13
              


%PDU:
% pdu.numAntPorts              % nAnt ports in SRS resourse.
                               % 0 = 1 port
                               % 1 = 2 ports
                               % 2 = 4 ports
                               
% pdu.numSymbols               % nSyms in SRS resourse.
                               % 0 = 1 symbol
                               % 1 = 2 symbols
                               % 2 = 4 symbols
                               
% pdu.numRepetitions          % number of time SRS repeated in slot
                              % 0 = 1 time
                              % 1 = 2 times
                              % 2 = 4 times

                                    
% pdu.timeStartPosition     % starting symbol of SRS resourse. 
                            % value: 0->13
                               
% pdu.sequenceId             % srs sequence ID. 
                             % value: 0->1023.
                               
% pdu.configIndex             % SRS bandwidth configuration index
                              % value: 0->63
                               
% pdu.bandwidthIndex          % SRS bandwidth index
                              % value: 0->3
                               
% pdu.combSize                 % Transmission comb size
                               % 0 = comb size 2
                               % 1 = comb size 4
                               
% pdu.combOffset               % Transmission comb offset
                               % value: 0->1 (combSize = 0)
                               % value: 0->3 (combSize = 1)
                               
% pdu.cyclicShift              % cyclic shift offset. 0-11.
                               % value: 0->7 (combSize = 0)
                               % value: 0->11 (combSize = 1)
                               
% pdu.frequencyPosition        % frequency domain position
                               % value: 0->67

% pdu.frequencyShift           % frequency domain shift
                               % value: 0->268


numAntPorts = combPdu.numAntPorts;                            
numSymbols = combPdu.numSymbols;            
timeStartPosition = combPdu.timeStartPosition;                                                     
sequenceId = combPdu.sequenceId;    
configIndex = combPdu.configIndex;
bandwidthIndex = combPdu.bandwidthIndex;                                    
combSize = combPdu.combSize;                                             
combOffset = combPdu.combOffset;           
cyclicShift = combPdu.cyclicShift;                                          
frequencyPosition = combPdu.frequencyPosition;     
frequencyShift = combPdu.frequencyShift; 


%%
%ALLOCATION

%time allocation:
startSym = find(timeStartPosition == symIdx) - 1;

switch numSymbols
    case{0}
        nSym = 1;
    case{1}
        nSym = 2;
    case{2}
        nSym = 4;
end

%frequency allocation:

%first lookup number of prbs in table 6.4.1.4.3-1
load('srs_bandwidth_table.mat');
nPrb = T(configIndex+1,2*bandwidthIndex+1);
Nb = T(configIndex+1,2*bandwidthIndex+2);

%next compute start prb (no freq hopping):
startPrb = frequencyShift;
for b = 0 : bandwidthIndex
    nb = mod(floor(4*frequencyPosition/nPrb),Nb);
    startPrb = startPrb + nb*nPrb;
end

%%
%COMB STRUCTURE


switch combSize
    case{0}
        combSize = 2;
    case{1}
        combSize = 4;
end

nScPrb = 12 / combSize;


%%
%SEQUENCE

% length of srs sequence:
n_srs = nScPrb*nPrb;

% zc prime:
load('primes.mat');
idx = find(p < n_srs,1,'last');
N_zc = p(idx);

% zc seqeunce number (no group/sequence hopping):
u = mod(sequenceId,30);
v = 0; 
q_bar = N_zc*(u+1)/31;
q = floor(q_bar+0.5) + v*(-1)^floor(2*q_bar);

% antenna ports:
switch numAntPorts
    case{0}
        numAntPorts = 1;
    case{1}
        numAntPorts = 2;
    case{2}
        numAntPorts = 4;
end

%%
%WRAP


%time/frequency allocation:
srsPar.startPrb  = startPrb;
srsPar.nPrb = nPrb;    
srsPar.startSym = startSym; 
srsPar.nSym = nSym;   

%comb structure:
srsPar.combSize = combSize; 
srsPar.nScPrb = nScPrb;
srsPar.combOffset = combOffset; 
                    
%sequence:
srsPar.N_zc = N_zc;       
srsPar.q = q;           
srsPar.cyclicShift = cyclicShift; 
srsPar.numAntPorts = numAntPorts; 

 












    
    
    
    
    
    
    
    
    
    
