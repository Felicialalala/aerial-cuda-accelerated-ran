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

function s_tx = srs_tx_main(gnb,srs)

% function computes the srs transmitted by all users

%outputs:
% s_tx --> transmitted srs. Dim: 4 x Nf x nSym x nRes

%%
%PARAMATERS

% gnb paramaters:
Nf = gnb.Nf;        % number of subcarrier in bwp

% uplink slot srs paramaters:
nSym = srs.nSym;                % total number of srs symbols 
symIdx = srs.symIdx;            % locations of srs symbols
nRes = srs.nRes;                % number of SRS resourses
srsPdu_cell = srs.pdu_cell;     % cell containing fapi srs pdu of each resourse


%%
%SRS PDU --> SRS PAR

%For each srs resourse, use FAPI/3gpp paramaters to derive "natural" paramaters (number
%of prbs, ZC sequence number, etc...)

srsPar_cell = cell(nRes,1);

for r = 0 : (nRes - 1)
    srsPar_cell{r+1} = derive_srsPdu(srsPdu_cell{r+1},symIdx);
end



%%
%SRS PAR --> COMB PAR

% GPU processes combs independently. If an SRS resourse has 4 antenna ports,
% there is a possibility that the SRS resourse consists of 2 combs. 
% Here we convert SRS paramaters to comb paramaters.


% first compute number of combs:
nComb = 0;

for r = 0 : (nRes - 1)
    srsPar = srsPar_cell{r + 1};
    
    if (srsPar.cyclicShift >= 6) && (srsPar.numAntPorts == 4) && (srsPar.combSize == 4)
        nComb = nComb + 2;
    else
        nComb = nComb + 1;
    end
end

% generate paramaters for each comb:
combPar_cell= cell(nComb,1);
count = 0;

for r = 0 : (nRes - 1)
    srsPar = srsPar_cell{r+1};
    
    if (srsPar.cyclicShift >= 6) && (srsPar.numAntPorts == 4) && (srsPar.combSize == 4)
        
        % antenna ports 0 and 2:
        combPar1 = srsPar;
        combPar1.numAntPorts = 2;
        combPar1.portMapping = [0 2];
        combPar1.resIdx = r;
        
        % antenna ports 1 and 3:
        combPar2 = srsPar;
        combPar2.numAntPorts = 2;
        combPar2.cyclicShift = mod(combPar2.cyclicShift + 3,12);
        combPar2.combOffset = mod(combPar2.combOffset + 2,4);
        combPar2.portMapping = [1 3];
        combPar2.resIdx = r;
        
        % store:
        combPar_cell{count + 1} = combPar1;
        combPar_cell{count + 2} = combPar2;
        count = count + 2;
    else
        
        % all antenna ports on same comb:
        combPar = srsPar;
        combPar.portMapping = 0 : (combPar.numAntPorts - 1);
        combPar.resIdx = r;
        
        % store:
        combPar_cell{count + 1} = combPar;
        count = count + 1;
    end
end


%%
%COMPUTE

s_tx = zeros(4,Nf,nSym,nRes);

for c = 0 : (nComb-1)
    combPar = combPar_cell{c+1};
    s_tx = srs_tx_kernel(s_tx,combPar);
end



end






