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

function deRmDeItlDynDesc = compute_polDeRmDeItlDynDesc(polarUciSegPrms)

% Function computes constants needed to polar deRm + deItl kernel

%inputs:
% polarUciSegPrms --> polar uci segment paramaters

%outputs:
% deRmDeItlDynDesc --> constants for polar deRm + deItl kernel

%%
%PARAMATERS

K_cw = polarUciSegPrms.K_cw; % number input bits per codeword (data + crc + possible zero insertion)
N_cw = polarUciSegPrms.N_cw; % polar codeword size (power of two)
E_cw = polarUciSegPrms.E_cw; % number of transmit bits per codeword

%%
%START

% Determine rate-matching method:
if E_cw >= N_cw
    rmMethod  = 0;
else
    if K_cw/E_cw <= 7/16
        rmMethod = 1;
    else
        rmMethod = 2;
    end
end

% Interleaved matrix buffer size:
T     = ceil((-1 + sqrt(1 + 8*E_cw)) / 2);

% Region 1 boundaries:
b                  = -(1 + 2*T);
lastRmIdx          = E_cw - 1;
lastRowIdxRegion1  = floor((-b - sqrt(b^2 - 8*lastRmIdx)) / 2);
lastColIdxRegion1  = lastRmIdx - lastRowIdxRegion1 * T + (lastRowIdxRegion1 - 1) * lastRowIdxRegion1 / 2;

% Region 1 sizes:
nBitsRegion1       = (lastRowIdxRegion1 + 1) * (lastColIdxRegion1 + 1);
nRowsRegion1       = lastRowIdxRegion1 + 1;
nColsRegion1       = lastColIdxRegion1 + 1;

% Region 2 sizes:
nRowsRegion2       = nRowsRegion1 - 1;
nColsRegion2       = ((T - nRowsRegion2 + 1) - nColsRegion1);
nBitsRegion1And2   = nBitsRegion1 + nColsRegion2 * nRowsRegion2;     

%%
%WRAP

deRmDeItlDynDesc = [];

deRmDeItlDynDesc.subBlockSize      = N_cw / 32;
deRmDeItlDynDesc.rmMethod          = rmMethod;
deRmDeItlDynDesc.T                 = T;
deRmDeItlDynDesc.nRowsRegion1      = nRowsRegion1;
deRmDeItlDynDesc.nColsRegion1      = nColsRegion1;
deRmDeItlDynDesc.nBitsRegion1      = nBitsRegion1;
deRmDeItlDynDesc.nRowsRegion2      = nRowsRegion2;
deRmDeItlDynDesc.nBitsRegion1And2  = nBitsRegion1And2;


end





