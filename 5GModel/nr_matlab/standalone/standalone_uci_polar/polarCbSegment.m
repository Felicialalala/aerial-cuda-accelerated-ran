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

function polCbs = polarCbSegment(polarUciSegPrms, uciSegPayload)

% Function segments a uci payload into one or two polar codeblocks

%inputs:
% polarUciSegPrm --> polar uci segment paramaters
% uciPayload     --> uci segment payload. Dim: A_seg x 1

%outputs:
% polCbs         --> polar codeblock(s) + crc bits + possible zero inserstion. Dim: K_cw x nCbs

%%
%PARAMATERS

nCbs           = polarUciSegPrms.nCbs;
zeroInsertFlag = polarUciSegPrms.zeroInsertFlag;
K_cw           = polarUciSegPrms.K_cw;
nCrcBits       = polarUciSegPrms.nCrcBits;

%%
%START
% Following 38.212 sections 6.3.1.2 and 5.2.1

nInfoCbBits = K_cw - nCrcBits;
polCbs      = zeros(nInfoCbBits, nCbs);

if (nCbs == 1)
    polCbs = uciSegPayload;
else
    polCbs(zeroInsertFlag + 1 : end, 1) = uciSegPayload(1 : (nInfoCbBits - zeroInsertFlag));
    polCbs(:, 2)                        = uciSegPayload(nInfoCbBits - zeroInsertFlag + 1 : end);
end


end

    
