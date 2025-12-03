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

function uciSegEncoded = uciSegPolarEncode(A_seg, E_seg, uciSegPayload)

% function fully polar encodes a uci segment: cb-segmentation + crc-attach +
% polar-cb-encode + block-interleave + rate-match + output-interleave + cb-concatenation

%inputs: 
% A_seg         --> number of uci segment information bits
% E_seg         --> number of uci segment transmit bits
% uciSegPayload --> uci segment payload. Dim: A_seg x 1

%outputs:
% uciSegEncoded --> polar encoded uci segment. Dim: E_seg x 1

%%
%START

% derive polar uci segment paramaters:
polarUciSegPrms = derive_polarUciSegPrms(A_seg, E_seg);

% cb segment:
polCbs = polarCbSegment(polarUciSegPrms, uciSegPayload);

% crc attachment:
crcStr           = num2str(polarUciSegPrms.nCrcBits);
polCbsCrcEncoded = zeros(polarUciSegPrms.K_cw, polarUciSegPrms.nCbs);

for cbIdx = 0 : (polarUciSegPrms.nCbs - 1)
    polCbsCrcEncoded(: ,cbIdx + 1) = add_CRC(polCbs(:, cbIdx + 1), crcStr);
end

% polar encode codeblocks:
cwBitTypes = compBitTypesKernel(polarUciSegPrms);
polCws     = zeros(polarUciSegPrms.N_cw, polarUciSegPrms.nCbs);

for cbIdx = 0 : (polarUciSegPrms.nCbs - 1)
    polCws(:, cbIdx + 1) = uplinkPolarCbEncoder(polarUciSegPrms, cwBitTypes, polCbsCrcEncoded(:, cbIdx + 1));
end

% rate-matching + interleave:
polCwsRmItl = zeros(polarUciSegPrms.E_cw, polarUciSegPrms.nCbs);

for cbIdx = 0 : (polarUciSegPrms.nCbs - 1)
    polCwsRmItl(:, cbIdx + 1) = uplinkPolarRmItl(polarUciSegPrms, polCws(:, cbIdx + 1));
end

% cb concatenation:
uciSegEncoded = zeros(polarUciSegPrms.E_seg,1);

if(polarUciSegPrms.nCbs == 1)
    uciSegEncoded = polCwsRmItl(:,1);
else
    E_cw = polarUciSegPrms.E_cw;
    uciSegEncoded(1 : E_cw)            = polCwsRmItl(:,1);
    uciSegEncoded((E_cw + 1) : 2*E_cw) = polCwsRmItl(:,2);
end
    











