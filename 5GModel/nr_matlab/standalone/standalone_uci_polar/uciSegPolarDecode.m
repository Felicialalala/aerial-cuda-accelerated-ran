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

function [uciSegEst, crcErrorFlag, interBuffers] = uciSegPolarDecode(A_seg, E_seg, listLength, uciSegLLRs)

% function fully performs full polar reciever chain:
% de-channel-interleave + de-rate-match + de-block-interleave +
% polar-decode + crc + cb-segment

%inputs: 
% A_seg         --> number of uci segment information bits
% E_seg         --> number of uci segment transmit bits
% listLength    --> listLength for CRC aided list decoder
% uciSegLLRs    --> uci segment LLRs. Dim: E_seg x 1

%outputs:
% uciSegEst    --> estimate of uci segment. Dim: A_seg x 1
% crcErrorFlag --> 1 if any crc fails, 0 otherwise

%%
%START

% derive polar uci segment paramaters:
polarUciSegPrms = derive_polarUciSegPrms(A_seg, E_seg);

% segment LLRs to codeword(s), deInterleave, deRateMatch:
deRmDeItlDynDesc = compute_polDeRmDeItlDynDesc(polarUciSegPrms);
cwLLRs           = pol_cwSeg_deRm_deItl(polarUciSegPrms, deRmDeItlDynDesc, uciSegLLRs);

% compute codeword tree types:
[~, cwTreeTypes] = compBitTypesKernel(polarUciSegPrms);

% run polar decoder:
nBitsPerCb      = polarUciSegPrms.K_cw - polarUciSegPrms.nCrcBits;
cbEsts          = zeros(nBitsPerCb, polarUciSegPrms.nCbs);
cbCrcErrorFlags = zeros(2,1);
crcErrorFlag    = 0;

for cbIdx = 1 : polarUciSegPrms.nCbs
    [cbEsts(:,cbIdx), cbCrcErrorFlags(cbIdx)] = polar_decoder(listLength,polarUciSegPrms.K_cw, polarUciSegPrms.N_cw, ... 
                                                      polarUciSegPrms.nCrcBits, cwLLRs(:,cbIdx), cwTreeTypes);
    
	if(cbCrcErrorFlags(cbIdx))
        crcErrorFlag = 1;
    end                                              
end

% combine cb estimates
if (polarUciSegPrms.nCbs == 1)
    uciSegEst = cbEsts;
else
    uciSegEst = combine_uciCbEsts(cbEsts(:,1), cbEsts(:,2), polarUciSegPrms);
end

%%
%INTERMEDIATE BUFFERS

interBuffers.polarUciSegPrms = polarUciSegPrms;
interBuffers.uciSegLLRs      = uciSegLLRs;
interBuffers.cwTreeTypes     = cwTreeTypes;
interBuffers.cbEsts          = cbEsts;
interBuffers.cwLLRs          = cwLLRs;
interBuffers.cbCrcErrorFlags = cbCrcErrorFlags;

end





    











