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

function ssburst = hSSBGen(ssb, carrier, ssb_type, SysPar)

N_ID_CELL = carrier.N_ID_CELL;
mu = carrier.mu;
fc = carrier.carrierFreq;
duplex = carrier.duplex;
caseType = ssb_type;
n_hf = ssb.n_hf;
pbch_payload = ssb.pduList{1}.bchPayload;
pbch_payload = dec2bin(pbch_payload, 24)' - '0';
forceSlotIdxFlag=SysPar.SimCtrl.genTV.forceSlotIdxFlag;
if forceSlotIdxFlag
    SFN = carrier.idxFrame;
else
    SFN = carrier.idxFrame-1;    
end
k_ssb = ssb.ssbSubcarrierOffset;

ssblock = zeros([240 4]);
ncellid = N_ID_CELL;

% generate PSS
pssSymbols = nrPSS(ncellid);
pssIndices = nrPSSIndices;

% generate SSS
sssSymbols = nrSSS(ncellid);
sssIndices = nrSSSIndices;

% generate PBCH indices
pbchIndices = nrPBCHIndices(ncellid);

% generate DMRS indices
dmrsIndices = nrPBCHDMRSIndices(ncellid);
% ssblock(dmrsIndices) = dmrsSymbols;

nSubframes = 10;
symbolsPerSlot = 14;
nSymbols = symbolsPerSlot * 2^mu * nSubframes;
nSymPerSubframe = symbolsPerSlot * 2^mu;

firstSymbolIndex = findSsbSymIdx(fc, duplex, caseType, SysPar.ssb);
L_max = length(firstSymbolIndex);

ssburst = zeros([240 nSymbols]);
ssblock = zeros([240 4]);
ssblock(pssIndices) = pssSymbols;
ssblock(sssIndices) = sssSymbols;

for ssbIndex = 1:length(firstSymbolIndex)
    if SysPar.ssb.ssbBitMap(ssbIndex)
        if L_max == 4
            i_SSB = mod(ssbIndex-1,4);
            ibar_SSB = i_SSB + 4*n_hf;
        else
            i_SSB = mod(ssbIndex-1,8);
            ibar_SSB = i_SSB;
        end
        v = i_SSB;

        if L_max == 64
            cw = nrBCH(pbch_payload,SFN,n_hf,L_max,ssbIndex-1,ncellid);
        else
            cw = nrBCH(pbch_payload,SFN,n_hf,L_max, k_ssb, ncellid);
        end

        pbchSymbols = nrPBCH(cw,ncellid,v);
        ssblock(pbchIndices) = pbchSymbols;

        dmrsSymbols = nrPBCHDMRS(ncellid,ibar_SSB);
        ssblock(dmrsIndices) = dmrsSymbols;
        
        if ssb.betaPss == 1      
            beta = 10^(3/20);
        elseif ssb.betaPss == 0
            beta = 1;
        else
            error('betaPss is not supported ...\n');
        end
        ssblock(:, 1) = beta * ssblock(:, 1);
        
        ssburst(:,firstSymbolIndex(ssbIndex) + (0:3) + 1) = ssblock;
    end
end

if forceSlotIdxFlag
    slotIdx = SysPar.SimCtrl.genTV.slotIdx(1);
    ssburst = ssburst(:, slotIdx*14+1:(slotIdx+1)*14);
end

return;
