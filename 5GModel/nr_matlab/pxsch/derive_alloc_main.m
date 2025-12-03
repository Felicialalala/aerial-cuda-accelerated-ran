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

function alloc = derive_alloc_main(pdu, dmrs, Xtf_remap)

Nt_dmrs = dmrs.Nt_dmrs;           % number of dmrs symbols
symIdx_dmrs = dmrs.symIdx_dmrs;   % indicies of dmrs symbols. Dim: Nt_dmrs x 1

if pdu.resourceAlloc == 0
%     rbBitmap_perbit = reshape(flipud(int2bit(pdu.rbBitmap,8,true)),[],1); % ordering: VRB0, VRB1,....
    rbBitmap_perbit = reshape(flipud(dec2bin(pdu.rbBitmap,8)')-'0',[],1); % ordering: VRB0, VRB1,....   
    reBitmap_perbit = logical(reshape(repmat(rbBitmap_perbit,[1,12]).',[],1));
    numValidREs = size(Xtf_remap,1);
    rbBitmap_perbit = rbBitmap_perbit(1:numValidREs/12);
    nPrb = sum(rbBitmap_perbit);
else
    nPrb = pdu.rbSize;
    rbBitmap_perbit = [];
end
nSym = pdu.NrOfSymbols;
startSym = pdu.StartSymbolIndex+1;

if symIdx_dmrs(1) < startSym
    error('Start symbol can not be after the first DMRS symbol');
end

symMask_data = zeros(14,1);
symMask_data(startSym : (startSym + nSym - 1)) = 1;
symMask_data(symIdx_dmrs) = 0;
symMask_data = logical(symMask_data);
symIdx_data = (1 : 14).';
symIdx_data(~symMask_data) = [];

Nf_data = 12 * nPrb;
Nt_data = length(symIdx_data);
if strcmp(pdu.type, 'pdsch')
    isDataPresent = 1;
else
    isDataPresent = bitand(uint16(pdu.pduBitmap),uint16(2^0));
end
% half of REs on DMRS symbols can be used for data
if pdu.numDmrsCdmGrpsNoData == 1
    N_data = Nf_data * (Nt_data + Nt_dmrs/2);
else
    N_data = Nf_data * Nt_data; 
end
if nargin == 3
    if pdu.resourceAlloc == 0        
        thisPdsch_remap = Xtf_remap(reBitmap_perbit(1:numValidREs), symIdx_data);
    else
        rbStart = pdu.rbStart;
        thisPdsch_remap = Xtf_remap(rbStart*12+1:(rbStart+nPrb)*12, symIdx_data);
    end
    nRE_reserved = sum(sum(thisPdsch_remap));
    N_data_used = N_data - nRE_reserved;
else
    N_data_used = N_data;
end    

alloc.resourceAlloc = pdu.resourceAlloc;
alloc.rbBitmap = rbBitmap_perbit;
alloc.nPrb = nPrb;
alloc.startPrb = pdu.rbStart + pdu.BWPStart + 1; %TODO remove to reconcile RAT1 with RAT0
alloc.BWPStart = pdu.BWPStart;
alloc.nSym = nSym;
alloc.startSym = startSym;
alloc.Nf_data = Nf_data;
alloc.Nt_data = Nt_data;
alloc.N_data = N_data;
alloc.N_data_used = N_data_used;
alloc.symIdx_data = symIdx_data;
alloc.portIdx = find(flip(pdu.dmrsPorts)); 
alloc.nl = pdu.nrOfLayers;
alloc.RNTI = pdu.RNTI;
alloc.dataScramblingId = pdu.dataScramblingId;
alloc.SCID = pdu.SCID;

return
