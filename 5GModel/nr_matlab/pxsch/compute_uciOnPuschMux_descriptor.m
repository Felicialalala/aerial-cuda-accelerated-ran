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

function uciOnPuschDeMuxDescr = compute_uciOnPuschMux_descriptor(nPrb, ...
    nDataSym, symIdx_data, symIdx_dmrs, nl, Qm, G_harq, G_harq_rvd, ...
    G_csi1, G_csi2, G_sch, nHarqBits, N_id, n_rnti,  numDmrsCdmGrpsNoData,...
    isDataPresent)

%%
%CONSTANTS

nBitsPerRe  =  nl * Qm;
nResPerSym  =  12 * nPrb;
nBitsPerSym =  nBitsPerRe * nResPerSym;
nDmrsSym    =  length(symIdx_dmrs);
nSym = nDmrsSym + nDataSym;
%%
%INITALIZE GRIDS

rvdHarqReGrids = cell(nSym,1);
harqReGrids    = cell(nSym,1);
csi1ReGrids    = cell(nSym,1);
csi2ReGrids    = cell(nSym,1);
schReGrids     = cell(nSym,1);

for i = 1 : nSym
    reGrid = {};
    reGrid.nRes           = 0;
    reGrid.ReStride       = 0;
    reGrid.rmBufferOffset = 0;
    
    rvdHarqReGrids{i} =  reGrid;
    harqReGrids{i}    =  reGrid;
    csi1ReGrids{i}    =  reGrid;
    csi2ReGrids{i}    =  reGrid;
    schReGrids{i}     =  reGrid;
end


%%
% MAP OFDM SYMBOLS TO UCI

startDataSymHarq = 0;
for i = 0 : (nDataSym - 1)
    if(symIdx_data(i + 1) > symIdx_dmrs(1))
        startDataSymHarq = i;
        break;
    end
end

startDataSymCsi1 = 0;
startDataSymCsi2 = 0;
startDataSymSch  = 0;

%%
%DETERMINE IF PUNCTURING USED

harqPunctFlag = 0;
if(nHarqBits <= 2) && (nHarqBits > 0)
    harqPunctFlag = 1;
elseif nHarqBits == 0
    harqPunctFlag = 1;
    G_harq = G_harq_rvd;
else
    G_harq_rvd = G_harq;
end

%%
%START
nAssignedHarqRvdRmBits   = 0;
nAssignedHarqRmBits      = 0;
nAssignedCsi1RmBits      = 0;
nAssignedCsi2RmBits      = 0;
nAssignedSchRmBits       = 0;

nUnassignedBitsInSymbol = ones(nDataSym + nDmrsSym,1) * nBitsPerSym;
nUnassignedResInSymbol  = ones(nDataSym + nDmrsSym,1) * nResPerSym;

symIdx_last = max(symIdx_data(end), symIdx_dmrs(end));
symIdx_first = min(symIdx_data(1), symIdx_dmrs(1));
symIdx_harq = symIdx_data(startDataSymHarq+1);

% first determine RE grids reserved for HARQ
for symIdx = symIdx_harq - symIdx_first : symIdx_last - symIdx_first 
    if(nAssignedHarqRvdRmBits > G_harq_rvd)
        break;
    end
    if(nUnassignedBitsInSymbol(symIdx + 1) == 0) || ismember(symIdx+symIdx_first, symIdx_dmrs)
        continue;
    end
    [rvdHarqReGrids{symIdx + 1}, nAssignedHarqRvdRmBits] = assign_RE_grid(nUnassignedBitsInSymbol(symIdx + 1), nUnassignedResInSymbol(symIdx + 1), ...
                                                                          nAssignedHarqRvdRmBits, G_harq_rvd, nBitsPerRe);

    [nUnassignedBitsInSymbol(symIdx + 1), nUnassignedResInSymbol(symIdx + 1)] = update_nUnassigned(nUnassignedBitsInSymbol(symIdx + 1), ...
                                                                                                    nUnassignedResInSymbol(symIdx + 1), ...
                                                                                                    rvdHarqReGrids{symIdx + 1},...
                                                                                                    nBitsPerRe);
end

% HARQ RM bits assigned to sub-grid of RVD grid
for symIdx = symIdx_harq - symIdx_first : symIdx_last - symIdx_first 
    nRvdRes  = rvdHarqReGrids{symIdx + 1}.nRes;
    nRvdBits = nRvdRes * nBitsPerRe;
    if(nAssignedHarqRmBits > G_harq)
        break;
    end
    if(nRvdBits == 0) || ismember(symIdx+symIdx_first, symIdx_dmrs)
        continue;
    end
    [harqReGrids{symIdx + 1}, nAssignedHarqRmBits] = assign_RE_grid(nRvdBits, nRvdRes, nAssignedHarqRmBits,...
                                                                    G_harq, nBitsPerRe);
end



% assign csi-p1 rm bits:
for symIdx = 0: symIdx_last - symIdx_first
    if(nAssignedCsi1RmBits >= G_csi1)
        break;
    end
    if(nUnassignedBitsInSymbol(symIdx + 1) == 0) || ismember(symIdx+symIdx_first, symIdx_dmrs)
        continue;
    end
    
    [csi1ReGrids{symIdx + 1}, nAssignedCsi1RmBits] = assign_RE_grid(nUnassignedBitsInSymbol(symIdx + 1), nUnassignedResInSymbol(symIdx + 1), ...
                                                                    nAssignedCsi1RmBits, G_csi1, nBitsPerRe); 
end


% update number of bits avaliable for assignment to SCH and CSI-P2
for symIdx = 0: symIdx_last - symIdx_first
    if ismember(symIdx+symIdx_first, symIdx_dmrs)
        continue;
    end
    if(harqPunctFlag) %If HARQ puncturing enabled, REs assigned to HARQ are also "assigned" to SCH or CSI-P2.
        [nUnassignedBitsInSymbol(symIdx + 1), nUnassignedResInSymbol(symIdx + 1)] = update_nUnassigned(nBitsPerSym,...
                                                                                                       nResPerSym,...
                                                                                                       csi1ReGrids{symIdx + 1},...
                                                                                                       nBitsPerRe);
    else
         [nUnassignedBitsInSymbol(symIdx + 1), nUnassignedResInSymbol(symIdx + 1)] = update_nUnassigned(nUnassignedBitsInSymbol(symIdx + 1), ...
                                                                                                        nUnassignedResInSymbol(symIdx + 1), ...
                                                                                                        csi1ReGrids{symIdx + 1}, ...
                                                                                                        nBitsPerRe);
    end
end

% assign csi-p2 rm bits:
for symIdx = 0: symIdx_last - symIdx_first 
    if(nAssignedCsi2RmBits >= G_csi2)
        break;
    end
    if(nUnassignedBitsInSymbol(symIdx + 1) == 0) || ismember(symIdx+symIdx_first, symIdx_dmrs)
        continue;
    end
    
    [csi2ReGrids{symIdx + 1}, nAssignedCsi2RmBits]  = assign_RE_grid(nUnassignedBitsInSymbol(symIdx + 1),...
                                                                     nUnassignedResInSymbol(symIdx + 1),...
                                                                     nAssignedCsi2RmBits,...
                                                                     G_csi2,...
                                                                     nBitsPerRe);
                                                                           
    [nUnassignedBitsInSymbol(symIdx + 1), nUnassignedResInSymbol(symIdx + 1)] = update_nUnassigned(nUnassignedBitsInSymbol(symIdx + 1), ...
                                                                                                   nUnassignedResInSymbol(symIdx + 1),...
                                                                                                   csi2ReGrids{symIdx + 1},...
                                                                                                   nBitsPerRe);
end

nDmrsSym = length(symIdx_dmrs);

% assign sch rm bits:

for symIdx = 0: symIdx_last - symIdx_first 
    if(nAssignedCsi2RmBits >= G_sch)
        break;
    end
    if(nUnassignedBitsInSymbol(symIdx + 1) == 0)
        if ~(numDmrsCdmGrpsNoData == 1 && isDataPresent && ismember(symIdx+symIdx_first, symIdx_dmrs))
            continue;
        end
    end
    
    reGrid = {};
    reGrid.rmBufferOffset = nAssignedSchRmBits;
    if ismember(symIdx+symIdx_first, symIdx_dmrs)
        if numDmrsCdmGrpsNoData == 1 && isDataPresent
            reGrid.nRes           = nResPerSym/2;
            reGrid.ReStride       = 2;
        else
            reGrid.nRes           = 0;
            reGrid.ReStride       = 1;
        end    
    else
        reGrid.nRes           = nUnassignedResInSymbol(symIdx + 1);
        reGrid.ReStride       = 1;
    end   

    schReGrids{symIdx + 1} = reGrid;
    nAssignedSchRmBits     = nAssignedSchRmBits + reGrid.nRes * nBitsPerRe;
end
    
uciOnPuschDeMuxDescr = {};

uciOnPuschDeMuxDescr.nBitsPerRe    = nBitsPerRe;
uciOnPuschDeMuxDescr.nResPerSym    = nResPerSym;
uciOnPuschDeMuxDescr.nDataSym      = nDataSym;
uciOnPuschDeMuxDescr.nDmrsSym      = nDmrsSym;
uciOnPuschDeMuxDescr.symIdx_data      = symIdx_data;
uciOnPuschDeMuxDescr.symIdx_dmrs      = symIdx_dmrs;
uciOnPuschDeMuxDescr.harqPunctFlag = harqPunctFlag;
uciOnPuschDeMuxDescr.G_harq        = G_harq;
uciOnPuschDeMuxDescr.qam           = Qm;
uciOnPuschDeMuxDescr.nl            = nl;
uciOnPuschDeMuxDescr.N_id          = N_id;
uciOnPuschDeMuxDescr.n_rnti        = n_rnti;

uciOnPuschDeMuxDescr.rvdHarqReGrids = rvdHarqReGrids;
uciOnPuschDeMuxDescr.harqReGrids    = harqReGrids;
uciOnPuschDeMuxDescr.csi1ReGrids    = csi1ReGrids;
uciOnPuschDeMuxDescr.csi2ReGrids    = csi2ReGrids;
uciOnPuschDeMuxDescr.schReGrids     = schReGrids;
    
end


function [reGrid, nAssignedRmBits] = assign_RE_grid(nUnassignedBitsInSymbol, nUnassignedResInSymbol, nAssignedRmBits, G, nBitsPerRe)
    reGrid = {};
    reGrid.rmBufferOffset = nAssignedRmBits;
    
    nUnassignedRmBits = G - nAssignedRmBits;
    if(nUnassignedRmBits > nUnassignedBitsInSymbol)
        reGrid.nRes     = nUnassignedResInSymbol;
        reGrid.ReStride = 1;
    else
        reGrid.nRes     = ceil(nUnassignedRmBits / nBitsPerRe);
        reGrid.ReStride = floor(nUnassignedBitsInSymbol / nUnassignedRmBits);
    end
    
    nAssignedRmBits = nAssignedRmBits + reGrid.nRes * nBitsPerRe;
end

function [nUnassignedBitsInSymbol, nUnassignedResInSymbol] = update_nUnassigned(nUnassignedBitsInSymbol, nUnassignedResInSymbol, symRmBitsToResMapping, nBitsPerRe)
    nUnassignedBitsInSymbol = nUnassignedBitsInSymbol - symRmBitsToResMapping.nRes * nBitsPerRe;
    nUnassignedResInSymbol  = nUnassignedResInSymbol - symRmBitsToResMapping.nRes;
end


        
        
        
        
        
        
        
        
        
        
    
    








        
        
        
        
        
        
        
        
        





