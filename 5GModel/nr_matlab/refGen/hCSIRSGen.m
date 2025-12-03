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

function Xtf = hCSIRSGen(csirs, carrier, table)

global SimCtrl

pduList = csirs.pduList;
nPdu = length(pduList);

Xtf = zeros(carrier.N_sc, carrier.N_symb_slot, carrier.numTxPort);
Xtf = Xtf(:);

for idxPdu = 1:nPdu
    pdu = pduList{idxPdu};
    
    carrier5g = nrCarrierConfig;
    carrier5g.NCellID = carrier.N_ID_CELL;
    carrier5g.NSizeGrid = carrier.N_grid_size_mu;
    carrier5g.NStartGrid = carrier.N_grid_start_mu;
    
    csirs5g = nrCSIRSConfig;
    switch pdu.CSIType
        case 0
            csirs5g.CSIRSType = 'nzp';
        case 1
            csirs5g.CSIRSType = 'nzp';
        case 2
            csirs5g.CSIRSType = 'zp';
    end
    
    csirs5g.RowNumber = pdu.Row;
    
    switch pdu.FreqDensity
        case 0
            csirs5g.Density = 'dot5even';
        case 1
            csirs5g.Density = 'dot5odd';
        case 2
            csirs5g.Density = 'one';
        case 3
            csirs5g.Density = 'three';
    end
    
    RowWithTwoSym = [13, 14, 16, 17];
    if ismember(pdu.Row, RowWithTwoSym)
        csirs5g.SymbolLocations = [pdu.SymbL0, pdu.SymbL1];
    else
        csirs5g.SymbolLocations = pdu.SymbL0;
    end
    nK = [1 1 1 1 1 4 2 2 6 3 4 4 3 3 3 4 4 4];
    Ports = [1 1 2 4 4 8 8 8 12 12 16 16 24 24 24 32 32 32];
    csirs5g.SubcarrierLocations = 2*[0:nK(csirs5g.RowNumber)-1];
    
    FreqDomain_bin = dec2bin(pdu.FreqDomain, 12) - '0';
    FreqDomain_flip = fliplr(FreqDomain_bin);
    switch pdu.Row
        case 1
            idxOne = find(FreqDomain_flip(1:4));
            ki =idxOne - 1;
        case 2
            idxOne = find(FreqDomain_flip(1:12));
            ki = idxOne - 1;
        case 4
            idxOne = find(FreqDomain_flip(1:3));
            ki = 4*(idxOne - 1);
        otherwise
            idxOne = find(FreqDomain_flip(1:6));
            ki = 2*(idxOne - 1);
    end
    csirs5g.SubcarrierLocations = ki(1:nK(csirs5g.RowNumber));
    
    csirs5g.NumRB = pdu.NrOfRBs;
    csirs5g.RBOffset = pdu.StartRB;
    csirs5g.NID = pdu.ScrambId;
    if SimCtrl.genTV.forceSlotIdxFlag
        carrier5g.NSlot = SimCtrl.genTV.slotIdx(1);
    else
        carrier5g.NSlot = mod((carrier.idxSlot + carrier.idxSubframe * ...
            carrier.N_slot_subframe_mu -1), carrier.N_slot_frame_mu);
    end
    carrier5g.SubcarrierSpacing = 15*2^carrier.mu;
    
    [ind,info_ind] = nrCSIRSIndices(carrier5g,csirs5g);
    [sym,info] = nrCSIRS(carrier5g,csirs5g);
        
    beta_db = (pdu.powerControlOffsetSS - 1)*3;
    beta = 10^(beta_db/20);
    sym = beta*sym;
    
    Xtf(ind) = sym;
end

Xtf = reshape(Xtf, [carrier.N_sc, carrier.N_symb_slot, carrier.numTxPort]);

return
