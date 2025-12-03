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

function [dcicw, sym, cgrid_all] = hPDCCHGen(mypdcchList, mycarrier)


cgrid_all = zeros(12*mycarrier.N_grid_size_mu,mycarrier.N_symb_slot);


nPdu = length(mypdcchList);

for idxPdu = 1:nPdu
    mypdcch = mypdcchList{idxPdu};
    
    nDCI = mypdcch.numDlDci;
    
    OccupiedSC = zeros(mycarrier.N_grid_size_mu*12, mycarrier.N_symb_slot);
    N_CCE = sum(mypdcch.coresetMap)*mypdcch.DurationSymbols;
    
    for idxDCI = 1:nDCI
        DCI = mypdcch.DCI{idxDCI};
        aggrL = DCI.AggregationLevel;
        rntiCrc = DCI.RNTI;
        
        if mypdcch.coresetIdx > 0
            rntiBits = DCI.ScramblingRNTI;
            nid = DCI.ScramblingId;
        else
            rntiBits = 0;
            nid = mycarrier.N_ID_CELL;
        end
        E = 2*9*6*aggrL;
        dcicw = nrDCIEncode(DCI.Payload(1:DCI.PayloadSizeBits)',rntiCrc,E);
        sym = nrPDCCH(dcicw,nid,rntiBits);
        
        carrier = nrCarrierConfig;
        
        global SimCtrl
        if SimCtrl.genTV.forceSlotIdxFlag
            carrier.NSlot = SimCtrl.genTV.slotIdx(1);
        else
            carrier.NSlot = mod((mycarrier.idxSlot + mycarrier.idxSubframe * ...
                mycarrier.N_slot_subframe_mu -1), mycarrier.N_slot_frame_mu);
        end
        carrier.NSizeGrid = mycarrier.N_grid_size_mu;
        carrier.NCellID = mycarrier.N_ID_CELL;
        carrier.SubcarrierSpacing = 15*2^mycarrier.mu;
        
        coreset = nrCORESETConfig;
        coreset.Duration = mypdcch.DurationSymbols;
        if mypdcch.CceRegMappingType
            coreset.CCEREGMapping = 'interleaved';
        else
            coreset.CCEREGMapping = 'noninterleaved';
        end
        
        coreset.FrequencyResources = mypdcch.coresetMap;
        if mypdcch.coresetIdx == 0
            coreset.REGBundleSize = 6;
            coreset.InterleaverSize = 2;
            coreset.ShiftIndex = mycarrier.N_ID_CELL;
        elseif strcmp(coreset.CCEREGMapping, 'interleaved')
            coreset.REGBundleSize = mypdcch.RegBundleSize;
            coreset.InterleaverSize = mypdcch.InterleaverSize;
            coreset.ShiftIndex = mypdcch.ShiftIndex;
        end
        
        
        coreset.CORESETID = mypdcch.coresetIdx;
        
        pdcch = nrPDCCHConfig;
        pdcch.NStartBWP = mypdcch.BWPStart;
        pdcch.NSizeBWP = mypdcch.BWPSize;
        
        if mypdcch.coresetIdx > 0
            pdcch.RNTI = DCI.RNTI; % it will impact the freq location of CCE
            pdcch.DMRSScramblingID = DCI.ScramblingId;
        else
            pdcch.RNTI = 0; % it will impact the freq location of CCE
            pdcch.DMRSScramblingID = mycarrier.N_ID_CELL;
            % Add this for Matlab R2021b, which only uses FrequencyResources
            % (no NStartBWP) to find Coreset RB location when CORESETID = 0.
            % Matlab R2020a uses both FrequencyResources and NStartBWP.
            if mod(pdcch.NStartBWP, 6) == 0
                %             RB6 = pdcch.NStartBWP/6;
                %             coreset.FrequencyResources = [zeros(1,RB6), coreset.FrequencyResources];
                pdcch.NStartBWP = 0;
                pdcch.NSizeBWP = mycarrier.N_grid_size_mu;
            else
                error('CORESET0 NStartBWP is not a multiple of 6 ...\n');
            end
        end
        pdcch.AggregationLevel = aggrL;
        pdcch.CORESET = coreset;
        
        pdcch.SearchSpace.StartSymbolWithinSlot = mypdcch.StartSymbolIndex;
        NumCandidates = floor(N_CCE./[1 2 4 8 16]);
        for k = 1:5
            if NumCandidates(k) > 8
                NumCandidates(k) = 8;
            elseif NumCandidates(k) == 8
                NumCandidates(k) = 6;
            end
            % Add to fix Matlab Error using nrSearchSpaceConfig/checkAL (line 185)
            % Element 2 of vector NumCandidates for aggregation level (2)
            % must be 0, 1, 2, 3, 4, 5, 6, or 8.
            if k == 2 || NumCandidates(k) == 7
                NumCandidates(k) = 6;
            end
        end
        pdcch.SearchSpace.NumCandidates = NumCandidates;
        pdcch.SearchSpace.CORESETID = mypdcch.coresetIdx;
        isCSS = mypdcch.isCSS;
        % Add to fix Matlab error:
        % RNTI must be equal to 0 for a 'common' SearchSpace.
        if isCSS
            pdcch.RNTI = 0;
        end
        if mypdcch.coresetIdx > 0 && ~isCSS
            pdcch.SearchSpace.SearchSpaceType = 'ue';
        else
            pdcch.SearchSpace.SearchSpaceType = 'common';
        end
        [allInd,allDMRSSym,allDMRSInd] = nrPDCCHSpace(carrier,pdcch);
        idxAggrL = round(log2(aggrL))+1;
        candidateSC =  allInd{idxAggrL};
        findCceIdx = 0;
        for idxCandidate = 1:pdcch.SearchSpace.NumCandidates(idxAggrL)
            occupied = sum(OccupiedSC(candidateSC(:, idxCandidate)));
            if ~ occupied
                OccupiedSC(candidateSC(:, idxCandidate)) = 1;
                findCceIdx = 1;
                break;
            end
        end
        if findCceIdx == 0
            error('DCI can not be allocated ...\n');
        end
        
%         qam_dB = (DCI.powerControlOffsetSS - 1) * 3;
        qam_dB = DCI.powerControlOffsetSSProfileNR;
        beta_qam = 10^(qam_dB/20);
%         dmrs_dB = (DCI.powerControlOffsetSS - 1) * 3;
        dmrs_dB = DCI.powerControlOffsetSSProfileNR;
        beta_dmrs = 10^(dmrs_dB/20);
        
        cgrid = zeros(12*mycarrier.N_grid_size_mu,mycarrier.N_symb_slot);
        cgrid(allInd{idxAggrL}(:,idxCandidate)) = beta_qam*sym;
        cgrid(allDMRSInd{idxAggrL}(:,idxCandidate)) = beta_dmrs*allDMRSSym{idxAggrL}(:,idxCandidate);
        
        cgrid_all = cgrid_all + cgrid;
        
    end
    
    % move coreset0 to the expected RB location (BWPStart)
    if mypdcch.coresetIdx == 0
        nRB_coreset0 = find(coreset.FrequencyResources, 1, 'last')*6;
        symIdx_coreset0 = [(mypdcch.StartSymbolIndex+1):mypdcch.DurationSymbols];
        cgrid_coreset0 = cgrid_all(1:nRB_coreset0*12, symIdx_coreset0);
        cgrid_all(1:nRB_coreset0*12, symIdx_coreset0) = 0;
        cgrid_all(mypdcch.BWPStart*12+(1:nRB_coreset0*12), symIdx_coreset0) = cgrid_coreset0;
    end
    
end

return


