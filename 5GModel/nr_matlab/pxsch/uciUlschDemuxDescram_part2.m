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

function [csi2_LLR_descr, ulsch_LLR_descr]= uciUlschDemuxDescram_part2(...
    csi2_LLR_descr, ulsch_LLR_descr, eqOutLLRs, uciOnPuschDeMuxDescr, ...
    nBitsCsi2, numDmrsCdmGrpsNoData, isDataPresent)
%%
%INPUT

harqPunctFlag  = uciOnPuschDeMuxDescr.harqPunctFlag;
nBitsPerRe     = uciOnPuschDeMuxDescr.nBitsPerRe;
nResPerSym     = uciOnPuschDeMuxDescr.nResPerSym;
nDataSym       = uciOnPuschDeMuxDescr.nDataSym;
nDmrsSym       = uciOnPuschDeMuxDescr.nDmrsSym;
symIdx_data    = uciOnPuschDeMuxDescr.symIdx_data;
symIdx_dmrs    = uciOnPuschDeMuxDescr.symIdx_dmrs;
G_harq         = uciOnPuschDeMuxDescr.G_harq;
n_rnti         = uciOnPuschDeMuxDescr.n_rnti;
N_id           = uciOnPuschDeMuxDescr.N_id;
qam            = uciOnPuschDeMuxDescr.qam;
nl             = uciOnPuschDeMuxDescr.nl;

rvdHarqReGrids = uciOnPuschDeMuxDescr.rvdHarqReGrids;
harqReGrids    = uciOnPuschDeMuxDescr.harqReGrids;
csi1ReGrids    = uciOnPuschDeMuxDescr.csi1ReGrids;
csi2ReGrids    = uciOnPuschDeMuxDescr.csi2ReGrids;
schReGrids     = uciOnPuschDeMuxDescr.schReGrids;

if numDmrsCdmGrpsNoData == 1
    eqOutLLRs = reshape(eqOutLLRs, nBitsPerRe, nResPerSym, nDataSym+nDmrsSym);
else
    eqOutLLRs = reshape(eqOutLLRs, nBitsPerRe, nResPerSym, nDataSym);
    eqOutLLRs_temp = zeros(nBitsPerRe, nResPerSym, nDataSym+nDmrsSym);
    symIdx_first = min(symIdx_data(1), symIdx_dmrs(1));
    eqOutLLRs_temp(:,:,symIdx_data-symIdx_first+1) = eqOutLLRs;    
    eqOutLLRs = eqOutLLRs_temp;
end

%%
%SCRAMBLING SEQUENCE

c_init = n_rnti*2^15 + N_id;
if numDmrsCdmGrpsNoData == 1
    c      = build_Gold_sequence(c_init, nBitsPerRe * nResPerSym * (nDataSym + nDmrsSym/2));
    c2 = zeros(nBitsPerRe, nResPerSym, nDataSym + nDmrsSym);
    idx_c = 0;
    symAll = [min(symIdx_data(1), symIdx_dmrs(1)) : max(symIdx_data(end), symIdx_dmrs(end))];
    symIdx_first = symAll(1);
    for symIdx = symAll
        if ismember(symIdx, symIdx_dmrs)
            c1 = c(idx_c + 1 : idx_c + nBitsPerRe * nResPerSym * 0.5);
            idx_c = idx_c + nBitsPerRe * nResPerSym * 0.5;
            c2(:, 2:2:end, symIdx - symIdx_first + 1) = reshape(c1, nBitsPerRe, nResPerSym/2);
        else
            c1 = c(idx_c + 1 : idx_c + nBitsPerRe * nResPerSym);
            idx_c = idx_c + nBitsPerRe * nResPerSym;
            c2(:, :, symIdx - symIdx_first + 1) = reshape(c1, nBitsPerRe, nResPerSym);
        end
    end
    c = c2;
else
    c      = build_Gold_sequence(c_init, nBitsPerRe * nResPerSym * nDataSym);
    c2 = zeros(nBitsPerRe, nResPerSym, nDataSym + nDmrsSym);
    idx_c = 0;
    symAll = [min(symIdx_data(1), symIdx_dmrs(1)) : max(symIdx_data(end), symIdx_dmrs(end))];
    symIdx_first = symAll(1);
    for symIdx = symAll
        if ~ismember(symIdx, symIdx_dmrs)        
            c1 = c(idx_c + 1 : idx_c + nBitsPerRe * nResPerSym);
            idx_c = idx_c + nBitsPerRe * nResPerSym;
            c2(:, :, symIdx - symIdx_first + 1) = reshape(c1, nBitsPerRe, nResPerSym);
        end
    end
    c = c2;
end

%%
%START

for symIdx = 0 : (nDataSym + nDmrsSym- 1)
    for reIdx = 0 : (nResPerSym - 1)  
        virtualReIdx  = reIdx;
        rvdHarqReFlag = 0;

        if(harqReGrids{symIdx + 1}.nRes > 0)
            % check if RE reserved to HARQ. Compute the number of REs
            % reserved to HARQ < reIdx.
            [rvdHarqReFlag, cumltNumRvdRes] = checkAssignment(rvdHarqReGrids{symIdx + 1}, reIdx);

            if(rvdHarqReFlag)
                % Check if RE assigend to HARQ. Compute the number of
                % REs assigned to HARQ < reIdx
                [assignFlag, cumltNumAssignedRes] = checkAssignment(harqReGrids{symIdx + 1}, cumltNumRvdRes);
                if(assignFlag)
                    continue;
                end
            else
                % Otherwise update virtualReIdx (RE index after all RVD HARQ REs removed)
                virtualReIdx = virtualReIdx - cumltNumRvdRes;
            end
        end

        if((csi1ReGrids{symIdx + 1}.nRes > 0) && (~rvdHarqReFlag))
            % Check if RE assigend to CSI-P1. Compute the number of CSI-P1 REs
            % assigned < reIdx.
            [assignFlag, cumltNumAssignedRes] = checkAssignment(csi1ReGrids{symIdx + 1}, virtualReIdx);

            if(assignFlag)
                continue;
            else
                % Otherwise update virtualReIdx 
                if(harqPunctFlag)
                    virtualReIdx = reIdx - cumltNumAssignedRes;        % RE index after all CSI-P1 REs removed (HARQ puncturing means REs assigned to both HARQ and SCH/CSI-P2)
                else
                    virtualReIdx = virtualReIdx - cumltNumAssignedRes; % RE index after all HARQ + CSI-P1 REs removed
                end
            end
        else
            if(harqPunctFlag) 
                virtualReIdx = reIdx; 
            end
        end
        
        if(csi2ReGrids{symIdx + 1}.nRes > 0)
            % Check if RE assigend to CSI-P2. Compute the number of CSI-P2 REs
            % assigned < reIdx.
            [assignFlag, cumltNumAssignedRes] = checkAssignment(csi2ReGrids{symIdx + 1}, virtualReIdx);

            if(assignFlag)
                % If assigned, descramble RE LLRs and save to CSI-P2 rateMatch LLR buffer
                reLLRs = eqOutLLRs(:,reIdx+1,symIdx+1);
                if((nBitsCsi2 == 1) && (qam > 1))
                    reLLRs = descrambleReLLRs_1bit_spx(reLLRs, reIdx, symIdx, c, nl, qam);
                else
                    reLLRs = descrambleReLLRs_standard(reLLRs, reIdx, symIdx, c);
                end

                rmBufferIdxs = csi2ReGrids{symIdx + 1}.rmBufferOffset + nBitsPerRe*cumltNumAssignedRes + (0 : (nBitsPerRe - 1));  
                csi2_LLR_descr(rmBufferIdxs + 1) = reLLRs;
                continue;
            else
                % Otherwise update virtualReIdx 
                virtualReIdx = virtualReIdx - cumltNumAssignedRes; % RE index after all HARQ + CSI-P1 + CSI-P2 REs removed (if no HARQ puncturing)
                                                                   % RE index after all CSI-P1 + CSI-P2 REs removed (if HARQ puncturing)
            end
        end
        
        % Descramble RE LLRs and save to SCH rateMatch LLR buffer
        if schReGrids{symIdx + 1}.ReStride == 2
            if mod(reIdx, 2) == 0 % dmrs RE on DMRS symbols
                continue;
            else
                virtualReIdx = (reIdx-1)/2;
            end
        end
        
        reLLRs = eqOutLLRs(:,reIdx+1,symIdx+1);
        reLLRs = descrambleReLLRs_standard(reLLRs, reIdx, symIdx, c);
        
        rmBufferIdxs = schReGrids{symIdx + 1}.rmBufferOffset + nBitsPerRe*virtualReIdx + (0 : (nBitsPerRe - 1));
        ulsch_LLR_descr(rmBufferIdxs + 1) = reLLRs;
    end
end

end


function [assignFlag, cumltNumAssignedRes] = checkAssignment(symRmBitsToResMapping, virtualReIdx)
    
    r = mod(virtualReIdx,symRmBitsToResMapping.ReStride);
    d = floor(virtualReIdx / symRmBitsToResMapping.ReStride);
    
    % Check if RE is assigned to the RM buffer
    if(r ~= 0)
        assignFlag = 0;
        if((d + 1) >= symRmBitsToResMapping.nRes)
            cumltNumAssignedRes = symRmBitsToResMapping.nRes;
        else
            cumltNumAssignedRes = (d + 1);
        end
    elseif(d >= symRmBitsToResMapping.nRes)
        assignFlag       = 0;
        cumltNumAssignedRes = symRmBitsToResMapping.nRes;
    else
        cumltNumAssignedRes = d;
        assignFlag          = 1;
    end
end


function reLLRs = descrambleReLLRs_standard(reLLRs, reIdx, symIdx, c)
    reLLRs = reLLRs .* (1 - 2 * c(:,reIdx + 1, symIdx + 1));
end


function reLLRs = descrambleReLLRs_1bit_spx(reLLRs, reIdx, symIdx, c, nl, qam)
    for layerIdx = 0 : (nl - 1)
        cc = c(layerIdx * qam + 1, reIdx + 1, symIdx + 1);
        reLLRs(layerIdx * qam + 1) = reLLRs(layerIdx * qam + 1) * (1 - 2 * cc);
        reLLRs(layerIdx * qam + 2) = reLLRs(layerIdx * qam + 2) * (1 - 2 * cc);
    end
end
