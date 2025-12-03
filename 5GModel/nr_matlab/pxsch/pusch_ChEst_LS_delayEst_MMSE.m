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

function [H_est,delay_mean_microsec,delay_spread_microsec,dbg_chest] = pusch_ChEst_LS_delayEst_MMSE(Xtf,puschTable,slotNumber,Nf,Nt,N_dmrs_id,...
        nl, portIdx, vec_scid, dmrsIdx, Nt_dmrs, nPrb, startPrb, ...
        delta_f,maxLength, numDmrsCdmGrpsNoData, enableTfPrcd,...
         N_slot_frame, N_symb_slot, puschIdentity,...
         groupOrSequenceHopping, AdditionalPosition, carrier, prgSize, enable_prg_chest, r_dmrs)
    global SimCtrl
    % initialize results buffer
    H_LS_est = cell(AdditionalPosition+1,1);
    H_est = cell(AdditionalPosition+1,1);
    nRxAnt = size(Xtf,3);
    for posDmrs = 1:AdditionalPosition+1
        H_LS_est{posDmrs} = zeros(size(Xtf, 1), nl, nRxAnt);
        H_est{posDmrs} = zeros(nRxAnt, nl, size(Xtf, 1));
    end
    % get LS ChEst
    tmp_R0 = zeros(AdditionalPosition+1, 1);
    tmp_R1 = zeros(AdditionalPosition+1, 1);
    
    if enable_prg_chest == 1
        if prgSize > 4
            error('Per-PRG ChEst only support PRG size 1, 2, 3, 4!')
        end
        numPRGs_this_ueg = ceil(nPrb/prgSize);
        numResidual_RPBs = mod(nPrb, prgSize);
        numPRBs_per_PRG = [prgSize*ones(floor(nPrb/prgSize),1);numResidual_RPBs]; 
    else
        numPRGs_this_ueg = 1;
        numPRBs_per_PRG = nPrb;
    end
    for posDmrs = 1:AdditionalPosition+1
        for idx_PRG = 1:numPRGs_this_ueg
            nPrb_this_PRG = numPRBs_per_PRG(idx_PRG);
            startPrb_this_PRG = startPrb+sum(numPRBs_per_PRG(1:idx_PRG-1));
            idx_REs_this_PRG = (startPrb-1)*12+1+sum(numPRBs_per_PRG(1:idx_PRG-1))*12:(startPrb-1)*12+sum(numPRBs_per_PRG(1:idx_PRG))*12;
            [tmp_H_LS_est, tmp_tmp_R0, tmp_tmp_R1] = apply_ChEst_LS_main(Xtf,puschTable,slotNumber,Nf,Nt,N_dmrs_id,...
            nl, portIdx, vec_scid, dmrsIdx{posDmrs}, Nt_dmrs, nPrb_this_PRG, startPrb_this_PRG, ...
            delta_f,maxLength, numDmrsCdmGrpsNoData, enableTfPrcd,...
             N_slot_frame, N_symb_slot, puschIdentity,...
             groupOrSequenceHopping, r_dmrs); 
            H_LS_est{posDmrs}(idx_REs_this_PRG,:,:) = tmp_H_LS_est(idx_REs_this_PRG,:,:);
            tmp_R0(posDmrs) = tmp_R0(posDmrs) + tmp_tmp_R0;
            tmp_R1(posDmrs) = tmp_R1(posDmrs) + tmp_tmp_R1;
        end
    end
    tmp_R0 = tmp_R0/numPRGs_this_ueg;
    tmp_R1 = tmp_R1/numPRGs_this_ueg;

    % est mean delay and delay spread
    R0 = zeros(AdditionalPosition+1, 1);
    R1 = zeros(AdditionalPosition+1, 1);
    R0 = mean(tmp_R0)*ones(AdditionalPosition+1,1);
    R1 = mean(tmp_R1)*ones(AdditionalPosition+1,1);
    
    delay_mean = zeros(AdditionalPosition+1, 1);
    delay_spread = zeros(AdditionalPosition+1, 1);
    if nl==1
        P_dmrs = 2;
    else
        P_dmrs = 4;
    end
    for posDmrs = 1:AdditionalPosition+1
        delay_mean(posDmrs) = -angle(R1(posDmrs))/2/pi/P_dmrs;
        delay_spread(posDmrs) = sqrt(2*(1-abs(R1(posDmrs))/real(R0(posDmrs))))/2/pi/P_dmrs;
    end
    delay_mean_microsec = delay_mean/delta_f*1e6;
    delay_spread_microsec = delay_spread/delta_f*1e6;
    if SimCtrl.alg.ChEst_enable_quantize_delay_spread_est
        quantization_levels_delay_spread_est_microsec = SimCtrl.alg.ChEst_quantize_levels_delay_spread_est*1e6;
        for posDmrs = 1:AdditionalPosition+1
            for idx_element = 1:length(quantization_levels_delay_spread_est_microsec)
                if quantization_levels_delay_spread_est_microsec(idx_element)>=delay_spread_microsec(posDmrs)
                    delay_spread_microsec(posDmrs) = quantization_levels_delay_spread_est_microsec(idx_element);
                    break;
                end
            end
            if delay_spread_microsec(posDmrs)>max(quantization_levels_delay_spread_est_microsec)
                delay_spread_microsec(posDmrs) = max(quantization_levels_delay_spread_est_microsec);
            end
        end
    end
    % do MMSE filtering
    for posDmrs = 1:AdditionalPosition+1
        % get MMSE filter coeff
        delay_mean_microsec_this_dmrs = delay_mean_microsec(posDmrs);
        if SimCtrl.alg.ChEst_enable_update_W 
            delay_spread_microsec_this_dmrs = delay_spread_microsec(posDmrs);
            mmse_rect_pdp_len_microsec = sqrt(12)*delay_spread_microsec_this_dmrs;
        else
            mmse_rect_pdp_len_microsec = SimCtrl.delaySpread*1e6;
        end
        tmp_table = derive_chest_mmse_coeff(carrier.mu, delay_mean_microsec_this_dmrs*1e-6, mmse_rect_pdp_len_microsec*1e-6);
        puschTable = update_puschTable_ChEst(puschTable,tmp_table);
        % MMSE filtering
        for idx_PRG = 1:numPRGs_this_ueg
            nPrb_this_PRG = numPRBs_per_PRG(idx_PRG); 
            startPrb_this_PRG = startPrb+sum(numPRBs_per_PRG(1:idx_PRG-1));
            idx_REs_this_PRG = (startPrb-1)*12+1+sum(numPRBs_per_PRG(1:idx_PRG-1))*12:(startPrb-1)*12+sum(numPRBs_per_PRG(1:idx_PRG))*12;
            tmp_H_est = apply_ChEst_MMSE_filter_main(H_LS_est{posDmrs},puschTable,slotNumber,Nf,Nt,N_dmrs_id,...
                nl, portIdx, vec_scid, dmrsIdx{posDmrs}, Nt_dmrs, nPrb_this_PRG, startPrb_this_PRG, ...
                delta_f,maxLength, numDmrsCdmGrpsNoData, enableTfPrcd,...
                 N_slot_frame, N_symb_slot, puschIdentity,...
                 groupOrSequenceHopping, r_dmrs);
            H_est{posDmrs}(:,:,idx_REs_this_PRG) = tmp_H_est(:,:,idx_REs_this_PRG);
        end
    end
    if SimCtrl.genTV.enable_logging_dbg_pusch_chest == 1
        dbg_chest.H_LS_est = H_LS_est;
        dbg_chest.R1 = R1;
        dbg_chest.delay_mean = delay_mean;
        dbg_chest.shiftSeq = tmp_table.shiftSeq;
        dbg_chest.shiftSeq4 = tmp_table.shiftSeq4;
        dbg_chest.unShiftSeq = tmp_table.unShiftSeq;
        dbg_chest.unShiftSeq4 = tmp_table.unShiftSeq4;
    else
        dbg_chest = [];
    end

end