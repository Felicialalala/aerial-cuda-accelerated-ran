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

function [LLRseq, LLR_demap] = softDemapper(X_est, Ree, table, nl, Nt_data, Nf, nPrb, startPrb, ...
    nl_offset, qam, n_scid, TdiMode, symIdx_data, symIdx_dmrs, dmrsIdx, numDmrsCdmGrpsNoData, isDataPresent)

global SimCtrl

L_UE = nl;
if numDmrsCdmGrpsNoData == 1
    symAll = [min(symIdx_data(1), symIdx_dmrs(1)) : max(symIdx_data(end), symIdx_dmrs(end))];
    nSym_data = length(symAll);
else
    symAll = symIdx_data;
    nSym_data = Nt_data;
end

d_qpsk = table.d_qpsk;
d_qam16 = table.d_qam16;
d_qam64 = table.d_qam64;
d_qam256 = table.d_qam256;

QAM4_LLR = table.QAM4_LLR;
QAM16_LLR = table.QAM16_LLR;
QAM64_LLR = table.QAM64_LLR;
QAM256_LLR = table.QAM256_LLR;

LLR = zeros(8,L_UE,Nf,nSym_data);

Nf_data = 12*nPrb;   % number of subcarriers in FEG
layerQam(1:nl) = qam;      % bits per qam transmited by each layer. Dim: nl x 1
scid_vect(1:nl) = n_scid;  % scrambling id used by each layer. Dim: nl x 1

%FEG frequency index: 
freqIdx = (startPrb - 1)*12 + 1 : (startPrb + nPrb - 1)*12;

%loop over layers:
for s = 1 : nl
    qam = layerQam(s);

    %extract d:
    switch qam
        case {1, 2}
            d = d_qpsk;
        case 4
            d = d_qam16;
        case 6
            d = d_qam64;
        case 8
            d = d_qam256;
    end
    
    if SimCtrl.useCuphySoftDemapper
        switch qam
            case {1, 2}
                T = QAM4_LLR;
            case 4
                T = QAM16_LLR;
            case 6
                T = QAM64_LLR;
            case 8
                T = QAM256_LLR;
        end
        % Truncate to an fp16 representation
        colfp16 = fp16nv(T,  SimCtrl.fp16AlgoSel);
        % Convert back to fp16 for interpolation below
        T = double(colfp16);
    end
    
    pam = qam / 2; %number of bits in PAM constellations
    
    x_est_vec = zeros(Nf_data*nSym_data, 1);
    N0_vec = zeros(Nf_data*nSym_data, 1);
    %compute LLRs:
    for f = 1 : Nf_data
        for t = 1 : nSym_data
            
            %extract estimated symbol and error variance:
%             x_est = X_est(freqIdx(f),t, portIdx(s) + 8*scid_vect(s));
            x_est = X_est(freqIdx(f),t, s + nl_offset);
%             x_est = X_est(freqIdx(f),t, portIdx(s));
%             N0 = Ree(portIdx(s),freqIdx(f));

            % Undo pi/2-BPSK rotation
            if qam == 1 
                if mod(f-1, 2) == 1
                    x_est = x_est * exp(-1j*3*pi/4);
                else
                    x_est = x_est * exp(-1j*pi/4);
                end
            end
            
            if TdiMode == 0
                N0 = Ree(s + nl_offset, freqIdx(f),1);
            elseif TdiMode == 1
                sym_idx = symAll(t);
                for idx_dmrsPos = 1:length(dmrsIdx)
                    if sym_idx <= dmrsIdx{idx_dmrsPos}(end)
                        break;
                    end
                end
                N0 = Ree(s + nl_offset, freqIdx(f), idx_dmrsPos);
            elseif TdiMode == 2
                N0 = Ree(s + nl_offset, freqIdx(f),t);
            
            end
            
            if SimCtrl.useCuphySoftDemapper == 0
                if qam == 1
                    llr = real(x_est);
                else
                    %estimate real/imag pam llrs:
                    llr_real = max_pam_LLR(real(x_est),N0,d,pam);
                    llr_imag = max_pam_LLR(imag(x_est),N0,d,pam);
                    %combine into qam llrs:
                    llr_matlab = zeros(qam,1);
                    for j = 1 : pam
                        llr_matlab(2*(j-1) + 1) = llr_real(j);
                        llr_matlab(2*(j-1) + 2) = llr_imag(j);
                    end
                    llr = llr_matlab;
                end
            elseif SimCtrl.useCuphySoftDemapper == 1
                llr_cuphy = cuPhySoftDemapper(x_est, T, qam, N0);
                llr = llr_cuphy;
            elseif SimCtrl.useCuphySoftDemapper == 2 % Model for cuPHY simplified soft demapper
                idx_sym = (f-1)*nSym_data + t;
                x_est_vec(idx_sym,1) = x_est;
                N0_vec(idx_sym,1) = N0;
                if (f == Nf_data) && (t == nSym_data) % trigger the simplified soft demapper                
                    fp_flag = SimCtrl.fp_flag_pusch_demapper;
                    fp_flag_out_llr = SimCtrl.fp_flag_pusch_demapper_out_llr;
                    llr_cuphy = cuPhySimplifiedSoftDemapper_Varray(x_est_vec, qam, N0_vec, fp_flag, fp_flag_out_llr);
                    llr = llr_cuphy.value;
                end
            elseif SimCtrl.useCuphySoftDemapper == 3
                % use ML to infer the LLR
                llr = llrNetSoftDemapper(x_est, T, qam, N0);
            end
            
            %embed into global llr:
%             LLR(1 : qam, portIdx(s) + 8*scid_vect(s),freqIdx(f),t) = llr;
            if SimCtrl.useCuphySoftDemapper == 2
                if (f == Nf_data) && (t == nSym_data)
                    % force NaN to 1
                    llr(isnan(llr)) = 1;
                    LLR(1 : qam, s, freqIdx,:) = permute(reshape(llr, [qam, nSym_data, Nf_data]),[1,3,2]);
                end
            else
                % force NaN to 1
                llr(isnan(llr)) = 1;
                LLR(1 : qam, s, freqIdx(f),t) = llr;
            end
        end
    end
end

LLR = real(LLR);

freqIdx = 12*(startPrb - 1) + 1 : 12*(startPrb + nPrb - 1);
% LLR_demap = LLR(1 : qam, portIdx + 8*n_scid, freqIdx, :);
LLR_demap = LLR(1 : qam, :, freqIdx, :);

%next demap:
LLRseq = LLR_demap(:);

return
