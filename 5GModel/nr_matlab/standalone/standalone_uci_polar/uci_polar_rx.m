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

function [cwBitTypes_cell, cwTreeTypes_cell, cwLLRs_cell, uciSegEsts_cell] = uci_polar_rx(nPolUciSegs, A_seg, E_seg, listLength, uciSegLLRs_cell, genTvFlag, tcNumber)

% Function runs uci polar reciever pipeline using cuPHY matlab model.
% Outputs intermediate buffers, option to save H5 test-vector.

%inputs:
% nPolUciSegs     --> number of polar uci segments to process
% A_seg           --> number of info bits per segment. Dim: nPolUciSegs x 1
% E_seg           --> number of tx bits per segment.   Dim: nPolUciSegs x 1
% uciSegLLRs_cell --> cell containing LLRs of uci segments. Dim: nPolUciSegs x 1
% genTvFlag       --> indicates if Tv should be generated
% tcNumber        --> test-case number, used to name TV H5 file

%outputs
% cwBitTypes_cell  --> cell containing cwBitTypes.            Dim: nPolUciSegs x 1
% cwTreeTypes_cell --> cell containing cwTreeTypes.           Dim: nPolUciSegs x 1
% cwLLRs_cell      --> cell containing codeword LLRs.         Dim: totNumPolCbs x 1
% uciSegEsts_cell  --> cell containing uci segment estimates. Dim: nPolUciSegs x 1

%%
% DERIVE UCI PARAMATERS

polarUciSegPrms_cell = cell(nPolUciSegs,1);

for segIdx = 0 : (nPolUciSegs - 1)
    polarUciSegPrms_cell{segIdx + 1} = derive_polarUciSegPrms(A_seg(segIdx + 1), E_seg(segIdx + 1));
end

[totNumPolCbs, polarCbPrms_cell, polarUciSegPrms_cell] = derive_polarCbPrms(nPolUciSegs, polarUciSegPrms_cell);


%%
% RUN 

% segment LLRs to codewords, deInterleave, deRateMatch
cwLLRs_cell = cell(totNumPolCbs,1);

for segIdx = 0 : (nPolUciSegs - 1)
    polarUciSegPrms  = polarUciSegPrms_cell{segIdx + 1};
    deRmDeItlDynDesc = compute_polDeRmDeItlDynDesc(polarUciSegPrms_cell{segIdx + 1});
    
    cwLLRs = pol_cwSeg_deRm_deItl(polarUciSegPrms, deRmDeItlDynDesc, uciSegLLRs_cell{segIdx + 1});
   
    for i = 0 : (polarUciSegPrms.nCbs - 1)
        cbIdx = polarUciSegPrms.childCbIdxs(i + 1);
        cwLLRs_cell{cbIdx + 1} = cwLLRs(:,i + 1);
    end
end

% Compute codeword types:
cwTreeTypes_cell = cell(nPolUciSegs,1);
cwBitTypes_cell  = cell(nPolUciSegs,1);

for segIdx = 0 : (nPolUciSegs - 1)
    [cwBitTypes_cell{segIdx + 1}, cwTreeTypes_cell{segIdx + 1}] = compBitTypesKernel(polarUciSegPrms_cell{segIdx + 1});
end


% run polar decoder:
cbEsts_cell   = cell(totNumPolCbs,1);
crcErrorFlags = zeros(totNumPolCbs,1);

for segIdx = 0 : (nPolUciSegs - 1)
    polarUciSegPrms = polarUciSegPrms_cell{segIdx + 1};
    cwTreeTypes     = cwTreeTypes_cell{segIdx + 1};
    
    for i = 0 : (polarUciSegPrms.nCbs - 1)
        cbIdx = polarUciSegPrms.childCbIdxs(i + 1);
        
        cwLLRs   = cwLLRs_cell{cbIdx + 1};
        [cbEst,crcErrorFlag]   = polar_decoder(listLength, polarUciSegPrms.K_cw, polarUciSegPrms.N_cw, polarUciSegPrms.nCrcBits, cwLLRs, cwTreeTypes);
               
        cbEsts_cell{cbIdx + 1}   = cbEst;
        crcErrorFlags(cbIdx + 1) = crcErrorFlag;
    end
end

% Combine cb Ests to form UCI seg estimate
uciSegEsts_cell = cell(nPolUciSegs, 1);

for segIdx = 0 : (nPolUciSegs - 1)
    polarUciSegPrms = polarUciSegPrms_cell{segIdx + 1};
    
    if(polarUciSegPrms.nCbs == 1)
        uciSegEsts_cell{segIdx + 1} = cbEsts_cell{polarUciSegPrms.childCbIdxs(1) + 1};
    else
        cbEst0 = cbEsts_cell{polarUciSegPrms.childCbIdxs(1) + 1};
        cbEst1 = cbEsts_cell{polarUciSegPrms.childCbIdxs(2) + 1};
        
        uciSegEsts_cell{segIdx + 1} = combine_uciCbEsts(cbEst0, cbEst1, polarUciSegPrms);
    end
end


        
        
        

%%
% SAVE TEST VECTOR

if(genTvFlag)
    % create h5 file
    tvDirName = 'GPU_test_input';
    tvName    = strcat('TVnr_',num2str(tcNumber),'_uciPolar');
    [status,msg] = mkdir(tvDirName); 

    h5File = H5F.create([tvDirName filesep tvName '.h5'], 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');


    % save polarUciSegPrms 
    polarUciSegPrms = [];
    for segIdx = 0 : (nPolUciSegs - 1)
        polarUciSegPrms(segIdx + 1).nCbs            = uint8(polarUciSegPrms_cell{segIdx + 1}.nCbs);
        polarUciSegPrms(segIdx + 1).K_cw            = uint16(polarUciSegPrms_cell{segIdx + 1}.K_cw);   
        polarUciSegPrms(segIdx + 1).E_cw            = uint32(polarUciSegPrms_cell{segIdx + 1}.E_cw);
        polarUciSegPrms(segIdx + 1).zeroInsertFlag  = uint8(polarUciSegPrms_cell{segIdx + 1}.zeroInsertFlag);
        polarUciSegPrms(segIdx + 1).n_cw            = uint8(polarUciSegPrms_cell{segIdx + 1}.n_cw);
        polarUciSegPrms(segIdx + 1).N_cw            = uint16(polarUciSegPrms_cell{segIdx + 1}.N_cw);
        polarUciSegPrms(segIdx + 1).E_seg           = uint32(polarUciSegPrms_cell{segIdx + 1}.E_seg);
        polarUciSegPrms(segIdx + 1).nCrcBits        = uint8(polarUciSegPrms_cell{segIdx + 1}.nCrcBits);
    end
    hdf5_write_nv_exp(h5File, 'polarUciSegPrms', polarUciSegPrms);

    % save cwTreeTypes
    for segIdx = 0 : (nPolUciSegs - 1)
        nameStr = strcat('cwTreeTypes',num2str(segIdx));
        hdf5_write_nv(h5File, nameStr, uint8(cwTreeTypes_cell{segIdx + 1}));
    end
    
    % save uci segment LLRs
    for segIdx = 0 : (nPolUciSegs - 1)
        uciSegLLRs = fp16nv(uciSegLLRs_cell{segIdx + 1}, 2);
        nameStr    = strcat('uciSegLLRs',num2str(segIdx));
        hdf5_write_nv(h5File, nameStr, uciSegLLRs);
    end
     
    % save codeword LLRs
    for cbIdx = 0 : (totNumPolCbs - 1)
        cwLLRs  = fp16nv(cwLLRs_cell{cbIdx + 1}, 2);
        nameStr = strcat('cwLLRs',num2str(cbIdx));
        hdf5_write_nv(h5File, nameStr, cwLLRs);
    end
    
    % save codeblock estimates
    for cbIdx = 0 : (totNumPolCbs - 1)
        cbEst_bits = cbEsts_cell{cbIdx + 1};
        
        nBits  = length(cbEst_bits);
        nWords = ceil(nBits / 32);
        
        cbEst_words = zeros(nWords,1);
        for wordIdx = 0 : (nWords - 1)
            for i = 0 : 31
                bitIdx = wordIdx*32 + i;
                
                if(bitIdx >= nBits)
                    break;
                else
                    cbEst_words(wordIdx+1) = cbEst_words(wordIdx+1) + cbEst_bits(bitIdx + 1)*2^i;
                end
            end
        end
        cbEst_words = uint32(cbEst_words);
        
        nameStr = strcat('cbEst',num2str(cbIdx));
        hdf5_write_nv(h5File, nameStr, cbEst_words);
    end
    
    % save uci seg estimates
    for segIdx = 0 : (nPolUciSegs - 1)
        uciSegEst_bits = uciSegEsts_cell{segIdx + 1};
        
        nBits  = length(uciSegEst_bits);
        nWords = ceil(nBits / 32);
        
        uciSegEst_words = zeros(nWords,1);
        for wordIdx = 0 : (nWords - 1)
            for i = 0 : 31
                bitIdx = wordIdx*32 + i;
                
                if(bitIdx >= nBits)
                    break;
                else
                    uciSegEst_words(wordIdx+1) = uciSegEst_words(wordIdx+1) + uciSegEst_bits(bitIdx + 1)*2^i;
                end
            end
        end
        uciSegEst_words = uint32(uciSegEst_words);
        
        nameStr = strcat('uciSegEst',num2str(segIdx));
        hdf5_write_nv(h5File, nameStr, uciSegEst_words);
    end
    
    % save crc error flags
    hdf5_write_nv(h5File, 'crcErrorFlags', uint8(crcErrorFlags));

    % sizes:
    sizes = [];
    sizes.nPolCws     = uint16(totNumPolCbs);
    sizes.nPolUciSegs = uint16(nPolUciSegs);
    hdf5_write_nv_exp(h5File, 'sizes', sizes);

    H5F.close(h5File);
end

