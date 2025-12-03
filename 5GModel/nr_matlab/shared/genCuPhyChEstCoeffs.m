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

% generate H5 file:
function out = genCuPhyChEstCoeffs()
   dirName = 'GPU_test_input';
   h5Name  = 'cuPhyChEstCoeffs';

   [status,msg] = mkdir(dirName);
   h5File  = H5F.create([dirName filesep h5Name '.h5'], 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');


   % intialize paramaters:
   testAlloc      = [];
   SysPar         = initSysPar(testAlloc);
   SysPar         = updateAlloc(SysPar);

   global SimCtrl
   SimCtrl = SysPar.SimCtrl;

   SysPar.carrier = updateCarrier(SysPar.carrier);
   SysPar.chan_BF = [];
   SysPar.Chan_DL = [];
   SysPar.Chan_UL = [];

   % generate coefficents:
   gNB              = initgNB(SysPar);
   puschTable       = gNB.Phy.Config.table;
   srsTable         = gNB.Phy.Config.table;
   push_rkhs_tables = gNB.Phy.Config.table.push_rkhs_tables;

   % save PUSCH ChEst filters:
   W_middle   = puschTable.W_middle;
   W_upper    = puschTable.W_upper;
   W_lower    = puschTable.W_lower;
   WFreq      = reshape([W_middle W_lower W_upper], [size(W_middle,1) size(W_middle, 2) 3]);
   hdf5_write_nv_exp(h5File, 'WFreq', single(WFreq));

   W4_middle   = puschTable.W4_middle;
   W4_upper    = puschTable.W4_upper;
   W4_lower    = puschTable.W4_lower;
   W4Freq      = reshape([W4_middle W4_lower W4_upper], [size(W4_middle,1) size(W4_middle, 2) 3]);
   hdf5_write_nv_exp(h5File, 'WFreq4', single(W4Freq));

   W3_padded                = puschTable.W3;
   W2_padded                = zeros(37,18);
   W2_padded(1 : 25, 1:12)  = puschTable.W2;
   W1_padded                = zeros(37,18);
   W1_padded(1 : 13, 1 : 6) = puschTable.W1;
   WSmallFreq               = reshape([W1_padded W2_padded W3_padded], [size(W1_padded,1) size(W1_padded, 2) 3]);
   hdf5_write_nv_exp(h5File, 'WFreqSmall', single(WSmallFreq));

   %save PUSCH ChEst sequences:
   s_grid   = puschTable.shiftSeq;
   s = puschTable.unShiftSeq;
   hdf5_write_nv(h5File, 'ShiftSeq', single(s_grid(1:8*6)),'fp16');
   hdf5_write_nv(h5File, 'UnShiftSeq', single(s(1:97)),'fp16');

   shiftSeq4   = puschTable.shiftSeq4;
   unShiftSeq4 = puschTable.unShiftSeq4;
   hdf5_write_nv(h5File, 'ShiftSeq4', single(shiftSeq4),'fp16');
   hdf5_write_nv(h5File, 'UnShiftSeq4', single(unShiftSeq4),'fp16');

   % save SRS ChEst filters:
   save_C_FP16_table_to_H5(srsTable.srsFocc,       'focc_table'      , h5File, SimCtrl.fp16AlgoSel);
   save_C_FP16_table_to_H5(srsTable.srsFocc_comb2, 'focc_table_comb2', h5File, SimCtrl.fp16AlgoSel);
   save_C_FP16_table_to_H5(srsTable.srsFocc_comb4, 'focc_table_comb4', h5File, SimCtrl.fp16AlgoSel);


   save_C_FP16_table_to_H5(srsTable.W_comb2_nPorts1_wide, 'W_comb2_nPorts1_wide', h5File, SimCtrl.fp16AlgoSel);
   save_C_FP16_table_to_H5(srsTable.W_comb2_nPorts2_wide, 'W_comb2_nPorts2_wide', h5File, SimCtrl.fp16AlgoSel);
   save_C_FP16_table_to_H5(srsTable.W_comb2_nPorts4_wide, 'W_comb2_nPorts4_wide', h5File, SimCtrl.fp16AlgoSel);
   save_C_FP16_table_to_H5(srsTable.W_comb2_nPorts8_wide, 'W_comb2_nPorts8_wide', h5File, SimCtrl.fp16AlgoSel);

   save_C_FP16_table_to_H5(srsTable.W_comb4_nPorts1_wide, 'W_comb4_nPorts1_wide', h5File, SimCtrl.fp16AlgoSel);
   save_C_FP16_table_to_H5(srsTable.W_comb4_nPorts2_wide, 'W_comb4_nPorts2_wide', h5File, SimCtrl.fp16AlgoSel);
   save_C_FP16_table_to_H5(srsTable.W_comb4_nPorts4_wide, 'W_comb4_nPorts4_wide', h5File, SimCtrl.fp16AlgoSel);
   save_C_FP16_table_to_H5(srsTable.W_comb4_nPorts6_wide, 'W_comb4_nPorts6_wide', h5File, SimCtrl.fp16AlgoSel);
   save_C_FP16_table_to_H5(srsTable.W_comb4_nPorts12_wide, 'W_comb4_nPorts12_wide', h5File, SimCtrl.fp16AlgoSel);

   save_C_FP16_table_to_H5(srsTable.W_comb2_nPorts1_narrow, 'W_comb2_nPorts1_narrow', h5File, SimCtrl.fp16AlgoSel);
   save_C_FP16_table_to_H5(srsTable.W_comb2_nPorts2_narrow, 'W_comb2_nPorts2_narrow', h5File, SimCtrl.fp16AlgoSel);
   save_C_FP16_table_to_H5(srsTable.W_comb2_nPorts4_narrow, 'W_comb2_nPorts4_narrow', h5File, SimCtrl.fp16AlgoSel);
   save_C_FP16_table_to_H5(srsTable.W_comb2_nPorts8_narrow, 'W_comb2_nPorts8_narrow', h5File, SimCtrl.fp16AlgoSel);

   save_C_FP16_table_to_H5(srsTable.W_comb4_nPorts1_narrow, 'W_comb4_nPorts1_narrow', h5File, SimCtrl.fp16AlgoSel);
   save_C_FP16_table_to_H5(srsTable.W_comb4_nPorts2_narrow, 'W_comb4_nPorts2_narrow', h5File, SimCtrl.fp16AlgoSel);
   save_C_FP16_table_to_H5(srsTable.W_comb4_nPorts4_narrow, 'W_comb4_nPorts4_narrow', h5File, SimCtrl.fp16AlgoSel);
   save_C_FP16_table_to_H5(srsTable.W_comb4_nPorts6_narrow, 'W_comb4_nPorts6_narrow', h5File, SimCtrl.fp16AlgoSel);
   save_C_FP16_table_to_H5(srsTable.W_comb4_nPorts12_narrow, 'W_comb4_nPorts12_narrow', h5File, SimCtrl.fp16AlgoSel);

   % save SRS deBias paramaters:
   debiasPrms = [];
   debiasPrms.noisEstDebias_comb2_nPorts1 = single(srsTable.noisEstDebias_comb2_nPorts1);
   debiasPrms.noisEstDebias_comb2_nPorts2 = single(srsTable.noisEstDebias_comb2_nPorts2);
   debiasPrms.noisEstDebias_comb2_nPorts4 = single(srsTable.noisEstDebias_comb2_nPorts4);
   debiasPrms.noisEstDebias_comb2_nPorts8 = single(srsTable.noisEstDebias_comb2_nPorts8);

   debiasPrms.noisEstDebias_comb4_nPorts1  = single(srsTable.noisEstDebias_comb4_nPorts1);
   debiasPrms.noisEstDebias_comb4_nPorts2  = single(srsTable.noisEstDebias_comb4_nPorts2);
   debiasPrms.noisEstDebias_comb4_nPorts4  = single(srsTable.noisEstDebias_comb4_nPorts4);
   debiasPrms.noisEstDebias_comb4_nPorts6  = single(srsTable.noisEstDebias_comb4_nPorts6);
   debiasPrms.noisEstDebias_comb4_nPorts12 = single(srsTable.noisEstDebias_comb4_nPorts12);
   hdf5_write_nv_exp(h5File, 'srsNoiseEstDebiasPrms', debiasPrms);

   % save PUSCH RKHS paramaters:
    push_rkhs_tables = puschTable.push_rkhs_tables;
    prbRksPrms = [];

    % per Prb RKHS prms:
    for i = 1 : push_rkhs_tables.rkhsPrms.nPrbSizes
        prbRksPrms(i).nZpDmrsSc           = uint16(push_rkhs_tables.prbPrms{i}.nZpDmrsSc);
        prbRksPrms(i).zpIdx               = uint8(push_rkhs_tables.prbPrms{i}.zpIdx);
        prbRksPrms(i).nCpInt              = uint8(push_rkhs_tables.prbPrms{i}.nCpInt);
        prbRksPrms(i).sumEigValues        = single(push_rkhs_tables.prbPrms{i}.sumEigValues);
    
        str1 = 'corr_half_nZpDmrsSc';
        str2 = num2str(i);
        str  = append(str1, str2);
        hdf5_write_nv(h5File, str, complex(single(push_rkhs_tables.prbPrms{i}.corr_half_nZpDmrsSc)),'fp16');
        
        str1 = 'eigVecCob';
        str2   = num2str(i);
        str    = append(str1, str2);
        buffer = push_rkhs_tables.eigVecCobTable{i};
        buffer = fp16nv(real(buffer), SimCtrl.fp16AlgoSel);
        hdf5_write_nv(h5File, str, buffer,'fp16');    
    
        str1 = 'corr';
        str2 = num2str(i);
        str  = append(str1, str2);
        hdf5_write_nv(h5File, str, complex(single(push_rkhs_tables.corrTable{i})),'fp16');
    
        str1 = 'eigVal';
        str2 = num2str(i);
        str  = append(str1, str2);
        hdf5_write_nv(h5File, str, real(single(push_rkhs_tables.eigValTable{i})),'fp16');
    
        str1 = 'interpCob';
        str2 = num2str(i);
        str  = append(str1, str2);
        hdf5_write_nv(h5File, str, real(single(push_rkhs_tables.interpCobTable{i}.')),'fp16');
    end

    % per zp prms:
    zpRksPrms = [];
    
    for zpIdx = 0 : (push_rkhs_tables.rkhsPrms.nZpSizes - 1)
        zpRksPrms(zpIdx + 1).secondStageFourierSize = uint8(push_rkhs_tables.zpFftPrms{zpIdx + 1}.secondStageFourierSize);
        zpRksPrms(zpIdx + 1).nZpDmrsSc              = uint16(push_rkhs_tables.zpFftPrms{zpIdx + 1}.nZpDmrsSc);
    
        str1 = 'zpInterpVec';
        str2 = num2str(zpIdx);
        str  = append(str1, str2);
        buffer = push_rkhs_tables.zpInterpVecTable{zpIdx + 1}.';
        buffer = fp16nv(real(buffer), SimCtrl.fp16AlgoSel);
        hdf5_write_nv(h5File, str, buffer,'fp16');
    
        str1 = 'zpDmrsScEigenVec';
        str2 = num2str(zpIdx);
        str  = append(str1, str2);
        buffer = push_rkhs_tables.zpDmrsScEigenVecTable{zpIdx + 1};
        buffer = fp16nv(real(buffer), SimCtrl.fp16AlgoSel);
        hdf5_write_nv(h5File, str, buffer,'fp16');
    
        str1 = 'zpSecondStageTwiddleFactors';
        str2 = num2str(zpIdx);
        str  = append(str1, str2);
        buffer = conj(push_rkhs_tables.zpFftPrms{zpIdx + 1}.secondStageTwiddleFactors);
        hdf5_write_nv(h5File, str, complex(single(buffer)),'fp16');
    
        str1 = 'zpSecondStageFourierPerm';
        str2 = num2str(zpIdx);
        str  = append(str1, str2);
        buffer = push_rkhs_tables.zpFftPrms{zpIdx + 1}.secondStageFourierPerm;
        buffer = uint8(buffer);
        hdf5_write_nv(h5File, str, buffer);
    end


    rkhsPrms               = [];
    rkhsPrms.nEigs         = uint8(push_rkhs_tables.rkhsPrms.nEigs);
    rkhsPrms.nZpDmrsScEigs = uint8(push_rkhs_tables.rkhsPrms.nZpDmrsScEigs);
    rkhsPrms.nIterpEigs    = uint8(push_rkhs_tables.rkhsPrms.nIterpEigs);
    rkhsPrms.nPrbSizes     = uint16(push_rkhs_tables.rkhsPrms.nPrbSizes);
    rkhsPrms.nZpSizes      = uint8(push_rkhs_tables.rkhsPrms.nZpSizes);
    
    hdf5_write_nv(h5File, 'zpRksPrms', zpRksPrms);
    hdf5_write_nv(h5File, 'rkhsPrms', rkhsPrms);
    hdf5_write_nv(h5File, 'prbRksPrms', prbRksPrms);

    % save SRS RKHS paramaters:
    srs_rkhs_tables = puschTable.srs_rkhs_tables;

    srsRkhsPrms        = [];
    rkhsGridPrms       = [];
    srsRkhsPrms.nGrids = uint8(srs_rkhs_tables.rkhsPrms.nGridSizes);

    for gridIdx = 0 : (srsRkhsPrms.nGrids - 1)
        rkhsGridPrms(gridIdx + 1).nEigs      = uint8(srs_rkhs_tables.gridPrms{gridIdx + 1}.nEig);
        rkhsGridPrms(gridIdx + 1).gridSize   = uint16(srs_rkhs_tables.gridPrms{gridIdx + 1}.gridSize);
        rkhsGridPrms(gridIdx + 1).zpGridSize = uint16(srs_rkhs_tables.gridPrms{gridIdx + 1}.zpGridSize);

        str1 = 'srsRkhs_eigenVecs_grid';
        str2 = num2str(gridIdx);
        str  = append(str1, str2);
        buffer = srs_rkhs_tables.eigenVecTable{gridIdx + 1};
        buffer = fp16nv(real(buffer), SimCtrl.fp16AlgoSel);
        hdf5_write_nv(h5File, str, buffer,'fp16');

        str1 = 'srsRkhs_eigValues_grid';
        str2 = num2str(gridIdx);
        str  = append(str1, str2);
        buffer = srs_rkhs_tables.eigValTable{gridIdx + 1};
        buffer = fp16nv(real(buffer), SimCtrl.fp16AlgoSel);
        hdf5_write_nv(h5File, str, buffer,'fp16');

        str1 = 'srsRkhs_eigenCorr_grid';
        str2 = num2str(gridIdx);
        str  = append(str1, str2);
        buffer = srs_rkhs_tables.corrTable{gridIdx + 1};
        buffer = fp16nv(complex(buffer), SimCtrl.fp16AlgoSel);
        hdf5_write_nv(h5File, str, complex(single(buffer)),'fp16');

        str1 = 'srsRkhs_secondStageTwiddleFactors_grid';
        str2 = num2str(gridIdx);
        str  = append(str1, str2);
        buffer = srs_rkhs_tables.secondStageTwiddleFactorsTable{gridIdx + 1};
        buffer = fp16nv(complex(buffer), SimCtrl.fp16AlgoSel);
        hdf5_write_nv(h5File, str, complex(single(buffer)),'fp16');

        str1 = 'srsRkhs_secondStageFourierPerm_grid';
        str2 = num2str(gridIdx);
        str  = append(str1, str2);
        buffer = srs_rkhs_tables.secondStageFourierPermTable{gridIdx + 1};
        buffer = uint8(buffer);
        hdf5_write_nv(h5File, str, buffer);
    end

    hdf5_write_nv(h5File, 'srsRkhsPrms', srsRkhsPrms);
    hdf5_write_nv(h5File, 'rkhsGridPrms', rkhsGridPrms);


   H5F.close(h5File);
   out = 0; %hack for matlab compiler
   st = dbstack;
   fprintf("Finished %s\n",st.name);
end


% function to cast complex fp64 table to complex fp16
function save_C_FP16_table_to_H5(table, table_name, h5File, fp16AlgoSel)
    table = reshape(fp16nv(real(table), fp16AlgoSel) + 1i*fp16nv(imag(table), fp16AlgoSel), [size(table)]);
    hdf5_write_nv(h5File, table_name, complex(single(table)));
end


