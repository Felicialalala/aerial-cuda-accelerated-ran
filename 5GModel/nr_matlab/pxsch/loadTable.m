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

function table = loadTable

load('type1_dmrs_table.mat');
table.fOCC_table = fOCC_table;
table.grid_table = grid_table;
table.tOCC_table = tOCC_table;
load('TBS_table.mat');
table.TBS_table = TBS_table;
load('Tanner_BG1.mat');
table.BG1_NeighborIndicies = NeighborIndicies;
table.BG1_NeighborPermutations_LS1 = NeighborPermutations_LS1;
table.BG1_NeighborPermutations_LS2 = NeighborPermutations_LS2;
table.BG1_NeighborPermutations_LS3 = NeighborPermutations_LS3;
table.BG1_NeighborPermutations_LS4 = NeighborPermutations_LS4;
table.BG1_NeighborPermutations_LS5 = NeighborPermutations_LS5;
table.BG1_NeighborPermutations_LS6 = NeighborPermutations_LS6;
table.BG1_NeighborPermutations_LS7 = NeighborPermutations_LS7;
table.BG1_NeighborPermutations_LS8 = NeighborPermutations_LS8;
table.BG1_numNeighbors = numNeighbors;
load('Tanner_BG2.mat');
table.BG2_NeighborIndicies = NeighborIndicies;
table.BG2_NeighborPermutations_LS1 = NeighborPermutations_LS1;
table.BG2_NeighborPermutations_LS2 = NeighborPermutations_LS2;
table.BG2_NeighborPermutations_LS3 = NeighborPermutations_LS3;
table.BG2_NeighborPermutations_LS4 = NeighborPermutations_LS4;
table.BG2_NeighborPermutations_LS5 = NeighborPermutations_LS5;
table.BG2_NeighborPermutations_LS6 = NeighborPermutations_LS6;
table.BG2_NeighborPermutations_LS7 = NeighborPermutations_LS7;
table.BG2_NeighborPermutations_LS8 = NeighborPermutations_LS8;
table.BG2_numNeighbors = numNeighbors;
load('McsTable.mat');
table.McsTable1 = McsTable1;
table.McsTable2 = McsTable2;
table.McsTable3 = McsTable3;
table.McsTable4 = McsTable4;
table.McsTable5 = McsTable5;
load('crcTable.mat');
table.crcTable_24A = crcTable_24A;
table.crcTable_24B = crcTable_24B;
table.crcTable_24C = crcTable_24C;
table.crcTable_16 = crcTable_16;
table.crcTable_11 = crcTable_11;
load('qam_mapping.mat');
table.BPSK_mapping = BPSK_mapping;
table.QPSK_mapping = QPSK_mapping;
table.QAM16_mapping = QAM16_mapping;
table.QAM64_mapping = QAM64_mapping;
table.QAM256_mapping = QAM256_mapping;
table.kBar_table = load('table_kBar.txt');
load table_prachCfg_FR1TDD;
load table_prachCfg_FR1FDD;
load table_prachCfg_FR2;
table.prachCfg_FR1TDD = table_prachCfg_FR1TDD;
table.prachCfg_FR1FDD = table_prachCfg_FR1FDD;
table.prachCfg_FR2 = table_prachCfg_FR2;
table.table_NCS_1p25k = load('table_NCS_1p25k.txt');
table.table_NCS_5k = load('table_NCS_5k.txt');
table.table_NCS_15kplus = load('table_NCS_15kplus.txt');
table.table_logIdx2u_839 = load('table_logIdx2u_839.txt');
table.table_logIdx2u_139 = load('table_logIdx2u_139.txt');
load('r_pucch.mat');
table.r = r;
load('tOCC_pucch.mat');
% Reshape tOCC from a 2D cell array to a Nx1 cell array of Nx1 cell arrays.
% This is necessary for Matlab python engine compatability.
tOCC_reshape = {};
for k=1:7
    tOCC_reshape{k} = {};
    for n=1:7
        tOCC_reshape{k}{n} = tOCC{k,n};
    end
end
table.tOCC = tOCC_reshape;
load('tOCC_pucch2.mat');
table.tOCC_cell = tOCC_cell;
load('srs_bandwidth_table.mat');
table.srs_BW_table = T;
load('srsLowPaprTable0.mat');
table.srsLowPaprTable0 = srsLowPaprTable0;
load('srsLowPaprTable1.mat');
table.srsLowPaprTable1 = srsLowPaprTable1;
load('srsPrimes.mat');
table.srsPrimes = srsPrimes;
load('srsFocc.mat');
table.srsFocc = conj(srsFocc);
table.srsFocc_comb2 = conj(fft(eye(8),8,1));
table.srsFocc_comb4 = conj(fft(eye(12),12,1));
load('primes.mat');
table.prime_table = p;
load('qam_dist.mat');
table.d_qpsk = d_qpsk;
table.d_qam16 = d_qam16;
table.d_qam64 = d_qam64;
table.d_qam256 = d_qam256;
load('noCdmTable.mat');
table.noCdmTable = noCdmTable;
load('fdCdm2Table.mat');
table.fdCdm2Table = fdCdm2Table;
load('cdm4Table.mat');
table.cdm4Table = cdm4Table;
load('cdm8Table.mat');
table.cdm8Table = cdm8Table;
load('csirsLocTable.mat');
table.csirsLocTable = csirsLocTable;
load('PM_W.mat');
table.PM_W = PM_W;
table.QAM4_LLR = load('table_QAM4_LLR.txt');
table.QAM16_LLR = load('table_QAM16_LLR.txt');
table.QAM64_LLR = load('table_QAM64_LLR.txt');
table.QAM256_LLR = load('table_QAM256_LLR.txt');
load('table_sinr2cqi.mat')
table.sinr2cqi = cqi_table_44PRBs_MMSE_IRC_noShr_fusedRx;