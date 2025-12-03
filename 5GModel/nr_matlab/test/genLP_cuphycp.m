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

function errFlag = genLP_cuphycp(caseSet)

caseList = [1:9];

for caseNum = caseList
    gen_launch_pattern_cuphycp(caseNum);
end

errFlag = 0;

return

function gen_launch_pattern_cuphycp(caseNum)

nSlot = 20;

switch caseNum
    case 1 % V_M3
        nCell = 3;
        PDSCH_TV1 = 'TV_fapi_V14-DS-08_slot0_MIMO2x16_PRB82_DataSyms10_qam64.h5';
        PUSCH_TV1 = 'TV_fapi_V12-US-08_snrdb40.00_MIMO4x16_PRB82_DataSyms10_qam16.h5';
        LP = [];
        LP.Cell_Configs = {PUSCH_TV1, PUSCH_TV1, PUSCH_TV1};
        LP = init_launchPattern(LP, nSlot, nCell);
        UL_slotIdx = [4, 9, 14, 19];
        DL_slotIdx = [0:19];
        [~,TcIdx] = ismember(UL_slotIdx, DL_slotIdx);
        DL_slotIdx(TcIdx) = [];
        for slotIdx = UL_slotIdx
            for cellIdx = 0:nCell-1
                LP.SCHED{slotIdx+1}.config{cellIdx+1}.channels = {PUSCH_TV1};
            end
        end
        for slotIdx = DL_slotIdx
            
            for cellIdx = 0:nCell-1
                LP.SCHED{slotIdx+1}.config{cellIdx+1}.channels = {PDSCH_TV1};
            end
        end
        LPfilename = 'V_M3';
    case 2 % F08_1C
        nCell = 1;
        PUSCH_TV1 = 'TVnr_CP_F08_US_01.1_gNB_FAPI_s5.h5';
        PDSCH_TV1 = 'TVnr_CP_F08_DS_01.1_gNB_FAPI_s5.h5';
        PDSCH_TV2 = 'TVnr_CP_F08_DS_01.2_gNB_FAPI_s5.h5';
        PDSCH_TV3 = 'TVnr_CP_F08_DS_01.1_126_gNB_FAPI_s5.h5';
        PBCH_TV1 = 'TVnr_1902_gNB_FAPI_s0.h5';
        PDCCH_UL_TV1 = 'demo_msg4_4_ant_pdcch_ul_gNB_FAPI_s7.h5';
        PDCCH_DL_TV1 = 'demo_msg2_4_ant_pdcch_dl_gNB_FAPI_s7.h5';
        LP = [];
        LP.Cell_Configs = {PUSCH_TV1};
        LP = init_launchPattern(LP, nSlot, nCell);
        LP.SCHED{1}.config{1}.channels = {PBCH_TV1, PDSCH_TV3};
        LP.SCHED{2}.config{1}.channels = {PDSCH_TV1};
        LP.SCHED{3}.config{1}.channels = {PDSCH_TV1};
        LP.SCHED{4}.config{1}.channels = {PDSCH_TV2};
        LP.SCHED{5}.config{1}.channels = {PUSCH_TV1};
        LP.SCHED{6}.config{1}.channels = {PDSCH_TV1};
        LP.SCHED{7}.config{1}.channels = {PDSCH_TV1};
        LP.SCHED{8}.config{1}.channels = {PDCCH_DL_TV1, PDCCH_UL_TV1, PDSCH_TV1};
        LP.SCHED{9}.config{1}.channels = {PDSCH_TV2};
        LP.SCHED{10}.config{1}.channels = {PUSCH_TV1};
        LP.SCHED{11}.config{1}.channels = {PDSCH_TV1};
        LP.SCHED{12}.config{1}.channels = {PDSCH_TV1};
        LP.SCHED{13}.config{1}.channels = {PDSCH_TV1};
        LP.SCHED{14}.config{1}.channels = {PDSCH_TV2};
        LP.SCHED{15}.config{1}.channels = {PUSCH_TV1};
        LP.SCHED{16}.config{1}.channels = {PDSCH_TV1};
        LP.SCHED{17}.config{1}.channels = {PDSCH_TV1};
        LP.SCHED{18}.config{1}.channels = {PDSCH_TV1};
        LP.SCHED{19}.config{1}.channels = {PDSCH_TV2};
        LP.SCHED{20}.config{1}.channels = {PUSCH_TV1};
        LPfilename = 'F08_1C';
    case 3 % F08_2C
        nCell = 2;
        PUSCH_TV1 = 'TVnr_CP_F08_US_01.1_gNB_FAPI_s5.h5';
        PDSCH_TV1 = 'TVnr_CP_F08_DS_01.1_gNB_FAPI_s5.h5';
        PDSCH_TV2 = 'TVnr_CP_F08_DS_01.2_gNB_FAPI_s5.h5';
        PDSCH_TV3 = 'TVnr_CP_F08_DS_01.1_126_gNB_FAPI_s5.h5';
        PBCH_TV1 = 'TVnr_1902_gNB_FAPI_s0.h5';
        PDCCH_UL_TV1 = 'demo_msg4_4_ant_pdcch_ul_gNB_FAPI_s7.h5';
        PDCCH_DL_TV1 = 'demo_msg2_4_ant_pdcch_dl_gNB_FAPI_s7.h5';
        LP = [];
        LP.Cell_Configs = {PUSCH_TV1, PUSCH_TV1};
        LP = init_launchPattern(LP, nSlot, nCell);
        for idxCell = 1:nCell
            LP.SCHED{1}.config{idxCell}.channels = {PBCH_TV1, PDSCH_TV3};
            LP.SCHED{2}.config{idxCell}.channels = {PDSCH_TV1};
            LP.SCHED{3}.config{idxCell}.channels = {PDSCH_TV1};
            LP.SCHED{4}.config{idxCell}.channels = {PDSCH_TV2};
            LP.SCHED{5}.config{idxCell}.channels = {PUSCH_TV1};
            LP.SCHED{6}.config{idxCell}.channels = {PDSCH_TV1};
            LP.SCHED{7}.config{idxCell}.channels = {PDSCH_TV1};
            LP.SCHED{8}.config{idxCell}.channels = {PDCCH_DL_TV1, PDCCH_UL_TV1, PDSCH_TV1};
            LP.SCHED{9}.config{idxCell}.channels = {PDSCH_TV2};
            LP.SCHED{10}.config{idxCell}.channels = {PUSCH_TV1};
            LP.SCHED{11}.config{idxCell}.channels = {PDSCH_TV1};
            LP.SCHED{12}.config{idxCell}.channels = {PDSCH_TV1};
            LP.SCHED{13}.config{idxCell}.channels = {PDSCH_TV1};
            LP.SCHED{14}.config{idxCell}.channels = {PDSCH_TV2};
            LP.SCHED{15}.config{idxCell}.channels = {PUSCH_TV1};
            LP.SCHED{16}.config{idxCell}.channels = {PDSCH_TV1};
            LP.SCHED{17}.config{idxCell}.channels = {PDSCH_TV1};
            LP.SCHED{18}.config{idxCell}.channels = {PDSCH_TV1};
            LP.SCHED{19}.config{idxCell}.channels = {PDSCH_TV2};
            LP.SCHED{20}.config{idxCell}.channels = {PUSCH_TV1};
        end
        LPfilename = 'F08_2C';
    case 4 % F08_3C
        nCell = 3;
        PUSCH_TV1 = 'TVnr_CP_F08_US_01.1_gNB_FAPI_s5.h5';
        PDSCH_TV1 = 'TVnr_CP_F08_DS_01.1_gNB_FAPI_s5.h5';
        PDSCH_TV2 = 'TVnr_CP_F08_DS_01.2_gNB_FAPI_s5.h5';
        PDSCH_TV3 = 'TVnr_CP_F08_DS_01.1_126_gNB_FAPI_s5.h5';
        PBCH_TV1 = 'TVnr_1902_gNB_FAPI_s0.h5';
        PDCCH_UL_TV1 = 'demo_msg4_4_ant_pdcch_ul_gNB_FAPI_s7.h5';
        PDCCH_DL_TV1 = 'demo_msg2_4_ant_pdcch_dl_gNB_FAPI_s7.h5';
        LP = [];
        LP.Cell_Configs = {PUSCH_TV1, PUSCH_TV1, PUSCH_TV1};
        LP = init_launchPattern(LP, nSlot, nCell);
        for idxCell = 1:nCell
            LP.SCHED{1}.config{idxCell}.channels = {PBCH_TV1, PDSCH_TV3};
            LP.SCHED{2}.config{idxCell}.channels = {PDSCH_TV1};
            LP.SCHED{3}.config{idxCell}.channels = {PDSCH_TV1};
            LP.SCHED{4}.config{idxCell}.channels = {PDSCH_TV2};
            LP.SCHED{5}.config{idxCell}.channels = {PUSCH_TV1};
            LP.SCHED{6}.config{idxCell}.channels = {PDSCH_TV1};
            LP.SCHED{7}.config{idxCell}.channels = {PDSCH_TV1};
            LP.SCHED{8}.config{idxCell}.channels = {PDCCH_DL_TV1, PDCCH_UL_TV1, PDSCH_TV1};
            LP.SCHED{9}.config{idxCell}.channels = {PDSCH_TV2};
            LP.SCHED{10}.config{idxCell}.channels = {PUSCH_TV1};
            LP.SCHED{11}.config{idxCell}.channels = {PDSCH_TV1};
            LP.SCHED{12}.config{idxCell}.channels = {PDSCH_TV1};
            LP.SCHED{13}.config{idxCell}.channels = {PDSCH_TV1};
            LP.SCHED{14}.config{idxCell}.channels = {PDSCH_TV2};
            LP.SCHED{15}.config{idxCell}.channels = {PUSCH_TV1};
            LP.SCHED{16}.config{idxCell}.channels = {PDSCH_TV1};
            LP.SCHED{17}.config{idxCell}.channels = {PDSCH_TV1};
            LP.SCHED{18}.config{idxCell}.channels = {PDSCH_TV1};
            LP.SCHED{19}.config{idxCell}.channels = {PDSCH_TV2};
            LP.SCHED{20}.config{idxCell}.channels = {PUSCH_TV1};
        end
        LPfilename = 'F08_3C';
    case 5 % F08_4C
        nCell = 4;
        PUSCH_TV1 = 'TVnr_CP_F08_US_01.1_gNB_FAPI_s5.h5';
        PDSCH_TV1 = 'TVnr_CP_F08_DS_01.1_gNB_FAPI_s5.h5';
        PDSCH_TV2 = 'TVnr_CP_F08_DS_01.2_gNB_FAPI_s5.h5';
        PDSCH_TV3 = 'TVnr_CP_F08_DS_01.1_126_gNB_FAPI_s5.h5';
        PBCH_TV1 = 'TVnr_1902_gNB_FAPI_s0.h5';
        PDCCH_UL_TV1 = 'demo_msg4_4_ant_pdcch_ul_gNB_FAPI_s7.h5';
        PDCCH_DL_TV1 = 'demo_msg2_4_ant_pdcch_dl_gNB_FAPI_s7.h5';
        LP = [];
        LP.Cell_Configs = {PUSCH_TV1, PUSCH_TV1, PUSCH_TV1, PUSCH_TV1};
        LP = init_launchPattern(LP, nSlot, nCell);
        for idxCell = 1:nCell
            LP.SCHED{1}.config{idxCell}.channels = {PBCH_TV1, PDSCH_TV3};
            LP.SCHED{2}.config{idxCell}.channels = {PDSCH_TV1};
            LP.SCHED{3}.config{idxCell}.channels = {PDSCH_TV1};
            LP.SCHED{4}.config{idxCell}.channels = {PDSCH_TV2};
            LP.SCHED{5}.config{idxCell}.channels = {PUSCH_TV1};
            LP.SCHED{6}.config{idxCell}.channels = {PDSCH_TV1};
            LP.SCHED{7}.config{idxCell}.channels = {PDSCH_TV1};
            LP.SCHED{8}.config{idxCell}.channels = {PDCCH_DL_TV1, PDCCH_UL_TV1, PDSCH_TV1};
            LP.SCHED{9}.config{idxCell}.channels = {PDSCH_TV2};
            LP.SCHED{10}.config{idxCell}.channels = {PUSCH_TV1};
            LP.SCHED{11}.config{idxCell}.channels = {PDSCH_TV1};
            LP.SCHED{12}.config{idxCell}.channels = {PDSCH_TV1};
            LP.SCHED{13}.config{idxCell}.channels = {PDSCH_TV1};
            LP.SCHED{14}.config{idxCell}.channels = {PDSCH_TV2};
            LP.SCHED{15}.config{idxCell}.channels = {PUSCH_TV1};
            LP.SCHED{16}.config{idxCell}.channels = {PDSCH_TV1};
            LP.SCHED{17}.config{idxCell}.channels = {PDSCH_TV1};
            LP.SCHED{18}.config{idxCell}.channels = {PDSCH_TV1};
            LP.SCHED{19}.config{idxCell}.channels = {PDSCH_TV2};
            LP.SCHED{20}.config{idxCell}.channels = {PUSCH_TV1};
        end
        LPfilename = 'F08_4C';
    case 6 % F13_1C
        nCell = 1;
        PUSCH_TV1 = 'TVnr_CP_F14_US_01_gNB_FAPI_s5.h5';
        PDSCH_TV1 = 'TVnr_CP_F14_DS_01_gNB_FAPI_s5.h5';
        LP = [];
        LP.Cell_Configs = {PUSCH_TV1};
        LP = init_launchPattern(LP, nSlot, nCell);
        
        UL_slotIdx = [4, 9, 14, 19];
        DL_slotIdx = [0:19];
        [~,TcIdx] = ismember(UL_slotIdx, DL_slotIdx);
        DL_slotIdx(TcIdx) = [];
        for slotIdx = UL_slotIdx
            LP.SCHED{slotIdx+1}.config{1}.channels = {PUSCH_TV1};
        end
        for slotIdx = DL_slotIdx
            LP.SCHED{slotIdx+1}.config{1}.channels = {PDSCH_TV1};
        end
        LPfilename = 'F13_1C';
    case 7 % F13_2C
        nCell = 2;
        PUSCH_TV1 = 'TVnr_CP_F14_US_01_gNB_FAPI_s5.h5';
        PUSCH_TV2 = 'TVnr_CP_F14_US_33_gNB_FAPI_s5.h5';
        PDSCH_TV1 = 'TVnr_CP_F14_DS_01_gNB_FAPI_s5.h5';
        PDSCH_TV2 = 'TVnr_CP_F14_DS_33_gNB_FAPI_s5.h5';
        LP = [];
        LP.Cell_Configs = {PUSCH_TV1, PUSCH_TV2};
        LP = init_launchPattern(LP, nSlot, nCell);
        
        UL_slotIdx = [4, 9, 14, 19];
        DL_slotIdx = [0:19];
        [~,TcIdx] = ismember(UL_slotIdx, DL_slotIdx);
        DL_slotIdx(TcIdx) = [];
        for slotIdx = UL_slotIdx
            LP.SCHED{slotIdx+1}.config{1}.channels = {PUSCH_TV1};
            LP.SCHED{slotIdx+1}.config{2}.channels = {PUSCH_TV2};
        end
        for slotIdx = DL_slotIdx
            LP.SCHED{slotIdx+1}.config{1}.channels = {PDSCH_TV1};
            LP.SCHED{slotIdx+1}.config{2}.channels = {PDSCH_TV2};            
        end
        LPfilename = 'F13_2C';          
    case 8 % F13_3C
        nCell = 3;
        PUSCH_TV1 = 'TVnr_CP_F14_US_01_gNB_FAPI_s5.h5';
        PUSCH_TV2 = 'TVnr_CP_F14_US_33_gNB_FAPI_s5.h5';
        PUSCH_TV3 = 'TVnr_CP_F14_US_34_gNB_FAPI_s5.h5';
        PDSCH_TV1 = 'TVnr_CP_F14_DS_01_gNB_FAPI_s5.h5';
        PDSCH_TV2 = 'TVnr_CP_F14_DS_33_gNB_FAPI_s5.h5';
        PDSCH_TV3 = 'TVnr_CP_F14_DS_34_gNB_FAPI_s5.h5';        
        LP = [];
        LP.Cell_Configs = {PUSCH_TV1, PUSCH_TV2, PUSCH_TV3};
        LP = init_launchPattern(LP, nSlot, nCell);
        
        UL_slotIdx = [4, 9, 14, 19];
        DL_slotIdx = [0:19];
        [~,TcIdx] = ismember(UL_slotIdx, DL_slotIdx);
        DL_slotIdx(TcIdx) = [];
        for slotIdx = UL_slotIdx
            LP.SCHED{slotIdx+1}.config{1}.channels = {PUSCH_TV1};
            LP.SCHED{slotIdx+1}.config{2}.channels = {PUSCH_TV2};
            LP.SCHED{slotIdx+1}.config{3}.channels = {PUSCH_TV3};
        end
        for slotIdx = DL_slotIdx
            LP.SCHED{slotIdx+1}.config{1}.channels = {PDSCH_TV1};
            LP.SCHED{slotIdx+1}.config{2}.channels = {PDSCH_TV2};            
            LP.SCHED{slotIdx+1}.config{3}.channels = {PDSCH_TV3};     
        end
        LPfilename = 'F13_3C';  
    case 9 % F13_4C
        nCell = 4;
        PUSCH_TV1 = 'TVnr_CP_F14_US_01_gNB_FAPI_s5.h5';
        PUSCH_TV2 = 'TVnr_CP_F14_US_33_gNB_FAPI_s5.h5';
        PUSCH_TV3 = 'TVnr_CP_F14_US_34_gNB_FAPI_s5.h5';
        PUSCH_TV4 = 'TVnr_CP_F14_US_35_gNB_FAPI_s5.h5';
        PDSCH_TV1 = 'TVnr_CP_F14_DS_01_gNB_FAPI_s5.h5';
        PDSCH_TV2 = 'TVnr_CP_F14_DS_33_gNB_FAPI_s5.h5';
        PDSCH_TV3 = 'TVnr_CP_F14_DS_34_gNB_FAPI_s5.h5';        
        PDSCH_TV4 = 'TVnr_CP_F14_DS_35_gNB_FAPI_s5.h5'; 
        LP = [];
        LP.Cell_Configs = {PUSCH_TV1, PUSCH_TV2, PUSCH_TV3, PUSCH_TV4};
        LP = init_launchPattern(LP, nSlot, nCell);
        
        UL_slotIdx = [4, 9, 14, 19];
        DL_slotIdx = [0:19];
        [~,TcIdx] = ismember(UL_slotIdx, DL_slotIdx);
        DL_slotIdx(TcIdx) = [];
        for slotIdx = UL_slotIdx
            LP.SCHED{slotIdx+1}.config{1}.channels = {PUSCH_TV1};
            LP.SCHED{slotIdx+1}.config{2}.channels = {PUSCH_TV2};
            LP.SCHED{slotIdx+1}.config{3}.channels = {PUSCH_TV3};
            LP.SCHED{slotIdx+1}.config{4}.channels = {PUSCH_TV4};
        end
        for slotIdx = DL_slotIdx
            LP.SCHED{slotIdx+1}.config{1}.channels = {PDSCH_TV1};
            LP.SCHED{slotIdx+1}.config{2}.channels = {PDSCH_TV2};            
            LP.SCHED{slotIdx+1}.config{3}.channels = {PDSCH_TV3};   
            LP.SCHED{slotIdx+1}.config{4}.channels = {PDSCH_TV4};
        end
        LPfilename = 'F13_4C'; 
end

tvDirName = 'GPU_test_input';
[status,msg] = mkdir(tvDirName);

TVname = sprintf('launch_pattern_%s', LPfilename);
yamlFileName = [tvDirName filesep TVname '.yaml'];

WriteYaml(yamlFileName, LP);

return

function LP = init_launchPattern(LP, nSlot, nCell)

for idxSlot = 1:nSlot
    for idxCell = 1:nCell        
        LP.SCHED{idxSlot}.config{idxCell}.cell_index = idxCell-1;
        LP.SCHED{idxSlot}.config{idxCell}.channels = {};
        LP.SCHED{idxSlot}.slot = idxSlot-1;
    end
end

return
