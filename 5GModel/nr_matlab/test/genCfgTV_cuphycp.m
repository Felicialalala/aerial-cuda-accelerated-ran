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

function [errCnt, nTV] = genCfgTV_cuphycp(caseSet)

if nargin == 0
    caseSet = 'full';
end

switch caseSet
    case 'full'
        nTV = 13;
        gen_F08_DS_01_1_126 = 1;
        gen_F08_DS_01_1     = 1;
        gen_F08_DS_01_2     = 1;
        gen_F14_DS_01       = 1;
        gen_F14_DS_33       = 1;
        gen_F14_DS_34       = 1;
        gen_F14_DS_35       = 1;
        gen_F01_US_09_1     = 1;
        gen_F08_US_01_1     = 1;
        gen_F14_US_01       = 1;
        gen_F14_US_33       = 1;
        gen_F14_US_34       = 1;
        gen_F14_US_35       = 1;
    case 'compact'
        nTV = 13;
        gen_F08_DS_01_1_126 = 1;
        gen_F08_DS_01_1     = 1;
        gen_F08_DS_01_2     = 1;
        gen_F14_DS_01       = 1;
        gen_F14_DS_33       = 1;
        gen_F14_DS_34       = 1;
        gen_F14_DS_35       = 1;
        gen_F01_US_09_1     = 1;  % seems not used
        gen_F08_US_01_1     = 1;
        gen_F14_US_01       = 1;
        gen_F14_US_33       = 1;
        gen_F14_US_34       = 1;
        gen_F14_US_35       = 1;
    case 'selected'
        nTV = 2;
        gen_F08_DS_01_1_126 = 1;
        gen_F08_DS_01_1     = 1;
        gen_F08_DS_01_2     = 1;
        gen_F14_DS_01       = 1;
        gen_F14_DS_33       = 1;
        gen_F14_DS_34       = 1;
        gen_F14_DS_35       = 1;
        gen_F01_US_09_1     = 1;
        gen_F08_US_01_1     = 1;
        gen_F14_US_01       = 1;
        gen_F14_US_33       = 1;
        gen_F14_US_34       = 1;
        gen_F14_US_35       = 1;
    otherwise
        error('caseSet is not supported...\n');
end

errCnt = 0;

[status,msg] = mkdir('GPU_test_input');

TC_list = [gen_F08_DS_01_1_126, gen_F08_DS_01_1, gen_F08_DS_01_2, ...
    gen_F14_DS_01, gen_F14_DS_33, gen_F14_DS_34, ...
    gen_F14_DS_35, gen_F01_US_09_1, gen_F08_US_01_1, ...
    gen_F14_US_01, gen_F14_US_33, gen_F14_US_34, gen_F14_US_35];

parfor idxTc = 1:length(TC_list)
    TcList = zeros(1, length(TC_list));
    TcList(idxTc) = TC_list(idxTc);
    errCnt = genCpTV(TcList) + errCnt;
end   

return

function errCnt = genCpTV(TcList)

gen_F08_DS_01_1_126 = TcList(1);
gen_F08_DS_01_1     = TcList(2);
gen_F08_DS_01_2     = TcList(3);
gen_F14_DS_01       = TcList(4);
gen_F14_DS_33       = TcList(5);
gen_F14_DS_34       = TcList(6);
gen_F14_DS_35       = TcList(7);
gen_F01_US_09_1     = TcList(8);
gen_F08_US_01_1     = TcList(9);
gen_F14_US_01       = TcList(10);
gen_F14_US_33       = TcList(11);
gen_F14_US_34       = TcList(12);
gen_F14_US_35       = TcList(13);

errCnt = 0;

slotIdx = 5;
forceSlotIdxFlag = 1;

%% F18_DS
SysPar = initSysPar;
pdsch0 = SysPar.pdsch{1};
SysPar = [];
SysPar.testAlloc.dl = 1;
SysPar.testAlloc.ul = 0;
SysPar.testAlloc.ssb = 0;
SysPar.testAlloc.pdcch = 0;
SysPar.testAlloc.pdsch = 1;
SysPar.testAlloc.csirs = 0;
SysPar.testAlloc.prach = 0;
SysPar.testAlloc.pucch = 0;
SysPar.testAlloc.pusch = 0;
SysPar.testAlloc.srs = 0;
SysPar.carrier.N_grid_size_mu = 273;
SysPar.carrier.mu = 1;
SysPar.carrier.Nant_gNB = 4;
SysPar.carrier.Nant_UE = 4;
SysPar.pdsch{1} = pdsch0;
SysPar.pdsch{1}.rbStart = 0;
SysPar.pdsch{1}.rbSize = 273;
SysPar.pdsch{1}.StartSymbolIndex = 2;
SysPar.pdsch{1}.NrOfSymbols = 12;
SysPar.pdsch{1}.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 0 0 0];
SysPar.pdsch{1}.nrOfLayers = 4;
SysPar.pdsch{1}.mcsTable = 1;
SysPar.SimCtrl.N_UE = 1;
SysPar.pdsch{1}.mcsIndex = 27;

% TV_api_cuphy_F08-DS-01.1_126_slot5_MIMO4x4_PRB126_DataSyms11_qam256.h5
% => TVnr_CP_F08_DS_01.1_126
if gen_F08_DS_01_1_126
    SysPar.pdsch{1}.rbSize = 126;
    SysPar.SimCtrl.genTV.forceSlotIdxFlag = forceSlotIdxFlag;
    SysPar.SimCtrl.genTV.slotIdx = slotIdx;
    if forceSlotIdxFlag == 0
         SysPar.SimCtrl.N_slot_run = slotIdx + 1;
    end
    fileName = ['GPU_test_input/TVnr_CP_F08_DS_01.1_126.yaml'];
    WriteYaml(fileName, SysPar);
    cfgFile = ['GPU_test_input/TVnr_CP_F08_DS_01.1_126.yaml'];
    TVfile = ['TVnr_CP_F08_DS_01.1_126'];
    errFlag = runSim(cfgFile, TVfile);
    errCnt = errCnt + errFlag;
end

% TV_api_cuphy_F08-DS-01.1_slot5_MIMO4x4_PRB273_DataSyms11_qam256.h5
% => TVnr_CP_F08_DS_01.1
if gen_F08_DS_01_1
    SysPar.pdsch{1}.rbSize = 273;
    SysPar.SimCtrl.genTV.forceSlotIdxFlag = forceSlotIdxFlag;
    SysPar.SimCtrl.genTV.slotIdx = slotIdx;
    if forceSlotIdxFlag == 0
         SysPar.SimCtrl.N_slot_run = slotIdx + 1;
    end    
    fileName = ['GPU_test_input/TVnr_CP_F08_DS_01.1.yaml'];
    WriteYaml(fileName, SysPar);
    cfgFile = ['GPU_test_input/TVnr_CP_F08_DS_01.1.yaml'];
    TVfile = ['TVnr_CP_F08_DS_01.1'];
    errFlag = runSim(cfgFile, TVfile);
    errCnt = errCnt + errFlag;
end

% TV_api_cuphy_F08-DS-01.1_slot5_MIMO4x4_PRB273_DataSyms9_qam256.h5
% => TVnr_CP_F08_DS_01.2
if gen_F08_DS_01_2
    SysPar.pdsch{1}.rbSize = 273;
    SysPar.pdsch{1}.NrOfSymbols = 10;
    SysPar.SimCtrl.genTV.forceSlotIdxFlag = forceSlotIdxFlag;
    SysPar.SimCtrl.genTV.slotIdx = slotIdx;
    if forceSlotIdxFlag == 0
         SysPar.SimCtrl.N_slot_run = slotIdx + 1;
    end    
    fileName = ['GPU_test_input/TVnr_CP_F08_DS_01.2.yaml'];
    WriteYaml(fileName, SysPar);
    cfgFile = ['GPU_test_input/TVnr_CP_F08_DS_01.2.yaml'];
    TVfile = ['TVnr_CP_F08_DS_01.2'];
    errFlag = runSim(cfgFile, TVfile);
    errCnt = errCnt + errFlag;
end

%% F14_DS
SysPar = initSysPar;
pdsch0 = SysPar.pdsch{1};
SysPar = [];
SysPar.testAlloc.dl = 1;
SysPar.testAlloc.ul = 0;
SysPar.testAlloc.ssb = 0;
SysPar.testAlloc.pdcch = 0;
SysPar.testAlloc.pdsch = 4;
SysPar.testAlloc.csirs = 0;
SysPar.testAlloc.prach = 0;
SysPar.testAlloc.pucch = 0;
SysPar.testAlloc.pusch = 0;
SysPar.testAlloc.srs = 0;
SysPar.carrier.N_grid_size_mu = 273;
SysPar.carrier.mu = 1;
SysPar.carrier.Nant_gNB = 16;
SysPar.carrier.Nant_UE = 4;
for idxUE = 1:4
    SysPar.pdsch{idxUE} = pdsch0;
    SysPar.pdsch{idxUE}.rbStart = 0;
    SysPar.pdsch{idxUE}.rbSize = 273;
    SysPar.pdsch{idxUE}.StartSymbolIndex = 2;
    SysPar.pdsch{idxUE}.NrOfSymbols = 12;
    SysPar.pdsch{idxUE}.DmrsSymbPos = [0 0 1 1 0 0 0 0 0 0 0 0 0 0];
    SysPar.pdsch{idxUE}.nrOfLayers = 4;
    SysPar.pdsch{idxUE}.mcsTable = 1;
    SysPar.pdsch{idxUE}.portIdx = mod((idxUE-1)*4+[0:3], 8);
    SysPar.pdsch{idxUE}.SCID = floor((idxUE-1)*4/8);
    SysPar.pdsch{idxUE}.idxUE = idxUE-1;
    SysPar.pdsch{idxUE}.seed = idxUE-1;
    SysPar.pdsch{idxUE}.mcsIndex = 27;
end
SysPar.SimCtrl.N_UE = 4;
SysPar.SimCtrl.genTV.forceSlotIdxFlag = forceSlotIdxFlag;
SysPar.SimCtrl.genTV.slotIdx = slotIdx;
if forceSlotIdxFlag == 0
    SysPar.SimCtrl.N_slot_run = slotIdx + 1;
end
% TV_api_cuphy_F14-DS-01_slot5_MIMO16x16_PRB273_DataSyms11_qam256.h5
% => TVnr_CP_F14_DS_01
if gen_F14_DS_01
    fileName = ['GPU_test_input/TVnr_CP_F14_DS_01.yaml'];
    WriteYaml(fileName, SysPar);
    cfgFile = ['GPU_test_input/TVnr_CP_F14_DS_01.yaml'];
    TVfile = ['TVnr_CP_F14_DS_01'];
    errFlag = runSim(cfgFile, TVfile);
    errCnt = errCnt + errFlag;
end

% TV_api_cuphy_F14-DS-33_slot5_MIMO16x16_PRB64_DataSyms11_qam256.h5
% => TVnr_CP_F14_DS_33
if gen_F14_DS_33
    for idxUE = 1:4
        SysPar.pdsch{idxUE}.rbSize = 64;
        SysPar.pdsch{idxUE}.mcsIndex = 27-4+1;
    end
    fileName = ['GPU_test_input/TVnr_CP_F14_DS_33.yaml'];
    WriteYaml(fileName, SysPar);
    cfgFile = ['GPU_test_input/TVnr_CP_F14_DS_33.yaml'];
    TVfile = ['TVnr_CP_F14_DS_33'];
    errFlag = runSim(cfgFile, TVfile);
    errCnt = errCnt + errFlag;
end

% TV_api_cuphy_F14-DS-34_slot5_MIMO4x16_PRB72_DataSyms11_qam64.h5
% => TVnr_CP_F14_DS_34
if gen_F14_DS_34
    SysPar.testAlloc.pdsch = 2;
    SysPar.carrier.Nant_UE = 2;
    SysPar.pdsch = [];
    for idxUE = 1:2
        SysPar.pdsch{idxUE} = pdsch0;
        SysPar.pdsch{idxUE}.rbStart = 0;
        SysPar.pdsch{idxUE}.rbSize = 72;
        SysPar.pdsch{idxUE}.StartSymbolIndex = 2;
        SysPar.pdsch{idxUE}.NrOfSymbols = 12;
        SysPar.pdsch{idxUE}.DmrsSymbPos = [0 0 1 1 0 0 0 0 0 0 0 0 0 0];
        SysPar.pdsch{idxUE}.nrOfLayers = 2;
        SysPar.pdsch{idxUE}.mcsTable = 1;
        SysPar.pdsch{idxUE}.portIdx = mod((idxUE-1)*2+[0:1], 8);
        SysPar.pdsch{idxUE}.SCID = floor((idxUE-1)*4/8);
        SysPar.pdsch{idxUE}.idxUE = idxUE-1;
        SysPar.pdsch{idxUE}.seed = idxUE-1;
        SysPar.pdsch{idxUE}.mcsIndex = 27-15+1;
    end
    SysPar.SimCtrl.N_UE = 2;
    SysPar.SimCtrl.genTV.forceSlotIdxFlag = forceSlotIdxFlag;
    SysPar.SimCtrl.genTV.slotIdx = slotIdx;
    if forceSlotIdxFlag == 0
         SysPar.SimCtrl.N_slot_run = slotIdx + 1;
    end    
    fileName = ['GPU_test_input/TVnr_CP_F14_DS_34.yaml'];
    WriteYaml(fileName, SysPar);
    cfgFile = ['GPU_test_input/TVnr_CP_F14_DS_34.yaml'];
    TVfile = ['TVnr_CP_F14_DS_34'];
    errFlag = runSim(cfgFile, TVfile);
    errCnt = errCnt + errFlag;
end

% TV_api_cuphy_F14-DS-35_slot5_MIMO2x16_PRB40_DataSyms11_qam16.h5
% => TVnr_CP_F14_DS_35
if gen_F14_DS_35
    SysPar.testAlloc.pdsch = 2;
    SysPar.carrier.Nant_UE = 1;
    SysPar.pdsch = [];
    for idxUE = 1:2
        SysPar.pdsch{idxUE} = pdsch0;
        SysPar.pdsch{idxUE}.rbStart = 0;
        SysPar.pdsch{idxUE}.rbSize = 40;
        SysPar.pdsch{idxUE}.StartSymbolIndex = 2;
        SysPar.pdsch{idxUE}.NrOfSymbols = 12;
        SysPar.pdsch{idxUE}.DmrsSymbPos = [0 0 1 1 0 0 0 0 0 0 0 0 0 0];
        SysPar.pdsch{idxUE}.nrOfLayers = 1;
        SysPar.pdsch{idxUE}.mcsTable = 1;
        SysPar.pdsch{idxUE}.portIdx = mod((idxUE-1)*1+0, 8);
        SysPar.pdsch{idxUE}.SCID = floor((idxUE-1)*4/8);
        SysPar.pdsch{idxUE}.idxUE = idxUE-1;
        SysPar.pdsch{idxUE}.seed = idxUE-1;
        SysPar.pdsch{idxUE}.mcsIndex = 27-21+1;
    end
    SysPar.SimCtrl.N_UE = 2;
    SysPar.SimCtrl.genTV.forceSlotIdxFlag = forceSlotIdxFlag;
    SysPar.SimCtrl.genTV.slotIdx = slotIdx;
    if forceSlotIdxFlag == 0
         SysPar.SimCtrl.N_slot_run = slotIdx + 1;
    end    
    fileName = ['GPU_test_input/TVnr_CP_F14_DS_35.yaml'];
    WriteYaml(fileName, SysPar);
    cfgFile = ['GPU_test_input/TVnr_CP_F14_DS_35.yaml'];
    TVfile = ['TVnr_CP_F14_DS_35'];
    errFlag = runSim(cfgFile, TVfile);
    errCnt = errCnt + errFlag;
end

%% F01-US
SysPar = initSysPar;
pusch0 = SysPar.pusch{1};
SysPar = [];
SysPar.testAlloc.dl = 0;
SysPar.testAlloc.ul = 1;
SysPar.testAlloc.ssb = 0;
SysPar.testAlloc.pdcch = 0;
SysPar.testAlloc.pdsch = 0;
SysPar.testAlloc.csirs = 0;
SysPar.testAlloc.prach = 0;
SysPar.testAlloc.pucch = 0;
SysPar.testAlloc.pusch = 1;
SysPar.testAlloc.srs = 0;
SysPar.carrier.N_grid_size_mu = 106;
SysPar.carrier.mu = 0;
SysPar.carrier.Nant_gNB = 4;
SysPar.carrier.Nant_UE = 2;
SysPar.pusch{1} = pusch0;
SysPar.pusch{1}.rbStart = 0;
SysPar.pusch{1}.rbSize = 104;
SysPar.pusch{1}.StartSymbolIndex = 0;
SysPar.pusch{1}.NrOfSymbols = 14;
SysPar.pusch{1}.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 0 0 0];
SysPar.pusch{1}.nrOfLayers = 2;
SysPar.pusch{1}.portIdx = [0, 1];
SysPar.pusch{1}.mcsTable = 1;
% SysPar.Chan{1}.type = 'TDLA30-10-Low';
% SysPar.SimCtrl.timeDomainSim = 1;
SysPar.SimCtrl.N_UE = 1;
SysPar.SimCtrl.genTV.forceSlotIdxFlag = forceSlotIdxFlag;
SysPar.SimCtrl.genTV.slotIdx = slotIdx;
if forceSlotIdxFlag == 0
    SysPar.SimCtrl.N_slot_run = slotIdx + 1;
end

% TV_cuphy_F01-US-09.1_snrdb40.00_MIMO2x4_PRB104_DataSyms13_qam256.h5
% => TVnr_CP_F01_US_09.1
if gen_F01_US_09_1
    SysPar.pusch{1}.mcsIndex = 27;
    SysPar = updateAlgFlag(SysPar);
    fileName = ['GPU_test_input/TVnr_CP_F01_US_09.1.yaml'];
    WriteYaml(fileName, SysPar);
    cfgFile = ['GPU_test_input/TVnr_CP_F01_US_09.1.yaml'];
    TVfile = ['TVnr_CP_F01_US_09.1'];
    errFlag = runSim(cfgFile, TVfile);
    errCnt = errCnt + errFlag;
end

%% F08-US
SysPar = initSysPar;
pusch0 = SysPar.pusch{1};
SysPar = [];
SysPar.testAlloc.dl = 0;
SysPar.testAlloc.ul = 1;
SysPar.testAlloc.ssb = 0;
SysPar.testAlloc.pdcch = 0;
SysPar.testAlloc.pdsch = 0;
SysPar.testAlloc.csirs = 0;
SysPar.testAlloc.prach = 0;
SysPar.testAlloc.pucch = 0;
SysPar.testAlloc.pusch = 1;
SysPar.testAlloc.srs = 0;
SysPar.carrier.N_grid_size_mu = 273;
SysPar.carrier.mu = 1;
SysPar.carrier.Nant_gNB = 4;
SysPar.carrier.Nant_UE = 2;
SysPar.pusch{1} = pusch0;
SysPar.pusch{1}.rbStart = 0;
SysPar.pusch{1}.rbSize = 272;
SysPar.pusch{1}.StartSymbolIndex = 0;
SysPar.pusch{1}.NrOfSymbols = 14;
SysPar.pusch{1}.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 0 0 0];
SysPar.pusch{1}.nrOfLayers = 2;
SysPar.pusch{1}.mcsTable = 1;
% SysPar.Chan{1}.type = 'TDLA30-10-Low';
% SysPar.SimCtrl.timeDomainSim = 1;
SysPar.SimCtrl.N_UE = 1;
SysPar.SimCtrl.genTV.forceSlotIdxFlag = forceSlotIdxFlag;
SysPar.SimCtrl.genTV.slotIdx = slotIdx;
if forceSlotIdxFlag == 0
    SysPar.SimCtrl.N_slot_run = slotIdx + 1;
end

% TV_cuphy_F08-US-01.1_snrdb40.00_MIMO1x4_PRB272_DataSyms13_qam256.h5
% => F08_US_01.1
if gen_F08_US_01_1
    SysPar.pusch{1}.mcsIndex = 27;
    SysPar = updateAlgFlag(SysPar);
    fileName = ['GPU_test_input/TVnr_CP_F08_US_01.1.yaml'];
    WriteYaml(fileName, SysPar);
    cfgFile = ['GPU_test_input/TVnr_CP_F08_US_01.1.yaml'];
    TVfile = ['TVnr_CP_F08_US_01.1'];
    errFlag = runSim(cfgFile, TVfile);
    errCnt = errCnt + errFlag;
end

%% F14-US
SysPar = initSysPar;
pusch0 = SysPar.pusch{1};
SysPar = [];
SysPar.testAlloc.dl = 0;
SysPar.testAlloc.ul = 1;
SysPar.testAlloc.ssb = 0;
SysPar.testAlloc.pdcch = 0;
SysPar.testAlloc.pdsch = 0;
SysPar.testAlloc.csirs = 0;
SysPar.testAlloc.prach = 0;
SysPar.testAlloc.pucch = 0;
SysPar.testAlloc.pusch = 2;
SysPar.testAlloc.srs = 0;
SysPar.carrier.N_grid_size_mu = 273;
SysPar.carrier.mu = 1;
SysPar.carrier.Nant_gNB = 16;
SysPar.carrier.Nant_UE = 4;
for idxUE = 1:2
    SysPar.pusch{idxUE} = pusch0;
    SysPar.pusch{idxUE}.rbStart = 0;
    SysPar.pusch{idxUE}.rbSize = 273;
    SysPar.pusch{idxUE}.StartSymbolIndex = 0;
    SysPar.pusch{idxUE}.NrOfSymbols = 12;
    SysPar.pusch{idxUE}.DmrsSymbPos = [0 0 1 1 0 0 0 0 0 0 0 0 0 0];
    SysPar.pusch{idxUE}.nrOfLayers = 4;
    SysPar.pusch{idxUE}.mcsTable = 1;
    SysPar.pusch{idxUE}.portIdx = (idxUE-1)*4+[0:3];
    SysPar.pusch{idxUE}.idxUE = idxUE-1;
    SysPar.pusch{idxUE}.seed = idxUE-1;
    SysPar.pusch{idxUE}.mcsIndex = 27;
end
% SysPar.Chan{1}.type = 'TDLA30-10-Low';
% SysPar.SimCtrl.timeDomainSim = 1;
SysPar.SimCtrl.N_UE = 2;
SysPar.SimCtrl.genTV.forceSlotIdxFlag = forceSlotIdxFlag;
SysPar.SimCtrl.genTV.slotIdx = slotIdx;
if forceSlotIdxFlag == 0
    SysPar.SimCtrl.N_slot_run = slotIdx + 1;
end

% TV_cuphy_F14-US-01_snrdb40.00_MIMO8x16_PRB272_DataSyms12_qam256.h5
% => F14_US_01
if gen_F14_US_01    
    SysPar = updateAlgFlag(SysPar);
    fileName = ['GPU_test_input/TVnr_CP_F14_US_01.yaml'];
    WriteYaml(fileName, SysPar);
    cfgFile = ['GPU_test_input/TVnr_CP_F14_US_01.yaml'];
    TVfile = ['TVnr_CP_F14_US_01'];
    errFlag = runSim(cfgFile, TVfile);
    errCnt = errCnt + errFlag;
end

% TV_cuphy_F14-US-33_snrdb40.00_MIMO8x16_PRB64_DataSyms12_qam256.h5
% => F14_US_33
if gen_F14_US_33
    for idxUE = 1:2
        SysPar.pusch{idxUE}.rbSize = 64;
        SysPar.pusch{idxUE}.mcsIndex = 27-4+1;
    end    
    SysPar = updateAlgFlag(SysPar);
    fileName = ['GPU_test_input/TVnr_CP_F14_US_33.yaml'];
    WriteYaml(fileName, SysPar);
    cfgFile = ['GPU_test_input/TVnr_CP_F14_US_33.yaml'];
    TVfile = ['TVnr_CP_F14_US_33'];
    errFlag = runSim(cfgFile, TVfile);
    errCnt = errCnt + errFlag;
end

% TV_cuphy_F14-US-34_snrdb40.00_MIMO2x16_PRB72_DataSyms12_qam64.h5
% => F14_US_34
if gen_F14_US_34
    SysPar.testAlloc.pusch = 2;
    SysPar.carrier.Nant_UE = 2;
    SysPar.pusch = [];
    for idxUE = 1:2
        SysPar.pusch{idxUE} = pusch0;
        SysPar.pusch{idxUE}.rbStart = 0;
        SysPar.pusch{idxUE}.rbSize = 72;
        SysPar.pusch{idxUE}.StartSymbolIndex = 0;
        SysPar.pusch{idxUE}.NrOfSymbols = 12;
        SysPar.pusch{idxUE}.DmrsSymbPos = [0 0 1 1 0 0 0 0 0 0 0 0 0 0];
        SysPar.pusch{idxUE}.nrOfLayers = 1;
        SysPar.pusch{idxUE}.mcsTable = 0;
        SysPar.pusch{idxUE}.portIdx = idxUE-1;
        SysPar.pusch{idxUE}.idxUE = idxUE-1;
        SysPar.pusch{idxUE}.seed = idxUE-1;
        SysPar.pusch{idxUE}.mcsIndex = 20;
    end
    SysPar.SimCtrl.N_UE = 2;
    SysPar.SimCtrl.genTV.forceSlotIdxFlag = forceSlotIdxFlag;
    SysPar.SimCtrl.genTV.slotIdx = slotIdx;
    if forceSlotIdxFlag == 0
         SysPar.SimCtrl.N_slot_run = slotIdx + 1;
    end    
    SysPar = updateAlgFlag(SysPar);
    fileName = ['GPU_test_input/TVnr_CP_F14_US_34.yaml'];
    WriteYaml(fileName, SysPar);
    cfgFile = ['GPU_test_input/TVnr_CP_F14_US_34.yaml'];
    TVfile = ['TVnr_CP_F14_US_34'];
    errFlag = runSim(cfgFile, TVfile);
    errCnt = errCnt + errFlag;
end

% TV_cuphy_F14-US-35_snrdb40.00_MIMO1x16_PRB40_DataSyms12_qam16.h5
% => F14_US_35
if gen_F14_US_35
    SysPar.testAlloc.pusch = 1;
    SysPar.carrier.Nant_UE = 1;
    SysPar.pusch = [];
    for idxUE = 1:1
        SysPar.pusch{idxUE} = pusch0;
        SysPar.pusch{idxUE}.rbStart = 0;
        SysPar.pusch{idxUE}.rbSize = 72;
        SysPar.pusch{idxUE}.StartSymbolIndex = 0;
        SysPar.pusch{idxUE}.NrOfSymbols = 12;
        SysPar.pusch{idxUE}.DmrsSymbPos = [0 0 1 1 0 0 0 0 0 0 0 0 0 0];
        SysPar.pusch{idxUE}.nrOfLayers = 1;
        SysPar.pusch{idxUE}.mcsTable = 0;
        SysPar.pusch{idxUE}.portIdx = idxUE-1;
        SysPar.pusch{idxUE}.idxUE = idxUE-1;
        SysPar.pusch{idxUE}.seed = idxUE-1;
        SysPar.pusch{idxUE}.mcsIndex = 13;
    end
    SysPar.SimCtrl.N_UE = 1;
    SysPar.SimCtrl.genTV.forceSlotIdxFlag = forceSlotIdxFlag;
    SysPar.SimCtrl.genTV.slotIdx = slotIdx;
    if forceSlotIdxFlag == 0
         SysPar.SimCtrl.N_slot_run = slotIdx + 1;
    end    
    SysPar = updateAlgFlag(SysPar);
    fileName = ['GPU_test_input/TVnr_CP_F14_US_35.yaml'];
    WriteYaml(fileName, SysPar);
    cfgFile = ['GPU_test_input/TVnr_CP_F14_US_35.yaml'];
    TVfile = ['TVnr_CP_F14_US_35'];
    errFlag = runSim(cfgFile, TVfile);
    errCnt = errCnt + errFlag;
end

return
