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

function [errCnt, nTV] = genCfgTV_perf(caseSet)

slotIdx = 0;

if nargin == 0
    caseSet = 'full';
end


switch caseSet
    case 'full'
        F01_DS_caseSet = 1:23;
        F01_US_caseSet = 1:23;
        F08_DS_caseSet = 1:23;
        F08_US_caseSet = 1:23;
        F14_DS_caseSet = 1:23;
        F14_US_caseSet = 1:23;
    case 'compact'
        F01_DS_caseSet = 1;
        F01_US_caseSet = 1;
        F08_DS_caseSet = 1;
        F08_US_caseSet = 1;
        F14_DS_caseSet = 1;
        F14_US_caseSet = 1;        
    case 'selected'
        F01_DS_caseSet = [];
        F01_US_caseSet = [];
        F08_DS_caseSet = [];
        F08_US_caseSet = [];
        F14_DS_caseSet = 1;
        F14_US_caseSet = 1;
    otherwise
        error('caseSet is not supported...\n');
end

nTV = length(F01_DS_caseSet) + length(F01_US_caseSet) + ...
      length(F08_DS_caseSet) + length(F08_US_caseSet) + ...
      length(F14_DS_caseSet) + length(F14_US_caseSet);
errCnt = 0;  

[status,msg] = mkdir('GPU_test_input');

% F01-PDSCH
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
SysPar.carrier.N_grid_size_mu = 106;
SysPar.carrier.mu = 0;
SysPar.carrier.Nant_gNB = 4;
SysPar.carrier.Nant_UE = 4; 
SysPar.pdsch{1} = pdsch0;
SysPar.pdsch{1}.rbStart = 0;
SysPar.pdsch{1}.rbSize = 106;
SysPar.pdsch{1}.StartSymbolIndex = 2;
SysPar.pdsch{1}.NrOfSymbols = 12;
SysPar.pdsch{1}.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 0 0 0];
SysPar.pdsch{1}.nrOfLayers = 4;
SysPar.pdsch{1}.mcsTable = 1;
SysPar.SimCtrl.N_UE = 1;
SysPar.SimCtrl.N_slot_run = slotIdx + 1;
SysPar.SimCtrl.genTV.slotIdx = slotIdx;
for idxCase = F01_DS_caseSet
    SysPar.pdsch{1}.mcsIndex = 27-idxCase+1;
    fileName = ['GPU_test_input/TVnr_F01_DS_', num2str(idxCase, '%02d'),'.yaml'];
    WriteYaml(fileName, SysPar);
    cfgFile = ['GPU_test_input/TVnr_F01_DS_', num2str(idxCase, '%02d'),'.yaml'];
    TVfile = ['TVnr_F01_DS_', num2str(idxCase, '%02d')];
    errFlag = runSim(cfgFile, TVfile);    
    errCnt = errCnt + errFlag;
end

% F01-PUSCH
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
SysPar.carrier.Nant_UE = 1; 
SysPar.pusch{1} = pusch0;
SysPar.pusch{1}.rbStart = 0;
SysPar.pusch{1}.rbSize = 106;
SysPar.pusch{1}.StartSymbolIndex = 0;
SysPar.pusch{1}.NrOfSymbols = 14;
SysPar.pusch{1}.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 0 0 0];
SysPar.pusch{1}.nrOfLayers = 1;
SysPar.pusch{1}.mcsTable = 1;
% SysPar.Chan{1}.type = 'TDLA30-10-Low';
% SysPar.SimCtrl.timeDomainSim = 1;
SysPar.SimCtrl.N_UE = 1;
SysPar.SimCtrl.N_slot_run = slotIdx + 1;
SysPar.SimCtrl.genTV.slotIdx = slotIdx;
for idxCase = F01_US_caseSet
    SysPar.pusch{1}.mcsIndex = 27-idxCase+1;
    fileName = ['GPU_test_input/TVnr_F01_US_', num2str(idxCase, '%02d'),'.yaml'];
    WriteYaml(fileName, SysPar);
    cfgFile = ['GPU_test_input/TVnr_F01_US_', num2str(idxCase, '%02d'),'.yaml'];
    TVfile = ['TVnr_F01_US_', num2str(idxCase, '%02d')];
    errFlag = runSim(cfgFile, TVfile);    
    errCnt = errCnt + errFlag;
end

% F08-PDSCH
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
SysPar.SimCtrl.N_slot_run = slotIdx + 1;
SysPar.SimCtrl.genTV.slotIdx = slotIdx;
for idxCase = F08_DS_caseSet
    SysPar.pdsch{1}.mcsIndex = 27-idxCase+1;
    fileName = ['GPU_test_input/TVnr_F08_DS_', num2str(idxCase, '%02d'),'.yaml'];
    WriteYaml(fileName, SysPar);
    cfgFile = ['GPU_test_input/TVnr_F08_DS_', num2str(idxCase, '%02d'),'.yaml'];
    TVfile = ['TVnr_F08_DS_', num2str(idxCase, '%02d')];
    errFlag = runSim(cfgFile, TVfile);    
    errCnt = errCnt + errFlag;
end

% F08-PUSCH
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
SysPar.SimCtrl.N_slot_run = slotIdx + 1;
SysPar.SimCtrl.genTV.slotIdx = slotIdx;
for idxCase = F08_US_caseSet
    SysPar.pusch{1}.mcsIndex = 27-idxCase+1;
    fileName = ['GPU_test_input/TVnr_F08_US_', num2str(idxCase, '%02d'),'.yaml'];
    WriteYaml(fileName, SysPar);
    cfgFile = ['GPU_test_input/TVnr_F08_US_', num2str(idxCase, '%02d'),'.yaml'];
    TVfile = ['TVnr_F08_US_', num2str(idxCase, '%02d')];
    errFlag = runSim(cfgFile, TVfile);    
    errCnt = errCnt + errFlag;
end

% F14-PDSCH
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
end
SysPar.SimCtrl.N_UE = 4;
SysPar.SimCtrl.N_slot_run = slotIdx + 1;
SysPar.SimCtrl.genTV.slotIdx = slotIdx;
for idxCase = F14_DS_caseSet
    for idxUE = 1:4
        SysPar.pdsch{idxUE}.mcsIndex = 27-idxCase+1;
    end
    fileName = ['GPU_test_input/TVnr_F14_DS_', num2str(idxCase, '%02d'),'.yaml'];
    WriteYaml(fileName, SysPar);
    cfgFile = ['GPU_test_input/TVnr_F14_DS_', num2str(idxCase, '%02d'),'.yaml'];
    TVfile = ['TVnr_F14_DS_', num2str(idxCase, '%02d')];
    errFlag = runSim(cfgFile, TVfile);    
    errCnt = errCnt + errFlag;
end

% F14-PUSCH
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
    SysPar.pusch{idxUE}.StartSymbolIndex = 2;
    SysPar.pusch{idxUE}.NrOfSymbols = 12;
    SysPar.pusch{idxUE}.DmrsSymbPos = [0 0 1 1 0 0 0 0 0 0 0 0 0 0];
    SysPar.pusch{idxUE}.nrOfLayers = 4;
    SysPar.pusch{idxUE}.mcsTable = 1;
    SysPar.pusch{idxUE}.portIdx = (idxUE-1)*4+[0:3];    
    SysPar.pusch{idxUE}.idxUE = idxUE-1;
end
% SysPar.Chan{1}.type = 'TDLA30-10-Low';
% SysPar.SimCtrl.timeDomainSim = 1;
SysPar.SimCtrl.N_UE = 2;
SysPar.SimCtrl.N_slot_run = slotIdx + 1;
SysPar.SimCtrl.genTV.slotIdx = slotIdx;
for idxCase = F14_US_caseSet
    for idxUE = 1:2
        SysPar.pusch{idxUE}.mcsIndex = 27-idxCase+1;
    end
    fileName = ['GPU_test_input/TVnr_F14_US_', num2str(idxCase, '%02d'),'.yaml'];
    WriteYaml(fileName, SysPar);
    cfgFile = ['GPU_test_input/TVnr_F14_US_', num2str(idxCase, '%02d'),'.yaml'];
    TVfile = ['TVnr_F14_US_', num2str(idxCase, '%02d')];
    errFlag = runSim(cfgFile, TVfile);
    errCnt = errCnt + errFlag;
end

return
