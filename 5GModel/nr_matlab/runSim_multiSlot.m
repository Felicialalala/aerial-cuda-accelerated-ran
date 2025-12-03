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

%
% This is the main program to run nrSimulator for algorithm design and
% new feature development. 
%
%

function [errFlag, SysParList, UE, gNB] = runSim_multiSlot(cfgListName, tvFileName)

% rng(0); move downwards so that we can read seed from cfgFie

testAlloc.dl = 0;
testAlloc.ul = 0;
testAlloc.ssb = 0;
testAlloc.pdcch = 0;
testAlloc.pdsch = 0;
testAlloc.csirs = 0;
testAlloc.prach = 0;
testAlloc.pucch = 0;
testAlloc.pusch = 0;
testAlloc.srs = 0;
cfgList = ReadYaml(cfgListName, 0, 0);
nSlot = length(cfgList.SCHED);
for idxSlot = 1:nSlot
    SysParList{idxSlot} = [];
    nChan = length(cfgList.SCHED{idxSlot}.config{1}.channels);    
    for idxChan = 1:nChan
        cfgFileName = cfgList.SCHED{idxSlot}.config{1}.channels{idxChan};
        SysPar_yaml = ReadYaml(cfgFileName, 0, 0);
        SysPar = initSysPar(testAlloc);
        SysPar_yaml = ReadYaml(cfgFileName, 0, 0);
        SysPar = updateStruct(SysPar, SysPar_yaml);
        if isdeployed
            if isempty(fieldnames(SysPar_yaml)) % in deployed mode with batchsim launch, SysPar_yaml should not be empty!
                error('Parsing cfg yaml file failed in deployed mode!')      % WriteYaml(fullfile(fileparts(cfgFileName),'cfg_template_sysparyaml.yaml'), SysPar_yaml);
            end
            if isfile(fullfile(ctfroot,'/runSim/cfg_template.yaml'))
                error(sprintf('%s should not be here! Please exclude it from your compilation!\n', fullfile(ctfroot,'/runSim/cfg_template.yaml')))
            end
        end
        fprintf('\nRead config from %s\n', cfgFileName);
        if nargin > 1
            SysPar.SimCtrl.genTV.enable = 1;
            SysPar.SimCtrl.genTV.TVname = tvFileName;
            SysPar.SimCtrl.genTV.forceSlotIdxFlag = 1;
        end
        if nargin == 3
            SysPar.SimCtrl.capSamp.enable = 1;
            SysPar.SimCtrl.capSamp.fileName = rxFileName;
        end
%         SysPar = updateAlgFlag(SysPar);
        if SysPar.testAlloc.dl
            SysParList{idxSlot}.SysParDl = SysPar;
        elseif SysPar.testAlloc.ul
            SysParList{idxSlot}.SysParUl = SysPar;
        end        
    end
end

[SysParList, UE, gNB] = nrSimulator(SysParList);

errFlag = 0;

return

function printUsage

fprintf('Usage: Run a single test case with user defined test configuration through input yaml file.\n');  
fprintf('1. Run genCfgTemplate to generate a configuration template file ''cfg_template.yaml''\n'); 
fprintf('2. Edit the template file with specific configuration and save it to another yaml file name\n');
fprintf('3. Run runSim(cfgFileName, tvFileName). For example, runSim(''pdsch_f14_cfg.yaml'', ''pdsch_f14'')\n');
fprintf('runSim() will run with the default configuration.\n');

return
