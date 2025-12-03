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

function genConformanceConfig

UlConfig =  {'G-FR1-A1-5',  7993;
             'G-FR1-A2-5',  7994;
             'G-FR1-A3-14', 7004;
             'G-FR1-A3-28', 7012;
             'G-FR1-A3-32', 7056;
             'G-FR1-A4-14', 7005;
             'G-FR1-A4-28', 7013;
             'G-FR1-A5-14', 7006;
             'G-FR1-A4-11', 7053};

[nUlConfig, ~] = size(UlConfig);
for idx = 1:nUlConfig
    testPerformance_pusch(UlConfig{idx, 2}, 0, 1, 99);
    cfgFileName = ['cfg-', num2str(UlConfig{idx, 2}), '.yaml'];
    SysPar = initSysPar; 
    SysPar_yaml = ReadYaml(cfgFileName, 0, 0);
    SysPar = updateStruct(SysPar, SysPar_yaml);
    SysPar.SimCtrl.genTV.enable = 1;
    SysPar.SimCtrl.genTV.FAPIyaml = 1;
    SysPar.SimCtrl.N_slot_run = 1;
    SysPar.SimCtrl.timeDomainSim = 0;
    SysPar.Chan{1} = cfgChan;
    WriteYaml(cfgFileName, SysPar);
    tvFileName = ['TVnr-', num2str(UlConfig{idx, 2}), '-', UlConfig{idx, 1}];
    runSim(cfgFileName, tvFileName);
end

DlConfig = {'NR-FR1-TM1.1',  120;
            'NR-FR1-TM1.2',  121;
            'NR-RF1-TM2',    122;
            'NR-FR1-TM2a',   123;
            'NR-FR1-TM3.1',  124;
            'NR-FR1-TM3.1a', 125;
            'NR-FR1-TM3.2',  126;
            'NR-FR1-TM3.3',  127};
[nDlConfig, ~] = size(DlConfig);          
for idx = 1:nDlConfig
    testCompGenTV_dlmix(DlConfig{idx, 2}, 'genCfg');
    cfgFileName = sprintf('cfg-%04d.yaml', DlConfig{idx, 2});
    SysPar = initSysPar; 
    SysPar_yaml = ReadYaml(cfgFileName, 0, 0);
    SysPar = updateStruct(SysPar, SysPar_yaml);
    SysPar.SimCtrl.genTV.enable = 1;
    SysPar.SimCtrl.genTV.FAPIyaml = 1;
    SysPar.SimCtrl.N_slot_run = 1;
    SysPar.SimCtrl.timeDomainSim = 0;
    SysPar.Chan{1} = cfgChan;
    WriteYaml(cfgFileName, SysPar);
    tvFileName = sprintf('TVnr-%04d-%s', DlConfig{idx, 2}, DlConfig{idx, 1});
    runSim(cfgFileName, tvFileName);
end        
        
return
    