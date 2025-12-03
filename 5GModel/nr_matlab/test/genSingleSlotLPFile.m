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

function errFlag = genSingleSlotLPFile(LPFileName, caseNum, slotIdx)

nCell = 1;
nSlot = 20;
LP = [];
TVname = sprintf('TVnr_%04d_gNB_FAPI_s%d.h5', caseNum, slotIdx);
LP.Cell_Configs = {TVname};
LP = init_launchPattern(LP, nSlot, nCell);
LP.SCHED{1}.config{1}.channels = {TVname};


tvDirName = 'GPU_test_input';
[status,msg] = mkdir(tvDirName);

TVname = sprintf('%s_9%04d', LPFileName, caseNum);
yamlFileName = [tvDirName filesep TVname '.yaml'];

WriteYaml(yamlFileName, LP);

errFlag = 0;

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
