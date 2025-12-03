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

function csirs = gen_csirs_config(csirs_cfg)

[nCfg, ~] = size(csirs_cfg);
csirs = [];

for idxCfg = 1:nCfg
    thisCsirs = cfgCsirs;
    thisCsirs.Row = csirs_cfg{idxCfg, 1};
    thisCsirs.CDMType = csirs_cfg{idxCfg, 2};
    thisCsirs.FreqDensity = csirs_cfg{idxCfg, 3};
    thisCsirs.StartRB = csirs_cfg{idxCfg, 4};
    thisCsirs.NrOfRBs = csirs_cfg{idxCfg, 5};
    thisCsirs.SymbL0 = csirs_cfg{idxCfg, 6};
    thisCsirs.SymbL1 = csirs_cfg{idxCfg, 7};
    thisCsirs.FreqDomain = cell2mat(csirs_cfg{idxCfg, 8});
    csirs{idxCfg} = thisCsirs;
end

return
