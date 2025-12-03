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

function [errCnt, nTV] = genCfgTV_perf_srs(caseSet)
 
if nargin == 0
    caseSet = ["F09", "F14"]; % all cases
end
 
 
for caseIdx = 1 : length(caseSet)
    switch caseSet(caseIdx)
        case "F09"
            % generate TVs based on standalone version
            % TODO: remove when cuphy phase 3 updated 
            get_SRS_TV(32, 272);
            copyfile('./GPU_test_input/GPU_TV_SRS_CH_EST_MIMO32x32_PRB272_SRS_SYM2_FP16.h5','./GPU_test_input/TV_cuphy_F09-SR-01_snrdb40.00_MIMO4x8_PRB272.h5');
            
            % generate TVs based on complete version
            [~, errCnt, nTV] = testCompGenTV_srs(8801);
            file = dir('./GPU_test_input/*TVnr_8801_SRS_gNB_CUPHY_*.h5');
            copyfile(['./GPU_test_input/', file.name],'./GPU_test_input/srs2_TV_cuphy_F09-SR-01_snrdb40.00_MIMO4x8_PRB272.h5');

            [~, errCnt, nTV] = testCompGenTV_srs(8802); % 20 MHz
            file = dir('./GPU_test_input/*TVnr_8802_SRS_gNB_CUPHY_*.h5');
            copyfile(['./GPU_test_input/', file.name],'./GPU_test_input/srs2_TV_cuphy_F09-SR-02_snrdb40.00_MIMO8x16_PRB48.h5');
            
        case "F14"
            % generate TVs based on standalone version
            % TODO: remove when cuphy phase 3 updated 
            get_SRS_TV(64, 272);
            copyfile('./GPU_test_input/GPU_TV_SRS_CH_EST_MIMO32x64_PRB272_SRS_SYM2_FP16.h5','./GPU_test_input/TV_cuphy_F14-SR-01_snrdb40.00_MIMO8x16_PRB272.h5');
            
            % generate TVs based on complete version
            [~, errCnt, nTV] = testCompGenTV_srs(8801);
            file = dir('./GPU_test_input/*TVnr_8801_SRS_gNB_CUPHY_*.h5');
            copyfile(['./GPU_test_input/', file.name],'./GPU_test_input/srs2_TV_cuphy_F14-SR-01_snrdb40.00_MIMO8x16_PRB272.h5');
            
            [~, errCnt, nTV] = testCompGenTV_srs(8802);
            file = dir('./GPU_test_input/*TVnr_8802_SRS_gNB_CUPHY_*.h5');
            copyfile(['./GPU_test_input/', file.name],'./GPU_test_input/srs2_TV_cuphy_F14-SR-42_snrdb40.00_MIMO8x16_PRB48.h5');
    end
end
 
end
