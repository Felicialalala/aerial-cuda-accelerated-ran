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

function [errCnt, nTV] = genCfgTV_perf_pucch(caseSet)

if nargin == 0
    caseSet = ["F08", "F09", "F14", "V15"]; % all cases
end


for caseIdx = 1 : length(caseSet)
    switch caseSet(caseIdx)
        case "F08"
            [~, errCnt, nTV] = testCompGenTV_pucch(6801);  % F08, full cell
            copyfile('./GPU_test_input/TVnr_6801_PUCCH_F3_gNB_CUPHY_s0p16.h5','./GPU_test_input/TV_cuphy_F08-UC-01_PRB273.h5');
                
            [~, errCnt, nTV] = testCompGenTV_pucch(6802);  % F08, ave cell
            copyfile('./GPU_test_input/TVnr_6802_PUCCH_F3_gNB_CUPHY_s0p5.h5','./GPU_test_input/TV_cuphy_F08-UC-02_PRB273.h5');
            
            [~, errCnt, nTV] = testCompGenTV_pucch(6805);  % F08, reduced complexity
            copyfile('./GPU_test_input/TVnr_6805_PUCCH_F1_gNB_CUPHY_s0p24.h5','./GPU_test_input/TV_cuphy_F08-UC-RC_PRB273.h5');
        
        case "F09"
            [~, errCnt, nTV] = testCompGenTV_pucch(6807);  % PF1 32 UEs
            copyfile('./GPU_test_input/TVnr_6807_PUCCH_F1_gNB_CUPHY_s0p32.h5','./GPU_test_input/TV_cuphy_F09-UC-40_PRB273.h5');
            
        case "V15"
            
            [~, errCnt, nTV] = testCompGenTV_pucch(6806);  % FDD 4T4R
            copyfile('./GPU_test_input/TVnr_6806_PUCCH_F1_gNB_CUPHY_s0p6.h5','./GPU_test_input/TV_cuphy_V15-UC-01_PRB106.h5');
            
        case "F14"
            [~, errCnt, nTV] = testCompGenTV_pucch(6803);  % F14, full cell
            copyfile('./GPU_test_input/TVnr_6803_PUCCH_F3_gNB_CUPHY_s0p36.h5','./GPU_test_input/TV_cuphy_F14-UC-01_PRB273.h5');
            
            [~, errCnt, nTV] = testCompGenTV_pucch(6804);  % F14, ave cell
            copyfile('./GPU_test_input/TVnr_6804_PUCCH_F3_gNB_CUPHY_s0p10.h5','./GPU_test_input/TV_cuphy_F14-UC-02_PRB273.h5');
            
            [~, errCnt, nTV] = testCompGenTV_pucch(6807);  % F14, full cell
            copyfile('./GPU_test_input/TVnr_6807_PUCCH_F1_gNB_CUPHY_s0p32.h5','./GPU_test_input/TV_cuphy_F14-UC-40_PRB273.h5');
    end
end

end