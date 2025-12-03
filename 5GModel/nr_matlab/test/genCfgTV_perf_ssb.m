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

function [errCnt, nTV] = genCfgTV_perf_ssb(caseSet)

if nargin == 0
    caseSet = ["F08", "F09", "F14", "V15"]; % all cases
end


for caseIdx = 1 : length(caseSet)
    switch caseSet(caseIdx)
        case "F08"
            [~, errCnt, nTV] = testCompGenTV_ssb(1801);  
            copyfile('./GPU_test_input/TVnr_1801_SSB_gNB_CUPHY_s0p0.h5','./GPU_test_input/TV_cuphy_F08-SS-01.h5');
            
            
        case "F09"
            [~, errCnt, nTV] = testCompGenTV_ssb(1801);  
            copyfile('./GPU_test_input/TVnr_1801_SSB_gNB_CUPHY_s0p0.h5','./GPU_test_input/TV_cuphy_F09-SS-01.h5');  
            
        case "F14"
            [~, errCnt, nTV] = testCompGenTV_ssb(1801);  % 1 SSB block
            copyfile('./GPU_test_input/TVnr_1801_SSB_gNB_CUPHY_s0p0.h5','./GPU_test_input/TV_cuphy_F14-SS-01.h5');
            
            [~, errCnt, nTV] = testCompGenTV_ssb(1803);  % 2 SSB blocks
            copyfile('./GPU_test_input/TVnr_1803_SSB_gNB_CUPHY_s0p1.h5','./GPU_test_input/TV_cuphy_F14-SS-02.h5');
        
       case "V15"
            [~, errCnt, nTV] = testCompGenTV_ssb(1802);  
            copyfile('./GPU_test_input/TVnr_1802_SSB_gNB_CUPHY_s0p0.h5','./GPU_test_input/TV_cuphy_V15-SS-01.h5');
           

    end
end

end