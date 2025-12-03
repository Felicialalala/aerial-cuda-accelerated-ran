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

function [errCnt, nTV] = genCfgTV_perf_csirs(caseSet)

if nargin == 0
    caseSet = ["F08", "F09", "F14"]; % all cases
end


for caseIdx = 1 : length(caseSet)
    switch caseSet(caseIdx)
        case "F08"
            [~, errCnt, nTV] = testCompGenTV_csirs(4801);  
            copyfile('./GPU_test_input/TVnr_4801_CSIRS_gNB_CUPHY_s0p0.h5','./GPU_test_input/TV_cuphy_F08-CR-01.h5');

            [~, errCnt, nTV] = testCompGenTV_csirs(4806);  
            copyfile('./GPU_test_input/TVnr_4806_CSIRS_gNB_CUPHY_s0p0.h5','./GPU_test_input/TV_cuphy_F08-CR-02.h5');
            
        case "V15"
            [~, errCnt, nTV] = testCompGenTV_csirs(4804);  
            copyfile('./GPU_test_input/TVnr_4804_CSIRS_gNB_CUPHY_s0p0.h5','./GPU_test_input/TV_cuphy_V15-CR-01.h5');
            
            
        case "F09"
            [~, errCnt, nTV] = testCompGenTV_csirs(4802);  
            copyfile('./GPU_test_input/TVnr_4802_CSIRS_gNB_CUPHY_s0p0.h5','./GPU_test_input/TV_cuphy_F09-CR-01.h5');

            [~, errCnt, nTV] = testCompGenTV_csirs(4807); % 20 MHz, 16 ports
            copyfile('./GPU_test_input/TVnr_4807_CSIRS_gNB_CUPHY_s0p0.h5','./GPU_test_input/TV_cuphy_F09-CR-02_PRB48.h5');
            
        case "F14"
            [~, errCnt, nTV] = testCompGenTV_csirs(4803);  
            copyfile('./GPU_test_input/TVnr_4803_CSIRS_gNB_CUPHY_s0p0.h5','./GPU_test_input/TV_cuphy_F14-CR-01.h5'); 
            
            [~, errCnt, nTV] = testCompGenTV_csirs(4805);  
            copyfile('./GPU_test_input/TVnr_4805_CSIRS_gNB_CUPHY_s0p0.h5','./GPU_test_input/TV_cuphy_F14-CR-42.h5');

    end
end

end