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

function errCnt = run5GModelCicd(cicdMode, relNum)

if nargin == 0
    cicdMode = 0;
    relNum = 10000;
    printUsage;
elseif nargin == 1
    relNum = 10000;
    printUsage;    
elseif ~ismember(cicdMode, [0, 1, 2])
    printUsage;
    error('cicdMode is not supported ...\n');
end

% Generate TVs based on cicdMode

mrSubSetMod = [0, 1]; % [mod(datenum(date), 5), 5];
nightlySubSetMod = [0, 1];

switch cicdMode
    case 0 % MR mode
        
        % Compact regression test for 5GModel
        [nTC,errCnt] = runRegression({'allTests'}, {'allChannels'}, 'compact', nightlySubSetMod, relNum);
        
        % delete TVs not supported by cuPHY.
        delete GPU_test_input/disabled_*;
    
    case 1 % nightly mode
        
        % Full regression test for 5GModel
        [nTC,errCnt] = runRegression({'allTests'}, {'allChannels'}, 'full', nightlySubSetMod, relNum);
        
        % delete TVs not supported by cuPHY.
        delete GPU_test_input/disabled_*;
        
    case 2 % compact TV only
        
        % Compact TV generation
        [nTC,errCnt] = runRegression({'TestVector'}, {'allChannels'}, 'compact', nightlySubSetMod, relNum);
        
        % delete TVs not supported by cuPHY.
        delete GPU_test_input/disabled_*;

    otherwise
        error('cicdMode is not supported ... \n');
end

return


function printUsage

fprintf('Usage: errCnt = run5GModelCicd(cicdMode, relNum) \n'); 
fprintf('cicdMode: 0 (MR pipeline), 1 (nightly pipeline), 2 (compact TV) \n');
fprintf('relNum: 10000 (latest), 2240 (Rel-22-4) \n');
fprintf('run5GModelCicd() = run5GModelCicd(0, 1000)\n');

return
