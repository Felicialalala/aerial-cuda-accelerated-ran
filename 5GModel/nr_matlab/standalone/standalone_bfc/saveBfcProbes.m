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

function saveBfcProbes(bfc,tvH,tvA0,tvA1,tvDinv,tvCoef,tvFrobNorm)

tvDirName = [pwd filesep 'GPU_test_input'];
tvNameSuffix = '';
if exist('fp16Data','var') && fp16Data
    tvNameSuffix = '_fp16'; 
end

tvName = sprintf('BfcCoef_MIMO%dx%d_NumCoef%d.mat', bfc.nLayers, bfc.L_gNB, bfc.nWeights);
tvDirName = 'GPU_test_input'; [status,msg] = mkdir(tvDirName);
save([tvDirName filesep tvName], 'bfc', 'tvH', 'tvCoef', 'tvA0', 'tvA1', 'tvDinv', 'tvFrobNorm');

end
