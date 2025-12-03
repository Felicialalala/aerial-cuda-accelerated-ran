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

function verifyBfcResultsLite(tv,gpuResult)

% Example invocation
if 0
  tv.dir = [pwd filesep 'GPU_test_input']; tv.fname = 'GPU_TV_BFC_COEF_MIMO16x64_N_COEF137.h5';
  gpuResult.dir = [pwd filesep 'GPU_results']; gpuResult.fname = 'gpu_out_GPU_TV_BFC_COEF_MIMO16x64_N_COEF137.h5';
  verifyBfcResultsLite(tv,gpuResult);
end

tvCoef = getfield(hdf5_load_nv([tv.dir filesep tv.fname]), 'Coef');
gpuCoef = getfield(hdf5_load_nv([gpuResult.dir filesep gpuResult.fname]), 'Coef');

enablePlots = 0;
compare(tvCoef, gpuCoef, 'BFC Coefficients', enablePlots);

end
