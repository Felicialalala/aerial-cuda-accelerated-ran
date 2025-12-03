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

% function outSamples = fh_bfp_compress(inSamples, iqWidth, bypass) 

% Compresses data compressed using block floating point algorithm (O-RAN
% WG4.CUS.0-v02.00, Section A.1.1
% inputs:
%   inSamples, uncompressed I+Q data arranged in PRBs (uni-dimensional
%   array)
%   iqWidth, bit width of compressed data (9 bit and 14 bit currently
%   supported)
%   bypass, flag to bypass compression (when set to 1)
% outputs:
%   outSamples, compressed I+Q data arranged in PRBs (uni-dimensional
%   array)


n=1;
inSamples = rand(1,12*n) + 1i*rand(1,12*n);


% Compress n PRB of data
if bypass
    outSamples = inSamples;
    break;
end

% Calculate exponent

% Calculate scaler

% Scale and quantize

% Calculate MSE
mse = mean(abs(inSamples - compSamples).^2);

