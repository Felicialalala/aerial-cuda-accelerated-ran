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

function y = fp16nv(x, fp16AlgoSel, returnDouble)
% fp16nv Convert input to FP16 precision
%   y = fp16nv(x) converts input x to FP16 precision using default algorithm (2)
%   y = fp16nv(x, fp16AlgoSel) uses specified algorithm selection
%   y = fp16nv(x, fp16AlgoSel, returnDouble) specifies return type:
%      - returnDouble = false (default): returns single precision
%      - returnDouble = true: returns double precision
%
% Inputs:
%   x - Input data
%   fp16AlgoSel - Algorithm selection (default: 2)
%   returnDouble - Return type flag (default: false)
%
% Output:
%   y - FP16 converted data in either single or double precision

% Set defaults for optional arguments
if nargin < 2
    fp16AlgoSel = 2;
end
if nargin < 3
    returnDouble = false;
end

switch fp16AlgoSel
    case 0 % Mathworks fixed point toolbox
        y = half(double(x));
    case 1 % Cleve fp16
        y = vfp16(double(x));
    case 2 % Nvidia fp16
        tensorDims = size(x);
        x = x(:);
               
        smallest_subnormal =  2^(-14-10);
        largest_subnormal = 2^(-14)*(1-2^(-10));
        smallest_normal = 2^(-14);
        largest_normal = (2-2^(-10))*2^15;
        
        y = x;
        posInfIdx = find(x > largest_normal);
        y(posInfIdx) = Inf;
        
        negInfIdx = find(x < -largest_normal);
        y(negInfIdx) = -Inf;
        
        zeroIdx = find(((x > -smallest_subnormal) + (x < smallest_subnormal)) > 1);
        y(zeroIdx) = 0;
        
        posSubnormalIdx = find(((x >= smallest_subnormal) + (x <= largest_subnormal)) > 1);
        y(posSubnormalIdx) = round(x(posSubnormalIdx)*2^(14+10))/2^(14+10);
        
        negSubnormalIdx = find(((x <= -smallest_subnormal) + (x >= -largest_subnormal)) > 1);
        stepSize = 2^(-14-10);
        y(negSubnormalIdx) = round(x(negSubnormalIdx)/stepSize)*stepSize;
        
        posNormalIdx = find(((x <= largest_normal) + (x > largest_subnormal)) > 1);
        x_posNormal = x(posNormalIdx);
        stepSize = 2.^(floor(log2(x_posNormal))-10);
        y(posNormalIdx) = round(x_posNormal./stepSize).*stepSize;
        
        negNormalIdx = find(((x >= -largest_normal) + (x < -largest_subnormal)) > 1);
        x_negNormal = x(negNormalIdx);
        stepSize = 2.^(floor(log2(-x_negNormal))-10);
        y(negNormalIdx) = round(x_negNormal./stepSize).*stepSize;
        
        y = reshape(y, tensorDims);
        
    otherwise
        error('fp16AlgoSel is not supported ...');
end

% Convert output type based on returnDouble flag
if returnDouble
    y = double(y);
else
    y = single(y);
end

return