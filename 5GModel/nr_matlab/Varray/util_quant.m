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

function out = util_quant(in, fp)
    if nargin == 1
        fp = 0;
    end
    if fp==0 %strcmp(fp, 'fp64') % ieee754
        out  = double(in);
    elseif fp==1 %strcmp(fp, 'fp32') % ieee754
        out  = single(in);
    elseif fp==2 %strcmp(fp, 'fp16') % ieee754
        out  = half(in);
    elseif fp==3 %strcmp(fp, 'fp16cleve') 
        if isreal(in)
            out  = vfp16(in);
        else
            real_part = vfp16(real(in));
            imag_part = vfp16(imag(in));
            out  = real_part + 1i*imag_part;
        end                             
    elseif fp==4 %strcmp(fp, 'fp16nv')
        size_array = size(in);
        if isreal(in)
            x = in(:);
        else
            tmp = in(:);
            x = [real(tmp); imag(tmp)];
        end
        len_x = length(x);
        
        smallest_subnormal =  2^(-14-10);
        largest_subnormal = 2^(-14)*(1-2^(-10));
%                 smallest_normal = 2^(-14);
        largest_normal = (2-2^(-10))*2^15;

        y = x;
        posInfIdx = (x > largest_normal);
        y(posInfIdx) = Inf;

        negInfIdx = (x < -largest_normal);
        y(negInfIdx) = -Inf;

        zeroIdx = (((x > -smallest_subnormal) + (x < smallest_subnormal)) > 1);
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

        if isreal(in)
            out = reshape(y, size_array);
        else
            out = reshape(y(1:len_x/2)+1i*y(len_x/2+1:end), size_array);
        end
    end
    out = double(out);
end