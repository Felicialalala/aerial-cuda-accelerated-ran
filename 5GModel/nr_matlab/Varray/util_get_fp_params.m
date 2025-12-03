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

function y = util_get_fp_params(fp_flag)
    switch fp_flag
        case 0 % FP64
            E  = 11; % Num of Exponent bits
            M  = 52; % Num of Mantissa bits
        case 1 % FP32
            E  = 8;
            M  = 23;
        case 2 % TF32
            E  = 8;
            M  = 10;
        case 3 % FP16
            E  = 5;
            M  = 10; 
        case 4 % BF16
            E  = 8;
            M  = 7;
        case 5 % FP8 E4M3
            E  = 4;
            M  = 3;
        case 6 % FP8 E5M2
            E  = 5;
            M  = 2;
    end
    Emax                = 2^(E-1)-1;            % Exponent of the largest normal number.   
    Emin                = 1-Emax;               % Exponent of the smallest normal number.
    Emin_subnormal      = Emin - M;             % Exponent of the smallest subnormal number.
    y.smallest_subnormal  = 2^Emin_subnormal;
    y.largest_subnormal   = 2^(Emin)*(1-2^(-M));
    y.smallest_normal     = 2^Emin;
    y.largest_normal      = 2^Emax * (2-2^(-M));
    y.E = E;
    y.M = M;
end