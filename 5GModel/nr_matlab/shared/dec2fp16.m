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

function y = dec2fp16(x)

   y = -9999999*ones(size(x));

   bit_15 = 2^15;
   bits_14_10 = 2^14 + 2^13 + 2^12 + 2^11 + 2^10;
   bit_10 = 2^10;
   bits_9_0 = 2^10 - 1;
   sign_bits = bitand(x,bit_15) / bit_15;
   exp_bits = bitand(x,bits_14_10) / bit_10 - 15;
   mant_bits = bitand(x,bits_9_0);

   % Normal numbers
   x_normals = bitor(exp_bits ~= -15, exp_bits ~= 16);
   y(x_normals) = (-1).^sign_bits(x_normals) .* (1 + mant_bits(x_normals)/1024) .* 2.^exp_bits(x_normals);

   % Subnormal numbers
   x_subnormals = (exp_bits == -15);
   y(x_subnormals) = (-1).^sign_bits(x_subnormals) .* (mant_bits(x_subnormals)/1024) .* 2.^(-14);

   % Infinity numbers
   x_infinities = bitand(exp_bits == 16, mant_bits == 0);
   y(x_infinities) = (-1).^sign_bits(x_infinities) .* inf;

   % NaN numbers
   x_nans = bitand(exp_bits == 16, mant_bits ~= 0);
   y(x_nans) = nan;
