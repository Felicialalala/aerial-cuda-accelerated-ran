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

function X_tf_fx = oranFL2FX(X_tf, iqWidth, beta)

Xtf = round(X_tf * beta);

limit = 2^(iqWidth-1);

real_Xtf = min(real(Xtf), limit-1);
imag_Xtf = min(imag(Xtf), limit-1);

real_Xtf = max(real_Xtf, -limit);
imag_Xtf = max(imag_Xtf, -limit);

real_Xtf = real_Xtf + 2^iqWidth * (real_Xtf < 0);
imag_Xtf = imag_Xtf + 2^iqWidth * (imag_Xtf < 0);

Xtf1 = [real_Xtf(:)'; imag_Xtf(:)'];
Xtf2 = Xtf1(:);
Xtf3 = dec2bin(Xtf2, iqWidth)';
Xtf4 = Xtf3(:)';
Xtf5 = reshape(Xtf4, 8, length(Xtf4)/8);
Xtf6 = bin2dec(Xtf5');

X_tf_fx = Xtf6;

return