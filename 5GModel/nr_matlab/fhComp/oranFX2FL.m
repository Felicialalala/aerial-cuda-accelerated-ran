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

function X_tf_fl = oranFX2FL(X_tf_fx, iqWidth, beta, X_tf_size)

X_tf_length = prod(X_tf_size);

Xtf6 = X_tf_fx;
Xtf7 = dec2bin(Xtf6, 8)';
Xtf8 = Xtf7(:);
Xtf9 = reshape(Xtf8, [iqWidth, 2*X_tf_length])';
Xtf_real = bin2dec(Xtf9(1:2:end, :));
Xtf_imag = bin2dec(Xtf9(2:2:end, :));

Xtf_real = Xtf_real - 2^iqWidth * (Xtf_real > 2^(iqWidth-1)-1);
Xtf_imag = Xtf_imag - 2^iqWidth * (Xtf_imag > 2^(iqWidth-1)-1);
Xtf_real = Xtf_real/beta;
Xtf_imag = Xtf_imag/beta;

X_tf_fl = Xtf_real + j*Xtf_imag;
X_tf_fl = reshape(X_tf_fl, X_tf_size);

return