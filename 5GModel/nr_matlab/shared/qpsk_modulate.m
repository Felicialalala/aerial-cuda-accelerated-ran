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

function d = qpsk_modulate(b,E)

% function modulates an array of bits into an array of qpsk symbols

%inputs:
% b --> bit array. Dim: E x 1
% E --> number of bits

% outputs:
% d --> qpsk symbols. Dim: E/2 x 1

%%
%START

M_sym = E / 2;
d = zeros(M_sym,1);

for i = 1 : M_sym
    bIdx = 2*(i-1);
    d(i) = 1/sqrt(2)*((1-2*b(bIdx+1)) + 1i*(1-2*b(bIdx+2)));
end


end
    
