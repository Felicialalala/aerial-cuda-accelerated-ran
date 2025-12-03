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

function [y_uv, x_uv, x_u] = genZcPreamble(L_RA, C_v, u)
%
% generate ZC sequence and preamble in freq domain 
%
genZcAlg = 1;

[~, q] = findDu(u, L_RA);

i = [0:L_RA-1];
x_u = exp(-1j*pi*u*i.*(i+1)/L_RA);
n = [0:L_RA-1];
x_uv = x_u(mod((n+C_v), L_RA)+1);       

switch genZcAlg 
    case 0 % normal method
        m = [0:L_RA-1];
        for nn = 0:L_RA-1
            y_uv(nn+1) = sum(x_uv(m+1).*exp(-1j*2*pi*m*nn/L_RA));
        end
        y_uv = y_uv/sqrt(L_RA);
    case 1
        % Simplied method
        % Use the closed-form expressions in https://goo.gl/rjzvJn 
        m = [0:L_RA-1];
        y_uv = conj(x_u(mod(q*m+C_v,L_RA)+1));
        y_uv = sum(x_u)*x_u(mod(C_v,L_RA)+1)*y_uv/sqrt(L_RA);          
    otherwise
        error('genZcAlg is not supperted ...\n');
end
