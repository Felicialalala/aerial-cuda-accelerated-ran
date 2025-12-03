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

function [d_u, q] = findDu(u, L_RA)

q = L_RA;
for qq = 0:L_RA-1 % Search range ???
    if mod(qq*u, L_RA) == 1
        q = qq;        
        break;
    end
end
if q == L_RA
    error('Can not find q < L_RA ... \n');
end


if q < L_RA/2
    d_u = q;
else
    d_u = L_RA - q;
end
