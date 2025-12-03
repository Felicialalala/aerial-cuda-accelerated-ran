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

function LLR = max_pam_LLR(y,N0,d,m)
 


%%
%SOFT BITS

y_soft = zeros(m,1);

y_soft(1) = y;

for i = 1 : (m - 1)
    y_soft(i + 1) = d * 2^(m - i) - abs(y_soft(i));
end


%%
%MIN DIST

min1_dist = zeros(m,1);
min2_dist = (abs(y_soft(end)) - d)^2;

for i = 1 : m
    min1_dist(i) = (d + abs(y_soft(i)))^2;
end

%%
%LLR

LLR = zeros(m,1);

for i = 1 : m
    
    if y_soft(i) >= 0
        LLR(i) = (min1_dist(i) - min2_dist) / N0;
    else
        LLR(i) = (min2_dist - min1_dist(i)) / N0;
    end
    
end






    
