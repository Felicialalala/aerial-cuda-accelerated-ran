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

wmArray_n5 = buildPolarWeightMetrics(5);
wmArray_n6 = buildPolarWeightMetrics(6);
wmArray_n7 = buildPolarWeightMetrics(7);
wmArray_n8 = buildPolarWeightMetrics(8);
wmArray_n9 = buildPolarWeightMetrics(9);
wmArray_n10 = buildPolarWeightMetrics(10);

save('wmArrays.mat','wmArray_n5','wmArray_n6','wmArray_n7','wmArray_n8','wmArray_n9','wmArray_n10');



function wmArray = buildPolarWeightMetrics(n)

%%
%PARAMATERS

N = 2^n;
polarMtx = zeros(N);

%%
%START

for bitIdx = 0 : (N-1)
    d             = zeros(N,1);
    d(bitIdx + 1) = 1;


    for i = 0 : (n - 1) %loop over log2(N) - 1 stages 

        s = 2^i;
        m = N / (2*s);

        %parallel start (N/2 parallel XORS)
        for j = 1 : m
            start_idx = 2*s*(j-1);

            for k = 1 : s
                d(start_idx + k) = xor(d(start_idx + k), d(start_idx + k + s) );
            end
        end
    end
    
    polarMtx(:,bitIdx + 1) = d;
end

wmArray = sum(polarMtx',2);

end
