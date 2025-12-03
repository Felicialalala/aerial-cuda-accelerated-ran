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

function rb0 = findPdcchSS(coresetIdx, slot_number, rnti, N_CCE, aggrL, M_candidate, isCSS)

p = coresetIdx;
n_s_f_u = slot_number;
n_CI = 0;
if p > 0 && ~isCSS 
    Y_p_n = rnti;
else
    Y_p_n = 0;
end
D = 65537;
if mod(p, 3) == 0
    A_p = 39827;
elseif mod(p, 3) == 1
    A_p = 39829;
elseif mod(p, 3) == 2
    A_p = 39839;
end       

for n = 0:n_s_f_u
    Y_p_n = mod(A_p*Y_p_n, D);
end

for idxCandidate = 1:M_candidate
    rb0(idxCandidate) = aggrL*mod((Y_p_n + floor((idxCandidate-1)*N_CCE/...
        (aggrL*M_candidate)) + n_CI), floor(N_CCE/aggrL));
end

return


