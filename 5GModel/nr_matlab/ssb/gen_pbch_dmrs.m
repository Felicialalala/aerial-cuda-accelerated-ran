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

function r = gen_pbch_dmrs(i_bar_ssb,N_id)

% function build pbch dmrs signal

%inputs:
% L_max     --> max number of SS blocks (4,8, or 64)
% block_idx --> SS block index (0 - L_max)
% n_hf      --> 0 or 1. Indiates is SS tx on first on second half-frame
% N_id      --> physical cell id

%outputs:
% r         --> pbch dmrs signal. Dim: 144 x 1

%%

c_init = 2^11*(i_bar_ssb + 1)*(floor(N_id/4) + 1) +...
    2^6*(i_bar_ssb + 1) + mod(N_id,4);

%next, compute Gold sequence:
c = build_Gold_sequence(c_init,288);

%finally, qpsk modulate:
r = qpsk_modulate(c,288);

end
