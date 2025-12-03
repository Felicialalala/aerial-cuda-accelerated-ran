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

function b = pbch_descrambling(e,E,N_id,L_max,block_idx)

%function performs pbch scrambling

%inputs:
% e         --> rate matched bits. Dim: E x 1
% E         --> number of transmit bits
% N_id      --> physical cell id
% L_max     --> max number of pbch blocks in pbch period (4, 8, or 64)
% block_idx --> pbch block index

%outputs:
% b        --> scrambled bits. Dim: E x 1

%%
%START

% extract LSB bits from block index, convert to integer:
if L_max == 4
    % two LSBs
    v = mod(block_idx,4);
else
    % three LSBs
    v = mod(block_idx,8);
end

% compute Gold sequence:
c = build_Gold_sequence(N_id,(v+1)*E);

% descramble:
b = (1-2*c(v*E+1:(v+1)*E)).* e;

end
