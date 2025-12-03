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

function TbCbs = generate_rnd_TB(PuschCfg)

%function generates a random transport block for user

%%
%PARAMATERS

BGN = PuschCfg.coding.BGN;  % 1 or 2. Indicates which base graph used
Zc = PuschCfg.coding.Zc;    % lifting size
C = PuschCfg.coding.C;      % number of codeblocks

%%
%START
rng('default');
if BGN == 1
    TbCbs = round(rand(Zc*22,C));
else
    TbCbs = round(rand(Zc*10,C));
end


end

