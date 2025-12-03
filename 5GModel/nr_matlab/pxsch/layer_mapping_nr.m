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

function TbLayerMapped = layer_mapping_nr(TbScramCbs, nl, qam)

%function performs layer mapping

%inputs:
% TbScramCbs    --> scrambled/encoded TB

%outputs:
% TbLayerMapped  --> layer mapped TB

%PARAMATERS


% nl = alloc.nl;            % number of layers transmited by user
% qam = coding.qam;  % number of bits/qam

N = length(TbScramCbs);     % number of bits
%START

TbLayerMapped = reshape(TbScramCbs,qam,nl,N / (qam*nl)); % qam x layer x RE
TbLayerMapped = permute(TbLayerMapped,[1 3 2]);          % qam x RE x layer
TbLayerMapped = TbLayerMapped(:);


return
