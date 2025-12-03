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

function cC2V = update_cC2V(c,cC2V,V2C,TannerPar)

%update cC2V for a single check node

%inputs:
%c    --> index of check node to be updated
%cC2V --> current cC2V messages. Dim: Zc x nC x 19
%V2C  --> current V2C messages. Dim: Zc x nC x 19
