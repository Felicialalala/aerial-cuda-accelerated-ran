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

function grid_size = bw2grid(bw, mu)
    BW_LUT = [5, 10,15,20,25,30,35,40,45,50,60,70,80,90,100];
    index = find(BW_LUT==bw);
    GridsSteps = [25,52,79,106,133,160,188,216,242,270, -1, -1, -1, -1,-1;
                  11,24,38, 51, 65, 78, 92,106,119,133,162,189,217,245,273;
                  -1,11,18, 24, 31, 38, 44, 51, 58, 65, 79, 93,107,121,135];
    grid_size = GridsSteps(mu+1,index);
return