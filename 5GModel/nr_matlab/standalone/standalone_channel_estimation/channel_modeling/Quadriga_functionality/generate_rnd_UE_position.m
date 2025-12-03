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

function l = generate_rnd_UE_position(l,LOS_flag)

%generate a rnd position for a UE.

%%
%START

r = 1000;
UE_angle = rand*pi/2 - pi/4;

UE_x = cos(UE_angle)*r;
UE_y = sin(UE_angle)*r;

if LOS_flag == 1
    l.rx_position = [UE_x ; UE_y ; 0];
else
    l.rx_position = [UE_x ; UE_y ; 1];
end

end


