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

function QL = generate_rnd_UE_track(QL,UE_speed,PAR)

%generate a random UE track (location of UE during subframe)

%inputs:
%QL --> Quadriga layout object
%UE_speed --> speed of UE (m/s)

%%
%TIME

dt = PAR.mod.dt;
Nt = PAR.mod.Nt;

t = dt * (0 : (Nt - 1));

%%
%VELOCITY

%generate rnd direction of travel:
UE_velocity = [randn(2,1) ; 0];
UE_velocity = UE_velocity / norm(UE_velocity);

%scale to speed:
UE_velocity = UE_speed * UE_velocity;

%%
%TRACK

UE_track = UE_velocity*t;

QL.track.positions = UE_track;
QL.track.no_snapshots = Nt;

