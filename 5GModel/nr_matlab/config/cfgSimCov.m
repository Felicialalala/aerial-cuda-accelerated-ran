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

function cfgSimCov(varargin)
    % function SysPar = initSysPar(varargin)
    %
    % This function initializes structures related to code coverage.
    %
    % Input:    Reserved
    %
    % Output:   SysPar: structure with all simulation configurations
    %

global SimCov;

if (isstruct(SimCov) == 0)
    % Do one-time setup
    SimCov.doCodeCoverage = 0;

    % NOTE: It is not currently possible to run this enhanced code coverage with parallel workers
    if (SimCov.doCodeCoverage == 1)

        % Currently the only thing we check here is Zc in pxsch/derive_coding_main.m
        SimCov.pxsch.Zc_finder.count = zeros(51,1);
        SimCov.pxsch.Zc_finder.mcsTable = zeros(51,1);
        SimCov.pxsch.Zc_finder.mcsIndex = zeros(51,1);
        SimCov.pxsch.Zc_finder.nrb = zeros(51,1);
        SimCov.pxsch.Zc_finder.nsym = zeros(51,1);
    end
end

% TODO: Only tested rv = 0, need to cover 1 and 2
% TODO: Is multi-codeblock needed?

% Line to print out pdsch CFG compliance table for Zc's (then prune the 0 entries)
%sprintf('%4d, %2d,      %3d,  %2d, %2d, %3d,   %3d,     %2d,  %4d, %4d,   %5d,     %4d;  Zc=%d\n',transpose([3015+transpose(1:51) SysCov_Zc_finder_mcsTable SysCov_Zc_finder_mcsIndex ones(51,1) zeros(51,1) SysCov_Zc_finder_nrb 2*ones(51,1) SysCov_Zc_finder_nsym 1*ones(51,1) zeros(51,1) zeros(51,1) zeros(51,1) transpose(Z)]))

% Line to print out pdsch CFG compliance table for modcodes
%L=29; sprintf('%4d, %2d,      %3d,  %2d, %2d, %3d,   %3d,     %2d,  %4d, %4d,   %5d,     %4d;  %% mcs sweep\n',transpose([3066+transpose(1:L) ones(L,1) transpose(0:L-1) ones(L,1) zeros(L,1) 273*ones(L,1) 2*ones(L,1) 3*ones(L,1) 1*ones(L,1) zeros(L,1) zeros(L,1) zeros(L,1)]))
%L=28; sprintf('%4d, %2d,      %3d,  %2d, %2d, %3d,   %3d,     %2d,  %4d, %4d,   %5d,     %4d;  %% mcs sweep\n',transpose([3095+transpose(1:L) 2*ones(L,1) transpose(0:L-1) ones(L,1) zeros(L,1) 273*ones(L,1) 2*ones(L,1) 3*ones(L,1) 1*ones(L,1) zeros(L,1) zeros(L,1) zeros(L,1)]))
%L=29; sprintf('%4d, %2d,      %3d,  %2d, %2d, %3d,   %3d,     %2d,  %4d, %4d,   %5d,     %4d;  %% mcs sweep\n',transpose([3123+transpose(1:L) 3*ones(L,1) transpose(0:L-1) ones(L,1) zeros(L,1) 273*ones(L,1) 2*ones(L,1) 3*ones(L,1) 1*ones(L,1) zeros(L,1) zeros(L,1) zeros(L,1)]))
