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

function plot_channel_SNR(SNR)

%function  plots the pdf and cdf of the channel estimation SNR

%%
%START

MIN = min(SNR) - 3;
MAX = max(SNR) + 3;

[~,density,xmesh,cdf]=kde(SNR,100,MIN,MAX);

figure
plot(xmesh,density,'linewidth',1.5);
grid on
xlabel('channel SNR');
ylabel('probability');
title('channel SNR pdf');

figure
semilogy(xmesh,cdf,'linewidth',1.5);
grid on
xlabel('channel SNR');
ylabel('probability');
title('channel SNR cdf');
ylim([10^(-2) (1 - 10^(-2))]);



