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

function compute_ChEst_snr(H,H_est)

% function computes average srs channel estimation snr


% inputs:
% H     --> uplink channel of all SRS users. Dim: L_gNB x 4 x Nf x nUe
% H_est --> estimated channel. Dim: 136 x 4 x L_gNB x nUe

nPrb = size(H,3) / 12;

%%
%SETUP

% extract estiamted subcarriers from true channel:
idx = 12 : 24 : (nPrb*12 - 1);
H = permute(H,[3 2 1 4]);   % now: Nf x 4 x L_gNB x nUe
H = H(idx+1,:,:,:);         % now: 136 x 4 x L_gNB x nUe


%%
%START

S = abs(H).^2;
E = abs(H - H_est).^2;

ChEst_snr = 10*log10(mean(S(:)) / mean(E(:)))

% E = mean(E,1);
% E = mean(E,3);
% E = squeeze(E);
% 
% -10*log10(E)

% 
% figure
% plot(abs(H_est(:,1,1)));
% hold on
% plot(abs(H(:,1,1)));
% legend('est channel','true channel','location','best');
% grid on
% xlabel('prb/2');
% ylabel('amplitude');






end

