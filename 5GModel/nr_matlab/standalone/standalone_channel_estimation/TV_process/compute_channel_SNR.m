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

function SNR = compute_channel_SNR(H_true,H_est,num_UE)

%compute the channel SNR for all the users

%inputs:
%H_true --> true channel
%H_est --> estimated channel
%num_UE --> number of UEs

%%
%START

SNR = zeros(num_UE,1);

for i = 1 : num_UE
    E = abs(H_true(:,:,i) - H_est(:,:,i)).^2;
    S = abs(H_true(:,:,i)).^2;
    
    SNR(i) = 10*log10(mean(S(:) / mean(E(:))));
end





