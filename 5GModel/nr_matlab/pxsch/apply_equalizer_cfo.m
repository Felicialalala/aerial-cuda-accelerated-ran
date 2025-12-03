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

function X_est = apply_equalizer_cfo(Y, W, symIdx_data, symIdx_dmrs, nl, Nf, cfo)

%function applies equalizer filter to compute soft estimates of the
%transmited signal

%inputs:
% Y     --> signal recieved by base station. Dim: Nf x Nt x L_BS
% W     --> equalizer filter. Dim: L_UE x L_BS x Nf

%outputs:
% X_est --> soft estimates. Dim: L_UE x Nf x nSym_data


L_UE = nl;            % max number of spatially multiplexed layers

%%
%START

nSym_data = length(symIdx_data);

%permute recieved signal:
Y = permute(Y,[3 1 2]); %now: L_BS x Nf x Nt

X_est = zeros(L_UE,Nf,nSym_data);

for f = 1 : Nf
    for t = 1 : nSym_data
        X_est(:,f,t) =  W(:,:,f) * Y(:,f,symIdx_data(t));
        for ll = 1:L_UE
            X_est(ll,f,t) = X_est(ll,f,t)*exp(-1i*2*pi*cfo(ll)*(symIdx_data(t)-symIdx_dmrs(1)));
        end
    end
end

X_est = permute(X_est,[2 3 1]);

end



