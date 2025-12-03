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

function X_est = apply_equalizer_tdi(Y, W, symIdx_data, symIdx_dmrs, nl, ...
    Nf, cfo_est_symbols_layers, to_est, enableCfoCorrection, enableToCorrection, TdiMode, ...
    numDmrsCdmGrpsNoData)

%function applies equalizer filter to_est compute soft estimates of the
%transmited signal

%inputs:
% Y     --> signal recieved by base station. Dim: Nf x Nt x L_BS
% W     --> equalizer filter. Dim: L_UE x L_BS x Nf

%outputs:
% X_est --> soft estimates. Dim: L_UE x Nf x nSym_data

global SimCtrl;
fp_flag = SimCtrl.fp_flag_pusch_equalizer;

L_UE    = nl;           % max number of spatially multiplexed layers
Y       = double(Y);    % note that the input data Y is in single format. Need to convert it to double format for the following calculation.

%%
%START

if numDmrsCdmGrpsNoData == 1
    symIdx_data = [min(symIdx_data(1), symIdx_dmrs(1)) : max(symIdx_data(end), symIdx_dmrs(end))];
end
nSym_data = length(symIdx_data);

imp_sel_flag = 1; % Hardcoded to 1 to select the Varray version by default
if  imp_sel_flag == 0 % this is the legacy 5GModel implementation
    %permute recieved signal:
    Y = permute(Y,[3 1 2]); %now: L_BS x Nf x Nt
    X_est = zeros(L_UE,Nf,nSym_data);
    for f = 1 : Nf
        for t = 1 : nSym_data
            idxSym = symIdx_data(t);
            %         X_est(:,f,t) =  W(:,:,f) * Y(:,f,symIdx_data(t));
            if TdiMode > 0
                X_est(:,f,t) =  W{t}(:,:,f) * Y(:,f,idxSym);
            else
                X_est(:,f,t) =  W(:,:,f) * Y(:,f,idxSym);
            end
            for ll = 1:L_UE
                if enableCfoCorrection
                    cfo_est_rot = exp(-1i*2*pi*cfo_est_symbols_layers(idxSym,ll)*(idxSym-symIdx_dmrs(1)));
                    X_est(ll,f,t) = X_est(ll,f,t)*cfo_est_rot;
                end
                if enableToCorrection
                    if f <= Nf/2
                        to_est_rot = exp(-1i*2*pi*to_est(ll)*(f-1-Nf/2));
                    else
                        to_est_rot = exp(-1i*2*pi*to_est(ll)*(f-Nf/2));
                    end
                    X_est(ll,f,t) = X_est(ll,f,t)*to_est_rot;
                end
            end
        end
    end
    X_est = permute(X_est,[2 3 1]);
else % this is the new 5GModel implementation in Varray
    %permute recieved signal:
    Y = permute(Y,[3 2 1]); %now: L_BS x Nt x Nf
    X_est = Varray(zeros(L_UE,nSym_data, Nf), fp_flag);
    for t = 1 : nSym_data
        idxSym = symIdx_data(t);
        % equalization
        if TdiMode > 0
            X_est(:,t,:) =  pagemtimes(Varray(W{t},fp_flag), Varray(Y(:,idxSym,:),fp_flag));
        else
            X_est(:,t,:) =  pagemtimes(Varray(W,fp_flag), Varray(Y(:,idxSym,:),fp_flag));
        end
        % compensate CFO
        if enableCfoCorrection
            cfo_est_rot_this_sym    = Varray(exp(-1i*2*pi*cfo_est_symbols_layers(idxSym,:)*(idxSym-symIdx_dmrs(1))).',fp_flag);
            cfo_est_rot_3d          = repmat(cfo_est_rot_this_sym, [1,1,Nf]);
            X_est(:,t,:)            = X_est(:,t,:).*cfo_est_rot_3d;
        end
        % compensate TO
        if enableToCorrection
            tmp_to_est_rot          = exp(-1i*2*pi*to_est(1,:).'*(-Nf/2+1:Nf/2)); % to_est: Ndmrs x Nlayers. In full-slot proc, to_est is the same over DMRS symbols.
            to_est_rot_3d           = Varray(permute(tmp_to_est_rot, [1,3,2]), fp_flag);
            X_est(:,t,:)            = X_est(:,t,:).*to_est_rot_3d;            
        end
    end
    X_est = getValue(permute(X_est,[3 2 1]));
end

return



