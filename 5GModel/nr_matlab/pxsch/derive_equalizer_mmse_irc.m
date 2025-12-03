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

function [W,Ree] = derive_equalizer_mmse_irc(H_est, nCov, nl, Nf, posDmrs, startPrb, nPrb, subslot_processing_option)

%function build mmse equalization filters, along with the mmse error
%variance

%inputs:
%H_est --> channel estimate. Dim: L_BS x L_UE x Nf

%outputs:
%W     --> mmse equalization filters. Dim: L_UE x L_BS x Nf
%Ree   --> mmse error variance. Dim: L_UE x Nf

global SimCtrl;

biasCorrection = 1; % option to enable bias correction
applyReeLimit = 1;

L_UE = nl;      % max number of spatially multiplexed layers
[L_BS, ~, ~] = size(H_est); % total number of bs antennas

W = zeros(L_UE,L_BS,Nf);
Ree = zeros(L_UE,Nf);
Rxx = eye(L_UE);

if applyReeLimit == 1
    maxReeInvVal = 10000; % this constant should align with cuPHY
    minReeVal = 1/maxReeInvVal; % this constant should align with cuPHY
    minRee = minReeVal.*ones(L_UE,1);
end
global SimCtrl;
fp_flag = SimCtrl.fp_flag_pusch_equalizer;
imp_sel_flag = 1; % Hardcoded to 1 to select the Varray version by default
if  imp_sel_flag == 0 % this is the legacy 5GModel implementation
    for f = 1 : Nf
    
        prbIdx = floor((f-1)/12) + 1;
        %[W(:,:,f), Ree(:,f)] = irc_v1(H_est(:,:,f), nCov(:,:,prbIdx), Rxx);
        [W(:,:,f), Ree(:,f)] = irc_v2(H_est(:,:,f), nCov(:,:,prbIdx,:), Rxx, posDmrs, subslot_processing_option);
    %     [W(:,:,f), Ree(:,f)] = irc_v3(H_est(:,:,f), nCov(:,:,prbIdx), Rxx);
        
        if biasCorrection == 1
            lambda = 1 ./ (1 - Ree(:,f));
            W(:,:,f) = diag(lambda) * W(:,:,f);
            Ree(:,f) = lambda .* Ree(:,f);
        end
    
        if applyReeLimit == 1
            Ree(:,f) = max(minRee, Ree(:,f));
        end

    end

else % this is the new 5GModel implementation in Varray
    [W, Ree] = irc_Varray(Varray(H_est,fp_flag), Varray(nCov,fp_flag), Varray(Rxx,fp_flag), posDmrs, startPrb, nPrb, subslot_processing_option, fp_flag);
    
    if biasCorrection == 1
        nRx = size(H_est,1);
        lambda = Varray(1.0, fp_flag) ./ (Varray(1.0, fp_flag) - Ree);
        lambda = Varray(min(getValue(Varray(SimCtrl.alg.pusch_equalizer_bias_corr_limit, fp_flag)), lambda.value), fp_flag); % this limiting is important
        lambda_3d = permute(repmat(lambda,[1,1,nRx]),[1,3,2]);
        W = lambda_3d.*W;
        Ree = lambda.*Ree;
    end
    % figure; Ree_before_corr = 0:0.01:1;lambda=1./(1-Ree_before_corr);postSINR=1./Ree_before_corr-1;semilogy(Ree_before_corr,lambda);hold on;semilogy(Ree_before_corr,postSINR); xlabel('Ree value before bias correction');ylabel('Value')
    enable_Ree_avg_per_PRB = 0;
    if enable_Ree_avg_per_PRB
        Ree_3d = reshape(Ree, [nl, 12, nPrb]);
        Ree_avg = mean(Ree_3d,2);
        Ree_3d_repeat = repmat(Ree_avg,[1,12,1]);
        Ree = reshape(Ree_3d_repeat,[nl, Nf]);
    end

    if applyReeLimit == 1
        Ree = Varray(max(getValue(Varray(minRee, fp_flag)), Ree.value), fp_flag);
    end

    W = W.value;
    Ree = Ree.value;
end


return

function [W_f, Ree] = irc_v1(H_est, nCov, Rxx)

    nCovInv = pinv(nCov);
    
    %compute Gram matrix:    
    G_f = H_est'*nCovInv*H_est + inv(Rxx);

    %compute error covariance:
    Ree_f = pinv(G_f);

    %compute filter:
    W_f = Ree_f * H_est' * nCovInv;

    %wrap:
    Ree = diag(Ree_f);
return

function [W_f, Ree] = irc_v2(H, nCov, Rxx, posDmrs,subslot_processing_option)

    if (subslot_processing_option == 0) || (subslot_processing_option == 1)
        % Force diagonal to be Real to make Choleksy happy
        nCov_tmp = nCov(:,:,1,posDmrs); % note that for full-slot proc (i.e., SimCtrl.subslot_processing_option == 0), nCov(:,:,1,1) and nCov(:,:,1,2) should be the same, both of which duplicate (schrinked) avg nCov.
        nCov = nCov_tmp - diag(nCov_tmp) + real(diag(nCov_tmp));    
        LInv = inv(tril(chol(nCov, 'lower')));   
    elseif subslot_processing_option == 2
        LInv = zeros(size(nCov(:,:,1,1)));
        for idx_dmrs_pos = 1:posDmrs
            nCov_tmp = nCov(:,:,1,idx_dmrs_pos); 
            nCov_tmp = nCov_tmp - diag(nCov_tmp) + real(diag(nCov_tmp));    
            LInv = LInv + inv(tril(chol(nCov_tmp, 'lower')));
        end
        LInv = LInv/posDmrs;        
    elseif subslot_processing_option == 3 
        nCov_avg = zeros(size(nCov(:,:,1,1)));
        for idx_dmrs_pos = 1:posDmrs
            nCov_tmp = nCov(:,:,1,idx_dmrs_pos); 
            nCov_avg = nCov_avg + nCov_tmp;            
        end
        nCov_avg = nCov_avg/posDmrs;
        nCov_avg = nCov_avg - diag(nCov_avg) + real(diag(nCov_avg));    
        LInv = inv(tril(chol(nCov_avg, 'lower')));
    end
    N = LInv*H;
    
    %nCovInv = pinv(nCov);
    
    %compute Gram matrix:    
    G_f = N'*N + inv(Rxx);

    %compute error covariance:
    Ree_f = pinv(G_f);

    %compute filter:
    W_f = Ree_f * N' * LInv;

    %wrap:
    Ree = diag(Ree_f);
return

function [W_f, Ree] = irc_v3(H, nCov, Rxx)
    
    %compute Gram matrix:    
    G_f = H*Rxx*H' + nCov;    

    %compute filter:
    W_f = H'*inv(G_f);

    %compute error covariance:
    Ree_f = eye(size(W_f,1)) - W_f*H;

    %wrap:
    Ree = diag(Ree_f);
return

function [W, Ree] = irc_Varray(H, nCov, Rxx, posDmrs, startPrb, nPrb,subslot_processing_option, fp_flag)
       
    if (subslot_processing_option == 0) || (subslot_processing_option == 1)
        prbIdxs = startPrb : (startPrb + nPrb - 1); 
        reIdxs = (startPrb-1)*12+1:(startPrb + nPrb - 1)*12;
        nCov_tmp        = nCov(:,:,prbIdxs,posDmrs); % note that for full-slot proc (i.e., SimCtrl.subslot_processing_option == 0), nCov(:,:,1,1) and nCov(:,:,1,2) should be the same, both of which duplicate (schrinked) avg nCov.
        flag_enable_normalize_nCov = 1;
        if flag_enable_normalize_nCov
            % scale nCov
            [nRx,~,nPRBs]   = size(getValue(nCov_tmp)); 
            idx_diag_nCov   = 1:nRx+1:nRx^2;
            nCov_2D         = reshape(nCov_tmp.value,[nRx^2,nPRBs]);
            nCov_diag_real  = real(nCov_2D(idx_diag_nCov,:));
            scale           = util_get_normalization_scale(nCov_diag_real);
        else
            scale           = 1.0;
        end
        scale_inv       = Varray(1/scale, fp_flag);% invert scale in double format then quantize it back to FPxx
        nCov_tmp        = Varray(nCov_tmp.value*scale, fp_flag); % let's scale nCov in double format, then quantize it back to FPxx
        % get sqrt of nCov, i.e., whitening matrix
        LInv_perPRB     = inv_tri(chol(nCov_tmp, 'lower',fp_flag), 'lower', fp_flag);  % Chol decompostion and inversion
        LInv_perRE      = repmat(LInv_perPRB,[1,1,1,12]);
        LInv_perRE_tmp  = permute(LInv_perRE, [1,2,4,3]);
        [nRx, ~, nPRBs] = size(nCov_tmp.value);
        nREs            = 12*nPRBs;
        LInv            = reshape(LInv_perRE_tmp,[nRx,nRx,nREs]);
    elseif subslot_processing_option == 2
        prbIdxs = startPrb : (startPrb + nPrb - 1); 
        reIdxs = (startPrb-1)*12+1:(startPrb + nPrb - 1)*12;
        LInv = Varray(zeros(size(nCov(:,:,:,1))), fp_flag);
        for idx_dmrs_pos = 1:posDmrs
            nCov_tmp        = nCov(:,:,prbIdxs,idx_dmrs_pos);
            flag_enable_normalize_nCov = 1;
            if flag_enable_normalize_nCov && (idx_dmrs_pos==1) % decide nCov scaling factor according to the first DMRS symbol only
                % scale nCov
                [nRx,~,nPRBs]   = size(getValue(nCov_tmp)); 
                idx_diag_nCov   = 1:nRx+1:nRx^2;
                nCov_2D         = reshape(nCov_tmp.value,[nRx^2,nPRBs]);
                nCov_diag_real  = real(nCov_2D(idx_diag_nCov,:));
                scale           = util_get_normalization_scale(nCov_diag_real);
            else
                if ~flag_enable_normalize_nCov
                    scale           = 1.0;
                end
            end
            scale_inv       = Varray(1/scale, fp_flag);% invert scale in double format then quantize it back to FPxx
            nCov_tmp        = Varray(nCov_tmp.value*scale, fp_flag); % let's scale nCov in double format, then quantize it back to FPxx
            % get sqrt of nCov, i.e., whitening matrix
            LInv_perPRB     = inv_tri(chol(nCov_tmp, 'lower',fp_flag), 'lower', fp_flag);  % Chol decompostion and inversion
            LInv_perRE      = repmat(LInv_perPRB,[1,1,1,12]);
            LInv_perRE_tmp  = permute(LInv_perRE, [1,2,4,3]);
            [nRx, ~, nPRBs] = size(nCov_tmp.value);
            nREs            = 12*nPRBs;
            LInv = LInv + reshape(LInv_perRE_tmp,[nRx,nRx,nREs]);
        end
        LInv = LInv/Varray(posDmrs, fp_flag); 
    else
        error('Undefined!')
    end
    N = pagemtimes(LInv, H(:,:,reIdxs));
        
    %compute Gram matrix:    
    G = pagemtimes(N,'ctranspose',N,'none') + Varray(repmat(inv(Rxx.value),[1,1,nREs]), fp_flag)*scale_inv;

    % LDL
    [L, D, U] = ldl(G, fp_flag);

    % use forward and backward substitition to get Ree and W_f
    % G([Ginv | W]) = ([I | M]) -> LDU([Ginv | W]) = ([I | M])
    nLayers = size(G.value,1);
    M = pagemtimes(N,'ctranspose', LInv,'none');
    A = horzcat(Varray(repmat(eye(nLayers),[1,1,nREs]),fp_flag),M);
    % forward sub
    tmp1 = forward_sub(L, A, fp_flag);
    % backward sub
    prod_DU = pagemtimes(D,U);
    tmp2 = backward_sub(prod_DU, tmp1, fp_flag);
    
    % assignment
    Ree_mat = tmp2(1:nLayers, 1:nLayers,:);
    idx_diag = 1:nLayers+1:nLayers^2;
    Ree_mat_tmp = reshape(Ree_mat, [nLayers^2, nREs]);
    Ree_tmp = Ree_mat_tmp(idx_diag,:);
    numPRBs_carrier = size(H.value,3);
    Ree = Varray(zeros(nLayers, numPRBs_carrier), fp_flag);
    W = Varray(zeros(nLayers,nRx,numPRBs_carrier), fp_flag);
    Ree(:,reIdxs) = Ree_tmp*scale_inv;
    W(:,:,reIdxs) = tmp2(1:nLayers,nLayers+1:(nLayers+nRx),:);    
return
