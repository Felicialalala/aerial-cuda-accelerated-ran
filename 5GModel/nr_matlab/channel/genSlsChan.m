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

clear all; close all;
rng(1);

tic;

global debugParam

debugParam.forceIndoor.enable = 0;
debugParam.forceIndoor.value = 0;

debugParam.forceLOS.enable = 0;
debugParam.forceLOS.value = 0;

debugParam.fixedLSP.enable = 0;
debugParam.fixedSSP.enable = 0;

%------------
% set simulation duration
%------------

N_frame = 100; % number of frames for simulations
[timeSeq, Nbatch, NbatchSamp] = genChanBatchParam(N_frame);

%------------
% Set environment, network layout, and antenna array parameters
%------------

% step-1a: set scenario
scenario = 'UMa'; % support 'UMa', 'UMi', 'RMa'

% step-1b: 
% set BS numbers and BS antenna type
nSite = 1;
nSecPerSite = 3;
nBs = nSite*nSecPerSite;

BsAnt.size = [1 4 1];
BsAnt.spacing = [0.5 0.5];
BsAnt.polarAngles = [45 -45];
BsAnt.element = 'directional';
BsAnt.orientationBase = 0;

% set UT numbers and UT antenna type
nUt = 10*nBs;

UtAnt.size = [1 2 1];
UtAnt.spacing = [0.5 0.5];
UtAnt.orientation = 0;
UtAnt.polarAngles = [90 0];
UtAnt.element = 'isotropic';

% set UT drop area
if nSite == 1
    dist = 1; % UT drop area = [1 x ISD, 1 x ISD]
elseif nSite <= 7
    dist = 3; % UT drop area = [3 x ISD, 3 x ISD]
elseif nSite <= 19
    dist = 5; % UT drop area = [5 x ISD, 5 x ISD]
else
    error('nSite is not supported ...\n');
end

% step-1c/f/e: set BS site and UT location/orientation, isIndoor and speed

% initialize BS site parameters (location, orientation)
Site = initSiteParam(scenario, nSite, nSecPerSite, BsAnt);

% initialize UT parameters (location, orientation, isIndoor and speed)
UT = initUtParam(scenario, nUt, dist, Site, UtAnt);

% step 1-g
fc = 2e9;
BW = 100e6;
lightSpeed = 3e8; % light speed
lambda_0 = lightSpeed/fc;

% constant
d2pi = pi/180;
pi2d = 180/pi;

%------------
% Large scale parameters for UT/Site links
%------------

for idxSite = 1:nSite
    x_Site = Site(idxSite).loc(1);
    y_Site = Site(idxSite).loc(2);
    h_BS = Site(idxSite).loc(3);
    for idxUt = 1:nUt
        x_UT = UT(idxUt).loc(1);
        y_UT = UT(idxUt).loc(2);
        h_UT = UT(idxUt).loc(3);
        isIndoor = UT(idxUt).isIndoor;

        % step 1-c: derive distances
        d_2D = sqrt((x_UT-x_Site)^2 + (y_UT-y_Site)^2);
        d_3D = sqrt((x_UT-x_Site)^2 + (y_UT-y_Site)^2 + (h_UT-h_BS)^2);
        d_2D_in = UT(idxUt).d_2D_in;
        d_2D_out = d_2D - d_2D_in;
        LinkParam(idxSite, idxUt).d_3D = d_3D;
        LinkParam(idxSite, idxUt).d_2D = d_2D;
        LinkParam(idxSite, idxUt).d_2D_out = d_2D_out;
        
        % step 1-c: derive LOS angles
        Site2UT_vec = (x_UT-x_Site)+j*(y_UT-y_Site); 
        LinkParam(idxSite, idxUt).phi_LOS_AOD = angle(Site2UT_vec)*pi2d;
        LinkParam(idxSite, idxUt).phi_LOS_AOA = angle(-Site2UT_vec)*pi2d;
        LinkParam(idxSite, idxUt).theta_LOS_ZOD = (pi-acos((h_BS-h_UT)/d_3D))*pi2d;
        LinkParam(idxSite, idxUt).theta_LOS_ZOA = 180-LinkParam(idxSite, idxUt).theta_LOS_ZOD;

        % step 2, assign LOS/NLOS
        Pr_LOS = derivePrLOS(scenario, d_2D_out, h_UT);     
        isLOS = (rand(1) < Pr_LOS);
        if debugParam.forceLOS.enable
            isLOS = debugParam.forceLOS.value;
        end
        LinkParam(idxSite, idxUt).isLOS = isLOS;

        % step 3, calculate passloss
        [PL, sigma_SF] = derivePL(scenario, isIndoor, isLOS, ...
            d_2D, d_2D_in, d_3D, h_BS, h_UT, fc);
        LinkParam(idxSite, idxUt).PL = PL;
        LinkParam(idxSite, idxUt).sigma_SF = sigma_SF;
    end

    % step 4, generate large scale parameters (LSP)
    Link2OneSite = LinkParam(idxSite, :);
    LinkLSParam = genLinkLSParam(scenario, fc, UT, Link2OneSite);
    for idxUt = 1:nUt
        LinkParam(idxSite, idxUt).SF = LinkLSParam(idxUt, 1);
        LinkParam(idxSite, idxUt).K = LinkLSParam(idxUt, 2);
        LinkParam(idxSite, idxUt).DS = LinkLSParam(idxUt, 3);
        LinkParam(idxSite, idxUt).ASD = LinkLSParam(idxUt, 4);
        LinkParam(idxSite, idxUt).ASA = LinkLSParam(idxUt, 5);
        LinkParam(idxSite, idxUt).ZSD = LinkLSParam(idxUt, 6);
        LinkParam(idxSite, idxUt).ZSA = LinkLSParam(idxUt, 7);
    end
end

%---------------
% Small scale parameters
%---------------

nRxAnt = prod(UtAnt.size);
nTxAnt = prod(BsAnt.size);
H = zeros(nUt, nSite, nSecPerSite, nRxAnt, nTxAnt, 24, Nbatch);
tao = zeros(nUt, nSite, 24);
pw = zeros(nUt, nSite, 24);

for idxUt = 1:nUt
    x_UT = UT(idxUt).loc(1);
    y_UT = UT(idxUt).loc(2);
    h_UT = UT(idxUt).loc(3);
    for idxSite = 1:nSite
        x_Site = Site(idxSite).loc(1);
        y_Site = Site(idxSite).loc(2);
        h_BS = Site(idxSite).loc(3);

        % load Site/UT link parameters
        phi_LOS_AOD = LinkParam(idxSite, idxUt).phi_LOS_AOD;
        phi_LOS_AOA = LinkParam(idxSite, idxUt).phi_LOS_AOA;
        theta_LOS_ZOD = LinkParam(idxSite, idxUt).theta_LOS_ZOD;
        theta_LOS_ZOA = LinkParam(idxSite, idxUt).theta_LOS_ZOA;
        isIndoor = UT(idxUt).isIndoor;
        d_2D_in = UT(idxUt).d_2D_in;
        d_2D = LinkParam(idxSite, idxUt).d_2D;
        d_2D_out = LinkParam(idxSite, idxUt).d_2D_out;
        d_3D = LinkParam(idxSite, idxUt).d_3D;
        isLOS = LinkParam(idxSite, idxUt).isLOS;
        if isLOS
            idxLOS = 1;
        else
            idxLOS = 2;
        end
        SF = LinkParam(idxSite, idxUt).SF;
        K = LinkParam(idxSite, idxUt).K;
        DS = LinkParam(idxSite, idxUt).DS;
        ASD = LinkParam(idxSite, idxUt).ASD;
        ASA = LinkParam(idxSite, idxUt).ASA;
        ZSD = LinkParam(idxSite, idxUt).ZSD;
        ZSA = LinkParam(idxSite, idxUt).ZSA;        
        LSParamTable = genLSParamTable(scenario, fc, d_2D, h_UT);
 
        % step 5, generate cluster delays
        nCluster = LSParamTable.nCluster(idxLOS);
        r_tao = LSParamTable.r_tao(idxLOS);

        [tao_n_NLOS, tao_n_LOS] = genClusterDelay(nCluster, DS, r_tao, isLOS, K);
        if isLOS
            tao_n = tao_n_LOS;
        else
            tao_n = tao_n_NLOS;
        end

        % step 6, generate cluster powers
        xi = LSParamTable.xi(idxLOS);
        [P_n_NLOS, P_n_LOS, idxWeakCluster] = genClusterPower(nCluster, tao_n_NLOS, r_tao, DS, xi, isLOS, K);
        if isLOS
            P_n = P_n_LOS;
        else
            P_n = P_n_NLOS;
        end
        tao_n(idxWeakCluster) = [];
        nCluster_updated = length(tao_n);

        % map 2 strongest clusters into 6 subclusters
        [~, strongest2clusters] = findMax(P_n, 2);
        P_n_updated = insertClusterPower(P_n, strongest2clusters);

        C_DS = LSParamTable.C_DS(idxLOS)*1e-9;        
        tao_n_updated = insertClusterDelay(tao_n, strongest2clusters, C_DS);

        % step 7, generate cluster angles
        C_phi_NLOS = LSParamTable.C_phi_NLOS;
        C_theta_NLOS = LSParamTable.C_theta_NLOS;
        mu_offset_ZOD = LSParamTable.mu_offset_ZOD(idxLOS);

        [phi_n_AOA, phi_n_AOD, theta_n_ZOA, theta_n_ZOD] = ...
            genClusterAngle(nCluster_updated, isLOS, K, P_n_NLOS, C_phi_NLOS, C_theta_NLOS, ...
            ASA, phi_LOS_AOA, ASD, phi_LOS_AOD, ZSA, theta_LOS_ZOA, ...
            isIndoor, ZSD, theta_LOS_ZOD, mu_offset_ZOD);

        % step 8, generate ray angles and couple rays
        RayOffsetAngles = LSParamTable.RayOffsetAngles;
        C_ASA = LSParamTable.C_ASA(idxLOS);
        C_ASD = LSParamTable.C_ASD(idxLOS);
        C_ZSA = LSParamTable.C_ZSA(idxLOS);
        mu_lgZSD = LSParamTable.mu_lgZSD(idxLOS);

        nRayPerCluster = LSParamTable.nRayPerCluster(idxLOS);
        [phi_n_m_AOA, phi_n_m_AOD, theta_n_m_ZOA, theta_n_m_ZOD] = ...
            genRayAngle(nCluster_updated, nRayPerCluster, phi_n_AOA, phi_n_AOD, ...
            theta_n_ZOA, theta_n_ZOD, C_ASA, C_ASD, C_ZSA, mu_lgZSD, ...
            RayOffsetAngles);
         
        % step 9: generate XPR
        mu_XPR = LSParamTable.mu_XPR(idxLOS);
        sigma_XPR = LSParamTable.sigma_XPR(idxLOS);
        kappa_n_m = genClusterXPR(nCluster_updated, nRayPerCluster, mu_XPR, sigma_XPR);

        % step 10: generate random phases
        PHI = pi*2*(rand(nCluster_updated, nRayPerCluster, 4)-0.5);

        % step 11: generate coefficients for each cluster
        RxAntSize = UT(idxUt).antSize;
        RxAntSpacing = UT(idxUt).antSpacing;
        RxAntPolarAngles = UT(idxUt).antPolarAngles;
        RxAntPattern = UT(idxUt).antElement;
        RxAntOrientation = UT(idxUt).antOrientation;
        nRxAnt = prod(RxAntSize);
        v_direction = [UT(idxUt).v(2), 0];
        v_speed = UT(idxUt).v(1);
        % dopplerHz = UT(idxUt).v(1)*fc/lightSpeed;
        
        for idxSec = 1:nSecPerSite
            TxAntSize = Site(idxSite).BS(idxSec).antSize;
            TxAntSpacing = Site(idxSite).BS(idxSec).antSpacing;
            TxAntPolarAngles = Site(idxSite).BS(idxSec).antPolarAngles;
            TxAntPattern = Site(idxSite).BS(idxSec).antElement;
            TxAntOrientation = Site(idxSite).BS(idxSec).antOrientation;
            nTxAnt = prod(TxAntSize);

            H_link = [];
            for u = 1:nRxAnt
                [m_RxAnt, n_RxAnt, p_RxAnt] = findAntLoc(u, RxAntSize);
                zetaRxAnt = RxAntPolarAngles(p_RxAnt);
                d_bar_rx_u = [0, (n_RxAnt-1)*RxAntSpacing(2), (m_RxAnt-1)*RxAntSpacing(1)]';
                for s = 1:nTxAnt
                    [m_TxAnt, n_TxAnt, p_TxAnt] = findAntLoc(s, TxAntSize);
                    zetaTxAnt = TxAntPolarAngles(p_TxAnt);
                    d_bar_tx_s = [0, (n_TxAnt-1)*TxAntSpacing(2), (m_TxAnt-1)*TxAntSpacing(1)]';
                    nn = 0;
                    H_u_s_NLOS = [];
                    for n = 1:nCluster_updated
                        raysInSubCluster = [];
                        if ismember(n, strongest2clusters)
                            raysInSubCluster{1} = [1 2 3 4 5 6 7 8 19 20];
                            raysInSubCluster{2} = [9, 10, 11, 12, 17, 18];
                            raysInSubCluster{3} = [13, 14, 15, 16];
                            nSubCluster = 3;
                        else
                            raysInSubCluster{1} = [1:20];
                            nSubCluster = 1;
                        end
                        for idxSubCluster = 1:nSubCluster
                            nn = nn + 1;
                            H_u_s_n = 0;
                            for m = raysInSubCluster{idxSubCluster}
                                theta_n_m_ZOA_prime = theta_n_m_ZOA(n, m) - RxAntOrientation(1);
                                phi_n_m_AOA_prime = phi_n_m_AOA(n, m) - RxAntOrientation(2);
                                [F_rx_u_theta, F_rx_u_phi] = calc_Field(RxAntPattern, theta_n_m_ZOA_prime, phi_n_m_AOA_prime, zetaRxAnt);
                                term1 = [F_rx_u_theta; F_rx_u_phi]';
                                kappa = kappa_n_m(n, m);
                                term2 = [exp(j*PHI(n, m, 1)), sqrt(1/kappa)*exp(j*PHI(n, m, 2));...
                                    sqrt(1/kappa)*exp(j*PHI(n, m, 3)), exp(j*PHI(n, m, 4))];
                                theta_n_m_ZOD_prime = theta_n_m_ZOD(n, m) - TxAntOrientation(1);
                                phi_n_m_AOD_prime = phi_n_m_AOD(n, m) - TxAntOrientation(2);
                                [F_tx_s_theta, F_tx_s_phi] = calc_Field(TxAntPattern, theta_n_m_ZOD_prime, phi_n_m_AOD_prime, zetaTxAnt);
                                term3 = [F_tx_s_theta; F_tx_s_phi];
                                r_head_rx_n_m = [sin(theta_n_m_ZOA(n, m)*d2pi)*cos(phi_n_m_AOA(n, m)*d2pi); ...
                                    sin(theta_n_m_ZOA(n, m)*d2pi)*sin(phi_n_m_AOA(n, m)*d2pi); ...
                                    cos(theta_n_m_ZOA(n, m)*d2pi)];
                                term4 = exp(j*2*pi*r_head_rx_n_m'*d_bar_rx_u);
                                r_head_tx_n_m = [sin(theta_n_m_ZOD(n, m)*d2pi)*cos(phi_n_m_AOD(n, m)*d2pi); ...
                                    sin(theta_n_m_ZOD(n, m)*d2pi)*sin(phi_n_m_AOD(n, m)*d2pi); ...
                                    cos(theta_n_m_ZOD(n, m)*d2pi)];
                                term5 = exp(j*2*pi*r_head_tx_n_m'*d_bar_tx_s);
                                v_bar = v_speed*[sin(v_direction(1))*cos(v_direction(2)), sin(v_direction(1))*sin(v_direction(2)), cos(v_direction(1))]';
                                term6 = exp(j*2*pi*(r_head_rx_n_m'*v_bar/lambda_0)*timeSeq);
                                % v_head = [sin(v_direction(1)*d2pi)*cos(v_direction(2)*d2pi), sin(v_direction(1)*d2pi)*sin(v_direction(2)*d2pi), cos(v_direction(1)*d2pi)]';
                                % term6 = exp(j*2*pi*(r_head_rx_n_m'*v_head*dopplerHz)*timeSeq);
                                rayCoeff = term1*term2*term3*term4*term5*term6; % equation 7.5-22 in 38.901
                                H_u_s_n = H_u_s_n + rayCoeff;
                            end % idxRay
                            H_u_s_NLOS(nn, :) = H_u_s_n*sqrt(P_n_updated(nn)/nRayPerCluster);
                        end % idxSubCluster
                    end % idxCluster
                    if isLOS
                        theta_LOS_ZOA_prime = theta_LOS_ZOA - RxAntOrientation(1);
                        phi_LOS_AOA_prime = phi_LOS_AOA - RxAntOrientation(2);
                        [F_rx_u_theta, F_rx_u_phi] = calc_Field(RxAntPattern, theta_LOS_ZOA_prime, phi_LOS_AOA_prime, zetaRxAnt);
                        term1 = [F_rx_u_theta; F_rx_u_phi]';
                        term2 = [1 0; 0 -1];
                        theta_LOS_ZOD_prime = theta_LOS_ZOD - TxAntOrientation(1);
                        phi_LOS_AOD_prime = phi_LOS_AOD - TxAntOrientation(2);
                        [F_tx_s_theta, F_tx_s_phi] = calc_Field(TxAntPattern, theta_LOS_ZOD_prime, phi_LOS_AOD_prime, zetaTxAnt);
                        term3 = [F_tx_s_theta; F_tx_s_phi];
                        r_head_rx_n_m = [sin(theta_LOS_ZOA*d2pi)*cos(phi_LOS_AOA*d2pi); ...
                            sin(theta_LOS_ZOA*d2pi)*sin(phi_LOS_AOA*d2pi); ...
                            cos(theta_LOS_ZOA*d2pi)];
                        term4 = exp(j*2*pi*r_head_rx_n_m'*d_bar_rx_u);
                        r_head_tx_n_m = [sin(theta_LOS_ZOD*d2pi)*cos(phi_LOS_AOD*d2pi); ...
                            sin(theta_LOS_ZOD*d2pi)*sin(phi_LOS_AOD*d2pi); ...
                            cos(theta_LOS_ZOD*d2pi)];
                        term5 = exp(j*2*pi*r_head_tx_n_m'*d_bar_tx_s);
                        v_bar = v_speed*[sin(v_direction(1))*cos(v_direction(2)), sin(v_direction(1))*sin(v_direction(2)), cos(v_direction(1))]';
                        term6 = exp(j*2*pi*(r_head_rx_n_m'*v_bar/lambda_0)*timeSeq);
                        % v_head = [sin(v_direction(1)*d2pi)*cos(v_direction(2)*d2pi), sin(v_direction(1)*d2pi)*sin(v_direction(2)*d2pi), cos(v_direction(1)*d2pi)]';
                        % term6 = exp(j*2*pi*(r_head_rx_n_m'*v_head*dopplerHz)*timeSeq);
                        rayCoeff = term1*term2*term3*exp(-j*2*pi*d_3D/lambda_0)*term4*term5*term6; % equation 7.5-22 in 38.901
                        H_u_s_LOS = rayCoeff;
                        K_R = 10^(K/10);
                        H_u_s = sqrt(1/(K_R+1)) * H_u_s_NLOS;
                        H_u_s(1, :) = sqrt(K_R/(K_R+1)) * H_u_s_LOS + H_u_s(1, :);
                        H_link(u, s, :, :) = H_u_s;
                    else
                        H_link(u, s, :, :) = H_u_s_NLOS;
                    end                    
                end % s (tx antenna element)
            end % u (rx antenna element)
            % Step 12: apply pathloss and shadowing
            pathGain = -LinkParam(idxSite, idxUt).PL + LinkParam(idxSite, idxUt).SF;
            H_RSRP = squeeze(H_link(:, :, :, 1));
            % H_power(idxUt, idxSite, idxSec) = sqrt(sum(abs(H_link(:)).^2/(nTxAnt*nRxAnt*Nbatch)));  
            H_power(idxUt, idxSite, idxSec) = sqrt(sum(abs(H_RSRP(:)).^2/(nTxAnt*nRxAnt))); 
            H_power_PL(idxUt, idxSite, idxSec) = 20*log10(H_power(idxUt, idxSite, idxSec)) + pathGain; 
            [d1, d2, d3, d4] = size(H_link);
           H(idxUt, idxSite, idxSec, 1:d1, 1:d2, 1:d3, 1:d4) = H_link;
        end % idxSec
        tao(idxUt, idxSite, 1:length(tao_n_updated)) = tao_n_updated; 
        pw(idxUt, idxSite, 1:length(P_n_updated)) = P_n_updated;
        nC(idxUt, idxSite) = length(tao_n_updated);
    end % idxSite
end % idxUt

for idxUt = 1:nUt
    Ut2BsPower = squeeze(H_power_PL(idxUt, :, :))';
    Ut2BsPower = Ut2BsPower(:);
    [rxPower, idxBs] = max(Ut2BsPower);
    UT(idxUt).connectedBS = [rxPower, idxBs];
end

% plotSite(Site, 1);
% plotUT(UT, 1);


% H: [nUt, nSite, nSector, nUtAnt, nBsAnt, nCluster, nBatch] 
fprintf(['\nGenerate channel coefficients H with dimension of \n' ...
    '[nUt, nSite, nSecPerSite, nUtAnt, nBsAnt, nCluster, nBatch] = [%d %d %d %d %d %d %d]\n'], size(H));
% tao: [nUt, nSite, nCluster]
fprintf(['\nGenerate channel delay tao with dimension of \n' ...
    '[nUt, nSite, nCluster] = [%d %d %d]\n\n'], size(tao));

% f_samp = 30e3*4096;
% [Hs, nTaps] = convertChannel(H, tao, pw, nC, f_samp);

toc;


function Site = initSiteParam(scenario, nSite, nSecPerSite, BsAnt)

% table 7.2-1
switch scenario
    case 'UMa'
        ISD = 500;
        BsHeight = 25; 
        theta_BS = 102; % 7.8-1
    case 'UMi'
        ISD = 200;
        BsHeight = 10; 
        theta_BS = 102; % 7.8-1        
    case 'RMa'
        ISD = 1732;
        BsHeight = 35; 
        theta_BS = 102; % 7.8-1  
    otherwise
        error('scenario is not supported ...\n')
end        

[x_site, y_site] = genSiteLocation(nSite, ISD);
phi_BS = [0, 120, -120];

for idxSite = 1:nSite
    Site(idxSite).loc = [x_site(idxSite), y_site(idxSite), BsHeight];
    for idxSec = 1:nSecPerSite
        Site(idxSite).BS(idxSec).antSize = BsAnt.size;
        Site(idxSite).BS(idxSec).antSpacing = BsAnt.spacing;        
        Site(idxSite).BS(idxSec).antOrientation = [theta_BS-90, ...
            BsAnt.orientationBase + phi_BS(idxSec)];
        Site(idxSite).BS(idxSec).antPolarAngles = BsAnt.polarAngles;
        Site(idxSite).BS(idxSec).antElement = BsAnt.element;
    end
end

end

function [x, y] = genSiteLocation(nSite, ISD)

% center site
x0 = 0;
y0 = 0;
% first loop (6 Sites next to center site)
x1 = cos(0:pi/3:(2*pi-pi/3));
y1 = sin(0:pi/3:(2*pi-pi/3));
% second loop (12 sites next to first loop)
xm = x1+circshift(x1, -1);
ym = y1+circshift(y1, -1);
x2 = [2*x1; xm];
x2 = x2(:)';
y2 = [2*y1; ym];
y2 = y2(:)';
% concatenate all 19 sites
x = [x0, x1, x2];
y = [y0, y1, y2];

% xy = x + j*y;
% plot(xy, '*');

if nSite > 19
    error('nSite > 19 is not supported ...\n');
else
    x = round(ISD * x(1:nSite)); % round to meter
    y = round(ISD * y(1:nSite)); % round to meter
end

end


function plotSite(Site, figNum)

figure(figNum);
hold on;
for idxSite = 1:length(Site)
    xy = Site(idxSite).loc(1) + j*Site(idxSite).loc(2);
    plot(complex(xy), 'ok', 'LineWidth',2, 'MarkerSize',10);
end

end

function UT = initUtParam(scenario, nUt, dist, Site, UtAnt)

global debugParam

% table 7.2-1
switch scenario
    case 'UMa'
        PrIndoor = 0.8;
        ISD = 500; 
        v_UT = 3e3/3600; 
        minSiteUtDist = 35;
        d_2D_in_max = 25; % 7.4.3.1
        theta_UE = 90;
    case 'UMi'
        PrIndoor = 0.8;
        ISD = 200; 
        v_UT = 3e3/3600; 
        minSiteUtDist = 10;
        d_2D_in_max = 25; % 7.4.3.1
        theta_UE = 90;        
    case 'RMa'
        PrIndoor = 0.5;
        ISD = 500; 
        v_UT = 3e3/3600; 
        minSiteUtDist = 35;
        d_2D_in_max = 10; % 7.4.3.1
        theta_UE = 90; 
    otherwise
        error('scenario is not supported ...\n')
end

for idxUt = 1:nUt
    isIndoor = (rand(1) < PrIndoor); % assign indoor UEs;
    if debugParam.forceIndoor.enable
        isIndoor = debugParam.forceIndoor.value;
    end
    if isIndoor
        d_2D_in = min(rand(1, 2)*d_2D_in_max); % 7.4.3.1
    else
        d_2D_in = 0;
    end
    UT(idxUt).d_2D_in = d_2D_in;
    UT(idxUt).isIndoor = isIndoor;
    h_UT = genhUT(scenario, isIndoor); % TR 36.873 Table 6-1
    Ut2dLoc = round(dist*ISD*(rand(1, 2)-0.5)); % round to meter
    % check UT/Site distance
    minDist = findMinDist(Ut2dLoc, Site);
    while minDist < minSiteUtDist
        Ut2dLoc = round(dist*ISD*(rand(1, 2)-0.5)); % round to meter
        minDist = findMinDist(Ut2dLoc, Site);
    end

    UT(idxUt).loc = [Ut2dLoc, h_UT];
    UT(idxUt).v = [v_UT, 360*(rand(1)-0.5)];
    UT(idxUt).antSize = UtAnt.size;
    UT(idxUt).antSpacing = UtAnt.spacing;
    UT(idxUt).antOrientation = [0, UtAnt.orientation];
    UT(idxUt).antPolarAngles = UtAnt.polarAngles;
    UT(idxUt).antElement = UtAnt.element;
end

end


function h_UT = genhUT(scenario, isIndoor)

% TR 36.873 Table 6-1
switch scenario
    case {'UMa', 'UMi', 'RMa'}
        UtHeight = 1.5;
        if isIndoor
            N_fl = floor(rand(1)*5)+4;
            n_fl = ceil(N_fl*rand(1));
        else
            n_fl = 1;
        end
        h_UT = 3*(n_fl-1) + UtHeight;
    otherwise
        error('scenario is not supported ...\n')
end

end

function minDist = findMinDist(Ut2dLoc, Site)

% find minimal distance between this UT and all Sites

nSite = length(Site);

minDist = 1e9;
for idxSite = 1:nSite
     thisDist = sqrt(sum(abs(Site(idxSite).loc(1:2)-Ut2dLoc).^2));
     if thisDist < minDist
         minDist = thisDist;
     end
end

end

function plotUT(UT, figNum)

figure(figNum);
hold on;
for idxUt = 1:length(UT)
    xy = UT(idxUt).loc(1) + j*UT(idxUt).loc(2);
    switch mod(UT(idxUt).connectedBS(2), 3)+1
        case 1
            plot(complex(xy), '+g', 'LineWidth',1, 'MarkerSize',5);
        case 2
            plot(complex(xy), '+b', 'LineWidth',1, 'MarkerSize',5);
        case 3
            plot(complex(xy), '+r', 'LineWidth',1, 'MarkerSize',5);
    end
    grid on;
end

end


function Pr_LOS = derivePrLOS(scenario, d_2D_out, h_UT)

% step 2: derive LOS probability

% table 7.4.2-1
switch scenario
    case 'UMa'
        if d_2D_out <= 18
            Pr_LOS = 1;
        else         
            if h_UT <= 13
                C_prime = 0;
            elseif h_UT <= 23
                C_prime = ((h_UT-13)/10)^1.5;
            else
                error('h_UT is not supported ...\n')
            end
            Pr_LOS = (18/d_2D_out + exp(-d_2D_out/63)*(1-18/d_2D_out))*...
                (1+C_prime*5/4*(d_2D_out/100)^3*exp(-d_2D_out/150));
        end
    case 'UMi'
        if d_2D_out <= 18
            Pr_LOS = 1;
        else         
            Pr_LOS = 18/d_2D_out + exp(-d_2D_out/36)*(1-18/d_2D_out);
        end
    case 'RMa'
        if d_2D_out <= 10
            Pr_LOS = 1;
        else         
            Pr_LOS = exp(-(d_2D_out-10)/1000);
        end        
    otherwise
        error('scenario is not supported ...\n')
end

end


function [PL, sigma_SF] = derivePL(scenario, isIndoor, isLOS, d_2D, d_2D_in, d_3D, h_BS, h_UT, fc)

% step 3: derive pathloss

lightSpeed = 3e8;

lgfc = log10(fc/1e9);
lgd3D = log10(d_3D);

% TR 38.901 Table 7.4.1-1
switch scenario
    case 'UMa'
        % derive g_d_2D
        if d_2D <= 18
            g_d_2D = 0;
        else
            g_d_2D = 5/4*(d_2D/100)^3*exp(-d_2D/150);
        end
        % derive h_E
        if h_UT < 13
            C_d_h = 0;
            h_E = 1;
        elseif h_UT <= 23            
            C_d_h = ((h_UT-13)/10)^1.5*g_d_2D;
            if rand(1) < 1/(1+C_d_h)
                h_E = 1;
            else
                h_E_vec = (12:3:h_UT-1.5);
                len_h_E_vec = length(h_E_vec);
                idx_h_E = ceil(rand(1)*len_h_E_vec);
                h_E = h_E_vec(idx_h_E);
            end
        else
            error('h_UT is not supported ...\n')
        end
        % derive d_prime_BP
        h_prime_Site = h_BS - h_E;
        h_prime_UT = h_UT - h_E;
        d_prime_BP = 4*h_prime_Site*h_prime_UT*fc/lightSpeed;
        % derive pathloss for outdoor
        if d_2D < d_prime_BP
            PL_LOS = 28 + 22*lgd3D + 20*lgfc;
        elseif d_2D < 5000
            PL_LOS = 28 + 40*lgd3D + 20*lgfc - ...
                9*log10(d_prime_BP^2 + (h_BS-h_UT)^2);
        else
            error('d_2D is not supported ...\n')
        end
        PL_prime = 13.54 + 39.08*lgd3D + 20*lgfc - 0.6*(h_UT-1.5);
        if isLOS
            PL = PL_LOS;
            sigma_SF = 4;
        else
            PL = max([PL_LOS, PL_prime]);
            sigma_SF = 6;
        end
        % derive pathloss for indoor
        if isIndoor
            if fc < 6e9 % table 7.4.3-3
                PL_tw = 20;
                PL_in = 0.5*d_2D_in;
                PL = PL + PL_tw + PL_in;
                sigma_SF = 7;
            else % table 7.4.3-2
                L_glass = 2 + 0.2*fc/1e9;
                L_concreate = 5 + 4*fc/1e9;
                L_IRRglass = 23 + 0.3*fc/1e9;
                O2I_model = 'low-pass'; % high/low-pass model
                if strcmp(O2I_model, 'low-pass')
                    PL_tw = 5 - 10*log10(0.3*10^(-L_glass/10) + ...
                        0.7*10^(-L_concreate/10));
                    PL_in = 0.5 * d_2D_in;
                    PL = PL + PL_tw + PL_in;
                    sigma_SF = 4.4;
                else
                    PL_tw = 5 - 10*log10(0.7*10^(-L_IRRglass/10) + ...
                        0.3*10^(-L_concreate/10));
                    PL_in = 0.5 * d_2D_in;
                    PL = PL + PL_tw + PL_in;
                    sigma_SF = 6.5;
                end
            end
        end
    case 'UMi'
        % derive g_d_2D
        if d_2D <= 18
            g_d_2D = 0;
        else
            g_d_2D = 5/4*(d_2D/100)^3*exp(-d_2D/150);
        end
        % derive h_E
        if h_UT < 13
            C_d_h = 0;
            h_E = 1;
        elseif h_UT <= 23            
            C_d_h = ((h_UT-13)/10)^1.5*g_d_2D;
            if rand(1) < 1/(1+C_d_h)
                h_E = 1;
            else
                h_E_vec = (12:3:h_UT-1.5);
                len_h_E_vec = length(h_E_vec);
                idx_h_E = ceil(rand(1)*len_h_E_vec);
                h_E = h_E_vec(idx_h_E);
            end
        else
            error('h_UT is not supported ...\n')
        end
        % derive d_prime_BP
        h_prime_Site = h_BS - h_E;
        h_prime_UT = h_UT - h_E;
        d_prime_BP = 4*h_prime_Site*h_prime_UT*fc/lightSpeed;
        % derive pathloss for outdoor
        if d_2D < d_prime_BP
            PL_LOS = 32.4 + 21*lgd3D + 20*lgfc;
        elseif d_2D < 5000
            PL_LOS = 32.4 + 40*lgd3D + 20*lgfc - ...
                9.5*log10(d_prime_BP^2 + (h_BS-h_UT)^2);
        else
            error('d_2D is not supported ...\n')
        end
        PL_prime = 22.4 + 35.3*lgd3D + 21.3*lgfc - 0.3*(h_UT-1.5);
        if isLOS
            PL = PL_LOS;
            sigma_SF = 4;
        else
            PL = max([PL_LOS, PL_prime]);
            sigma_SF = 7.82;
        end
        % derive pathloss for indoor
        if isIndoor
            if fc < 6e9 % table 7.4.3-3
                PL_tw = 20;
                PL_in = 0.5*d_2D_in;
                PL = PL + PL_tw + PL_in;
                sigma_SF = 7;
            else % table 7.4.3-2
                L_glass = 2 + 0.2*fc/1e9;
                L_concreate = 5 + 4*fc/1e9;
                L_IRRglass = 23 + 0.3*fc/1e9;
                O2I_model = 'low-pass'; % high/low-pass model
                if strcmp(O2I_model, 'low-pass')
                    PL_tw = 5 - 10*log10(0.3*10^(-L_glass/10) + ...
                        0.7*10^(-L_concreate/10));
                    PL_in = 0.5 * d_2D_in;
                    PL = PL + PL_tw + PL_in;
                    sigma_SF = 4.4;
                else
                    PL_tw = 5 - 10*log10(0.7*10^(-L_IRRglass/10) + ...
                        0.3*10^(-L_concreate/10));
                    PL_in = 0.5 * d_2D_in;
                    PL = PL + PL_tw + PL_in;
                    sigma_SF = 6.5;
                end
            end
        end
    case 'RMa'
        % derive d_BP
        d_BP = 2*pi*h_BS*h_UT*fc/lightSpeed;
        % derive pathloss for outdoor
        if d_2D < d_BP
            h = 5;
            PL_LOS = 20*log10(40*pi*d_3D*fc*1e-9/3) + min(0.03*h^1.72, 10)*lgd3D ...
                -min(0.044*h^1.72, 14.77) + 0.002*log10(h)*d_3D;
            sigma_SF = 4;
        elseif d_2D < 10000
            PL_LOS = 20*log10(40*pi*d_BP*fc*1e-9/3) + min(0.03*h^1.72, 10)*log10(d_BP) ...
                -min(0.044*h^1.72, 14.77) + 0.002*log10(h)*d_BP + 40*log10(d_3D/d_BP);
            sigma_SF = 6;
        else
            error('d_2D is not supported ...\n')
        end
        W = 20;
        PL_prime = 161.04 - 7.1*log10(W) + 7.5*log10(h) - (24.37 - 3.7*(h/h_BS)^2)*log10(h_BS) ...
            + (43.42-3.1*log10(h_BS))*(log10(d_3D)-3) + 20*lgfc ...
            - (3.2*(log10(11.75*h_UT))^2-4.97);
        if isLOS
            PL = PL_LOS;
        else
            PL = max([PL_LOS, PL_prime]);
            sigma_SF = 8;
        end
        % derive pathloss for indoor
        if isIndoor
            if fc < 6e9 % table 7.4.3-3
                PL_tw = 20;
                PL_in = 0.5*d_2D_in;
                PL = PL + PL_tw + PL_in;
                sigma_SF = 7;
            else % table 7.4.3-2
                L_glass = 2 + 0.2*fc/1e9;
                L_concreate = 5 + 4*fc/1e9;
                L_IRRglass = 23 + 0.3*fc/1e9;
                O2I_model = 'low-pass'; % only low-pass model
                if strcmp(O2I_model, 'low-pass')
                    PL_tw = 5 - 10*log10(0.3*10^(-L_glass/10) + ...
                        0.7*10^(-L_concreate/10));
                    PL_in = 0.5 * d_2D_in;
                    PL = PL + PL_tw + PL_in;
                    sigma_SF = 4.4;
                else
                    PL_tw = 5 - 10*log10(0.7*10^(-L_IRRglass/10) + ...
                        0.3*10^(-L_concreate/10));
                    PL_in = 0.5 * d_2D_in;
                    PL = PL + PL_tw + PL_in;
                    sigma_SF = 6.5;
                end
            end
        end
    otherwise
        error('scenario is not supported ...\n')
end

end        


function LSParamTable = genLSParamTable(scenario, fc, d_2D, h_UT)

% step 4: derive large scale parameters

switch scenario
    case 'UMa'
        % table 7.5-6
        if fc < 6e9 % Note 6
            fc = 6e9;
        end
        lgfc = log10(fc/1e9);
        LSParamTable.mu_lgDS = [-6.955-0.0963*lgfc, -6.28-0.204*lgfc];
        LSParamTable.sigma_lgDS = [0.66, 0.39];
        LSParamTable.mu_lgASD = [1.06+0.1114*lgfc, 1.5-0.1144*lgfc];
        LSParamTable.sigma_lgASD = [0.28, 0.28];
        LSParamTable.mu_lgASA = [1.81, 2.08-0.27*lgfc];
        LSParamTable.sigma_lgASA = [0.2, 0.11];
        LSParamTable.mu_lgZSA = [0.95, 1.512-0.3236*lgfc];
        LSParamTable.sigma_lgZSA = [0.16, 0.16];
        LSParamTable.mu_K = 9;
        LSParamTable.sigma_K = 3.5;
        LSParamTable.r_tao = [2.5, 2.3];
        LSParamTable.mu_XPR = [8, 7];
        LSParamTable.sigma_XPR = [4, 3];
        LSParamTable.nCluster = [12, 20];
        LSParamTable.nRayPerCluster = [20, 20];
        LSParamTable.C_DS = [max([0.25, 6.5622-3.4084*lgfc]), ...
            max([0.25, 6.5622-3.4084*lgfc])];
        LSParamTable.C_ASD = [5, 2];
        LSParamTable.C_ASA = [11, 15];
        LSParamTable.C_ZSA = [7, 7];
        LSParamTable.xi = [3, 3];
        % table 7.5-2
        LSParamTable.C_phi_NLOS = 1.146;
        % table 7.5-4
        LSParamTable.C_theta_NLOS = 1.104;
        % table 7.5-3
        LSParamTable.RayOffsetAngles = [0.0447, -0.0447, 0.1413, -0.1413, 0.2492, -0.2492, ...
            0.3715, -0.3715, 0.5129, -0.5129, 0.6797, -0.6797, 0.8844, -0.8844, ...
            1.1481, -1.1481, 1.5195, -1.5195, 2.1551, -2.1551];
        % table 7.5-7
        LSParamTable.mu_lgZSD = [max([-0.5, -2.1*(d_2D/1000)-0.01*(h_UT-1.5)+0.75]), ...
            max([-0.5, -2.1*(d_2D/1000)-0.01*(h_UT-1.5)+0.9])];
        LSParamTable.sigma_lgZSD = [0.4, 0.49];
        LSParamTable.mu_offset_ZOD = [0, 7.66*lgfc-5.96 ...
            - 10^((0.208*lgfc-0.782)*log10(max([25, d_2D]))...
            + (2.03-0.13*lgfc)-0.07*(h_UT-1.5))];      
    case 'UMi'
        % table 7.5-6
        if fc < 6e9 % Note 6
            fc = 6e9;
        end
        lg1fc = log10(1+fc/1e9);
        LSParamTable.mu_lgDS = [-7.14-0.24*lg1fc, -6.83-0.24*lg1fc];
        LSParamTable.sigma_lgDS = [0.38, 0.28+0.16*lg1fc];
        LSParamTable.mu_lgASD = [1.21-0.05*lg1fc, 1.53-0.23*lg1fc];
        LSParamTable.sigma_lgASD = [0.41, 0.33+0.11*lg1fc];
        LSParamTable.mu_lgASA = [1.73-0.08*lg1fc, 1.81-0.08*lg1fc];
        LSParamTable.sigma_lgASA = [0.28+0.014*lg1fc, 0.3+0.05*lg1fc];
        LSParamTable.mu_lgZSA = [0.73-0.1*lg1fc, 0.92-0.04*lg1fc];
        LSParamTable.sigma_lgZSA = [0.34-0.04*lg1fc, 0.41-0.07*lg1fc];
        LSParamTable.mu_K = 9;
        LSParamTable.sigma_K = 5;
        LSParamTable.r_tao = [3, 2.1];
        LSParamTable.mu_XPR = [9, 8];
        LSParamTable.sigma_XPR = [3, 3];
        LSParamTable.nCluster = [12, 19];
        LSParamTable.nRayPerCluster = [20, 20];
        LSParamTable.C_DS = [5, 11];
        LSParamTable.C_ASD = [3, 10];
        LSParamTable.C_ASA = [17, 22];
        LSParamTable.C_ZSA = [7, 7];
        LSParamTable.xi = [3, 3];
        % table 7.5-2
        LSParamTable.C_phi_NLOS = 1.146;
        % table 7.5-4
        LSParamTable.C_theta_NLOS = 1.104;
        % table 7.5-3
        LSParamTable.RayOffsetAngles = [0.0447, -0.0447, 0.1413, -0.1413, 0.2492, -0.2492, ...
            0.3715, -0.3715, 0.5129, -0.5129, 0.6797, -0.6797, 0.8844, -0.8844, ...
            1.1481, -1.1481, 1.5195, -1.5195, 2.1551, -2.1551];
        % table 7.5-8        
        h_BS = 10;
        LSParamTable.mu_lgZSD = [max([-0.21, -14.8*(d_2D/1000)-0.01*abs(h_UT-h_BS)+0.83]), ...
            max([-0.5, -3.1*(d_2D/1000)+0.01*max(h_UT-h_BS, 0)+0.2])];
        LSParamTable.sigma_lgZSD = [0.35, 0.35];
        LSParamTable.mu_offset_ZOD = [0, -10^(-1.5*log10(max([10, d_2D]))+3.3)];  
    case 'RMa'
        % table 7.5-6
        if fc < 6e9 % Note 6
            fc = 6e9;
        end
        lgfc = log10(fc/1e9);
        LSParamTable.mu_lgDS = [-7.49, -7.43];
        LSParamTable.sigma_lgDS = [0.55, 0.48];
        LSParamTable.mu_lgASD = [0.9, 0.95];
        LSParamTable.sigma_lgASD = [0.38, 0.45];
        LSParamTable.mu_lgASA = [1.52, 1.52];
        LSParamTable.sigma_lgASA = [0.24, 0.13];
        LSParamTable.mu_lgZSA = [0.47, 0.58];
        LSParamTable.sigma_lgZSA = [0.4, 0.37];
        LSParamTable.mu_K = 7;
        LSParamTable.sigma_K = 4;
        LSParamTable.r_tao = [3.8, 1.7];
        LSParamTable.mu_XPR = [12, 7];
        LSParamTable.sigma_XPR = [4, 3];
        LSParamTable.nCluster = [11, 10];
        LSParamTable.nRayPerCluster = [20, 20];
        LSParamTable.C_DS = [0, 0];
        LSParamTable.C_ASD = [2, 2];
        LSParamTable.C_ASA = [3, 3];
        LSParamTable.C_ZSA = [3, 3];
        LSParamTable.xi = [3, 3];
        % table 7.5-2
        LSParamTable.C_phi_NLOS = 1.146;
        % table 7.5-4
        LSParamTable.C_theta_NLOS = 1.104;
        % table 7.5-3
        LSParamTable.RayOffsetAngles = [0.0447, -0.0447, 0.1413, -0.1413, 0.2492, -0.2492, ...
            0.3715, -0.3715, 0.5129, -0.5129, 0.6797, -0.6797, 0.8844, -0.8844, ...
            1.1481, -1.1481, 1.5195, -1.5195, 2.1551, -2.1551];
        % table 7.5-9
        LSParamTable.mu_lgZSD = [max([-1, -0.17*(d_2D/1000)-0.01*(h_UT-1.5)+0.22]), ...
            max([-1, -0.19*(d_2D/1000)-0.01*(h_UT-1.5)+0.28])];
        LSParamTable.sigma_lgZSD = [0.34, 0.30];
        LSParamTable.mu_offset_ZOD = [0, atan((35-3.5)/d_2D)-atan((35-1.5)/d_2D)];          
    otherwise
        error('scenario is not supported ...\n')
end

end   


function [tao_n, tao_n_LOS] = genClusterDelay(nCluster, DS, r_tao, isLOS, K)

% step 5: generate cluster delays

global debugParam

for n = 1:nCluster
    tao_prime_n(n) = -r_tao*DS*log(rand(1));
    if debugParam.fixedSSP.enable
        tao_prime_n(n) = -r_tao*DS*log(n/nCluster);
    end
end
tao_n = sort(tao_prime_n);
tao_n = tao_n-tao_n(1);
if isLOS
    C_tao = 0.7705-0.0433*K+0.0002*K^2+0.000017*K^3;
    tao_n_LOS = tao_n/C_tao;
else
    tao_n_LOS = [];
end

end

function [P_n, P_n_LOS, idxWeakCluster] = genClusterPower(nCluster, tao_n, r_tao, DS, xi, isLOS, K)

% step 6: generate cluster power

global debugParam

for n = 1:nCluster
    Z_n(n) = xi * randn(1);
    if debugParam.fixedSSP.enable
        Z_n(n) = 0;
    end
    P_prime_n(n) = exp(-tao_n(n)*(r_tao-1)/(r_tao*DS))*10^(-Z_n(n)/10);
end
P_n = P_prime_n/sum(P_prime_n);

if isLOS
    K_R = 10^(K/10);
    P1_LOS = K_R/(K_R+1);
    P_n_LOS = 1/(K_R+1)*P_n;
    P_n_LOS(1) = P_n_LOS(1) + P1_LOS;    
else
    P_n_LOS = [];
end

% Discard clusters with power 25 dB lower than max power cluster
if isLOS
   P_thr = max(P_n_LOS) * 10^(-25/10);
   idxWeakCluster = find(P_n_LOS < P_thr);
   P_n_LOS(idxWeakCluster) = [];
else
   P_thr = max(P_n) * 10^(-25/10);
   idxWeakCluster = find(P_n < P_thr);
   P_n(idxWeakCluster) = [];
end

% TBD: discard clusters with power 25 dB lower than max power cluster
% P_thr = max(P_n) * 10^(-25/10);
% P_n = P_n .* (P_n > P_thr);

end

function [phi_n_AOA, phi_n_AOD, theta_n_ZOA, theta_n_ZOD] = ...
    genClusterAngle(nCluster, isLOS, K, P_n, C_phi_NLOS, C_theta_NLOS, ...
    ASA, phi_LOS_AOA, ASD, phi_LOS_AOD, ZSA, theta_LOS_ZOA, ...
    isIndoor, ZSD, theta_LOS_ZOD, mu_offset_ZOD)

global debugParam 

% step 7, generate cluster angles

% AOA
if isLOS
    C_phi = C_phi_NLOS*(1.1035-0.028*K-0.002*K^2+0.0001*K^3);
else
    C_phi = C_phi_NLOS;
end
phi_prime_AOA = 2*(ASA/1.4)*sqrt(-log(P_n/max(P_n)))/C_phi;

for n = 1:nCluster
    Xn = sign(rand(1)-0.5);
    Yn = ASA/7*randn(1);
    if debugParam.fixedSSP.enable
        Xn = (-1)^n;
        Yn = 0;
    end
    phi_n_AOA(n) = Xn*phi_prime_AOA(n) + Yn + phi_LOS_AOA;
end

if isLOS
    % phi_n_AOA = phi_n_AOA - phi_n_AOA(1);
    phi_n_AOA(1) = phi_LOS_AOA;
end

% AOD
phi_prime_AOD = 2*(ASD/1.4)*sqrt(-log(P_n/max(P_n)))/C_phi;
for n = 1:nCluster
    Xn = sign(rand(1)-0.5);
    Yn = ASD/7*randn(1);
    if debugParam.fixedSSP.enable
        Xn = (-1)^n;
        Yn = 0;
    end    
    phi_n_AOD(n) = Xn*phi_prime_AOD(n) + Yn + phi_LOS_AOD;
end

if isLOS
    % phi_n_AOD = phi_n_AOD - phi_n_AOD(1);
    phi_n_AOD(1) = phi_LOS_AOD;
end

% ZOA
if isLOS
    C_theta = C_theta_NLOS*(1.3086+0.0339*K-0.0077*K^2+0.0002*K^3);
else
    C_theta = C_theta_NLOS;
end
theta_prime_ZOA = -ZSA*log(P_n/max(P_n))/C_theta;
for n = 1:nCluster
    Xn = sign(rand(1)-0.5);
    Yn = ZSA/7*randn(1);
    if debugParam.fixedSSP.enable
        Xn = (-1)^n;
        Yn = 0;
    end    
    if isIndoor
        theta_bar_ZOA = 90; 
    else
        theta_bar_ZOA = theta_LOS_ZOA;
    end
    theta_n_ZOA(n) = Xn*theta_prime_ZOA(n) + Yn + theta_bar_ZOA;
end

if isLOS
    % theta_n_ZOA = theta_n_ZOA - theta_n_ZOA(1) + theta_LOS_ZOA;
    theta_n_ZOA(1) = theta_LOS_ZOA;
end

% ZOD
theta_prime_ZOD = -ZSD*log(P_n/max(P_n))/C_theta;
for n = 1:nCluster
    Xn = sign(rand(1)-0.5);
    Yn = ZSD/7*randn(1);
    if debugParam.fixedSSP.enable
        Xn = (-1)^n;
        Yn = 0;
    end    
    theta_n_ZOD(n) = Xn*theta_prime_ZOD(n) + Yn + theta_LOS_ZOD + mu_offset_ZOD;   
end

if isLOS
    % theta_n_ZOD = theta_n_ZOD - theta_n_ZOD(1) + theta_LOS_ZOD;
    theta_n_ZOD(1) = theta_LOS_ZOD;
end

end


function [phi_n_m_AOA, phi_n_m_AOD, theta_n_m_ZOA, theta_n_m_ZOD] = ...
    genRayAngle(nCluster, nRayPerCluster, phi_n_AOA, phi_n_AOD, ...
    theta_n_ZOA, theta_n_ZOD, C_ASA, C_ASD, C_ZSA, mu_lgZSD, ...
    RayOffsetAngles)

% step 8, generate ray angles and couple rays within a cluster

for n = 1:nCluster
    idxASD = randperm(nRayPerCluster);
    idxASA = randperm(nRayPerCluster);
    idxZSD = randperm(nRayPerCluster);
    idxZSA = randperm(nRayPerCluster);
    for m = 1:nRayPerCluster
        phi_n_m_AOA(n, m) = phi_n_AOA(n) + C_ASA * RayOffsetAngles(idxASA(m));
        phi_n_m_AOD(n, m) = phi_n_AOD(n) + C_ASD * RayOffsetAngles(idxASD(m));
        temp = theta_n_ZOA(n) + C_ZSA * RayOffsetAngles(idxZSA(m));
        if temp >= 180
            theta_n_m_ZOA(n, m) = 360-temp;
        else
            theta_n_m_ZOA(n, m) = temp;
        end
        temp = theta_n_ZOD(n) + (3/8)*10^mu_lgZSD * RayOffsetAngles(idxZSD(m));
        if temp >= 180
            theta_n_m_ZOD(n, m) = 360-temp;
        else
            theta_n_m_ZOD(n, m) = temp;
        end
    end
end

end

function kappa_n_m = genClusterXPR(nCluster, nRayPerCluster, mu_XPR, sigma_XPR)

% step 9: generate XPR

for n = 1:nCluster
    for m = 1:nRayPerCluster
        X_n_m = sigma_XPR*randn(1) + mu_XPR;
        kappa_n_m(n, m) = 10^(X_n_m/10);
    end
end

end

function [mAnt, nAnt, pAnt] = findAntLoc(u, AntSize)

M = AntSize(1);
N = AntSize(2);
P = AntSize(3);

u = u-1;
pAnt = mod(u, P)+1;
nAnt = mod(floor(u/P), N)+1;
mAnt = mod(floor(u/(P*N)), M)+1;

end

function A_dB_3D = calc_A_dB_3D(theta, phi)

theta_3dB = 65;
SLA_v = 30;

theta = (theta<0).*(-theta) + (theta>0).*theta;
theta = (theta>180).*(360-theta) + (theta<=180).*theta;
% if theta < 0
%     % warning('theta < 0 \n');
%     theta = -theta;
% elseif theta > 180
%     % warning('theta > 180 \n');
%     theta = 360-theta;
% end
A_dB_theta = -min(12*((theta-90)/theta_3dB).^2, SLA_v);

phi_3dB = 65;
A_max = 30;
phi = (phi>180).*(phi-360) + (phi<=180).*phi;
phi = (phi<-180).*(phi+360) + (phi>=-180).*phi;
% if phi > 180
%     % warning('phi > 180 \n');
%     phi = phi - 360;
% elseif phi < -180
%     % warning('phi < -180 \n');
%     phi = phi + 360;
% end
A_dB_phi = -min(12*(phi/phi_3dB).^2, A_max);

A_dB_3D = -min(-(A_dB_theta + A_dB_phi), A_max);

end

function [F_theta, F_phi] = calc_Field(antPattern, theta, phi, zeta)

switch(antPattern)
    case 'directional'
        G_E_max = 8; 
        A_dB_3D = G_E_max + calc_A_dB_3D(theta, phi);
        A = 10.^(A_dB_3D/10);
    case 'isotropic'
        A = ones(size(theta));
    otherwise
        error('antPettern is not supported ...\n')
end

F_theta = sqrt(A)*cos(zeta*pi/180);
F_phi = sqrt(A)*sin(zeta*pi/180);

end

function [maxValue, maxIndex] = findMax(A, n)

% find the max n elements values and indices for vector A

[B, I] = sort(A, 'descend'); 
maxValue = B(1:n);
maxIndex = I(1:n);

end

function P_n_updated = insertClusterPower(P_n, strongest2Clusters)

% replace the power of 2 strongest clusters with 6 sub-clusters

if isempty(strongest2Clusters)
    P_n_updated = P_n;
else
    idx1 = min(strongest2Clusters);
    idx2 = max(strongest2Clusters);

    P_n_updated = [P_n(1:idx1-1), P_n(idx1)*10/20, P_n(idx1)*6/20, ...
        P_n(idx1)*4/20, P_n(idx1+1:idx2-1), P_n(idx2)*10/20, P_n(idx2)*6/20, ...
        P_n(idx2)*4/20, P_n(idx2+1:end)];
end

end



function tao_n_updated = insertClusterDelay(tao_n, strongest2Clusters, C_DS)

% replace the delay of 2 strongest clusters with 6 sub-clusters

if isempty(strongest2Clusters)
    tao_n_updated = tao_n;
else
    idx1 = min(strongest2Clusters);
    idx2 = max(strongest2Clusters);

    tao_n_updated = [tao_n(1:idx1-1), tao_n(idx1), tao_n(idx1) + 1.28*C_DS, ...
        tao_n(idx1) + 2.56*C_DS, tao_n(idx1+1:idx2-1), tao_n(idx2), ...
        tao_n(idx2) + 1.28*C_DS, tao_n(idx2) + 2.56*C_DS, tao_n(idx2+1:end)];
end

end

function LinkLSParam = genLinkLSParam(scenario, fc, UT, Link)

global debugParam

% Step-4: Generate LSP for links to one site

% 38.901 table 7
switch scenario
    case 'UMa'
        CorrDist_DS = [30 40];
        CorrDist_ASD = [18 50];
        CorrDist_ASA = [15 50];
        CorrDist_SF = [37 50];
        CorrDist_K = [12];
        CorrDist_ZSA = [15 50];
        CorrDist_ZSD = [15 50];
        sqrt_C_LOS = chol(...
        [   1    0 -0.4 -0.5 -0.5    0 -0.8;...
            0    1 -0.4    0 -0.2    0    0;...
         -0.4 -0.4    1  0.4  0.8 -0.2    0;...
         -0.5    0  0.4    1    0  0.5    0;...
         -0.5 -0.2  0.8    0    1 -0.3  0.4;...
            0    0 -0.2  0.5 -0.3    1    0;...
         -0.8    0    0    0  0.4    0    1]); 
        sqrt_C_NLOS = chol(...
        [   1 -0.4 -0.6 -0.4    0    0;...
         -0.4    1  0.4  0.6 -0.5    0;...
         -0.6  0.4    1  0.4  0.5 -0.1;...
         -0.4  0.6  0.4    1    0    0;...
            0 -0.5  0.5    0    1    0;...
            0    0 -0.1    0    0    1]);        
    case 'UMi'
        CorrDist_DS = [7 10];
        CorrDist_ASD = [8 10];
        CorrDist_ASA = [8 9];
        CorrDist_SF = [10 13];
        CorrDist_K = [15];
        CorrDist_ZSA = [12 10];
        CorrDist_ZSD = [12 10];
        sqrt_C_LOS = chol(...
        [   1  0.5 -0.4 -0.5 -0.4    0    0;...
          0.5    1 -0.7 -0.2 -0.3    0    0;...
         -0.4 -0.7    1  0.5  0.8    0  0.2;...
         -0.5 -0.2  0.5    1  0.4  0.5  0.3;...
         -0.4 -0.3  0.8  0.4    1    0    0;...
            0    0    0  0.5    0    1    0;...
            0    0  0.2  0.3    0    0    1]); 
        sqrt_C_NLOS = chol(...
        [   1 -0.7    0 -0.4    0    0;...
         -0.7    1    0  0.4 -0.5    0;...
            0    0    1    0  0.5  0.5;...
         -0.4  0.4    0    1    0  0.2;...
            0 -0.5  0.5    0    1    0;...
            0    0  0.5  0.2    0    1]);      
    case 'RMa'
        CorrDist_DS = [50 36];
        CorrDist_ASD = [25 30];
        CorrDist_ASA = [35 40];
        CorrDist_SF = [37 120];
        CorrDist_K = [40];
        CorrDist_ZSA = [15 50];
        CorrDist_ZSD = [15 50];
        sqrt_C_LOS = chol(...
        [   1     0  -0.5     0     0  0.01 -0.17;...
            0     1     0     0     0     0 -0.02;...
         -0.5     0     1     0     0 -0.05  0.27;...
            0     0     0     1     0  0.73 -0.14;...
            0     0     0     0     1  -0.2  0.24;...
         0.01     0 -0.05  0.73 -0.12     1 -0.07;...
        -0.17 -0.02  0.27 -0.14  0.24 -0.07     1]); 
        sqrt_C_NLOS = chol(...
        [   1 -0.5   0.6     0  -0.04  -0.25;...
         -0.5    1  -0.4     0   -0.1   -0.4;...
          0.6 -0.4     1     0   0.42  -0.27;...
            0    0     0     1  -0.18   0.26;...
        -0.04 -0.1  0.42 -0.18      1  -0.27;...
        -0.25 -0.4 -0.27  0.26  -0.27      1]);          
    otherwise
        error('scenario is not supported ...\n')
end

nUt = length(UT);

[max_X, min_X, max_Y, min_Y] = findUtBoundary(UT);

for idxLOS = 1:2
    CRN_DS(idxLOS, :, :) = genCRN(max_X, min_X, max_Y, min_Y, CorrDist_DS(idxLOS));
    CRN_ASD(idxLOS, :, :) = genCRN(max_X, min_X, max_Y, min_Y, CorrDist_ASD(idxLOS));
    CRN_ASA(idxLOS, :, :) = genCRN(max_X, min_X, max_Y, min_Y, CorrDist_ASA(idxLOS));
    CRN_SF(idxLOS, :, :) = genCRN(max_X, min_X, max_Y, min_Y, CorrDist_SF(idxLOS));
    if idxLOS == 1
        CRN_K(idxLOS, :, :) = genCRN(max_X, min_X, max_Y, min_Y, CorrDist_K(idxLOS));
    end
    CRN_ZSA(idxLOS, :, :) = genCRN(max_X, min_X, max_Y, min_Y, CorrDist_ZSA(idxLOS));
    CRN_ZSD(idxLOS, :, :) = genCRN(max_X, min_X, max_Y, min_Y, CorrDist_ZSD(idxLOS));
end

for idxUt = 1:nUt
    isLOS = Link(idxUt).isLOS;
    if isLOS
        idxLOS = 1;
    else
        idxLOS = 2;
    end
    idx_X = UT(idxUt).loc(1) - min_X + 1;
    idx_Y = UT(idxUt).loc(2) - min_Y + 1;
    UT_DS = CRN_DS(idxLOS, idx_X, idx_Y);
    UT_ASD = CRN_ASD(idxLOS, idx_X, idx_Y);
    UT_ASA = CRN_ASA(idxLOS, idx_X, idx_Y);
    UT_SF = CRN_SF(idxLOS, idx_X, idx_Y);
    if isLOS
        UT_K = CRN_K(idxLOS, idx_X, idx_Y);
    end
    UT_ZSA = CRN_ZSA(idxLOS, idx_X, idx_Y);
    UT_ZSD = CRN_ZSD(idxLOS, idx_X, idx_Y);
    if isLOS
        xi = [UT_SF, UT_K, UT_DS, UT_ASD, UT_ASA, UT_ZSD, UT_ZSA]';
        s_tilda = (sqrt_C_LOS*xi);
    else
        xi = [UT_SF, UT_DS, UT_ASD, UT_ASA, UT_ZSD, UT_ZSA]';
        s_tilda = (sqrt_C_NLOS*xi);
    end

    d_2D = Link(idxUt).d_2D;
    h_UT = UT(idxUt).loc(3);
    sigma_SF = Link(idxUt).sigma_SF;
    LSParamTable = genLSParamTable(scenario, fc, d_2D, h_UT);
    if isLOS
        mu = [0, LSParamTable.mu_K(idxLOS), LSParamTable.mu_lgDS(idxLOS), LSParamTable.mu_lgASD(idxLOS), ...
            LSParamTable.mu_lgASA(idxLOS), LSParamTable.mu_lgZSD(idxLOS), LSParamTable.mu_lgZSA(idxLOS)]';
        sigma = [sigma_SF, LSParamTable.sigma_K(idxLOS), LSParamTable.sigma_lgDS(idxLOS), LSParamTable.sigma_lgASD(idxLOS), ...
            LSParamTable.sigma_lgASA(idxLOS), LSParamTable.sigma_lgZSD(idxLOS), LSParamTable.sigma_lgZSA(idxLOS)]';
        s = s_tilda .* sigma + mu;
        if debugParam.fixedLSP.enable
            s = mu;
        end
        LinkLSParam(idxUt, 1:2) = s(1:2);
        LinkLSParam(idxUt, 3:7) = 10.^s(3:7);
    else
        mu = [0, LSParamTable.mu_lgDS(idxLOS), LSParamTable.mu_lgASD(idxLOS), ...
            LSParamTable.mu_lgASA(idxLOS), LSParamTable.mu_lgZSD(idxLOS), LSParamTable.mu_lgZSA(idxLOS)]';
        sigma = [sigma_SF, LSParamTable.sigma_lgDS(idxLOS), LSParamTable.sigma_lgASD(idxLOS), ...
            LSParamTable.sigma_lgASA(idxLOS), LSParamTable.sigma_lgZSD(idxLOS), LSParamTable.sigma_lgZSA(idxLOS)]';
        s = s_tilda .* sigma + mu;
        if debugParam.fixedLSP.enable
            s = mu;
        end        
        LinkLSParam(idxUt, 1) = s(1);
        LinkLSParam(idxUt, 2) = 0;
        LinkLSParam(idxUt, 3:7) = 10.^s(2:6);        
    end
end

LinkLSParam(:, 4) = min(LinkLSParam(:, 4), 104);
LinkLSParam(:, 5) = min(LinkLSParam(:, 5), 104);
LinkLSParam(:, 6) = min(LinkLSParam(:, 6), 52);
LinkLSParam(:, 7) = min(LinkLSParam(:, 7), 52);

end

function CRN = genCRN(max_X, min_X, max_Y, min_Y, CorrDist)

% generate correlated random numbers
% WINNER II - 3.3.1

D = 3*CorrDist;
nX = round(max_X - min_X + 1 + 2 * D);
nY = round(max_Y - min_Y + 1 + 2 * D);

% exponential correlation filter
if CorrDist == 0
    h = 1;
else
    h = exp(-abs(-D:D)/CorrDist);
    % h = h/sqrt(sum(h.^2)); % power normalization
end

% generate gaussian noise
gn = randn(nX, nY);

% 2-dimensional convolution
CRN1 = conv2(h, h, gn);

% remove head and tail
CRN2 = CRN1(2*D+1:end-2*D, 2*D+1:end-2*D);

% power normalization
CRN = CRN2/sqrt(mean(abs(CRN2(:)).^2));

end

function [max_X, min_X, max_Y, min_Y] = findUtBoundary(UT)

nUt = length(UT);

max_X = -1e5;
min_X = 1e5;
max_Y = -1e5;
min_Y = 1e5;
for idxUt = 1:nUt
    if UT(idxUt).loc(1) > max_X
        max_X = UT(idxUt).loc(1);
    end
    if UT(idxUt).loc(1) < min_X
        min_X = UT(idxUt).loc(1);
    end
    if UT(idxUt).loc(2) > max_Y
        max_Y = UT(idxUt).loc(2);
    end
    if UT(idxUt).loc(2) < min_Y
        min_Y = UT(idxUt).loc(2);
    end    
end

end


function [timeSeq, Nbatch, NbatchSamp] = genChanBatchParam(N_frame)

f_samp = 30e3*4096; % sampling rate = delta_f * Nfft
T_frame = 10e-3; % frame duation = 10 ms
lenSamp = N_frame*T_frame*f_samp; % number of samples

% update rate of quasi-static channel
% 30e3 is 100x larger than max doppler freq (300Hz). It should be fast
% enough to capture the channel variation over time
f_batch = 30e3; % batch (channel realization) frequency (rate)
NbatchSamp = round(f_samp/f_batch); 
Nbatch = ceil(lenSamp/NbatchSamp); % number of channel realization
timeSeq = [0:Nbatch-1]/f_samp*NbatchSamp;

end

function  [Hs, nTaps] = convertChannel(H, tao, pw, nC, f_samp)

[nUt, nSite, nSecPerSite, nUtAnt, nBsAnt, maxCluster, nBatch] = size(H);

maxDelay = max(tao(:));
maxTap = round(maxDelay*f_samp)+1;

Hs = zeros(nUt, nSite, nSecPerSite, nUtAnt, nBsAnt, maxTap, nBatch);

for idxUt = 1:nUt
    for idxSite = 1:nSite
        nCluster = nC(idxUt, idxSite);
        thisTao = squeeze(tao(idxUt, idxSite, 1:nCluster));
        maxDelay = max(thisTao);
        maxTap = round(maxDelay*f_samp)+1;
        for idxCluster = 1:nCluster
            idxTap = round(thisTao(idxCluster)*f_samp) + 1;
            Hs(idxUt, idxSite, :, :, :, idxTap, :) = ...
                H(idxUt, idxSite, :, :, :, idxCluster, :) + ...
                Hs(idxUt, idxSite, :, :, :, idxTap, :);
        end
        nTaps(idxUt, idxSite) = maxTap;
    end
end

end
