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

function Chan = genCDL(Chan, SysPar)
%
% Generate time domain fading channel filters based on TS 38.901
% The fading channel is quasi-static over a number of samples 
% instead of changing over every sample.

% factor coverting degree to radian
d2pi = pi/180;

DelayProfile = Chan.DelayProfile;
DelaySpread = Chan.DelaySpread;
dopplerHz = Chan.MaximumDopplerShift;
f_samp = Chan.f_samp;
T_samp = Chan.T_samp;
N_frame = SysPar.SimCtrl.N_frame;
lenSamp = N_frame*10e-3*f_samp;

UeAntSize = Chan.UE_AntArraySize;
UeAntSpacing = Chan.UE_AntSpacing; 
UeAntPolarAngles = Chan.UE_AntPolarizationAngles; 
UeAntPattern = Chan.UE_AntPattern; 

BsAntSize = Chan.gNB_AntArraySize;
BsAntSpacing = Chan.gNB_AntSpacing; 
BsAntPolarAngles = Chan.gNB_AntPolarizationAngles; 
BsAntPattern = Chan.gNB_AntPattern; 

% BsAntOrientation = [0 0 0];
% UeAntOrientation = [0 0 0];
v_direction = [90, 0];

switch DelayProfile
    case 'CDL-A' 
        load CDLparam;
        DPA = CDLparam.DPA_A;
        PCP = CDLparam.PCP_A;
    case 'CDL-B'
        load CDLparam;
        DPA = CDLparam.DPA_B;
        PCP = CDLparam.PCP_B;
    case 'CDL-C'
        load CDLparam;
        DPA = CDLparam.DPA_C;
        PCP = CDLparam.PCP_C;
    case 'CDL-D'
        load CDLparam;
        DPA = CDLparam.DPA_D;
        PCP = CDLparam.PCP_D;
    case 'CDL-E'
        load CDLparam;
        DPA = CDLparam.DPA_E;
        PCP = CDLparam.PCP_E;
    case 'CDL_customized'
        DPA = Chan.CDL_DPA;
        PCP = Chan.CDL_PCP;
    otherwise 
        error('chanType is not supported ... \n');
end
% swapTransmitAndReceive(cdl); % 38.901 CDL parameter table is based on DL

% set delay/power/angle
PathDelays = DPA(:,1)';
AveragePathGains = DPA(:,2)';
AnglesAoD = DPA(:,4)';
AnglesAoA = DPA(:,3)';
AnglesZoD = DPA(:,6)';
AnglesZoA = DPA(:,5)';
% set per-cluster parameters
c_ASD = PCP(2);
c_ASA = PCP(1);
c_ZSD = PCP(4);
c_ZSA = PCP(3);
XPR = PCP(5);

% 38.901 table 7.5-3
RayOffsetAngles = [0.0447, -0.0447, 0.1413, -0.1413, 0.2492, -0.2492, ...
    0.3715, -0.3715, 0.5129, -0.5129, 0.6797, -0.6797, 0.8844, -0.8844, ...
    1.1481, -1.1481, 1.5195, -1.5195, 2.1551, -2.1551];

PathDelays = PathDelays*DelaySpread;

numBsAnt = prod(BsAntSize);
numUeAnt = prod(UeAntSize);
numClusters = length(PathDelays);
numRays = 20;

% Quantize to FIR filter with T_samp interval
firTapMap = round(PathDelays/T_samp) + 1;  % map to the same tap dure to sampling
firPw = 10.^(AveragePathGains/10);
lenFir = round(PathDelays(end)/T_samp)+1;
firPw = firPw/sum(firPw);  % normalize PDP

% update rate of quasi-static channel
% 15e3 is 50x larger than max doppler freq (300Hz). It should be fast
% enough to capture the channel variation over time
f_batch = 15e3; 
NbatchSamp = round(f_samp / f_batch); 
Nbatch = ceil(lenSamp / NbatchSamp); % Number of channel realizations
timeSeq = [0:Nbatch-1] * T_samp * NbatchSamp;

H = zeros(numUeAnt, numBsAnt, lenFir, Nbatch);
PHI = 2 * pi * (rand(numClusters, numRays, 4) - 0.5);

% Precompute ray coupling indices
rayCoupling = zeros(4, lenFir, numRays);
for i = 1:4  % ASD ASA ZSD ZSA
    for firNzIdx = 1:lenFir
        rayCoupling(i, firNzIdx, :) = randperm(numRays); % Generate a random permutation for each row
    end
end

% Precompute antenna locations and polar angles
ueAntLocs = zeros(3, numUeAnt); % [x; y; z] for each Rx antenna
bsAntLocs = zeros(3, numBsAnt); % [x; y; z] for each Tx antenna
ueAntPolarAngles = zeros(1, numUeAnt); % Polar angles for Rx antennas
bsAntPolarAngles = zeros(1, numBsAnt); % Polar angles for Tx antennas

for u = 1:numUeAnt
    [mAnt, uAnt, pAnt] = findAntLoc(u, UeAntSize);
    ueAntLocs(:, u) = [0; (uAnt-1) * UeAntSpacing(2); (mAnt-1) * UeAntSpacing(1)];
    ueAntPolarAngles(u) = UeAntPolarAngles(pAnt);
end

for s = 1:numBsAnt
    [mAnt, uAnt, pAnt]  = findAntLoc(s, BsAntSize);
    bsAntLocs(:, s) = [0; (uAnt-1) * BsAntSpacing(2); (mAnt-1) * BsAntSpacing(1)];
    bsAntPolarAngles(s) = BsAntPolarAngles(pAnt);
end

% Precompute inverse square root of kappa for XPR
kappa_inv_sqrt = sqrt(1 / (10^(XPR / 10)));

% Main loop, generate CDL channel coefficients based on 38.901 7.7.1 
for u = 1:numUeAnt
    d_bar_rx_u = ueAntLocs(:, u);
    zetaUeAnt = ueAntPolarAngles(u);

    for s = 1:numBsAnt
        d_bar_tx_s = bsAntLocs(:, s);
        zetaBsAnt = bsAntPolarAngles(s);

        for n = 1:numClusters
            H_u_s_n = zeros(1, Nbatch);

            % Precompute angles for the cluster
            phi_n_AOD = AnglesAoD(n);
            phi_n_AOA = AnglesAoA(n);
            theta_n_ZOD = AnglesZoD(n);
            theta_n_ZOA = AnglesZoA(n);

            % Extract ray coupling indices for the cluster
            idxASD = squeeze(rayCoupling(1, n, :));
            idxASA = squeeze(rayCoupling(2, n, :));
            idxZSD = squeeze(rayCoupling(3, n, :));
            idxZSA = squeeze(rayCoupling(4, n, :));

            % Loop over rays within the cluster
            for m = 1:numRays           
                % Compute angles for the current ray
                phi_n_m_AOD = phi_n_AOD + c_ASD * RayOffsetAngles(idxASD(m));
                phi_n_m_AOA = phi_n_AOA + c_ASA * RayOffsetAngles(idxASA(m));
                theta_n_m_ZOD = theta_n_ZOD + c_ZSD * RayOffsetAngles(idxZSD(m));
                theta_n_m_ZOA = theta_n_ZOA + c_ZSA * RayOffsetAngles(idxZSA(m));

                % Compute field components and terms
                [F_rx_u_theta, F_rx_u_phi] = calc_Field(UeAntPattern, theta_n_m_ZOA, phi_n_m_AOA, zetaUeAnt);
                term1 = [F_rx_u_theta; F_rx_u_phi]';

                PHI_4 = squeeze(PHI(n, m, :));
                term2 = [exp(1j * PHI_4(1)), kappa_inv_sqrt * exp(1j * PHI_4(2)); ...
                         kappa_inv_sqrt * exp(1j * PHI_4(3)), exp(1j * PHI_4(4))];

                [F_tx_s_theta, F_tx_s_phi] = calc_Field(BsAntPattern, theta_n_m_ZOD, phi_n_m_AOD, zetaBsAnt);
                term3 = [F_tx_s_theta; F_tx_s_phi];

                r_head_rx_n_m = [sin(theta_n_m_ZOA*d2pi)*cos(phi_n_m_AOA*d2pi); ...
                                 sin(theta_n_m_ZOA*d2pi)*sin(phi_n_m_AOA*d2pi); ...
                                 cos(theta_n_m_ZOA*d2pi)];
                term4 = exp(1j * 2 * pi * r_head_rx_n_m' * d_bar_rx_u);

                r_head_tx_n_m = [sin(theta_n_m_ZOD*d2pi)*cos(phi_n_m_AOD*d2pi); ...
                                 sin(theta_n_m_ZOD*d2pi)*sin(phi_n_m_AOD*d2pi); ...
                                 cos(theta_n_m_ZOD*d2pi)];
                term5 = exp(1j * 2 * pi * r_head_tx_n_m' * d_bar_tx_s);

                % v_bar = v_speed*[sin(v_direction(1))*cos(v_direction(2)), sin(v_direction(1))*sin(v_direction(2)), cos(v_direction(1))]';
                % term6 = exp(j*2*pi*(r_head_rx_n_m'*v_bar/lambda_0)*timeSeq);
                v_head_rx = [sin(v_direction(1)*d2pi)*cos(v_direction(2)*d2pi), ...
                             sin(v_direction(1)*d2pi)*sin(v_direction(2)*d2pi), ...
                             cos(v_direction(1)*d2pi)]';
                term6 = exp(1j * 2*pi*(r_head_rx_n_m' * v_head_rx) * dopplerHz .* timeSeq);

                % Compute and accumulate channel coefficients in one step
                H_u_s_n = H_u_s_n + term1 * term2 * term3 * term4 * term5 * term6; % equation 7.5-22 in 38.901
            end

            % Update channel matrix
            H(u, s, firTapMap(n), :) = H(u, s, firTapMap(n), :) + ...
                                       reshape(sqrt(firPw(n)) / sqrt(numRays) * H_u_s_n, [1, 1, 1, Nbatch]);
        end
    end
end

% swap to ensure dimensions [Nin, Nout, lenFir, Nbatch]
% H was created as (numUeAnt, numBsAnt, lenFir, Nbatch);
if strcmp(Chan.link_direction, 'Downlink')
    Chan.chanMatrix = permute(H, [2, 1, 3, 4]);
else
    Chan.chanMatrix = H;
end
Chan.lenFir = lenFir;
Chan.NbatchSamp = NbatchSamp;

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
A_dB_theta = -min(12*((theta-90)/theta_3dB)^2, SLA_v);

phi_3dB = 65;
A_max = 30;
A_dB_phi = -min(12*(phi/phi_3dB)^2, A_max);

A_dB_3D = -min(-(A_dB_theta + A_dB_phi), A_max);

end

function [F_theta, F_phi] = calc_Field(antPattern, theta, phi, zeta)

switch(antPattern)
    case '38.901'
        G_E_max = 8; 
        A_dB_3D = G_E_max + calc_A_dB_3D(theta, phi);
        A = 10^(A_dB_3D/10);
    case 'isotropic'
        A = 1;
    otherwise
        error('antPettern is not supported ...\n')
end

F_theta = sqrt(A)*cos(zeta*pi/180);
F_phi = sqrt(A)*sin(zeta*pi/180);

end
