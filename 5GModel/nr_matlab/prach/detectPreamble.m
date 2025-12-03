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

function prach  = detectPreamble(prach, carrier, SimCtrl)
%
% detect preamble index 
%

genTV = SimCtrl.prach.genTV;
if genTV
    TVname = SimCtrl.prach.TVname;
end

y_uv_rx = prach.y_uv_rx;
N_CS = prach.N_CS;
K = prach.K;
y_u_ref = prach.y_u_ref;
u_ref = prach.u_ref;
C_v_ref = prach.C_v_ref;
uCount = prach.uCount;
L_RA = prach.L_RA; 
Nrep = prach.NrepRx;
mu = carrier.mu;
delta_f_RA = prach.delta_f_RA;
N_nc = prach.N_nc;


if L_RA == 839
    Nfft = 1024;
else
    Nfft = 256;
end
prach.Nfft = Nfft;

% One sample in L_RA scale equals to Nfft/L_RA samples in Nfft scale
zoneSearchGap = 1*floor(Nfft/L_RA);    

% Set preamble detection SNR threshold
% Test with preamble format 0 and B4, mu = 0 and 1. 
% May need optimization for other combinations.
if L_RA == 839
    if mu == 0
        thr0 = 7.5; 
        if uCount > 1
            thr0 = 10;
        end
    elseif mu == 1
        thr0 = 6;
        if uCount > 1
            thr0 = 8.5;
        end
    end
else
    if mu == 0
        thr0 = 9;
        if N_nc == 2
            thr0 = 6;
        end
    elseif mu == 1
        thr0 = 7.5;
        if N_nc == 2
            thr0 = 5;
        end
    end
end

prmbCount = 0;
for idxU = 0:uCount-1
    % load reference preamble in freq domain for one u
    y_u0_tx = y_u_ref(idxU+1, :);
    u1 = u_ref(prmbCount+1);
    Nzone = sum(u1 == u_ref);
    [Nant, Nsamp] = size(y_uv_rx);
    Nsamp = Nsamp/Nrep;
    for idxAnt = 1:Nant
        y_uv_rx_rep = y_uv_rx(idxAnt,:);
        y_uv_rx_mean = zeros(1, Nsamp);
        % average for repeatitive preamble
        rep_start = 1;        
        for idxNc = 1:N_nc
            y_uv_rx_mean = zeros(1, Nsamp);
            if idxNc < N_nc
                step = floor(Nrep/N_nc);
            else
                step = Nrep - (N_nc-1)*floor(Nrep/N_nc);
            end
            for idxRep = rep_start:rep_start+step-1
                y_uv_rx_mean = y_uv_rx_mean + ...
                    y_uv_rx_rep(Nsamp*(idxRep-1)+1:Nsamp*idxRep);
            end
            y_uv_rx_mean = y_uv_rx_mean/step;
            rep_start = rep_start + step;
            % multiplication in freq domain
            z_u = y_uv_rx_mean.*conj(y_u0_tx);
            
            % convert to time domain and calculate power of each sample
            pdp_nc(idxNc, :) = abs(ifft(z_u, Nfft)*(Nfft/L_RA)).^2;
        end
        pdp = mean(pdp_nc, 1);
        % Right shift zoneSearchGap samples to avoid misdetection of the
        % strongest path at beginning of zone(k) to zone(k-1)
        pdp= [pdp(end-zoneSearchGap+1:end), pdp(1:end-zoneSearchGap)];
        for idxZone = 0:Nzone-1
            
            % find each zone's location
            zone_start = mod(Nfft-ceil(C_v_ref(prmbCount+idxZone+1)...
                *Nfft/L_RA), Nfft)+1;
            zone_end = zone_start + floor(N_CS*Nfft/L_RA);
            
            % calculate each zone's mean/max power and stronges path location
            pdp_zone = pdp(zone_start:zone_end);
            pdp_zone_power(idxAnt, idxZone+1) = mean(pdp_zone);
            [pdp_zone_max(idxAnt, idxZone+1), ...
                pdp_max_loc(idxAnt, idxZone+1)] = max(pdp_zone);
            pdp_max_loc(idxAnt, idxZone+1) = ...
                pdp_max_loc(idxAnt, idxZone+1)-zoneSearchGap-1;
        end % idxZone = 0:Nzone-1
        
    end % idxAnt = 1:Nant
    
    for idxZone = 0:Nzone-1
        pdp_zone_power_mean(prmbCount+idxZone+1) = ...
            mean(pdp_zone_power(:, idxZone+1));
        pdp_zone_max_mean(prmbCount+idxZone+1) = ...
            mean(pdp_zone_max(:, idxZone+1));
        [~, maxIdx] = max(pdp_zone_max(:, idxZone+1));
        pdp_max_loc_mean(prmbCount+idxZone+1) = ...
            pdp_max_loc(maxIdx, idxZone+1);                  
    end
  
    prmbCount = prmbCount + Nzone;
    if prmbCount >= 64
        break;
    end
end

% estimate noise floor
np1 = mean(pdp_zone_power_mean);
thr1 = thr0*np1;
idxNoiseZone = find(pdp_zone_max_mean < thr1);
np2 = mean(pdp_zone_power_mean(idxNoiseZone));
thr2 = max(np2*thr0, 1e-2); % to reduce false alarm

if SimCtrl.prach.plotFigure
    figure; plot(10*log10(pdp_zone_max_mean)); grid on;
end   

detIdx = 0;
Nprmb = 64;
for prmbIdx = 0:Nprmb-1
    if pdp_zone_max_mean(prmbIdx+1) > thr2
        detIdx = detIdx + 1;
        peak_det(detIdx) = pdp_zone_max_mean(prmbIdx+1);
        prmbIdx_det(detIdx) = prmbIdx;  
        delay_samp_det(detIdx) = max(0, pdp_max_loc_mean(prmbIdx+1));
    end
end

prach.pdp_zone_max_mean = pdp_zone_max_mean;
prach.pdp_max_loc_mean = pdp_max_loc_mean;
prach.thr2 = thr2;
prach.thr0 = thr0;
prach.detIdx = detIdx;


if detIdx > 0    
    snr_det = 10*log10(peak_det/np2)-10*log10(L_RA);
    prach.peak_det = peak_det;
    prach.prmbIdx_det = prmbIdx_det;
    prach.delay_samp_det = delay_samp_det;
    delay_time_det = delay_samp_det/(Nfft*delta_f_RA);
    prach.delay_time_det = delay_time_det;
    prach.snr_det = snr_det;
else
    prach.peak_det = 0;
    prach.prmbIdx_det = -1;
    prach.delay_samp_det = 0;
    prach.delay_time_det = 0;
    prach.snr_det = 100;
    
end

if genTV
    saveTV_prach(TVname, prach, carrier);
end

return

function saveTV_prach(TVname, prach, carrier)

tvDirName = 'TV_prach';
if ~exist(tvDirName, 'dir')
    mkdir(tvDirName);
end

% save input
y_uv_rx = prach.y_uv_rx;
y_u_ref = prach.y_u_ref;
prachParams.N_CS = uint32(prach.N_CS);
prachParams.uCount = uint32(prach.uCount);
prachParams.L_RA = uint32(prach.L_RA);
prachParams.N_rep = uint32(prach.NrepRx);
prachParams.mu = uint32(carrier.mu);
prachParams.delta_f_RA = uint32(prach.delta_f_RA);
prachParams.thr0 = single(prach.thr0);
[Nant, ~] = size(y_uv_rx);
prachParams.N_ant = uint32(Nant);
prachParams.Nfft = uint32(prach.Nfft);
prachParams.N_nc = uint32(prach.N_nc);

h5File  = H5F.create([tvDirName filesep TVname '.h5'], 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');
hdf5_write_nv(h5File, 'y_uv_rx', single(y_uv_rx.'));
hdf5_write_nv(h5File, 'y_u_ref', single(y_u_ref.')); 
hdf5_write_nv(h5File, 'prachParams', prachParams);

% save output
prmbIdx_det = prach.prmbIdx_det;
delay_time_det = prach.delay_time_det;
detIdx = prach.detIdx;
peak_det = prach.peak_det;
hdf5_write_nv(h5File, 'detIdx', uint32(detIdx));
hdf5_write_nv(h5File, 'prmbIdx_det', uint32(prmbIdx_det)); 
hdf5_write_nv(h5File, 'delay_time_det', single(delay_time_det)); 
hdf5_write_nv(h5File, 'peak_det', single(peak_det)); 

H5F.close(h5File);

return
