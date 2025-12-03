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

function prach = convertPreamble(prach, carrier, nodeType)

isUE = strcmp(nodeType, 'UE');

% read paramters from input
mu = carrier.mu;
startRaSym = prach.startRaSym;
delta_f_RA = prach.delta_f_RA;
N_CP_RA = prach.N_CP_RA;
K = prach.K;
k1 = prach.k1;
kBar = prach.kBar;
T_c = carrier.T_c;
T_samp = carrier.T_samp;
N_u = prach.N_u;
N_u_mu = carrier.N_u_mu;
L_RA = prach.L_RA;
if isUE
    y_uv = prach.y_uv;
%     betaPrach = prach.betaPrach;
end
n_slot_RA_sel = prach.n_slot_RA_sel;
k_const = carrier.k_const;
Nfft = carrier.Nfft;

% find starting time for preamble
startRaSym = startRaSym + n_slot_RA_sel*14;
for ll = 0:startRaSym
    % find t_start_RA
    if ll == 0
        t_start_l_mu = 0;
        N_CP_l_mu = 144*k_const*2^(-mu) + 16*k_const;
    else
        if ll-1 == 0 || ll-1 == 7*2^mu
            % increase CP length for symbol 0 and 7
            N_CP_l_mu = 144*k_const*2^(-mu) + 16*k_const;
        else
            N_CP_l_mu = 144*k_const*2^(-mu);
        end
        t_start_l_mu = t_start_l_mu + (N_u_mu + N_CP_l_mu)*T_c;
    end
end
if startRaSym == 0 || startRaSym == 7*2^mu
    N_CP_l_mu = 144*k_const*2^(-mu) + 16*k_const;
else
    N_CP_l_mu = 144*k_const*2^(-mu);
end

t_start_RA = t_start_l_mu;

% find CP length for preamble
switch delta_f_RA
    case {1250, 5000} % for long preamble
        n = 0;
    case {15000, 30000, 60000, 120000} % for short preamble
        % n is the number of times that preamble overlaps with symbol 0 and 7
        n = 0;
        if t_start_RA <= 0 && (t_start_RA + (N_u + N_CP_RA)*T_c > 0) 
            n = n + 1;
        end
        if t_start_RA <= 0.5e-3 && ...
                (t_start_RA + (N_u + N_CP_RA)*T_c > 0.5e-3) 
            n = n + 1;
        end
end
N_CP_l_RA = N_CP_RA + n*16*k_const; % increase cp length based on n    
    
% 0: large FFT approach; 
% 1: filter + up/down sample + modulate
genPreambleAlg = 1; 

prach.y_uv_rx = [];
if isUE
    % for UE to convert preamble from freq domain to time domain
    N_samp_RA = N_u*T_c/T_samp;
    Nfft_RA = Nfft*K;
    Nrep = N_samp_RA/Nfft_RA;
    if genPreambleAlg == 0 % large FFT based
        % calculate starting subcarrier
        startSC = mod(K*k1+kBar, Nfft_RA);
        startSC = startSC + 1; % skip DC 
        y_uv_pad = zeros(1, Nfft_RA);
        y_uv_pad(mod(startSC:startSC+L_RA-1, Nfft_RA)+1) = y_uv;
        % do IFFT
        s_l = ifft(y_uv_pad)*sqrt(Nfft_RA);
        lenCP = N_CP_l_RA*T_c/T_samp;
        s_l_rep = [];
        % do repetition
        for idxRep = 1:Nrep
            s_l_rep = [s_l_rep, s_l];
        end
        s_l = s_l_rep;
        % add CP
        s_l = [s_l(end-lenCP+1:end), s_l];
%         s_l = betaPrach*s_l;
        preambleSamp = s_l;
        prach.Nrep = Nrep;       
    elseif genPreambleAlg == 1 % FFT + filter based
        if prach.K == 1 % no need to resample
            startSC = mod(K*k1+kBar, Nfft_RA);
            startSC = startSC + 1; % skip DC 
            y_uv_pad = zeros(1, Nfft_RA);
            y_uv_pad(mod(startSC:startSC+L_RA-1, Nfft_RA)+1) = y_uv;
            s_l = ifft(y_uv_pad)*sqrt(Nfft_RA);
            lenCP = N_CP_l_RA*T_c/T_samp;
            s_l_rep = [];
            for idxRep = 1:Nrep
                s_l_rep = [s_l_rep, s_l];
            end
            s_l = s_l_rep;
            s_l = [s_l(end-lenCP+1:end), s_l];
%             s_l = betaPrach*s_l; 
            preambleSamp = s_l;
            prach.Nrep = Nrep;                   
        else % need to resample
            % load preamble into ifft input
            fft_in = zeros(1, Nfft);
            fft_in(Nfft/2-floor(L_RA/2)+2:Nfft/2+ceil(L_RA/2)+1) = y_uv;
            % ifft
            fft_out = ifft(fftshift(fft_in))*sqrt(Nfft);
            % repeat
            fft_out_rep = [];
            for idxRep = 1:Nrep
                fft_out_rep = [fft_out_rep, fft_out];
            end
            % add CP
            lenCP = N_CP_l_RA*T_c/T_samp/K;
            fft_out = fft_out_rep;
            lenGuard = 10; % guard samples for filter convolution
            samp = [fft_out(end-lenCP-lenGuard+1:end), fft_out, ...
                fft_out(1:lenGuard)];
            lenSampIn = length(samp);
            % multi-stage upsampling and filtering
            for idxStage = 1:prach.NfirStage
                sampRate = prach.sampRate(idxStage);
                % samp = upsample(samp, sampRate)*sqrt(sampRate);
                lenSamp = length(samp);
                samp = [samp; zeros(sampRate-1, lenSamp)];
                samp = reshape(samp, 1, lenSamp*sampRate)*sqrt(sampRate);
                coef = prach.fir{idxStage}.coef;
                samp = conv(coef, samp);
            end            
            dly = floor(length(samp) - (lenSampIn-2*lenGuard)*K)/2;
            samp = samp(dly+1:end-dly);                      
            % modulation
            modFreq = (K*k1+kBar + floor(L_RA/2))*delta_f_RA;
            lenCP = N_CP_l_RA*T_c/T_samp;
            tt = ([0:length(samp)-1]-lenCP)*T_samp;
            samp = samp.*exp(1j*2*pi*modFreq*tt);
            s_l = samp;            
            preambleSamp = s_l;
            prach.Nrep = Nrep;       
        end
    end
    preambleSampStart = round(t_start_RA/T_samp); 
    % To fit for per slot process
    preambleSampStart = mod(preambleSampStart, carrier.N_samp_slot);
    prach.preambleSamp = preambleSamp;
    prach.preambleSamp_test = preambleSamp/sqrt(Nfft);
    prach.preambleSampStart = preambleSampStart;
else
    % for gNB to convert from time domain to freq domain 
    % extract time domain samples
    rxSamp = prach.rxSamp;
    if genPreambleAlg == 1 && prach.K == 1
        preambleSampStart = round(t_start_RA/T_samp);
        % To fit for per slot process
        preambleSampStart = mod(preambleSampStart, carrier.N_samp_slot);
        preambleSampEnd = floor(preambleSampStart + ...
        (N_CP_l_RA + N_u)*T_c/T_samp - 1);      
    else
        preambleSampStart = round((t_start_RA + N_CP_l_RA*T_c)/T_samp);
        % To fit for per slot process
%         preambleSampStart = mod(preambleSampStart, carrier.N_samp_slot);        
        preambleSampEnd = round(preambleSampStart + N_u*T_c/T_samp - 1);
    end
    [Nant, ~] = size(rxSamp);
    for idxAnt = 1:Nant
        s_l = rxSamp(idxAnt, preambleSampStart+1:preambleSampEnd+1);
        N_samp_RA = N_u*T_c/T_samp;
        Nfft_RA = Nfft*K;
        Nrep = N_samp_RA/Nfft_RA;
        % convert from time domain to freq domain
        if genPreambleAlg == 0 % large FFT based approach
            y_uv = [];
            for idxRep = 1:Nrep
                fft_in = s_l((idxRep-1)*Nfft_RA+1:idxRep*Nfft_RA);
                fft_out = fft(fft_in, Nfft_RA);
                startSC = mod(K*k1+kBar, Nfft_RA);
                startSC = startSC + 1; % skip DC 
                y_uv = [y_uv, fft_out(mod(startSC:startSC+L_RA-1, Nfft_RA)+1)];
            end
            y_uv = y_uv/sqrt(Nfft_RA);
            prach.y_uv_rx(idxAnt,:) = y_uv;   
            prach.NrepRx = Nrep;
        elseif genPreambleAlg == 1 % FFT + filter based
            if prach.K == 1 % no need to resampling
                y_uv = [];
                fftDist = (N_u_mu + 144*k_const*2^(-mu))*T_c/T_samp;
                lenCP = 144*k_const*2^(-mu)*T_c/T_samp;
                startIdx = N_CP_l_mu*T_c/T_samp;
                shiftBase = (N_CP_l_RA - N_CP_l_mu)*T_c/T_samp;
                [~, lenS] = size(s_l);                
                NrepRx = 0;
                for idxRep = 1:Nrep
                    oran_mode = 1;
                    if oran_mode % PRACH FFT window not sync with PUSCH
                        fft_in = s_l(startIdx+1:startIdx+Nfft_RA);
    %                         fft_in = circshift(fft_in, NrepRx*lenCP-shiftBase, 2);
                        fft_out = fft(fft_in, Nfft_RA);
                        fft_out = fft_out.*exp(-1j*2*pi*(0:Nfft_RA-1)*...
                                (-shiftBase)/Nfft_RA);
                        startSC = mod(K*k1+kBar, Nfft_RA);
                        startSC = startSC + 1; % skip DC
                        y_uv = [y_uv, fft_out(mod(startSC:startSC+L_RA-1, Nfft_RA)+1)];
                        startIdx = startIdx + Nfft;
                        NrepRx = NrepRx + 1;                                                
                    else % PRACH FFT window sync with PUSCH
                        if startIdx + Nfft_RA <= lenS
                            fft_in = s_l(startIdx+1:startIdx+Nfft_RA);
    %                         fft_in = circshift(fft_in, NrepRx*lenCP-shiftBase, 2);
                            fft_out = fft(fft_in, Nfft_RA);
                            fft_out = fft_out.*exp(-1j*2*pi*(0:Nfft_RA-1)*...
                                (NrepRx*lenCP-shiftBase)/Nfft_RA);
                            startSC = mod(K*k1+kBar, Nfft_RA);
                            startSC = startSC + 1; % skip DC
                            y_uv = [y_uv, fft_out(mod(startSC:startSC+L_RA-1, Nfft_RA)+1)];
                            startIdx = startIdx + fftDist;
                            NrepRx = NrepRx + 1;
                        else
                            break;
                        end
                    end
                end                                                                   
                y_uv = y_uv/sqrt(Nfft_RA);
                prach.y_uv_rx(idxAnt,:) = y_uv; 
                prach.NrepRx = NrepRx;
            else % need to resample
                % demodualte
                samp = s_l;
                modFreq = (K*k1+kBar + floor(L_RA/2))*delta_f_RA;
                samp = samp.*exp(-1j*2*pi*modFreq*[0:length(samp)-1]*T_samp);
                lenSampIn = length(samp);
                % multi-stage filtering and downsampling
                for idxStage = prach.NfirStage:-1:1
                    sampRate = prach.sampRate(idxStage);
                    coef = prach.fir{idxStage}.coef;
                    samp = conv(coef, samp)*sqrt(sampRate);
                    % samp = downsample(samp, sampRate);
                    samp = samp(1:sampRate:end);
                end
                % compensate for filter delay
                dly = floor((length(samp) - lenSampIn/K)/2);
                samp = samp(dly+1:end-dly);
                s_l = samp;
                y_uv = [];
                for idxRep = 1:Nrep % for each repeatition                      
                    fft_in = s_l((idxRep-1)*Nfft+1:idxRep*Nfft);
                    % fft
                    fft_out = fftshift(fft(fft_in, Nfft));
                    % grab preamble from fft output 
                    y_uv = [y_uv, fft_out(Nfft/2-floor(L_RA/2)+2: ...
                        Nfft/2+ceil(L_RA/2)+1)];
                end
                y_uv = y_uv/sqrt(Nfft);
                prach.y_uv_rx(idxAnt,:) = y_uv;
                prach.NrepRx = Nrep;                
            end        
        end
    end
end



return



