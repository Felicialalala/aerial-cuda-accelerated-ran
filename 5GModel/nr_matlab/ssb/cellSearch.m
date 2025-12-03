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

function csResult = cellSearch(rxSamp, Nfft)

plotCorr = 0;
enableToEst = 0;
nScPSS = 127;
nScSSB = 240;
nLeft0PSS = 56;
nRight0PSS = 57;

persistent lpFilt

if isempty(lpFilt)
    lpFilt = designfilt('lowpassfir', 'PassbandFrequency', ceil(nScPSS/2)/(Nfft/2),'StopbandFrequency', (nScSSB/2)/(Nfft/2), 'PassbandRipple', 0.5,'StopbandAttenuation', 40, 'DesignMethod', 'equiripple');
end

% ToBeAdded modulate SSB to the center

[nSamp, nAnt] = size(rxSamp);

% calculate reference PSS waveform in time domain
for idxPss = 0:2
    pss_ref(:, idxPss+1) = build_pss(idxPss);
    pss_pad0 = [zeros(nLeft0PSS, 1); pss_ref(:, idxPss+1); zeros(nRight0PSS, 1)];
    pss_t(:, idxPss+1) = sqrt(nScSSB)*ifft(fftshift(pss_pad0));
end

% low pass filtering in time domain for SSB bandwidth(nScSSB subcarriers)
for idxAnt = 1:nAnt
    y1 = filter(lpFilt, rxSamp(:, idxAnt));
    y1 = circshift(y1, -floor(length(lpFilt.Coefficients)/2));
    y2(:, idxAnt) = resample(y1, nScSSB, Nfft);
end
[nSamp2, ~] = size(y2);

% PSS detection
pssDetected = 0;
for idxSamp = 1:nSamp2-nScSSB
    for idxPss = 0:2
        for idxAnt = 1:nAnt
            y3(idxAnt) = abs(sum(y2(idxSamp:idxSamp+nScSSB-1, nAnt).*conj(pss_t(:, idxPss+1))));
            y3_norm(idxAnt) = sum(abs(y2(idxSamp:idxSamp+nScSSB-1, nAnt)).^2);
            y3_ratio(idxAnt) = y3(idxAnt)/sqrt(y3_norm(idxAnt));
        end
        [y_corr(idxPss+1), idx] = max(y3_ratio);
        y_peak(idxPss+1) = y3(idx);
        y_ant(idxPss+1) = idx;
    end
    [pkRatio, idx] = max(y_corr);
    pk = y_peak(idx);
    ant = y_ant(idx);
    pssThreshold = 4;
    pkRatioVec(idxSamp) = pkRatio;
    pkVec(idxSamp) = pk;
    if pkRatio > pssThreshold && pssDetected == 0
        pssDetected = 1;
        pssIdx = idx-1;
        pssPeak = pk;
        pssRatio = pkRatio;
        pssAnt = ant;
        pssSampIdx = idxSamp;
%         break;
    end
end

if plotCorr
    figure; 
    subplot(2,2,1); plot(pkVec); title('PSS peak'); grid on;
    subplot(2,2,2); plot(pkRatioVec); title('PSS ratio'); grid on;
end

if pssDetected
    % grab SSB block in freq domain
    lenCP = (Nfft/4096)*288;
    lenSym = Nfft + lenCP;
    ssbStart = floor((pssSampIdx) * Nfft/nScSSB) - lenCP;
    ssbSamp = rxSamp(ssbStart +1:ssbStart + 4*lenSym, :);
    for idxSym = 1:4
        for idxAnt = 1:nAnt
            fftStart = (idxSym-1)*lenSym + lenCP + 1;
            fftEnd = (idxSym-1)*lenSym + lenCP + Nfft;
            xt = ssbSamp(fftStart:fftEnd, idxAnt);
            xf = sqrt(1/Nfft)*fftshift(fft(xt, Nfft));
            ssbTf(:, idxSym, idxAnt) = xf(Nfft/2-120+1:Nfft/2+120);
        end
    end
    if enableToEst
        % estimate timing offset based on PSS
        pss_rx = ssbTf(nLeft0PSS+1:nLeft0PSS+nScPss, 1, pssAnt);
        pss_corr = pss_rx.*conj(pss_ref(:, pssIdx+1));
        phaseRot = angle(sum(pss_corr(2:end).*conj(pss_corr(1:end-1))));
        toSamp = phaseRot*Nfft/(2*pi);
    end
    % SSS detection
    sss_rx = ssbTf(nLeft0PSS+1:nLeft0PSS+nScPSS, 3, pssAnt);
    sss_norm = sum(abs(sss_rx).^2);
    for idxSss = 0:335
        sss_ref = build_sss2(idxSss, pssIdx);
        sss_corr(idxSss+1) = abs(sum(sss_rx.*conj(sss_ref)));
    end
    [sssPeak, idx] = max(sss_corr);
    sssRatio = sssPeak/sqrt(sss_norm);
    sssIdx = idx-1;
    if plotCorr
        subplot(2,2,3); plot(sss_corr); title('SSS peak'); grid on;
        subplot(2,2,4); plot(sss_corr/sqrt(sss_norm)); title('SSS ratio'); grid on;
        pause(1);
    end
    nCellId = pssIdx + sssIdx * 3;
end

if pssDetected
    csResult.pssDetected = 1;
    csResult.pssIdx = pssIdx;
    csResult.pssPeak = pssPeak;
    csResult.pssRatio = pssRatio;
    csResult.pssAnt = pssAnt;
    csResult.pssSampIdx = pssSampIdx;
    csResult.sssPeak = sssPeak;
    csResult.sssRatio = sssRatio;
    csResult.sssIdx = sssIdx;
    csResult.nCellId = nCellId;
else
    csResult.pssDetected = 0;
end

csResult

return

function d_sss = build_sss2(N_id1, N_id2)

% function builds the Secondary Synchronization Sequence (SSS)

%inputs:
% N_id  --> physical cell id

%outputs:
% d_sss --> sss. Dim: 127 x 1

%%
%START

% N_id2 = mod(N_id,3);
% N_id1 = (N_id - N_id2) / 3;

load('sss_x_seq.mat');

m0 = 15*floor(N_id1/112) + 5*N_id2;
m1 = mod(N_id1,112);

idx0 = mod((0:126) + m0,127);
idx1 = mod((0:126) + m1,127);

d_sss = x0(idx0 + 1) .* x1(idx1 + 1);

return
