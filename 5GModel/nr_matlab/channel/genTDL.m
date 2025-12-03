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

function Chan = genTDL(Chan, SysPar)
%
% Generate time domain TDL fading channel filters based on TS 38.141
% The fading channel is quasi-static over a number of samples 
% instead of changing over every sample.
%

chanType = Chan.type;
f_samp = Chan.f_samp;
T_samp = Chan.T_samp;
N_frame = SysPar.SimCtrl.N_frame;
lenSamp = N_frame*10e-3*f_samp;
Nin = Chan.Nin;
Nout = Chan.Nout;

pdp_A = [...
            0       -15.5;
            10      0;
            15      -5.1;
            20      -5.1;
            25      -9.6;
            50      -8.2;
            65      -13.1;
            75      -11.5;
            105     -11.0;
            135  	-16.2;
            150	    -16.6;
            290   	-26.2
            ];
pdp_B = [...
            0       0;
            10      -2.2;
            20      -0.6;
            30      -0.6;
            35      -0.3;
            45      -1.2;
            55      -5.9;
            120     -2.2;
            170     -0.8;
            245 	-6.3;
            330     -7.5;
            480     -7.1
            ];
pdp_C = [...
            0       -6.9;
            65      0;
            70      -7.7;
            190     -2.5;
            195     -2.4;
            200     -9.9;
            240     -8.0;
            325     -6.6;
            520     -7.1;
            1045	-13.0;
            1510	-14.2;
            2595	-16.0
            ];

% load power delay profile model
switch chanType
    case 'TDLA30-5-Low'
        pdp = pdp_A;
        f_doppler = 5;
        MIMOcorrlation = 'Low';
    case 'TDLA30-10-Low' 
        pdp = pdp_A;
        f_doppler = 10;
        MIMOcorrlation = 'Low';        
    case 'TDLB100-400-Low'
        pdp = pdp_B;
        f_doppler = 400;
        MIMOcorrlation = 'Low';           
    case 'TDLC300-100-Low'
        pdp = pdp_C;
        f_doppler = 100;
        MIMOcorrlation = 'Low'; 
    case 'TDL'
        if strcmp(Chan.DelayProfile,'TDL-A')
            pdp = pdp_A;
        elseif strcmp(Chan.DelayProfile,'TDL-B')
            pdp = pdp_B;
        elseif strcmp(Chan.DelayProfile,'TDL-C')
            pdp = pdp_C;
        else
            error('Undefined PDP!')
        end
        f_doppler = Chan.MaximumDopplerShift;
        MIMOcorrlation = Chan.MIMOCorrelation;
    otherwise 
        error('chanType is not supported ... \n');
end

% load MIMO correlation matrix
switch MIMOcorrlation
    case 'Low'
        corrMatrix = diag(ones(1, Nin*Nout)); 
    otherwise
        error('MIMO correlation type is not supported ...\n');
end

Npath = 48;
% quantize to fir filter with T_samp interval
PathDelays = pdp(:,1);
firTapMap = round(PathDelays * 1e-9 / T_samp) + 1;  % map to the same tap dure to sampling
firPw = 10.^(pdp(:,2)/10);
lenFir = round(PathDelays(end) * 1e-9 / T_samp) + 1;
firPw = sqrt(firPw/(sum(firPw)*double(Npath))); % sqrt(Npath) needs to be multipled to ensure over long term E{||chanMatrix(txAntIdx, rxAntIdx, :, idxBatch)||_2^2} = 1 for all (txAntIdx, rxAntIdx, idxBatch)
NtapPdp = length(PathDelays);  % number of taps in PDP table
% generate doppler spectrum mask

% update rate of quasi-static channel
% 15e3 is 50x larger than max doppler freq (300Hz). It should be fast
% enough to capture the channel variation over time
f_batch = 15e3; 
NbatchSamp = round(f_samp/f_batch); 
Nbatch = ceil(lenSamp/NbatchSamp); % number of channel realization

genChan_alg = 0;
if genChan_alg == 0    
    % generate Rayleigh fading for each Tap based on MATLAB Fading Channel in Communication Toolbox
    % Ref: https://www.mathworks.com/help/comm/ug/fading-channels.html
    thetaRand = 2*rand(Nin, Nout, NtapPdp, Npath,2)*pi;
    chanMatrix = zeros(Nin, Nout, lenFir, Nbatch);
    timeSeq = [0:Nbatch-1]*T_samp*NbatchSamp;
    
    % Superimpose N paths 
    for idxIn = 1:Nin
        for idxOut = 1:Nout
            for idxTap = 1:NtapPdp
                hsum = zeros(1, Nbatch);
                for idxPath = 1:Npath
                    alpha_0 = pi/4/Npath * idxTap/(NtapPdp+2);
                    % real part
                    freqRand_real = f_doppler * cos(pi/2/Npath * (idxPath - 0.5) + alpha_0);
                    % imag part
                    freqRand_imag = f_doppler * cos(pi/2/Npath * (idxPath - 0.5) - alpha_0);

                    % Note size(chanMatrix) = [Nin, Nout, lenFir, Nbatch]
                    h1 = firPw(idxTap) * ( ...   % normalization coe
                        cos(2*pi*freqRand_real*timeSeq + thetaRand(idxIn, idxOut, idxTap, idxPath, 1)) + ...  % real part
                        + 1j * cos(2*pi*freqRand_imag*timeSeq + thetaRand(idxIn, idxOut, idxTap, idxPath, 2))...  % imag part
                        );
                        
                    hsum = h1 + hsum;
                end
                chanMatrix(idxIn, idxOut, firTapMap(idxTap),:) = chanMatrix(idxIn, idxOut, firTapMap(idxTap),:) +  reshape(hsum, [1, 1, 1, Nbatch]);
            end
        end
    end
    % apply MIMO correlation: TODO, currently no effect
    for idxTap = 1:lenFir
        if ismember(idxTap, firTapMap)
            for idxBatch = 1:Nbatch
                tap1 = reshape(chanMatrix(:, :, idxTap, idxBatch), [1, Nin*Nout]);
                tap2 = (corrMatrix*tap1.').';
                chanMatrix(:, :, idxTap, idxBatch) = reshape(tap2, [Nin, Nout]);
            end
        end
    end

    % normalize power for chanMatrix
    % chanMatrix = chanMatrix/sqrt(sum(sum(sum(sum(abs(chanMatrix).^2))))/(Nin*Nout*Nbatch));
else    
    f_sc = f_batch/Nbatch;    % Subcarrier spacing with FFT size of Nbatch
    sc_doppler = floor(f_doppler/f_sc)-1; % subcarrier corresponding to doppler freq
    
    fv = (0:sc_doppler)*f_sc/f_doppler; % f/f_doppler
    mask1 = sqrt(1./sqrt(1-fv.^2)); % single side spectrum
    lenMask1 = length(mask1);
    mask2 = zeros(1, Nbatch);    % full mask
    mask2(1:lenMask1) = mask1;
    mask2(end-lenMask1+2:end) = fliplr(mask1(2:end));
    
    % filter AWGN vector with doppler spectrum
    chanMatrix = zeros(Nin, Nout, lenFir, Nbatch);
    for idxIn = 1:Nin
        for idxOut = 1:Nout
            for idxTap = find(firPw)
                % generate AWGN sequence of size Nbatch and apply power delay profile
                tap1 = sqrt(firPw(idxTap))*sqrt(0.5)*(randn(1, Nbatch)+1j*randn(1, Nbatch));
                tap2 = ifft(fft(tap1, Nbatch).*mask2); % multiply with doppler spectrum mask
                chanMatrix(idxIn, idxOut, idxTap,:) = tap2;
            end
        end
    end
    
    % apply MIMO correlation
    for idxTap = find(firPw)
        for idxBatch = 1:Nbatch
            tap1 = reshape(chanMatrix(:, :, idxTap, idxBatch), [1, Nin*Nout]);
            tap2 = (corrMatrix*tap1.').';
            chanMatrix(:, :, idxTap, idxBatch) = reshape(tap2, [Nin, Nout]);
        end
    end
    
    % normalize power for chanMatrix
    % chanMatrix = chanMatrix/sqrt(sum(sum(sum(sum(abs(chanMatrix).^2))))/(Nin*Nout*Nbatch));
end

Chan.chanMatrix = chanMatrix;
Chan.lenFir = lenFir;
Chan.NbatchSamp = NbatchSamp;

return
