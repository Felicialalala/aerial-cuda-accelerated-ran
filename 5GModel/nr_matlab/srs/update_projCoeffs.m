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

function projCoeffs = update_projCoeffs(projCoeffs, eq_coeff, old_eqCoeff, boxIdxs_freq, boxIdxs_ver, boxIdxs_hor, eqBoxIdx_freq, eqBoxIdx_ver, eqBoxIdx_hor, corr_freq, corr_gnbVertAnt, corr_gnbHorAnt, nKeptBoxes, freqNumEigs, verNumEigs, horNumEigs)


            % projCoeffs = update_projCoeffs(projCoeffs, updateEqCoeff, zeros(scNumEigs, verNumEigs, horNumEigs), ...
            %                                 projCoeffFreqIntIdx, projCoeffVerAntIntIdx, projCoeffHorAntIntIdx, eqCoeffFreqIntIdx(nEqBoxes + 1), eqCoeffVerAntIntIdx(nEqBoxes + 1), ...
            %                                 eqCoeffHorAntIntIdx(nEqBoxes
            %                                 + 1), scCorr, verAntCorr,
            %                                 horAntCorr, nOccupiedBoxes, scNumEigs, verNumEigs, horNumEigs);\\


% corr_idx_freq = mod(-eqIntIdx_freq + intIdxs_freq, nFreq) + 1;
% corr_idx_ver  = mod(-eqIntIdx_ver + intIdxs_ver, nGnbAntVer) + 1;
% corr_idx_hor  = mod(-eqIntIdx_hor + intIdxs_hor, nGnbAntHor) + 1;

% C_freq = corr_freq(:,:,corr_idx_freq);
% C_ver  = corr_gnbVertAnt(:,:,corr_idx_ver);
% C_hor  = corr_gnbHorAnt(:,:,corr_idx_hor);


 
C_freq = zeros(freqNumEigs, freqNumEigs, nKeptBoxes);
C_ver  = zeros(verNumEigs,  verNumEigs,  nKeptBoxes);
C_hor  = zeros(horNumEigs,  horNumEigs,  nKeptBoxes);

for boxIdx = 0 : (nKeptBoxes - 1)
    freqDist                 = boxIdxs_freq(boxIdx + 1) - eqBoxIdx_freq;
    freqDist_abs             = abs(freqDist);
    C_freq(:, :, boxIdx + 1) = corr_freq(:, :, freqDist_abs + 1);
    if(freqDist < 0)
        C_freq(:, :, boxIdx + 1) = conj(C_freq(:, :, boxIdx + 1));
    end

    verDist                 = boxIdxs_ver(boxIdx + 1) - eqBoxIdx_ver;
    verDist_abs             = abs(verDist);
    C_ver(:, :, boxIdx + 1) = corr_gnbVertAnt(:, :, verDist_abs + 1);
    if(verDist < 0)
        C_ver(:, :, boxIdx + 1) = conj(C_ver(:, :, boxIdx + 1));
    end

    horDist                 = boxIdxs_hor(boxIdx + 1) - eqBoxIdx_hor;
    horDist_abs             = abs(horDist);
    C_hor(:, :, boxIdx + 1) = corr_gnbHorAnt(:, :, horDist_abs + 1);
    if(horDist < 0)
        C_hor(:, :, boxIdx + 1) = conj(C_hor(:, :, boxIdx + 1));
    end
end

delta_corr = eq_coeff - old_eqCoeff;
delta_corr = repmat(delta_corr, 1, 1, 1, nKeptBoxes);



% frequency
delta_corr = reshape(delta_corr, freqNumEigs, verNumEigs * horNumEigs, nKeptBoxes); % freq x ver x hor
delta_corr = pagemtimes(C_freq, delta_corr);

% vertical
delta_corr = pagetranspose(delta_corr); % ver x hor x freq
delta_corr = reshape(delta_corr, verNumEigs, horNumEigs * freqNumEigs, nKeptBoxes);
delta_corr = pagemtimes(C_ver, delta_corr);

% hor
delta_corr = pagetranspose(delta_corr); % hor x freq x ver
delta_corr = reshape(delta_corr, horNumEigs, freqNumEigs * verNumEigs,nKeptBoxes);
delta_corr = pagemtimes(C_hor, delta_corr);

% reshape
delta_corr = pagetranspose(delta_corr); % freq x ver x hor
delta_corr = reshape(delta_corr, freqNumEigs, verNumEigs, horNumEigs, nKeptBoxes);

projCoeffs = projCoeffs - delta_corr;

a = 2;


