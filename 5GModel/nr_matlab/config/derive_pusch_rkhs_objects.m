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

function push_rkhs_tables = derive_pusch_rkhs_objects

%%
% PARAMATERS


zpDmrsScSizes  = [32 64 128 256 512 1024];
nZpDmrsScSizes = length(zpDmrsScSizes);
nZpDmrsScEigs  = 6;
nIterpEigs     = 6;

nPrbSizes = 64;
nEigs     = 3;


%%
% ZP EIGENVECTORS

zpDmrsScEigenVecTable  = [];
zpInterpVecTable       = [];
zpFftPrms              = [];

for i = 0 : (nZpDmrsScSizes - 1)

    % zero padded DMRS eigenvectors:
    nZpDmrsSc = zpDmrsScSizes(i + 1);

    f_dmrs = 0 : (nZpDmrsSc - 1);
    f_dmrs = f_dmrs.';

    K                            = sinc_tbf((f_dmrs - f_dmrs') / nZpDmrsSc);
    [V,~]                        = eigs(K, nZpDmrsScEigs);
    zpDmrsScEigenVecTable{i + 1} = V.';

    % interpolation filter:
    f_interp = -1 : (2*nZpDmrsSc - 1);
    f_interp = f_interp.' / 2;

    K     = sinc_tbf((f_interp - f_interp') / nZpDmrsSc);
    [V,D] = eigs(K, nIterpEigs);
    V     = V * diag(diag(D).^(1/2));

    zpInterpVecTable{i + 1} = V;

    % zero-padded FFT paramaters:
    zpFftPrm = [];

    secondStageFourierSize        = floor(nZpDmrsSc / 32);
    % secondStageTwiddleFactors = [];

    t = (0 : 31).';
    secondStageTwiddleFactors = zeros(32,secondStageFourierSize);

    for k = 0 : (secondStageFourierSize - 1)
        secondStageTwiddleFactors(:,k + 1) = exp(-2*pi*1i*k*t / nZpDmrsSc);
    end

    secondStageFourierPerm = bitrevorder(0 : (secondStageFourierSize - 1));

    zpFftPrm.secondStageFourierSize    = secondStageFourierSize;
    zpFftPrm.secondStageTwiddleFactors = secondStageTwiddleFactors;
    zpFftPrm.secondStageFourierPerm    = secondStageFourierPerm;
    zpFftPrm.nZpDmrsSc                 = nZpDmrsSc;

    zpFftPrms{i + 1} = zpFftPrm;
end

%%
% EIGENVECTORS

corrTable      = [];
eigVecCobTable = [];
eigValTable    = [];
interpCobTable = [];
prbPrms        = cell(nPrbSizes,1);

for nPrb = 1 : nPrbSizes
    prbPrm  = [];

    % compute number of Zp dmrs SCs
    nDmrsSc = nPrb * 6;
    if(nPrb <= 4)
        nZpDmrsSc = min(2^(ceil(log2(nDmrsSc)) + 2 ), 2048);
    else
        nZpDmrsSc = min(2^(ceil(log2(nDmrsSc)) + 1), 2048);
    end

    % prb eigenvectors
    f_dmrs = 0 : (nDmrsSc - 1);
    f_dmrs = f_dmrs.';

    K     = sinc_tbf((f_dmrs - f_dmrs') / nZpDmrsSc);
    [V,D] = eigs(K, nEigs);


    eigValTable{nPrb} = diag(D);

    % change of basis: Zp eigenvectors to allocation eigenvectors
    zpIdx = log2(nZpDmrsSc) - 5;
    V_zp  = zpDmrsScEigenVecTable{zpIdx + 1}.';

    cob                  = V_zp(1:nDmrsSc,:) \ V;
    eigVecCobTable{nPrb} = cob.';

    % eigenvector correlation
    % totalLength = 1 / (60*10^3);
    nCpInt      = floor(nZpDmrsSc * 0.1386);
    

    corr = zeros(nZpDmrsSc, nEigs, nEigs);
    for i = 1 : nEigs
        for j = 1 : nEigs
            c           = ifft(conj(V(:,i)) .* V(:,j),nZpDmrsSc) * nZpDmrsSc;
            corr(:,i,j) = c;
        end
    end
    corr_half_nZpDmrsSc = squeeze(corr(nZpDmrsSc/2 + 1, :, :));

    corr            = permute(corr, [2 3 1]);
    corr            = corr(:, : ,1 : nCpInt);
    corrTable{nPrb} = corr;

    % interpolation cob
    interpVec = zpInterpVecTable{zpIdx + 1};
    
    dmrsIdxs = 0 : (nDmrsSc - 1);
    dmrsIdxs = 2*dmrsIdxs + 1;

    VV                   = zeros(2*nZpDmrsSc + 1, nEigs);
    VV(dmrsIdxs + 1, :)  = V;
    cob = interpVec' * VV * diag(diag(D).^(-1));
    interpCobTable{nPrb} = cob;


    % store Prb paramaters:
    prbPrm.nZpDmrsSc           = nZpDmrsSc;
    prbPrm.zpIdx               = zpIdx;
    prbPrm.nCpInt              = nCpInt;
    prbPrm.sumEigValues        = sum(diag(D));
    prbPrm.corr_half_nZpDmrsSc = corr_half_nZpDmrsSc;


    prbPrms{nPrb} = prbPrm; 
end

%%
% WRAP

rkhsPrms               = [];
rkhsPrms.nEigs         = nEigs;
rkhsPrms.nZpDmrsScEigs = nZpDmrsScEigs;
rkhsPrms.nIterpEigs    = nIterpEigs;
rkhsPrms.nPrbSizes     = nPrbSizes;
rkhsPrms.nZpSizes      = nZpDmrsScSizes;
rkhsPrms.hammingFlag   = 1;


push_rkhs_tables                       = [];
push_rkhs_tables.eigVecCobTable        = eigVecCobTable;
push_rkhs_tables.corrTable             = corrTable;
push_rkhs_tables.zpDmrsScEigenVecTable = zpDmrsScEigenVecTable;
push_rkhs_tables.rkhsPrms              = rkhsPrms;
push_rkhs_tables.eigValTable           = eigValTable;
push_rkhs_tables.interpCobTable        = interpCobTable;
push_rkhs_tables.zpInterpVecTable      = zpInterpVecTable;
push_rkhs_tables.prbPrms               = prbPrms;
push_rkhs_tables.zpFftPrms             = zpFftPrms;

end






