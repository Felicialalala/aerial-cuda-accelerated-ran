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

function srs_rkhs_tables = derive_srs_rkhs_objects

%%
% PARAMATERS

gridSizes        = [4 8 816];
nGridSizes       = length(gridSizes);
zpGridSizes      = zeros(1,nGridSizes);
nEigsPerGridSize = [3 3 3];

%%
% EIGENVECTORS

eigenVecTable = [];
eigValTable   = [];
corrTable     = [];
secondStageTwiddleFactorsTable = cell(nGridSizes, 1);
secondStageFourierPermTable    = cell(nGridSizes, 1);

for i = 0 : (nGridSizes - 1)

    % grid:
    gridSize           = gridSizes(i + 1);
    zpGridSize         = 2^(ceil(log2(gridSize)));
    zpGridSizes(i + 1) = zpGridSize;
    grid               = 0 : (gridSize - 1);
    grid               = grid.';

    % eigenvectors/eigenvalues:
    nEigs                = nEigsPerGridSize(i + 1);
    K                    = sinc_tbf((grid - grid') / zpGridSize);
    [V, D]               = eigs(K, nEigs);
    eigenVecTable{i + 1} = V;
    eigValTable{i + 1}   = diag(D);


    % eigenvector correlation:
    V_zp                  = zeros(zpGridSize, nEigs);
    V_zp(1 : gridSize, :) = V;

    corr = zeros(zpGridSize, nEigs, nEigs);
    for j = 1 : nEigs
        for k = 1 : nEigs
            c           = ifft(conj(V_zp(:,j)) .* V_zp(:,k), zpGridSize) * zpGridSize;
            corr(:,j,k) = c;
        end
    end
    
    if(gridSize > 200)
        lambda = 2.31*10^(-6) * 120 * 10^3;
        nCpInt = ceil(lambda *  zpGridSize);
        corr   = corr(1 : nCpInt, :, :);
    end

    corrTable{i + 1} = permute(corr, [2 3 1]);


    secondStageFourierSize        = floor(zpGridSize / 32);

    if(secondStageFourierSize > 1)
        t = (0 : 31).';
        secondStageTwiddleFactors = zeros(32,secondStageFourierSize);
    
        for k = 0 : (secondStageFourierSize - 1)
            secondStageTwiddleFactors(:,k + 1) = exp(2*pi*1i*k*t / zpGridSize);
        end
    
        secondStageFourierPerm = bitrevorder(0 : (secondStageFourierSize - 1));

        secondStageTwiddleFactorsTable{i + 1} = secondStageTwiddleFactors;
        secondStageFourierPermTable{i + 1}    = secondStageFourierPerm;
    end



end


%%
% WRAP

rkhsPrms                  = [];
rkhsPrms.nEigs            = nEigs;
rkhsPrms.nGridSizes       = nGridSizes;
rkhsPrms.gridSizes        = gridSizes;
rkhsPrms.nEigsPerGridSize = nEigsPerGridSize;

gridPrms = cell(rkhsPrms.nGridSizes, 1);
for i = 0 : (rkhsPrms.nGridSizes - 1)
    gridPrms{i + 1}.nEig       = nEigsPerGridSize(i + 1);
    gridPrms{i + 1}.gridSize   = gridSizes(i + 1);
    gridPrms{i + 1}.zpGridSize = zpGridSizes(i + 1);
end


srs_rkhs_tables               = [];
srs_rkhs_tables.rkhsPrms      = rkhsPrms;
srs_rkhs_tables.eigenVecTable = eigenVecTable;
srs_rkhs_tables.eigValTable   = eigValTable;
srs_rkhs_tables.corrTable     = corrTable;
srs_rkhs_tables.gridPrms      = gridPrms;

srs_rkhs_tables.secondStageTwiddleFactorsTable = secondStageTwiddleFactorsTable;
srs_rkhs_tables.secondStageFourierPermTable    = secondStageFourierPermTable;


end





