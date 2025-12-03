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

% function csirsLocTable = genCsirsLocTable

for Row = 1:18
    carrier = nrCarrierConfig;
    csirs = nrCSIRSConfig;
    csirs.RowNumber = Row;
    if Row == 1
        csirs.Density = 'three';
    end
    nK = [1 1 1 1 1 4 2 2 6 3 4 4 3 3 3 4 4 4];
    Ports = [1 1 2 4 4 8 8 8 12 12 16 16 24 24 24 32 32 32];
    % Add the CDM configuration array - corresponds to row2nCdm used throughout the codebase
    CDMGroups = [1 1 2 2 2 2 2 4 2 4 2 4 2 4 8 2 4 8];
    csirs.SubcarrierLocations = [0:nK(Row)-1];
    switch Row
        case {13, 14, 16, 17}
            csirs.SymbolLocations = [0, 4];
        otherwise
            csirs.SymbolLocations = 0;
    end
    
    [ind,info_ind] = nrCSIRSIndices(carrier,csirs);
    KBarLBar = info_ind.KBarLBar{1, 1};
    if ismember(Row, [13, 14, 16, 17])
        lenKBarLBar = length(KBarLBar);
        for idxKBarLBar = 1:lenKBarLBar
            if idxKBarLBar > lenKBarLBar/2
                KBarLBar{idxKBarLBar}(2) = KBarLBar{idxKBarLBar}(2) - csirs.SymbolLocations(2);
            end
        end
    end
    csirsLocTable.KBarLBar{Row} = KBarLBar;
    csirsLocTable.CDMGroupIndices{Row} = info_ind.CDMGroupIndices{1,1};
    csirsLocTable.KPrime{Row} = info_ind.KPrime{1,1};
    csirsLocTable.LPrime{Row} = info_ind.LPrime{1,1};
    csirsLocTable.Ports{Row} = Ports(Row);
    csirsLocTable.CDMGroups{Row} = CDMGroups(Row);
end

% Save the updated table
save('csirsLocTable.mat', 'csirsLocTable');

% return
