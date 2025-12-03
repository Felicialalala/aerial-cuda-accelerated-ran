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

function [nBitsCsi2]= csiP2PayloadSizeCalc(rank, nCsirsPorts, N1, N2, ...
    nCsiReports, csiReportingBand, codebookType, codebookMode, isCqi, isLi)

                                            
% This function calculates CSI part 2 payload size 

% Ref.: TS 38.212 Sec. 6.3.1.1.2.1

% Input parameters
 
% rank:                 rank reported by UE in CSI-P1 payload
% nCsirsPorts:          number of CSI RS ports (Candidate values = 2,4,8,12,16,24,32)
% N1:                   number of antenna ports in first dimension (horizontal)
% N2:                   number of antenna ports in second dimension (vertical)
% nCsiReports:          number of CSI reports for CSI part 2 (>=1)
% csiReporting band:    wideband, or subband, or both
% codebookType:         type1SinglePanel, or type1MultiPanel, or type2, or type2PortSelection 
% codebookMode:         '0' (N/A) or '1' or '2' (non zero value only needed for type 1 codebook)
% isCqi:                whether CSI-P2 contains CQI (wideband or subband): 0 or 1
% isLi:                 whether CSI-P2 contains LI : 0 or 1

%==============================================================
% Output 

% nBitsCsi2:            total bitwidth for CSI-P2 payload

%==============================================================
% Assumptions:

% CSI Part 2 payload only carries wideband PMI (i.e. no CQI/LI/subband parameters) for a single CSI-report,
% with the following attributes:

% nCsirsPorts = 4
% N1 = 2
% N2 = 1
% nCsiReports = 1
% csiReportingBand = 'wideband'
% codebookType = 'type1SinglePanel'
% codebookMode = 1
% isCqi = 0
% isLi = 0

%%
% Defaults paramaters (other then rank)

if nargin == 1
    nCsirsPorts = 4;
    N1 = 2;
    N2 = 1;
    nCsiReports = 1;
    csiReportingBand = 'wideband';
    codebookType = 'type1SinglePanel';
    codebookMode = 1;
    isCqi = 0;
    isLi = 0;
end


%% Input validation
% NOTE: The validation is for 21-3 feature support, may evolve in future. 

if nCsirsPorts ~=4
    warning ('Number of CSI-RS ports should be 4 for POC1'); % for POC1 (21-3)
end

if nCsiReports~=1
    error ('Invalid input: only 1 CSI-report is supported');
end

switch csiReportingBand 
    case 'wideband'
        if (isCqi || isLi)
            error('Invalid input: CQI or LI reporting on CSI-P2 is not supported');
        end
    otherwise
        error('Invalid input: only wideband is supported');
end

switch codebookType
    case 'type1SinglePanel'
        if codebookMode ~=1
            warning ('CodebookMode should be 1 for POC1'); % for POC 1 (21-3)
        end
    otherwise 
        error('Invalid input: Only type1SinglePanel codebook is supported');
end

%% CSI-P2 payload estimation

% TS 38.214 Table 5.2.2.2.1-2: Supported Configurations of (N1,N2);(O1,O2)

switch nCsirsPorts
    case 2
        % valid number of ports but N/A (N1,N2);(O1,O2)
    case 4
        if (N1 == 2) && (N2 == 1)
        O1 = 4; % DFT oversampling (beam sweeping step in first dimension)
        O2 = 1; % DFT oversampling (beam sweeping step in second dimension)
        else
            error('Invalid input for N1-N2');
        end
    case 8
        if (N1 == 2) && (N2 == 2)
            O1 = 4;
            O2 = 4;
        elseif (N1 == 4) && (N2 == 1)
            O1 = 4;
            O2 = 1;
        else
            error ('Invalid input for N1-N2');
        end
    case 12        
        if (N1 == 3) && (N2 == 2)
            O1 = 4;
            O2 = 4;
        elseif (N1 == 6) && (N2 == 1)
            O1 = 4;
            O2 = 1;  
        else
            error ('Invalid input for N1-N2');
        end
    case 16
      if (N1 == 4) && (N2 == 2)
            O1 = 4;
            O2 = 4;
        elseif (N1 == 8) && (N2 == 1)
            O1 = 4;
            O2 = 1;  
        else
            error ('Invalid input for N1-N2');
      end  
    case 24
        if (((N1 == 4) && (N2 == 3)) || ((N1 == 6) && (N2 == 2)))
            O1 = 4;
            O2 = 4;
        elseif (N1 == 12) && (N2 == 1)
            O1 = 4;
            O2 = 1;  
        else
            error ('Invalid input for N1-N2');
        end
    case 32
        if (((N1 == 4) && (N2 == 4)) || ((N1 == 8) && (N2 == 2)))
            O1 = 4;
            O2 = 4;
        elseif (N1 == 16) && (N2 == 1)
            O1 = 4;
            O2 = 1;  
        else
            error ('Invalid input for N1-N2');
        end
    otherwise
        error('Invalid input for nCsirsPorts');
end

%%
% TS 38.212 Table 6.3.1.1.2-1 (PMI of codebookType = type1SinglePanel)

nBitsCsi2 = 0; % This is a placeholder for future addition of CQI/L1 bits 

switch codebookType
    case 'type1SinglePanel'
        if (rank ==1)
            if (nCsirsPorts > 2)
                if (N2 == 1)
                    switch codebookMode
                        case 1
                            i11 = ceil(log2(N1 * O1));  % Information field X1 for wideband PMI
                            i12 = ceil(log2(N2 * O2));  % Information field X1 for wideband PMI
                            i2 = 2;                     % Information field X2 for wideband PMI or per subband PMI
                            nBitsCsi2 = nBitsCsi2 + i11 + i12 + i2;
                        case 2
                            i11 = ceil(log2((N1 * O1)/2));
                            i12 = 0;
                            i2 = 4;
                            nBitsCsi2 = nBitsCsi2 + i11 + i12 + i2;
                        otherwise
                            error ('Invalid input: codebookMode should be 1 or 2');
                    end
                elseif (N2 > 1)
                    switch codebookMode
                        case 1
                            i11 = ceil(log2(N1 * O1));
                            i12 = ceil(log2(N2 * O2));
                            i2 = 2;
                            nBitsCsi2 = nBitsCsi2 + i11 + i12 + i2;                                
                        case 2
                            i11 = ceil(log2((N1 * O1)/2));
                            i12 = ceil(log2((N2 * O2)/2));
                            i2 = 4;
                            nBitsCsi2 = nBitsCsi2 + i11 + i12 + i2;
                        otherwise
                            error ('Invalid input: codebookMode should be 1 or 2');
                    end
                else
                    error ('Invalid input: N2 should be >=1');
                end
            else % nCsirsPorts =2
                nBitsCsi2 = nBitsCsi2 + 2; % PMI bitwidth = 2 
            end
        elseif (rank ==2)
            if nCsirsPorts == 2
                nBitsCsi2 = nBitsCsi2 + 1; % PMI bitwidth = 1 
            elseif nCsirsPorts== 4          % N2 = 1
                switch codebookMode
                    case 1
                        i11 = ceil(log2(N1 * O1));
                        i12 = ceil(log2(N2 * O2));
                        i13 = 1;                    % Information field X1 for wideband PMI
                        i2 = 1;
                        nBitsCsi2 = nBitsCsi2 + i11 + i12 + i13 + i2;
                    case 2
                        i11 = ceil(log2((N1 * O1)/2));
                        i12 = 0;
                        i13 = 1;
                        i2 = 3;
                        nBitsCsi2 = nBitsCsi2 + i11 + i12 + i13 + i2;
                    otherwise
                        error ('Invalid input: codebookMode should be 1 or 2');
                end
            else % nCsirsPorts >4
                if N2 > 1
                  switch codebookMode
                    case 1
                        i11 = ceil(log2(N1 * O1));
                        i12 = ceil(log2(N2 * O2));
                        i13 = 2;
                        i2 = 1;
                        nBitsCsi2 = nBitsCsi2 + i11 + i12 + i13 + i2;
                    case 2
                        i11 = ceil(log2((N1 * O1)/2));
                        i12 = ceil(log2((N2 * O2)/2));
                        i13 = 2;
                        i2 = 3;
                        nBitsCsi2 = nBitsCsi2 + i11 + i12 + i13 + i2;
                    otherwise
                        error ('Invalid input: codebookMode should be 1 or 2');
                  end  
                else % N2 = 1
                   switch codebookMode
                    case 1
                        i11 = ceil(log2(N1 * O1));
                        i12 = ceil(log2(N2 * O2));
                        i13 = 2;
                        i2 = 1;
                        nBitsCsi2 = nBitsCsi2 + i11 + i12 + i13 + i2;
                    case 2
                        i11 = ceil(log2((N1 * O1)/2));
                        i12 = 0;
                        i13 = 2;
                        i2 = 3;
                        nBitsCsi2 = nBitsCsi2 + i11 + i12 + i13 + i2;
                    otherwise
                        error ('Invalid input: codebookMode should be 1 or 2');
                   end  
                end
            end
        elseif ((rank ==3) || (rank ==4))
            if nCsirsPorts == 4
                i11 = ceil(log2(N1 * O1));
                i12 = ceil(log2(N2 * O2));
                i13 = 0;
                i2 = 1;
                nBitsCsi2 = nBitsCsi2 + i11 + i12 + i13 + i2;
            elseif ((nCsirsPorts == 8) || (nCsirsPorts == 12))
                i11 = ceil(log2(N1 * O1));
                i12 = ceil(log2(N2 * O2));
                i13 = 2;
                i2 = 1;
                nBitsCsi2 = nBitsCsi2 + i11 + i12 + i13 + i2;
            elseif nCsirsPorts > 16
                i11 = ceil(log2((N1 * O1)/2));
                i12 = ceil(log2(N2 * O2));
                i13 = 2;
                i2 = 1;
                nBitsCsi2 = nBitsCsi2 + i11 + i12 + i13 + i2;
            else
                error('Invalid value for nCsirsPorts');
            end
        elseif ((rank == 5) || (rank ==6))
            i11 = ceil(log2(N1 * O1));
            i12 = ceil(log2(N2 * O2));
            i2 = 1;
            nBitsCsi2 = nBitsCsi2 + i11 + i12 + i2;
        elseif ((rank == 7) || (rank ==8))
            if ((N1 == 4) && (N2 == 1))
             i11 = ceil(log2((N1 * O1)/2));
             i12 = ceil(log2(N2 * O2));
             i2 = 1;
             nBitsCsi2 = nBitsCsi2 + i11 + i12 + i2;  
            elseif ((N1 > 2) && (N2 == 2))
               i11 = ceil(log2(N1 * O1));
               i12 = ceil(log2((N2 * O2)/2));
               i2 = 1;
               nBitsCsi2 = nBitsCsi2 + i11 + i12 + i2;
            elseif (((N1 > 4) && (N2 == 1)) || ((N1 == 2) && (N2 == 2)) || ((N1 > 2) && (N2 > 2)))
                i11 = ceil(log2(N1 * O1));
                i12 = ceil(log2(N2 * O2));
                i2 = 1;
                nBitsCsi2 = nBitsCsi2 + i11 + i12 + i2; 
            else
                error ('Invalid value for N1-N2');
            end
        else 
            error ('Invalid value of rank');
        end
    otherwise
        error('Invalid input: Only type1-SinglePanel codebook is supported');
end            
return