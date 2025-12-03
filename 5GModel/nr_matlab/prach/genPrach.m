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

function Xtf = genPrach(pdu, table, carrier, Xtf)
% function prach = genPreambleSeq(prach, carrier, nodeType)
%
% This function generates PRACH paramters, ZC sequence and preamble in 
% freq domain.
%
% Input:    prach: prach related configuration 
%           carrier: carrier related configuration
%           nodeType: 'UE' or 'gNB'
%
% Output:   prach: add fields for PRACH paramters, ZC sequence and 
%           preamble in freq domain
%

PrachParams = derive_prach_params(pdu, table, carrier);

Xtf = genPrach_cuphy(PrachParams, Xtf);

global SimCtrl;

if SimCtrl.prachFalseAlarmTest
    Xtf = Xtf * 0;
end

idxSlot = carrier.idxSlotInFrame;
idxPdu = pdu.prachPduIdx-1;

if SimCtrl.genTV.enableUE && SimCtrl.genTV.cuPHY && ...
        ismember(idxSlot, SimCtrl.genTV.slotIdx)
    TVname = [SimCtrl.genTV.TVname, '_UE_CUPHY_s', num2str(idxSlot),...
        'p', num2str(idxPdu)];
    saveTV_prach(SimCtrl.genTV.tvDirName, TVname,  PrachParams, Xtf);
    SimCtrl.genTV.idx = SimCtrl.genTV.idx + 1;
end

return


function PrachParams = derive_prach_params(pdu, table, carrier)

prachTable = table;

% read input from carrier and prach
FR = carrier.FR;
duplex = carrier.duplex;
mu = carrier.mu;
prachCfgIdx = pdu.configurationIndex;
restrictedSet = pdu.restrictedSet;
rootSequenceIndex = pdu.rootSequenceIndex;
% zeroCorrelationZone = pdu.zeroCorrelationZone;
prmbIdx = pdu.prmbIdx;
preambleFormat = pdu.prachFormat;
N_CS = pdu.numCs;

% find dela_f_RA and L_RA
switch preambleFormat
    case {'0', '1', '2'}
        delta_f_RA = 1250;
        L_RA = 839;
    case '3'
        delta_f_RA = 5000;
        L_RA = 839;
    otherwise
        delta_f_RA = 15000*2^mu;
        L_RA = 139;
end    

switch preambleFormat
    case {'0'}
        Nrep = 1;
    case 'B4'
        Nrep = 11;
    case '1'
        Nrep = 2;
    otherwise
        error('preambleFormat is not supported ...\n');
end 

% % find N_CS from zeroCorrelationZone
% switch delta_f_RA
%     case 1250
%         switch restrictedSet
%             case 0
%                 typeIdx = 1;
%             case 1
%                 typeIdx = 2;
%             case 2
%                 typeIdx = 3;
%             otherwise
%                 error('restrictedSet is not supported ...\n');
%         end
%         % 3GPP TS 38.211 V15.4.0 Table 6.3.3.1-5
%         N_CS_table = prachTable.table_NCS_1p25k;
%         N_CS = N_CS_table(zeroCorrelationZone + 1, typeIdx);        
%     case 5000
%         switch restrictedSet
%             case 0
%                 typeIdx = 1;
%             case 1
%                 typeIdx = 2;
%             case 2
%                 typeIdx = 3;
%             otherwise
%                 error('restrictedSet is not supported ...\n');                
%         end
%         % 3GPP TS 38.211 V15.4.0 Table 6.3.3.1-6
%         N_CS_table = prachTable.table_NCS_5k;
%         N_CS = N_CS_table(zeroCorrelationZone + 1, typeIdx); 
%     otherwise
%         % 3GPP TS 38.211 V15.4.0 Table 6.3.3.1-7
%         N_CS_table = prachTable.table_NCS_15kplus;
%         N_CS = N_CS_table(zeroCorrelationZone + 1);
% end

% load logIdx2u table for logical root mapping
if L_RA == 839
    % 3GPP TS 38.211 V15.4.0 Table 6.3.3.1-3
    logIdx2u_table = prachTable.table_logIdx2u_839;
elseif L_RA == 139
    % 3GPP TS 38.211 V15.4.0 Table 6.3.3.1-4
    logIdx2u_table = prachTable.table_logIdx2u_139;
else
    error('L_RA length is not supported ... \n');
end

% generate ZC sequence and preamble
 
% calculate u and C_v
[u, C_v] = findZcPar(prmbIdx, rootSequenceIndex, ...
    L_RA, restrictedSet, N_CS, logIdx2u_table);

PrachParams.L_RA = L_RA;
PrachParams.C_v = C_v;
PrachParams.u = u;
PrachParams.Nrep = Nrep;

return

function Xtf = genPrach_cuphy(PrachParams, Xtf)

%
% generate ZC sequence and preamble in freq domain 
%

L_RA = PrachParams.L_RA;
C_v = PrachParams.C_v;
u = PrachParams.u;
Nrep = PrachParams.Nrep;

genZcAlg = 1;

[~, q] = findDu(u, L_RA);

i = [0:L_RA-1];
x_u = exp(-1j*pi*u*i.*(i+1)/L_RA);
n = [0:L_RA-1];
x_uv = x_u(mod((n+C_v), L_RA)+1);       

switch genZcAlg 
    case 0 % normal method
        m = [0:L_RA-1];
        for nn = 0:L_RA-1
            y_uv(nn+1) = sum(x_uv(m+1).*exp(-1j*2*pi*m*nn/L_RA));
        end
        y_uv = y_uv/sqrt(L_RA);
    case 1
        % Simplied method
        % Use the closed-form expressions in https://goo.gl/rjzvJn 
        m = [0:L_RA-1];
        y_uv = conj(x_u(mod(q*m+C_v,L_RA)+1));
        y_uv = sum(x_u)*x_u(mod(C_v,L_RA)+1)*y_uv/sqrt(L_RA);          
    otherwise
        error('genZcAlg is not supperted ...\n');
end

y_uv = repmat(y_uv, [1, Nrep]);        % now: Nf_srs x nSym x numAntPorts 

Xtf(1:length(y_uv),1) = y_uv;

return


function saveTV_prach(tvDirName, TVname, PrachParams, Xtf)

[status,msg] = mkdir(tvDirName);

PrachParams.L_RA = uint32(PrachParams.L_RA);
PrachParams.C_v = uint32(PrachParams.C_v);
PrachParams.u = uint32(PrachParams.u);

h5File  = H5F.create([tvDirName filesep TVname '.h5'], 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');
hdf5_write_nv(h5File, 'PrachParams', PrachParams);
hdf5_write_nv(h5File, 'X_tf', single(Xtf));
H5F.close(h5File);

return


