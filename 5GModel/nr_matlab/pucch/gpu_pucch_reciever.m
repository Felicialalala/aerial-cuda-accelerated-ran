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

function b_est = gpu_pucch_reciever(Y, nUe_pucch,Pucch_common,Pucch_ue_cell,Pucch_receiver,sp)


% function applies pucch recieve pipeline to base station recieved signal

%inputs:
% Y             --> recieved signal. Dim: Nf x Nt x L_BS
% nUe_pucch     --> number of pucch users. (1-42).
% Pucch_common  --> pucch paramaters shared by all users
% Pucch_ue_cell --> cell containing user specific pucch paramaters. Dim: nUe_pucch x 1
% Pucch_receiver --> structure containing filters and sequences needed by pucch receiver

%outputs:
% b_est  --> estimates of transmitted bit(s). Dim: nUe_pucch x 2

% Test vector in HDF5 file format
% TODO FIXME: HDF5 file generated only for (last slot, SNR step).
% Could generate a separate HDF5 file per (SNR step, slot), if needed.
save_str = strcat('./GPU_test_input/',sp.sim.opt.testCase,'.h5');
tvName = sprintf(save_str);
h5File  = H5F.create([tvName], 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');
hdf5_write_nv(h5File, 'Y_input', single(Y)); % Input to PUCCH receiver, complex number, single precision, matlab layout: {Nf x Nt x L_Bs}
hdf5_write_nv(h5File, 'nUe_pucch', uint32(nUe_pucch)); % number of UEs

% HDF5 file - Dataset values shared across all antennas. TODO potentially use a compound type
hdf5_write_nv(h5File, 'startSym', uint32(Pucch_common.startSym));  % start symbol, [0, 10] for PUCCH format 1
hdf5_write_nv(h5File, 'nSym', uint32(Pucch_common.nSym)); % total number of symbols: [4, 14] for PUCCH format 1
hdf5_write_nv(h5File, 'prbIdx', uint32(Pucch_common.prbIdx)); % index of current PUCCH PRB
hdf5_write_nv(h5File, 'u', uint32(Pucch_common.u)); % index of low-PARP sequence [0, 29]
hdf5_write_nv(h5File, 'L_BS', uint32(Pucch_common.L_BS)); % number of base station antennas

hdf5_write_nv(h5File, 'mu', uint32(Pucch_common.mu)); % numerology
hdf5_write_nv(h5File, 'slotNumber', uint32(Pucch_common.slotNumber));
hdf5_write_nv(h5File, 'hoppingId', uint32(Pucch_common.hoppingId));


%%
%PARAMATERS

%Pucch_common
startSym = Pucch_common.startSym;     % index of starting pucch symbol (c 0 indexing!) (0-10) 
nSym = Pucch_common.nSym;             % number of pucch symbols. (1 - 14)
nSym_data = Pucch_common.nSym_data;   % number of pucch data symbols. (1-7)
nSym_dmrs = Pucch_common.nSym_dmrs;   % number of pucch dmrs symbols. (1-7)
prbIdx = Pucch_common.prbIdx;         % index of pucch prb (c 0 indexing!)
u = Pucch_common.u;                   % index of low-papr sequence (c 0 indexing!) (0-29)
L_BS = Pucch_common.L_BS;             % number of base-station antennas. (1,2,4,8, or 16).

%Pucch_receiver
cs_freq = Pucch_receiver.cs_freq;     % frequency representation of cyclic shifts. Dim: 12 x 12
r = Pucch_receiver.r;                 % low papr pucch sequences. Dim: 12 x 30
tOCC_cell = Pucch_receiver.tOCC_cell; % cell containing time orthogonal covering codes. Dim: 7 x 1
Wf = Pucch_receiver.Wf;               % frequency ChEst filter (real valued). Dim: 12 x 12
s = Pucch_receiver.s;                 % delay shift sequence. Dim: 12 x 1
Wt_cell = Pucch_receiver.Wt_cell;     % Cell of time ChEst filters (real valued). Dim: 11 x 1

hdf5_write_nv(h5File, 'Wf', single(real(Pucch_receiver.Wf))); % 12 x 12; keep real part

%Writing cell (matlab type) to an HDF5 file
for i = 1 : 11
    wt_dataset_name = sprintf('Wt_cell_%d', i-1);
    hdf5_write_nv(h5File, wt_dataset_name, single(real(Pucch_receiver.Wt_cell{i})));
end


%%
%LOAD SEQUENCES/FILTERS

% load cell-specific low papr sequence:
r = r(:,u+1);

% load data/dmrs tOCC:
tOCC_data = tOCC_cell{nSym_data};
tOCC_dmrs = tOCC_cell{nSym_dmrs};

% load time filter:
Wt = Wt_cell{nSym - 3};

%%
%PUCCH COMMMON

% steps 1-3 are performed once for all users

%%
%STEP 1

% extract pucch signal from total recieved signal 
%(careful with difference between matlab 1 indexing and c 0 index)

freq_idx = (prbIdx*12 + 1) : 12*(prbIdx + 1);
time_idx = (startSym + 1) : (startSym + nSym);

Y_pucch = Y(freq_idx,time_idx,:);

%%
%STEP 2

% seperate pucch signal into dmrs and data signals

Y_dmrs = zeros(12,nSym_dmrs,L_BS);
for i = 1 : nSym_dmrs
    Y_dmrs(:,i,:) = Y_pucch(:,2*(i-1) + 1,:);
end

Y_data = zeros(12,nSym_data,L_BS);
for i = 1 : nSym_data
    Y_data(:,i,:) = Y_pucch(:,2*(i-1) + 2,:);
end

%%
%STEP 3

% remove cell-specific covering code. Center dmrs signal.
for i = 1 : nSym_dmrs
    for bs = 1 : L_BS
        Y_dmrs(:,i,bs) = conj(r) .* s .* Y_dmrs(:,i,bs);
    end
end

for i = 1 : nSym_data
    for bs = 1 : L_BS
        Y_data(:,i,bs) = conj(r) .* Y_data(:,i,bs);
    end
end

% Intermediate output for debugging purposes.
hdf5_write_nv(h5File, 'Y_dmrs_after_step3', single(Y_dmrs)); % complex numbers
hdf5_write_nv(h5File, 'Y_data_after_step3', single(Y_data)); % complex numbers

%%
%PUCCH USER SPECIFIC

% steps 4-6 are perform seperatly for each user (can be done in parallel)

% store bit estimates:
b_est = zeros(nUe_pucch,2);

for iue = 1 : nUe_pucch

    % extract user specific pucch paramaters:
    Pucch_ue = Pucch_ue_cell{iue};
    cs = Pucch_ue.cs;            % index of cyclic shifts. Dim: nSym x 1. (c 0 indexing!)
    tOCCidx = Pucch_ue.tOCCidx;  % index of time cover-code. (c 0 index!)
    nBits = Pucch_ue.nBits;      % 1 or 2. Number of transmitted bits

    % UE-specific information; separate HDF5 file dataset per UE.
    pucch_ue_dataset_name = sprintf('pucch_ue_%d', iue-1);
    hdf5_write_nv(h5File, strcat(pucch_ue_dataset_name, '_tOCCidx'), uint32(tOCCidx));
    hdf5_write_nv(h5File, strcat(pucch_ue_dataset_name, '_nBits'), uint32(nBits)); %uint32 in [1, 2]
    hdf5_write_nv(h5File, strcat(pucch_ue_dataset_name, '_cs0'), uint32(Pucch_ue.cs0));
    
    %%
    %STEP 4
    
    % remove user specific time and frequency codes
    Y_dmrs_iue = zeros(12,nSym_dmrs,L_BS);

    % Debugging
    %cs_freq_dmrs_vals = zeros(12, nSym_dmrs, L_BS);
    %cs_times = zeros(12, nSym_dmrs, L_BS);

    for i = 1 : nSym_dmrs
        for bs = 1 : L_BS
            Y_dmrs_iue(:,i,bs) = conj(tOCC_dmrs(tOCCidx+1,i)) * conj(cs_freq(:,cs(2*(i-1)+1)+1)) .* ...
                Y_dmrs(:,i,bs);
            cs_freq_dmrs_vals(:,i,bs) = cs_freq(:,cs(2*(i-1)+1)+1);
            cs_times_vals(:,i,bs) = cs(2*(i-1)+1)+1;
        end
    end

    % Intermediate output for debugging purposes.
    %hdf5_write_nv(h5File, strcat(pucch_ue_dataset_name, '_cs_freq_dmrs_vals'), single(cs_freq_dmrs_vals));
    %hdf5_write_nv(h5File, strcat(pucch_ue_dataset_name, '_cs_times_vals'), single(cs_times_vals));
    
    Y_data_iue = zeros(12,nSym_data,L_BS);
    for i = 1 : nSym_data
        for bs = 1 : L_BS
            Y_data_iue(:,i,bs) = conj(tOCC_data(tOCCidx+1,i)) * conj(cs_freq(:,cs(2*(i-1)+2)+1)) .* ...
                Y_data(:,i,bs);
        end
    end

    % Intermediate output for debugging purposes
    %pucch_ue_dataset_name = sprintf('Y_ue_%d', iue-1);
    %hdf5_write_nv(h5File, strcat(pucch_ue_dataset_name, '_dmrs_after_step4'), single(Y_dmrs_iue)); % complex numbers
    %hdf5_write_nv(h5File, strcat(pucch_ue_dataset_name, '_data_after_step4'), single(Y_data_iue)); % complex numbers
    
    %%
    %STEP 5
    
    % estimate users channel on the data symbols, undo centering
    
    H_est_iue = zeros(12,nSym_data,L_BS);
    %H_est_iue_part1  = zeros(12,nSym_dmrs,L_BS);
    temp_Wt = Wt;
    
    for bs = 1 : L_BS
        H_est_iue(:,:,bs) = (Wf * Y_dmrs_iue(:,:,bs)) * Wt;
        H_est_iue_part1(:,:,bs) = (Wf * Y_dmrs_iue(:,:,bs));
    end

    % Intermediate output for debugging purposes
    %pucch_hest_dataset_name = sprintf('pre_shift_Hest_ue_%d', iue-1);
    %hdf5_write_nv(h5File, strcat(pucch_hest_dataset_name, '_in_step5'), single(H_est_iue));

    %pucch_hest_dataset_name = sprintf('pre_shift_Hest_ue_%d', iue-1);
    %hdf5_write_nv(h5File, strcat(pucch_hest_dataset_name, '_part1_in_step5'), single(H_est_iue_part1));
    
    for i = 1 : nSym_data
        for bs = 1 : L_BS
            H_est_iue(:,i,bs) = conj(s) .* H_est_iue(:,i,bs);
        end
    end

    %pucch_hest_dataset_name = sprintf('Hest_ue_%d', iue-1);
    %hdf5_write_nv(h5File, strcat(pucch_hest_dataset_name, '_after_step5'), single(H_est_iue));

    %%
    %STEP 6
    
    % estimate users bits
    
    m = Y_data_iue .* conj(H_est_iue);
    qam_est = sum(m(:));

    % Each per-UE dataset contains a single complex number. Used for debugging purposes
    pucch_qam_est_name = sprintf('qam_est_ue_%d', iue-1);
    hdf5_write_nv(h5File, pucch_qam_est_name, single(qam_est));
    
    % for two bits, slice the real and imag axis
    if nBits == 2
        if real(qam_est) <= 0
            b_est(iue,1) = 1;
        else
            b_est(iue,1) = 0;
        end

        if imag(qam_est) <= 0
            b_est(iue,2) = 1;
        else
            b_est(iue,2) = 0;
        end
    end
    
    % for one bit slice the 1 + 1i axis
    if nBits == 1
        a = 1 - 1i;
        qam_est = a*qam_est;

        if real(qam_est) <= 0
            b_est(iue,1) = 1;
        else
            b_est(iue,1) = 0;
        end
    end
end

% PUCCH receiver's final output. Unused bits are set to 0.
hdf5_write_nv(h5File, 'bit_est_step6', uint32(b_est));
H5F.close(h5File);
