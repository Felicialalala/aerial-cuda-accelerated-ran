/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _CUMAC_CP_TV_HPP_
#define _CUMAC_CP_TV_HPP_

#include <unistd.h>

#include "api.h"
#include "cumac_app.hpp"
#include "cumac_pfm_sort.h"

#include "hdf5hpp.hpp"
#include "cuphy_hdf5.hpp"
#include "nvlog.hpp"

#include <chrono>

#define CONFIG_CUMAC_TV_PATH "testVectors/cumac/"

typedef struct cumac_cp_tv
{
    uint32_t parsed = 0;

    struct cumac::cumacSchedulerParam params{};
    cumac_buf_num_t buf_num{};

    float *avgRates = nullptr;
    uint8_t *cellAssoc = nullptr;
    uint8_t *cellAssocActUe = nullptr;
    int8_t *tbErrLast = nullptr;
    uint16_t *cellId = nullptr;

    // data buffer pointers
    uint16_t *CRNTI = nullptr;                 // C-RNTIs of all active UEs in the cell
    uint16_t *srsCRNTI = nullptr;              // C-RNTIs of the UEs that have refreshed SRS channel estimates in the cell.
    uint8_t *prgMsk = nullptr;                 // Bit map for the availability of each PRG for allocation
    float *postEqSinr = nullptr;               // Array of the per-PRG per-layer post-equalizer SINRs of all active UEs in the cell
    float *wbSinr = nullptr;                   // Array of wideband per-layer post-equalizer SINRs of all active UEs in the cell
    cuComplex *estH_fr = nullptr;              // For FP32. Array of the subband (per-PRG) SRS channel estimate coefficients for all active UEs in the cell
    cuComplex *estH_fr_half = nullptr;         // For FP16. Array of the subband (per-PRG) SRS channel estimate coefficients for all active UEs in the cell
    cuComplex *prdMat = nullptr;               // Array of the precoder/beamforming weights for all active UEs in the cell
    cuComplex *detMat = nullptr;               // Array of the detector/beamforming weights for all active UEs in the cell
    float *sinVal = nullptr;                   // Array of the per-UE, per-PRG, per-layer singular values obtained from the SVD of the channel matrix
    float *avgRatesActUe = nullptr;            // Array of the long-term average data rates of all active UEs in the cell
    uint16_t *prioWeightActUe = nullptr;       // For priority-based UE selection. Priority weights of all active UEs in the cell
    int8_t *tbErrLastActUe = nullptr;          // TB decoding error indicators of all active UEs in the cell
    int8_t *newDataActUe = nullptr;            // Indicators of initial transmission/retransmission for all active UEs in the cell
    int16_t *allocSolLastTxActUe = nullptr;    // The PRG allocation solution for the last transmissions of all active UEs in the cell
    int16_t *mcsSelSolLastTxActUe = nullptr;   // MCS selection solution for the last transmissions of all active UEs in the cell
    uint8_t *layerSelSolLastTxActUe = nullptr; //

    uint16_t *setSchdUePerCellTTI = nullptr; // Set of IDs of the selected UEs for the cell
    int16_t *allocSol = nullptr;             // PRB group allocation solution for all active UEs in the cell
    int16_t *mcsSelSol = nullptr;            // MCS selection solution for all active UEs in the cell
    uint8_t *layerSelSol = nullptr;          // Layer selection solution for all active UEs in the cell

    std::vector<cumac_pfm_cell_info_t> pfmCellInfo; // PFM sorting input buffer
    std::vector<cumac_pfm_output_cell_info_t> pfmSortSol; // PFM sorting output buffer
} cumac_cp_tv_t;

int parse_tv_file(cumac_cp_tv_t &tv, std::string tv_file);
int parse_group_tv(cumac_cp_tv_t &tv, int cell_num);
int check_bytes(const char *name1, const char *name2, void *buf1, void *buf2, size_t nbytes);
cumac_cp_tv_t *get_cumac_tv_ptr();

bool pfm_load_tv_H5(const std::string& tv_name, std::vector<cumac_pfm_cell_info_t>& pfm_cell_info, std::vector<cumac_pfm_output_cell_info_t>& pfm_output_cell_info);
bool pfm_validate_tv_h5(const std::string& tv_name, const std::vector<cumac_pfm_cell_info_t>& pfm_cell_info, const std::vector<cumac_pfm_output_cell_info_t>& pfm_output_cell_info);

#define CUMAC_VALIDATE_BYTES(buf1, buf2, nbytes) check_bytes(#buf1, #buf2, (buf1), (buf2), nbytes)

#endif // _CUMAC_CP_TV_HPP_