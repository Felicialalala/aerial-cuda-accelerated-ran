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

#if !defined(CSIRSRX_HPP_INCLUDED_)
#define CSIRSRX_HPP_INCLUDED_

#if defined(__cplusplus)
extern "C" {
#endif /* defined(__cplusplus) */

/**
 * @brief Struct that tracks all necessary parameters for CSI-RS computation.
 */
typedef struct _CsirsRxParams
{
    uint16_t       startRb;                             /*!< RB where this CSI resource starts. Expected value < 273*/
    uint16_t       nRb;                                 /*!< Number of RBs across which this CSI resource spans. Expected value <= 273-startRb */
    uint8_t        row;                                 /*!< Row entry into the CSI resource location table. Valid values 1-18 */
    uint8_t        freqDensity;
    uint8_t        li[2];                               /*!< Time domain location L0 and L1. 0 <= Valid value < OFDM_SYMBOLS_PER_SLOT */
    uint8_t        seqIndexCount;                       /*!< Index count in seq table for given CDM type */
    uint8_t        genEvenRB;                           /*!< Used for rho = 0.5. Value 0 or 1. */
    uint8_t        idxSlotInFrame;                      /*!< slot index in frame */
    uint16_t       scrambId;                            /*!< ScramblingId of CSI-RS */
    uint8_t        ki[CUPHY_CSIRS_MAX_KI_INDEX_LENGTH]; /*!< reference location of CSI-RS in frequency domain (k0,k1,k2,k3,k4,k5) */
    cuphyCsiType_t csiType;                             /*!< CSI Type, 0: TRS, 1: CSI-RS NZP, 2: CSI-RS ZP. Only  CSI-RS NZP supported currently */
    cuphyCdmType_t cdmType;                             /*!< CDM Type, 0: noCDM, 1: fd-CDM2, 2: cdm4-FD2-TD2, 3: cdm8-FD2-TD4 */
    float          alpha;                               /*!< alpha = (X==1) ? rho : 2*rho */
    float          rho;                                 /*!< density*/
    float          beta;                                /*!< Power scaling factor */
    uint16_t       cell_index;
    uint8_t        enablePrcdBf;                        /*!< Enable pre-coding for this CSI-RS*/
    uint16_t       pmwPrmIdx;                           /*!< Index to pre-coding matrix array, i.e., to the pPmwParams array of the
                                                             cuphyCsirsDynPrms_t struct */
} CsirsRxParams;


typedef struct _CsirsRxUeParams
{
    uint16_t cellIdx;
    uint32_t csirsRxParamsStartIdx;
    uint32_t csirsRxChEstBufStartIdx;
    uint16_t nCsirs; 
    uint8_t  nRxAnt; 
    uint16_t nRxRe;
    
} CsirsRxUeParams;


void kernelSelectCsirsRxChEst(cuphyCsirsRxChEstLaunchCfg_t* pLaunchCfg, int numUes, int maxNumRrcParamPerUe, int maxNumPrbPerUe);

#if defined(__cplusplus)
} /* extern "C" */
#endif /* defined(__cplusplus) */

#endif // CSIRSRX_HPP_INCLUDED_
