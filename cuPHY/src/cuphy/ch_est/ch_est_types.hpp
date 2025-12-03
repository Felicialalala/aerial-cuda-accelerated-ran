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

#ifndef CUPHY_CH_EST_TYPES_HPP
#define CUPHY_CH_EST_TYPES_HPP

namespace ch_est  {

// PRB RKHS descriptor
struct prbRkhsDesc_t final
{
    uint8_t   zpIdx;          // zero padding index, used to load pZpDmrsScEigenVec and pZpInterpVec
    __half    sumEigValues;   // sum of prb eignValues

    cuphyTensorInfo2_t  tInfoEigVecCob;            // eigenvector change-of-basis
    cuphyTensorInfo2_t  tInfoCorr_half_nZpDmrsSc;  // correlation between eignevectors seperated by nZpDmrsSc / 2 intervals. Dim: nEigs x nEigs
    cuphyTensorInfo3_t  tInfoCorr;                 // correlation between eigenvectors seperated by 0 to (nCpInt - 1) intervals. Dim: nEigs x nEigs x nCpInt
    cuphyTensorInfo1_t  tInfoEigVal;               // Prb eigenvalues. Dim: nEigs x 1
    cuphyTensorInfo2_t  tInfoInterpCob;            // change of basis between prb and interpolation eigenvectors. Dim: nIterpEigs x nEigs
};

// zero-padded RKHS descriptor
struct zpRkhsDesc_t final
{
    cuphyTensorInfo2_t tInfoZpDmrsScEigenVec;          // zero-padded eigenvector tensor
    cuphyTensorInfo2_t tInfoZpInterpVec;               // interpolation eigenvector tensor
    cuphyTensorInfo2_t tInfoSecondStageTwiddleFactors; // twiddle factors for second stage of FFT
    cuphyTensorInfo1_t tInfoSecondStageFourierPerm;    // input permutation for input to second stage Fourier transform
};

// Tensor parameters needed to access Channel estimator input/output tensors
template <size_t NDim>
struct puschRxChEstTensorPrm final
{
    void* pAddr;
    int   strides[NDim];
};
template <size_t NDim>
using puschRxChEstTensorPrm_t = puschRxChEstTensorPrm<NDim>;

// Channel estimator static descriptor
struct puschRxChEstStatDescr_t final
{
    puschRxChEstTensorPrm_t<CUPHY_PUSCH_RX_CH_EST_N_DIM_FREQ_INTERP_COEFS> tPrmFreqInterpCoefs;       // (N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER + N_INTER_DMRS_GRID_FREQ_SHIFT, N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER, 3), 3 filters: 1 for middle, 1 lower edge and 1 upper edge
    puschRxChEstTensorPrm_t<CUPHY_PUSCH_RX_CH_EST_N_DIM_FREQ_INTERP_COEFS> tPrmFreqInterpCoefs4;      // 25 x 24 x 3
    puschRxChEstTensorPrm_t<CUPHY_PUSCH_RX_CH_EST_N_DIM_FREQ_INTERP_COEFS> tPrmFreqInterpCoefsSmall;  // 37 x 18 x 3

    puschRxChEstTensorPrm_t<CUPHY_PUSCH_RX_CH_EST_N_DIM_SHIFT_SEQ>         tPrmShiftSeq;         // (N_DATA_PRB*N_DMRS_GRID_TONES_PER_PRB, N_DMRS_SYMS)
    puschRxChEstTensorPrm_t<CUPHY_PUSCH_RX_CH_EST_N_DIM_UNSHIFT_SEQ>       tPrmUnShiftSeq;       // (N_DATA_PRB*N_DMRS_INTERP_TONES_PER_GRID*N_DMRS_GRIDS_PER_PRB + N_INTER_DMRS_GRID_FREQ_SHIFT)

    puschRxChEstTensorPrm_t<CUPHY_PUSCH_RX_CH_EST_N_DIM_SHIFT_SEQ>         tPrmShiftSeq4;        // Small shift sequence. Dim: 24 x 1
    puschRxChEstTensorPrm_t<CUPHY_PUSCH_RX_CH_EST_N_DIM_SHIFT_SEQ>         tPrmUnShiftSeq4;      // Small un-shift sequence. Dim: 49 x 1

    const uint32_t *                                                       pSymbolRxStatus;      // Pointer to GPU array indicating if symbol received for all cells scheduled for each time slot

    // RKHS paramaters:
    prbRkhsDesc_t  prbRkhsDescs[MAX_N_PRBS_SUPPORTED];
    zpRkhsDesc_t   zpRkhsDescs[NUM_RKHS_ZP];
};

struct foccPrm_t final
{
    uint8_t layerIdx;
};

struct toccPrm_t final
{
    uint8_t   foccBitMask;
    foccPrm_t foccPrms[2];
};

struct gridPrm_t final
{
    uint8_t   toccBitMask;
    toccPrm_t toccPrms[2];
};

enum rkhsNoiseEstMethod_t
{
    USE_EMPTY_DMRS_GRID    = 0,
    USE_EMPTY_FOCC         = 1,
    USE_QUITE_FOCC_REGIONS = 2
};

struct computeBlocksCommonPrms_t final
{
    uint16_t nPrb;     // number of PRBs in the compute block
    uint8_t  zpIdx;    // zero-padding index

    rkhsNoiseEstMethod_t noiseEstMethod;         // method to measure noise
    float                nNoiseMeasurments;      // total number of noise measurments
    uint16_t             noiseRegionFirstIntIdx; // first time interval for noise measurments
    uint16_t             nNoiseIntsPerFocc;      // per-focc number of time intervals for noise measurment
    uint16_t             nNoiseIntsPerGrid;      // per-grid number of time intervals for noise measurment
};

struct __align__(32) rkhsUeGrpPrms_t
{
    uint8_t    gridBitMask;
    gridPrm_t  gridPrms[2];
    uint8_t    gridIdxs[MAX_N_LAYERS_PUSCH];

    computeBlocksCommonPrms_t computeBlocksCommonPrms;
};

struct __align__(32) rkhsComputeBlockPrms_t
{
    uint16_t ueGrpIdx;                 // ueGrp that compute block belongs to
    uint16_t startInputPrb;            // first prb input to compute block
    uint16_t nOutputSc;                // number of sc output by compute block
    uint16_t startOutputScInBlock;     // fist sc outputed within compute block
    uint16_t scOffsetIntoChEstBuff;    // sc output offset into chEst buffer
};

// Channel estimator dynamic descriptor
struct puschRxChEstDynDescr_t final
{
    uint8_t                   chEstTimeInst; // Time domain instance of channel estimation
    uint8_t                   dmrsSymPos[N_MAX_DMRS_SYMS];
    cuphyPuschRxUeGrpPrms_t*  pDrvdUeGrpPrms;
    uint32_t                  hetCfgUeGrpMap[MAX_N_USER_GROUPS_SUPPORTED]; // Mapping of Heterogenous config to UE group
    uint8_t*                  pPreSubSlotWaitKernelStatusGpu;
    uint8_t*                  pPostSubSlotWaitKernelStatusGpu;
    uint16_t                  waitTimeOutPreEarlyHarqUs;       // timeout threshold for wait kernel prior to starting early HARQ processing
    uint16_t                  waitTimeOutPostEarlyHarqUs;      // timeout threshold for wait kernel after finishing early HARQ processing
    uint64_t                  mPuschStartTimeNs;      // start time as reference point to measure timeout
    rkhsUeGrpPrms_t           rkhsUeGrpPrms[MAX_N_USER_GROUPS_SUPPORTED];
    rkhsComputeBlockPrms_t    rkhsCompBlockPrms[MAX_N_USER_GROUPS_SUPPORTED];
    uint8_t                   nSymPreSubSlotWaitKernel; // the number of OFDM symbols to wait in pre-sub-slot-processing wait kernel
    uint8_t                   nSymPostSubSlotWaitKernel; // the number of OFDM symbols to wait in post-sub-slot-processing wait kernel
};

// Channel estimator kernel arguments (supplied via descriptors)
typedef struct _puschRxChEstKernelArgs
{
    // puschRxDescrs_t const* pPuschRxDescrs;
    // puschRxChEstStatDescr_t const* pStatDescr;
    puschRxChEstStatDescr_t* pStatDescr; // pointer to an array of static descriptors (1 to CUPHY_PUSCH_RX_CH_EST_N_HOM_CFG)
    puschRxChEstDynDescr_t*  pDynDescr;  // pointer to an array of CUPHY_PUSCH_RX_CH_EST_N_HOM_CFG dynamic descriptors
} puschRxChEstKernelArgs_t;

// Forward declaration for launch parameters
using puschRxChEstKernelArgsArr_t = std::array<puschRxChEstKernelArgs_t, CUPHY_PUSCH_RX_CH_EST_ALL_ALGS_N_MAX_HET_CFGS>;
using puschRxChEstDynDescrVec_t   = std::array<puschRxChEstDynDescr_t, CUPHY_PUSCH_RX_CH_EST_ALL_ALGS_N_MAX_HET_CFGS>;

} // namespace ch_est

#endif //CUPHY_CH_EST_TYPES_HPP
