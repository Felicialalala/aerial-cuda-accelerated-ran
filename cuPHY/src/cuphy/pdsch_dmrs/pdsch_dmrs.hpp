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

#if !defined(PDSCH_DMRS_HPP_INCLUDED_)
#define PDSCH_DMRS_HPP_INCLUDED_

#if defined(__cplusplus)
extern "C" {
#endif /* defined(__cplusplus) */

struct pdschDmrsDescr
{
    PdschDmrsParams* dmrs_params;
    int num_TBs;
};
typedef struct pdschDmrsDescr pdschDmrsDescr_t;

struct pdschCsirsPrepDescr
{
    uint16_t*          reMapArray;
    int                bufferSizeInBytes;
    cuphyCsirsRrcDynPrm_t * csirsParams;
    int                numParams;
    uint32_t*          offsets;
    uint32_t           totalOffsets;
    uint32_t*          cellIndexArray;
    PdschUeGrpParams*  ueGrpParams;
    PdschDmrsParams*   dmrsParams;
    uint16_t           maxBWP;
};
typedef struct pdschCsirsPrepDescr pdschCsirsPrepDescr_t;

cuphyStatus_t CUPHYWINAPI cuphySetupPdschCsirsPreprocessing(cuphyPdschCsirsPrepLaunchConfig_t pdschCsirsLaunchConfig,
                                                          void*        re_map_array_addr,
                                                          cuphyCsirsRrcDynPrm_t* d_params,
                                                          size_t       numParams,
                                                          uint32_t     total_offsets,
                                                          uint32_t*    d_offsets,
                                                          uint32_t*    d_cellIndex,
                                                          uint16_t                num_ue_groups,
                                                          PdschUeGrpParams*       d_ue_grp_params,
                                                          PdschDmrsParams*        d_dmrs_params,
                                                          uint16_t                max_BWP,
                                                          uint16_t                num_cells,
                                                          void*                   cpu_desc,
                                                          void*                   gpu_desc,
                                                          uint8_t                 enable_desc_async_copy,
                                                          cudaStream_t stream);

#if defined(__cplusplus)
} /* extern "C" */
#endif /* defined(__cplusplus) */

#endif // PDSCH_DMRS_HPP_INCLUDED_
