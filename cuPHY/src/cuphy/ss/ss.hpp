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

#if !defined(SS_HPP_INCLUDED_)
#define SS_HPP_INCLUDED_

#if defined(__cplusplus)
extern "C" {
#endif /* defined(__cplusplus) */

/**
 * @brief: Generate time/freq domain subcarriers for SSB pipeline.
 * @param[in]  d_x_tx: pointer to the PBCH rate matched output
 * @param[out] d_tfSignal: array of all num_cells output buffers; one buffer per cell
 * @param[in]  d_ssb_params: SSB parameters for all num_SSB SSBs
 * @param[in] d_per_cell_params: cell specific parameters for all num_cells cells
 * @param[in] num_SSBs: number of SSBs
 * @param[in] num_cells: number of cells
 * @param[in] stream: CUDA stream for kernel launch
 * @return CUPHY_STATUS_SUCCESS or CUPHY_STATUS_INVALID_ARGUMENT or CUPHY_STATUS_INTERNAL_ERROR.
 */
cuphyStatus_t cuphyRunSsbMapper(const uint8_t*                  d_x_tx,
                                __half2**                       d_tfSignal,
                                const cuphyPerSsBlockDynPrms_t* d_ssb_params,
                                const cuphyPerCellSsbDynPrms_t* d_per_cell_params,
                                const cuphyPmWOneLayer_t*       d_pmw_params,
                                uint16_t                        num_SSBs,
                                uint16_t                        num_cells,
                                cudaStream_t                    stream,
                                cuphySsbMapperLaunchCfg_t*      pSsbMapperCfg);

/**
 * @brief: Set kernel launch configurations for SSB Mapper kernel
 * @param[in, out] pLaunchCfg: pointer to launch configuration for SSB mapper kernel
 * @param[in] num_SSBs: number of SSBs
 * @return CUPHY_STATUS_SUCCESS or CUPHY_STATUS_INVALID_ARGUMENT.
 */
cuphyStatus_t kernelSelectSsbMapper(cuphySsbMapperLaunchCfg_t* pLaunchCfg,
                                    uint16_t                   num_SSBs);

#if defined(__cplusplus)
} /* extern "C" */
#endif /* defined(__cplusplus) */

#endif // SS_HPP_INCLUDED_
