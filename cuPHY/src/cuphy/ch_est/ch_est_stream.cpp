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

#include "cuphy.hpp"

#include "ch_est_stream.hpp"

namespace ch_est {
void launchKernelsImpl(const cudaStream_t                                            stream,
                       const gsl_lite::span<const cuphyPuschRxChEstLaunchCfgs_t>          launchCfgs,
                       const std::uint32_t                                           startChEstInstIdx,
                       const CUDA_KERNEL_NODE_PARAMS cuphyPuschRxChEstLaunchCfg_t::* kernelDriver)
{
    for(uint32_t chEstInstIdx = startChEstInstIdx; chEstInstIdx < CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST; ++chEstInstIdx)
    {
        const auto& cfg  = launchCfgs[chEstInstIdx];
        for(uint32_t hetCfgIdx = 0; hetCfgIdx < cfg.nCfgs; ++hetCfgIdx)
        {
            const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = cfg.cfgs[hetCfgIdx].*kernelDriver;
            if(!kernelNodeParamsDriver.func) { continue; }
            CU_CHECK_EXCEPTION(launch_kernel(kernelNodeParamsDriver, stream));
        }
    }
}

void launchKernels0SlotImpl(const cudaStream_t                                            stream,
                            const gsl_lite::span<const cuphyPuschRxChEstLaunchCfgs_t>          launchCfgs,
                            const CUDA_KERNEL_NODE_PARAMS cuphyPuschRxChEstLaunchCfg_t::* kernelDriver)
{
    const auto& cfg  = launchCfgs[0];
    for(uint32_t hetCfgIdx = 0; hetCfgIdx < cfg.nCfgs; ++hetCfgIdx)
    {
        const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = cfg.cfgs[hetCfgIdx].*kernelDriver;
        if(!kernelNodeParamsDriver.func) { continue; }
        CU_CHECK_EXCEPTION(launch_kernel(kernelNodeParamsDriver, stream));
    }
}

void ChestStream::launchKernels(cudaStream_t stream) {
    // If early HARQ is enabled, then we already ran the first channel estimation kernel on the
    // first DMRS symbol. When using delay estimation, this will have already accumulated contributions
    // from the first DMRS symbol, so we skip the first symbol here. We still run all of the second kernels
    // because the second kernel on the first DMRS symbol will now use delay estimations accumulated over
    // all DMRS symbols whereas the kernel that ran in early HARQ only used the first DMRS symbol.
    const uint32_t startChEstInstIdx =
            (m_earlyHarqModeEnabled && m_chEstAlgo==PUSCH_CH_EST_ALGO_TYPE_MULTISTAGE_MMSE_WITH_DELAY_EST) ?
            1 : 0;
    /////////////////////////////////// full-slot processing ////////////////////////////////////////////
    launchKernelsImpl(stream, m_chEstLaunchCfgs, startChEstInstIdx);
    if(m_chEstAlgo == PUSCH_CH_EST_ALGO_TYPE_MULTISTAGE_MMSE_WITH_DELAY_EST)
    {
        launchSecondaryKernels(stream);
    }
}

void ChestStream::launchSecondaryKernels(cudaStream_t stream) {
    constexpr std::uint32_t startChEstInstIdx = 0;
    launchKernelsImpl(stream, m_chEstLaunchCfgs, startChEstInstIdx,
                      &cuphyPuschRxChEstLaunchCfg_t::kernelNodeParamsDriverSecond);
}

void ChestStream::launchKernels0Slot(cudaStream_t stream) {
    launchKernels0SlotImpl(stream, m_chEstLaunchCfgs);
    if(m_chEstAlgo == PUSCH_CH_EST_ALGO_TYPE_MULTISTAGE_MMSE_WITH_DELAY_EST)
    {
        launchSecondaryKernels0Slot(stream);
    }
}

void ChestStream::launchSecondaryKernels0Slot(cudaStream_t stream) {
    launchKernels0SlotImpl(stream, m_chEstLaunchCfgs,
                           &cuphyPuschRxChEstLaunchCfg_t::kernelNodeParamsDriverSecond);
}

} // namespace ch_est
