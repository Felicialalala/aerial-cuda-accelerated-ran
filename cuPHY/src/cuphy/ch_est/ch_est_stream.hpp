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

#ifndef CUPHY_CHEST_STREAM_HPP
#define CUPHY_CHEST_STREAM_HPP

#include "IStream.hpp"
#include "cuphy.h"

namespace ch_est {

/**
 * @brief StreamChest implement Stream CUDA operations
 *        while the Nodes Chest class types are in charge
 *        of CUDA graph related  operations.
 */
class ChestStream final : public IChestStream {
public:
    /**
     * @brief Construct channel estimate stream
     * @param chEstLaunchCfgs channel estimate kernel launch configs
     * @param earlyHarqModeEnabled true/false for Early HARQ mode
     * @param chEstAlgo The channel estimate to use.
     */
    ChestStream(const gsl_lite::span<const cuphyPuschRxChEstLaunchCfgs_t> chEstLaunchCfgs,
                const bool earlyHarqModeEnabled,
                const cuphyPuschChEstAlgoType_t chEstAlgo) :
            m_earlyHarqModeEnabled{earlyHarqModeEnabled},
            m_chEstAlgo{chEstAlgo},
            m_chEstLaunchCfgs{chEstLaunchCfgs} {}

    /**
     * @brief Launch primary kernels
     * @param stream CUDA stream to use
     * @throw cuda_driver_exception in case of exception
     */
    void launchKernels(cudaStream_t stream) final;

    /**
     * @brief Launch primary kernel, first slot only
     * @param stream CUDA stream to use
     * @throw cuda_driver_exception in case of exception
     */
    void launchKernels0Slot(cudaStream_t stream) final;

    /**
     * @brief Launch secondary kernels
     * @param stream CUDA stream to use
     * @throw cuda_driver_exception in case of exception
     */
    void launchSecondaryKernels(cudaStream_t stream) final;

    /**
     * @brief Launch secondary kernel, first slot only
     * @param stream CUDA stream to use
     * @throw cuda_driver_exception in case of exception
     */
    void launchSecondaryKernels0Slot(cudaStream_t stream) final;

    /**
     * @brief toggle true/false on Early HARQ
     * @param earlyHarqModeEnabled true/false
     */
    void setEarlyHarqModeEnabled(const bool earlyHarqModeEnabled) {
        m_earlyHarqModeEnabled = earlyHarqModeEnabled;
    }
private:
    bool m_earlyHarqModeEnabled{};
    cuphyPuschChEstAlgoType_t m_chEstAlgo{};
    // Array - need span<> instead.
    const gsl_lite::span<const cuphyPuschRxChEstLaunchCfgs_t> m_chEstLaunchCfgs{};
};

} // namespace ch_est

#endif //CUPHY_CHEST_STREAM_HPP
