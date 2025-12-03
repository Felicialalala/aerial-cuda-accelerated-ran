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

#ifndef CUPHY_CHEST_ISTREAM_HPP
#define CUPHY_CHEST_ISTREAM_HPP

#include <gsl-lite/gsl-lite.hpp>

#include "driver_types.h"
#include "cuphy.h"

namespace ch_est {

// Helper functions for stream operations

/**
 * @brief Launch kernels from launchCfgs. Loop over the launchCfgs and if the kernel function is
 * defined, call launch_kernel
 * @param stream Stream to use for launching the kernel
 * @param launchCfgs launch configs, non-owning view
 * @param startChEstInstIdx the channel index to start with, in the configs array.
 * @param kernelDriver function to use. This provides flexibility to which function to launch
 */
void launchKernelsImpl(cudaStream_t                                                  stream,
                       gsl_lite::span<const cuphyPuschRxChEstLaunchCfgs_t>                launchCfgs,
                       std::uint32_t                                                 startChEstInstIdx = 0,
                       const CUDA_KERNEL_NODE_PARAMS cuphyPuschRxChEstLaunchCfg_t::* kernelDriver      = &cuphyPuschRxChEstLaunchCfg_t::kernelNodeParamsDriver);

/**
 * @brief Launch kernels from launchCfgs[0]. I.e. slot 0 only.
 * Loop over the cfs of this slot 0, and if the kernel function is defined, call launch_kernel.
 * @param stream Stream to use for launching the kernel
 * @param launchCfgs launch configs, non-owning view
 * @param kernelDriver function to use. This provides flexibility to which function to launch
 */
void launchKernels0SlotImpl(cudaStream_t                                                  stream,
                            gsl_lite::span<const cuphyPuschRxChEstLaunchCfgs_t>                launchCfgs,
                            const CUDA_KERNEL_NODE_PARAMS cuphyPuschRxChEstLaunchCfg_t::* kernelDriver = &cuphyPuschRxChEstLaunchCfg_t::kernelNodeParamsDriver);

/**
 * @class IChestStream
 *
 * @brief Provides interface to running kernels using a provided cudaStream_t
 *       stream instance/handle.
 */
class IChestStream {
public:
    IChestStream() = default;
    virtual ~IChestStream() = default;
    IChestStream(const IChestStream& streamChest) = default;
    IChestStream& operator=(const IChestStream& streamChest) = default;
    IChestStream(IChestStream&& streamChest) = default;
    IChestStream& operator=(IChestStream&& streamChest) = default;

    /**
     * @brief Launch primary kernels
     * @param stream CUDA stream to use
     * @throw cuda_driver_exception in case of exception
     */
    virtual void launchKernels(cudaStream_t stream) = 0;

    /**
     * @brief Launch primary kernel, first slot only
     * @param stream CUDA stream to use
     * @throw cuda_driver_exception in case of exception
     */
    virtual void launchKernels0Slot(cudaStream_t stream) = 0;

    /**
     * @brief Launch secondary kernels
     * @param stream CUDA stream to use
     * @throw cuda_driver_exception in case of exception
     */
    virtual void launchSecondaryKernels(cudaStream_t stream) = 0;

    /**
     * @brief Launch secondary kernel, first slot only
     * @param stream CUDA stream to use
     * @throw cuda_driver_exception in case of exception
     */
    virtual void launchSecondaryKernels0Slot(cudaStream_t stream) = 0;
};

} // namespace ch_est

#endif //CUPHY_CHEST_ISTREAM_HPP
