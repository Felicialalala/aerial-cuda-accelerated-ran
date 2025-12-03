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

#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../cuphy/cuphy.h"
#include "../cuphy/common_utils.hpp"

template <typename T>
__device__ inline void copyStrided(const T* __restrict__ src,
                                   T* __restrict__ dst,
                                   std::size_t count)
{
    const std::size_t threadIndex = (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    const std::size_t stride      = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for(std::size_t i = threadIndex; i < count; i += stride) {
        dst[i] = src[i];
    }
}

__global__ void copyKernel(const std::uint8_t* __restrict__ srcBytes,
                                  std::uint8_t* __restrict__ dstBytes,
                                  std::size_t numBytes)
{
    if(numBytes == 0) {
        return;
    }

    const std::uintptr_t srcAddr = reinterpret_cast<std::uintptr_t>(srcBytes);
    const std::uintptr_t dstAddr = reinterpret_cast<std::uintptr_t>(dstBytes);
    static constexpr std::size_t align = 16;
    const std::size_t srcMod = static_cast<std::size_t>(srcAddr % align);
    const std::size_t dstMod = static_cast<std::size_t>(dstAddr % align);

    if(srcMod != dstMod) {
        copyStrided<std::uint8_t>(srcBytes, dstBytes, numBytes);
        return;
    }

    const std::size_t head = (align - srcMod) % align;
    std::size_t remaining = numBytes;
    const std::uint8_t* srcHead = srcBytes;
    std::uint8_t* dstHead = dstBytes;

    if(head != 0 && remaining != 0) {
        const std::size_t headBytes = head <= remaining ? head : remaining;
        copyStrided<std::uint8_t>(srcHead, dstHead, headBytes);
        srcHead += headBytes;
        dstHead += headBytes;
        remaining -= headBytes;
    }

    const std::size_t numVec = remaining / align;
    if(numVec > 0) {
        const uint4* srcVec = reinterpret_cast<const uint4*>(srcHead);
        uint4* dstVec = reinterpret_cast<uint4*>(dstHead);
        copyStrided<uint4>(srcVec, dstVec, numVec);
        const std::size_t vecBytes = numVec * align;
        srcHead += vecBytes;
        dstHead += vecBytes;
        remaining -= vecBytes;
    }

    if(remaining > 0) {
        copyStrided<std::uint8_t>(srcHead, dstHead, remaining);
    }
}

cuphyStatus_t cuphylaunchKernelCopy(void* devicePtr,
                                     void* pinnedHostPtr,
                                     std::size_t numBytes,
                                     cudaMemcpyKind direction,
                                     cudaStream_t stream)
{
    if(numBytes == 0) {
        return CUPHY_STATUS_SUCCESS;
    }
    

    std::uint8_t* mappedHostDevicePtr = nullptr;
    CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&mappedHostDevicePtr), pinnedHostPtr, 0));

    const std::uint8_t* srcBytes = nullptr;
    std::uint8_t* dstBytes = nullptr;

    if(direction == cudaMemcpyDeviceToHost) {
        srcBytes = reinterpret_cast<const std::uint8_t*>(devicePtr);
        dstBytes = mappedHostDevicePtr;
    } else if(direction == cudaMemcpyHostToDevice) {
        srcBytes = mappedHostDevicePtr;
        dstBytes = reinterpret_cast<std::uint8_t*>(devicePtr);
    } else {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }

    if(srcBytes == nullptr || dstBytes == nullptr) {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }

    static constexpr int blockSize = 1024;

    const std::size_t blocksNeeded = (numBytes + static_cast<std::size_t>(blockSize) - 1) / static_cast<std::size_t>(blockSize);
    const int gridSize = static_cast<int>(blocksNeeded > 65535 ? 65535 : blocksNeeded);
    dim3 grid(gridSize);
    dim3 block(blockSize);
    void* args[] = {
        const_cast<void*>(reinterpret_cast<const void*>(&srcBytes)),
        reinterpret_cast<void*>(&dstBytes),
        const_cast<void*>(static_cast<const void*>(&numBytes))
    };
    cudaError_t launchErrRt = cudaLaunchKernel(reinterpret_cast<const void*>(&copyKernel), grid, block, args, 0, stream);
    if(launchErrRt != cudaSuccess) {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    return CUPHY_STATUS_SUCCESS;
}


