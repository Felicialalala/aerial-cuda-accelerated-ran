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

#include <cuda.h>
#include "cuphy.h"


struct cuphyRmDecoder
{
};

struct rmDecoderDynDescr
{
    cuphyRmCwPrm_t*      pCwPrmsGpu;         
    uint16_t             nCws;
    uint32_t*            pRmCodebook;
};
typedef struct rmDecoderDynDescr rmDecoderDynDescr_t;

// simplexDecoder kernel arguments (supplied via descriptors)
typedef struct
{
    rmDecoderDynDescr_t* pDynDescr;
} rmDecoderKernelArgs_t;

const int PUCCH_F2_RM_MAX_N = 32;

#include "cuphy_context.hpp"
#include "tensor_desc.hpp"
class rmDecoder : public cuphyRmDecoder
{
public:
    rmDecoder(const cuphy_i::context& ctx);

    void init(void* pMemoryFootprint=nullptr);

    void setup( uint16_t                   nCws,
                cuphyRmCwPrm_t*            pCwPrmsGpu,
                bool                       enableCpuToGpuDescrAsyncCpy, // option to copy descriptors from CPU to GPU
                rmDecoderDynDescr_t*       pCpuDynDesc,                 // pointer to descriptor in cpu
                void*                      pGpuDynDesc,                 // pointer to descriptor in gpu
                cuphyRmDecoderLaunchCfg_t* pLaunchCfg,                  // pointer to launch configuration
                cudaStream_t               strm);                       // stream to perform copy

    ~rmDecoder();
    static void getDescrInfo(size_t& dynDescrSizeBytes, size_t& dynDescrAlignBytes);

private:
    int      deviceIndex_;            // index of device associated with context
    uint64_t cc_;                     // compute capability (major << 32) | minor
    int      sharedMemPerBlockOptin_; // maximum shared memory per block usable by option
    int      multiProcessorCount_;    // number of multiprocessors on device

    void*                         pDstAddr_;
    const void*                   pSrcAddrs_;
    cuphyDataType_t               srcType_;
    int                           mode_;
    int                           numCW_;
    uint16_t                      CW_PER_BLOCK_;
    int                           *d_E_;
    int                           *d_two_to_K_;
    float                         *d_codebook_f_;
    uint32_t                      *d_codebook_u32_;

    rmDecoderKernelArgs_t m_kernelArgs;
    void kernelSelect(uint16_t  nCws, cuphyRmDecoderLaunchCfg_t* pLaunchCfg);
};
