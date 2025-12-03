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

const int SIMPLEX_DECODER_MAX_E = 1536;

struct cuphySimplexDecoder
{
};

struct simplexDecoderDynDescr
{
    cuphySimplexCwPrm_t* pCwPrmsGpu;         
    uint16_t             nCws;
};
typedef struct simplexDecoderDynDescr simplexDecoderDynDescr_t;

// simplexDecoder kernel arguments (supplied via descriptors)
typedef struct
{
    simplexDecoderDynDescr_t* pDynDescr;
} simplexDecoderKernelArgs_t;

class SimplexDecoder : public cuphySimplexDecoder
{
public:
    SimplexDecoder();

    void setup(uint16_t                        nCws,
               cuphySimplexCwPrm_t*            pCwPrmsCpu, 
               cuphySimplexCwPrm_t*            pCwPrmsGpu, 
               bool                            enableCpuToGpuDescrAsyncCpy, // option to copy descriptors from CPU to GPU
               simplexDecoderDynDescr_t*       pCpuDynDesc,                 // pointer to descriptor in cpu
               void*                           pGpuDynDesc,                 // pointer to descriptor in gpu
               cuphySimplexDecoderLaunchCfg_t* pLaunchCfg,                  // pointer to launch configuration
               cudaStream_t                    strm);                       // stream to perform copy

    static void getDescrInfo(size_t& dynDescrSizeBytes, size_t& dynDescrAlignBytes);

private:
    uint8_t  CW_PER_BLOCK_;


    simplexDecoderKernelArgs_t m_kernelArgs;
    void kernelSelect(uint16_t  nCws, cuphySimplexDecoderLaunchCfg_t* pLaunchCfg);
};