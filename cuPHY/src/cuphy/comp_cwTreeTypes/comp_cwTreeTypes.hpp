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

#include "../cuphy.h"

// Implementation of CompCwTreeTypes interface exposed as an opaque data type to abstract out implementation
// details (compCwTreeTypes  C++ class). compCwTreeTypes is implemented as a C++ class which inherits
// from this interface structure defiend as an empty shell (opaque type is a struct since the interface is C
// compatible). Pointer to the opaque type is also exposed in the interface as a handle to the underlying
// implementation.
struct cuphyCompCwTreeTypes
{};

struct compCwTypesDynDescr
{
    uint8_t**                    pCwTreeTypesAddrs;
    const cuphyPolarUciSegPrm_t* pPolarUciSegPrms;
};
typedef struct compCwTypesDynDescr compCwTreeTypesDynDescr_t;

// compCwTypes kernel arguments (supplied via descriptors)
typedef struct
{
    compCwTreeTypesDynDescr_t* pDynDescr;
} compCwTreeTypesKernelArgs_t;

// Class implementation of compCwTreeTypes
class compCwTreeTypes : public cuphyCompCwTreeTypes {
public:
    compCwTreeTypes()                       = default;
    ~compCwTreeTypes()                      = default;
    compCwTreeTypes(compCwTreeTypes const&) = delete;
    compCwTreeTypes& operator=(compCwTreeTypes const&) = delete;

    // setup object state and dynamic component descriptor in prepration towards execution
    void setup(uint16_t                         nPolUciSegs,                 // number of polar UCI segments
               const cuphyPolarUciSegPrm_t*     pPolUciSegPrmsCpu,           // starting adreass of polar UCI segment parameters (CPU)
               const cuphyPolarUciSegPrm_t*     pPolUciSegPrmsGpu,           // starting adreass of polar UCI segment parameters (GPU)
               uint8_t**                        pCwTreeTypesAddrs,           // pointer to cwTreeTypes addresses
               compCwTreeTypesDynDescr_t*       pCpuDynDesc,                 // pointer to dynamic descriptor in cpu
               void*                            pGpuDynDesc,                 // pointer to dynamic descriptor in gpu
               uint8_t                          enableCpuToGpuDescrAsyncCpy, // option to copy cpu descriptors from cpu to gpu
               cuphyCompCwTreeTypesLaunchCfg_t* pLaunchCfg,                  // pointer to rate matching launch configuration
               cudaStream_t                     strm);                                           // stream to perform copy

    void kernelSelect(uint16_t                         nPolUciSegs,
                      const cuphyPolarUciSegPrm_t*     pPolUciSegPrmsCpu,
                      cuphyCompCwTreeTypesLaunchCfg_t* pLaunchCfg);

    static void getDescrInfo(size_t& dynDescrSizeBytes, size_t& dynDescrAlignBytes);

    compCwTreeTypesKernelArgs_t m_kernelArgs;
};
