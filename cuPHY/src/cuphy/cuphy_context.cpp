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

#include "cuphy_context.hpp"
#include "cuphy_internal.h"

namespace cuphy_i
{
    
////////////////////////////////////////////////////////////////////////
// cuphy_i::context::context()
context::context()
{
    //------------------------------------------------------------------
    // Retrieve the device that will be associated with this context
    cudaError_t e = cudaGetDevice(&deviceIndex_);
    if(cudaSuccess != e)
    {
        throw cuda_exception(e);
    }
    //------------------------------------------------------------------
    // Retrieve individual device attributes as needed; no need to call cudaGetDeviceProperties
    int cc_major = 0;
    e = cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, deviceIndex_);
    if(cudaSuccess != e)
    {
        throw cuda_exception(e);
    }
    int cc_minor = 0;
    e = cudaDeviceGetAttribute(&cc_minor, cudaDevAttrComputeCapabilityMinor, deviceIndex_);
    if(cudaSuccess != e)
    {
        throw cuda_exception(e);
    }
    cc_                     = make_cc_uint64(cc_major, cc_minor);

    e = cudaDeviceGetAttribute(&sharedMemPerBlockOptin_, cudaDevAttrMaxSharedMemoryPerBlockOptin, deviceIndex_);
    if(cudaSuccess != e)
    {
        throw cuda_exception(e);
    }

    e = cudaDeviceGetAttribute(&multiProcessorCount_, cudaDevAttrMultiProcessorCount, deviceIndex_);
    if(cudaSuccess != e)
    {
        throw cuda_exception(e);
    }

    //------------------------------------------------------------------
    softDemapperContext_.reset(new soft_demapper_context);
};

} // namespace cuphy_i
