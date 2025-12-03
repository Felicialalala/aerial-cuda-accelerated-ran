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

#if !defined(CUPHY_RNG_HPP_INCLUDED_)
#define CUPHY_RNG_HPP_INCLUDED_

#include "cuphy.h"
#include "cuphy_internal.h"
#include "tensor_desc.hpp"
#include <curand_kernel.h>

////////////////////////////////////////////////////////////////////////
// cuphyRNG
// Empty base class for internal random number generator class, used by
// forward declaration in public-facing cuphy.h.
struct cuphyRNG
{
};


namespace cuphy_i // cuphy internal
{

////////////////////////////////////////////////////////////////////////
// rng
class rng : public cuphyRNG
{
public:
    //------------------------------------------------------------------
    // rng()
    // Constructor
    rng(unsigned long long seed, cudaStream_t s);
    //------------------------------------------------------------------
    // normal()
    // Populate the given tensor with random values from a Gaussian
    // (normal) distribution
    cuphyStatus_t normal(const tensor_desc&    t,
                         void*                 p,
                         const cuphyVariant_t& mean,
                         const cuphyVariant_t& stddev,
                         cudaStream_t          strm);
    //------------------------------------------------------------------
    // uniform()
    // Populate the given tensor with random values from a uniform
    // distribution.
    cuphyStatus_t uniform(const tensor_desc&    t,
                          void*                 p,
                          const cuphyVariant_t& min_v,
                          const cuphyVariant_t& max_v,
                          cudaStream_t          strm);

private:
    cuphy_i::unique_device_ptr<curandState> randStates_;
};


} // namespace cuphy_i

#endif // !defined(CUPHY_RNG_HPP_INCLUDED_)
