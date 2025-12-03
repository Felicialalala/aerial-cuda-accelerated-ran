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

#if !defined(LDPC_MS_CTA_SIMD_FLOODING_HPP_INCLUDED_)
#define LDPC_MS_CTA_SIMD_FLOODING_HPP_INCLUDED_

// Min-sum, Single Cooperative Thread Array, SIMD, Flooding LDPC Implementation

#include "ldpc.hpp"

////////////////////////////////////////////////////////////////////////
// ldpc
namespace ldpc
{
////////////////////////////////////////////////////////////////////////
// decode_ms_cta_simd_flooding()
cuphyStatus_t decode_ms_cta_simd_flooding(LDPC_output_t&      tDst,
                                          const_tensor_pair&  tLLR,
                                          const LDPC_config&  config,
                                          float               normalization,
                                          cuphyLDPCResults_t* results,
                                          void*               workspace,
                                          cudaStream_t        strm);

////////////////////////////////////////////////////////////////////////
// decode_ms_cta_simd_flooding_workspace_size()
std::pair<bool, size_t> decode_ms_cta_simd_flooding_workspace_size(const LDPC_config& cfg);

} // namespace ldpc

#endif // !defined(LDPC_MS_CTA_LAYERED_HPP_INCLUDED_)
