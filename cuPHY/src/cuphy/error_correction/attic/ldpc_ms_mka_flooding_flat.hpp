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

#if !defined(LDPC_MS_MKA_FLOODING_FLAT_HPP_INCLUDED_)
#define LDPC_MS_MKA_FLOODING_FLAT_HPP_INCLUDED_

// Min-sum, Multi-kernel w/Atomic, Flooding , Flat BG Table LDPC Implementation

#include "ldpc.hpp"

////////////////////////////////////////////////////////////////////////
// ldpc
namespace ldpc
{
////////////////////////////////////////////////////////////////////////
// decode_multi_kernel_atomic_flat()
cuphyStatus_t decode_multi_kernel_atomic_flat(LDPC_output_t&      tDst,
                                              const_tensor_pair&  tLLR,
                                              const LDPC_config&  config,
                                              float               normalization,
                                              cuphyLDPCResults_t* results,
                                              void*               workspace,
                                              cudaStream_t        strm);

////////////////////////////////////////////////////////////////////////
// decode_multi_kernel_atomic_flat_workspace_size()
std::pair<bool, size_t> decode_multi_kernel_atomic_flat_workspace_size(const LDPC_config& config);

} // namespace ldpc

#endif // !defined(LDPC_MS_MKA_FLOODING_FLAT_HPP_INCLUDED_)
