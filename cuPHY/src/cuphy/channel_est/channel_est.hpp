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

#if !defined(CHANNEL_EST_HPP_INCLUDED_)
#define CHANNEL_EST_HPP_INCLUDED_

#include "tensor_desc.hpp"

////////////////////////////////////////////////////////////////////////
// channel_est
namespace channel_est
{
//----------------------------------------------------------------------
// mmse_1D_time_frequency()
void mmse_1D_time_frequency(tensor_pair&       tDst,
                            const_tensor_pair& tSymbols,
                            const_tensor_pair& tFreqFilters,
                            const_tensor_pair& tTimeFilters,
                            const_tensor_pair& tFreqIndices,
                            const_tensor_pair& tTimeIndices,
                            cudaStream_t       strm);

} // namespace channel_est

#endif // !defined(CHANNEL_EST_HPP_INCLUDED_)
