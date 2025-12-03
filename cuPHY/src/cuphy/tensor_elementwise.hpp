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

#if !defined(CUPHY_TENSOR_ELEMENTWISE_HPP_INCLUDED_)
#define CUPHY_TENSOR_ELEMENTWISE_HPP_INCLUDED_

#include "cuphy.h"
#include "tensor_desc.hpp"

namespace cuphy_i
{

////////////////////////////////////////////////////////////////////////
// tensor_elementwise()
// Note that for some operations, input B is optional.
cuphyStatus_t tensor_elementwise(const tensor_desc&      tDst,
                                 void*                   dstAddr,
                                 const tensor_desc&      tSrcA,
                                 const void*             srcAddrA,
                                 const cuphyVariant_t*   alpha,
                                 cuphyTensorDescriptor_t tSrcB,
                                 const void*             srcAddrB,
                                 const cuphyVariant_t*   beta,
                                 cuphyElementWiseOp_t    elemOp,
                                 cudaStream_t            strm);


} // namespace cuphy_i

#endif // !defined(CUPHY_TENSOR_TILE_HPP_INCLUDED_)
