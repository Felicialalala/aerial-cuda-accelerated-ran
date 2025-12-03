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

#if !defined(CUPHY_SOFT_DEMAPPER_HPP_INCLUDED_)
#define CUPHY_SOFT_DEMAPPER_HPP_INCLUDED_

#include "cuphy.h"
#include "cuphy_internal.h"
#include "tensor_desc.hpp"

namespace cuphy_i // cuphy internal
{

class context;
////////////////////////////////////////////////////////////////////////
// soft_demapper_context
// Storage for read-only resources required for some implementations of
// soft demapping
class soft_demapper_context
{
public:
    //------------------------------------------------------------------
    // soft_demapper_context()
    soft_demapper_context();
    //------------------------------------------------------------------
    // QAM_tex()
    const mipmapped_texture& QAM_tex() const { return QAMtex_; }
private:
    mipmapped_texture QAMtex_;
};

////////////////////////////////////////////////////////////////////////
// soft_demap()
cuphyStatus_t soft_demap(context&     ctx,
                         tensor_desc& tLLR,
                         void*        pLLR,
                         tensor_desc& tSym,
                         const void*  pSym,
                         int          log2_QAM,
                         float        noiseVariance,
                         cudaStream_t strm);

} // namespace cuphy_i

#endif // !defined(CUPHY_SOFT_DEMAPPER_HPP_INCLUDED_)
