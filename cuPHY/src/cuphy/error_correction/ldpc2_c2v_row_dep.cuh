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

#if !defined(LDPC2_C2V_ROW_DEP_CUH_INCLUDED_)
#define LDPC2_C2V_ROW_DEP_CUH_INCLUDED_

#include "ldpc2.cuh"

namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// C2V_row_dep
// Check-to-variable processor, with row-dependent processing selection
template <typename T,
          int      BG>
class C2V_row_dep
{
public:
    //------------------------------------------------------------------
    // init()
    __device__
    void init()
    {
        c2v_storage.init();
    }
private:
    //typedef __half            app_t;
    
    typedef cC2V_storage_t<T> c2v_storage_t;
    //------------------------------------------------------------------
    // Data
    c2v_storage_t c2v_storage;

};
          


} // namespace ldpc2

#endif // !defined(LDPC2_C2V_ROW_DEP_CUH_INCLUDED_)

