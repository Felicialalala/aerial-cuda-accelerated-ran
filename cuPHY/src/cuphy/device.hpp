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

#if !defined(DEVICE_HPP_INCLUDED_)
#define DEVICE_HPP_INCLUDED_

#include "cuphy_internal.h"

////////////////////////////////////////////////////////////////////////
// cuphy_i
namespace cuphy_i // cuphy internal
{
//----------------------------------------------------------------------
// device
class device //
{
public:
    device(int device_index = 0);
    int        multiProcessorCount()    const { return properties_.multiProcessorCount; }
    size_t     sharedMemPerBlock()      const { return properties_.sharedMemPerBlock; }
    size_t     sharedMemPerBlockOptin() const { return properties_.sharedMemPerBlockOptin; }
    static int get_count();
    static int get_current();

private:
    int            index_;
    cudaDeviceProp properties_;
};

} // namespace cuphy_i

#endif // !defined(DEVICE_HPP_INCLUDED_)
