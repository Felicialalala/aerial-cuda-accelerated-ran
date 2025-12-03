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

#ifndef PYCUPHY_DEBUG_HPP
#define PYCUPHY_DEBUG_HPP

#include <string>
#include "cuphy.hpp"
#include "hdf5hpp.hpp"

namespace pycuphy {

class H5DebugDump {

public:
    H5DebugDump(const std::string& filename);
    ~H5DebugDump();

    void dump(const std::string& name, const cuphy::tensor_ref& tensor, cudaStream_t cuStream);

private:
    std::unique_ptr<hdf5hpp::hdf5_file> m_hdf5File;
};

}


#endif // PYCUPHY_DEBUG_HPP
