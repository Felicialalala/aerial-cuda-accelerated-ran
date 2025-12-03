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

#include "cuphy_hdf5.hpp"
#include "hdf5hpp.hpp"
#include "cuphy.hpp"
#include "pycuphy_debug.hpp"

namespace pycuphy {

H5DebugDump::H5DebugDump(const std::string& filename) {
    m_hdf5File.reset(new hdf5hpp::hdf5_file(hdf5hpp::hdf5_file::create(filename.c_str())));
    hdf5hpp::hdf5_file::open(filename.c_str());
}

H5DebugDump::~H5DebugDump() {
    m_hdf5File->close();
}

void H5DebugDump::dump(const std::string& name, const cuphy::tensor_ref& tensor, cudaStream_t cuStream) {
    cuphy::write_HDF5_dataset(*m_hdf5File, tensor, tensor.desc(), name.c_str(), cuStream);
}

}
