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

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>

namespace pycuphy {

[[nodiscard]] pybind11::tuple as_tuple(const std::vector<size_t>& vec);

// Implements __cuda_array_interface__ (version 3):
// https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html
template <typename T>
class cuda_array_t final
{
public:
    explicit cuda_array_t(intptr_t addr,
                          const std::vector<size_t> &shape,
                          const std::vector<size_t> &strides,
                          bool readonly = false);

    explicit cuda_array_t(const pybind11::object& array);

    [[nodiscard]] void* get_device_ptr() const { return device_ptr; }
    [[nodiscard]] auto get_shape() const { return shape_; }
    [[nodiscard]] auto get_ndim() const { return shape_.size(); }
    [[nodiscard]] auto get_strides() const { return strides_; }
    [[nodiscard]] bool is_readonly() const { return readonly; }
    [[nodiscard]] bool has_stride_info() const { return has_strides; }
    [[nodiscard]] size_t get_size() const;
    [[nodiscard]] pybind11::dict get_interface_dict() const;

private:
    void* device_ptr{};
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    bool readonly{};
    bool has_strides{};

    void verify_dtype(const std::string &typestr) const;
    [[nodiscard]] std::string get_typestr() const;
};

using cuda_array_float = cuda_array_t<float>;
using cuda_array_complex_float = cuda_array_t<std::complex<float>>;
using cuda_array_uint32 = cuda_array_t<uint32_t>;
using cuda_array_uint8 = cuda_array_t<uint8_t>;


}  // namespace pycuphy