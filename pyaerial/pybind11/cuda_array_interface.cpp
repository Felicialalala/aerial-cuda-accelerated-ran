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

#include <vector>
#include <complex>
#include <cuda_fp16.h>
#include "cuda_array_interface.hpp"

namespace py = pybind11;

namespace pycuphy {

py::tuple as_tuple(const std::vector<size_t>& vec) {
    py::tuple result(vec.size());
    for (size_t i = 0; i < vec.size(); ++i)
    {
        result[i] = vec[i];
    }
    return result;
}

template <typename T>
cuda_array_t<T>::cuda_array_t(intptr_t addr,
                              const std::vector<size_t> &shape,
                              const std::vector<size_t> &strides,
                              bool readonly):
device_ptr(reinterpret_cast<void *>(addr)),
shape_(shape),
strides_(strides),
readonly(readonly),
has_strides(!strides.empty()) {
    if (shape_.empty())
    {
        throw std::runtime_error("Shape cannot be empty!");
    }
}

template <typename T>
cuda_array_t<T>::cuda_array_t(const py::object& array) {
    py::dict interface = array.attr("__cuda_array_interface__");

    // shape (required)
    auto shape_tuple = interface["shape"].cast<py::tuple>();
    shape_.clear();
    for (const auto &dim : shape_tuple)
    {
        shape_.push_back(dim.cast<size_t>());
    }

    // data pointer (required)
    auto data_tuple = interface["data"].cast<py::tuple>();
    void* raw_ptr = reinterpret_cast<void*>(data_tuple[0].cast<uintptr_t>());
    // for zero-size arrays, pointer should be nullptr
    device_ptr = (get_size() == 0) ? nullptr : raw_ptr;
    readonly = data_tuple[1].cast<bool>();

    // strides (optional)
    has_strides = false;
    if (interface.contains("strides"))
    {
        py::object strides_obj = interface["strides"];
        if (!strides_obj.is_none())
        {
            auto strides_tuple = strides_obj.cast<py::tuple>();
            strides_.clear();
            for (const auto &stride : strides_tuple)
            {
                strides_.push_back(stride.cast<size_t>());
            }
            has_strides = true;
        }
    }

    auto typestr = interface["typestr"].cast<std::string>();
    verify_dtype(typestr);
}

template <typename T>
size_t cuda_array_t<T>::get_size() const {
    return std::accumulate(shape_.cbegin(), shape_.cend(), 1, std::multiplies<size_t>{});
}

template <typename T>
py::dict cuda_array_t<T>::get_interface_dict() const {
    py::dict interface;

    // required fields
    interface["shape"] = as_tuple(shape_);
    interface["typestr"] = get_typestr();
    interface["data"] = py::make_tuple(reinterpret_cast<uintptr_t>(device_ptr), readonly);
    interface["version"] = 3;

    // optional fields
    if (has_strides) {
        interface["strides"] = as_tuple(strides_);
    }
    else {
        interface["strides"] = py::none();
    }

    // we don't currently support mask or descr
    interface["mask"] = py::none();
    interface["descr"] = py::none();

    return interface;
}

template <typename T>
void cuda_array_t<T>::verify_dtype(const std::string &typestr) const
{
    if constexpr (std::is_same_v<T, float>)
    {
        if (typestr != "<f4")
        {
            throw std::runtime_error("Type mismatch: expected float32!");
        }
    }
    else if constexpr (std::is_same_v<T, int>)
    {
        if (typestr != "<i4")
        {
            throw std::runtime_error("Type mismatch: expected int!");
        }
    }
    else if constexpr (std::is_same_v<T, uint8_t>)
    {
        if (typestr != "|u1" && typestr != "<u1")
        {
            throw std::runtime_error("Type mismatch: expected uint8!");
        }
    }
    else if constexpr (std::is_same_v<T, uint16_t>)
    {
        // Allow uint16 wrapper to accept uint32 arrays for automatic conversion
        if (typestr != "<u2" && typestr != "<u4")
        {
            throw std::runtime_error("Type mismatch: expected uint16 or uint32 (for auto-conversion)!");
        }
    }
    else if constexpr (std::is_same_v<T, uint32_t>)
    {
        if (typestr != "<u4")
        {
            throw std::runtime_error("Type mismatch: expected uint32!");
        }
    }
    else if constexpr (std::is_same_v<T, __half>)
    {
        if (typestr != "<f2")
        {
            throw std::runtime_error("Type mismatch: expected float16!");
        }
    }
    else if constexpr (std::is_same_v<T, double>)
    {
        if (typestr != "<f8")
        {
            throw std::runtime_error("Type mismatch: expected float64!");
        }
    }
    else if constexpr (std::is_same_v<T, std::complex<float>>)
    {
        if (typestr != "<c8")
        {
            throw std::runtime_error("Type mismatch: expected complex64!");
        }
    }
    else if constexpr (std::is_same_v<T, std::complex<double>>)
    {
        if (typestr != "<c16")
        {
            throw std::runtime_error("Type mismatch: expected complex128!");
        }
    }
}

template <typename T>
std::string cuda_array_t<T>::get_typestr() const
{
    if constexpr (std::is_same_v<T, float>)
    {
        return "<f4";
    }
    else if constexpr (std::is_same_v<T, int>)
    {
        return "<i4";
    }
    else if constexpr (std::is_same_v<T, uint8_t>)
    {
        return "<u1";
    }
    else if constexpr (std::is_same_v<T, uint16_t>)
    {
        return "<u2";
    }
    else if constexpr (std::is_same_v<T, uint32_t>)
    {
        return "<u4";
    }
    else if constexpr (std::is_same_v<T, __half>)
    {
        return "<f2";
    }
    else if constexpr (std::is_same_v<T, double>)
    {
        return "<f8";
    }
    else if constexpr (std::is_same_v<T, std::complex<float>>)
    {
        return "<c8";
    }
    else if constexpr (std::is_same_v<T, std::complex<double>>)
    {
        return "<c16";
    }
    throw std::runtime_error("Unsupported dtype!");
}

template class cuda_array_t<int>;
template class cuda_array_t<uint8_t>;
template class cuda_array_t<uint16_t>;
template class cuda_array_t<uint32_t>;
template class cuda_array_t<float>;
template class cuda_array_t<__half>;
template class cuda_array_t<std::complex<float>>;

}  // namespace pycuphy