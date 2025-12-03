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

#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "cuphy.h"
#include "cuphy.hpp"
#include "cuphy_api.h"
#include "utils.cuh"
#include "pycuphy_util.hpp"
#include "tensor_desc.hpp"
#include "cuda_array_interface.hpp"

namespace py = pybind11;

namespace pycuphy {


template <typename T>
py::array_t<T> deviceToNumpy(uint64_t deviceAddr,
                             uint64_t hostAddr,
                             const py::list& dimensions,
                             uint64_t cuStream) {
    int nDim = dimensions.size();

    // T needs to be either std::complex<float> or float.
    // TODO: Fix this hack where we need to determine types based on T as it restricts
    // the use of this function.
    cuphyDataType_t deviceDataType, hostDataType;
    if(std::is_same<T, std::complex<float>>::value) {
        deviceDataType = CUPHY_C_16F;
        hostDataType = CUPHY_C_32F;
    }
    else if(std::is_same<T, float>::value) {
        deviceDataType = CUPHY_R_32F;
        hostDataType = CUPHY_R_32F;
    }
    else {
        throw std::runtime_error("deviceToNumpy: Unsupported data type!");
    }

    if(nDim == 2) {
        int dim0 = dimensions[0].cast<int>();
        int dim1 = dimensions[1].cast<int>();

        cuphy::tensor_device deviceTensor = cuphy::tensor_device((void*)deviceAddr, deviceDataType, dim0, dim1, cuphy::tensor_flags::align_tight);
        cuphy::tensor_pinned hostTensor = cuphy::tensor_pinned((void*)hostAddr, hostDataType, dim0, dim1, cuphy::tensor_flags::align_tight);

        hostTensor.convert(deviceTensor, (cudaStream_t)cuStream);
        CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)cuStream));
        return hostToNumpy<T>((T*)hostAddr, dim0, dim1);
    }

    else if(nDim ==3) {
        int dim0 = dimensions[0].cast<int>();
        int dim1 = dimensions[1].cast<int>();
        int dim2 = dimensions[2].cast<int>();

        cuphy::tensor_device deviceTensor = cuphy::tensor_device((void*)deviceAddr, deviceDataType, dim0, dim1, dim2, cuphy::tensor_flags::align_tight);
        cuphy::tensor_pinned hostTensor = cuphy::tensor_pinned((void*)hostAddr, hostDataType, dim0, dim1, dim2, cuphy::tensor_flags::align_tight);

        hostTensor.convert(deviceTensor, (cudaStream_t)cuStream);
        CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)cuStream));
        return hostToNumpy<T>((T*)hostAddr, dim0, dim1, dim2);
    }

    else {
        throw std::runtime_error("\nInvalid tensor dimensions!\n");
    }
}

// Explicit instantiations.
template py::array_t<float> deviceToNumpy(uint64_t deviceAddr,
                                          uint64_t hostAddr,
                                          const py::list& dimensions,
                                          uint64_t cuStream);
template py::array_t<std::complex<float>> deviceToNumpy(uint64_t deviceAddr,
                                                        uint64_t hostAddr,
                                                        const py::list& dimensions,
                                                        uint64_t cuStream);


void complexHalfToComplexFloat(uint64_t pComplexHalfAddr,
                               uint64_t pComplexFloatAddr,
                               const py::list& dimensions,
                               uint64_t cuStream) {
    const int nDim = dimensions.size();
    if(nDim == 2) {
        const int dim0 = dimensions[0].cast<int>();
        const int dim1 = dimensions[1].cast<int>();

        cuphy::tensor_device src = cuphy::tensor_device((void*)pComplexHalfAddr, CUPHY_C_16F, dim0, dim1, cuphy::tensor_flags::align_tight);
        cuphy::tensor_device target = cuphy::tensor_device((void*)pComplexFloatAddr, CUPHY_C_32F, dim0, dim1, cuphy::tensor_flags::align_tight);

        target.convert(src, (cudaStream_t)cuStream);
        CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)cuStream));
    }

    else if(nDim ==3) {
        const int dim0 = dimensions[0].cast<int>();
        const int dim1 = dimensions[1].cast<int>();
        const int dim2 = dimensions[2].cast<int>();

        cuphy::tensor_device src = cuphy::tensor_device((void*)pComplexHalfAddr, CUPHY_C_16F, dim0, dim1, dim2, cuphy::tensor_flags::align_tight);
        cuphy::tensor_device target = cuphy::tensor_device((void*)pComplexFloatAddr, CUPHY_C_32F, dim0, dim1, dim2, cuphy::tensor_flags::align_tight);

        target.convert(src, (cudaStream_t)cuStream);
        CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)cuStream));
    }

    else {
        throw std::runtime_error("\nInvalid tensor dimensions!\n");
    }
}


template <typename T>
void fromNumpyBitArray(float* src, T* dst, uint32_t npDim0, uint32_t npDim1) {

    static_assert(std::is_integral<T>::value, "Integral destination required.");

    const int ELEMENT_SIZE = sizeof(T) * 8;

    for(uint32_t col = 0; col < npDim1; col++) {
        for(int row = 0; row < npDim0; row += ELEMENT_SIZE) {
            T bits = 0;

            for(int o = 0; o < ELEMENT_SIZE; o++) {
                if(row + o < npDim0) {
                    float bit = *(src + (npDim1 * (row + o) + col));
                    T bit_0 = (T)bit & 0x1;
                    bits |= (bit_0 << o);
                }
            }

            // Target address. Set the data.
            T* dstElem = dst + (row / ELEMENT_SIZE) + (npDim0 / ELEMENT_SIZE) * col;
            *dstElem = bits;
        }
    }
}

// Explicit instantiations.
template void fromNumpyBitArray(float* src, uint32_t* dst, uint32_t npDim0, uint32_t npDim1);


template <typename T>
void toNumpyBitArray(T* src, float* dst, uint32_t dstDim0, uint32_t dstDim1) {

    static_assert(std::is_integral<T>::value, "Integral source required.");

    const int ELEMENT_SIZE = sizeof(T) * 8;

    uint32_t srcDim0 = (dstDim0 + ELEMENT_SIZE - 1) / ELEMENT_SIZE;
    for(uint32_t col = 0; col < dstDim1; col++) {
        for(uint32_t row = 0; row < dstDim0; row += ELEMENT_SIZE) {
            T* srcElem = src + srcDim0 * col + (row / ELEMENT_SIZE);
            for(int o = 0; o < ELEMENT_SIZE && (row + o < dstDim0); o++) {
                T bit = ((*srcElem & (1 << o)) >> o) & 1;
                float* dstElem = dst + dstDim1 * (row + o) + col;
                *dstElem = (float)bit;
            }
        }
    }
}

// Explicit instantiations.
template void toNumpyBitArray(uint32_t* src, float* dst, uint32_t dstDim0, uint32_t dstDim1);
template void toNumpyBitArray(uint8_t* src, float* dst, uint32_t dstDim0, uint32_t dstDim1);


template <typename T>
py::array_t<T> hostToNumpy(T* dataPtr, uint32_t dim0) {
    return py::array_t<T>(
        {dim0},  // Shape
        {sizeof(T)},  // Strides (in bytes) for each index
        dataPtr  // The data pointer
    );
}

// Explicit instantiations.
template py::array_t<float> hostToNumpy(float* dataPtr, uint32_t dim0);
template py::array_t<std::complex<float>> hostToNumpy(std::complex<float>* dataPtr, uint32_t dim0);


template <typename T>
py::array_t<T> hostToNumpy(T* dataPtr, uint32_t dim0, uint32_t dim1) {
    return py::array_t<T>(
        {dim0, dim1},  // Shape
        {sizeof(T), sizeof(T) * dim0},  // Strides (in bytes) for each index
        dataPtr  // The data pointer
    );
}

// Explicit instantiations.
template py::array_t<float> hostToNumpy(float* dataPtr, uint32_t dim0, uint32_t dim1);
template py::array_t<std::complex<float>> hostToNumpy(std::complex<float>* dataPtr, uint32_t dim0, uint32_t dim1);


template <typename T>
py::array_t<T> hostToNumpy(T* dataPtr, uint32_t dim0, uint32_t dim1, uint32_t dim2) {
    return py::array_t<T>(
        {dim0, dim1, dim2},  // Shape
        {sizeof(T), sizeof(T) * dim0, sizeof(T) * dim0 * dim1},  // Strides (in bytes) for each index
        dataPtr  // The data pointer
    );
}

// Explicit instantiations.
template py::array_t<float> hostToNumpy(float* dataPtr, uint32_t dim0, uint32_t dim1, uint32_t dim2);
template py::array_t<std::complex<float>> hostToNumpy(std::complex<float>* dataPtr, uint32_t dim0, uint32_t dim1, uint32_t dim2);


template <typename T>
py::array_t<T> hostToNumpy(T* dataPtr, uint32_t dim0, uint32_t dim1, uint32_t dim2, uint32_t dim3) {
    return py::array_t<T>(
        {dim0, dim1, dim2, dim3},  // Shape
        {sizeof(T), sizeof(T) * dim0, sizeof(T) * dim0 * dim1, sizeof(T) * dim0 * dim1 * dim2},  // Strides (in bytes) for each index
        dataPtr  // The data pointer
    );
}

// Explicit instantiations.
template py::array_t<float> hostToNumpy(float* dataPtr, uint32_t dim0, uint32_t dim1, uint32_t dim2, uint32_t dim3);
template py::array_t<std::complex<float>> hostToNumpy(std::complex<float>* dataPtr, uint32_t dim0, uint32_t dim1, uint32_t dim2, uint32_t dim3);


template <typename T>
py::array_t<T> hostToNumpy(T* dataPtr, uint32_t dim0, uint32_t dim1, uint32_t dim2, uint32_t dim3, uint32_t dim4) {
    return py::array_t<T>(
        {dim0, dim1, dim2, dim3, dim4},  // Shape
        {sizeof(T), sizeof(T) * dim0, sizeof(T) * dim0 * dim1, sizeof(T) * dim0 * dim1 * dim2, sizeof(T) * dim0 * dim1 * dim2 * dim3},  // Strides (in bytes) for each index
        dataPtr  // The data pointer
    );
}

// Explicit instantiations.
template py::array_t<float> hostToNumpy(float* dataPtr, uint32_t dim0, uint32_t dim1, uint32_t dim2, uint32_t dim3, uint32_t dim4);
template py::array_t<std::complex<float>> hostToNumpy(std::complex<float>* dataPtr, uint32_t dim0, uint32_t dim1, uint32_t dim2, uint32_t dim3, uint32_t dim4);

template <typename T>
py::array_t<T> hostToNumpy(T* dataPtr, const std::vector<size_t>& dims, const std::vector<size_t>& strides) {
    std::vector<size_t> strides_;
    // Default strides.
    if(strides.empty()) {
        static constexpr auto stride = sizeof(T);
        std::transform(begin(dims), end(dims), back_inserter(strides_),
        [stride = stride] (const auto e) mutable {
            const auto prev = std::exchange(stride, stride * e);
            return prev; }
        );
    }
    else {
        strides_ = strides;
    }
    return py::array_t<T>(dims, strides_, dataPtr);
}

// Explicit instantiations.
template py::array_t<float> hostToNumpy(float* dataPtr, const std::vector<size_t>& dims, const std::vector<size_t>& strides);
template py::array_t<int> hostToNumpy(int* dataPtr, const std::vector<size_t>& dims, const std::vector<size_t>& strides);
template py::array_t<std::complex<float>> hostToNumpy(std::complex<float>* dataPtr, const std::vector<size_t>& dims, const std::vector<size_t>& strides);

template <typename T, int flags>
cuphy::tensor_device deviceFromNumpy(const py::array& py_array,
                                     void * inputDevPtr,
                                     cuphyDataType_t convertFromType,
                                     cuphyDataType_t convertToType,
                                     cuphy::tensor_flags hostTensorFlags,
                                     cuphy::tensor_flags deviceTensorFlags,
                                     cudaStream_t cuStream) {
    py::array_t<T, flags> array = py_array;
    py::buffer_info buf = array.request();

    std::vector<int> dims(buf.ndim);
    std::copy(buf.shape.begin(), buf.shape.end(), dims.begin());
    std::vector<int> strides(buf.ndim);
    std::copy(buf.strides.begin(), buf.strides.end(), strides.begin());

    cuphy::tensor_layout layout(buf.ndim, dims.data(), strides.data());
    cuphy::tensor_info info(convertToType, layout);
    cuphy::tensor_pinned hostTensor(convertFromType, layout, hostTensorFlags);
    cuphy::tensor_device deviceTensor(inputDevPtr, info, deviceTensorFlags);

    // Copy the array to pinned memory first.
    size_t nBytes = buf.size * sizeof(T);
    memcpy(hostTensor.addr(), buf.ptr, nBytes);

    // Obtain a tensor in device memory.
    deviceTensor.convert(hostTensor, cuStream);
    CUDA_CHECK(cudaStreamSynchronize(cuStream));

    return deviceTensor;
}

// Explicit instantiations.
template cuphy::tensor_device deviceFromNumpy<float>(
    const py::array& py_array,
    void * inputDevPtr,
    cuphyDataType_t convertFromType,
    cuphyDataType_t convertToType,
    cuphy::tensor_flags hostTensorFlags,
    cuphy::tensor_flags deviceTensorFlags,
    cudaStream_t cuStream);
template cuphy::tensor_device deviceFromNumpy<std::complex<float>>(
    const py::array& py_array,
    void * inputDevPtr,
    cuphyDataType_t convertFromType,
    cuphyDataType_t convertToType,
    cuphy::tensor_flags hostTensorFlags,
    cuphy::tensor_flags deviceTensorFlags,
    cudaStream_t cuStream);
template cuphy::tensor_device deviceFromNumpy<uint8_t>(
    const py::array& py_array,
    void * inputDevPtr,
    cuphyDataType_t convertFromType,
    cuphyDataType_t convertToType,
    cuphy::tensor_flags hostTensorFlags,
    cuphy::tensor_flags deviceTensorFlags,
    cudaStream_t cuStream);
template cuphy::tensor_device deviceFromNumpy<float, py::array::f_style | py::array::forcecast>(
    const py::array& py_array,
    void * inputDevPtr,
    cuphyDataType_t convertFromType,
    cuphyDataType_t convertToType,
    cuphy::tensor_flags hostTensorFlags,
    cuphy::tensor_flags deviceTensorFlags,
    cudaStream_t cuStream);
template cuphy::tensor_device deviceFromNumpy<std::complex<float>, py::array::f_style | py::array::forcecast>(
    const py::array& py_array,
    void * inputDevPtr,
    cuphyDataType_t convertFromType,
    cuphyDataType_t convertToType,
    cuphy::tensor_flags hostTensorFlags,
    cuphy::tensor_flags deviceTensorFlags,
    cudaStream_t cuStream);


template <typename T, int flags>
cuphy::tensor_device deviceFromNumpy(const py::array& py_array,
                                     void * inputDevPtr,
                                     cuphyDataType_t convertFromType,
                                     cuphyDataType_t convertToType,
                                     cuphy::tensor_flags tensorDescFlags,
                                     cudaStream_t cuStream) {
    return deviceFromNumpy<T, flags>(py_array,
                                     inputDevPtr,
                                     convertFromType,
                                     convertToType,
                                     cuphy::tensor_flags::align_tight,
                                     tensorDescFlags,
                                     cuStream);
}


// Explicit instantiations.
template cuphy::tensor_device deviceFromNumpy<float>(
    const py::array& py_array,
    void * inputDevPtr,
    cuphyDataType_t convertFromType,
    cuphyDataType_t convertToType,
    cuphy::tensor_flags tensorDescFlags,
    cudaStream_t cuStream);
template cuphy::tensor_device deviceFromNumpy<std::complex<float>>(
    const py::array& py_array,
    void * inputDevPtr,
    cuphyDataType_t convertFromType,
    cuphyDataType_t convertToType,
    cuphy::tensor_flags tensorDescFlags,
    cudaStream_t cuStream);
template cuphy::tensor_device deviceFromNumpy<uint8_t>(
    const py::array& py_array,
    void * inputDevPtr,
    cuphyDataType_t convertFromType,
    cuphyDataType_t convertToType,
    cuphy::tensor_flags tensorDescFlags,
    cudaStream_t cuStream);
template cuphy::tensor_device deviceFromNumpy<float, py::array::f_style | py::array::forcecast>(
    const py::array& py_array,
    void * inputDevPtr,
    cuphyDataType_t convertFromType,
    cuphyDataType_t convertToType,
    cuphy::tensor_flags tensorDescFlags,
    cudaStream_t cuStream);
template cuphy::tensor_device deviceFromNumpy<std::complex<float>, py::array::f_style | py::array::forcecast>(
    const py::array& py_array,
    void * inputDevPtr,
    cuphyDataType_t convertFromType,
    cuphyDataType_t convertToType,
    cuphy::tensor_flags tensorDescFlags,
    cudaStream_t cuStream);
template cuphy::tensor_device deviceFromNumpy<__half, py::array::f_style | py::array::forcecast>(
    const py::array& py_array,
    void * inputDevPtr,
    cuphyDataType_t convertFromType,
    cuphyDataType_t convertToType,
    cuphy::tensor_flags tensorDescFlags,
    cudaStream_t cuStream);
template cuphy::tensor_device deviceFromNumpy<int, py::array::f_style | py::array::forcecast>(
    const py::array& py_array,
    void * inputDevPtr,
    cuphyDataType_t convertFromType,
    cuphyDataType_t convertToType,
    cuphy::tensor_flags tensorDescFlags,
    cudaStream_t cuStream);


template <typename T, int flags>
cuphy::tensor_device deviceFromNumpy(const py::array& py_array,
                                     cuphyDataType_t convertFromType,
                                     cuphyDataType_t convertToType,
                                     cuphy::tensor_flags tensorDescFlags,
                                     cudaStream_t cuStream) {
    py::array_t<T, flags> array = py_array;
    py::buffer_info buf = array.request();

    std::vector<int> dims(buf.ndim);
    std::copy(buf.shape.begin(), buf.shape.end(), dims.begin());
    std::vector<int> strides(buf.ndim);
    std::copy(buf.strides.begin(), buf.strides.end(), strides.begin());

    cuphy::tensor_layout layout(buf.ndim, dims.data(), strides.data());
    cuphy::tensor_pinned hostTensor(convertFromType, layout, cuphy::tensor_flags::align_tight);
    cuphy::tensor_device deviceTensor(convertToType, layout, tensorDescFlags);

    // Copy the array to pinned memory first.
    size_t nBytes = buf.size * sizeof(T);
    memcpy(hostTensor.addr(), buf.ptr, nBytes);

    // Obtain a tensor in device memory.
    deviceTensor.convert(hostTensor, cuStream);
    CUDA_CHECK(cudaStreamSynchronize(cuStream));

    return deviceTensor;
}

// Explicit instantiations.
template cuphy::tensor_device deviceFromNumpy<float>(
    const py::array& py_array,
    cuphyDataType_t convertFromType,
    cuphyDataType_t convertToType,
    cuphy::tensor_flags tensorDescFlags,
    cudaStream_t cuStream);
template cuphy::tensor_device deviceFromNumpy<std::complex<float>>(
    const py::array& py_array,
    cuphyDataType_t convertFromType,
    cuphyDataType_t convertToType,
    cuphy::tensor_flags tensorDescFlags,
    cudaStream_t cuStream);
template cuphy::tensor_device deviceFromNumpy<uint8_t>(
    const py::array& py_array,
    cuphyDataType_t convertFromType,
    cuphyDataType_t convertToType,
    cuphy::tensor_flags tensorDescFlags,
    cudaStream_t cuStream);
template cuphy::tensor_device deviceFromNumpy<float, py::array::f_style | py::array::forcecast>(
    const py::array& py_array,
    cuphyDataType_t convertFromType,
    cuphyDataType_t convertToType,
    cuphy::tensor_flags tensorDescFlags,
    cudaStream_t cuStream);
template cuphy::tensor_device deviceFromNumpy<std::complex<float>, py::array::f_style | py::array::forcecast>(
    const py::array& py_array,
    cuphyDataType_t convertFromType,
    cuphyDataType_t convertToType,
    cuphy::tensor_flags tensorDescFlags,
    cudaStream_t cuStream);


template <typename T>
cuphy::tensor_device deviceFromCudaArray(const cuda_array_t<T>& cudaArray,
                                         void* pDevAddr,
                                         cuphyDataType_t convertFromType,
                                         cuphyDataType_t convertToType,
                                         cuphy::tensor_flags tensorDescFlags,
                                         cudaStream_t cuStream) {

    const std::vector<size_t>& vdims = cudaArray.get_shape();

    std::vector<int> dims(std::size(vdims));
    std::copy(vdims.begin(), vdims.end(), dims.begin());

    auto bufAddr = cudaArray.get_device_ptr();

    cuphy::tensor_layout layout(vdims.size(), dims.data(), nullptr);
    cuphy::tensor_info toInfo(convertToType, layout);

    if(!pDevAddr) {  // Additional device memory not pre-allocated.
        if(convertToType != convertFromType) {
            cuphy::tensor_info fromInfo(convertFromType, layout);
            cuphy::tensor_device fromTensor(bufAddr, fromInfo, tensorDescFlags);

            // Allocates device memory for the different type of array.
            cuphy::tensor_device toTensor(toInfo, tensorDescFlags);
            toTensor.convert(fromTensor, cuStream);
            return toTensor;
        }

        // Just wrap the cuPy array into cuPHY tensor (same device memory).
        cuphy::tensor_device toTensor(bufAddr, toInfo, tensorDescFlags);
        return toTensor;
    }

    // Pre-allocated device memory - copy the cuPy array content over.
    cuphy::tensor_info fromInfo(convertFromType, layout);
    cuphy::tensor_device fromTensor(bufAddr, fromInfo, tensorDescFlags);
    cuphy::tensor_device toTensor(pDevAddr, toInfo, tensorDescFlags);
    toTensor.convert(fromTensor, cuStream);
    return toTensor;
}

// Explicit instantiations.
template cuphy::tensor_device deviceFromCudaArray(const cuda_array_t<uint32_t>& cudaArray,
                                                  void* pDevAddr,
                                                  cuphyDataType_t convertFromType,
                                                  cuphyDataType_t convertToType,
                                                  cuphy::tensor_flags tensorDescFlags,
                                                  cudaStream_t cuStream);
template cuphy::tensor_device deviceFromCudaArray(const cuda_array_t<uint8_t>& cudaArray,
                                                  void* pDevAddr,
                                                  cuphyDataType_t convertFromType,
                                                  cuphyDataType_t convertToType,
                                                  cuphy::tensor_flags tensorDescFlags,
                                                  cudaStream_t cuStream);
template cuphy::tensor_device deviceFromCudaArray(const cuda_array_t<__half>& cudaArray,
                                                  void* pDevAddr,
                                                  cuphyDataType_t convertFromType,
                                                  cuphyDataType_t convertToType,
                                                  cuphy::tensor_flags tensorDescFlags,
                                                  cudaStream_t cuStream);
template cuphy::tensor_device deviceFromCudaArray(const cuda_array_t<float>& cudaArray,
                                                  void* pDevAddr,
                                                  cuphyDataType_t convertFromType,
                                                  cuphyDataType_t convertToType,
                                                  cuphy::tensor_flags tensorDescFlags,
                                                  cudaStream_t cuStream);
template cuphy::tensor_device deviceFromCudaArray(const cuda_array_t<std::complex<float>>& cudaArray,
                                                  void* pDevAddr,
                                                  cuphyDataType_t convertFromType,
                                                  cuphyDataType_t convertToType,
                                                  cuphy::tensor_flags tensorDescFlags,
                                                  cudaStream_t cuStream);

template <typename T>
cuda_array_t<T> deviceToCudaArray(void* pDevAddr,
                                  void* pAddr,
                                  const std::vector<size_t>& shape,
                                  cuphyDataType_t devDataType,
                                  cuphyDataType_t cudaArrayDataType,
                                  cuphy::tensor_flags tensorDescFlags,
                                  cudaStream_t cuStream) {
    std::vector<size_t> strides;
    // F-order strides as in cuPHY.
    static constexpr auto stride = sizeof(T);
    std::transform(begin(shape), end(shape), back_inserter(strides),
    [stride = stride] (const auto e) mutable {
        const auto prev = std::exchange(stride, stride * e);
        return prev; }
    );

    return deviceToCudaArray<T>(pDevAddr,
                                pAddr,
                                shape,
                                strides,
                                devDataType,
                                cudaArrayDataType,
                                tensorDescFlags,
                                cuStream);
}

// Explicit instantiations.
template cuda_array_t<int> deviceToCudaArray(void* pDevAddr,
                                             void* pAddr,
                                             const std::vector<size_t>& shape,
                                             cuphyDataType_t devDataType,
                                             cuphyDataType_t cudaArrayDataType,
                                             cuphy::tensor_flags tensorDescFlags,
                                             cudaStream_t cuStream);
template cuda_array_t<__half> deviceToCudaArray(void* pDevAddr,
                                                void* pAddr,
                                                const std::vector<size_t>& shape,
                                                cuphyDataType_t devDataType,
                                                cuphyDataType_t cudaArrayDataType,
                                                cuphy::tensor_flags tensorDescFlags,
                                                cudaStream_t cuStream);
template cuda_array_t<float> deviceToCudaArray(void* pDevAddr,
                                               void* pAddr,
                                               const std::vector<size_t>& shape,
                                               cuphyDataType_t devDataType,
                                               cuphyDataType_t cudaArrayDataType,
                                               cuphy::tensor_flags tensorDescFlags,
                                               cudaStream_t cuStream);
template cuda_array_t<std::complex<float>> deviceToCudaArray(void* pDevAddr,
                                                             void* pAddr,
                                                             const std::vector<size_t>& shape,
                                                             cuphyDataType_t devDataType,
                                                             cuphyDataType_t cudaArrayDataType,
                                                             cuphy::tensor_flags tensorDescFlags,
                                                             cudaStream_t cuStream);


template <typename T>
cuda_array_t<T> deviceToCudaArray(void* pDevAddr,
                                  void* pAddr,
                                  const std::vector<size_t>& shape,
                                  const std::vector<size_t>& strides,
                                  cuphyDataType_t devDataType,
                                  cuphyDataType_t cudaArrayDataType,
                                  cuphy::tensor_flags tensorDescFlags,
                                  cudaStream_t cuStream) {
    std::vector<int> dims(std::size(shape));
    std::copy(shape.begin(), shape.end(), dims.begin());
    cuphy::tensor_layout layout(dims.size(), dims.data(), nullptr);

    cuphy::tensor_info cudaArrayInfo(cudaArrayDataType, layout);
    cuphy::tensor_device cudaArray(pAddr, cudaArrayInfo, tensorDescFlags);

    cuphy::tensor_info devInfo(devDataType, layout);
    cuphy::tensor_device devTensor(pDevAddr, devInfo, tensorDescFlags);

    cudaArray.convert(devTensor, cuStream);

    return cuda_array_t<T>(reinterpret_cast<intptr_t>(pAddr), shape, strides);
}

// Explicit instantiations.
template cuda_array_t<int> deviceToCudaArray(void* pDevAddr,
                                             void* pAddr,
                                             const std::vector<size_t>& shape,
                                             const std::vector<size_t>& strides,
                                             cuphyDataType_t devDataType,
                                             cuphyDataType_t cudaArrayDataType,
                                             cuphy::tensor_flags tensorDescFlags,
                                             cudaStream_t cuStream);
template cuda_array_t<__half> deviceToCudaArray(void* pDevAddr,
                                                void* pAddr,
                                                const std::vector<size_t>& shape,
                                                const std::vector<size_t>& strides,
                                                cuphyDataType_t devDataType,
                                                cuphyDataType_t cudaArrayDataType,
                                                cuphy::tensor_flags tensorDescFlags,
                                                cudaStream_t cuStream);
template cuda_array_t<float> deviceToCudaArray(void* pDevAddr,
                                               void* pAddr,
                                               const std::vector<size_t>& shape,
                                               const std::vector<size_t>& strides,
                                               cuphyDataType_t devDataType,
                                               cuphyDataType_t cudaArrayDataType,
                                               cuphy::tensor_flags tensorDescFlags,
                                               cudaStream_t cuStream);
template cuda_array_t<std::complex<float>> deviceToCudaArray(void* pDevAddr,
                                                             void* pAddr,
                                                             const std::vector<size_t>& shape,
                                                             const std::vector<size_t>& strides,
                                                             cuphyDataType_t devDataType,
                                                             cuphyDataType_t cudaArrayDataType,
                                                             cuphy::tensor_flags tensorDescFlags,
                                                             cudaStream_t cuStream);


template <typename T>
cuda_array_t<T> deviceToCudaArray(void* pAddr,
                                  const std::vector<size_t> &shape) {
    std::vector<size_t> strides;
    static constexpr auto stride = sizeof(T);
    std::transform(begin(shape), end(shape), back_inserter(strides),
    [stride = stride] (const auto e) mutable {
        const auto prev = std::exchange(stride, stride * e);
        return prev; }
    );
    return deviceToCudaArray<T>(pAddr, shape, strides);
}

// Explicit instantiations.
template cuda_array_t<int> deviceToCudaArray(void* pAddr,
                                             const std::vector<size_t> &shape);
template cuda_array_t<uint8_t> deviceToCudaArray(void* pAddr,
                                                 const std::vector<size_t> &shape);
template cuda_array_t<uint32_t> deviceToCudaArray(void* pAddr,
                                                 const std::vector<size_t> &shape);
template cuda_array_t<__half> deviceToCudaArray(void* pAddr,
                                                const std::vector<size_t> &shape);
template cuda_array_t<float> deviceToCudaArray(void* pAddr,
                                               const std::vector<size_t> &shape);
template cuda_array_t<std::complex<float>>deviceToCudaArray(void* pAddr,
                                                            const std::vector<size_t> &shape);


template <typename T>
cuda_array_t<T> deviceToCudaArray(void* pAddr,
                                  const std::vector<size_t> &shape,
                                  const std::vector<size_t> &strides) {
    return cuda_array_t<T>(reinterpret_cast<intptr_t>(pAddr), shape, strides);
}

// Explicit instantiations.
template cuda_array_t<int> deviceToCudaArray(void* pAddr,
                                             const std::vector<size_t> &shape,
                                             const std::vector<size_t> &strides);
template cuda_array_t<uint8_t> deviceToCudaArray(void* pAddr,
                                                 const std::vector<size_t> &shape,
                                                 const std::vector<size_t> &strides);
template cuda_array_t<uint32_t> deviceToCudaArray(void* pAddr,
                                                 const std::vector<size_t> &shape,
                                                 const std::vector<size_t> &strides);
template cuda_array_t<__half> deviceToCudaArray(void* pAddr,
                                                const std::vector<size_t> &shape,
                                                const std::vector<size_t> &strides);
template cuda_array_t<float> deviceToCudaArray(void* pAddr,
                                               const std::vector<size_t> &shape,
                                               const std::vector<size_t> &strides);
template cuda_array_t<std::complex<float>>deviceToCudaArray(void* pAddr,
                                                            const std::vector<size_t> &shape,
                                                            const std::vector<size_t> &strides);

template <typename T>
std::unique_ptr<cuda_array_t<T>> deviceToCudaArrayPtr(void* pAddr, const std::vector<size_t>& shape) {
    std::vector<size_t> strides;
    // F-order strides as in cuPHY.
    static constexpr auto stride = sizeof(T);
    std::transform(begin(shape), end(shape), back_inserter(strides),
    [stride = stride] (const auto e) mutable {
        const auto prev = std::exchange(stride, stride * e);
        return prev; }
    );
    return deviceToCudaArrayPtr<T>(pAddr, shape, strides);
}

// Explicit instantiations.
template std::unique_ptr<cuda_array_t<uint32_t>> deviceToCudaArrayPtr(void* pAddr, const std::vector<size_t>& shape);
template std::unique_ptr<cuda_array_t<uint8_t>> deviceToCudaArrayPtr(void* pAddr, const std::vector<size_t>& shape);
template std::unique_ptr<cuda_array_t<__half>> deviceToCudaArrayPtr(void* pAddr, const std::vector<size_t>& shape);
template std::unique_ptr<cuda_array_t<float>> deviceToCudaArrayPtr(void* pAddr, const std::vector<size_t>& shape);
template std::unique_ptr<cuda_array_t<std::complex<float>>> deviceToCudaArrayPtr(void* pAddr, const std::vector<size_t>& shape);

template <typename T>
std::unique_ptr<cuda_array_t<T>> deviceToCudaArrayPtr(void* pAddr, const std::vector<size_t>& shape, const std::vector<size_t>& strides) {
    return std::make_unique<cuda_array_t<T>>(reinterpret_cast<intptr_t>(pAddr), shape, strides);
}

// Explicit instantiations.
template std::unique_ptr<cuda_array_t<uint32_t>> deviceToCudaArrayPtr(void* pAddr, const std::vector<size_t>& shape, const std::vector<size_t>& strides);
template std::unique_ptr<cuda_array_t<uint8_t>> deviceToCudaArrayPtr(void* pAddr, const std::vector<size_t>& shape, const std::vector<size_t>& strides);
template std::unique_ptr<cuda_array_t<__half>> deviceToCudaArrayPtr(void* pAddr, const std::vector<size_t>& shape, const std::vector<size_t>& strides);
template std::unique_ptr<cuda_array_t<float>> deviceToCudaArrayPtr(void* pAddr, const std::vector<size_t>& shape, const std::vector<size_t>& strides);
template std::unique_ptr<cuda_array_t<std::complex<float>>> deviceToCudaArrayPtr(void* pAddr, const std::vector<size_t>& shape, const std::vector<size_t>& strides);


template <typename T>
T* numpyArrayToPtr(const py::array& py_array) {
    py::array_t<T, py::array::f_style | py::array::forcecast> array = py_array;
    py::buffer_info buf = array.request();
    T* ptr = static_cast<T*>(buf.ptr);
    return ptr;
}

// Explicit instantiations.
template uint8_t* numpyArrayToPtr(const py::array& py_array);
template uint16_t* numpyArrayToPtr(const py::array& py_array);
template uint32_t* numpyArrayToPtr(const py::array& py_array);
template float* numpyArrayToPtr(const py::array& py_array);


template <typename T>
void copyTensorData(cuphy::tensor_ref& tensor, T& tInfo) {
    tInfo.pAddr              = tensor.addr();
    const tensor_desc& tDesc = static_cast<const tensor_desc&>(*(tensor.desc().handle()));
    tInfo.elemType           = tDesc.type();
    std::copy_n(tDesc.layout().strides.begin(), std::extent<decltype(tInfo.strides)>::value, tInfo.strides);
}


// Explicit instantiations.
template void copyTensorData(cuphy::tensor_ref& tensor, cuphyTensorInfo1_t& tInfo);
template void copyTensorData(cuphy::tensor_ref& tensor, cuphyTensorInfo2_t& tInfo);
template void copyTensorData(cuphy::tensor_ref& tensor, cuphyTensorInfo3_t& tInfo);
template void copyTensorData(cuphy::tensor_ref& tensor, cuphyTensorInfo4_t& tInfo);
template void copyTensorData(cuphy::tensor_ref& tensor, cuphyTensorInfo5_t& tInfo);

template <typename T>
void copyTensorData(const cuphy::tensor_device& tensor, T& tInfo) {
    tInfo.pAddr              = tensor.addr();
    const tensor_desc& tDesc = static_cast<const tensor_desc&>(*(tensor.desc().handle()));
    tInfo.elemType           = tDesc.type();
    std::copy_n(tDesc.layout().strides.begin(), std::extent<decltype(tInfo.strides)>::value, tInfo.strides);
}

// Explicit instantiations.
template void copyTensorData(const cuphy::tensor_device& tensor, cuphyTensorInfo1_t& tInfo);
template void copyTensorData(const cuphy::tensor_device& tensor, cuphyTensorInfo2_t& tInfo);
template void copyTensorData(const cuphy::tensor_device& tensor, cuphyTensorInfo3_t& tInfo);
template void copyTensorData(const cuphy::tensor_device& tensor, cuphyTensorInfo4_t& tInfo);
template void copyTensorData(const cuphy::tensor_device& tensor, cuphyTensorInfo5_t& tInfo);


void expandDmrsBitmap(const pybind11::list& bitMask,
                      const uint8_t startSym,
                      const uint8_t numPdschSyms,
                      uint16_t& dmrsSymLoc,
                      uint64_t& dataSymLoc,
                      uint8_t& numDmrsSyms,
                      uint8_t& numDataSyms)
{
    dmrsSymLoc = 0;
    dataSymLoc = 0;
    numDmrsSyms = 0;
    numDataSyms = 0;

    // Process each bit in the mask
    for (int i = startSym; i < (startSym + numPdschSyms); i++) {
        const auto bit = bitMask[i].cast<int>();

        if (bit == 1) {
            // DMRS symbol - set bit in dmrsSymLoc
            dmrsSymLoc |= ((static_cast<uint16_t>(i) & 0xF) << (numDmrsSyms * 4));
            numDmrsSyms++;
        } else {
            // Data symbol - set bit in dataSymLoc
            dataSymLoc |= ((static_cast<uint64_t>(i) & 0xF) << (numDataSyms * 4));
            numDataSyms++;
        }
    }
}


void readRbBitmap(const py::array& rbBitmap, uint32_t* pRbBitmap) {
    // Get array buffer and length
    const py::buffer_info& buf = rbBitmap.request();
    const size_t length = buf.size;
    const auto* data = static_cast<const uint8_t*>(buf.ptr);
    std::memcpy(pRbBitmap, data, length);
}


void readPrecodingMatrix(const py::array_t<std::complex<float>>& pyPrecodingMatrix, __half2* precodingMatrix, uint8_t& numPorts) {

    const py::buffer_info& buf = pyPrecodingMatrix.request();
    const auto* data = std::bit_cast<const std::complex<float>*>(buf.ptr);

    // Set number of ports (columns)
    numPorts = static_cast<uint8_t>(buf.shape[1]);

    // Copy and convert each complex value to __half2
    for (size_t idx = 0; idx < buf.size; idx++) {
        precodingMatrix[idx].x = __float2half(data[idx].real());
        precodingMatrix[idx].y = __float2half(data[idx].imag());
    }
}


uint32_t get_tb_size(const int numSymbols,
                     const int numDmrsSymbols,
                     const int numPrbs,
                     const int numLayers,
                     const float codeRate,
                     const uint32_t modOrder,
                     const int numDmrsCdmGrpsNoData) {
    const int numDmrsCdmGrpsNoData1Symbols = (numDmrsCdmGrpsNoData == 1) ? numDmrsSymbols : 0;
    uint32_t numCbs = 0, tbSize = 0;
    get_TB_size_and_num_CBs(numSymbols,
                            numPrbs,
                            numLayers,
                            codeRate,
                            modOrder,
                            numCbs,
                            tbSize,
                            numDmrsCdmGrpsNoData1Symbols);
    return tbSize;
}


}  // pycuphy
