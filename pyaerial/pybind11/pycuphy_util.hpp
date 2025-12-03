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

#ifndef PYCUPHY_UTIL_HPP
#define PYCUPHY_UTIL_HPP

#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "cuphy.h"
#include "cuphy.hpp"
#include "cuphy_api.h"
#include "cuda_array_interface.hpp"

namespace py = pybind11;

namespace pycuphy {

template <typename T>
py::array_t<T> deviceToNumpy(uint64_t deviceAddr,
                             uint64_t hostAddr,
                             const py::list& dimensions,
                             uint64_t cuStream);

void complexHalfToComplexFloat(uint64_t pComplexHalfAddr,
                               uint64_t pComplexFloatAddr,
                               const py::list& dimensions,
                               uint64_t cuStream);

template <typename T>
void fromNumpyBitArray(float* src, T* dst, uint32_t npDim0, uint32_t npDim1);


template <typename T>
void toNumpyBitArray(T* src, float* dst, uint32_t dstDim0, uint32_t dstDim1);


template <typename T>
__attribute__((visibility("default")))
py::array_t<T> hostToNumpy(T* dataPtr, uint32_t dim0);

template <typename T>
__attribute__((visibility("default")))
py::array_t<T> hostToNumpy(T* dataPtr, uint32_t dim0, uint32_t dim1);


template <typename T>
__attribute__((visibility("default")))
py::array_t<T> hostToNumpy(T* dataPtr, uint32_t dim0, uint32_t dim1, uint32_t dim2);


template <typename T>
__attribute__((visibility("default")))
py::array_t<T> hostToNumpy(T* dataPtr, uint32_t dim0, uint32_t dim1, uint32_t dim2, uint32_t dim3);


template <typename T>
__attribute__((visibility("default")))
py::array_t<T> hostToNumpy(T* dataPtr, uint32_t dim0, uint32_t dim1, uint32_t dim2, uint32_t dim3, uint32_t dim4);

template <typename T>
__attribute__((visibility("default")))
py::array_t<T> hostToNumpy(T* dataPtr, const std::vector<size_t>& dims, const std::vector<size_t>& strides);

template <typename T, int flags = py::array::c_style | py::array::forcecast>
__attribute__((visibility("default")))
cuphy::tensor_device deviceFromNumpy(const py::array& py_array,
                                     void * inputDevPtr,
                                     cuphyDataType_t convertFromType,
                                     cuphyDataType_t convertToType,
                                     cuphy::tensor_flags tensorDescFlags,
                                     cudaStream_t cuStream);

template <typename T, int flags = py::array::c_style | py::array::forcecast>
__attribute__((visibility("default")))
cuphy::tensor_device deviceFromNumpy(const py::array& py_array,
                                     void * inputDevPtr,
                                     cuphyDataType_t convertFromType,
                                     cuphyDataType_t convertToType,
                                     cuphy::tensor_flags hostTensorFlags,
                                     cuphy::tensor_flags deviceTensorFlags,
                                     cudaStream_t cuStream);

template <typename T, int flags = py::array::c_style | py::array::forcecast>
__attribute__((visibility("default")))
cuphy::tensor_device deviceFromNumpy(const py::array& py_array,
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
                                         cudaStream_t cuStream);

template <typename T>
cuda_array_t<T> deviceToCudaArray(void* pDevAddr,
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
                                  cudaStream_t cuStream);

template <typename T>
cuda_array_t<T> deviceToCudaArray(void* pAddr,
                                  const std::vector<size_t> &shape);

template <typename T>
cuda_array_t<T> deviceToCudaArray(void* pAddr,
                                  const std::vector<size_t> &shape,
                                  const std::vector<size_t> &strides);

template <typename T>
std::unique_ptr<cuda_array_t<T>> deviceToCudaArrayPtr(void* pAddr, const std::vector<size_t>& shape);

template <typename T>
std::unique_ptr<cuda_array_t<T>> deviceToCudaArrayPtr(void* pAddr, const std::vector<size_t>& shape, const std::vector<size_t> &strides);


template <typename T>
T* numpyArrayToPtr(const py::array& py_array);

template <typename T>
void copyTensorData(cuphy::tensor_ref& tensor, T& tInfo);

template <typename T>
void copyTensorData(const cuphy::tensor_device& tensor, T& tInfo);

/**
 * @brief Expands a DMRS bitmap into separate bit positions for DMRS and data symbols
 *
 * Processes a bitmap mask indicating which OFDM symbols are DMRS (1) and which are data (0).
 * For each DMRS symbol, its position is encoded in 4 bits in dmrsSymLoc.
 * For each data symbol, its position is encoded in 4 bits in dataSymLoc.
 *
 * @param[in] bitMask Python list containing the bitmap (1 for DMRS, 0 for data)
 * @param[in] startSym Starting symbol index to process
 * @param[in] numPdschSyms Number of PDSCH symbols to process from startSym
 * @param[out] dmrsSymLoc Encoded positions of DMRS symbols (4 bits per position)
 * @param[out] dataSymLoc Encoded positions of data symbols (4 bits per position)
 * @param[out] numDmrsSyms Total number of DMRS symbols found
 * @param[out] numDataSyms Total number of data symbols found
 */
void expandDmrsBitmap(const pybind11::list& bitMask,
                      const uint8_t startSym,
                      const uint8_t numPdschSyms,
                      uint16_t& dmrsSymLoc,
                      uint64_t& dataSymLoc,
                      uint8_t& numDmrsSyms,
                      uint8_t& numDataSyms);

/**
 * @brief Reads a resource block bitmap from a Python array into a C++ array
 *
 * Copies the resource block bitmap data directly from the Python array buffer
 * to the provided destination buffer.
 *
 * @param[in] rbBitmap Python array containing the resource block bitmap
 * @param[out] pRbBitmap Pointer to the destination buffer for the bitmap data
 */
void readRbBitmap(const py::array& rbBitmap, uint32_t* pRbBitmap);

/**
 * @brief Converts a precoding matrix from Numpy complex float format to CUDA half2 format
 *
 * Reads a complex floating-point precoding matrix from Python and converts each complex value
 * to a CUDA __half2 type, where the real part is stored in x and the imaginary part in y.
 * Also extracts the number of ports from the second dimension of the matrix.
 *
 * @param[in] pyPrecodingMatrix Python array containing complex float precoding coefficients
 * @param[out] precodingMatrix Destination buffer for converted __half2 values
 * @param[out] numPorts Number of antenna ports (extracted from matrix columns)
 */
void readPrecodingMatrix(const py::array_t<std::complex<float>>& pyPrecodingMatrix,
                         __half2* precodingMatrix,
                         uint8_t& numPorts);

/**
 * @brief Calculates the transport block size (in bits) based on the number of symbols, PRBs, layers, code rate, modulation order, and number of DMRS CDM groups without data symbols
 *
 * This function computes the transport block size (TB) for a given set of parameters:
 * - Number of symbols for the transport block not including DMRS symbols
 * - Number of DMRS symbols
 * - Number of PRBs
 * - Number of layers
 * - Code rate
 * - Modulation order
 * - Number of DMRS CDM groups without data symbols
 *
 * The function uses the get_TB_size_and_num_CBs function to compute the TB size and number of code blocks (CBs).
 *
 * @param[in] numSymbols Number of symbols for the transport block not including DMRS symbols
 * @param[in] numDmrsSymbols Number of DMRS symbols
 * @param[in] numPrbs Number of PRBs
 * @param[in] numLayers Number of layers
 * @param[in] codeRate Code rate
 * @param[in] modOrder Modulation order
 * @param[in] numDmrsCdmGrpsNoData Number of DMRS CDM groups without data symbols
 * @return Transport block size (TB) in bits
 */
__attribute__((visibility("default")))
[[nodiscard]] uint32_t get_tb_size(int numSymbols,
                                   int numDmrsSymbols,
                                   int numPrbs,
                                   int numLayers,
                                   float codeRate,
                                   uint32_t modOrder,
                                   int numDmrsCdmGrpsNoData);
};

#endif // PYCUPHY_UTIL_HPP
