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

#ifndef PYCUPHY_LDPC_HPP
#define PYCUPHY_LDPC_HPP

#include <pybind11/pybind11.h>

#include <vector>
#include "cuphy.h"
#include "util.hpp"
#include "cuphy.hpp"
#include "cuda_array_interface.hpp"
#include "pusch_utils.hpp"
#include "pycuphy_params.hpp"
#include "ldpc/ldpc_api.hpp"


namespace pycuphy {

//////////////////////////////////////////////////////////////////////////////
// Common utility functions

/**
 * @brief Set PDSCH transport block parameters
 *
 * @param tbParams Output parameter structure to be filled
 * @param tbStartAddr Pointer to transport block start address
 * @param tbStartOffset Offset into the transport block
 * @param tbSize Size of the transport block in bits
 * @param cumulativeTbSizePadding Cumulative padding size
 * @param codeRate Code rate for LDPC encoding
 * @param rateMatchLen Length after rate matching
 * @param qamMod QAM modulation order
 * @param numCodedBits Number of coded bits
 * @param rv Redundancy version
 * @param numLayers Number of layers
 * @param cinit Scrambling initialization value
 */
void setPdschPerTbParams(PdschPerTbParams& tbParams,
                         uint8_t* tbStartAddr,
                         uint32_t tbStartOffset,
                         uint32_t tbSize,
                         uint32_t cumulativeTbSizePadding,
                         float codeRate,
                         uint32_t rateMatchLen,
                         uint8_t qamMod,
                         uint32_t numCodedBits,
                         uint8_t rv,
                         uint8_t numLayers,
                         uint32_t cinit);

/**
 * @brief Set per transport block parameters for LDPC processing
 *
 * @param tbParams Output parameter structure to be filled
 * @param ldpcParams LDPC parameters structure to be filled
 * @param tbSize Size of the transport block in bits
 * @param codeRate Code rate for LDPC encoding
 * @param qamMod QAM modulation order
 * @param ndi New data indicator
 * @param rv Redundancy version
 * @param rateMatchLen Length after rate matching
 * @param cinit Scrambling initialization value
 * @param userGroupIdx User group index
 * @param numLayers Number of layers
 * @param numUeGrpLayers Number of UE group layers
 * @param layerMapArray Layer mapping array
 * @param nDmrsCdmGrpsNoData Number of DMRS CDM groups without data
 */
void setPerTbParams(PerTbParams& tbParams,
                    cuphyLDPCParams& ldpcParams,
                    uint32_t tbSize,
                    float codeRate,
                    uint8_t qamMod,
                    uint32_t ndi,
                    uint32_t rv,
                    uint32_t rateMatchLen,
                    uint32_t cinit,
                    uint32_t userGroupIdx,
                    uint8_t numLayers,
                    uint8_t numUeGrpLayers,
                    const std::vector<uint32_t>& layerMapArray,
                    uint8_t nDmrsCdmGrpsNoData = 2);

/**
 * @brief Read PDSCH DMRS parameters from Python configuration objects.
 *
 * This function extracts DMRS parameters from a list of PDSCH configurations and populates
 * a vector of PdschDmrsParams structs used by cuPHY.
 *
 * @param[in] pdschConfigs List of Python PDSCH configuration objects containing DMRS parameters
 * @param[in] slot Slot number
 * @param[in] cellId Physical cell ID
 * @param[in] nPrbDlBwp Number of PRBs in downlink bandwidth part
 * @param[in] cellOutputTensorAddr Output tensor address for the cell
 * @param[out] numTbs Number of transport blocks
 * @param[out] dmrsPrms Buffer to store the extracted DMRS parameters
 */

void readDmrsParams(const std::vector<pybind11::object>& pdschConfigs,
                    uint32_t slot,
                    uint16_t cellId,
                    uint16_t nPrbDlBwp,
                    __half2* cellOutputTensorAddr,
                    uint32_t& numTbs,
                    cuphy::buffer<PdschDmrsParams, cuphy::pinned_alloc>& dmrsPrms);

/**
 * @brief Read PDSCH transport block parameters from Python configuration objects.
 *
 * This function extracts transport block parameters from a list of PDSCH configurations and populates
 * a vector of PdschPerTbParams structs used by cuPHY.
 *
 * @param[in] pdschConfigs List of Python PDSCH configuration objects containing transport block parameters
 * @param[in] tbInputAddr Input tensor address for the transport blocks
 * @param[out] numTbs Number of transport blocks
 * @param[out] tbPrms Buffer to store the extracted transport block parameters
  */

void readTbParams(const std::vector<pybind11::object>& pdschConfigs,
                  uint8_t* tbInputAddr,
                  uint32_t& numTbs,
                  cuphy::buffer<PdschPerTbParams, cuphy::pinned_alloc>& tbPrms);



/**
 * @brief Core LDPC encoder class for CUDA processing
 *
 * This class handles the low-level LDPC encoding operations on CUDA devices.
 * It manages CUDA resources and provides the interface for running encoding
 * operations on input data.
 */
class LdpcEncoder final {
public:
    /**
     * @brief Construct a new LDPC encoder
     *
     * @param outputDevicePtr Pointer to device memory for output
     * @param cuStream CUDA stream for asynchronous operations
     */
    explicit LdpcEncoder(void* outputDevicePtr, cudaStream_t cuStream);

    /**
     * @brief Encode input data using LDPC encoding
     *
     * @param inputData Input tensor containing data to encode
     * @param tbParams Transport block parameters
     * @return const cuphy::tensor_device& Reference to output tensor
     */
    [[nodiscard]] const cuphy::tensor_device& encode(const cuphy::tensor_device& inputData,
                                                     const std::vector<PdschPerTbParams>& tbParams);

    /**
     * @brief Get the code block size
     *
     * @return uint32_t Code block size
     */
    [[nodiscard]] uint32_t getCbSize() const { return m_effN; }

    /**
     * @brief Set puncturing mode
     *
     * @param puncture Puncturing mode (0 or 1)
     */
    void setPuncturing(uint8_t puncture);

private:
    void*                           m_outputDevicePtr;  ///< Device pointer for output
    cuphy::tensor_device            m_dOutputTensor;    ///< Output tensor
    uint8_t                         m_puncture{};       ///< Puncturing mode
    uint32_t                        m_effN{};           ///< Effective code block size
    cuphy::buffer<uint8_t, cuphy::device_alloc> m_dLdpcDesc;  ///< LDPC descriptor (device)
    cuphy::buffer<uint8_t, cuphy::pinned_alloc> m_hLdpcDesc;  ///< LDPC descriptor (host)
    cuphy::buffer<uint8_t, cuphy::device_alloc> m_dWorkspace; ///< Device workspace
    cuphy::buffer<uint8_t, cuphy::pinned_alloc> m_hWorkspace; ///< Host workspace
    cudaStream_t m_cuStream{};                          ///< CUDA stream handle
};

/**
 * @brief Python interface for LDPC encoding
 *
 * This class provides a Python-friendly interface for LDPC encoding,
 * handling the conversion between Python and C++ data types and managing
 * the underlying CUDA resources.
 */
class __attribute__((visibility("default"))) PyLdpcEncoder final {
public:
    /**
     * @brief Construct a new Python LDPC encoder
     *
     * @param outputDevicePtr Pointer to device memory for output
     * @param cuStream CUDA stream for asynchronous operations
     */
    explicit PyLdpcEncoder(uint64_t outputDevicePtr, uint64_t cuStream);

    /**
     * @brief Encode input data using LDPC encoding
     *
     * @param inputData Input tensor containing data to encode
     * @param tbSize Transport block size
     * @param codeRate Code rate for LDPC encoding
     * @param rv Redundancy version
     * @return const cuda_array_t<uint32_t>& Reference to output tensor
     */
    [[nodiscard]] const cuda_array_t<uint32_t>& encode(
        const cuda_array_t<uint32_t>& inputData,
        uint32_t tbSize,
        float codeRate,
        int rv
    );

    /**
     * @brief Set puncturing mode
     *
     * @param puncture Puncturing mode (0 or 1)
     */
    void setPuncturing(uint8_t puncture);

    /**
     * @brief Get the code block size
     *
     * @return uint32_t Code block size
     */
    [[nodiscard]] uint32_t getCbSize() const { return m_ldpcEncoder.getCbSize(); }

private:
    std::unique_ptr<cuda_array_t<uint32_t>> m_encodedBits;  ///< Encoded bits buffer
    std::vector<PdschPerTbParams> m_tbParams;               ///< Transport block parameters
    LdpcEncoder m_ldpcEncoder;                              ///< Core encoder instance
    cudaStream_t m_cuStream{};                              ///< CUDA stream handle
};


/**
 * @brief Core CSI-RS re-mapper class for CUDA processing
 *
 * This class handles the low-level CSI-RS re-mapping operations on CUDA devices.
 * It manages CUDA resources and provides the interface for running re-mapping
 * operations on CSI-RS parameters.
 */
class CsiRsReMapper final {
public:
    explicit CsiRsReMapper(uint16_t nPrbDlBwp, cudaStream_t cuStream);

    /**
     * @brief Run CSI-RS mapping
     *
     * @param dmrsParams DMRS parameters (device memory)
     * @param csiRsParams CSI-RS parameters (host memory)
     * @param csiRsParamsDev CSI-RS parameters (device memory)
     * @param numTbs Number of transport blocks
     * @param numCsiRsConfigs Number of CSI-RS configurations
     * @param numUeGrps Number of UE groups
     */
    [[nodiscard]] void* run(const cuphy::buffer<PdschDmrsParams, cuphy::device_alloc>& dmrsParams,
                            const cuphy::buffer<cuphyCsirsRrcDynPrm_t, cuphy::pinned_alloc>& csiRsParams,
                            const cuphy::buffer<cuphyCsirsRrcDynPrm_t, cuphy::device_alloc>& csiRsParamsDev,
                            const cuphy::buffer<PdschUeGrpParams, cuphy::device_alloc>& ueGrpPrms,
                            const uint32_t numTbs,
                            const uint32_t numCsiRsConfigs,
                            const uint32_t numUeGrps);

private:
    uint16_t m_nPrbDlBwp{};            ///< Number of PRBs in downlink bandwidth part
    cudaStream_t m_cuStream{};          ///< CUDA stream handle

    cuphy::buffer<uint8_t, cuphy::device_alloc> m_dCsiRsPrepDesc;   ///< CSI-RS preparation descriptor (device)
    cuphy::buffer<uint8_t, cuphy::pinned_alloc> m_hCsiRsPrepDesc;   ///< CSI-RS preparation descriptor (host)
    cuphy::buffer<uint32_t, cuphy::device_alloc> m_dCsiRsOffsets;   ///< Device CSI-RS offsets
    cuphy::buffer<uint32_t, cuphy::pinned_alloc> m_hCsiRsOffsets;   ///< Host CSI-RS offsets
    cuphy::buffer<uint32_t, cuphy::device_alloc> m_dCsiRsCellIndex; ///< Device CSI-RS cell index
    cuphy::buffer<uint32_t, cuphy::pinned_alloc> m_hCsiRsCellIndex; ///< Host CSI-RS cell index

    cuphy::unique_device_ptr<uint16_t> m_reMap;                     ///< RE map array for CSI-RS
};


enum class EnableScrambling : bool {ENABLED, DISABLED};

/**
 * @brief Core LDPC rate matching class for CUDA processing
 *
 * This class handles the low-level LDPC rate matching operations on CUDA devices.
 * It manages CUDA resources and provides the interface for running rate matching
 * operations on LDPC-encoded data.
 */
class LdpcRateMatch final {
public:
    /**
     * @brief Construct a new LDPC rate matcher
     *
     * @param scrambling Enable/disable scrambling
     * @param nPrbDlBwp Number of PRBs in downlink bandwidth part
     * @param maxNumTbs Maximum number of transport blocks
     * @param maxNumCodeBlocks Maximum number of code blocks
     * @param cuStream CUDA stream for asynchronous operations
     */
    explicit LdpcRateMatch(EnableScrambling scrambling,
                           uint16_t nPrbDlBwp,
                           uint32_t maxNumTbs,
                           uint32_t maxNumCodeBlocks,
                           cudaStream_t cuStream);

    /**
     * @brief Perform rate matching on input data
     *
     * @param inputBits Input tensor containing LDPC-encoded data
     * @param tbParams Transport block parameters (host memory)
     * @param tbParamsDev Transport block parameters (device memory)
     * @param numTbs Number of transport blocks to process
     * @return const cuphy::tensor_device& Reference to output tensor
     */
    [[nodiscard]] const cuphy::tensor_device& rateMatch(const cuphy::tensor_device& inputBits,
                                                        const cuphy::buffer<PdschPerTbParams, cuphy::pinned_alloc>& tbParams,
                                                        const cuphy::buffer<PdschPerTbParams, cuphy::device_alloc>& tbParamsDev,
                                                        const cuphy::buffer<PdschDmrsParams, cuphy::pinned_alloc>& dmrsParams,
                                                        const cuphy::buffer<PdschDmrsParams, cuphy::device_alloc>& dmrsParamsDev,
                                                        const cuphy::buffer<cuphyCsirsRrcDynPrm_t, cuphy::pinned_alloc>& csiRsParams,
                                                        const cuphy::buffer<cuphyCsirsRrcDynPrm_t, cuphy::device_alloc>& csiRsParamsDev,
                                                        uint32_t numTbs,
                                                        uint32_t numCsiRsConfigs);

    /**
     * @brief Perform rate matching and modulation layer mapping
     *
     * This function performs rate matching and modulation and layer mapping on the input bits,
     * and writes the output to the output tensor.
     *
     * @param inputBits Input tensor containing LDPC-encoded data
     * @param outputTensor Output tensor for rate-matched data
     * @param tbParams Transport block parameters (host memory)
     * @param tbParamsDev Transport block parameters (device memory)
     * @param dmrsParams DMRS parameters (host memory)
     * @param dmrsParamsDev DMRS parameters (device memory)
     * @param csiRsParams CSI-RS parameters (host memory)
     * @param csiRsParamsDev CSI-RS parameters (device memory)
     * @param numTbs Number of transport blocks to process
     * @param numCsiRsConfigs Number of CSI-RS configurations
     * @param numUeGrps Number of UE groups
     * @param enableModLayerMap Enable/disable modulation layer mapping
     */
    void run(const cuphy::tensor_device& inputBits,
             const cuphy::tensor_device& outputTensor,
             const cuphy::buffer<PdschPerTbParams, cuphy::pinned_alloc>& tbParams,
             const cuphy::buffer<PdschPerTbParams, cuphy::device_alloc>& tbParamsDev,
             const cuphy::buffer<PdschDmrsParams, cuphy::pinned_alloc>& dmrsParams,
             const cuphy::buffer<PdschDmrsParams, cuphy::device_alloc>& dmrsParamsDev,
             const cuphy::buffer<cuphyCsirsRrcDynPrm_t, cuphy::pinned_alloc>& csiRsParams,
             const cuphy::buffer<cuphyCsirsRrcDynPrm_t, cuphy::device_alloc>& csiRsParamsDev,
             uint32_t numTbs,
             uint32_t numCsiRsConfigs,
             uint32_t numUeGrps,
             uint8_t enableModLayerMap);

    /**
     * @brief Get the number of rate-matched bits per code block
     *
     * @return const std::vector<uint32_t>& Vector of bit counts per code block
     */
    [[nodiscard]] const std::vector<uint32_t>& getNumRmBitsPerCb() const { return m_numRmBitsPerCb; }

private:

    cuphy::tensor_device m_dOutputTensor;           ///< Output tensor
    cuphy::unique_device_ptr<uint32_t> m_rmOutput;  ///< Rate matching output buffer
    std::vector<uint32_t> m_numRmBitsPerCb;         ///< Number of rate matched output bits per code block
    EnableScrambling      m_scrambling{};           ///< Enable/disable scrambling
    uint16_t              m_nPrbDlBwp{};            ///< Number of PRBs in downlink bandwidth part

    cuphy::buffer<uint8_t, cuphy::device_alloc> m_dRmDesc;          ///< Rate matching descriptor (device)
    cuphy::buffer<uint8_t, cuphy::pinned_alloc> m_hRmDesc;          ///< Rate matching descriptor (host)
    cuphy::buffer<uint32_t, cuphy::device_alloc> m_dRmWorkspace;    ///< Device rate matching workspace
    cuphy::buffer<uint32_t, cuphy::pinned_alloc> m_hRmWorkspace;    ///< Host rate matching workspace

    CsiRsReMapper m_csiRsReMapper;                                    ///< CSI-RS re-mapper instance
    cuphy::buffer<PdschUeGrpParams, cuphy::pinned_alloc> m_ueGrpPrms;       ///< UE group parameters (host)
    cuphy::buffer<PdschUeGrpParams, cuphy::device_alloc> m_ueGrpPrmsDev;   ///< UE group parameters (device)
    cudaStream_t m_cuStream{};                                          ///< CUDA stream handle

};


/**
 * @brief Python interface for LDPC rate matching
 *
 * This class provides a Python-friendly interface for LDPC rate matching,
 * handling the conversion between Python and C++ data types and managing
 * the underlying CUDA resources.
*/
class __attribute__((visibility("default"))) PyLdpcRateMatch final {
public:
    /**
     * @brief Construct a new Python LDPC rate matcher
     *
     * @param scrambling Enable/disable scrambling
     * @param nPrbDlBwp Number of PRBs in downlink bandwidth part
     * @param maxNumTbs Maximum number of transport blocks
     * @param maxNumCodeBlocks Maximum number of code blocks
     * @param cuStream CUDA stream handle for asynchronous operations
     */
    explicit PyLdpcRateMatch(EnableScrambling scrambling,
                             uint16_t nPrbDlBwp,
                             uint32_t maxNumTbs,
                             uint32_t maxNumCodeBlocks,
                             uint64_t cuStream);

    /**
     * @brief Perform rate matching on input bits
     *
     * @param inputBits Input bits to be rate matched
     * @param tbSizes Transport block sizes per TB
     * @param codeRates Code rates per TB
     * @param rateMatchLens Target lengths for rate matching per TB
     * @param modOrders QAM modulation orders per TB
     * @param numLayers Number of layers per TB
     * @param redundancyVersions Redundancy versions per TB
     * @param cinits Scrambling initialization values per TB
     * @return const cuda_array_t<uint32_t>& Rate-matched output bits
     */
    [[nodiscard]] const cuda_array_t<uint32_t>& rateMatch(const cuda_array_uint32& inputBits,
                                                          const std::vector<uint32_t>& tbSizes,
                                                          const std::vector<float>& codeRates,
                                                          const std::vector<uint32_t>& rateMatchLens,
                                                          const std::vector<uint8_t>& modOrders,
                                                          const std::vector<uint8_t>& numLayers,
                                                          const std::vector<uint8_t>& redundancyVersions,
                                                          const std::vector<uint32_t>& cinits);

    /**
     * @brief Apply rate matching, modulation layer mapping
     *
     * @param inputBits Input bits before rate matching
     * @param txBuffer Transmission buffer for mapped symbols
     * @param pdschConfigs PDSCH configuration parameters
     * @param csiRsConfigs Optional CSI-RS configuration parameters
     */
    void rmModLayerMap(const cuda_array_uint32& inputBits,
                       cuda_array_complex_float& txBuffer,
                       const std::vector<pybind11::object>& pdschConfigs,
                       const std::vector<pybind11::object>& csiRsConfigs = {});

    /**
     * @brief Get the number of rate-matched bits per code block
     *
     * @return const std::vector<uint32_t>& Vector of bit counts per code block
     */
    [[nodiscard]] const std::vector<uint32_t>& getNumRmBitsPerCb() const { return m_ldpcRateMatch.getNumRmBitsPerCb(); }

private:
    /**
     * @brief Convert Python array to CUDA tensor
     *
     * @param inputBits Input bits in Python array format
     * @return cuphy::tensor_device CUDA tensor containing input bits
     */
    [[nodiscard]] cuphy::tensor_device getInputTensor(const cuda_array_uint32& inputBits);

    std::unique_ptr<cuda_array_uint32> m_rmBits;       ///< Rate-matched bits buffer
    cuphy::unique_device_ptr<__half2> m_txBufHalf;     ///< Half-precision transmission buffer

    cuphy::buffer<PdschPerTbParams, cuphy::pinned_alloc> m_tbParams;     ///< Transport block parameters (host)
    cuphy::buffer<PdschPerTbParams, cuphy::device_alloc> m_tbParamsDev;  ///< Transport block parameters (device)
    cuphy::buffer<PdschDmrsParams, cuphy::pinned_alloc> m_dmrsParams;    ///< DMRS parameters (host)
    cuphy::buffer<PdschDmrsParams, cuphy::device_alloc> m_dmrsParamsDev; ///< DMRS parameters (device)
    cuphy::buffer<cuphyCsirsRrcDynPrm_t, cuphy::pinned_alloc> m_csiRsParams;     ///< CSI-RS parameters (host)
    cuphy::buffer<cuphyCsirsRrcDynPrm_t, cuphy::device_alloc> m_csiRsParamsDev;  ///< CSI-RS parameters (device)

    cudaStream_t m_cuStream{};           ///< CUDA stream handle
    LdpcRateMatch m_ldpcRateMatch;       ///< Core rate matching instance
    uint16_t m_nPrbDlBwp{};              ///< Number of PRBs in downlink bandwidth part
};



/**
 * @brief Core LDPC derate matching class for CUDA processing
 *
 * This class handles the low-level LDPC derate matching operations on CUDA devices.
 * It manages CUDA resources and provides the interface for running derate matching
 * operations on received data.
 */
class LdpcDerateMatch final {
public:
    /**
     * @brief Construct a new LDPC derate matcher
     *
     * @param scrambling Enable/disable scrambling
     * @param cuStream CUDA stream for asynchronous operations
     * @param fpConfig Floating point configuration
     */
    explicit LdpcDerateMatch(const bool scrambling, const cudaStream_t cuStream, int fpConfig = 3);
    ~LdpcDerateMatch();

    /**
     * @brief Perform derate matching on input LLRs
     *
     * @param llrs Vector of input LLR tensors
     * @param deRmOutput Output buffer for derate matched data
     * @param puschParams PUSCH parameters
     */
    void derateMatch(const std::vector<cuphy::tensor_ref>& llrs, void** deRmOutput, PuschParams& puschParams);

    /**
     * @brief Perform derate matching on input LLRs with tensor parameters
     *
     * @param inputLlrs Vector of input LLR tensor parameters
     * @param deRmOutput Output buffer for derate matched data
     * @param puschParams PUSCH parameters
     * @param nUes Number of UEs
     */
    void derateMatch(const std::vector<cuphyTensorPrm_t>& inputLlrs,
                     void** deRmOutput,
                     const PerTbParams* pTbPrmsCpu,
                     const PerTbParams* pTbPrmsGpu,
                     int nUes);

private:
    void destroy();  ///< Clean up resources

    size_t m_dynDescrSizeBytes;  ///< Dynamic descriptor size in bytes
    cuphy::buffer<uint8_t, cuphy::pinned_alloc> m_dynDescrBufCpu;  ///< CPU descriptor buffer
    cuphy::buffer<uint8_t, cuphy::device_alloc> m_dynDescrBufGpu;  ///< GPU descriptor buffer
    cuphyPuschRxRateMatchHndl_t m_puschRmHndl{};  ///< PUSCH rate matching handle
    cudaStream_t m_cuStream{};                     ///< CUDA stream handle
};

/**
 * @brief Python interface for LDPC derate matching
 *
 * This class provides a Python-friendly interface for LDPC derate matching,
 * handling the conversion between Python and C++ data types and managing
 * the underlying CUDA resources.
 */
class __attribute__((visibility("default"))) PyLdpcDerateMatch final {
public:
    /**
     * @brief Construct a new Python LDPC derate matcher
     *
     * @param scrambling Enable/disable scrambling
     * @param cuStream CUDA stream for asynchronous operations
     */
    explicit PyLdpcDerateMatch(const bool scrambling, const uint64_t cuStream);
    ~PyLdpcDerateMatch();

    [[nodiscard]] const std::vector<cuda_array_t<__half>>& derateMatch(const std::vector<cuda_array_t<__half>>& inputLlrs,
                                                                      const std::vector<uint32_t>& tbSizes,
                                                                      const std::vector<float>& codeRates,
                                                                      const std::vector<uint32_t>& rateMatchLengths,
                                                                      const std::vector<uint8_t>& qamMods,
                                                                      const std::vector<uint8_t>& numLayers,
                                                                      const std::vector<uint32_t>& rvs,
                                                                      const std::vector<uint32_t>& ndis,
                                                                      const std::vector<uint32_t>& cinits,
                                                                      const std::vector<uint32_t>& userGroupIdxs);


private:
    static constexpr uint32_t LINEAR_ALLOC_PAD_BYTES = 128;  ///< Linear allocation padding in bytes
    cuphy::linear_alloc<LINEAR_ALLOC_PAD_BYTES, cuphy::device_alloc> m_linearAlloc;  ///< Linear allocator
    [[nodiscard]] size_t getBufferSize() const;  ///< Get required buffer size
    LdpcDerateMatch m_derateMatch;      ///< Core derate matching instance
    cudaStream_t m_cuStream{};          ///< CUDA stream handle
    std::vector<cuphy::tensor_device> m_inputLlrTensors;  ///< Input LLR tensors
    void** m_deRmOutput;  ///< Derate matching output buffer
    std::vector<cuda_array_t<__half>> m_pyDeRmOutput;     ///< Python derate matching output
};



/**
 * @brief Core LDPC decoder class for CUDA processing
 *
 * This class handles the low-level LDPC decoding operations on CUDA devices.
 * It manages CUDA resources and provides the interface for running decoding
 * operations on received data.
 */
class LdpcDecoder final {
public:
    /**
     * @brief Construct a new LDPC decoder
     *
     * @param cuStream CUDA stream for asynchronous operations
     */
    explicit LdpcDecoder(const cudaStream_t cuStream);

    /**
     * @brief Decode input LLRs
     *
     * @param deRmOutput Output buffer for derate matched data
     * @param puschParams PUSCH parameters
     * @return void* Decoded output
     */
    [[nodiscard]] void* decode(void** deRmOutput, PuschParams& puschParams);

    /**
     * @brief Decode input LLRs with tensor parameters
     *
     * @param deRmLlr Vector of derate matched LLR tensors
     * @param tbPrmsCpu Transport block parameters (host memory)
     * @param ldpcParams LDPC parameters
     * @return void* Decoded output
     */
    [[nodiscard]] void* decode(const std::vector<void*>& deRmLlr,
                               const std::vector<PerTbParams>& tbPrmsCpu,
                               const cuphyLDPCParams& ldpcParams);

    /**
     * @brief Get the soft outputs
     *
     * @return const std::vector<void*>& Soft outputs
     */
    [[nodiscard]] const std::vector<void*>& getSoftOutputs() const;

private:
    static constexpr uint32_t LINEAR_ALLOC_PAD_BYTES = 128;
    cuphy::linear_alloc<LINEAR_ALLOC_PAD_BYTES, cuphy::device_alloc> m_linearAlloc;
    [[nodiscard]] size_t getBufferSize() const;

    // Output address on device.
    void* m_ldpcOutput;

    // Output address of soft outputs on device.
    std::vector<void*> m_ldpcSoftOutput;

    cuphy::context      m_ctx;
    cuphy::LDPC_decoder m_decoder;
    cudaStream_t        m_cuStream;

    // Normalization factor for min-sum.
    float               m_normalizationFactor;

};

/**
 * @brief Python interface for LDPC decoding
 *
 * This class provides a Python-friendly interface for LDPC decoding,
 * handling the conversion between Python and C++ data types and managing
 * the underlying CUDA resources.
 */
class __attribute__((visibility("default"))) PyLdpcDecoder final {
public:
    /**
     * @brief Construct a new Python LDPC decoder
     *
     * @param cuStream CUDA stream for asynchronous operations
     */
    explicit PyLdpcDecoder(const uint64_t cuStream);

    /**
     * @brief Set the number of decoding iterations
     *
     * @param numIterations Number of iterations
     */
    void setNumIterations(const uint32_t numIterations);

    /**
     * @brief Set the throughput mode
     *
     * @param throughputMode Throughput mode
     */
    void setThroughputMode(const uint8_t throughputMode);

    [[nodiscard]] const std::vector<cuda_array_t<__half>>& decode(
        const std::vector<cuda_array_t<__half>>& inputLlrs,
        const std::vector<uint32_t>& tbSizes,
        const std::vector<float>& codeRates,
        const std::vector<uint32_t>& rvs,
        const std::vector<uint32_t>& rateMatchLengths
    );

    [[nodiscard]] const std::vector<cuda_array_t<float>>& getSoftOutputs();

private:
    static constexpr uint32_t LINEAR_ALLOC_PAD_BYTES = 128;
    cuphy::linear_alloc<LINEAR_ALLOC_PAD_BYTES, cuphy::device_alloc> m_linearAlloc;
    [[nodiscard]] size_t getBufferSize() const;

    LdpcDecoder m_decoder;
    cudaStream_t m_cuStream;

    std::vector<cuphy::tensor_device> m_inputLlrTensors;
    cuphyLDPCParams m_ldpcParams;
    std::vector<PerTbParams> m_tbParams;

    // Outputs to Python.
    std::vector<cuda_array_t<__half>> m_ldpcOutput;
    std::vector<cuda_array_t<float>> m_softOutput;

};

}

#endif // PYCUPHY_LDPC_HPP