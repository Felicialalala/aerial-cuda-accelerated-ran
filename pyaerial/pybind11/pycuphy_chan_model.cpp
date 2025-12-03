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

#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "pycuphy_chan_model.hpp"
#include "cuda_array_interface.hpp"

namespace py = pybind11;

namespace pycuphy {

/*-------------------------------       OFDM modulation class       -------------------------------*/
template <typename Tscalar, typename Tcomplex>
OfdmModulateWrapper<Tscalar, Tcomplex>::OfdmModulateWrapper(cuphyCarrierPrms_t* cuphyCarrierPrms, py::array_t<std::complex<Tscalar>> freqDataInCpu, uintptr_t streamHandle) :
m_cuStrm((cudaStream_t)streamHandle),
m_externGpuAlloc(0)
{
    // buffer size from config
    m_freqDataInSizeDl = (cuphyCarrierPrms -> N_sc) * (cuphyCarrierPrms -> N_bsLayer) * (cuphyCarrierPrms -> N_symbol_slot);
    m_freqDataInSizeUl = (cuphyCarrierPrms -> N_sc) * (cuphyCarrierPrms -> N_ueLayer) * (cuphyCarrierPrms -> N_symbol_slot);
    // get buffer info from the NumPy array
    py::buffer_info buf = freqDataInCpu.request();
    m_freqDataInCpu = static_cast<Tcomplex*>(buf.ptr);

    // allocate GPU buffer
    CUDA_CHECK(cudaMalloc((void**) &(m_freqDataInGpu), sizeof(Tcomplex) * std::max(m_freqDataInSizeDl, m_freqDataInSizeUl)));
    m_ofdmModulateHandle = new ofdm_modulate::ofdmModulate<Tscalar, Tcomplex>(cuphyCarrierPrms, m_freqDataInGpu, m_cuStrm);
}

template <typename Tscalar, typename Tcomplex>
OfdmModulateWrapper<Tscalar, Tcomplex>::OfdmModulateWrapper(cuphyCarrierPrms_t* cuphyCarrierPrms, uintptr_t freqDataInGpu, uintptr_t streamHandle) :
m_cuStrm((cudaStream_t)streamHandle),
m_externGpuAlloc(1)
{
    m_freqDataInGpu = (Tcomplex*)freqDataInGpu;
    m_ofdmModulateHandle = new ofdm_modulate::ofdmModulate<Tscalar, Tcomplex>(cuphyCarrierPrms, m_freqDataInGpu, m_cuStrm);
}

template <typename Tscalar, typename Tcomplex>
OfdmModulateWrapper<Tscalar, Tcomplex>::~OfdmModulateWrapper()
{
    if (!m_externGpuAlloc) // cudaMalloc internally, need to free GPU memory
    {
        cudaFree(m_freqDataInGpu);
    }
    delete m_ofdmModulateHandle;
}

template <typename Tscalar, typename Tcomplex>
void OfdmModulateWrapper<Tscalar, Tcomplex>::run(py::array_t<std::complex<Tscalar>> freqDataInCpu, uint8_t enableSwapTxRx)
{
    uint32_t freqDataInSize = (enableSwapTxRx ? m_freqDataInSizeUl : m_freqDataInSizeDl);
    if(freqDataInCpu.size() != 0) // new input numpy array, need to copy new data to GPU
    {
        // get buffer info from the NumPy array
        py::buffer_info buf = freqDataInCpu.request();
        Tcomplex* freqDataInCpuNew = static_cast<Tcomplex*>(buf.ptr);
        assert(buf.size == freqDataInSize); // check data size match

        cudaMemcpyAsync(m_freqDataInGpu, freqDataInCpuNew, sizeof(Tcomplex) * freqDataInSize, cudaMemcpyHostToDevice, m_cuStrm);
    }
    else
    {
        if (!m_externGpuAlloc) // use numpy array, need to copy new data to GPU
        {
            cudaMemcpyAsync(m_freqDataInGpu, m_freqDataInCpu, sizeof(Tcomplex) * freqDataInSize, cudaMemcpyHostToDevice, m_cuStrm);
        }
    }
    m_ofdmModulateHandle -> run(enableSwapTxRx, m_cuStrm);
}

/*-------------------------------       OFDM demodulation class       -------------------------------*/
template <typename Tscalar, typename Tcomplex>
OfdmDeModulateWrapper<Tscalar, Tcomplex>::OfdmDeModulateWrapper(cuphyCarrierPrms_t * cuphyCarrierPrms, uintptr_t timeDataInGpu, py::array_t<std::complex<Tscalar>> freqDataOutCpu, bool prach, bool perAntSamp, uintptr_t streamHandle) :
m_cuStrm((cudaStream_t)streamHandle),
m_perAntSamp(perAntSamp),
m_externGpuAlloc(0)
{
    // buffer size from config
    m_freqDataOutSizeDl = (cuphyCarrierPrms -> N_sc) * (cuphyCarrierPrms -> N_ueLayer) * (cuphyCarrierPrms -> N_symbol_slot);
    m_freqDataOutSizeUl = (cuphyCarrierPrms -> N_sc) * (cuphyCarrierPrms -> N_bsLayer) * (cuphyCarrierPrms -> N_symbol_slot);
    if (m_perAntSamp)
    {
        m_freqDataOutSizeDl *= cuphyCarrierPrms -> N_bsLayer;
        m_freqDataOutSizeUl *= cuphyCarrierPrms -> N_ueLayer;
    }
    // get buffer info from the NumPy array
    py::buffer_info buf = freqDataOutCpu.request();
    m_freqDataOutCpu = static_cast<Tcomplex*>(buf.ptr);

    // allocate GPU buffer
    CUDA_CHECK(cudaMalloc((void**) &(m_freqDataOutGpu), sizeof(Tcomplex) * std::max(m_freqDataOutSizeDl, m_freqDataOutSizeUl)));
    m_ofdmDeModulateHandle = new ofdm_demodulate::ofdmDeModulate<Tscalar, Tcomplex>(cuphyCarrierPrms, (Tcomplex*)timeDataInGpu, m_freqDataOutGpu, prach, perAntSamp, m_cuStrm);
}

template <typename Tscalar, typename Tcomplex>
OfdmDeModulateWrapper<Tscalar, Tcomplex>::OfdmDeModulateWrapper(cuphyCarrierPrms_t * cuphyCarrierPrms, uintptr_t timeDataInGpu, uintptr_t freqDataOutGpu, bool prach, bool perAntSamp, uintptr_t streamHandle) :
m_cuStrm((cudaStream_t)streamHandle),
m_perAntSamp(perAntSamp),
m_externGpuAlloc(1)
{
    // buffer size from config
    m_freqDataOutSizeDl = (cuphyCarrierPrms -> N_sc) * (cuphyCarrierPrms -> N_ueLayer) * (cuphyCarrierPrms -> N_symbol_slot);
    m_freqDataOutSizeUl = (cuphyCarrierPrms -> N_sc) * (cuphyCarrierPrms -> N_bsLayer) * (cuphyCarrierPrms -> N_symbol_slot);
    if (m_perAntSamp)
    {
        m_freqDataOutSizeDl *= cuphyCarrierPrms -> N_bsLayer;
        m_freqDataOutSizeUl *= cuphyCarrierPrms -> N_ueLayer;
    }
    m_freqDataOutCpu = nullptr;
    m_freqDataOutGpu = (Tcomplex*) freqDataOutGpu;
    m_ofdmDeModulateHandle = new ofdm_demodulate::ofdmDeModulate<Tscalar, Tcomplex>(cuphyCarrierPrms, (Tcomplex*)timeDataInGpu, m_freqDataOutGpu, prach, perAntSamp, m_cuStrm);
}

template <typename Tscalar, typename Tcomplex>
OfdmDeModulateWrapper<Tscalar, Tcomplex>::~OfdmDeModulateWrapper()
{
    if (!m_externGpuAlloc) // cudaMalloc internally, need to free GPU memory
    {
        cudaFree(m_freqDataOutGpu);
    }
    delete m_ofdmDeModulateHandle;
}

template <typename Tscalar, typename Tcomplex>
void OfdmDeModulateWrapper<Tscalar, Tcomplex>::run(py::array_t<std::complex<Tscalar>> freqDataOutCpu, uint8_t enableSwapTxRx)
{
    m_ofdmDeModulateHandle -> run(enableSwapTxRx, m_cuStrm);
    uint32_t freqDataOutSize = (enableSwapTxRx ? m_freqDataOutSizeUl : m_freqDataOutSizeDl);
    if(freqDataOutCpu.size() != 0) // new output numpy array, need to copy new data from GPU
    {
        py::buffer_info buf = freqDataOutCpu.request();
        Tcomplex* freqDataOutCpuNew = static_cast<Tcomplex*>(buf.ptr);
        assert(buf.size == freqDataOutSize); // check data size match

        cudaMemcpyAsync(freqDataOutCpuNew, m_freqDataOutGpu, sizeof(Tcomplex) * freqDataOutSize, cudaMemcpyDeviceToHost, m_cuStrm);
    }
    else if (!m_externGpuAlloc)
    {
        cudaMemcpyAsync(m_freqDataOutCpu, m_freqDataOutGpu, sizeof(Tcomplex) * freqDataOutSize, cudaMemcpyDeviceToHost, m_cuStrm);
    }
    cudaStreamSynchronize(m_cuStrm);
}

/*-------------------------------       TDL channel class       -------------------------------*/
template <typename Tscalar, typename Tcomplex>
TdlChanWrapper<Tscalar, Tcomplex>::TdlChanWrapper(tdlConfig_t* tdlCfg, py::array_t<std::complex<Tscalar>> txSigInCpu, uint16_t randSeed, uintptr_t streamHandle) :
m_cuStrm((cudaStream_t)streamHandle),
m_runMode(tdlCfg -> runMode),
m_nLink((tdlCfg -> nCell) * (tdlCfg -> nUe)),
m_tdlCfg(tdlCfg),
m_externGpuAlloc(0)
{
    // buffer size from config
    if (m_tdlCfg -> procSigFreq == 1) // proc tx signal in freq domain
    {
        m_txSigSizeDl = m_nLink * (m_tdlCfg -> N_sc) * (m_tdlCfg -> batchLen.size()) * (m_tdlCfg -> nBsAnt);
        m_txSigSizeUl = m_nLink * (m_tdlCfg -> N_sc) * (m_tdlCfg -> batchLen.size()) * (m_tdlCfg -> nUeAnt);
    }
    else
    {
        m_txSigSizeDl = m_nLink * (tdlCfg -> sigLenPerAnt) * (tdlCfg -> nBsAnt);
        m_txSigSizeUl = m_nLink * (tdlCfg -> sigLenPerAnt) * (tdlCfg -> nUeAnt);
    }
    m_rxSigSizeDl = m_txSigSizeUl;
    m_rxSigSizeUl = m_txSigSizeDl;
    // get buffer info from the NumPy array
    py::buffer_info buf = txSigInCpu.request();
    m_txSigInCpu = static_cast<Tcomplex*>(buf.ptr);

    // allocate GPU buffer and copy data
    CUDA_CHECK(cudaMalloc((void**) &(m_txSigInGpu), sizeof(Tcomplex) * std::max(m_txSigSizeUl, m_txSigSizeDl)));
    tdlCfg -> txSigIn = m_txSigInGpu;
    m_tdlChanHandle = new tdlChan<Tscalar, Tcomplex>(tdlCfg, randSeed, m_cuStrm);
}

template <typename Tscalar, typename Tcomplex>
TdlChanWrapper<Tscalar, Tcomplex>::TdlChanWrapper(tdlConfig_t* tdlCfg, uintptr_t txSigInGpu, uint16_t randSeed, uintptr_t streamHandle) :
m_cuStrm((cudaStream_t)streamHandle),
m_runMode(tdlCfg -> runMode),
m_nLink((tdlCfg -> nCell) * (tdlCfg -> nUe)),
m_tdlCfg(tdlCfg),
m_externGpuAlloc(1)
{
    m_txSigInGpu = (cuComplex*) txSigInGpu;
    tdlCfg -> txSigIn = m_txSigInGpu;
    m_tdlChanHandle = new tdlChan<Tscalar, Tcomplex>(tdlCfg, randSeed, m_cuStrm);
}

template <typename Tscalar, typename Tcomplex>
TdlChanWrapper<Tscalar, Tcomplex>::~TdlChanWrapper()
{
    if (!m_externGpuAlloc) // cudaMalloc internally, need to free GPU memory
    {
        cudaFree(m_txSigInGpu);
    }
    delete m_tdlChanHandle;
}

template <typename Tscalar, typename Tcomplex>
void TdlChanWrapper<Tscalar, Tcomplex>::run(float refTime0, uint8_t enableSwapTxRx, uint8_t txColumnMajorInd)
{
    if (!m_externGpuAlloc) // cudaMalloc internally, need to copy GPU memory
    {
        cudaMemcpyAsync(m_txSigInGpu, m_txSigInCpu, sizeof(Tcomplex) * (enableSwapTxRx ? m_txSigSizeUl : m_txSigSizeDl), cudaMemcpyHostToDevice, m_cuStrm);
    }
    m_tdlChanHandle -> run(refTime0, enableSwapTxRx, txColumnMajorInd);
}

template <typename Tscalar, typename Tcomplex>
void TdlChanWrapper<Tscalar, Tcomplex>::run(py::array_t<std::complex<Tscalar>> txFreqSigInCpu, py::array_t<std::complex<Tscalar>> rxFreqSigOutCpu, float refTime0, uint8_t enableSwapTxRx, uint8_t txColumnMajorInd)
{
    if (!m_externGpuAlloc) // cudaMalloc internally, need to free GPU memory
    {
        py::buffer_info buf_in = txFreqSigInCpu.request();
        m_txSigInCpu = static_cast<Tcomplex*>(buf_in.ptr);

        cudaMemcpyAsync(m_txSigInGpu, m_txSigInCpu, sizeof(Tcomplex) * (enableSwapTxRx ? m_txSigSizeUl : m_txSigSizeDl), cudaMemcpyHostToDevice, m_cuStrm);
        m_tdlChanHandle -> run(refTime0, enableSwapTxRx, txColumnMajorInd);
        py::buffer_info buf_out = rxFreqSigOutCpu.request();
        cudaMemcpyAsync(static_cast<void*>(buf_out.ptr), m_tdlChanHandle -> getRxSigOut(), sizeof(Tcomplex) * (enableSwapTxRx ? m_rxSigSizeUl : m_rxSigSizeDl), cudaMemcpyDeviceToHost, m_cuStrm);
        cudaStreamSynchronize(m_cuStrm);
    }
    else
    {
        fprintf(stderr, "Input tx sample must be copied to external GPU memory!\n");
        exit(EXIT_FAILURE);
    }
}

template <typename Tscalar, typename Tcomplex> 
void TdlChanWrapper<Tscalar, Tcomplex>::dumpCir(py::array_t<std::complex<Tscalar>> cirCpu)
{
    // buffer size from config
    uint32_t timeChanSize = m_tdlChanHandle -> getTimeChanSize();
    
    // get buffer info from the NumPy array
    py::buffer_info buf = cirCpu.request();
    assert(buf.size == timeChanSize); // check data size match

    // copy CIR
    cudaMemcpyAsync(buf.ptr, m_tdlChanHandle -> getTimeChan(), sizeof(Tcomplex) * timeChanSize, cudaMemcpyDeviceToHost, m_cuStrm);
}

template <typename Tscalar, typename Tcomplex> 
void TdlChanWrapper<Tscalar, Tcomplex>::dumpCfrPrbg(py::array_t<std::complex<Tscalar>> cfrPrbg)
{
    // dump CFR on PRBG
    if(m_runMode > 0 && m_runMode < 3)
    {
        // buffer size from config
        uint32_t freqChanPrbgSize = m_tdlChanHandle -> getFreqChanPrbgSize();
        
        // get buffer info from the NumPy array
        py::buffer_info buf = cfrPrbg.request();
        assert(buf.size == freqChanPrbgSize); // check data size match

        // copy CFR on PRBG
        cudaMemcpyAsync(buf.ptr, m_tdlChanHandle -> getFreqChanPrbg(), sizeof(Tcomplex) * freqChanPrbgSize, cudaMemcpyDeviceToHost, m_cuStrm);
    }
}

template <typename Tscalar, typename Tcomplex> 
void TdlChanWrapper<Tscalar, Tcomplex>::dumpCfrSc(py::array_t<std::complex<Tscalar>> cfrSc)
{
    // dump CFR on SC
    if(m_runMode > 1 && m_runMode < 3)
    {
        // buffer size from config
        uint32_t freqChanScSizePerLink = m_tdlChanHandle -> getFreqChanScPerLinkSize();
        Tcomplex ** freqChanSc = m_tdlChanHandle -> getFreqChanScHostArray(); // CFR on SC is saved using pointer of pointers
        
        // get buffer info from the NumPy array
        py::buffer_info buf = cfrSc.request();

        // copy CFR on SC
        Tcomplex * freqChScCpuOut = static_cast<Tcomplex*>(buf.ptr);
        for (uint16_t linkIdx = 0; linkIdx < m_nLink; linkIdx ++)
        {
            cudaMemcpyAsync(freqChScCpuOut, freqChanSc[linkIdx], sizeof(Tcomplex) * freqChanScSizePerLink, cudaMemcpyDeviceToHost, m_cuStrm);
            freqChScCpuOut += freqChanScSizePerLink; // CPU address for CFR on SC of next link
        }
    }
}

/*-------------------------------       CDL channel class       -------------------------------*/
template <typename Tscalar, typename Tcomplex>
CdlChanWrapper<Tscalar, Tcomplex>::CdlChanWrapper(cdlConfig_t* cdlCfg, py::array_t<std::complex<Tscalar>> txSigInCpu, uint16_t randSeed, uintptr_t streamHandle) :
m_cuStrm((cudaStream_t)streamHandle),
m_runMode(cdlCfg -> runMode),
m_nLink((cdlCfg -> nCell) * (cdlCfg -> nUe)),
m_cdlCfg(cdlCfg),
m_externGpuAlloc(0)
{
    m_nBsAnt = std::accumulate(m_cdlCfg -> bsAntSize.begin(), m_cdlCfg -> bsAntSize.end(), 1U, std::multiplies<uint32_t>());
    m_nUeAnt = std::accumulate(m_cdlCfg -> ueAntSize.begin(), m_cdlCfg -> ueAntSize.end(), 1U, std::multiplies<uint32_t>());
    // buffer size from config
    if (m_cdlCfg -> procSigFreq == 1) // proc tx signal in freq domain
    {
        m_txSigSizeDl = m_nLink * (m_cdlCfg -> N_sc) * (m_cdlCfg -> batchLen.size()) * m_nBsAnt;
        m_txSigSizeUl = m_nLink * (m_cdlCfg -> N_sc) * (m_cdlCfg -> batchLen.size()) * m_nUeAnt;
    }
    else
    {
        m_txSigSizeDl = m_nLink * (cdlCfg -> sigLenPerAnt) * m_nBsAnt;
        m_txSigSizeUl = m_nLink * (cdlCfg -> sigLenPerAnt) * m_nUeAnt;
    }
    m_rxSigSizeDl = m_txSigSizeUl;
    m_rxSigSizeUl = m_txSigSizeDl;
    // get buffer info from the NumPy array
    py::buffer_info buf = txSigInCpu.request();
    m_txSigInCpu = static_cast<Tcomplex*>(buf.ptr);

    // allocate GPU buffer and copy data
    CUDA_CHECK(cudaMalloc((void**) &(m_txSigInGpu), sizeof(Tcomplex) * std::max(m_txSigSizeUl, m_txSigSizeDl)));
    cdlCfg -> txSigIn = m_txSigInGpu;
    m_cdlChanHandle = new cdlChan<Tscalar, Tcomplex>(cdlCfg, randSeed, m_cuStrm);
}

template <typename Tscalar, typename Tcomplex>
CdlChanWrapper<Tscalar, Tcomplex>::CdlChanWrapper(cdlConfig_t* cdlCfg, uintptr_t txSigInGpu, uint16_t randSeed, uintptr_t streamHandle) :
m_cuStrm((cudaStream_t)streamHandle),
m_runMode(cdlCfg -> runMode),
m_nLink((cdlCfg -> nCell) * (cdlCfg -> nUe)),
m_cdlCfg(cdlCfg),
m_externGpuAlloc(1)
{
    m_txSigInGpu = (cuComplex*) txSigInGpu;
    cdlCfg -> txSigIn = m_txSigInGpu;
    m_cdlChanHandle = new cdlChan<Tscalar, Tcomplex>(cdlCfg, randSeed, m_cuStrm);
}

template <typename Tscalar, typename Tcomplex>
CdlChanWrapper<Tscalar, Tcomplex>::~CdlChanWrapper()
{
    if (!m_externGpuAlloc) // cudaMalloc internally, need to free GPU memory
    {
        cudaFree(m_txSigInGpu);
    }
    delete m_cdlChanHandle;
}

template <typename Tscalar, typename Tcomplex>
void CdlChanWrapper<Tscalar, Tcomplex>::run(float refTime0, uint8_t enableSwapTxRx, uint8_t txColumnMajorInd)
{
    if (!m_externGpuAlloc) // cudaMalloc internally, need to copy GPU memory
    {
        cudaMemcpyAsync(m_txSigInGpu, m_txSigInCpu, sizeof(Tcomplex) * (enableSwapTxRx ? m_txSigSizeUl : m_txSigSizeDl), cudaMemcpyHostToDevice, m_cuStrm);
    }
    m_cdlChanHandle -> run(refTime0, enableSwapTxRx, txColumnMajorInd);
}

template <typename Tscalar, typename Tcomplex>
void CdlChanWrapper<Tscalar, Tcomplex>::run(py::array_t<std::complex<Tscalar>> txFreqSigInCpu, py::array_t<std::complex<Tscalar>> rxFreqSigOutCpu, float refTime0, uint8_t enableSwapTxRx, uint8_t txColumnMajorInd)
{
    if (!m_externGpuAlloc) // cudaMalloc internally, need to free GPU memory
    {
        py::buffer_info buf_in = txFreqSigInCpu.request();
        m_txSigInCpu = static_cast<Tcomplex*>(buf_in.ptr);

        cudaMemcpyAsync(m_txSigInGpu, m_txSigInCpu, sizeof(Tcomplex) * (enableSwapTxRx ? m_txSigSizeUl : m_txSigSizeDl), cudaMemcpyHostToDevice, m_cuStrm);
        m_cdlChanHandle -> run(refTime0, enableSwapTxRx, txColumnMajorInd);
        py::buffer_info buf_out = rxFreqSigOutCpu.request();
        cudaMemcpyAsync(static_cast<void*>(buf_out.ptr), m_cdlChanHandle -> getRxSigOut(), sizeof(Tcomplex) * (enableSwapTxRx ? m_rxSigSizeUl : m_rxSigSizeDl), cudaMemcpyDeviceToHost, m_cuStrm);
        cudaStreamSynchronize(m_cuStrm);
    }
    else
    {
        fprintf(stderr, "Input tx sample must be copied to external GPU memory!\n");
        exit(EXIT_FAILURE);
    }
}

template <typename Tscalar, typename Tcomplex> 
void CdlChanWrapper<Tscalar, Tcomplex>::dumpCir(py::array_t<std::complex<Tscalar>> cirCpu)
{
    // buffer size from config
    uint32_t timeChanSize = m_cdlChanHandle -> getTimeChanSize();
    
    // get buffer info from the NumPy array
    py::buffer_info buf = cirCpu.request();
    assert(buf.size == timeChanSize); // check data size match

    // copy CIR
    cudaMemcpyAsync(buf.ptr, m_cdlChanHandle -> getTimeChan(), sizeof(Tcomplex) * timeChanSize, cudaMemcpyDeviceToHost, m_cuStrm);
}

template <typename Tscalar, typename Tcomplex> 
void CdlChanWrapper<Tscalar, Tcomplex>::dumpCfrPrbg(py::array_t<std::complex<Tscalar>> cfrPrbg)
{
    // dump CFR on PRBG
    if(m_runMode > 0 && m_runMode < 3)
    {
        // buffer size from config
        uint32_t freqChanPrbgSize = m_cdlChanHandle -> getFreqChanPrbgSize();
        
        // get buffer info from the NumPy array
        py::buffer_info buf = cfrPrbg.request();
        assert(buf.size == freqChanPrbgSize); // check data size match

        // copy CFR on PRBG
        cudaMemcpyAsync(buf.ptr, m_cdlChanHandle -> getFreqChanPrbg(), sizeof(Tcomplex) * freqChanPrbgSize, cudaMemcpyDeviceToHost, m_cuStrm);
    }
}

template <typename Tscalar, typename Tcomplex> 
void CdlChanWrapper<Tscalar, Tcomplex>::dumpCfrSc(py::array_t<std::complex<Tscalar>> cfrSc)
{
    // dump CFR on SC
    if(m_runMode > 1 && m_runMode < 3)
    {
        // buffer size from config
        uint32_t freqChanScSizePerLink = m_cdlChanHandle -> getFreqChanScPerLinkSize();
        Tcomplex ** freqChanSc = m_cdlChanHandle -> getFreqChanScHostArray(); // CFR on SC is saved using pointer of pointers
        
        // get buffer info from the NumPy array
        py::buffer_info buf = cfrSc.request();

        // copy CFR on SC
        Tcomplex * freqChScCpuOut = static_cast<Tcomplex*>(buf.ptr);
        for (uint16_t linkIdx = 0; linkIdx < m_nLink; linkIdx ++)
        {
            cudaMemcpyAsync(freqChScCpuOut, freqChanSc[linkIdx], sizeof(Tcomplex) * freqChanScSizePerLink, cudaMemcpyDeviceToHost, m_cuStrm);
            freqChScCpuOut += freqChanScSizePerLink; // CPU address for CFR on SC of next link
        }
    }
}

/*-------------------------------       add Gaussian noise class       -------------------------------*/
template <typename Tscalar, typename Tcomplex>
GauNoiseAdderWrapper<Tscalar, Tcomplex>::GauNoiseAdderWrapper(uint32_t nThreads, int seed, uintptr_t streamHandle) :
m_cuStrm((cudaStream_t)streamHandle)
{
    m_gauNoiseAdder = new GauNoiseAdder<Tcomplex>(nThreads, seed, m_cuStrm);
}

template <typename Tscalar, typename Tcomplex>
GauNoiseAdderWrapper<Tscalar, Tcomplex>::~GauNoiseAdderWrapper()
{
    delete m_gauNoiseAdder;
}

template <typename Tscalar, typename Tcomplex>
void GauNoiseAdderWrapper<Tscalar, Tcomplex>::addNoise(py::array_t<std::complex<Tscalar>> noisySignal, uintptr_t d_signal, uint32_t signalSize, float snr_db)
{
    m_gauNoiseAdder -> addNoise((Tcomplex*)d_signal, signalSize, snr_db);
    // get buffer info from the NumPy array
    py::buffer_info buf = noisySignal.request();
    assert(noisySignal.size() == signalSize); // check data size match

    // copy noisy signal
    cudaMemcpyAsync(buf.ptr, (Tcomplex*)d_signal, sizeof(Tcomplex) * signalSize, cudaMemcpyDeviceToHost, m_cuStrm);
    cudaStreamSynchronize(m_cuStrm);
}

/*-------------------------------       Stochastic Channel Model class       -------------------------------*/
template <typename Tscalar, typename Tcomplex>
StatisChanModelWrapper<Tscalar, Tcomplex>::StatisChanModelWrapper(
    const SimConfig& sim_config,
    const SystemLevelConfig& system_level_config,
    const LinkLevelConfig& link_level_config,
    const ExternalConfig& external_config,
    uint32_t randSeed,
    uintptr_t streamHandle) :
m_cuStrm((cudaStream_t)streamHandle),
m_randSeed(randSeed)
{
    m_statisChanModelHandle = new statisChanModel<Tscalar, Tcomplex>(
        &sim_config, &system_level_config, &link_level_config, &external_config,
        m_randSeed, m_cuStrm);
    m_cpuOnlyMode = sim_config.cpu_only_mode;
}

// Constructor with just sim_config and system_level_config
template <typename Tscalar, typename Tcomplex>
StatisChanModelWrapper<Tscalar, Tcomplex>::StatisChanModelWrapper(
    const SimConfig& sim_config,
    const SystemLevelConfig& system_level_config,
    uint32_t randSeed,
    uintptr_t streamHandle) :
m_cuStrm((cudaStream_t)streamHandle),
m_randSeed(randSeed)
{
    m_statisChanModelHandle = new statisChanModel<Tscalar, Tcomplex>(
        &sim_config, &system_level_config, nullptr, nullptr,
        m_randSeed, m_cuStrm);
    m_cpuOnlyMode = sim_config.cpu_only_mode;
}

template <typename Tscalar, typename Tcomplex>
StatisChanModelWrapper<Tscalar, Tcomplex>::~StatisChanModelWrapper()
{
    delete m_statisChanModelHandle;
}

template <typename Tscalar, typename Tcomplex>
void StatisChanModelWrapper<Tscalar, Tcomplex>::run(
    float refTime,
    uint8_t continuous_fading,
    py::object activeCell,
    py::object activeUt,
    py::object utNewLoc,
    py::object utNewVelocity,
    py::object cir_coe,
    py::object cir_norm_delay,
    py::object cir_n_taps,
    py::object cfr_sc,
    py::object cfr_prbg) {
    
    // Convert activeCell to vector
    std::vector<uint16_t> activeCellVec;
    if (!activeCell.is_none()) {
        if (py::isinstance<py::list>(activeCell)) {
            auto cellList = activeCell.cast<py::list>();
            for (auto item : cellList) {
                activeCellVec.push_back(item.cast<uint16_t>());
            }
        }
    }
    
    // Convert activeUt to nested vector
    std::vector<std::vector<uint16_t>> activeUtVec;
    if (!activeUt.is_none()) {
        if (py::isinstance<py::list>(activeUt)) {
            auto utList = activeUt.cast<py::list>();
            for (auto item : utList) {
                if (py::isinstance<py::list>(item)) {
                    std::vector<uint16_t> utVec;
                    auto innerList = item.cast<py::list>();
                    for (auto ut : innerList) {
                        utVec.push_back(ut.cast<uint16_t>());
                    }
                    activeUtVec.push_back(utVec);
                }
            }
        }
    }
    
    // Convert utNewLoc to vector of Coordinates
    std::vector<Coordinate> utNewLocVec;
    if (!utNewLoc.is_none()) {
        if (py::isinstance<py::array>(utNewLoc)) {
            auto arr = utNewLoc.cast<py::array_t<float>>();
            auto buf = arr.unchecked<2>();
            for (py::ssize_t i = 0; i < buf.shape(0); i++) {
                Coordinate coord;
                coord.x = buf(i, 0);
                coord.y = buf(i, 1);
                coord.z = buf(i, 2);
                utNewLocVec.push_back(coord);
            }
        }
    }
    
    // Convert utNewVelocity to vector of float3
    std::vector<float3> utNewVelocityVec;
    if (!utNewVelocity.is_none()) {
        if (py::isinstance<py::array>(utNewVelocity)) {
            auto arr = utNewVelocity.cast<py::array_t<float>>();
            auto buf = arr.unchecked<2>();
            for (py::ssize_t i = 0; i < buf.shape(0); i++) {
                float3 vel;
                vel.x = buf(i, 0);
                vel.y = buf(i, 1);
                vel.z = buf(i, 2);
                utNewVelocityVec.push_back(vel);
            }
        }
    }
    
    // Convert per-cell array parameters to vectors of pointers
    std::vector<Tcomplex*> cir_coe_ptrs;
    std::vector<uint16_t*> cir_norm_delay_ptrs;
    std::vector<uint16_t*> cir_n_taps_ptrs;
    std::vector<Tcomplex*> cfr_sc_ptrs;
    std::vector<Tcomplex*> cfr_prbg_ptrs;
    
    // Helper lambda to extract device pointers from list of CuPy arrays
    auto extract_cuda_array_ptrs = [](py::object obj, auto& ptr_vec) {
        if (!obj.is_none() && py::isinstance<py::list>(obj)) {
            auto array_list = obj.cast<py::list>();
            for (auto item : array_list) {
                uintptr_t device_ptr = 0;
                
                // Handle CuPy arrays directly using __cuda_array_interface__
                if (py::hasattr(item, "__cuda_array_interface__")) {
                    auto array_interface = item.attr("__cuda_array_interface__");
                    if (py::isinstance<py::dict>(array_interface)) {
                        auto interface_dict = array_interface.cast<py::dict>();
                        if (interface_dict.contains("data")) {
                            auto data_info = interface_dict["data"];
                            if (py::isinstance<py::tuple>(data_info)) {
                                auto data_tuple = data_info.cast<py::tuple>();
                                if (data_tuple.size() > 0) {
                                    device_ptr = data_tuple[0].cast<uintptr_t>();
                                }
                            }
                        }
                    }
                }
                // Also try CPU arrays with __array_interface__ as fallback
                else if (py::hasattr(item, "__array_interface__")) {
                    auto array_interface = item.attr("__array_interface__");
                    if (py::isinstance<py::dict>(array_interface)) {
                        auto interface_dict = array_interface.cast<py::dict>();
                        if (interface_dict.contains("data")) {
                            auto data_info = interface_dict["data"];
                            if (py::isinstance<py::tuple>(data_info)) {
                                auto data_tuple = data_info.cast<py::tuple>();
                                if (data_tuple.size() > 0) {
                                    device_ptr = data_tuple[0].cast<uintptr_t>();
                                }
                            }
                        }
                    }
                }
                // Fallback: try pycuphy wrapper with get_device_ptr()
                else if (py::hasattr(item, "get_device_ptr")) {
                    auto ptr_value = item.attr("get_device_ptr")();
                    if (py::isinstance<py::int_>(ptr_value)) {
                        device_ptr = ptr_value.cast<uintptr_t>();
                    }
                }
                // Fallback: try PyTorch-style data_ptr()
                else if (py::hasattr(item, "data_ptr")) {
                    device_ptr = item.attr("data_ptr")().cast<uintptr_t>();
                }
                
                ptr_vec.push_back(reinterpret_cast<typename std::remove_reference_t<decltype(ptr_vec)>::value_type>(device_ptr));
            }
        }
    };
    
    // Extract pointers from Python objects:
    // - In GPU mode, prefer __cuda_array_interface__ and device pointers
    // - In CPU-only mode, fall back to __array_interface__ host pointers
    extract_cuda_array_ptrs(cir_coe, cir_coe_ptrs);
    extract_cuda_array_ptrs(cir_norm_delay, cir_norm_delay_ptrs);
    extract_cuda_array_ptrs(cir_n_taps, cir_n_taps_ptrs);
    extract_cuda_array_ptrs(cfr_sc, cfr_sc_ptrs);
    extract_cuda_array_ptrs(cfr_prbg, cfr_prbg_ptrs);

    if (m_cpuOnlyMode == 0) {
        // Debug GPU pointers only in GPU mode
        printf("DEBUG: Extracted GPU pointers from pybind11:\n");
        printf("  cir_coe_ptrs: %zu pointers\n", cir_coe_ptrs.size());
        for (size_t i = 0; i < cir_coe_ptrs.size(); ++i) {
            printf("    Cell %zu: cir_coe=0x%lx\n", i, reinterpret_cast<uintptr_t>(cir_coe_ptrs[i]));
        }
        printf("  cir_norm_delay_ptrs: %zu pointers\n", cir_norm_delay_ptrs.size());
        for (size_t i = 0; i < cir_norm_delay_ptrs.size(); ++i) {
            printf("    Cell %zu: cir_norm_delay=0x%lx\n", i, reinterpret_cast<uintptr_t>(cir_norm_delay_ptrs[i]));
        }
        printf("  cir_n_taps_ptrs: %zu pointers\n", cir_n_taps_ptrs.size());
        for (size_t i = 0; i < cir_n_taps_ptrs.size(); ++i) {
            printf("    Cell %zu: cir_n_taps=0x%lx\n", i, reinterpret_cast<uintptr_t>(cir_n_taps_ptrs[i]));
        }
        printf("  cfr_sc_ptrs: %zu pointers\n", cfr_sc_ptrs.size());
        for (size_t i = 0; i < cfr_sc_ptrs.size(); ++i) {
            printf("    Cell %zu: cfr_sc=0x%lx\n", i, reinterpret_cast<uintptr_t>(cfr_sc_ptrs[i]));
        }
        printf("  cfr_prbg_ptrs: %zu pointers\n", cfr_prbg_ptrs.size());
        for (size_t i = 0; i < cfr_prbg_ptrs.size(); ++i) {
            printf("    Cell %zu: cfr_prbg=0x%lx\n", i, reinterpret_cast<uintptr_t>(cfr_prbg_ptrs[i]));
        }
    }

    // Call the C++ method with vectors of pointers
    m_statisChanModelHandle->run(refTime, continuous_fading, activeCellVec, activeUtVec, 
                               utNewLocVec, utNewVelocityVec, cir_coe_ptrs, cir_norm_delay_ptrs, 
                               cir_n_taps_ptrs, cfr_sc_ptrs, cfr_prbg_ptrs);
    // sync stream in GPU mode
    if (m_cpuOnlyMode == 0) {
        cudaStreamSynchronize(m_cuStrm);
    }
}

template <typename Tscalar, typename Tcomplex>
void StatisChanModelWrapper<Tscalar, Tcomplex>::run_link_level(
    float refTime0,
    uint8_t continuous_fading,
    uint8_t enableSwapTxRx,
    uint8_t txColumnMajorInd) {
    
    // Create empty vectors and empty pointer vectors for unused parameters
    std::vector<uint16_t> empty_cells;
    std::vector<std::vector<uint16_t>> empty_uts;
    std::vector<Coordinate> empty_locs;
    std::vector<float3> empty_velocities;
    std::vector<Tcomplex*> empty_cir_coe;
    std::vector<uint16_t*> empty_cir_norm_delay;
    std::vector<uint16_t*> empty_cir_n_taps;
    std::vector<Tcomplex*> empty_cfr_sc;
    std::vector<Tcomplex*> empty_cfr_prbg;
    
    // Call the run method with all parameters
    m_statisChanModelHandle->run(refTime0, continuous_fading, empty_cells, empty_uts,
                               empty_locs, empty_velocities, empty_cir_coe, empty_cir_norm_delay,
                               empty_cir_n_taps, empty_cfr_sc, empty_cfr_prbg);
}

template <typename Tscalar, typename Tcomplex>
void StatisChanModelWrapper<Tscalar, Tcomplex>::dump_los_nlos_stats(py::array_t<float> lost_nlos_stats) {
    float* stats_ptr = nullptr;
    if (lost_nlos_stats.size() > 0) {
        stats_ptr = lost_nlos_stats.mutable_data();
    }
    m_statisChanModelHandle->dump_los_nlos_stats(stats_ptr);
}

template <typename Tscalar, typename Tcomplex>
void StatisChanModelWrapper<Tscalar, Tcomplex>::dump_pathloss_shadowing_stats(
    py::array_t<float> pathloss_shadowing,
    py::array_t<int> activeCell,
    py::array_t<int> activeUt) {
    
    // pathloss_shadowing is required
    if (pathloss_shadowing.size() == 0) {
        throw std::invalid_argument("pathloss_shadowing array cannot be empty");
    }
    
    float* pathloss_shadowing_ptr = pathloss_shadowing.mutable_data();
    
    // Convert numpy arrays to vectors
    std::vector<uint16_t> activeCellVec;
    std::vector<uint16_t> activeUtVec;
    
    if (activeCell.size() > 0) {
        activeCellVec.reserve(activeCell.size());
        const int* cell_data = activeCell.data();
        for (size_t i = 0; i < activeCell.size(); ++i) {
            activeCellVec.push_back(static_cast<uint16_t>(cell_data[i]));
        }
    }
    
    if (activeUt.size() > 0) {
        activeUtVec.reserve(activeUt.size());
        const int* ut_data = activeUt.data();
        for (size_t i = 0; i < activeUt.size(); ++i) {
            activeUtVec.push_back(static_cast<uint16_t>(ut_data[i]));
        }
    }
    // Call the C++ method with vectors
    m_statisChanModelHandle->dump_pathloss_shadowing_stats(pathloss_shadowing_ptr, activeCellVec, activeUtVec);
}

template <typename Tscalar, typename Tcomplex>
void StatisChanModelWrapper<Tscalar, Tcomplex>::dump_topology_to_yaml(const std::string& filename) {
    m_statisChanModelHandle->dump_topology_to_yaml(filename);
}

template <typename Tscalar, typename Tcomplex>
void StatisChanModelWrapper<Tscalar, Tcomplex>::saveSlsChanToH5File(std::string_view filename_ending) {
    m_statisChanModelHandle->saveSlsChanToH5File(filename_ending);
}

}