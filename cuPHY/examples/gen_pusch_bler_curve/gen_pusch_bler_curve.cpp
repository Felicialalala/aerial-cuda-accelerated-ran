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

#include <cuda_runtime.h>
#include "cuda.h"
#include "cuda_fp16.h"
#include "cuphy.h"
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include "hdf5hpp.hpp"
#include "cuphy_hdf5.hpp"
#include "cuphy.hpp"
#include "datasets.hpp"
#include "srs_rx.hpp"
#include "cuphy_channels.hpp"
#include "pusch_rx.hpp"
#include "gen_pusch_bler_curve.hpp"


#include <fstream>
#include <cstring>
#include <iostream>
#include <unistd.h> // for getcwd()
#include <dirent.h> // opendir, readdir
#include <errno.h>
#include <sys/stat.h> // for mkdir
#include "fading_chan.cuh"
using namespace std;



void gen_pusch_bler_curve(std::string puschInputFilename, std::string puschDbgFilename, std::string tberFilename, std::vector<float> snrVec, uint32_t nItrPerSnr, bool quickBlerFlag, uint32_t quickBlerNumTbErrs, uint8_t seed)
{
    // ---------------------------------------------------------------
    // Initialize main stream

    cuphy::stream cuStrmMain;
    
    // Input file
    hdf5hpp::hdf5_file inputFile            = hdf5hpp::hdf5_file::open(puschInputFilename.c_str());
    
    // Setup parameters
    float TTIlen = 0.0005f; // length of each TTI, assuming numerology 1
    hdf5hpp::hdf5_dataset dset_carrier              = inputFile.open_dataset("carrier_pars");
    hdf5hpp::hdf5_dataset_elem dset_carrier_elem    = dset_carrier[0];
    hdf5hpp::hdf5_dataset dset_chan                 = inputFile.open_dataset("chan_pars");
    hdf5hpp::hdf5_dataset_elem dset_chan_elem       = dset_chan[0];
    // chanType: 0 - AWGN, 1 - TDL, 2 - CDL, 3 - P2P 
    uint8_t chanType                               = dset_chan_elem["chanType"].as<uint8_t>();
    uint16_t N_ueLayer = 0;
    if (chanType == 2)  // CDL, need to check antenna size
    { 
        std::vector<uint8_t> ueAntSize = dset_chan_elem["ueAntSize"].as<std::vector<uint8_t>>();
        N_ueLayer = std::accumulate(ueAntSize.begin(), ueAntSize.end(), 1, std::multiplies<uint8_t>());
    }
    else
    {
        N_ueLayer = dset_chan_elem["numUeAnt"].as<uint16_t>();
    }
    
    // validate chanType
    // currently only AWGN, TDL, and CDL are implemented
    assert (chanType == 0 || chanType == 1 || chanType == 2);

    // ---------------------------------------------------------------
    // Open output files

    std::unique_ptr<hdf5hpp::hdf5_file> debugFile;
    if(!puschDbgFilename.empty())
    {
        debugFile.reset(new hdf5hpp::hdf5_file(hdf5hpp::hdf5_file::create(puschDbgFilename.c_str())));
    }

    std::unique_ptr<hdf5hpp::hdf5_file> tberFile;
    bool                     tberOutputFlag = false;
    hdf5hpp::hdf5_file       tberHdf5File;
    if(!tberFilename.empty())
    {
        tberOutputFlag = true;
        tberFile.reset(new hdf5hpp::hdf5_file(hdf5hpp::hdf5_file::create(tberFilename.c_str())));
        tberHdf5File = hdf5hpp::hdf5_file::open(tberFilename.c_str());
    }

    // -------------------------------------------------------------------
    // Load PUSCH datasets

    std::vector<std::string> puschInputFilenameVec(1);
    puschInputFilenameVec[0] = puschInputFilename;

    StaticApiDataset  puschStaticDataset(puschInputFilenameVec, cuStrmMain.handle(), puschDbgFilename);
    DynApiDataset     puschDynDataset(puschInputFilenameVec,    cuStrmMain.handle());
    EvalDataset       evalDataset(puschInputFilenameVec,        cuStrmMain.handle());
    cudaStreamSynchronize(cuStrmMain.handle()); // synch to ensure data copied

    puschDynDataset.puschDynPrm.cpuCopyOn  = 1;

    //---------------------------------------------------------------
    // Determine number of UCI Cbs 

    cuphyPuschUePrm_t& uePrms = puschDynDataset.uePrmsVec[0];
    uint32_t nHarqUcis = 0;
    uint32_t nCsi1Ucis = 0;
    uint32_t nCsi2Ucis = 0;

    if(uePrms.pUciPrms != NULL)
    {
        if(uePrms.pUciPrms->nBitsHarq > 0)
            nHarqUcis = 1;

        if(uePrms.pUciPrms->nBitsCsi1 > 0)
            nCsi1Ucis = 1;
    }

    if((uePrms.pduBitmap >> 5) & 1)
        nCsi2Ucis = 1;


    // ---------------------------------------------------------------
    // Create receiver

    cuphy::pusch_rx pusch_rx(puschStaticDataset.puschStatPrms, cuStrmMain.handle());
    cudaStreamSynchronize(cuStrmMain.handle()); // synch to ensure data copied

    //---------------------------------------------------------------
    // Allocate Harq buffer

    const cuphyPuschBatchPrmHndl_t puschBatchPrmHndl = nullptr;
    puschDynDataset.puschDynPrm.setupPhase = PUSCH_SETUP_PHASE_1;
    pusch_rx.setup(puschDynDataset.puschDynPrm, puschBatchPrmHndl);
    
    uint32_t maxNumRmLLRs = 7938048;;
    std::vector<cuphy::buffer<uint8_t, cuphy::device_alloc>> m_harqBuffers(puschDynDataset.DataOut.totNumTbs);
    m_harqBuffers[0] = std::move(cuphy::buffer<uint8_t, cuphy::device_alloc>(maxNumRmLLRs * sizeof(__half)));

    puschDynDataset.puschDynPrm.pDataInOut->pHarqBuffersInOut[0] = static_cast<uint8_t*>(m_harqBuffers[0].addr());

    //-----------------------------------------------------------------
    // Allocate PUSCH output buffers

    cuphy::tensor_pinned decodedTbDataCpu(CUPHY_R_8U, 183456, cuphy::tensor_flags::align_tight);
    cuphy::tensor_pinned decodedCbCrc(CUPHY_R_32U, 1000, cuphy::tensor_flags::align_tight);
    cuphy::tensor_pinned decodedTbCrc(CUPHY_R_32U, 10, cuphy::tensor_flags::align_tight);

    puschDynDataset.DataOut.pTbPayloads = static_cast<uint8_t*>(decodedTbDataCpu.addr());
    puschDynDataset.DataOut.pCbCrcs     = static_cast<uint32_t*>(decodedCbCrc.addr());
    puschDynDataset.DataOut.pTbCrcs     = static_cast<uint32_t*>(decodedTbCrc.addr());

    //------------------------------------------------------------------
    // Received signal buffers

    cuphy::rng rng(seed, 0, cuStrmMain.handle());

    uint32_t nSubcarriers  = puschStaticDataset.cellStatPrmVec[0].nPrbUlBwp*12;
    uint32_t nGnbAnt       = puschStaticDataset.cellStatPrmVec[0].nRxAnt;
    uint32_t slotSizeBytes = nSubcarriers * nGnbAnt * OFDM_SYMBOLS_PER_SLOT * sizeof(__half2);

    cuphy::tensor_ref tNoiseFreeTxSlot;
    tNoiseFreeTxSlot.desc().set(CUPHY_C_16F,  nSubcarriers, OFDM_SYMBOLS_PER_SLOT, N_ueLayer, cuphy::tensor_flags::align_tight);
    tNoiseFreeTxSlot.set_addr(puschDynDataset.DataIn.pTDataRx->pAddr);

    cuphy::tensor_device tNoisyRxSlot(CUPHY_C_16F, nSubcarriers, OFDM_SYMBOLS_PER_SLOT, nGnbAnt, cuphy::tensor_flags::align_tight);
    puschDynDataset.DataIn.pTDataRx->pAddr = tNoisyRxSlot.addr();
    
    /*------------------------ Create fading channel ------------------------*/
    uint8_t phychanType = 0; // PUSCH
    fadingChan<__half2>* fadeChanPtr = new fadingChan<__half2>(static_cast<__half2*>(tNoiseFreeTxSlot.addr()), static_cast<__half2*>(tNoisyRxSlot.addr()), cuStrmMain.handle(), chanType, static_cast<uint16_t>(seed), phychanType);
    fadeChanPtr->setup(inputFile, 1/*enableSwapTxRx*/);

    // -----------------------------------------------------------------
    // Generate Tber curves


    int nSnrSteps = snrVec.size();
    cuphy::typed_tensor<CUPHY_R_32F, cuphy::pinned_alloc> tTberCpu(nSnrSteps);
    cuphy::typed_tensor<CUPHY_R_32F, cuphy::pinned_alloc> tSnrCpu(nSnrSteps);
    cuphy::typed_tensor<CUPHY_R_32U, cuphy::pinned_alloc> tNumTbsCpu(nSnrSteps);
    cuphy::typed_tensor<CUPHY_R_32U, cuphy::pinned_alloc> tNumTbErrsCpu(nSnrSteps);
    cuphy::typed_tensor<CUPHY_R_32U, cuphy::pinned_alloc> tNumHarqUciSegErrsCpu(nSnrSteps);
    cuphy::typed_tensor<CUPHY_R_32U, cuphy::pinned_alloc> tNumHarqUciSegsCpu(nSnrSteps);
    cuphy::typed_tensor<CUPHY_R_32U, cuphy::pinned_alloc> tNumCsi1UciSegErrsCpu(nSnrSteps);
    cuphy::typed_tensor<CUPHY_R_32U, cuphy::pinned_alloc> tNumCsi1UciSegsCpu(nSnrSteps);
    cuphy::typed_tensor<CUPHY_R_32U, cuphy::pinned_alloc> tNumCsi2UciSegErrsCpu(nSnrSteps);
    cuphy::typed_tensor<CUPHY_R_32U, cuphy::pinned_alloc> tNumCsi2UciSegsCpu(nSnrSteps);


    for(int snrStepIdx = 0; snrStepIdx < nSnrSteps; snrStepIdx++)
    {
        float snr            = snrVec[snrStepIdx];
        float noiseVariance  = pow(10.0, -snr / 10.0);
        cuComplex m32_0      = make_cuFloatComplex(0, 0);
        cuComplex stddev32_1 = make_cuFloatComplex(sqrt(noiseVariance / 2), sqrt(noiseVariance / 2));

        tNumTbsCpu(snrStepIdx)            = 0;
        tNumTbErrsCpu(snrStepIdx)         = 0;
        tNumHarqUciSegErrsCpu(snrStepIdx) = 0;
        tNumHarqUciSegsCpu(snrStepIdx)    = 0;
        tNumCsi1UciSegErrsCpu(snrStepIdx) = 0;
        tNumCsi1UciSegsCpu(snrStepIdx)    = 0;
        tNumCsi2UciSegErrsCpu(snrStepIdx) = 0;
        tNumCsi2UciSegsCpu(snrStepIdx)    = 0;


        for(int itrIdx = 0; itrIdx < nItrPerSnr; ++itrIdx)
        {

            // channel fading/noise
            fadeChanPtr -> run(TTIlen * itrIdx, snr, 1 /*enableSwapTxRx*/);
            cudaStreamSynchronize(cuStrmMain.handle()); // synch to ensure data copied

            // setup receiver:
            puschDynDataset.puschDynPrm.setupPhase = PUSCH_SETUP_PHASE_1;
            pusch_rx.setup(puschDynDataset.puschDynPrm, puschBatchPrmHndl);
            CUDA_CHECK(cudaMemsetAsync(m_harqBuffers[0].addr(), 0, puschDynDataset.DataOut.h_harqBufferSizeInBytes[0], cuStrmMain.handle()));
            puschDynDataset.puschDynPrm.setupPhase = PUSCH_SETUP_PHASE_2;
            pusch_rx.setup(puschDynDataset.puschDynPrm, puschBatchPrmHndl);

            // run receiver:
            pusch_rx.run(PUSCH_RUN_ALL_PHASES);

            // check Tb CRC:
            CUDA_CHECK(cudaStreamSynchronize(cuStrmMain.handle())); 
            uint32_t tbCrc = puschDynDataset.DataOut.pTbCrcs[0];
            if(tbCrc > 0)
            {
                tTberCpu(snrStepIdx) += 1;
                tNumTbErrsCpu(snrStepIdx) += 1;
            }
            tNumTbsCpu(snrStepIdx) += 1;

            if(quickBlerFlag && (tNumTbErrsCpu(snrStepIdx) == quickBlerNumTbErrs))
            {
                break;
            }

            // check uci errors:
            evalDataset.computeNumUciCbErrors(puschDynDataset, false);
            tNumHarqUciSegErrsCpu(snrStepIdx)   += evalDataset.nHarqUciErrors;
            tNumHarqUciSegsCpu(snrStepIdx)      += nHarqUcis;
            tNumCsi1UciSegErrsCpu(snrStepIdx)   += evalDataset.nCsi1UciErrors;
            tNumCsi1UciSegsCpu(snrStepIdx)      += nCsi1Ucis;
            tNumCsi2UciSegErrsCpu(snrStepIdx)   += evalDataset.nCsi2UciErrors;
            tNumCsi2UciSegsCpu(snrStepIdx)      += nCsi2Ucis;
        }
        tTberCpu(snrStepIdx) =  static_cast<float>(tNumTbErrsCpu(snrStepIdx)) / static_cast<float>(tNumTbsCpu(snrStepIdx));

        if(quickBlerFlag && (tNumTbErrsCpu(snrStepIdx) == 0))
        {
            break;
        }
    }

    for(int snrStepIdx = 0; snrStepIdx < nSnrSteps; snrStepIdx++)
    {
        tSnrCpu(snrStepIdx) = snrVec[snrStepIdx];
    }
    
    // cleanup
    if (fadeChanPtr) {
        delete fadeChanPtr;
    }
        
    // -----------------------------------------------------------------
    // Save tber to H5

    if(tberOutputFlag)
    {
        cuphy::typed_tensor<CUPHY_R_32F, cuphy::device_alloc> tTberGpu(nSnrSteps);
        cuphy::typed_tensor<CUPHY_R_32F, cuphy::device_alloc> tSnrGpu(nSnrSteps);
        cuphy::typed_tensor<CUPHY_R_32U, cuphy::device_alloc> tNumTbsGpu(nSnrSteps);
        cuphy::typed_tensor<CUPHY_R_32U, cuphy::device_alloc> tNumTbErrsGpu(nSnrSteps);
        cuphy::typed_tensor<CUPHY_R_32U, cuphy::device_alloc> tNumHarqUciSegErrsGpu(nSnrSteps);
        cuphy::typed_tensor<CUPHY_R_32U, cuphy::device_alloc> tNumHarqUciSegsGpu(nSnrSteps);
        cuphy::typed_tensor<CUPHY_R_32U, cuphy::device_alloc> tNumCsi1UciSegErrsGpu(nSnrSteps);
        cuphy::typed_tensor<CUPHY_R_32U, cuphy::device_alloc> tNumCsi1UciSegsGpu(nSnrSteps);
        cuphy::typed_tensor<CUPHY_R_32U, cuphy::device_alloc> tNumCsi2UciSegErrsGpu(nSnrSteps);
        cuphy::typed_tensor<CUPHY_R_32U, cuphy::device_alloc> tNumCsi2UciSegsGpu(nSnrSteps);

        CUDA_CHECK_EXCEPTION(cudaMemcpyAsync(static_cast<void*>(tTberGpu.addr()), static_cast<void*>(tTberCpu.addr()), nSnrSteps * sizeof(float), cudaMemcpyHostToDevice, cuStrmMain.handle()));
        CUDA_CHECK_EXCEPTION(cudaMemcpyAsync(static_cast<void*>(tSnrGpu.addr()) , static_cast<void*>(tSnrCpu.addr()) , nSnrSteps * sizeof(float), cudaMemcpyHostToDevice, cuStrmMain.handle()));
        CUDA_CHECK_EXCEPTION(cudaMemcpyAsync(static_cast<void*>(tNumTbsGpu.addr()) , static_cast<void*>(tNumTbsCpu.addr()) , nSnrSteps * sizeof(uint32_t), cudaMemcpyHostToDevice, cuStrmMain.handle()));
        CUDA_CHECK_EXCEPTION(cudaMemcpyAsync(static_cast<void*>(tNumTbErrsGpu.addr()) , static_cast<void*>(tNumTbErrsCpu.addr()) , nSnrSteps * sizeof(uint32_t), cudaMemcpyHostToDevice, cuStrmMain.handle()));
        CUDA_CHECK_EXCEPTION(cudaMemcpyAsync(static_cast<void*>(tNumHarqUciSegErrsGpu.addr()) , static_cast<void*>(tNumHarqUciSegErrsCpu.addr()) , nSnrSteps * sizeof(uint32_t), cudaMemcpyHostToDevice, cuStrmMain.handle()));
        CUDA_CHECK_EXCEPTION(cudaMemcpyAsync(static_cast<void*>(tNumHarqUciSegsGpu.addr()) , static_cast<void*>(tNumHarqUciSegsCpu.addr()) , nSnrSteps * sizeof(uint32_t), cudaMemcpyHostToDevice, cuStrmMain.handle()));
        CUDA_CHECK_EXCEPTION(cudaMemcpyAsync(static_cast<void*>(tNumCsi1UciSegErrsGpu.addr()) , static_cast<void*>(tNumCsi1UciSegErrsCpu.addr()) , nSnrSteps * sizeof(uint32_t), cudaMemcpyHostToDevice, cuStrmMain.handle()));
        CUDA_CHECK_EXCEPTION(cudaMemcpyAsync(static_cast<void*>(tNumCsi1UciSegsGpu.addr()) , static_cast<void*>(tNumCsi1UciSegsCpu.addr()) , nSnrSteps * sizeof(uint32_t), cudaMemcpyHostToDevice, cuStrmMain.handle()));
        CUDA_CHECK_EXCEPTION(cudaMemcpyAsync(static_cast<void*>(tNumCsi2UciSegErrsGpu.addr()) , static_cast<void*>(tNumCsi2UciSegErrsCpu.addr()) , nSnrSteps * sizeof(uint32_t), cudaMemcpyHostToDevice, cuStrmMain.handle()));
        CUDA_CHECK_EXCEPTION(cudaMemcpyAsync(static_cast<void*>(tNumCsi2UciSegsGpu.addr()) , static_cast<void*>(tNumCsi2UciSegsCpu.addr()) , nSnrSteps * sizeof(uint32_t), cudaMemcpyHostToDevice, cuStrmMain.handle()));
        CUDA_CHECK(cudaStreamSynchronize(cuStrmMain.handle()));

        cuphy::write_HDF5_dataset(tberHdf5File, tSnrGpu,  "snr", cuStrmMain.handle());
        cuphy::write_HDF5_dataset(tberHdf5File, tTberGpu, "tber", cuStrmMain.handle());
        cuphy::write_HDF5_dataset(tberHdf5File, tNumTbsGpu, "numTBs", cuStrmMain.handle());
        cuphy::write_HDF5_dataset(tberHdf5File, tNumTbErrsGpu, "numErrTBs", cuStrmMain.handle());
        cuphy::write_HDF5_dataset(tberHdf5File, tNumHarqUciSegErrsGpu, "numErrHarqUcis", cuStrmMain.handle());
        cuphy::write_HDF5_dataset(tberHdf5File, tNumHarqUciSegsGpu, "numHarqUcis", cuStrmMain.handle());
        cuphy::write_HDF5_dataset(tberHdf5File, tNumCsi1UciSegErrsGpu, "numErrCsi1Ucis", cuStrmMain.handle());
        cuphy::write_HDF5_dataset(tberHdf5File, tNumCsi1UciSegsGpu, "numCsi1Ucis", cuStrmMain.handle());
        cuphy::write_HDF5_dataset(tberHdf5File, tNumCsi2UciSegErrsGpu, "numErrCsi2Ucis", cuStrmMain.handle());
        cuphy::write_HDF5_dataset(tberHdf5File, tNumCsi2UciSegsGpu, "numCsi2Ucis", cuStrmMain.handle());
        CUDA_CHECK(cudaStreamSynchronize(cuStrmMain.handle()));

        tberHdf5File.close();
    }
}
