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

#include "sls_chan.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand.h>
#include <cassert>
#include <cstdint>
#include <hdf5.h>

// Kernel to initialize curandState array with unique seeds for each thread
__global__ void initCurandStatesKernel(curandState* states, const uint32_t numStates, const uint32_t baseSeed) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numStates) {
        // Initialize each state with a unique seed
        // Use large spacing between seeds to ensure statistical independence
        curand_init(baseSeed + tid * 1000U, 0, 0, &states[tid]);
    }
}

// Function to find BS boundaries
inline void findBsBoundary(const std::vector<CellParam>& cellParams, float ISD, float& maxX, float& minX, float& maxY, float& minY) {
    if (cellParams.empty()) {
        maxX = minX = maxY = minY = 0.0f;
        return;
    }
    maxX = cellParams[0].loc.x;
    minX = cellParams[0].loc.x;
    maxY = cellParams[0].loc.y;
    minY = cellParams[0].loc.y;
    for (const auto& cell : cellParams) {
        if (cell.loc.x > maxX) maxX = cell.loc.x;
        if (cell.loc.x < minX) minX = cell.loc.x;
        if (cell.loc.y > maxY) maxY = cell.loc.y;
        if (cell.loc.y < minY) minY = cell.loc.y;
    }
    float cellRadius = ISD / std::sqrt(3.0f);
    maxX += cellRadius;
    minX -= cellRadius;
    maxY += ISD / 2.0f;
    minY -= ISD / 2.0f;

    maxX = roundf(maxX);
    minX = roundf(minX);
    maxY = roundf(maxY);
    minY = roundf(minY);
}

template <typename Tscalar, typename Tcomplex>
slsChan<Tscalar, Tcomplex>::slsChan(const SimConfig* simConfig, const SystemLevelConfig* sysConfig, const ExternalConfig* extConfig, uint32_t randSeed, cudaStream_t strm)
: m_simConfig(simConfig)
    , m_sysConfig(sysConfig)
    , m_extConfig(extConfig)
    , m_randSeed(randSeed)
    , m_strm(strm)
    , m_gen(randSeed)
    , m_uniformDist(0.0f, 1.0f)
    , m_normalDist(0.0f, 1.0f)
    , m_updatePerTTILinkParams(true)
    , m_updatePLAndPenetrationLoss(true)
    , m_updateAllLSPs(true)
    , m_updateLosState(true)
    , m_lastAllocatedLinks(0)
    , m_lastAllocatedActiveLinks(0)
    , m_cirCoe(nullptr)
    , m_cirNormDelay(nullptr)
    , m_cirNtaps(nullptr)
    , m_freqChanPrbg(nullptr)
    , m_freqChanSc(nullptr)
    , m_rxSigOut(nullptr)
    , m_d_curandStates(nullptr)
    , m_maxCurandStates(0)
{
    // Perform BS and UE dropping
    bsUeDropping();
#ifdef SLS_DEBUG_
    dumpTopologyToYaml("network_topology.yaml");
#endif
    // Initialize antenna panel configuration
    initializeAntPanelConfig();

    // if extConfig is not nullptr, override cell and UT parameters
    if (m_extConfig != nullptr) {
        setup(m_extConfig);
    }

    // Initialize link parameters - one per site-UT pair (co-sited sectors share link parameters)
    m_linkParams.resize(m_topology.nSite * m_topology.nUT);
    // Initialize cluster parameters vector with the correct size
    m_clusterParams.resize(m_topology.nSite * m_topology.nUT);
    
    // Find BS boundaries first
    findBsBoundary(m_topology.cellParams, m_topology.ISD, m_maxX, m_minX, m_maxY, m_minY);
    
    // calculate common link parameters
    // this also sets the nUeAnt and nBsAnt that will be used in memory allocation
    calCmnLinkParams();
    
    // allocate GPU memory for internal data structures
    allocateStaticGpuMem();
    
    // allocate GPU memory for dynamic data structures, use 1 link initially
    allocateDynamicGpuMem(1);
    
    // generate CRN and calculate link/cluster parameters
    if (m_simConfig->cpu_only_mode == 1) {
        generateCRN();
        calLinkParam();
        calClusterRay();
    } else {
        generateCRNGPU();
        calLinkParamGPU();
        calClusterRayGPU();
    }
}


template <typename Tscalar, typename Tcomplex>
void slsChan<Tscalar, Tcomplex>::run(const float refTime,
                                    const uint8_t continuous_fading,
                                    const std::vector<uint16_t>& activeCell,
                                    const std::vector<std::vector<uint16_t>>& activeUt,
                                    const std::vector<Coordinate>& utNewLoc,
                                    const std::vector<float3>& utNewVelocity,
                                    const std::vector<Tcomplex*>& cirCoePerCell,
                                    const std::vector<uint16_t*>& cirNormDelayPerCell,
                                    const std::vector<uint16_t*>& cirNTapsPerCell,
                                    const std::vector<Tcomplex*>& cfrScPerCell,
                                    const std::vector<Tcomplex*>& cfrPrbgPerCell)
{
    m_refTime = refTime;

    // Update UT locations if provided
    if (!utNewLoc.empty()) {
        for (uint16_t uid = 0; uid < m_topology.nUT; ++uid) {
            if (uid < utNewLoc.size()) {
                m_topology.utParams[uid].loc.x = utNewLoc[uid].x;
                m_topology.utParams[uid].loc.y = utNewLoc[uid].y;
                m_topology.utParams[uid].loc.z = utNewLoc[uid].z;
            }
        }
        m_updatePerTTILinkParams = true;
    }

    // Update UT velocities if provided
    if (!utNewVelocity.empty()) {
        for (uint16_t uid = 0; uid < m_topology.nUT; ++uid) {
            if (uid < utNewVelocity.size()) {
                m_topology.utParams[uid].velocity[0] = utNewVelocity[uid].x;
                m_topology.utParams[uid].velocity[1] = utNewVelocity[uid].y;
                m_topology.utParams[uid].velocity[2] = utNewVelocity[uid].z;

                // Assert that velocity.z is approximately zero
                // 38.901 specification: "UT mobility (horizontal plane only)"
                assert(std::abs(utNewVelocity[uid].z) < 1e-6f);

                // update UT antenna orientation to the angle of (x, y) in degrees
                const float orientation_deg = std::atan2(utNewVelocity[uid].y, utNewVelocity[uid].x) * 180.0f / M_PI;
                m_topology.utParams[uid].antPanelOrientation[1] = orientation_deg;
            }
        }
    }

    // update UT new location and velocity to GPU (GPU mode only)
    if (m_simConfig->cpu_only_mode == 0) {
        if (!utNewLoc.empty() || !utNewVelocity.empty()) {
            CHECK_CUDAERROR(cudaMemcpyAsync(m_d_utParams, m_topology.utParams.data(), m_topology.utParams.size() * sizeof(UtParam), cudaMemcpyHostToDevice, m_strm));
        }
    }
    
    // calculate number of links
    uint32_t nActiveLinks = 0;
    if (activeCell.empty() && activeUt.empty()) {
        nActiveLinks = m_topology.nSector * m_topology.nUT;
    }
    else {
        for (const auto& uts : activeUt) {
            nActiveLinks += uts.size();
        }
    }

    // check memory allocation
    allocateDynamicGpuMem(nActiveLinks);
    
    // Set per-cell pointers based on internal memory mode
    // Mode 0: External CIR/CFR, Mode 1: Internal CIR/External CFR, Mode 2: Internal CIR/CFR
    if (m_simConfig->internal_memory_mode == 0) {
        // Mode 0: Use external memory for both CIR and CFR
        m_cirCoePerCell = cirCoePerCell;
        m_cirNormDelayPerCell = cirNormDelayPerCell;
        m_cirNtapsPerCell = cirNTapsPerCell;
        m_freqChanScPerCell = cfrScPerCell;
        m_freqChanPrbgPerCell = cfrPrbgPerCell;
    } else if (m_simConfig->internal_memory_mode == 1) {
        // Mode 1: Use internal CIR, external CFR
        // CIR pointers already set by allocateDynamicGpuMem
        // Set CFR pointers to external memory
        m_freqChanScPerCell = cfrScPerCell;
        m_freqChanPrbgPerCell = cfrPrbgPerCell;
#ifdef SLS_DEBUG_
        printf("DEBUG: Mode 1 - Set external CFR pointers:\n");
        printf("  m_freqChanScPerCell.size() = %zu\n", m_freqChanScPerCell.size());
        printf("  m_freqChanPrbgPerCell.size() = %zu\n", m_freqChanPrbgPerCell.size());
        for (size_t i = 0; i < std::min(m_freqChanScPerCell.size(), (size_t)3); i++) {
            printf("  m_freqChanScPerCell[%zu] = %p\n", i, (void*)m_freqChanScPerCell[i]);
        }
#endif
    } else if (m_simConfig->internal_memory_mode == 2) {
        // Mode 2: Use internal memory for both CIR and CFR
        // All pointers already set by allocateDynamicGpuMem
    }
    
    // Update active link indices based on active cells and UTs
    updateActiveLinkInd(activeCell, activeUt);
    
#ifdef SLS_DEBUG_
    // Debug: Print m_activeLinkParams after updateActiveLinkInd
    printf("DEBUG: m_activeLinkParams after updateActiveLinkInd:\n");
    printf("  m_activeLinkParams.size(): %zu\n", m_activeLinkParams.size());
    for (size_t i = 0; i < m_activeLinkParams.size() && i < 10; i++) {
        printf("  Link %zu: cid=%d, uid=%d, linkIdx=%u, lspReadIdx=%u\n", i, 
               m_activeLinkParams[i].cid, m_activeLinkParams[i].uid,
               m_activeLinkParams[i].linkIdx, m_activeLinkParams[i].lspReadIdx);
    }
    printf("  nActiveLinks (from input): %u\n", nActiveLinks);
    printf("  Expected: nActiveLinks should equal m_activeLinkParams.size()\n");
#endif
    
    assert(m_activeLinkParams.size() == nActiveLinks);
    // Synchronize stream for all processing (GPU mode only)
    if (m_simConfig->cpu_only_mode == 0) {
        CHECK_CUDAERROR(cudaStreamSynchronize(m_strm));
    }

    // Regenerate large scale parameters if not in continuous fading mode or location changed
    if (!continuous_fading || m_updatePerTTILinkParams) {
        m_updatePLAndPenetrationLoss = m_sysConfig->enable_per_tti_lsp >= 1;
        m_updateAllLSPs = m_sysConfig->enable_per_tti_lsp == 2;
        if (m_simConfig->cpu_only_mode == 1) {
            calLinkParam();
        } else {
            calLinkParamGPU();
        }
    }

    // update cluster ray parameters, only for enable_per_tti_lsp mode 2
    if (m_updateAllLSPs) {
        if (m_simConfig->cpu_only_mode == 1) {
            calClusterRay();
        } else {
            calClusterRayGPU();
        }
    }

    // reset update flags
    m_updatePerTTILinkParams = false;
    m_updatePLAndPenetrationLoss = false;
    m_updateAllLSPs = false;

    // synchronize stream for all processing (GPU mode only)
    if (m_simConfig->cpu_only_mode == 0) {
        CHECK_CUDAERROR(cudaStreamSynchronize(m_strm));
    }
    
    // Generate Channel Impulse Response (CIR)
    if (m_simConfig->cpu_only_mode == 1) {
        generateCIR();
    } else {
        generateCIRGPU();
    }
    // Synchronize stream for all processing (GPU mode only)
    if (m_simConfig->cpu_only_mode == 0) {
        CHECK_CUDAERROR(cudaStreamSynchronize(m_strm));
    }

    // Generate Channel Frequency Response (CFR) if run_mode > 0
    if (m_simConfig->run_mode > 0) {
#ifdef SLS_DEBUG_
        printf("DEBUG: About to call CFR generation, run_mode=%d\n", m_simConfig->run_mode);
#endif
        if (m_simConfig->cpu_only_mode == 1) {
            generateCFR();
        } else {
            generateCFRGPU();
        }
#ifdef SLS_DEBUG_
        printf("DEBUG: CFR generation completed\n");
#endif
    }
    
    // Process transmitted samples if required
    if (m_simConfig->tx_sig_in != nullptr) {
        processTxSamples();
    }
    // Synchronize stream for all processing (GPU mode only)
    if (m_simConfig->cpu_only_mode == 0) {
        CHECK_CUDAERROR(cudaStreamSynchronize(m_strm));
    }
    
    // Copy data from internal contiguous storage to external per-cell arrays if needed
    if (m_simConfig->internal_memory_mode >= 1) {
        // If external per-cell arrays are provided, copy from internal contiguous storage
        if ((!cirCoePerCell.empty() && cirCoePerCell[0] != nullptr) ||
            (!cirNormDelayPerCell.empty() && cirNormDelayPerCell[0] != nullptr) ||
            (!cirNTapsPerCell.empty() && cirNTapsPerCell[0] != nullptr) ||
            (!cfrScPerCell.empty() && cfrScPerCell[0] != nullptr) ||
            (!cfrPrbgPerCell.empty() && cfrPrbgPerCell[0] != nullptr)) {
            
            // Temporarily store external arrays for copying
            m_cirCoePerCell = cirCoePerCell;
            m_cirNormDelayPerCell = cirNormDelayPerCell;
            m_cirNtapsPerCell = cirNTapsPerCell;
            m_freqChanScPerCell = cfrScPerCell;
            m_freqChanPrbgPerCell = cfrPrbgPerCell;
            
            copyContiguousToPerCell(activeCell, activeUt);
        }
    }

    // Synchronize stream (GPU mode only)
    if (m_simConfig->cpu_only_mode == 0) {
        CHECK_CUDAERROR(cudaStreamSynchronize(m_strm));
        // Check for errors
        CHECK_CUDAERROR(cudaGetLastError());
    }
}


template <typename Tscalar, typename Tcomplex>
void slsChan<Tscalar, Tcomplex>::reset()
{
    // calculate common link parameters
    calCmnLinkParams();
    // generate CRN
    if (m_simConfig->cpu_only_mode == 1) {
        generateCRN();
    } else {
        generateCRNGPU();
    }
    // set per TTI flag to true
    m_updatePerTTILinkParams = true;
    m_updatePLAndPenetrationLoss = true;
    m_updateAllLSPs = true;
    m_updateLosState = true;  // Reset LOS state so it will be regenerated
    // those flags will be reset in run()
}

template <typename Tscalar, typename Tcomplex>
void slsChan<Tscalar, Tcomplex>::saveSlsChanToH5File(std::string_view filenameEnding)
{
    // Create HDF5 file with optional filename ending
    std::string outFilename = "slsChanData_" + std::to_string(m_topology.nSite) + "sites_" + 
                             std::to_string(m_topology.nUT) + "uts" + std::string(filenameEnding) + ".h5";
    hid_t slsH5File = H5Fcreate(outFilename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (slsH5File < 0) {
        fprintf(stderr, "Failed to create HDF5 file: %s\n", outFilename.c_str());
        exit(EXIT_FAILURE);
    }

    // Calculate total number of links
    const uint32_t nTotalLinks = m_topology.nSite * m_topology.nUT;
    
    // Dump Link Parameters
    if (!m_linkParams.empty() && m_linkParams.size() >= nTotalLinks) {
        // Create compound datatype for LinkParams
        hid_t linkParamsType = H5Tcreate(H5T_COMPOUND, sizeof(LinkParams));
        H5Tinsert(linkParamsType, "d2d", HOFFSET(LinkParams, d2d), H5T_NATIVE_FLOAT);
        H5Tinsert(linkParamsType, "d2d_in", HOFFSET(LinkParams, d2d_in), H5T_NATIVE_FLOAT);
        H5Tinsert(linkParamsType, "d2d_out", HOFFSET(LinkParams, d2d_out), H5T_NATIVE_FLOAT);
        H5Tinsert(linkParamsType, "d3d", HOFFSET(LinkParams, d3d), H5T_NATIVE_FLOAT);
        H5Tinsert(linkParamsType, "d3d_in", HOFFSET(LinkParams, d3d_in), H5T_NATIVE_FLOAT);
        H5Tinsert(linkParamsType, "phi_LOS_AOD", HOFFSET(LinkParams, phi_LOS_AOD), H5T_NATIVE_FLOAT);
        H5Tinsert(linkParamsType, "theta_LOS_ZOD", HOFFSET(LinkParams, theta_LOS_ZOD), H5T_NATIVE_FLOAT);
        H5Tinsert(linkParamsType, "phi_LOS_AOA", HOFFSET(LinkParams, phi_LOS_AOA), H5T_NATIVE_FLOAT);
        H5Tinsert(linkParamsType, "theta_LOS_ZOA", HOFFSET(LinkParams, theta_LOS_ZOA), H5T_NATIVE_FLOAT);
        H5Tinsert(linkParamsType, "losInd", HOFFSET(LinkParams, losInd), H5T_NATIVE_UINT8);
        H5Tinsert(linkParamsType, "pathloss", HOFFSET(LinkParams, pathloss), H5T_NATIVE_FLOAT);
        H5Tinsert(linkParamsType, "SF", HOFFSET(LinkParams, SF), H5T_NATIVE_FLOAT);
        H5Tinsert(linkParamsType, "K", HOFFSET(LinkParams, K), H5T_NATIVE_FLOAT);
        H5Tinsert(linkParamsType, "DS", HOFFSET(LinkParams, DS), H5T_NATIVE_FLOAT);
        H5Tinsert(linkParamsType, "ASD", HOFFSET(LinkParams, ASD), H5T_NATIVE_FLOAT);
        H5Tinsert(linkParamsType, "ASA", HOFFSET(LinkParams, ASA), H5T_NATIVE_FLOAT);
        H5Tinsert(linkParamsType, "mu_lgZSD", HOFFSET(LinkParams, mu_lgZSD), H5T_NATIVE_FLOAT);
        H5Tinsert(linkParamsType, "sigma_lgZSD", HOFFSET(LinkParams, sigma_lgZSD), H5T_NATIVE_FLOAT);
        H5Tinsert(linkParamsType, "mu_offset_ZOD", HOFFSET(LinkParams, mu_offset_ZOD), H5T_NATIVE_FLOAT);
        H5Tinsert(linkParamsType, "ZSD", HOFFSET(LinkParams, ZSD), H5T_NATIVE_FLOAT);
        H5Tinsert(linkParamsType, "ZSA", HOFFSET(LinkParams, ZSA), H5T_NATIVE_FLOAT);

        // Create dataspace and dataset for link parameters
        hsize_t linkDims[1] = {nTotalLinks};
        hid_t linkDataspace = H5Screate_simple(1, linkDims, nullptr);
        hid_t linkDataset = H5Dcreate2(slsH5File, "linkParams", linkParamsType, linkDataspace,
                                      H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        
        // Write link parameters data
        H5Dwrite(linkDataset, linkParamsType, H5S_ALL, H5S_ALL, H5P_DEFAULT, m_linkParams.data());
        
        // Cleanup
        H5Dclose(linkDataset);
        H5Sclose(linkDataspace);
        H5Tclose(linkParamsType);
    }

    // Dump Cluster Parameters
    if (!m_clusterParams.empty() && m_clusterParams.size() >= nTotalLinks) {
        // Create compound datatype for ClusterParams
        hid_t clusterParamsType = H5Tcreate(H5T_COMPOUND, sizeof(ClusterParams));
        H5Tinsert(clusterParamsType, "nCluster", HOFFSET(ClusterParams, nCluster), H5T_NATIVE_UINT16);
        H5Tinsert(clusterParamsType, "nRayPerCluster", HOFFSET(ClusterParams, nRayPerCluster), H5T_NATIVE_UINT16);
        
        // Add arrays - note: for arrays we need to create array types
        hsize_t maxClustersDim[1] = {ClusterParams::MAX_CLUSTERS};
        hsize_t strongest2ClustersDim[1] = {2};
        hsize_t maxRaysClustersDim[1] = {ClusterParams::MAX_CLUSTERS * ClusterParams::MAX_RAYS};
        hsize_t randomPhasesDim[1] = {ClusterParams::MAX_CLUSTERS * ClusterParams::MAX_RAYS * 4};
        
        hid_t floatArrayType = H5Tarray_create2(H5T_NATIVE_FLOAT, 1, maxClustersDim);
        hid_t uint16ArrayType = H5Tarray_create2(H5T_NATIVE_UINT16, 1, strongest2ClustersDim);
        hid_t floatRaysArrayType = H5Tarray_create2(H5T_NATIVE_FLOAT, 1, maxRaysClustersDim);
        hid_t floatPhasesArrayType = H5Tarray_create2(H5T_NATIVE_FLOAT, 1, randomPhasesDim);
        
        H5Tinsert(clusterParamsType, "delays", HOFFSET(ClusterParams, delays), floatArrayType);
        H5Tinsert(clusterParamsType, "powers", HOFFSET(ClusterParams, powers), floatArrayType);
        H5Tinsert(clusterParamsType, "strongest2clustersIdx", HOFFSET(ClusterParams, strongest2clustersIdx), uint16ArrayType);
        H5Tinsert(clusterParamsType, "phi_n_AoA", HOFFSET(ClusterParams, phi_n_AoA), floatArrayType);
        H5Tinsert(clusterParamsType, "phi_n_AoD", HOFFSET(ClusterParams, phi_n_AoD), floatArrayType);
        H5Tinsert(clusterParamsType, "theta_n_ZOD", HOFFSET(ClusterParams, theta_n_ZOD), floatArrayType);
        H5Tinsert(clusterParamsType, "theta_n_ZOA", HOFFSET(ClusterParams, theta_n_ZOA), floatArrayType);
        H5Tinsert(clusterParamsType, "xpr", HOFFSET(ClusterParams, xpr), floatRaysArrayType);
        H5Tinsert(clusterParamsType, "randomPhases", HOFFSET(ClusterParams, randomPhases), floatPhasesArrayType);
        H5Tinsert(clusterParamsType, "phi_n_m_AoA", HOFFSET(ClusterParams, phi_n_m_AoA), floatRaysArrayType);
        H5Tinsert(clusterParamsType, "phi_n_m_AoD", HOFFSET(ClusterParams, phi_n_m_AoD), floatRaysArrayType);
        H5Tinsert(clusterParamsType, "theta_n_m_ZOD", HOFFSET(ClusterParams, theta_n_m_ZOD), floatRaysArrayType);
        H5Tinsert(clusterParamsType, "theta_n_m_ZOA", HOFFSET(ClusterParams, theta_n_m_ZOA), floatRaysArrayType);

        // Create dataspace and dataset for cluster parameters
        hsize_t clusterDims[1] = {nTotalLinks};
        hid_t clusterDataspace = H5Screate_simple(1, clusterDims, nullptr);
        hid_t clusterDataset = H5Dcreate2(slsH5File, "clusterParams", clusterParamsType, clusterDataspace,
                                         H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        
        // Write cluster parameters data
        H5Dwrite(clusterDataset, clusterParamsType, H5S_ALL, H5S_ALL, H5P_DEFAULT, m_clusterParams.data());
        
        // Cleanup array types
        H5Tclose(floatArrayType);
        H5Tclose(uint16ArrayType);
        H5Tclose(floatRaysArrayType);
        H5Tclose(floatPhasesArrayType);
        
        // Cleanup dataset
        H5Dclose(clusterDataset);
        H5Sclose(clusterDataspace);
        H5Tclose(clusterParamsType);
    }

    // dump active link parameters
#ifdef SLS_DEBUG_
    printf("DEBUG: m_activeLinkParams.size() = %zu\n", m_activeLinkParams.size());
#endif
    if (!m_activeLinkParams.empty()) {
        // Create compound datatype for activeLink
        hid_t activeLinkType = H5Tcreate(H5T_COMPOUND, sizeof(activeLink<Tcomplex>));
        H5Tinsert(activeLinkType, "cid", HOFFSET(activeLink<Tcomplex>, cid), H5T_NATIVE_UINT16);
        H5Tinsert(activeLinkType, "uid", HOFFSET(activeLink<Tcomplex>, uid), H5T_NATIVE_UINT16);
        H5Tinsert(activeLinkType, "linkIdx", HOFFSET(activeLink<Tcomplex>, linkIdx), H5T_NATIVE_UINT32);
        H5Tinsert(activeLinkType, "lspReadIdx", HOFFSET(activeLink<Tcomplex>, lspReadIdx), H5T_NATIVE_UINT32);

        // Create dataspace and dataset for active link parameters
        hsize_t activeLinkDims[1] = {m_activeLinkParams.size()};
        hid_t activeLinkDataspace = H5Screate_simple(1, activeLinkDims, nullptr);
        hid_t activeLinkDataset = H5Dcreate2(slsH5File, "activeLinkParams", activeLinkType, activeLinkDataspace,
                                           H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        
        // Write active link parameters data
        H5Dwrite(activeLinkDataset, activeLinkType, H5S_ALL, H5S_ALL, H5P_DEFAULT, m_activeLinkParams.data());
        
        // Cleanup
        H5Dclose(activeLinkDataset);
        H5Sclose(activeLinkDataspace);
        H5Tclose(activeLinkType);
    }

    // dump CIR per cell
#ifdef SLS_DEBUG_
    printf("DEBUG: m_cirCoePerCell.size() = %zu\n", m_cirCoePerCell.size());
#endif
    if (!m_cirCoePerCell.empty() && m_cirCoePerCell[0] != nullptr) {
        const uint32_t nCells = static_cast<uint32_t>(m_cirCoePerCell.size());
        const uint32_t nUtsPerCell = m_topology.nUT;
        const uint32_t nCirCoeff = m_cmnLinkParams.nUeAnt * m_cmnLinkParams.nBsAnt * N_MAX_TAPS;
        
        // Create groups for CIR data
        hid_t cirGroup = H5Gcreate2(slsH5File, "cirPerCell", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        
        // Dump CIR coefficients per cell
        for (uint32_t cellIdx = 0; cellIdx < nCells; ++cellIdx) {
            if (m_cirCoePerCell[cellIdx] != nullptr) {
                // Create dataset name
                std::string datasetName = "cirCoe_cell" + std::to_string(cellIdx);
                
                // Create dataspace (nUtsPerCell x nCirCoeff complex values)
                hsize_t cirDims[2] = {nUtsPerCell, nCirCoeff};
                hid_t cirDataspace = H5Screate_simple(2, cirDims, nullptr);
                
                // Create complex datatype
                hid_t complexType = H5Tcreate(H5T_COMPOUND, sizeof(Tcomplex));
                H5Tinsert(complexType, "real", 0, H5T_NATIVE_FLOAT);
                H5Tinsert(complexType, "imag", sizeof(float), H5T_NATIVE_FLOAT);
                
                // Create dataset
                hid_t cirDataset = H5Dcreate2(cirGroup, datasetName.c_str(), complexType, cirDataspace,
                                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                
                // Copy data from GPU to host if needed, or use CPU pointer directly
                Tcomplex* dataPtr = nullptr;
                std::vector<Tcomplex> hostData;
                
                if (m_simConfig->cpu_only_mode == 0 && m_simConfig->internal_memory_mode >= 1) {
                    // GPU memory - copy to host
                    hostData.resize(nUtsPerCell * nCirCoeff);
                    CHECK_CUDAERROR(cudaMemcpy(hostData.data(), m_cirCoePerCell[cellIdx], 
                                               nUtsPerCell * nCirCoeff * sizeof(Tcomplex), 
                                               cudaMemcpyDeviceToHost));
                    dataPtr = hostData.data();
                } else {
                    // CPU memory - use pointer directly
                    dataPtr = m_cirCoePerCell[cellIdx];
                }
                
                // Write data
                H5Dwrite(cirDataset, complexType, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataPtr);
                
                // Cleanup
                H5Dclose(cirDataset);
                H5Sclose(cirDataspace);
                H5Tclose(complexType);
                
                // Write UE-to-row mapping for this cell
                // This allows the analysis script to know which row corresponds to which UE
                if (cellIdx < m_activeUt.size() && !m_activeUt[cellIdx].empty()) {
                    const std::string mappingName = "ue_mapping_cell" + std::to_string(cellIdx);
                    
                    // Create dataspace for UE mapping (1D array)
                    const hsize_t mappingDims[1] = {m_activeUt[cellIdx].size()};
                    const hid_t mappingDataspace = H5Screate_simple(1, mappingDims, nullptr);
                    
                    // Create dataset for UE mapping
                    const hid_t mappingDataset = H5Dcreate2(cirGroup, mappingName.c_str(), H5T_NATIVE_UINT16,
                                                            mappingDataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                    
                    // Write UE mapping
                    H5Dwrite(mappingDataset, H5T_NATIVE_UINT16, H5S_ALL, H5S_ALL, H5P_DEFAULT, m_activeUt[cellIdx].data());
                    
                    // Cleanup
                    H5Dclose(mappingDataset);
                    H5Sclose(mappingDataspace);
                }
            }
        }
        
        // Dump CIR normalized delays per cell
        if (!m_cirNormDelayPerCell.empty() && m_cirNormDelayPerCell[0] != nullptr) {
            for (uint32_t cellIdx = 0; cellIdx < nCells; ++cellIdx) {
                if (m_cirNormDelayPerCell[cellIdx] != nullptr) {
                    std::string datasetName = "cirNormDelay_cell" + std::to_string(cellIdx);
                    
                    hsize_t delayDims[2] = {nUtsPerCell, N_MAX_TAPS};
                    hid_t delayDataspace = H5Screate_simple(2, delayDims, nullptr);
                    hid_t delayDataset = H5Dcreate2(cirGroup, datasetName.c_str(), H5T_NATIVE_UINT16, delayDataspace,
                                                  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                    
                    // Copy data from GPU to host if needed, or use CPU pointer directly
                    uint16_t* delayPtr = nullptr;
                    std::vector<uint16_t> hostDelayData;
                    
                    if (m_simConfig->cpu_only_mode == 0 && m_simConfig->internal_memory_mode >= 1) {
                        // GPU memory - copy to host
                        hostDelayData.resize(nUtsPerCell * N_MAX_TAPS);
                        CHECK_CUDAERROR(cudaMemcpy(hostDelayData.data(), m_cirNormDelayPerCell[cellIdx], 
                                                   nUtsPerCell * N_MAX_TAPS * sizeof(uint16_t), 
                                                   cudaMemcpyDeviceToHost));
                        delayPtr = hostDelayData.data();
                    } else {
                        // CPU memory - use pointer directly
                        delayPtr = m_cirNormDelayPerCell[cellIdx];
                    }
                    
                    H5Dwrite(delayDataset, H5T_NATIVE_UINT16, H5S_ALL, H5S_ALL, H5P_DEFAULT, delayPtr);
                    
                    H5Dclose(delayDataset);
                    H5Sclose(delayDataspace);
                }
            }
        }
        
        // Dump CIR number of taps per cell
        if (!m_cirNtapsPerCell.empty() && m_cirNtapsPerCell[0] != nullptr) {
            for (uint32_t cellIdx = 0; cellIdx < nCells; ++cellIdx) {
                if (m_cirNtapsPerCell[cellIdx] != nullptr) {
                    std::string datasetName = "cirNtaps_cell" + std::to_string(cellIdx);
                    
                    hsize_t ntapsDims[1] = {nUtsPerCell};
                    hid_t ntapsDataspace = H5Screate_simple(1, ntapsDims, nullptr);
                    hid_t ntapsDataset = H5Dcreate2(cirGroup, datasetName.c_str(), H5T_NATIVE_UINT16, ntapsDataspace,
                                                  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                    
                    // Copy data from GPU to host if needed, or use CPU pointer directly
                    uint16_t* ntapsPtr = nullptr;
                    std::vector<uint16_t> hostNtapsData;
                    
                    if (m_simConfig->cpu_only_mode == 0 && m_simConfig->internal_memory_mode >= 1) {
                        // GPU memory - copy to host
                        hostNtapsData.resize(nUtsPerCell);
                        CHECK_CUDAERROR(cudaMemcpy(hostNtapsData.data(), m_cirNtapsPerCell[cellIdx], 
                                                   nUtsPerCell * sizeof(uint16_t), 
                                                   cudaMemcpyDeviceToHost));
                        ntapsPtr = hostNtapsData.data();
                    } else {
                        // CPU memory - use pointer directly
                        ntapsPtr = m_cirNtapsPerCell[cellIdx];
                    }
                    
                    H5Dwrite(ntapsDataset, H5T_NATIVE_UINT16, H5S_ALL, H5S_ALL, H5P_DEFAULT, ntapsPtr);
                    
                    H5Dclose(ntapsDataset);
                    H5Sclose(ntapsDataspace);
                }
            }
        }
        
        H5Gclose(cirGroup);
    }

    // dump CFR on PRBG per cell
#ifdef SLS_DEBUG_
    printf("DEBUG: m_freqChanPrbgPerCell.size() = %zu\n", m_freqChanPrbgPerCell.size());
#endif
    if (!m_freqChanPrbgPerCell.empty() && m_freqChanPrbgPerCell[0] != nullptr) {
        const uint32_t nCells = static_cast<uint32_t>(m_freqChanPrbgPerCell.size());
        const uint32_t nUtsPerCell = m_topology.nUT;
        const uint32_t nPrbgCoeff = m_cmnLinkParams.nUeAnt * m_cmnLinkParams.nBsAnt * m_simConfig->n_prb;
        
        // Create group for CFR PRBG data
        hid_t cfrPrbgGroup = H5Gcreate2(slsH5File, "cfrPrbgPerCell", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        
        for (uint32_t cellIdx = 0; cellIdx < nCells; ++cellIdx) {
            if (m_freqChanPrbgPerCell[cellIdx] != nullptr) {
                std::string datasetName = "cfrPrbg_cell" + std::to_string(cellIdx);
                
                hsize_t cfrDims[2] = {nUtsPerCell, nPrbgCoeff};
                hid_t cfrDataspace = H5Screate_simple(2, cfrDims, nullptr);
                
                // Create complex datatype
                hid_t complexType = H5Tcreate(H5T_COMPOUND, sizeof(Tcomplex));
                H5Tinsert(complexType, "real", 0, H5T_NATIVE_FLOAT);
                H5Tinsert(complexType, "imag", sizeof(float), H5T_NATIVE_FLOAT);
                
                hid_t cfrDataset = H5Dcreate2(cfrPrbgGroup, datasetName.c_str(), complexType, cfrDataspace,
                                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                
                // Copy data from GPU to host if needed, or use CPU pointer directly
                Tcomplex* cfrPtr = nullptr;
                std::vector<Tcomplex> hostCfrData;
                
                if (m_simConfig->cpu_only_mode == 0 && m_simConfig->internal_memory_mode >= 1) {
                    // GPU memory - copy to host
                    hostCfrData.resize(nUtsPerCell * nPrbgCoeff);
                    CHECK_CUDAERROR(cudaMemcpy(hostCfrData.data(), m_freqChanPrbgPerCell[cellIdx], 
                                               nUtsPerCell * nPrbgCoeff * sizeof(Tcomplex), 
                                               cudaMemcpyDeviceToHost));
                    cfrPtr = hostCfrData.data();
                } else {
                    // CPU memory - use pointer directly
                    cfrPtr = m_freqChanPrbgPerCell[cellIdx];
                }
                
                H5Dwrite(cfrDataset, complexType, H5S_ALL, H5S_ALL, H5P_DEFAULT, cfrPtr);
                
                H5Dclose(cfrDataset);
                H5Sclose(cfrDataspace);
                H5Tclose(complexType);
            }
        }
        
        H5Gclose(cfrPrbgGroup);
    }

    // dump CFR on SC per cell
#ifdef SLS_DEBUG_
    printf("DEBUG: m_freqChanScPerCell.size() = %zu\n", m_freqChanScPerCell.size());
#endif
    if (!m_freqChanScPerCell.empty() && m_freqChanScPerCell[0] != nullptr) {
        const uint32_t nCells = static_cast<uint32_t>(m_freqChanScPerCell.size());
        const uint32_t nUtsPerCell = m_topology.nUT;
        uint32_t nScCoeff{};
        
        // Determine number of subcarriers based on run mode
        switch (m_simConfig->run_mode) {
            case 2:
            case 3:
                nScCoeff = m_cmnLinkParams.nUeAnt * m_cmnLinkParams.nBsAnt * m_simConfig->n_prb * 12; // PRBs * 12 SCs
                break;
            case 4:
                nScCoeff = m_cmnLinkParams.nUeAnt * m_cmnLinkParams.nBsAnt * m_simConfig->fft_size; // FFT size
                break;
            default:
                nScCoeff = 0;
                break;
        }
        
        if (nScCoeff > 0) {
            // Create group for CFR SC data
            hid_t cfrScGroup = H5Gcreate2(slsH5File, "cfrScPerCell", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            
            for (uint32_t cellIdx = 0; cellIdx < nCells; ++cellIdx) {
                if (m_freqChanScPerCell[cellIdx] != nullptr) {
                    std::string datasetName = "cfrSc_cell" + std::to_string(cellIdx);
                    
                    hsize_t cfrDims[2] = {nUtsPerCell, nScCoeff};
                    hid_t cfrDataspace = H5Screate_simple(2, cfrDims, nullptr);
                    
                    // Create complex datatype
                    hid_t complexType = H5Tcreate(H5T_COMPOUND, sizeof(Tcomplex));
                    H5Tinsert(complexType, "real", 0, H5T_NATIVE_FLOAT);
                    H5Tinsert(complexType, "imag", sizeof(float), H5T_NATIVE_FLOAT);
                    
                    hid_t cfrDataset = H5Dcreate2(cfrScGroup, datasetName.c_str(), complexType, cfrDataspace,
                                                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                    
                    // Copy data from GPU to host if needed, or use CPU pointer directly
                    Tcomplex* cfrScPtr = nullptr;
                    std::vector<Tcomplex> hostCfrData;
                    
                    if (m_simConfig->cpu_only_mode == 0 && m_simConfig->internal_memory_mode >= 1) {
                        // GPU memory - copy to host
                        hostCfrData.resize(nUtsPerCell * nScCoeff);
                        CHECK_CUDAERROR(cudaMemcpy(hostCfrData.data(), m_freqChanScPerCell[cellIdx], 
                                                   nUtsPerCell * nScCoeff * sizeof(Tcomplex), 
                                                   cudaMemcpyDeviceToHost));
                        cfrScPtr = hostCfrData.data();
                    } else {
                        // CPU memory - use pointer directly
                        cfrScPtr = m_freqChanScPerCell[cellIdx];
                    }
                    
                    H5Dwrite(cfrDataset, complexType, H5S_ALL, H5S_ALL, H5P_DEFAULT, cfrScPtr);
                    
                    H5Dclose(cfrDataset);
                    H5Sclose(cfrDataspace);
                    H5Tclose(complexType);
                }
            }
            
            H5Gclose(cfrScGroup);
        }
    }

    // dump basic configuration metadata (always available)
    {
        hid_t configGroup = H5Gcreate2(slsH5File, "configurationMetadata", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t scalarSpace = H5Screate(H5S_SCALAR);
        
        // Common link parameters (always available)
        hid_t linkGroup = H5Gcreate2(configGroup, "linkParams", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        
        hid_t nUeAntDataset = H5Dcreate2(linkGroup, "nUeAnt", H5T_NATIVE_UINT8, scalarSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(nUeAntDataset, H5T_NATIVE_UINT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, &m_cmnLinkParams.nUeAnt);
        H5Dclose(nUeAntDataset);
        
        hid_t nBsAntDataset = H5Dcreate2(linkGroup, "nBsAnt", H5T_NATIVE_UINT8, scalarSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(nBsAntDataset, H5T_NATIVE_UINT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, &m_cmnLinkParams.nBsAnt);
        H5Dclose(nBsAntDataset);
        
        H5Gclose(linkGroup);
        
        // Simulation configuration (if available)
        if (m_simConfig) {
            hid_t simGroup = H5Gcreate2(configGroup, "simulation", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            
            hid_t nPrbDataset = H5Dcreate2(simGroup, "n_prb", H5T_NATIVE_INT, scalarSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Dwrite(nPrbDataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &m_simConfig->n_prb);
            H5Dclose(nPrbDataset);
            
            hid_t fftSizeDataset = H5Dcreate2(simGroup, "fft_size", H5T_NATIVE_INT, scalarSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Dwrite(fftSizeDataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &m_simConfig->fft_size);
            H5Dclose(fftSizeDataset);
            
            hid_t centerFreqDataset = H5Dcreate2(simGroup, "center_freq_hz", H5T_NATIVE_FLOAT, scalarSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Dwrite(centerFreqDataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &m_simConfig->center_freq_hz);
            H5Dclose(centerFreqDataset);
            
            hid_t bandwidthDataset = H5Dcreate2(simGroup, "bandwidth_hz", H5T_NATIVE_FLOAT, scalarSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Dwrite(bandwidthDataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &m_simConfig->bandwidth_hz);
            H5Dclose(bandwidthDataset);
            
            hid_t runModeDataset = H5Dcreate2(simGroup, "run_mode", H5T_NATIVE_INT, scalarSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Dwrite(runModeDataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &m_simConfig->run_mode);
            H5Dclose(runModeDataset);
            
            H5Gclose(simGroup);
        }
        
        // System configuration (if available)
        if (m_sysConfig) {
            hid_t sysGroup = H5Gcreate2(configGroup, "system", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            
            hid_t isdDataset = H5Dcreate2(sysGroup, "isd", H5T_NATIVE_FLOAT, scalarSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Dwrite(isdDataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &m_sysConfig->isd);
            H5Dclose(isdDataset);
            
            hid_t nSiteDataset = H5Dcreate2(sysGroup, "n_site", H5T_NATIVE_UINT32, scalarSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Dwrite(nSiteDataset, H5T_NATIVE_UINT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, &m_sysConfig->n_site);
            H5Dclose(nSiteDataset);
            
            hid_t nUtDataset = H5Dcreate2(sysGroup, "n_ut", H5T_NATIVE_UINT32, scalarSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Dwrite(nUtDataset, H5T_NATIVE_UINT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, &m_sysConfig->n_ut);
            H5Dclose(nUtDataset);
            
            H5Gclose(sysGroup);
        }
        
        H5Sclose(scalarSpace);
        H5Gclose(configGroup);
    }

    // dump complete network topology structure
#ifdef SLS_DEBUG_
    printf("DEBUG: m_topology.cellParams.size() = %zu\n", m_topology.cellParams.size());
    printf("DEBUG: m_topology.utParams.size() = %zu\n", m_topology.utParams.size());
#endif
    
    {
        hid_t topologyGroup = H5Gcreate2(slsH5File, "topology", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t scalarSpace = H5Screate(H5S_SCALAR);
        
        // Save topology-level parameters
        hid_t nSiteDataset = H5Dcreate2(topologyGroup, "nSite", H5T_NATIVE_UINT32, scalarSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(nSiteDataset, H5T_NATIVE_UINT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, &m_topology.nSite);
        H5Dclose(nSiteDataset);
        
        hid_t nSectorDataset = H5Dcreate2(topologyGroup, "nSector", H5T_NATIVE_UINT32, scalarSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(nSectorDataset, H5T_NATIVE_UINT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, &m_topology.nSector);
        H5Dclose(nSectorDataset);
        
        hid_t nUTDataset = H5Dcreate2(topologyGroup, "nUT", H5T_NATIVE_UINT32, scalarSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(nUTDataset, H5T_NATIVE_UINT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, &m_topology.nUT);
        H5Dclose(nUTDataset);
        
        hid_t isdDataset = H5Dcreate2(topologyGroup, "ISD", H5T_NATIVE_FLOAT, scalarSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(isdDataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &m_topology.ISD);
        H5Dclose(isdDataset);
        
        hid_t bsHeightDataset = H5Dcreate2(topologyGroup, "bsHeight", H5T_NATIVE_FLOAT, scalarSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(bsHeightDataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &m_topology.bsHeight);
        H5Dclose(bsHeightDataset);
        
        hid_t minDistDataset = H5Dcreate2(topologyGroup, "minBsUeDist2d", H5T_NATIVE_FLOAT, scalarSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(minDistDataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &m_topology.minBsUeDist2d);
        H5Dclose(minDistDataset);
        
        hid_t maxDistIndoorDataset = H5Dcreate2(topologyGroup, "maxBsUeDist2dIndoor", H5T_NATIVE_FLOAT, scalarSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(maxDistIndoorDataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &m_topology.maxBsUeDist2dIndoor);
        H5Dclose(maxDistIndoorDataset);
        
        hid_t indoorPercentDataset = H5Dcreate2(topologyGroup, "indoorUtPercent", H5T_NATIVE_FLOAT, scalarSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(indoorPercentDataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &m_topology.indoorUtPercent);
        H5Dclose(indoorPercentDataset);
        
        // Add derived parameters
        const uint32_t nSectorPerSite = m_topology.nSector / m_topology.nSite;
        hid_t nSectorPerSiteDataset = H5Dcreate2(topologyGroup, "nSectorPerSite", H5T_NATIVE_UINT32, scalarSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(nSectorPerSiteDataset, H5T_NATIVE_UINT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, &nSectorPerSite);
        H5Dclose(nSectorPerSiteDataset);
        
        H5Sclose(scalarSpace);
        
        // Save cell parameters as part of topology
        if (!m_topology.cellParams.empty()) {
            // Create compound datatype for CellParam
            hid_t cellParamType = H5Tcreate(H5T_COMPOUND, sizeof(CellParam));
            H5Tinsert(cellParamType, "cid", HOFFSET(CellParam, cid), H5T_NATIVE_UINT16);
            H5Tinsert(cellParamType, "siteId", HOFFSET(CellParam, siteId), H5T_NATIVE_UINT32);
            
            // Coordinate struct
            hid_t locType = H5Tcreate(H5T_COMPOUND, sizeof(Coordinate));
            H5Tinsert(locType, "x", HOFFSET(Coordinate, x), H5T_NATIVE_FLOAT);
            H5Tinsert(locType, "y", HOFFSET(Coordinate, y), H5T_NATIVE_FLOAT);
            H5Tinsert(locType, "z", HOFFSET(Coordinate, z), H5T_NATIVE_FLOAT);
            H5Tinsert(cellParamType, "loc", HOFFSET(CellParam, loc), locType);
            
            H5Tinsert(cellParamType, "antPanelIdx", HOFFSET(CellParam, antPanelIdx), H5T_NATIVE_UINT32);

            // Create dataspace and dataset for cell parameters
            hsize_t cellDims[1] = {m_topology.cellParams.size()};
            hid_t cellDataspace = H5Screate_simple(1, cellDims, nullptr);
            hid_t cellDataset = H5Dcreate2(topologyGroup, "cellParams", cellParamType, cellDataspace,
                                         H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            
            H5Dwrite(cellDataset, cellParamType, H5S_ALL, H5S_ALL, H5P_DEFAULT, m_topology.cellParams.data());
            
            H5Dclose(cellDataset);
            H5Sclose(cellDataspace);
            H5Tclose(locType);
            H5Tclose(cellParamType);
        }

        // Save UT parameters as part of topology
        if (!m_topology.utParams.empty()) {
            // Create individual datasets for UT parameters to avoid offsetof issues with inheritance
            hid_t utGroup = H5Gcreate2(topologyGroup, "utParams", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hsize_t utCount = m_topology.utParams.size();
        hid_t utDataspace = H5Screate_simple(1, &utCount, nullptr);
        
        // Extract and save individual parameter arrays
        std::vector<uint32_t> uids;
        std::vector<float> locs_x, locs_y, locs_z;
        std::vector<uint8_t> outdoor_inds;
        std::vector<uint32_t> antPanelIdxs;
        std::vector<float> velocities_x, velocities_y, velocities_z;
        std::vector<float> d_2d_ins;
        
        uids.reserve(utCount);
        locs_x.reserve(utCount); locs_y.reserve(utCount); locs_z.reserve(utCount);
        outdoor_inds.reserve(utCount);
        antPanelIdxs.reserve(utCount);
        velocities_x.reserve(utCount); velocities_y.reserve(utCount); velocities_z.reserve(utCount);
        d_2d_ins.reserve(utCount);
        
        for (const auto& ut : m_topology.utParams) {
            uids.push_back(ut.uid);
            locs_x.push_back(ut.loc.x);
            locs_y.push_back(ut.loc.y);
            locs_z.push_back(ut.loc.z);
            outdoor_inds.push_back(ut.outdoor_ind);
            antPanelIdxs.push_back(ut.antPanelIdx);
            velocities_x.push_back(ut.velocity[0]);
            velocities_y.push_back(ut.velocity[1]);
            velocities_z.push_back(ut.velocity[2]);
            d_2d_ins.push_back(ut.d_2d_in);
        }
        
        // Create and write individual datasets
        hid_t uidDataset = H5Dcreate2(utGroup, "uid", H5T_NATIVE_UINT32, utDataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(uidDataset, H5T_NATIVE_UINT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, uids.data());
        H5Dclose(uidDataset);
        
        hid_t locXDataset = H5Dcreate2(utGroup, "loc_x", H5T_NATIVE_FLOAT, utDataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(locXDataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, locs_x.data());
        H5Dclose(locXDataset);
        
        hid_t locYDataset = H5Dcreate2(utGroup, "loc_y", H5T_NATIVE_FLOAT, utDataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(locYDataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, locs_y.data());
        H5Dclose(locYDataset);
        
        hid_t locZDataset = H5Dcreate2(utGroup, "loc_z", H5T_NATIVE_FLOAT, utDataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(locZDataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, locs_z.data());
        H5Dclose(locZDataset);
        
        hid_t outdoorDataset = H5Dcreate2(utGroup, "outdoor_ind", H5T_NATIVE_UINT8, utDataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(outdoorDataset, H5T_NATIVE_UINT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, outdoor_inds.data());
        H5Dclose(outdoorDataset);
        
        hid_t antPanelDataset = H5Dcreate2(utGroup, "antPanelIdx", H5T_NATIVE_UINT32, utDataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(antPanelDataset, H5T_NATIVE_UINT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, antPanelIdxs.data());
        H5Dclose(antPanelDataset);
        
        hid_t velXDataset = H5Dcreate2(utGroup, "velocity_x", H5T_NATIVE_FLOAT, utDataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(velXDataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, velocities_x.data());
        H5Dclose(velXDataset);
        
        hid_t velYDataset = H5Dcreate2(utGroup, "velocity_y", H5T_NATIVE_FLOAT, utDataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(velYDataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, velocities_y.data());
        H5Dclose(velYDataset);
        
        hid_t velZDataset = H5Dcreate2(utGroup, "velocity_z", H5T_NATIVE_FLOAT, utDataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(velZDataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, velocities_z.data());
        H5Dclose(velZDataset);
        
        hid_t d2dDataset = H5Dcreate2(utGroup, "d_2d_in", H5T_NATIVE_FLOAT, utDataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(d2dDataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_2d_ins.data());
        H5Dclose(d2dDataset);
        
            H5Sclose(utDataspace);
            H5Gclose(utGroup);
        }
        
        H5Gclose(topologyGroup);
    }

    // dump simulation and system configuration
    {
        hid_t configGroup = H5Gcreate2(slsH5File, "configuration", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t scalarSpace = H5Screate(H5S_SCALAR);
        
        // Simulation config
        if (m_simConfig) {
            hid_t simGroup = H5Gcreate2(configGroup, "simulation", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            
            hid_t nPrbDataset = H5Dcreate2(simGroup, "n_prb", H5T_NATIVE_INT, scalarSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Dwrite(nPrbDataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &m_simConfig->n_prb);
            H5Dclose(nPrbDataset);
            
            hid_t fftSizeDataset = H5Dcreate2(simGroup, "fft_size", H5T_NATIVE_INT, scalarSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Dwrite(fftSizeDataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &m_simConfig->fft_size);
            H5Dclose(fftSizeDataset);
            
            hid_t centerFreqDataset = H5Dcreate2(simGroup, "center_freq_hz", H5T_NATIVE_FLOAT, scalarSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Dwrite(centerFreqDataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &m_simConfig->center_freq_hz);
            H5Dclose(centerFreqDataset);
            
            hid_t bandwidthDataset = H5Dcreate2(simGroup, "bandwidth_hz", H5T_NATIVE_FLOAT, scalarSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Dwrite(bandwidthDataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &m_simConfig->bandwidth_hz);
            H5Dclose(bandwidthDataset);
            
            hid_t runModeDataset = H5Dcreate2(simGroup, "run_mode", H5T_NATIVE_INT, scalarSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Dwrite(runModeDataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &m_simConfig->run_mode);
            H5Dclose(runModeDataset);
            
            hid_t memModeDataset = H5Dcreate2(simGroup, "internal_memory_mode", H5T_NATIVE_INT, scalarSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Dwrite(memModeDataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &m_simConfig->internal_memory_mode);
            H5Dclose(memModeDataset);
            
            H5Gclose(simGroup);
        }
        
        // System config
        if (m_sysConfig) {
            hid_t sysGroup = H5Gcreate2(configGroup, "system", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            
            // Save system level config parameters
            hid_t isdDataset = H5Dcreate2(sysGroup, "isd", H5T_NATIVE_FLOAT, scalarSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Dwrite(isdDataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &m_sysConfig->isd);
            H5Dclose(isdDataset);
            
            hid_t nSiteDataset = H5Dcreate2(sysGroup, "n_site", H5T_NATIVE_UINT32, scalarSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Dwrite(nSiteDataset, H5T_NATIVE_UINT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, &m_sysConfig->n_site);
            H5Dclose(nSiteDataset);
            
            hid_t nUtDataset = H5Dcreate2(sysGroup, "n_ut", H5T_NATIVE_UINT32, scalarSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Dwrite(nUtDataset, H5T_NATIVE_UINT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, &m_sysConfig->n_ut);
            H5Dclose(nUtDataset);
            
            H5Gclose(sysGroup);
        }
        
        H5Sclose(scalarSpace);
        H5Gclose(configGroup);
    }

    // dump runtime metadata
    {
        hid_t runtimeGroup = H5Gcreate2(slsH5File, "runtimeMetadata", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t scalarSpace = H5Screate(H5S_SCALAR);
        
        hid_t refTimeDataset = H5Dcreate2(runtimeGroup, "refTime", H5T_NATIVE_FLOAT, scalarSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(refTimeDataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &m_refTime);
        H5Dclose(refTimeDataset);
        
        hid_t randSeedDataset = H5Dcreate2(runtimeGroup, "randSeed", H5T_NATIVE_UINT32, scalarSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(randSeedDataset, H5T_NATIVE_UINT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, &m_randSeed);
        H5Dclose(randSeedDataset);
        
        hid_t allocLinksDataset = H5Dcreate2(runtimeGroup, "lastAllocatedLinks", H5T_NATIVE_UINT64, scalarSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        uint64_t allocLinks = static_cast<uint64_t>(m_lastAllocatedLinks);
        H5Dwrite(allocLinksDataset, H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, &allocLinks);
        H5Dclose(allocLinksDataset);
        
        hid_t allocActiveLinksDataset = H5Dcreate2(runtimeGroup, "lastAllocatedActiveLinks", H5T_NATIVE_UINT64, scalarSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        uint64_t allocActiveLinks = static_cast<uint64_t>(m_lastAllocatedActiveLinks);
        H5Dwrite(allocActiveLinksDataset, H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, &allocActiveLinks);
        H5Dclose(allocActiveLinksDataset);
        
        // Spatial bounds
        hid_t maxXDataset = H5Dcreate2(runtimeGroup, "maxX", H5T_NATIVE_FLOAT, scalarSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(maxXDataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &m_maxX);
        H5Dclose(maxXDataset);
        
        hid_t minXDataset = H5Dcreate2(runtimeGroup, "minX", H5T_NATIVE_FLOAT, scalarSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(minXDataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &m_minX);
        H5Dclose(minXDataset);
        
        hid_t maxYDataset = H5Dcreate2(runtimeGroup, "maxY", H5T_NATIVE_FLOAT, scalarSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(maxYDataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &m_maxY);
        H5Dclose(maxYDataset);
        
        hid_t minYDataset = H5Dcreate2(runtimeGroup, "minY", H5T_NATIVE_FLOAT, scalarSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(minYDataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &m_minY);
        H5Dclose(minYDataset);
        
        H5Sclose(scalarSpace);
        H5Gclose(runtimeGroup);
    }

    // dump common link parameters (critical for analysis)
    {
        hid_t cmnLinkGroup = H5Gcreate2(slsH5File, "commonLinkParams", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t scalarSpace = H5Screate(H5S_SCALAR);
        
        // Array dataspace for [3] arrays (NLOS, LOS, O2I)
        hsize_t arrayDims[1] = {3};
        hid_t arraySpace = H5Screate_simple(1, arrayDims, nullptr);
        
        // Large-scale parameters
        hid_t muLgDsDataset = H5Dcreate2(cmnLinkGroup, "mu_lgDS", H5T_NATIVE_FLOAT, arraySpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(muLgDsDataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, m_cmnLinkParams.mu_lgDS);
        H5Dclose(muLgDsDataset);
        
        hid_t sigmaLgDsDataset = H5Dcreate2(cmnLinkGroup, "sigma_lgDS", H5T_NATIVE_FLOAT, arraySpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(sigmaLgDsDataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, m_cmnLinkParams.sigma_lgDS);
        H5Dclose(sigmaLgDsDataset);
        
        hid_t muLgAsdDataset = H5Dcreate2(cmnLinkGroup, "mu_lgASD", H5T_NATIVE_FLOAT, arraySpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(muLgAsdDataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, m_cmnLinkParams.mu_lgASD);
        H5Dclose(muLgAsdDataset);
        
        hid_t muLgAsaDataset = H5Dcreate2(cmnLinkGroup, "mu_lgASA", H5T_NATIVE_FLOAT, arraySpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(muLgAsaDataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, m_cmnLinkParams.mu_lgASA);
        H5Dclose(muLgAsaDataset);
        
        hid_t muLgZsaDataset = H5Dcreate2(cmnLinkGroup, "mu_lgZSA", H5T_NATIVE_FLOAT, arraySpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(muLgZsaDataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, m_cmnLinkParams.mu_lgZSA);
        H5Dclose(muLgZsaDataset);
        
        // Key system parameters
        hid_t nLinkDataset = H5Dcreate2(cmnLinkGroup, "nLink", H5T_NATIVE_UINT32, scalarSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(nLinkDataset, H5T_NATIVE_UINT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, &m_cmnLinkParams.nLink);
        H5Dclose(nLinkDataset);
        
        hid_t nUeAntDataset = H5Dcreate2(cmnLinkGroup, "nUeAnt", H5T_NATIVE_UINT32, scalarSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(nUeAntDataset, H5T_NATIVE_UINT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, &m_cmnLinkParams.nUeAnt);
        H5Dclose(nUeAntDataset);
        
        hid_t nBsAntDataset = H5Dcreate2(cmnLinkGroup, "nBsAnt", H5T_NATIVE_UINT32, scalarSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(nBsAntDataset, H5T_NATIVE_UINT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, &m_cmnLinkParams.nBsAnt);
        H5Dclose(nBsAntDataset);
        
        hid_t lambda0Dataset = H5Dcreate2(cmnLinkGroup, "lambda_0", H5T_NATIVE_FLOAT, scalarSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(lambda0Dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &m_cmnLinkParams.lambda_0);
        H5Dclose(lambda0Dataset);
        
        H5Sclose(arraySpace);
        H5Sclose(scalarSpace);
        H5Gclose(cmnLinkGroup);
    }
    
    // Close HDF5 file
    // Dump SystemLevelConfig
    if (m_sysConfig != nullptr) {
        // Create compound datatype for SystemLevelConfig
        hid_t sysConfigType = H5Tcreate(H5T_COMPOUND, sizeof(SystemLevelConfig));
        
        // Add enum for Scenario
        hid_t scenarioEnumType = H5Tenum_create(H5T_NATIVE_INT);
        int scenarioVal;
        scenarioVal = static_cast<int>(Scenario::UMa);
        H5Tenum_insert(scenarioEnumType, "UMa", &scenarioVal);
        scenarioVal = static_cast<int>(Scenario::UMi);
        H5Tenum_insert(scenarioEnumType, "UMi", &scenarioVal);
        scenarioVal = static_cast<int>(Scenario::RMa);
        H5Tenum_insert(scenarioEnumType, "RMa", &scenarioVal);
        
        // Insert SystemLevelConfig fields
        H5Tinsert(sysConfigType, "scenario", HOFFSET(SystemLevelConfig, scenario), scenarioEnumType);
        H5Tinsert(sysConfigType, "isd", HOFFSET(SystemLevelConfig, isd), H5T_NATIVE_FLOAT);
        H5Tinsert(sysConfigType, "n_site", HOFFSET(SystemLevelConfig, n_site), H5T_NATIVE_UINT32);
        H5Tinsert(sysConfigType, "n_sector_per_site", HOFFSET(SystemLevelConfig, n_sector_per_site), H5T_NATIVE_UINT8);
        H5Tinsert(sysConfigType, "n_ut", HOFFSET(SystemLevelConfig, n_ut), H5T_NATIVE_UINT32);
        H5Tinsert(sysConfigType, "optional_pl_ind", HOFFSET(SystemLevelConfig, optional_pl_ind), H5T_NATIVE_UINT8);
        H5Tinsert(sysConfigType, "o2i_building_penetr_loss_ind", HOFFSET(SystemLevelConfig, o2i_building_penetr_loss_ind), H5T_NATIVE_UINT8);
        H5Tinsert(sysConfigType, "o2i_car_penetr_loss_ind", HOFFSET(SystemLevelConfig, o2i_car_penetr_loss_ind), H5T_NATIVE_UINT8);
        H5Tinsert(sysConfigType, "enable_near_field_effect", HOFFSET(SystemLevelConfig, enable_near_field_effect), H5T_NATIVE_UINT8);
        H5Tinsert(sysConfigType, "enable_non_stationarity", HOFFSET(SystemLevelConfig, enable_non_stationarity), H5T_NATIVE_UINT8);
        
        // Add array types for force_los_prob and force_ut_speed
        hsize_t arrayDim2[1] = {2};
        hid_t floatArray2Type = H5Tarray_create2(H5T_NATIVE_FLOAT, 1, arrayDim2);
        H5Tinsert(sysConfigType, "force_los_prob", HOFFSET(SystemLevelConfig, force_los_prob), floatArray2Type);
        H5Tinsert(sysConfigType, "force_ut_speed", HOFFSET(SystemLevelConfig, force_ut_speed), floatArray2Type);
        
        H5Tinsert(sysConfigType, "force_indoor_ratio", HOFFSET(SystemLevelConfig, force_indoor_ratio), H5T_NATIVE_FLOAT);
        H5Tinsert(sysConfigType, "disable_pl_shadowing", HOFFSET(SystemLevelConfig, disable_pl_shadowing), H5T_NATIVE_UINT8);
        H5Tinsert(sysConfigType, "disable_small_scale_fading", HOFFSET(SystemLevelConfig, disable_small_scale_fading), H5T_NATIVE_UINT8);
        H5Tinsert(sysConfigType, "enable_per_tti_lsp", HOFFSET(SystemLevelConfig, enable_per_tti_lsp), H5T_NATIVE_UINT8);
        H5Tinsert(sysConfigType, "enable_propagation_delay", HOFFSET(SystemLevelConfig, enable_propagation_delay), H5T_NATIVE_UINT8);

        // Create dataspace and dataset for SystemLevelConfig (single instance)
        hsize_t sysConfigDims[1] = {1};
        hid_t sysConfigDataspace = H5Screate_simple(1, sysConfigDims, nullptr);
        hid_t sysConfigDataset = H5Dcreate2(slsH5File, "systemLevelConfig", sysConfigType, sysConfigDataspace,
                                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        
        // Write SystemLevelConfig data
        H5Dwrite(sysConfigDataset, sysConfigType, H5S_ALL, H5S_ALL, H5P_DEFAULT, m_sysConfig);
        
        // Cleanup
        H5Dclose(sysConfigDataset);
        H5Sclose(sysConfigDataspace);
        H5Tclose(floatArray2Type);
        H5Tclose(scenarioEnumType);
        H5Tclose(sysConfigType);
    }

    // Dump SimConfig
    if (m_simConfig != nullptr) {
        // Create compound datatype for SimConfig
        hid_t simConfigType = H5Tcreate(H5T_COMPOUND, sizeof(SimConfig));
        
        // Insert SimConfig fields (note: skipping tx_sig_in pointer as it's not serializable)
        H5Tinsert(simConfigType, "link_sim_ind", HOFFSET(SimConfig, link_sim_ind), H5T_NATIVE_INT);
        H5Tinsert(simConfigType, "center_freq_hz", HOFFSET(SimConfig, center_freq_hz), H5T_NATIVE_FLOAT);
        H5Tinsert(simConfigType, "bandwidth_hz", HOFFSET(SimConfig, bandwidth_hz), H5T_NATIVE_FLOAT);
        H5Tinsert(simConfigType, "sc_spacing_hz", HOFFSET(SimConfig, sc_spacing_hz), H5T_NATIVE_FLOAT);
        H5Tinsert(simConfigType, "fft_size", HOFFSET(SimConfig, fft_size), H5T_NATIVE_INT);
        H5Tinsert(simConfigType, "n_prb", HOFFSET(SimConfig, n_prb), H5T_NATIVE_INT);
        H5Tinsert(simConfigType, "n_prbg", HOFFSET(SimConfig, n_prbg), H5T_NATIVE_INT);
        H5Tinsert(simConfigType, "n_snapshot_per_slot", HOFFSET(SimConfig, n_snapshot_per_slot), H5T_NATIVE_INT);
        H5Tinsert(simConfigType, "run_mode", HOFFSET(SimConfig, run_mode), H5T_NATIVE_INT);
        H5Tinsert(simConfigType, "internal_memory_mode", HOFFSET(SimConfig, internal_memory_mode), H5T_NATIVE_INT);
        H5Tinsert(simConfigType, "freq_convert_type", HOFFSET(SimConfig, freq_convert_type), H5T_NATIVE_INT);
        H5Tinsert(simConfigType, "sc_sampling", HOFFSET(SimConfig, sc_sampling), H5T_NATIVE_INT);
        H5Tinsert(simConfigType, "proc_sig_freq", HOFFSET(SimConfig, proc_sig_freq), H5T_NATIVE_INT);

        // Create dataspace and dataset for SimConfig (single instance)
        hsize_t simConfigDims[1] = {1};
        hid_t simConfigDataspace = H5Screate_simple(1, simConfigDims, nullptr);
        hid_t simConfigDataset = H5Dcreate2(slsH5File, "simConfig", simConfigType, simConfigDataspace,
                                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        
        // Write SimConfig data
        H5Dwrite(simConfigDataset, simConfigType, H5S_ALL, H5S_ALL, H5P_DEFAULT, m_simConfig);
        
        // Cleanup
        H5Dclose(simConfigDataset);
        H5Sclose(simConfigDataspace);
        H5Tclose(simConfigType);
    }

    // dump antenna panel configurations
    if (m_antPanelConfig && !m_antPanelConfig->empty()) {
        // Create compound datatype for AntPanelConfig
        hid_t antPanelType = H5Tcreate(H5T_COMPOUND, sizeof(AntPanelConfig));
        
        // Basic fields
        H5Tinsert(antPanelType, "nAnt", HOFFSET(AntPanelConfig, nAnt), H5T_NATIVE_UINT16);
        H5Tinsert(antPanelType, "antModel", HOFFSET(AntPanelConfig, antModel), H5T_NATIVE_UINT8);
        
        // Array types for antenna configuration
        hsize_t antSizeDims[1] = {5};
        hid_t antSizeArrayType = H5Tarray_create2(H5T_NATIVE_UINT16, 1, antSizeDims);
        H5Tinsert(antPanelType, "antSize", HOFFSET(AntPanelConfig, antSize), antSizeArrayType);
        
        hsize_t antSpacingDims[1] = {4};
        hid_t antSpacingArrayType = H5Tarray_create2(H5T_NATIVE_FLOAT, 1, antSpacingDims);
        H5Tinsert(antPanelType, "antSpacing", HOFFSET(AntPanelConfig, antSpacing), antSpacingArrayType);
        
        hsize_t antPolarAnglesDims[1] = {2};
        hid_t antPolarAnglesArrayType = H5Tarray_create2(H5T_NATIVE_FLOAT, 1, antPolarAnglesDims);
        H5Tinsert(antPanelType, "antPolarAngles", HOFFSET(AntPanelConfig, antPolarAngles), antPolarAnglesArrayType);
        
        // Antenna pattern arrays
        hsize_t antThetaDims[1] = {181};
        hid_t antThetaArrayType = H5Tarray_create2(H5T_NATIVE_FLOAT, 1, antThetaDims);
        H5Tinsert(antPanelType, "antTheta", HOFFSET(AntPanelConfig, antTheta), antThetaArrayType);
        
        hsize_t antPhiDims[1] = {360};
        hid_t antPhiArrayType = H5Tarray_create2(H5T_NATIVE_FLOAT, 1, antPhiDims);
        H5Tinsert(antPanelType, "antPhi", HOFFSET(AntPanelConfig, antPhi), antPhiArrayType);
        
        // Create dataspace and dataset for antenna panels
        hsize_t antPanelDims[1] = {m_antPanelConfig->size()};
        hid_t antPanelDataspace = H5Screate_simple(1, antPanelDims, nullptr);
        hid_t antPanelDataset = H5Dcreate2(slsH5File, "antennaPanels", antPanelType, antPanelDataspace,
                                         H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        
        // Write antenna panel data
        H5Dwrite(antPanelDataset, antPanelType, H5S_ALL, H5S_ALL, H5P_DEFAULT, m_antPanelConfig->data());
        
        // Close resources
        H5Dclose(antPanelDataset);
        H5Sclose(antPanelDataspace);
        H5Tclose(antPhiArrayType);
        H5Tclose(antThetaArrayType);
        H5Tclose(antPolarAnglesArrayType);
        H5Tclose(antSpacingArrayType);
        H5Tclose(antSizeArrayType);
        H5Tclose(antPanelType);
    }

    H5Fclose(slsH5File);
    printf("SLS channel data saved to: %s\n", outFilename.c_str());
}



template <typename Tscalar, typename Tcomplex>
slsChan<Tscalar, Tcomplex>::~slsChan()
{
    // Cleanup resources based on runtime mode
    const bool cpuOnly = (m_simConfig && m_simConfig->cpu_only_mode == 1);
    if (!cpuOnly) {
        // Free static GPU memory
        if (m_d_cellParams) cudaFree(m_d_cellParams);
        if (m_d_utParams) cudaFree(m_d_utParams);
        if (m_d_sysConfig) cudaFree(m_d_sysConfig);
        if (m_d_simConfig) cudaFree(m_d_simConfig);
        if (m_d_cmnLinkParams) cudaFree(m_d_cmnLinkParams);
        if (m_d_linkParams) cudaFree(m_d_linkParams);
        if (m_d_clusterParams) cudaFree(m_d_clusterParams);
        if (m_d_antPanelConfigs) cudaFree(m_d_antPanelConfigs);
        if (m_d_activeLinkParams) cudaFree(m_d_activeLinkParams);
        // Clean up CRN arrays
        if (m_d_crnLos) {
            float* losGrids[7];
            CHECK_CUDAERROR(cudaMemcpy(losGrids, m_d_crnLos, 7 * sizeof(float*), cudaMemcpyDeviceToHost));
            for (int i = 0; i < 7; i++) {
                if (losGrids[i]) cudaFree(losGrids[i]);
            }
            cudaFree(m_d_crnLos);
        }
        if (m_d_crnNlos) {
            float* nlosGrids[6];
            CHECK_CUDAERROR(cudaMemcpy(nlosGrids, m_d_crnNlos, 6 * sizeof(float*), cudaMemcpyDeviceToHost));
            for (int i = 0; i < 6; i++) {
                if (nlosGrids[i]) cudaFree(nlosGrids[i]);
            }
            cudaFree(m_d_crnNlos);
        }
        if (m_d_crnO2i) {
            float* o2iGrids[6];
            CHECK_CUDAERROR(cudaMemcpy(o2iGrids, m_d_crnO2i, 6 * sizeof(float*), cudaMemcpyDeviceToHost));
            for (int i = 0; i < 6; i++) {
                if (o2iGrids[i]) cudaFree(o2iGrids[i]);
            }
            cudaFree(m_d_crnO2i);
        }
        if (m_d_corrDistLos) cudaFree(m_d_corrDistLos);
        if (m_d_corrDistNlos) cudaFree(m_d_corrDistNlos);
        if (m_d_corrDistO2i) cudaFree(m_d_corrDistO2i);
        if (m_d_curandStates) cudaFree(m_d_curandStates);
        if (m_simConfig) {
            if (m_simConfig->internal_memory_mode >= 1) {
                if (m_cirCoe) cudaFree(m_cirCoe);
                if (m_cirNormDelay) cudaFree(m_cirNormDelay);
                if (m_cirNtaps) cudaFree(m_cirNtaps);
            }
            if (m_simConfig->internal_memory_mode == 2) {
                if (m_freqChanPrbg) cudaFree(m_freqChanPrbg);
                if (m_freqChanSc) cudaFree(m_freqChanSc);
            }
            if (m_rxSigOut) cudaFree(m_rxSigOut);
        }
    } else {
        // CPU-only: free host allocations if any
        if (m_simConfig) {
            if (m_simConfig->internal_memory_mode >= 1) {
                if (m_cirCoe) free(m_cirCoe);
                if (m_cirNormDelay) free(m_cirNormDelay);
                if (m_cirNtaps) free(m_cirNtaps);
            }
            if (m_simConfig->internal_memory_mode == 2) {
                if (m_freqChanPrbg) free(m_freqChanPrbg);
                if (m_freqChanSc) free(m_freqChanSc);
            }
            if (m_rxSigOut) free(m_rxSigOut);
        }
    }
}


template <typename Tscalar, typename Tcomplex>
void slsChan<Tscalar, Tcomplex>::allocateStaticGpuMem()
{
    if (m_simConfig && m_simConfig->cpu_only_mode == 1) {
        // Initialize additional GPU pointers to nullptr for CPU-only mode
        m_d_cellParams = nullptr;
        m_d_utParams = nullptr;
        m_d_sysConfig = nullptr;
        m_d_simConfig = nullptr;
        m_d_cmnLinkParams = nullptr;
        m_d_linkParams = nullptr;
        m_d_clusterParams = nullptr;
        m_d_antPanelConfigs = nullptr;
        m_d_activeLinkParams = nullptr;
        m_d_crnLos = nullptr;
        m_d_crnNlos = nullptr;
        m_d_crnO2i = nullptr;
        m_d_curandStates = nullptr;
        m_maxCurandStates = 0;
        return;
    }
    // Allocate additional GPU memory for static allocations used in calLinkParamGPU
    CHECK_CUDAERROR(cudaMalloc((void**)&m_d_cellParams, m_topology.cellParams.size() * sizeof(CellParam)));
    CHECK_CUDAERROR(cudaMalloc((void**)&m_d_utParams, m_topology.utParams.size() * sizeof(UtParam)));
    CHECK_CUDAERROR(cudaMalloc((void**)&m_d_sysConfig, sizeof(SystemLevelConfig)));
    CHECK_CUDAERROR(cudaMalloc((void**)&m_d_simConfig, sizeof(SimConfig)));
    CHECK_CUDAERROR(cudaMalloc((void**)&m_d_cmnLinkParams, sizeof(CmnLinkParams)));
    // Allocate for site-UT pairs (co-sited sectors share link parameters)
    CHECK_CUDAERROR(cudaMalloc((void**)&m_d_linkParams, m_topology.nSite * m_sysConfig->n_ut * sizeof(LinkParams)));
    // Allocate cluster parameters for site-UT pairs (co-sited sectors share cluster parameters)
    CHECK_CUDAERROR(cudaMalloc((void**)&m_d_clusterParams, m_topology.nSite * m_sysConfig->n_ut * sizeof(ClusterParams)));
    
    // Copy static data to GPU
    CHECK_CUDAERROR(cudaMemcpyAsync(m_d_cellParams, m_topology.cellParams.data(), m_topology.cellParams.size() * sizeof(CellParam), cudaMemcpyHostToDevice, m_strm));
    CHECK_CUDAERROR(cudaMemcpyAsync(m_d_utParams, m_topology.utParams.data(), m_topology.utParams.size() * sizeof(UtParam), cudaMemcpyHostToDevice, m_strm));
    CHECK_CUDAERROR(cudaMemcpyAsync(m_d_sysConfig, m_sysConfig, sizeof(SystemLevelConfig), cudaMemcpyHostToDevice, m_strm));
    CHECK_CUDAERROR(cudaMemcpyAsync(m_d_simConfig, m_simConfig, sizeof(SimConfig), cudaMemcpyHostToDevice, m_strm));

    // Allocate additional memory for small-scale functions
    CHECK_CUDAERROR(cudaMalloc((void**)&m_d_antPanelConfigs, m_antPanelConfig->size() * sizeof(AntPanelConfig)));
    // Note: m_d_activeLinkParams will be allocated dynamically as its size depends on active links
    m_d_activeLinkParams = nullptr;
    
    // Initialize correlation distance pointers
    m_d_corrDistLos = nullptr;
    m_d_corrDistNlos = nullptr;
    m_d_corrDistO2i = nullptr;
    
    // Initialize CRN allocated size tracker
    const uint16_t nSite = m_topology.nSite;
    m_crnAllocatedNSite = nSite;
    m_crnGridsAllocated = false;  // Individual grids will be allocated on first generateCRNGPU() call
    
    // Calculate CRN grid size based on spatial bounds (similar to CPU code)
    m_crnGridSize = (uint32_t)((m_maxX - m_minX + 1.0f) * (m_maxY - m_minY + 1.0f));
    
    // Allocate GPU memory for CRN pointer arrays with proper nSite multiplier
    CHECK_CUDAERROR(cudaMalloc((void**)&m_d_crnLos, nSite * 7 * sizeof(float*)));   // nSite * 7 LSPs for LOS
    CHECK_CUDAERROR(cudaMalloc((void**)&m_d_crnNlos, nSite * 6 * sizeof(float*)));  // nSite * 6 LSPs for NLOS (no K)
    CHECK_CUDAERROR(cudaMalloc((void**)&m_d_crnO2i, nSite * 6 * sizeof(float*)));   // nSite * 6 LSPs for O2I (no K)
    
    // Initialize CRN seed
    m_crnSeed = m_randSeed;  // Fixed seed for reproducible results
    
    // Allocate GPU memory for correlation distances
    CHECK_CUDAERROR(cudaMalloc((void**)&m_d_corrDistLos, 7 * sizeof(float)));
    CHECK_CUDAERROR(cudaMalloc((void**)&m_d_corrDistNlos, 6 * sizeof(float)));
    CHECK_CUDAERROR(cudaMalloc((void**)&m_d_corrDistO2i, 6 * sizeof(float)));
    
    // Copy antenna panel config to GPU
    CHECK_CUDAERROR(cudaMemcpyAsync(m_d_antPanelConfigs, m_antPanelConfig->data(), m_antPanelConfig->size() * sizeof(AntPanelConfig), cudaMemcpyHostToDevice, m_strm));
    
    // Calculate and allocate universal curandState array
    // Based on maximum grid configurations across all kernels:
    // - Large-scale kernels: nSite * (nUT/threadsPerBlock + 1) blocks * threadsPerBlock
    // - Small-scale kernels: nActiveLinks * nSnapshots * maxThreadsPerBlock
    // - CRN generation kernels: Calculate dynamically based on actual grid dimensions
    const uint32_t maxThreadsPerKernel = 1024;
    
    // Calculate actual CRN thread requirements dynamically
    // Use maximum correlation distance from an scenario
    float maxCorrDist = 0.0f;
    switch (m_sysConfig->scenario) {
        case Scenario::UMa:
            maxCorrDist = 50.0f;
            break;
        case Scenario::UMi:
            maxCorrDist = 15.0f;
            break;
        case Scenario::RMa:
            maxCorrDist = 120.0f;
            break;
        default:
            fprintf(stderr, "ERROR: Invalid scenario: %d\n", m_sysConfig->scenario);
            exit(EXIT_FAILURE);
            break;
    }
    
    // Calculate blocks needed with fixed maximum blocks and threads per block
    // Use fixed resources and calculate elements per thread dynamically
    const int maxCrnBlocks = 128;  // Maximum blocks for CRN kernels (can be adjusted)
    const int threadsPerBlock = 256;  // Fixed threads per block (can be adjusted)
    
    uint32_t maxCrnThreads = maxCrnBlocks * threadsPerBlock;
    
#ifdef SLS_DEBUG_
    // Calculate debug info only when needed
    float D = 3.0f * maxCorrDist;
    int paddedNX = (int)roundf(m_maxX - m_minX + 1.0f + 2.0f * D);
    int paddedNY = (int)roundf(m_maxY - m_minY + 1.0f + 2.0f * D);
    int L = (maxCorrDist == 0.0f) ? 1 : (2 * (int)D + 1);
    int finalNX = paddedNX - L + 1;
    int finalNY = paddedNY - L + 1;
    int totalElements = finalNX * finalNY;
    int totalThreads = maxCrnBlocks * threadsPerBlock;
    int elementsPerThread = (totalElements + totalThreads - 1) / totalThreads;
    printf("DEBUG: CRN thread calculation - Grid: %dx%d (%d elements) -> MaxBlocks: %d -> ThreadsPerBlock: %d -> ElementsPerThread: %d -> TotalThreads: %u\n",
           finalNX, finalNY, totalElements, maxCrnBlocks, threadsPerBlock, elementsPerThread, maxCrnThreads);
#endif
    
    const uint32_t maxBlocks = std::max({
        m_topology.nSite * ((m_topology.nUT + 511) / 512),  // Large-scale kernels  
        static_cast<uint32_t>(m_topology.nSite * m_topology.nUT),  // Small-scale active links estimate
        static_cast<uint32_t>(m_simConfig->n_snapshot_per_slot),  // Snapshot dimension
        (maxCrnThreads + maxThreadsPerKernel - 1) / maxThreadsPerKernel  // CRN threads converted to blocks
    });
    m_maxCurandStates = maxBlocks * maxThreadsPerKernel;
    
    // Allocate curandState array
    CHECK_CUDAERROR(cudaMalloc((void**)&m_d_curandStates, m_maxCurandStates * sizeof(curandState)));
    
    // Initialize curandState array with unique seeds
    const uint32_t numBlocks = (m_maxCurandStates + (uint32_t)threadsPerBlock - 1) / (uint32_t)threadsPerBlock;
    initCurandStatesKernel<<<numBlocks, (uint32_t)threadsPerBlock, 0, m_strm>>>(
        m_d_curandStates, m_maxCurandStates, m_randSeed);
    CHECK_CUDAERROR(cudaGetLastError());
    
#ifdef SLS_DEBUG_
    printf("DEBUG: Allocated %u curandStates (%.2f MB)\n", 
           m_maxCurandStates, m_maxCurandStates * sizeof(curandState) / (1024.0f * 1024.0f));
#endif
}

template <typename Tscalar, typename Tcomplex>
void slsChan<Tscalar, Tcomplex>::allocateDynamicGpuMem(uint32_t nLinks)
{
    // only allocated for internal memory allocation if existing are not sufficient
    if (nLinks <= m_lastAllocatedLinks) {
        return;
    }
    m_lastAllocatedLinks = nLinks;

    // Early return for CPU-only mode - skip all GPU allocations
    if (m_simConfig && m_simConfig->cpu_only_mode == 1) {
        // Allocate CIR memory for modes 1 and 2 (internal CIR)
        if (m_simConfig->internal_memory_mode >= 1) {
            // Reallocate cirCoe with NULL check using contiguous layout: (total_active_links, n_snapshots, n_ut_ant, n_bs_ant, max_taps)
            Tcomplex* temp_cirCoe = (Tcomplex*)realloc(m_cirCoe, m_lastAllocatedLinks *
                                m_simConfig->n_snapshot_per_slot *
                                m_cmnLinkParams.nUeAnt * 
                                m_cmnLinkParams.nBsAnt * 
                                N_MAX_TAPS *
                                sizeof(Tcomplex));
            if (temp_cirCoe == nullptr) {
                fprintf(stderr, "Failed to reallocate m_cirCoe\n");
                return;
            }
            m_cirCoe = temp_cirCoe;

            // Reallocate cirNormDelay with NULL check
            uint16_t* temp_cirNormDelay = (uint16_t*)realloc(m_cirNormDelay, m_lastAllocatedLinks * N_MAX_TAPS * sizeof(uint16_t));
            if (temp_cirNormDelay == nullptr) {
                fprintf(stderr, "Failed to reallocate m_cirNormDelay\n");
                return;
            }
            m_cirNormDelay = temp_cirNormDelay;

            // Reallocate cirNtaps with NULL check
            uint16_t* temp_cirNtaps = (uint16_t*)realloc(m_cirNtaps, m_lastAllocatedLinks * sizeof(uint16_t));
            if (temp_cirNtaps == nullptr) {
                fprintf(stderr, "Failed to reallocate m_cirNtaps\n");
                return;
            }
            m_cirNtaps = temp_cirNtaps;
            
            // Initialize cirNtaps to zero
            memset(m_cirNtaps, 0, m_lastAllocatedLinks * sizeof(uint16_t));
        }
    
        // Allocate CFR memory for mode 2 only (internal CFR)
        if (m_simConfig->internal_memory_mode == 2) {
            if (m_simConfig->run_mode == 0) {
                m_freqChanPrbg = nullptr;
                m_freqChanSc = nullptr;
            }
            else if (m_simConfig->run_mode == 1) {
                // Reallocate freqChanPrbg with NULL check
                Tcomplex* temp_freqChanPrbg = (Tcomplex*)realloc(m_freqChanPrbg, m_lastAllocatedLinks *
                                m_simConfig->n_snapshot_per_slot *
                                m_cmnLinkParams.nUeAnt * 
                                m_cmnLinkParams.nBsAnt * 
                                m_simConfig->n_prbg *
                                sizeof(Tcomplex));
                if (temp_freqChanPrbg == nullptr) {
                    fprintf(stderr, "Failed to reallocate m_freqChanPrbg\n");
                    return;
                }
                m_freqChanPrbg = temp_freqChanPrbg;
                m_freqChanSc = nullptr;
            } else if (m_simConfig->run_mode == 2 || m_simConfig->run_mode == 3) {
                // Mode 2: CIR and CFR on SC (n_prb * 12 subcarriers) 
                // Mode 3: CIR and CFR on PRB/SC (same as mode 2)
                Tcomplex* temp_freqChanPrbg = (Tcomplex*)realloc(m_freqChanPrbg, m_lastAllocatedLinks *
                                m_simConfig->n_snapshot_per_slot *
                                m_cmnLinkParams.nUeAnt * 
                                m_cmnLinkParams.nBsAnt * 
                                m_simConfig->n_prbg *
                                sizeof(Tcomplex));
                if (temp_freqChanPrbg == nullptr) {
                    fprintf(stderr, "Failed to reallocate m_freqChanPrbg\n");
                    return;
                }
                m_freqChanPrbg = temp_freqChanPrbg;

                // Reallocate freqChanSc with NULL check
                Tcomplex* temp_freqChanSc = (Tcomplex*)realloc(m_freqChanSc, m_lastAllocatedLinks *
                                m_simConfig->n_snapshot_per_slot *
                                m_cmnLinkParams.nUeAnt * 
                                m_cmnLinkParams.nBsAnt * 
                                m_simConfig->n_prb * 12 *
                                sizeof(Tcomplex));
                if (temp_freqChanSc == nullptr) {
                    fprintf(stderr, "Failed to reallocate m_freqChanSc\n");
                    return;
                }
                m_freqChanSc = temp_freqChanSc;
            } else if (m_simConfig->run_mode == 4) {
                // Mode 4: CIR and CFR on all N_FFT subcarriers
                Tcomplex* temp_freqChanPrbg = (Tcomplex*)realloc(m_freqChanPrbg, m_lastAllocatedLinks *
                                m_simConfig->n_snapshot_per_slot *
                                m_cmnLinkParams.nUeAnt * 
                                m_cmnLinkParams.nBsAnt * 
                                m_simConfig->n_prbg *
                                sizeof(Tcomplex));
                if (temp_freqChanPrbg == nullptr) {
                    fprintf(stderr, "Failed to reallocate m_freqChanPrbg\n");
                    return;
                }
                m_freqChanPrbg = temp_freqChanPrbg;

                // Reallocate freqChanSc for N_FFT subcarriers
                Tcomplex* temp_freqChanSc = (Tcomplex*)realloc(m_freqChanSc, m_lastAllocatedLinks *
                                m_simConfig->n_snapshot_per_slot *
                                m_cmnLinkParams.nUeAnt * 
                                m_cmnLinkParams.nBsAnt * 
                                m_simConfig->fft_size *
                                sizeof(Tcomplex));
                if (temp_freqChanSc == nullptr) {
                    fprintf(stderr, "Failed to reallocate m_freqChanSc\n");
                    return;
                }
                m_freqChanSc = temp_freqChanSc;
            }
        }
        return;
    }
    
    // GPU mode: Allocate GPU memory for active link parameters
    if (m_d_activeLinkParams) {
        cudaFree(m_d_activeLinkParams);
        m_d_activeLinkParams = nullptr;
    }
    CHECK_CUDAERROR(cudaMalloc((void**)&m_d_activeLinkParams, m_lastAllocatedLinks * sizeof(activeLink<Tcomplex>)));
    
    // Below is for CIR, CFR on SC, CFR on PRB, allocated based on internal memory mode
    // Mode 0: External CIR/CFR, Mode 1: Internal CIR/External CFR, Mode 2: Internal CIR/CFR
    if (m_simConfig->internal_memory_mode == 0) {
        return;  // No internal allocation needed for mode 0
    }
    
    // Note: Internal modes use contiguous allocation for performance
    // External arrays (if provided) will be copied to/from this contiguous storage
    
    // Free previously allocated internal memory based on memory mode
    // Mode 1: Free CIR arrays only, Mode 2: Free both CIR and CFR arrays
    if (m_simConfig->internal_memory_mode >= 1) {
        // Free CIR arrays (allocated in modes 1 and 2)
        if (m_cirCoe) {
            cudaFree(m_cirCoe);
            m_cirCoe = nullptr;
        }
        if (m_cirNormDelay) {
            cudaFree(m_cirNormDelay);
            m_cirNormDelay = nullptr;
        }
        if (m_cirNtaps) {
            cudaFree(m_cirNtaps);
            m_cirNtaps = nullptr;
        }
    }
    
    if (m_simConfig->internal_memory_mode == 2) {
        // Free CFR arrays (allocated in mode 2 only)
        if (m_freqChanPrbg) {
            cudaFree(m_freqChanPrbg);
            m_freqChanPrbg = nullptr;
        }
        if (m_freqChanSc) {
            cudaFree(m_freqChanSc);
            m_freqChanSc = nullptr;
        }
    }

    // Allocate CIR memory for modes 1 and 2 (internal CIR)
    if (m_simConfig->internal_memory_mode >= 1) {
        // allocate CIR memory using contiguous layout: (total_active_links, n_snapshots, n_ut_ant, n_bs_ant, max_taps)
        CHECK_CUDAERROR(cudaMalloc((void**)&m_cirCoe, m_lastAllocatedLinks *
                              m_simConfig->n_snapshot_per_slot *
                              m_cmnLinkParams.nUeAnt * 
                              m_cmnLinkParams.nBsAnt * 
                              N_MAX_TAPS *
                              sizeof(Tcomplex)));    
        CHECK_CUDAERROR(cudaMalloc((void**)&m_cirNormDelay, m_lastAllocatedLinks * N_MAX_TAPS * sizeof(uint16_t)));
        CHECK_CUDAERROR(cudaMalloc((void**)&m_cirNtaps, m_lastAllocatedLinks * sizeof(uint16_t)));
        
        // Initialize cirNtaps to zero
        cudaMemset(m_cirNtaps, 0, m_lastAllocatedLinks * sizeof(uint16_t));
    }

    // Allocate CFR memory for mode 2 only (internal CFR)
    if (m_simConfig->internal_memory_mode == 2) {
        if (m_simConfig->run_mode == 0) {
            // Mode 0: CIR only
            m_freqChanPrbg = nullptr;
            m_freqChanSc = nullptr;
        }
        else if (m_simConfig->run_mode == 1 || m_simConfig->run_mode == 3) {
            // Mode 1: CIR and CFR on PRBG only
            CHECK_CUDAERROR(cudaMalloc((void**)&m_freqChanPrbg, m_lastAllocatedLinks *
                              m_simConfig->n_snapshot_per_slot *
                              m_cmnLinkParams.nUeAnt * 
                              m_cmnLinkParams.nBsAnt * 
                              m_simConfig->n_prbg *
                              sizeof(Tcomplex)));
            m_freqChanSc = nullptr;
        } else if (m_simConfig->run_mode == 2 || m_simConfig->run_mode == 3) {
            // Mode 2: CIR and CFR on SC (n_prb * 12 subcarriers)
            // Mode 3: CIR and CFR on PRB/SC (same as mode 2)
            CHECK_CUDAERROR(cudaMalloc((void**)&m_freqChanPrbg, m_lastAllocatedLinks *
                              m_simConfig->n_snapshot_per_slot *
                              m_cmnLinkParams.nUeAnt * 
                              m_cmnLinkParams.nBsAnt * 
                              m_simConfig->n_prbg *
                              sizeof(Tcomplex)));
            CHECK_CUDAERROR(cudaMalloc((void**)&m_freqChanSc, m_lastAllocatedLinks *
                              m_simConfig->n_snapshot_per_slot *
                              m_cmnLinkParams.nUeAnt * 
                              m_cmnLinkParams.nBsAnt * 
                              m_simConfig->n_prb * 12 *
                              sizeof(Tcomplex)));
        } else if (m_simConfig->run_mode == 4) {
            // Mode 4: CIR and CFR on all N_FFT subcarriers
            CHECK_CUDAERROR(cudaMalloc((void**)&m_freqChanPrbg, m_lastAllocatedLinks *
                              m_simConfig->n_snapshot_per_slot *
                              m_cmnLinkParams.nUeAnt * 
                              m_cmnLinkParams.nBsAnt * 
                              m_simConfig->n_prbg *
                              sizeof(Tcomplex)));
            CHECK_CUDAERROR(cudaMalloc((void**)&m_freqChanSc, m_lastAllocatedLinks *
                              m_simConfig->n_snapshot_per_slot *
                              m_cmnLinkParams.nUeAnt * 
                              m_cmnLinkParams.nBsAnt * 
                              m_simConfig->fft_size *
                              sizeof(Tcomplex)));
        }
    }
}

template <typename Tscalar, typename Tcomplex>
void slsChan<Tscalar, Tcomplex>::updateActiveLinkInd(
    const std::vector<uint16_t>& activeCell,
    const std::vector<std::vector<uint16_t>>& activeUt) {
    
#ifdef SLS_DEBUG_
    // Debug: Print input parameters
    printf("DEBUG: updateActiveLinkInd called with:\n");
    printf("  activeCell.size(): %zu\n", activeCell.size());
    printf("  activeUt.size(): %zu\n", activeUt.size());
    for (size_t i = 0; i < activeCell.size() && i < 10; i++) {
        printf("  activeCell[%zu]: %u\n", i, activeCell[i]);
    }
#endif
    
    // save of previous active link parameters
    std::vector<activeLink<Tcomplex>> prevActiveLinkParams = m_activeLinkParams;
    
    // Clear the existing active link parameters
    m_activeLinkParams.clear();
    
    // Handle the case where both vectors are empty (use all links)
    // Create temporary vectors to represent all cells and all UTs, then reuse existing logic
    std::vector<uint16_t> tempActiveCell;
    std::vector<std::vector<uint16_t>> tempActiveUt;
    
    if (activeCell.empty() && activeUt.empty()) {
#ifdef SLS_DEBUG_
        printf("DEBUG: Both activeCell and activeUt are empty, creating all links\n");
        printf("DEBUG: Creating temporary vectors: nSector=%u, nUT=%u\n", m_topology.nSector, m_topology.nUT);
#endif
        
        // Create vectors representing all cells with all UTs
        tempActiveCell.reserve(m_topology.nSector);
        tempActiveUt.reserve(m_topology.nSector);
        
        for (uint16_t cid = 0; cid < m_topology.nSector; ++cid) {
            tempActiveCell.push_back(cid);
            
            // Add all UTs for this cell
            std::vector<uint16_t> allUts;
            allUts.reserve(m_topology.nUT);
            for (uint16_t uid = 0; uid < m_topology.nUT; ++uid) {
                allUts.push_back(uid);
            }
            tempActiveUt.push_back(std::move(allUts));
        }
        
#ifdef SLS_DEBUG_
        printf("DEBUG: Temporary activeCell: [");
        for (size_t i = 0; i < tempActiveCell.size(); i++) {
            printf("%u", tempActiveCell[i]);
            if (i < tempActiveCell.size() - 1) printf(", ");
        }
        printf("]\n");
        printf("DEBUG: Temporary activeUt: [");
        for (size_t i = 0; i < tempActiveUt.size(); i++) {
            printf("{");
            for (size_t j = 0; j < tempActiveUt[i].size(); j++) {
                printf("%u", tempActiveUt[i][j]);
                if (j < tempActiveUt[i].size() - 1) printf(", ");
            }
            printf("}");
            if (i < tempActiveUt.size() - 1) printf(", ");
        }
        printf("]\n");
#endif
    }
    
    // Store active cell and UE mappings for H5 file generation
    // Use temporary vectors if in "all links" mode, otherwise use provided vectors
    m_activeCell = activeCell.empty() ? tempActiveCell : activeCell;
    m_activeUt = activeUt.empty() ? tempActiveUt : activeUt;
    
    // Initialize per-cell pointer vectors for internal memory mode
    if (m_simConfig->internal_memory_mode >= 1) {
        m_cirCoePerCell.clear();
        m_cirNormDelayPerCell.clear();
        m_cirNtapsPerCell.clear();
        
        m_cirCoePerCell.resize(m_activeCell.size(), nullptr);
        m_cirNormDelayPerCell.resize(m_activeCell.size(), nullptr);
        m_cirNtapsPerCell.resize(m_activeCell.size(), nullptr);
        
        // Only clear CFR vectors for mode 2 (internal CFR)
        // For mode 1, CFR vectors contain external pointers set in run() - don't clear them!
        if (m_simConfig->internal_memory_mode == 2) {
            m_freqChanPrbgPerCell.clear();
            m_freqChanScPerCell.clear();
            
            m_freqChanPrbgPerCell.resize(m_activeCell.size(), nullptr);
            m_freqChanScPerCell.resize(m_activeCell.size(), nullptr);
        }
        
#ifdef SLS_DEBUG_
        printf("DEBUG: Initialized per-cell pointer vectors for %zu cells (mode %d)\n", 
               m_activeCell.size(), m_simConfig->internal_memory_mode);
        printf("  CIR vectors cleared and resized\n");
        if (m_simConfig->internal_memory_mode == 2) {
            printf("  CFR vectors cleared and resized (mode 2)\n");
        } else {
            printf("  CFR vectors preserved from external setup (mode 1)\n");
        }
#endif
    }
    
    // Otherwise, create links between active cells and their corresponding active UTs
#ifdef SLS_DEBUG_
    printf("DEBUG: Processing active cells and UTs\n");
#endif
    uint32_t linkIdx = 0;
    for (size_t i = 0; i < m_activeCell.size(); ++i) {
        uint16_t cid = m_activeCell[i];
#ifdef SLS_DEBUG_
        printf("DEBUG: Processing cell %zu (cid=%u)\n", i, cid);
#endif
        
        // Check if we have active UTs for this cell
        if (i < m_activeUt.size() && !m_activeUt[i].empty()) {
#ifdef SLS_DEBUG_
            printf("DEBUG: Cell %u has %zu active UTs, starting at linkIdx=%u\n", cid, m_activeUt[i].size(), linkIdx);
#endif
            
            // Set per-cell pointers for internal memory mode
            if (m_simConfig->internal_memory_mode >= 1) {
                // Set CIR pointers to the appropriate positions in contiguous buffers
                if (m_cirCoe) {
                    const size_t cirCoeOffset = linkIdx * m_simConfig->n_snapshot_per_slot * 
                                              m_cmnLinkParams.nUeAnt * m_cmnLinkParams.nBsAnt * N_MAX_TAPS;
                    m_cirCoePerCell[i] = m_cirCoe + cirCoeOffset;
                }
                
                if (m_cirNormDelay) {
                    const size_t cirNormDelayOffset = linkIdx * N_MAX_TAPS;
                    m_cirNormDelayPerCell[i] = m_cirNormDelay + cirNormDelayOffset;
                }
                
                if (m_cirNtaps) {
                    const size_t cirNtapsOffset = linkIdx;
                    m_cirNtapsPerCell[i] = m_cirNtaps + cirNtapsOffset;
                }
                
                // Handle CFR buffers for mode 2
                if (m_simConfig->internal_memory_mode >= 2) {
                    if (m_freqChanPrbg) {
                        const size_t freqChanPrbgOffset = linkIdx * m_simConfig->n_snapshot_per_slot * 
                                                         m_cmnLinkParams.nUeAnt * m_cmnLinkParams.nBsAnt * m_simConfig->n_prbg;
                        m_freqChanPrbgPerCell[i] = m_freqChanPrbg + freqChanPrbgOffset;
                    }
                    
                    if (m_freqChanSc) {
                        uint32_t scPerLink = (m_simConfig->run_mode == 4) ? m_simConfig->fft_size : (m_simConfig->n_prb * 12);
                        const size_t freqChanScOffset = linkIdx * m_simConfig->n_snapshot_per_slot * 
                                                       m_cmnLinkParams.nUeAnt * m_cmnLinkParams.nBsAnt * scPerLink;
                        m_freqChanScPerCell[i] = m_freqChanSc + freqChanScOffset;
                    }
                }
                
#ifdef SLS_DEBUG_
                printf("DEBUG: Set per-cell pointers for cell %u (array idx %zu):\n", cid, i);
                printf("  m_cirCoePerCell[%zu] = %p (offset: %zu links)\n", i, (void*)m_cirCoePerCell[i], linkIdx);
                printf("  m_cirNormDelayPerCell[%zu] = %p\n", i, (void*)m_cirNormDelayPerCell[i]);
                printf("  m_cirNtapsPerCell[%zu] = %p\n", i, (void*)m_cirNtapsPerCell[i]);
#endif
            }
            
            // Add links for each active UT in this cell
            uint32_t utIdxInCell = 0;  // Track UT index within this cell
            for (uint16_t uid : m_activeUt[i]) {
                uint32_t lspReadIdx = m_topology.cellParams[cid].siteId*m_sysConfig->n_ut + uid;
                
                // Choose memory layout based on internal memory mode
                Tcomplex* cirCoePtr = nullptr;
                uint16_t* cirNormDelayPtr = nullptr;
                uint16_t* cirNtapsPtr = nullptr;
                Tcomplex* freqChanPrbgPtr = nullptr;
                Tcomplex* freqChanScPtr = nullptr;
                
                if (m_simConfig->internal_memory_mode >= 1) {
                    // Internal modes: Use contiguous storage with global linkIdx offset
                    const size_t cirCoeOffset = linkIdx * m_simConfig->n_snapshot_per_slot * m_cmnLinkParams.nUeAnt * m_cmnLinkParams.nBsAnt * N_MAX_TAPS;
                    const size_t cirNormDelayOffset = linkIdx * N_MAX_TAPS;
                    const size_t cirNtapsOffset = linkIdx;
                    
                    // Bounds checking for internal memory access
                    if (linkIdx >= m_lastAllocatedLinks) {
                        printf("ERROR: linkIdx %u >= m_lastAllocatedLinks %zu for cell %u, UT %u\n", 
                               linkIdx, m_lastAllocatedLinks, cid, uid);
                        throw std::runtime_error("Link index out of bounds");
                    }
                    
                    cirCoePtr = m_cirCoe ? m_cirCoe + cirCoeOffset : nullptr;
                    cirNormDelayPtr = m_cirNormDelay ? m_cirNormDelay + cirNormDelayOffset : nullptr;
                    cirNtapsPtr = m_cirNtaps ? m_cirNtaps + cirNtapsOffset : nullptr;
                    
                    if (m_simConfig->internal_memory_mode == 2) {
                        // Mode 2: Internal CFR as well
                        size_t freqChanPrbgOffset = 0, freqChanScOffset = 0;
                        if (m_freqChanPrbg) {
                            freqChanPrbgOffset = linkIdx * m_simConfig->n_snapshot_per_slot * m_cmnLinkParams.nUeAnt * m_cmnLinkParams.nBsAnt * m_simConfig->n_prbg;
                            freqChanPrbgPtr = m_freqChanPrbg + freqChanPrbgOffset;
                        }
                        if (m_freqChanSc) {
                            // Calculate offset based on run mode: mode 2/3 uses n_prb*12, mode 4 uses fft_size
                            uint32_t scPerLink = (m_simConfig->run_mode == 4) ? m_simConfig->fft_size : (m_simConfig->n_prb * 12);
                            freqChanScOffset = linkIdx * m_simConfig->n_snapshot_per_slot * m_cmnLinkParams.nUeAnt * m_cmnLinkParams.nBsAnt * scPerLink;
                            freqChanScPtr = m_freqChanSc + freqChanScOffset;
                        }
                    } else {
                        // Mode 1: External CFR (use per-cell if provided)
                        if (i < m_freqChanPrbgPerCell.size() && m_freqChanPrbgPerCell[i] != nullptr) {
                            size_t freqChanPrbgOffset = utIdxInCell * m_simConfig->n_snapshot_per_slot * m_cmnLinkParams.nUeAnt * m_cmnLinkParams.nBsAnt * m_simConfig->n_prbg;
                            freqChanPrbgPtr = m_freqChanPrbgPerCell[i] + freqChanPrbgOffset;
                        }
                        if (i < m_freqChanScPerCell.size() && m_freqChanScPerCell[i] != nullptr) {
                            uint32_t scPerLink = (m_simConfig->run_mode == 4) ? m_simConfig->fft_size : (m_simConfig->n_prb * 12);
                            size_t freqChanScOffset = utIdxInCell * m_simConfig->n_snapshot_per_slot * m_cmnLinkParams.nUeAnt * m_cmnLinkParams.nBsAnt * scPerLink;
                            freqChanScPtr = m_freqChanScPerCell[i] + freqChanScOffset;
#ifdef SLS_DEBUG_
                            printf("DEBUG: Mode 1 CFR SC pointer assignment: cell=%u, utIdxInCell=%u, linkIdx=%u\n", cid, utIdxInCell, linkIdx);
                            printf("  m_freqChanScPerCell[%zu] = %p, offset = %zu, final ptr = %p\n", 
                                   i, (void*)m_freqChanScPerCell[i], freqChanScOffset, (void*)freqChanScPtr);
#endif
                        }
                    }
                } else {
                    // External mode (0): Use per-cell storage with cell-specific offsets
                    const size_t cirCoeOffset = utIdxInCell * m_simConfig->n_snapshot_per_slot * m_cmnLinkParams.nUeAnt * m_cmnLinkParams.nBsAnt * N_MAX_TAPS;
                    const size_t cirNormDelayOffset = utIdxInCell * N_MAX_TAPS;
                    const size_t cirNtapsOffset = utIdxInCell;
                    
                    cirCoePtr = (i < m_cirCoePerCell.size() && m_cirCoePerCell[i] != nullptr) ? m_cirCoePerCell[i] + cirCoeOffset : nullptr;
                    cirNormDelayPtr = (i < m_cirNormDelayPerCell.size() && m_cirNormDelayPerCell[i] != nullptr) ? m_cirNormDelayPerCell[i] + cirNormDelayOffset : nullptr;
                    cirNtapsPtr = (i < m_cirNtapsPerCell.size() && m_cirNtapsPerCell[i] != nullptr) ? m_cirNtapsPerCell[i] + cirNtapsOffset : nullptr;
                    
                    if (i < m_freqChanPrbgPerCell.size() && m_freqChanPrbgPerCell[i] != nullptr) {
                        size_t freqChanPrbgOffset = utIdxInCell * m_simConfig->n_snapshot_per_slot * m_cmnLinkParams.nUeAnt * m_cmnLinkParams.nBsAnt * m_simConfig->n_prbg;
                        freqChanPrbgPtr = m_freqChanPrbgPerCell[i] + freqChanPrbgOffset;
                    }
                    if (i < m_freqChanScPerCell.size() && m_freqChanScPerCell[i] != nullptr) {
                        uint32_t scPerLink = (m_simConfig->run_mode == 4) ? m_simConfig->fft_size : (m_simConfig->n_prb * 12);
                        size_t freqChanScOffset = utIdxInCell * m_simConfig->n_snapshot_per_slot * m_cmnLinkParams.nUeAnt * m_cmnLinkParams.nBsAnt * scPerLink;
                        freqChanScPtr = m_freqChanScPerCell[i] + freqChanScOffset;
                    }
                }
                
#ifdef SLS_DEBUG_
                printf("DEBUG: Creating link %u: Cell %u -> UT %u (lspReadIdx=%u, mode=%u)\n", 
                       linkIdx, cid, uid, lspReadIdx, m_simConfig->internal_memory_mode);
#endif

                m_activeLinkParams.push_back(activeLink<Tcomplex>{
                    cid,
                    uid,
                    linkIdx,
                    lspReadIdx,
                    cirCoePtr,
                    cirNormDelayPtr,
                    cirNtapsPtr,
                    freqChanPrbgPtr,
                    freqChanScPtr
                });
                linkIdx++;
                utIdxInCell++;
            }
        } else {
#ifdef SLS_DEBUG_
            printf("DEBUG: Cell %u has no active UTs (i=%zu, m_activeUt.size()=%zu, m_activeUt[i].empty()=%s)\n", 
                   cid, i, m_activeUt.size(), (i < m_activeUt.size()) ? (m_activeUt[i].empty() ? "true" : "false") : "N/A");
#endif
        }
    }

    // copy active link parameters to GPU (GPU mode only)
    if (m_simConfig->cpu_only_mode == 0) {
        CHECK_CUDAERROR(cudaMemcpyAsync(m_d_activeLinkParams, m_activeLinkParams.data(), m_activeLinkParams.size() * sizeof(activeLink<Tcomplex>), cudaMemcpyHostToDevice, m_strm));
    }
    
    // check if active link parameters are changed (only check cid and uid, not memory)
    if (m_activeLinkParams.size() != prevActiveLinkParams.size()) {
        m_updatePerTTILinkParams = true;
    } else {
        for (size_t i = 0; i < m_activeLinkParams.size(); i++) {
            if (m_activeLinkParams[i].cid != prevActiveLinkParams[i].cid || 
                m_activeLinkParams[i].uid != prevActiveLinkParams[i].uid) {
                m_updatePerTTILinkParams = true;
                break;
            }
        }
    }
    
#ifdef SLS_DEBUG_
    printf("DEBUG: Total links created: %u\n", linkIdx);
    printf("DEBUG: Expected links: %u\n", 
           static_cast<uint32_t>(activeCell.empty() && activeUt.empty() ? 
                                 m_topology.nSector * m_topology.nUT : 
                                 [&]() { uint32_t total = 0; for (const auto& uts : m_activeUt) total += uts.size(); return total; }()));
    
    // Verify per-cell pointers are within bounds
    if (m_simConfig->internal_memory_mode >= 1) {
        printf("DEBUG: Per-cell pointer verification:\n");
        printf("  Total allocated links: %zu\n", m_lastAllocatedLinks);
        printf("  Per-cell arrays size: %zu\n", m_cirCoePerCell.size());
        
        for (size_t i = 0; i < m_cirCoePerCell.size(); i++) {
            if (m_cirCoePerCell[i] != nullptr) {
                // Calculate offset from base pointer
                size_t offset = (m_cirCoePerCell[i] - m_cirCoe) / 
                              (m_simConfig->n_snapshot_per_slot * m_cmnLinkParams.nUeAnt * m_cmnLinkParams.nBsAnt * N_MAX_TAPS);
                printf("  Cell array[%zu]: ptr offset = %zu links", i, offset);
                if (offset >= m_lastAllocatedLinks) {
                    printf(" **ERROR: OFFSET OUT OF BOUNDS!**");
                }
                printf("\n");
            }
        }
    }
#endif
}

template <typename Tscalar, typename Tcomplex>
void slsChan<Tscalar, Tcomplex>::copyContiguousToPerCell(
    const std::vector<uint16_t>& activeCell,
    const std::vector<std::vector<uint16_t>>& activeUt) {
    
    if (m_simConfig->internal_memory_mode == 0) {
        return; // No internal storage to copy from
    }
    
#ifdef SLS_DEBUG_
    printf("DEBUG: Copying from contiguous internal storage to per-cell external arrays\n");
#endif
    
    uint32_t linkIdx = 0;
    for (size_t i = 0; i < activeCell.size(); ++i) {
        if (i < activeUt.size() && !activeUt[i].empty()) {
            uint32_t utIdxInCell = 0;
            for (uint16_t uid : activeUt[i]) {
                // Copy CIR data if external per-cell arrays are provided
                if (m_simConfig->internal_memory_mode >= 1) {
                    if (i < m_cirCoePerCell.size() && m_cirCoePerCell[i] != nullptr && m_cirCoe != nullptr) {
                        const size_t contiguousOffset = linkIdx * m_simConfig->n_snapshot_per_slot * m_cmnLinkParams.nUeAnt * m_cmnLinkParams.nBsAnt * N_MAX_TAPS;
                        const size_t perCellOffset = utIdxInCell * m_simConfig->n_snapshot_per_slot * m_cmnLinkParams.nUeAnt * m_cmnLinkParams.nBsAnt * N_MAX_TAPS;
                        const size_t copySize = m_simConfig->n_snapshot_per_slot * m_cmnLinkParams.nUeAnt * m_cmnLinkParams.nBsAnt * N_MAX_TAPS;
                        
                        if (m_simConfig->cpu_only_mode == 1) {
                            memcpy(m_cirCoePerCell[i] + perCellOffset, m_cirCoe + contiguousOffset, copySize * sizeof(Tcomplex));
                        } else {
                            CHECK_CUDAERROR(cudaMemcpyAsync(m_cirCoePerCell[i] + perCellOffset, m_cirCoe + contiguousOffset, copySize * sizeof(Tcomplex), cudaMemcpyDeviceToDevice, m_strm));
                        }
                    }
                    
                    if (i < m_cirNormDelayPerCell.size() && m_cirNormDelayPerCell[i] != nullptr && m_cirNormDelay != nullptr) {
                        const size_t contiguousOffset = linkIdx * N_MAX_TAPS;
                        const size_t perCellOffset = utIdxInCell * N_MAX_TAPS;
                        
                        if (m_simConfig->cpu_only_mode == 1) {
                            memcpy(m_cirNormDelayPerCell[i] + perCellOffset, m_cirNormDelay + contiguousOffset, N_MAX_TAPS * sizeof(uint16_t));
                        } else {
                            CHECK_CUDAERROR(cudaMemcpyAsync(m_cirNormDelayPerCell[i] + perCellOffset, m_cirNormDelay + contiguousOffset, N_MAX_TAPS * sizeof(uint16_t), cudaMemcpyDeviceToDevice, m_strm));
                        }
                    }
                    
                    if (i < m_cirNtapsPerCell.size() && m_cirNtapsPerCell[i] != nullptr && m_cirNtaps != nullptr) {
                        const size_t contiguousOffset = linkIdx;
                        const size_t perCellOffset = utIdxInCell;
                        
                        if (m_simConfig->cpu_only_mode == 1) {
                            memcpy(m_cirNtapsPerCell[i] + perCellOffset, m_cirNtaps + contiguousOffset, sizeof(uint16_t));
                        } else {
                            CHECK_CUDAERROR(cudaMemcpyAsync(m_cirNtapsPerCell[i] + perCellOffset, m_cirNtaps + contiguousOffset, sizeof(uint16_t), cudaMemcpyDeviceToDevice, m_strm));
                        }
                    }
                }
                
                // Copy CFR data if mode 2 and external per-cell arrays are provided
                if (m_simConfig->internal_memory_mode == 2) {
                    if (i < m_freqChanPrbgPerCell.size() && m_freqChanPrbgPerCell[i] != nullptr && m_freqChanPrbg != nullptr) {
                        const size_t contiguousOffset = linkIdx * m_simConfig->n_snapshot_per_slot * m_cmnLinkParams.nUeAnt * m_cmnLinkParams.nBsAnt * m_simConfig->n_prbg;
                        const size_t perCellOffset = utIdxInCell * m_simConfig->n_snapshot_per_slot * m_cmnLinkParams.nUeAnt * m_cmnLinkParams.nBsAnt * m_simConfig->n_prbg;
                        const size_t copySize = m_simConfig->n_snapshot_per_slot * m_cmnLinkParams.nUeAnt * m_cmnLinkParams.nBsAnt * m_simConfig->n_prbg;
                        
                        if (m_simConfig->cpu_only_mode == 1) {
                            memcpy(m_freqChanPrbgPerCell[i] + perCellOffset, m_freqChanPrbg + contiguousOffset, copySize * sizeof(Tcomplex));
                        } else {
                            CHECK_CUDAERROR(cudaMemcpyAsync(m_freqChanPrbgPerCell[i] + perCellOffset, m_freqChanPrbg + contiguousOffset, copySize * sizeof(Tcomplex), cudaMemcpyDeviceToDevice, m_strm));
                        }
                    }
                    
                    if (i < m_freqChanScPerCell.size() && m_freqChanScPerCell[i] != nullptr && m_freqChanSc != nullptr) {
                        uint32_t scPerLink = (m_simConfig->run_mode == 4) ? m_simConfig->fft_size : (m_simConfig->n_prb * 12);
                        const size_t contiguousOffset = linkIdx * m_simConfig->n_snapshot_per_slot * m_cmnLinkParams.nUeAnt * m_cmnLinkParams.nBsAnt * scPerLink;
                        const size_t perCellOffset = utIdxInCell * m_simConfig->n_snapshot_per_slot * m_cmnLinkParams.nUeAnt * m_cmnLinkParams.nBsAnt * scPerLink;
                        const size_t copySize = m_simConfig->n_snapshot_per_slot * m_cmnLinkParams.nUeAnt * m_cmnLinkParams.nBsAnt * scPerLink;
                        
                        if (m_simConfig->cpu_only_mode == 1) {
                            memcpy(m_freqChanScPerCell[i] + perCellOffset, m_freqChanSc + contiguousOffset, copySize * sizeof(Tcomplex));
                        } else {
                            CHECK_CUDAERROR(cudaMemcpyAsync(m_freqChanScPerCell[i] + perCellOffset, m_freqChanSc + contiguousOffset, copySize * sizeof(Tcomplex), cudaMemcpyDeviceToDevice, m_strm));
                        }
                    }
                }
                
                linkIdx++;
                utIdxInCell++;
            }
        }
    }
}

template <typename Tscalar, typename Tcomplex>
void slsChan<Tscalar, Tcomplex>::copyPerCellToContiguous(
    const std::vector<uint16_t>& activeCell,
    const std::vector<std::vector<uint16_t>>& activeUt) {
    
    if (m_simConfig->internal_memory_mode == 0) {
        return; // No internal storage to copy to
    }
    
#ifdef SLS_DEBUG_
    printf("DEBUG: Copying from per-cell external arrays to contiguous internal storage\n");
#endif
    
    uint32_t linkIdx = 0;
    for (size_t i = 0; i < activeCell.size(); ++i) {
        if (i < activeUt.size() && !activeUt[i].empty()) {
            uint32_t utIdxInCell = 0;
            for (uint16_t uid : activeUt[i]) {
                // Copy CIR data from external per-cell arrays if available
                if (m_simConfig->internal_memory_mode >= 1) {
                    if (i < m_cirCoePerCell.size() && m_cirCoePerCell[i] != nullptr && m_cirCoe != nullptr) {
                        const size_t contiguousOffset = linkIdx * m_simConfig->n_snapshot_per_slot * m_cmnLinkParams.nUeAnt * m_cmnLinkParams.nBsAnt * N_MAX_TAPS;
                        const size_t perCellOffset = utIdxInCell * m_simConfig->n_snapshot_per_slot * m_cmnLinkParams.nUeAnt * m_cmnLinkParams.nBsAnt * N_MAX_TAPS;
                        const size_t copySize = m_simConfig->n_snapshot_per_slot * m_cmnLinkParams.nUeAnt * m_cmnLinkParams.nBsAnt * N_MAX_TAPS;
                        
                        if (m_simConfig->cpu_only_mode == 1) {
                            memcpy(m_cirCoe + contiguousOffset, m_cirCoePerCell[i] + perCellOffset, copySize * sizeof(Tcomplex));
                        } else {
                            CHECK_CUDAERROR(cudaMemcpyAsync(m_cirCoe + contiguousOffset, m_cirCoePerCell[i] + perCellOffset, copySize * sizeof(Tcomplex), cudaMemcpyDeviceToDevice, m_strm));
                        }
                    }
                    
                    if (i < m_cirNormDelayPerCell.size() && m_cirNormDelayPerCell[i] != nullptr && m_cirNormDelay != nullptr) {
                        const size_t contiguousOffset = linkIdx * N_MAX_TAPS;
                        const size_t perCellOffset = utIdxInCell * N_MAX_TAPS;
                        
                        if (m_simConfig->cpu_only_mode == 1) {
                            memcpy(m_cirNormDelay + contiguousOffset, m_cirNormDelayPerCell[i] + perCellOffset, N_MAX_TAPS * sizeof(uint16_t));
                        } else {
                            CHECK_CUDAERROR(cudaMemcpyAsync(m_cirNormDelay + contiguousOffset, m_cirNormDelayPerCell[i] + perCellOffset, N_MAX_TAPS * sizeof(uint16_t), cudaMemcpyDeviceToDevice, m_strm));
                        }
                    }
                    
                    if (i < m_cirNtapsPerCell.size() && m_cirNtapsPerCell[i] != nullptr && m_cirNtaps != nullptr) {
                        const size_t contiguousOffset = linkIdx;
                        const size_t perCellOffset = utIdxInCell;
                        
                        if (m_simConfig->cpu_only_mode == 1) {
                            memcpy(m_cirNtaps + contiguousOffset, m_cirNtapsPerCell[i] + perCellOffset, sizeof(uint16_t));
                        } else {
                            CHECK_CUDAERROR(cudaMemcpyAsync(m_cirNtaps + contiguousOffset, m_cirNtapsPerCell[i] + perCellOffset, sizeof(uint16_t), cudaMemcpyDeviceToDevice, m_strm));
                        }
                    }
                }
                
                linkIdx++;
                utIdxInCell++;
            }
        }
    }
    
    // Ensure all copies are complete (GPU mode only)
    if (m_simConfig->cpu_only_mode == 0) {
        CHECK_CUDAERROR(cudaStreamSynchronize(m_strm));
    }
}

// Constants for antenna pattern calculation
constexpr float theta_3dB = 65.0f;  // 3dB beamwidth in theta direction (degrees)
constexpr float phi_3dB = 65.0f;    // 3dB beamwidth in phi direction (degrees)
constexpr float A_m = 30.0f;        // Maximum attenuation (dB)
constexpr float SLA_v = 30.0f;      // Side lobe attenuation in vertical direction (dB)

// Antenna pattern calculation functions
inline void calc_AntPattern(int antModel, float* antTheta, float* antPhi) {
    if (antModel == 0) {  // Isotropic pattern
        // Set all gains to 0 dB (1 in linear scale)
        for (int i = 0; i <= 180; i++) {
            antTheta[i] = 0.0f;
        }
        for (int i = 0; i < 360; i++) {
            antPhi[i] = 0.0f;
        }
    }
    else if (antModel == 1) {  // Directional pattern (3GPP)
        // Calculate theta pattern (0 to 180 degrees)
        for (int i = 0; i <= 180; i++) {
            float theta = static_cast<float>(i);  // theta from 0 to 179
            
            // Calculate vertical pattern
            float A_v = -std::min(12.0f * powf((theta - 90.0f) / theta_3dB, 2), SLA_v);
            antTheta[i] = A_v;
        }

        // Calculate phi pattern (0 to 359 degrees)
        for (int i = 0; i < 360; i++) {
            float phi = static_cast<float>(i);  // phi from 0 to 359
            
            // Calculate horizontal pattern
            float phi_wrap = phi >= 180.0f ? phi - 360.0f : phi;  // wrap phi from [0, 359] to [-180, 179]
            float A_h = -std::min(12.0f * powf(phi_wrap / phi_3dB, 2), A_m);
            antPhi[i] = A_h;
        }
    }
}

template<typename Tscalar, typename Tcomplex>
void slsChan<Tscalar, Tcomplex>::initializeAntPanelConfig() {
    try {
        if (m_extConfig != nullptr && !m_extConfig->ant_panel_config.empty()) {
            m_antPanelConfig = &(m_extConfig->ant_panel_config);
        } else {
            m_ownAntPanelConfig.resize(2);  // BS and UE panels
            // BS antenna panel configuration (index 0)
            // Initialize antSize array [M_g, N_g, M, N, P]
            m_ownAntPanelConfig[0].antSize[0] = 1;  // M_g
            m_ownAntPanelConfig[0].antSize[1] = 1;  // N_g
            m_ownAntPanelConfig[0].antSize[2] = 1;  // M
            m_ownAntPanelConfig[0].antSize[3] = 2;  // N
            m_ownAntPanelConfig[0].antSize[4] = 2;  // P
            // calculate nAnt based on antSize
            m_ownAntPanelConfig[0].nAnt = m_ownAntPanelConfig[0].antSize[0] * m_ownAntPanelConfig[0].antSize[1] * m_ownAntPanelConfig[0].antSize[2] * m_ownAntPanelConfig[0].antSize[3] * m_ownAntPanelConfig[0].antSize[4];
            // Initialize antSpacing array [d_g_h, d_g_v, d_h, d_v]
            m_ownAntPanelConfig[0].antSpacing[0] = 0.0f;   // d_g_h
            m_ownAntPanelConfig[0].antSpacing[1] = 0.0f;   // d_g_v
            m_ownAntPanelConfig[0].antSpacing[2] = 0.5f;   // d_h
            m_ownAntPanelConfig[0].antSpacing[3] = 0.5f;   // d_v
            // Initialize antPolarAngles array [roll_angle_first_polz, roll_angle_second_polz]
            m_ownAntPanelConfig[0].antPolarAngles[0] = 45.0f;   // roll_angle_first_polz
            m_ownAntPanelConfig[0].antPolarAngles[1] = -45.0f;  // roll_angle_second_polz
            m_ownAntPanelConfig[0].antModel = 1;  // 0: isotropic, 1: directional, 2: direct pattern
            // UE antenna panel configuration (index 1)
            // Initialize antSize array [M_g, N_g, M, N, P]
            m_ownAntPanelConfig[1].antSize[0] = 1;  // M_g
            m_ownAntPanelConfig[1].antSize[1] = 1;  // N_g
            m_ownAntPanelConfig[1].antSize[2] = 2;  // M
            m_ownAntPanelConfig[1].antSize[3] = 2;  // N
            m_ownAntPanelConfig[1].antSize[4] = 1;  // P
            // calculate nAnt based on antSize
            m_ownAntPanelConfig[1].nAnt = m_ownAntPanelConfig[1].antSize[0] * m_ownAntPanelConfig[1].antSize[1] * m_ownAntPanelConfig[1].antSize[2] * m_ownAntPanelConfig[1].antSize[3] * m_ownAntPanelConfig[1].antSize[4];
            // Initialize antSpacing array [d_g_h, d_g_v, d_h, d_v]
            m_ownAntPanelConfig[1].antSpacing[0] = 0.0f;   // d_g_h
            m_ownAntPanelConfig[1].antSpacing[1] = 0.0f;   // d_g_v
            m_ownAntPanelConfig[1].antSpacing[2] = 0.5f;   // d_h
            m_ownAntPanelConfig[1].antSpacing[3] = 0.5f;   // d_v
            // Initialize antPolarAngles array [roll_angle_first_polz, roll_angle_second_polz]
            m_ownAntPanelConfig[1].antPolarAngles[0] = 0.0f;   // roll_angle_first_polz
            m_ownAntPanelConfig[1].antPolarAngles[1] = 0.0f;   // roll_angle_second_polz
            m_ownAntPanelConfig[1].antModel = 0;  // 0: isotropic, 1: directional, 2: direct pattern
            
            // Calculate antenna patterns for owned config before making it const
            for (int i = 0; i < m_ownAntPanelConfig.size(); i++) {
                calc_AntPattern(m_ownAntPanelConfig[i].antModel, m_ownAntPanelConfig[i].antTheta, m_ownAntPanelConfig[i].antPhi);
            }
            
            m_antPanelConfig = static_cast<const std::vector<AntPanelConfig>*>(&m_ownAntPanelConfig);
        }
    }
    catch (const std::exception& e) {
        printf("Error in initializeAntPanelConfig: %s\n", e.what());
        throw;
    }
    // Note: For external configs, assume antenna patterns are already calculated
}

template <typename Tscalar, typename Tcomplex>
void slsChan<Tscalar, Tcomplex>::setup(const ExternalConfig* extConfig) {
    // Override cell parameters if provided
    if (!extConfig->cell_config.empty()) {
        for (const auto& ext_cell : extConfig->cell_config) {
            // Find matching cell by cid
            for (auto& cell : m_topology.cellParams) {
                if (cell.cid == ext_cell.cid) {
                    // Always override with external config values (they can legitimately be 0)
                    cell.siteId = ext_cell.siteId;
                    cell.loc = ext_cell.loc;
                    cell.antPanelIdx = ext_cell.antPanelIdx;
                    std::memcpy(cell.antPanelOrientation, ext_cell.antPanelOrientation, 
                              sizeof(float) * 3);
                    break;
                }
            }
        }
    }

    // Override UT parameters if provided
    if (!extConfig->ut_config.empty()) {
        for (const auto& ext_ut : extConfig->ut_config) {
            // Find matching UT by uid
            for (auto& ut : m_topology.utParams) {
                if (ut.uid == ext_ut.uid) {
                    // Always override with external config values (they can legitimately be 0)
                    ut.loc = ext_ut.loc;
                    ut.outdoor_ind = ext_ut.outdoor_ind;
                    // d_2d_in is now handled internally with default value -1.0f
                    ut.antPanelIdx = ext_ut.antPanelIdx;
                    std::memcpy(ut.antPanelOrientation, ext_ut.antPanelOrientation, 
                              sizeof(float) * 3);
                    std::memcpy(ut.velocity, ext_ut.velocity, sizeof(float) * 3);
                    break;
                }
            }
        }
    }

    // override antenna panel configuration if provided
    if (!extConfig->ant_panel_config.empty()) {
        // Create a deep copy of external config to ensure data persistence
        m_ownAntPanelConfig.clear();
        m_ownAntPanelConfig.reserve(extConfig->ant_panel_config.size());
        
        for (const auto& ext_panel : extConfig->ant_panel_config) {
            AntPanelConfig local_panel{};
            
            // Copy all fields
            local_panel.nAnt = ext_panel.nAnt;
            std::memcpy(local_panel.antSize, ext_panel.antSize, sizeof(local_panel.antSize));
            std::memcpy(local_panel.antSpacing, ext_panel.antSpacing, sizeof(local_panel.antSpacing));
            std::memcpy(local_panel.antPolarAngles, ext_panel.antPolarAngles, sizeof(local_panel.antPolarAngles));
            local_panel.antModel = ext_panel.antModel;
            
            // Copy antenna patterns (deep copy of arrays)
            std::memcpy(local_panel.antTheta, ext_panel.antTheta, sizeof(local_panel.antTheta));
            std::memcpy(local_panel.antPhi, ext_panel.antPhi, sizeof(local_panel.antPhi));
            
            m_ownAntPanelConfig.push_back(local_panel);
        }
        
        // Calculate antenna patterns for panels with antModel 0 or 1
        for (auto& ant_panel : m_ownAntPanelConfig) {
            if (ant_panel.antModel == 0 || ant_panel.antModel == 1) {
                calc_AntPattern(ant_panel.antModel, ant_panel.antTheta, ant_panel.antPhi);
            }
        }
        
        // Update the pointer to use our owned configuration
        m_antPanelConfig = static_cast<const std::vector<AntPanelConfig>*>(&m_ownAntPanelConfig);
    }
}

template <typename Tscalar, typename Tcomplex>
void slsChan<Tscalar, Tcomplex>::dump_los_nlos_stats(float* lost_nlos_stats) {
    if (lost_nlos_stats == nullptr) {
        return;
    }
    
    // Initialize array to 0 (NLOS)
    const uint32_t total_elements = m_sysConfig->n_sector_per_site * m_sysConfig->n_site * m_sysConfig->n_ut;
    std::memset(lost_nlos_stats, 0, total_elements * sizeof(float));
    
    // Fill the array with LOS/NLOS stats from link parameters
    // Array is organized as [n_sector, n_ut]
    for (uint32_t linkIdx = 0; linkIdx < m_linkParams.size(); ++linkIdx) {
        // Calculate sector and UT indices from link index
        uint32_t sectorIdx = linkIdx / m_sysConfig->n_ut;
        uint32_t utIdx = linkIdx % m_sysConfig->n_ut;
        
        // Bounds checking
        if (sectorIdx < (m_sysConfig->n_sector_per_site * m_sysConfig->n_site) && 
            utIdx < m_sysConfig->n_ut) {
            
            // Store LOS indicator: 1.0f for LOS, 0.0f for NLOS
            uint32_t arrayIdx = sectorIdx * m_sysConfig->n_ut + utIdx;
            lost_nlos_stats[arrayIdx] = static_cast<float>(m_linkParams[linkIdx].losInd);
        }
    }
}

template <typename Tscalar, typename Tcomplex>
void slsChan<Tscalar, Tcomplex>::dump_pathloss_shadowing_stats(float* pathloss_shadowing,
                                                              const std::vector<uint16_t>& activeCell,
                                                              const std::vector<uint16_t>& activeUt) {
    if (pathloss_shadowing == nullptr) {
        return;
    }
    
    // Step 1: Determine what to dump and prepare dimensions
    const bool dump_all_cells = activeCell.empty();
    const bool dump_all_uts = activeUt.empty();
    
    // Calculate dimensions and total elements based on mode
    uint32_t num_cells, num_uts, total_elements;
    const uint32_t total_sectors = m_sysConfig->n_sector_per_site * m_sysConfig->n_site;
    
    if (dump_all_cells && dump_all_uts) {
        // Case 1: Both empty - dump all links
        num_cells = total_sectors;
        num_uts = m_sysConfig->n_ut;
        total_elements = num_cells * num_uts;
    } else if (dump_all_cells && !dump_all_uts) {
        // Case 2: activeCell empty, activeUt has values - dump all cells for specified UTs
        num_cells = total_sectors;
        num_uts = activeUt.size();
        total_elements = num_cells * num_uts;
    } else if (!dump_all_cells && dump_all_uts) {
        // Case 3: activeUt empty, activeCell has values - dump all UTs for specified cells
        num_cells = activeCell.size();
        num_uts = m_sysConfig->n_ut;
        total_elements = num_cells * num_uts;
    } else {
        // Case 4: Both have values - dump only specified combinations
        num_cells = activeCell.size();
        num_uts = activeUt.size();
        total_elements = num_cells * num_uts;
    }
    
    // Step 2: Initialize output array
    std::memset(pathloss_shadowing, 0, total_elements * sizeof(float));
    
    // Step 3: Fill the array with pathloss + shadowing data
    for (uint32_t cell_idx = 0; cell_idx < num_cells; ++cell_idx) {
        // Get actual cell ID based on mode
        uint16_t cid;
        if (dump_all_cells) {
            cid = cell_idx;  // Use cell_idx directly as cell ID
        } else {
            cid = activeCell[cell_idx];  // Use specified cell ID
        }
        
        if (cid >= m_topology.cellParams.size()) continue;
        
        const uint16_t site_id = m_topology.cellParams[cid].siteId;
        
        for (uint32_t ue_idx = 0; ue_idx < num_uts; ++ue_idx) {
            // Get actual UE ID based on mode
            uint16_t uid;
            if (dump_all_uts) {
                uid = ue_idx;  // Use ue_idx directly as UE ID
            } else {
                uid = activeUt[ue_idx];  // Use specified UE ID
            }
            
            // Calculate global link index from cell and UT IDs
            const uint32_t linkIdx = site_id * m_sysConfig->n_ut + uid;
            
            // Bounds checking for link parameters
            if (linkIdx < m_linkParams.size() && 
                cid < total_sectors && 
                uid < m_sysConfig->n_ut) {
                
                // Calculate array index in the output
                const uint32_t arrayIdx = cell_idx * num_uts + ue_idx;
                
                // Store total loss = - (pathloss - shadow fading) (both in dB)
                // The sign of the shadow fading is defined so that positive SF means more received power at UT than predicted by the path loss model
                pathloss_shadowing[arrayIdx] = - (m_linkParams[linkIdx].pathloss - m_linkParams[linkIdx].SF);
            }
        }
    }
}

// Explicit template instantiations
// This must come after all template definitions are included
template class slsChan<float, float2>;
// template class slsChan<__half, __half2>;  // Disabled for now