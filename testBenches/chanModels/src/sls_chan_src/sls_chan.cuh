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

#include <string>
#include <vector>
#include <cmath>      // For math functions like log10, pow, exp
#include <cuda_runtime.h>
#include <cuda/std/complex>
#include <cuda_fp16.h>  // for __half and __half2
#include <random>
#include <curand_kernel.h>
#include <curand.h>
#include <cstdint>  // for uint8_t, uint16_t, uint32_t
#include <cassert>
#include <cstring>
#include "sls_table.h"
#include "../chanModelsDataset.hpp"
#include "../fastFadingCommon.cuh"

#define N_MAX_TAPS 24
// #define SLS_DEBUG_  // for debugging output
// #define CALIBRATION_CFG_  // for calibration config (min BS-UT is 0, diff generation of d_2d_in)

// Topology parameters
struct TopologyParam {
    uint32_t nSite;  // Number of sites
    uint32_t nSector;  // Number of sectors per site
    uint32_t n_sector_per_site;  // Number of sectors per site
    uint32_t nUT;  // Number of user terminals
    std::vector<CellParam> cellParams;  // Base station parameters
    std::vector<UtParam> utParams;  // User terminal parameters
    float ISD;  // Inter-site distance
    float bsHeight;  // Base station height
    float minBsUeDist2d;  // Minimum 2D distance between base station and user terminal
    float maxBsUeDist2dIndoor;  // Maximum 2D distance between base station and user terminal in indoor
    float indoorUtPercent;  // Percentage of user terminals in indoor
};

// Link parameters
// 81 bytes
struct LinkParams {
    float d2d;  // 2D distance between BS and UT in meters
    float d2d_in;  // 2D distance in indoor environment in meters
    float d2d_out;  // 2D distance in outdoor environment in meters
    float d3d;  // 3D distance between BS and UT in meters
    float d3d_in;  // 3D distance in indoor environment in meters
    float d3d_out;  // 3D distance in outdoor environment in meters
    float phi_LOS_AOD;  // Line-of-Sight (LOS) Azimuth Angle of Departure in degrees
    float theta_LOS_ZOD;  // Line-of-Sight (LOS) Zenith Angle of Departure in degrees
    float phi_LOS_AOA;  // Line-of-Sight (LOS) Azimuth Angle of Arrival in degrees
    float theta_LOS_ZOA;  // Line-of-Sight (LOS) Zenith Angle of Arrival in degrees    
    uint8_t losInd;  // Line-of-Sight indicator (1: LOS, 0: NLOS)
    float pathloss;  // Path loss in dB
    float SF;  // Shadow Fading in dB
    float K;  // K-factor (Ricean factor) in dB
    float DS; // Delay Spread in seconds
    float ASD; // Azimuth Spread of Departure in degrees
    float ASA; // Azimuth Spread of Arrival in degrees
    float mu_lgZSD; // Mean of log10(Zenith Spread of Departure) in degrees
    float sigma_lgZSD; // Standard deviation of log10(Zenith Spread of Departure) in degrees
    float mu_offset_ZOD; // Mean of offset of ZOD
    float ZSD; // Zenith Spread of Departure in degrees
    float ZSA; // Zenith Spread of Arrival in degrees
};

// Common link parameters
// Updated bytes count due to array size changes
struct CmnLinkParams {
    // Large-scale parameters
    float mu_lgDS[3];  // [NLOS, LOS, O2I] mean of log10(DS)
    float sigma_lgDS[3];  // [NLOS, LOS, O2I] std of log10(DS)
    float mu_lgASD[3];  // [NLOS, LOS, O2I] mean of log10(ASD)
    float sigma_lgASD[3];  // [NLOS, LOS, O2I] std of log10(ASD)
    float mu_lgASA[3];  // [NLOS, LOS, O2I] mean of log10(ASA)
    float sigma_lgASA[3];  // [NLOS, LOS, O2I] std of log10(ASA)
    float mu_lgZSA[3];  // [NLOS, LOS, O2I] mean of log10(ZSA)
    float sigma_lgZSA[3];  // [NLOS, LOS, O2I] std of log10(ZSA)
    float mu_K[3];  // [NLOS, LOS, O2I] mean of K
    float sigma_K[3];  // [NLOS, LOS, O2I] std of K
    float r_tao[3];  // [NLOS, LOS, O2I] delay scaling factor
    float mu_XPR[3];  // [NLOS, LOS, O2I] mean of XPR
    float sigma_XPR[3];  // [NLOS, LOS, O2I] std of XPR
    uint16_t nCluster[3];  // [NLOS, LOS, O2I] number of clusters
    uint16_t nRayPerCluster[3];  // [NLOS, LOS, O2I] number of rays per cluster
    
    // cluster parameters
    float C_DS[3];  // [NLOS, LOS, O2I] cluster DS (not used with O2I)
    float C_ASD[3];  // [NLOS, LOS, O2I] cluster ASD
    float C_ASA[3];  // [NLOS, LOS, O2I] cluster ASA
    float C_ZSA[3];  // [NLOS, LOS, O2I] cluster ZSA
    float xi[3];  // [NLOS, LOS, O2I] cluster shadowing std
    float C_phi_LOS;  // LOS azimuth offset, need to be scaled based on K
    float C_phi_NLOS;  // NLOS azimuth offset
    float C_phi_O2I;  // O2I azimuth offset
    float C_theta_LOS;  // LOS elevation offset, need to be scaled based on K
    float C_theta_NLOS;  // NLOS elevation offset
    float C_theta_O2I;  // O2I elevation offset
    
    // updated lgfc
    float lgfc;
    
    // Ray offset angles (no need to set since they are same for all scenarios)
    // float RayOffsetAngles[20];

    // Correlation matrices for LOS and NLOS cases
    float sqrtCorrMatLos[LOS_MATRIX_SIZE * LOS_MATRIX_SIZE];
    float sqrtCorrMatNlos[NLOS_MATRIX_SIZE * NLOS_MATRIX_SIZE];
    float sqrtCorrMatO2i[O2I_MATRIX_SIZE * O2I_MATRIX_SIZE];

    uint32_t nLink;  // Number of links from a Site to a UE
    uint32_t nUeAnt; // Max number of UE antennas. TODO: should be the same for all UEs
    uint32_t nBsAnt; // Max number of BS antennas. TODO: should be the same for all BSs
    float lambda_0;  // Wavelength in meters
    
    // Subcluster ray definitions (3GPP Table 7.5-5)
    static constexpr int nSubCluster = 3;
    static constexpr int maxRaysInSubCluster = 10;
    int raysInSubClusterSizes[nSubCluster];  // {10, 6, 4}
    uint16_t raysInSubCluster0[10];  // {0, 1, 2, 3, 4, 5, 6, 7, 18, 19}
    uint16_t raysInSubCluster1[6];   // {8, 9, 10, 11, 16, 17}
    uint16_t raysInSubCluster2[4];   // {12, 13, 14, 15}
};

// Cluster parameters
struct ClusterParams {
    // Use maximum possible sizes
    static constexpr uint8_t MAX_CLUSTERS = 20;
    static constexpr uint8_t MAX_RAYS = 20;
    
    // Actual number of clusters and rays for this instance
    uint16_t nCluster = 0;
    uint16_t nRayPerCluster = 0;
    
    // Arrays with maximum size
    float delays[MAX_CLUSTERS];
    float powers[MAX_CLUSTERS];
    uint16_t strongest2clustersIdx[2];
    float phi_n_AoA[MAX_CLUSTERS];
    float phi_n_AoD[MAX_CLUSTERS];
    float theta_n_ZOD[MAX_CLUSTERS];
    float theta_n_ZOA[MAX_CLUSTERS];
    float xpr[MAX_CLUSTERS * MAX_RAYS];
    float randomPhases[MAX_CLUSTERS * MAX_RAYS * 4];
    float phi_n_m_AoA[MAX_CLUSTERS * MAX_RAYS];
    float phi_n_m_AoD[MAX_CLUSTERS * MAX_RAYS];
    float theta_n_m_ZOD[MAX_CLUSTERS * MAX_RAYS];
    float theta_n_m_ZOA[MAX_CLUSTERS * MAX_RAYS];
};

// Active link parameters
template <typename Tcomplex>
struct activeLink {
    // Default constructor to initialize all pointer members to nullptr
    activeLink() : cirCoe(nullptr), cirNormDelay(nullptr), cirNtaps(nullptr), 
                  freqChanPrbg(nullptr), freqChanSc(nullptr) {}

    // Constructor with all parameters
    activeLink(uint16_t cid_, uint16_t uid_, uint32_t linkIdx_, uint32_t lspReadIdx_,
              Tcomplex* cirCoe_, uint16_t* cirNormDelay_, uint16_t* cirNtaps_,
              Tcomplex* freqChanPrbg_, Tcomplex* freqChanSc_)
        : cid(cid_), uid(uid_), linkIdx(linkIdx_), lspReadIdx(lspReadIdx_),
          cirCoe(cirCoe_), cirNormDelay(cirNormDelay_), cirNtaps(cirNtaps_),
          freqChanPrbg(freqChanPrbg_), freqChanSc(freqChanSc_) {}

    // link indexes
    uint16_t cid;
    uint16_t uid;
    uint32_t linkIdx;
    uint32_t lspReadIdx;
    
    // place to save the channel coefficients
    Tcomplex * cirCoe;
    uint16_t * cirNormDelay;
    uint16_t * cirNtaps;
    Tcomplex * freqChanPrbg;  // CFR on PRBG level
    Tcomplex * freqChanSc;    // CFR on SC level - reused for mode 2/3 (n_prb*12) and mode 4 (N_FFT)
};

// Use types from chanModelsApi.hpp
using scenario_t = Scenario;
using cmnLinkParams_t = CmnLinkParams;
using clusterParams_t = ClusterParams;
using linkParams_t = LinkParams;
using topologyParam_t = TopologyParam;

// Random number type enum
enum class RandomNumberType {
    UNIFORM,
    NORMAL
};

// Main channel model class
template <typename Tscalar, typename Tcomplex>
class slsChan {
public:
    /**
     * @brief Constructor for SLS channel class
     * 
     * @param simConfig Simulation configuration
     * @param sysConfig System level configuration
     * @param randSeed Random seed for simulation
     * @param strm CUDA stream to run SLS class
     */
    slsChan(const SimConfig * simConfig, const SystemLevelConfig * sysConfig, const ExternalConfig * extConfig, uint32_t randSeed, cudaStream_t strm);

    /**
     * @brief Setup SLS channel class by overriding the default configuration
     * 
     * @param extConfig External configuration
     */
    void setup(const ExternalConfig * extConfig);

    /**
     * @brief Run SLS channel generation
     * 
     * @param refTime Timestamp for the start of transmitted symbol (default: 0.0f)
     * @param continuous_fading Flag to enable continuous fading (default: 1)
     * @param activeCell Vector of active cell indices (default: empty)
     * @param activeUt Vector of vectors containing active UT indices per cell (default: empty)
     * @param utNewLoc Vector of new UT locations (default: empty)
     * @param utNewVelocity Vector of new UT mobility parameters (default: empty)
     * @param cirCoePerCell Vector of pointers to store CIR coefficients per cell (default: empty)
     * @param cirNormDelayPerCell Vector of pointers to store CIR normalized delays per cell (default: empty)
     * @param cirNTapsPerCell Vector of pointers to store number of CIR taps per cell (default: empty)
     * @param cfrScPerCell Vector of pointers to store CFR on subcarriers per cell (default: empty)
     * @param cfrPrbgPerCell Vector of pointers to store CFR on PRBGs per cell (default: empty)
     */
    void run(const float refTime = 0.0f,
             const uint8_t continuous_fading = 1,
             const std::vector<uint16_t>& activeCell = {},
             const std::vector<std::vector<uint16_t>>& activeUt = {},
             const std::vector<Coordinate>& utNewLoc = {},
             const std::vector<float3>& utNewVelocity = {},
             const std::vector<Tcomplex*>& cirCoePerCell = {},
             const std::vector<uint16_t*>& cirNormDelayPerCell = {},
             const std::vector<uint16_t*>& cirNTapsPerCell = {},
             const std::vector<Tcomplex*>& cfrScPerCell = {},
             const std::vector<Tcomplex*>& cfrPrbgPerCell = {});

    /**
     * @brief Get the Channel Impulse Response (CIR) coefficients
     * 
     * @param cellIdx Cell index to access
     * @return Tcomplex* Pointer to CIR coefficient data for the specified cell
     * @note CIR is saved as a row-major 1D array [nUe, nBatch, nUeAnt, nBsAnt, firNzLen] per cell
     *       CIR is saved as sparse matrix. Works for both internal and external memory modes.
     */
    Tcomplex* getCirCoe(uint32_t cellIdx = 0) {
        if (m_cirCoePerCell.empty()) {
            // Internal memory mode: return owning pointer (contiguous allocation)
            if (cellIdx == 0) {
                return m_cirCoe;
            }
            // For cellIdx != 0 in internal mode, would need offset calculation
            return nullptr;
        }
        if (cellIdx >= m_cirCoePerCell.size()) {
            return nullptr;
        }
        return m_cirCoePerCell[cellIdx];
    };

    /**
     * @brief Get the Channel Impulse Response (CIR) tap indices
     * 
     * @param cellIdx Cell index to access
     * @return uint16_t* Pointer to CIR tap indices for the specified cell
     * @note Works for both internal and external memory modes.
     */
    uint16_t* getCirIndex(uint32_t cellIdx = 0) {
        if (m_cirNormDelayPerCell.empty()) {
            // Internal memory mode: return owning pointer (contiguous allocation)
            if (cellIdx == 0) {
                return m_cirNormDelay;
            }
            // For cellIdx != 0 in internal mode, would need offset calculation
            return nullptr;
        }
        if (cellIdx >= m_cirNormDelayPerCell.size()) {
            return nullptr;
        }
        return m_cirNormDelayPerCell[cellIdx];
    };

    /**
     * @brief Get the number of CIR taps
     * 
     * @param cellIdx Cell index to access  
     * @return uint16_t* Pointer to number of CIR taps for the specified cell
     * @note Works for both internal and external memory modes.
     */
    uint16_t* getCirNtaps(uint32_t cellIdx = 0) {
        if (m_cirNtapsPerCell.empty()) {
            // Internal memory mode: return owning pointer (contiguous allocation)
            if (cellIdx == 0) {
                return m_cirNtaps;
            }
            // For cellIdx != 0 in internal mode, would need offset calculation
            return nullptr;
        }
        if (cellIdx >= m_cirNtapsPerCell.size()) {
            return nullptr;
        }
        return m_cirNtapsPerCell[cellIdx];
    };

    /**
     * @brief Get Channel Frequency Response (CFR) on PRBG
     * 
     * @param cellIdx Cell index to access
     * @return Tcomplex* Pointer to CFR on PRBG data for the specified cell
     * @note CFR on PRBG is saved as a row-major 1D array [nUe, nBatch, nUeAnt, nBsAnt, nPRBG] per cell
     *       Works for both internal and external memory modes.
     */
    Tcomplex* getFreqChanPrbg(uint32_t cellIdx = 0) {
        if (m_freqChanPrbgPerCell.empty()) {
            // Internal memory mode: return owning pointer (contiguous allocation)
            if (cellIdx == 0) {
                return m_freqChanPrbg;
            }
            // For cellIdx != 0 in internal mode, would need offset calculation
            return nullptr;
        }
        if (cellIdx >= m_freqChanPrbgPerCell.size()) {
            return nullptr;
        }
        return m_freqChanPrbgPerCell[cellIdx];
    };

    /**
     * @brief Get Channel Frequency Response (CFR) on SC
     * 
     * @param cellIdx Cell index to access
     * @return Tcomplex* Pointer to CFR on SC data for the specified cell
     * @note CFR per cell is a row-major 1D array [nUe, nBatch, nUeAnt, nBsAnt, nSc]
     *       Works for both internal and external memory modes.
     */
    Tcomplex* getFreqChanSc(uint32_t cellIdx = 0) {
        if (m_freqChanScPerCell.empty()) {
            // Internal memory mode: return owning pointer (contiguous allocation)
            if (cellIdx == 0) {
                return m_freqChanSc;
            }
            // For cellIdx != 0 in internal mode, would need offset calculation
            return nullptr;
        }
        if (cellIdx >= m_freqChanScPerCell.size()) {
            return nullptr;
        }
        return m_freqChanScPerCell[cellIdx];
    };

    /**
     * @brief Get received signal output
     * 
     * @return Tcomplex* Pointer to received signal data
     * @note Rx samples is saved as a row-major 1D array [nCell, nUe, nUeAnt or nBsAnt, sigLenPerAnt]
     */
    Tcomplex* getRxSigOut() {return m_rxSigOut;};

    /**
     * @brief Reset the SLS channel class by regenerating random numbers
     */
    void reset();



    /**
     * @brief Destructor
     */
    ~slsChan();
                          
    /**
     * @brief Generate network topology
     */
    void generateTopology() {
        bsUeDropping();  // Call the private implementation
    }

    /**
     * @brief Get GPU memory usage in MB
     * @return float Memory usage in MB (returns 0 in CPU-only mode)
     */
    float getGpuMemUseMB() {
        // Skip GPU memory query in CPU-only mode to avoid creating CUDA context
        if (m_simConfig && m_simConfig->cpu_only_mode == 1) {
            return 0.0f;
        }
        size_t free, total;
        cudaMemGetInfo(&free, &total);
        return (total - free) / (1024.0f * 1024.0f);
    }

    /**
     * @brief Dump network topology to YAML file
     * @param filename Output YAML filename
     */
    void dumpTopologyToYaml(const std::string& filename);

    /**
     * @brief Dump LOS/NLOS statistics for all links
     * 
     * @param lost_nlos_stats Pointer to array for storing LOS/NLOS stats, dimension: [n_sector, n_ut]
     *                        Values: 0 = NLOS (Non-Line-of-Sight), 1 = LOS (Line-of-Sight)
     */
    void dump_los_nlos_stats(float* lost_nlos_stats);

    /**
     * @brief Dump pathloss and shadowing statistics (negative value in dB)
     * 
     * @param pathloss_shadowing Pointer to array for storing pathloss+shadowing stats (required)
     *                          If activeCell and activeUt are provided: dimension [activeCell.size(), activeUt.size()]
     *                          If activeCell or activeUt are empty: use dimension n_sector*n_site or n_ut for the empty one
     *                          Values are total loss = - (pathloss - shadow_fading) in dB
     * @param activeCell Vector of active cell IDs (optional, empty vector dumps all cells)
     * @param activeUt Vector of active UT IDs (optional, empty vector dumps all UEs)
     */
    void dump_pathloss_shadowing_stats(float* pathloss_shadowing,
                                     const std::vector<uint16_t>& activeCell = {},
                                     const std::vector<uint16_t>& activeUt = {});

    /**
     * @brief Save SLS channel data to H5 file for debugging
     * @param filenameEnding Optional string to append to filename
     */
    void saveSlsChanToH5File(std::string_view filenameEnding = "");

    const SystemLevelConfig * m_sysConfig;
    const SimConfig * m_simConfig;
    const ExternalConfig * m_extConfig;
    uint32_t m_randSeed;
    cudaStream_t m_strm;
    bool m_updatePerTTILinkParams;  // update link parameters (for each TTI)
    bool m_updatePLAndPenetrationLoss;  // update pathloss and penetration loss
    bool m_updateAllLSPs;  // update all LSPs
    bool m_updateLosState;  // update LOS/NLOS state (only true at start or after reset)

private:
    /**
     * @brief Perform BS and UE dropping in the network topology
     * 
     * This function handles the spatial distribution of base stations and user equipment
     * according to the specified network topology parameters. It determines the positions
     * of all network elements based on the configuration settings.
     */
    void bsUeDropping();

    /**
     * @brief Initialize antenna panel configuration
     */
    void initializeAntPanelConfig();

    /**
     * @brief Calculate Large Scale Parameters (LSP)
     * 
     * Generates and calculates the large scale parameters including:
     * - Delay Spread (DS)
     * - Angular Spread of Arrival (ASA)
     * - Angular Spread of Departure (ASD)
     * - Shadow Fading (SF)
     * - K-factor
     * These parameters are correlated according to 3GPP specifications.
     */
    inline void calLsp(scenario_t scenario, bool isLos, float fc, bool optionalPlInd,
                      float d_2d, float d_3d, float d_2d_in, float d_3d_in,
                      float h_bs, float h_ut, float phi_los_aod, float phi_los_aoa,
                      float theta_los_zod, float theta_los_zoa,
                      std::mt19937& gen, std::normal_distribution<float>& normalDist,
                      float& ds, float& asd, float& asa, float& sf, float& k,
                      float& zsd, float& zsa, bool isSameSite = false, bool isSameFloor = false,
                      float utX = 0.0f, float utY = 0.0f,
                      const corrDist_t& corrDist = {50.0f, 50.0f, 50.0f, 37.0f, 12.0f, 50.0f, 50.0f},
                      const std::vector<std::vector<std::vector<float>>>& crn = {},
                      float maxX = 0.0f, float minX = 0.0f, float maxY = 0.0f, float minY = 0.0f);

    /**
     * @brief Calculate cluster and ray parameters
     * 
     * Generates cluster and ray parameters including:
     * - Cluster delays
     * - Cluster powers
     * - Arrival and departure angles
     * - Ray coupling
     * - Cross-polarization ratios (XPR)
     * - Initial phases
     */
    void calClusterRay();

    /**
     * @brief Generate Common Random Numbers (CRN) for correlated LSP generation
     */
    void generateCRN();

    /**
     * @brief Generate Channel Impulse Response (CIR)
     * 
     */
    void generateCIR();

    /**
     * @brief Generate Channel Frequency Response (CFR)
     * 
     */
    void generateCFR();

    /**
     * @brief Process transmitted samples
     * 
     * Handles the processing of transmitted signal samples through the channel.
     */
    void processTxSamples() {
        // TODO: Implement this function
    }

    /**
     * @brief Allocate GPU memory for internal data structures
     */
    void allocateStaticGpuMem();
    void allocateDynamicGpuMem(uint32_t nLink);
    
    /**
     * @brief Copy data from contiguous internal storage to per-cell external arrays
     * 
     * Used when internal memory mode is enabled but external per-cell arrays are provided
     */
    void copyContiguousToPerCell(const std::vector<uint16_t>& activeCell,
                                const std::vector<std::vector<uint16_t>>& activeUt);
    
    /**
     * @brief Copy data from per-cell external arrays to contiguous internal storage
     * 
     * Used when internal memory mode is enabled and external per-cell arrays need to be processed
     */
    void copyPerCellToContiguous(const std::vector<uint16_t>& activeCell,
                                const std::vector<std::vector<uint16_t>>& activeUt);

    /**
     * @brief Calculate common link parameters
     */
    void calCmnLinkParams();

    /**
     * @brief Calculate link parameters
     */
    void calLinkParam();

    /**
     * @brief Calculate link parameters using GPU
     */
    void calLinkParamGPU();
    
    /**
     * @brief Calculate cluster ray parameters using GPU
     */
    void calClusterRayGPU();
    
    /**
     * @brief Generate Channel Impulse Response using GPU
     */
    void generateCIRGPU();
    
    /**
     * @brief Generate Channel Frequency Response using GPU
     */
    void generateCFRGPU();
    
    /**
     * @brief Generate Common Random Numbers (CRN) on GPU for correlated LSP generation
     */
    void generateCRNGPU();

    /**
     * @brief Update active link indices based on active cells and UTs
     * 
     * @param activeCell Vector of active cell indices
     * @param activeUt Vector of vectors containing active UT indices per cell
     * 
     * If both inputs are empty, creates links for all BS-UE pairs.
     * Otherwise, creates links between each active cell and its corresponding active UTs.
     */
    void updateActiveLinkInd(const std::vector<uint16_t>& activeCell,
                           const std::vector<std::vector<uint16_t>>& activeUt);

    uint32_t m_nSiteUeLink;
    size_t m_lastAllocatedSize;
    // Add topology parameters
    topologyParam_t m_topology;  // Network topology parameters
    
    // Internal data structures
    size_t m_lastAllocatedLinks;
    size_t m_lastAllocatedActiveLinks;  // Track allocated active link memory
    float m_refTime;  // reference time for CIR and CFR generation
    
    /** Memory ownership clarification */
    
    /** OWNING POINTERS - Internal contiguous storage (for modes 1,2 - performance)
     * 
     * MEMORY OWNERSHIP CONTRACT:
     * - These pointers OWN the memory they point to
     * - Memory is allocated via cudaMalloc() in allocateInternalMemory()
     * - Memory is deallocated via cudaFree() in deallocateInternalMemory()
     * - NEVER call delete[] on these pointers - use cudaFree() only
     * - These pointers become invalid after calling deallocateInternalMemory()
     */
    Tcomplex* m_cirCoe;        // OWNING: Contiguous CIR coefficients for internal allocation
    uint16_t* m_cirNormDelay;  // OWNING: Contiguous CIR indices for internal allocation  
    uint16_t* m_cirNtaps;      // OWNING: Contiguous number of CIR taps for internal allocation
    Tcomplex* m_freqChanPrbg;  // OWNING: Contiguous CFR on PRBG data for internal allocation
    Tcomplex* m_freqChanSc;    // OWNING: Contiguous CFR on SC data for internal allocation
    
    /** NON-OWNING POINTERS - External per-cell views (for mode 0 and API compatibility) */
    // IMPORTANT: These vectors contain raw pointers that DO NOT own the memory.
    // The memory is owned and managed by external code (user-provided arrays).
    // Do NOT delete these pointers - they are views into externally managed memory.
    std::vector<Tcomplex*> m_cirCoePerCell;         // NON-OWNING: Per-cell views into external CIR arrays
    std::vector<uint16_t*> m_cirNormDelayPerCell;   // NON-OWNING: Per-cell views into external delay arrays
    std::vector<uint16_t*> m_cirNtapsPerCell;       // NON-OWNING: Per-cell views into external ntaps arrays
    std::vector<Tcomplex*> m_freqChanPrbgPerCell;   // NON-OWNING: Per-cell views into external PRBG arrays
    std::vector<Tcomplex*> m_freqChanScPerCell;     // NON-OWNING: Per-cell views into external SC arrays
    
    /** TODO: Consider clarifying ownership of this pointer */
    Tcomplex* m_rxSigOut;    // Raw pointer for received signal data (ownership TBD)

    // antenna panel configuration
    const std::vector<AntPanelConfig>* m_antPanelConfig; ///< Pointer to the active antenna panel config (external or owned)
    std::vector<AntPanelConfig>  m_ownAntPanelConfig; ///< Owned antenna panel config if no external config is provided 

    // Add link parameters
    // for generate correlated random variables
    float m_maxX;
    float m_minX;
    float m_maxY;
    float m_minY;
    std::vector<std::vector<std::vector<std::vector<float>>>> m_crnLos;  // [site][LSP][x][y]: SF, K, DS, ASD, ASA, ZSD, ZSA per site
    std::vector<std::vector<std::vector<std::vector<float>>>> m_crnNlos;  // [site][LSP][x][y]: SF, DS, ASD, ASA, ZSD, ZSA per site
    std::vector<std::vector<std::vector<std::vector<float>>>> m_crnO2i;  // [site][LSP][x][y]: SF, DS, ASD, ASA, ZSD, ZSA per site

    cmnLinkParams_t m_cmnLinkParams;
    std::vector<linkParams_t> m_linkParams;  // Link-specific parameters

    std::vector<ClusterParams> m_clusterParams;  // Cluster parameters for each link

    // Random number generators
    std::mt19937 m_gen;  // Mersenne Twister random number generator
    std::uniform_real_distribution<float> m_uniformDist;  // Uniform distribution for [0,1)
    std::normal_distribution<float> m_normalDist;  // Normal distribution with mean 0 and std dev 1

    // active link indices
    std::vector<activeLink<Tcomplex>> m_activeLinkParams;
    
    // Store active cell and UE mappings for H5 file generation
    std::vector<uint16_t> m_activeCell;
    std::vector<std::vector<uint16_t>> m_activeUt;

    // GPU buffers
    // Additional GPU memory pointers for static allocations
    CellParam* m_d_cellParams;
    UtParam* m_d_utParams;
    SystemLevelConfig* m_d_sysConfig;
    SimConfig* m_d_simConfig;
    CmnLinkParams* m_d_cmnLinkParams;
    LinkParams* m_d_linkParams;
    ClusterParams* m_d_clusterParams;
    
    // Additional pointers for small-scale functions
    AntPanelConfig* m_d_antPanelConfigs;
    activeLink<Tcomplex>* m_d_activeLinkParams;
    
    // Common Random Number (CRN) arrays for correlated LSP generation
    float** m_d_crnLos;   // CRN for LOS scenarios [nSite * 7 LSPs] - indexed as [siteIdx * 7 + lspIdx]
    float** m_d_crnNlos;  // CRN for NLOS scenarios [nSite * 6 LSPs] - indexed as [siteIdx * 6 + lspIdx]
    float** m_d_crnO2i;  // CRN for O2I scenarios [nSite * 6 LSPs] - indexed as [siteIdx * 6 + lspIdx]
    uint32_t m_crnSeed;   // Seed for CRN generation
    uint32_t m_crnGridSize; // Grid size for CRN (calculated from spatial bounds)
    uint16_t m_crnAllocatedNSite; // Number of sites for which CRN was allocated (to detect size changes)
    bool m_crnGridsAllocated; // Track if individual CRN grids have been allocated
    
    // GPU correlation distance arrays (allocated on GPU)
    float* m_d_corrDistLos;   // Correlation distances for LOS case [7 LSPs]
    float* m_d_corrDistNlos;  // Correlation distances for NLOS case [6 LSPs]
    float* m_d_corrDistO2i;  // Correlation distances for O2I case [6 LSPs]
    
    // Universal curandState array for consistent random number generation across kernel launches
    curandState* m_d_curandStates;  // OWNING: Pre-initialized curandState array for all threads
    uint32_t m_maxCurandStates;     // Maximum number of curandState elements allocated
};
