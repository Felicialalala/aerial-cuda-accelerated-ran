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

#ifndef CHAN_MODELS_API_HPP
#define CHAN_MODELS_API_HPP

#include <vector>
#include <cuda_runtime.h>
#include <string>
#include <memory>
#include "chanModelsDataset.hpp"
#include "tdl_chan_src/tdl_chan.cuh"
#include "cdl_chan_src/cdl_chan.cuh"
#include "sls_chan_src/sls_chan.cuh"
#include "config_reader.hpp"

// Main channel model class
template <typename Tscalar, typename Tcomplex>
class statisChanModel {
public:
    statisChanModel(const SimConfig* sim_config,
                const SystemLevelConfig* system_level_config,
                const LinkLevelConfig* link_level_config,
                const ExternalConfig* external_config,
                uint32_t randSeed,
                cudaStream_t strm = nullptr);

    ~statisChanModel() {
        // Only destroy the stream if we created it (i.e., if it wasn't passed in)
        if (m_owns_stream && m_strm != nullptr) {
            cudaError_t st = cudaStreamDestroy(m_strm);
            if (st != cudaSuccess) {
                // Log error instead of throwing (destructors should not throw)
                fprintf(stderr, "Warning: statisChanModel cudaStreamDestroy failed: %s\n", 
                        cudaGetErrorString(st));
            }
        }
    }

    // Delete copy constructor and assignment operator
    statisChanModel(const statisChanModel&) = delete;
    statisChanModel& operator=(const statisChanModel&) = delete;

    // for system level simulation
    void run(const float refTime = 0.0f,
             const uint8_t continuous_fading = 1,
             const std::vector<uint16_t>& activeCell = {},
             const std::vector<std::vector<uint16_t>>& activeUt = {},
             const std::vector<Coordinate>& utNewLoc = {},
             const std::vector<float3>& utNewVelocity = {},
             const std::vector<Tcomplex*>& cir_coe = {},
             const std::vector<uint16_t*>& cir_norm_delay = {},
             const std::vector<uint16_t*>& cir_n_taps = {},
             const std::vector<Tcomplex*>& cfr_sc = {},
             const std::vector<Tcomplex*>& cfr_prbg = {});

    // for link level simulation
    void run(const float refTime0 = 0.0f,
             const uint8_t continuous_fading = 1,
             const uint8_t enableSwapTxRx = 0,
             const uint8_t txColumnMajorInd = 0);

    void reset();
    void dump_los_nlos_stats(float* lost_nlos_stats = nullptr);
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
    void dump_topology_to_yaml(const std::string& filename);
    
    /**
     * Save SLS channel data to H5 file for debugging
     * 
     * @param filenameEnding Optional string to append to filename
     */
    void saveSlsChanToH5File(std::string_view filenameEnding = "");

private:
    const SimConfig* m_sim_config;
    const SystemLevelConfig* m_system_level_config;
    const LinkLevelConfig* m_link_level_config;
    const ExternalConfig* m_external_config;
    uint32_t m_rand_seed;
    cudaStream_t m_strm = nullptr;
    bool m_owns_stream = false;  // Flag to track if we created the stream
    
    // link level channel models, TDL and CDL
    // TODO: add AWGN channel model
    // TDL channel model
    std::unique_ptr<tdlConfig_t> m_tdl_chan_cfg;
    std::unique_ptr<tdlChan<Tscalar, Tcomplex>> m_tdl_chan;
    // CDL channel model
    std::unique_ptr<cdlConfig_t> m_cdl_chan_cfg;
    std::unique_ptr<cdlChan<Tscalar, Tcomplex>> m_cdl_chan;
    // system level channel models
    // support UMa, UMi, RMa
    std::unique_ptr<slsChan<Tscalar, Tcomplex>> m_sls_chan;
};

// Explicit template instantiations 
template class statisChanModel<float, cuComplex>;
// template class statisChanModel<__half, __half2>;  // Disabled for now

// Custom exception class for channel model errors
class ChannelModelError : public std::runtime_error {
public:
    explicit ChannelModelError(const std::string& message) 
        : std::runtime_error("Channel Model Error: " + message) {}
};

#endif // CHAN_MODELS_API_HPP