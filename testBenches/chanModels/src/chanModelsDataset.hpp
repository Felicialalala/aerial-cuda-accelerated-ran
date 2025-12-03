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

#ifndef CHAN_MODELS_DATASET_HPP
#define CHAN_MODELS_DATASET_HPP

#include <vector>
#include <cstdint>  // for uint8_t, uint16_t, uint32_t
#include <cuda_runtime.h>
#include <cuda_fp16.h>  // For __half and __half2
#include <string>
#include <random>  // For random number generation
#include <array>
#include <algorithm>  // for std::copy
#include <stdexcept>  // for std::invalid_argument

// Scenario enum
enum class Scenario {
    UMa,
    UMi,
    RMa,
    Indoor,  // TODO: Not supported yet
    InF,     // TODO: Not supported yet
    SMa      // TODO: Not supported yet
};

// Coordinate structure
struct Coordinate {
    float x = 0.0f;  // x-coordinate in global coordinate system
    float y = 0.0f;  // y-coordinate in global coordinate system
    float z = 0.0f;  // z-coordinate in global coordinate system
    
    // Default constructor
    Coordinate() = default;
    
    // Constructor with parameters
    Coordinate(float x, float y, float z) : x(x), y(y), z(z) {}
};

// Antenna panel parameters
struct AntPanelConfig {
    uint16_t nAnt = 4;  // Number of antennas in the array
    uint16_t antSize[5] = {1, 1, 1, 2, 2};  // Dimensions (M_g,N_g,M,N,P)
    float antSpacing[4] = {0, 0, 0.5, 0.5};  // Spacing in wavelengths
    float antTheta[181] = {0.0f};  // Antenna pattern A(theta, phi=0) in dB, size should be 181 (0-180 degrees)
    float antPhi[360] = {0.0f};    // Antenna pattern A(theta=90, phi) in dB, size should be 360 (0-360 degrees)
    float antPolarAngles[2] = {45, -45};  // Polar angles
    uint8_t antModel = 1;  // 0: isotropic, 1: directional, 2: direct pattern
    
    // Default constructor
    AntPanelConfig() = default;
    
    // Basic constructor
    AntPanelConfig(uint16_t nAnt, uint8_t antModel = 1) : nAnt(nAnt), antModel(antModel) {}
    
    // Constructor with antenna size array for antMode 0 and 1
    // report error if antModel is not 0 or 1
    AntPanelConfig(uint16_t nAnt, const std::array<uint16_t, 5>& antSize, const std::array<float, 4>& antSpacing, const std::array<float, 2>& antPolarAngles, uint8_t antModel = 1) {
        if (antModel != 0 && antModel != 1) {
            throw std::invalid_argument("antModel must be 0 or 1");
        }
        this->nAnt = nAnt;
        this->antModel = antModel;
        std::copy(antSize.begin(), antSize.end(), this->antSize);
        std::copy(antSpacing.begin(), antSpacing.end(), this->antSpacing);
        std::copy(antPolarAngles.begin(), antPolarAngles.end(), this->antPolarAngles);
    }
    
    // Constructor with antenna size array using direct pattern
    AntPanelConfig(uint16_t nAnt, const std::array<uint16_t, 5>& antSize, const std::array<float, 4>& antSpacing, const std::array<float, 181>& antTheta, const std::array<float, 360>& antPhi, const std::array<float, 2>& antPolarAngles, uint8_t antModel = 2) 
        : nAnt(nAnt), antModel(antModel) {
        std::copy(antSize.begin(), antSize.end(), this->antSize);
        std::copy(antSpacing.begin(), antSpacing.end(), this->antSpacing);
        std::copy(antTheta.begin(), antTheta.end(), this->antTheta);
        std::copy(antPhi.begin(), antPhi.end(), this->antPhi);
        std::copy(antPolarAngles.begin(), antPolarAngles.end(), this->antPolarAngles);
    }
};

// UT parameters for public API (user-configurable parameters only)
struct UtParamCfg {
    uint32_t uid = 0;  // Global UE ID
    Coordinate loc;  // UE location
    uint8_t outdoor_ind = 0;  // 0: indoor, 1: outdoor
    uint32_t antPanelIdx = 0;  // Antenna panel configuration index
    float antPanelOrientation[3] = {0, 0, 0};  // (theta, phi, slant offset)
    float velocity[3] = {0, 0, 0};  // (vx, vy, vz), abs(velocity_direction) = speed in m/s, vz = 0 per 3GPP spec)
    
    // Default constructor
    UtParamCfg() = default;
    
    // Basic constructor
    UtParamCfg(uint32_t uid, const Coordinate& loc, uint8_t outdoor_ind = 0, uint32_t antPanelIdx = 0)
        : uid(uid), loc(loc), outdoor_ind(outdoor_ind), antPanelIdx(antPanelIdx) {}
    
    // Constructor with velocity and antenna panel orientation
    UtParamCfg(uint32_t uid, const Coordinate& loc, uint8_t outdoor_ind, uint32_t antPanelIdx, const float antPanelOrientation[3], const float velocity[3])
        : uid(uid), loc(loc), outdoor_ind(outdoor_ind), antPanelIdx(antPanelIdx) {
        this->antPanelOrientation[0] = antPanelOrientation[0];
        this->antPanelOrientation[1] = antPanelOrientation[1];
        this->antPanelOrientation[2] = antPanelOrientation[2];
        this->velocity[0] = velocity[0];
        this->velocity[1] = velocity[1];
        this->velocity[2] = velocity[2];
    }
};

// UT parameters for internal implementation (inherits public API + adds internal fields)
struct UtParam : public UtParamCfg {
    float d_2d_in = -1.0f;  //!< 2D distance of an indoor UE (internal use, default -1)
    float o2i_penetration_loss = 0.0f;  //!< O2I building penetration loss in dB (UT-specific, same for all BSs per 3GPP TR 38.901 Section 7.4.3)
    
    // Default constructor
    UtParam() = default;
    
    // Constructor from public API struct
    UtParam(const UtParamCfg& cfg) : UtParamCfg(cfg) {}
};

// Cell parameters
struct CellParam {
    uint32_t cid = 0;  // Global cell ID
    uint32_t siteId = 0;  // Site ID for LSP access
    Coordinate loc;  // Cell location
    uint32_t antPanelIdx = 0;  // Antenna panel configuration index
    float antPanelOrientation[3] = {0, 0, 0};  // (theta, phi, slant offset)
    
    // Default constructor
    CellParam() = default;
    
    // Basic constructor
    CellParam(uint32_t cid, uint32_t siteId, Coordinate loc, uint32_t antPanelIdx)
        : cid(cid), siteId(siteId), loc(loc), antPanelIdx(antPanelIdx) {}
    
    // Constructor with parameters
    CellParam(uint32_t cid, uint32_t siteId, Coordinate loc, uint32_t antPanelIdx, const float antPanelOrientation[3]) 
        : cid(cid), siteId(siteId), loc(loc), antPanelIdx(antPanelIdx) {
        this->antPanelOrientation[0] = antPanelOrientation[0];
        this->antPanelOrientation[1] = antPanelOrientation[1];
        this->antPanelOrientation[2] = antPanelOrientation[2];
    }
};

// System-level configuration
struct SystemLevelConfig {
    Scenario scenario = Scenario::UMa;
    float isd = 1732.0f;  // Inter-site distance in meters
    uint32_t n_site = 1;  // Number of sites
    uint8_t n_sector_per_site = 3;  // Sectors per site
    uint32_t n_ut = 100;  // Total number of UTs
    uint8_t optional_pl_ind = 0;  // 0: standard pathloss, 1: optional pathloss
    uint8_t o2i_building_penetr_loss_ind = 1;  // 0: none, 1: low-loss, 2: 50% low-loss, 50% high-loss, 3: 100% high-loss
    uint8_t o2i_car_penetr_loss_ind = 0;  // 0: none, 1: basic, 2: 50% basic, 50% metallized, 3: 100% metallized
    uint8_t enable_near_field_effect = 0;  // 0: disable, 1: enable
    uint8_t enable_non_stationarity = 0;  // 0: disable, 1: enable
    float force_los_prob[2] = {-1, -1};  // Force LOS probability
    float force_ut_speed[2] = {-1, -1};  // Force UT speed in m/s
    float force_indoor_ratio = -1;  // Force indoor ratio
    uint8_t disable_pl_shadowing = 0;  // 0: calculate, 1: disable
    uint8_t disable_small_scale_fading = 0;  // 0: calculate, 1: disable
    uint8_t enable_per_tti_lsp = 1;  // 0: disable, 1: update PL/O2I/shadowing, 2: update all
    uint8_t enable_propagation_delay = 1;  // 0: disable propagation delay in CIR generation, 1: enable propagation delay in CIR generation. Propagation delay is link-specific, distance / speed of light
    
    // Default constructor
    SystemLevelConfig() = default;
    
    // Basic constructor
    SystemLevelConfig(Scenario scenario, uint32_t n_site, uint8_t n_sector_per_site, uint32_t n_ut, float isd = 1732.0f)
        : scenario(scenario), isd(isd), n_site(n_site), n_sector_per_site(n_sector_per_site), n_ut(n_ut) {}
    
    // Constructor with full parameters
    SystemLevelConfig(Scenario scenario, uint32_t n_site, uint8_t n_sector_per_site, uint32_t n_ut, float isd, uint8_t optional_pl_ind, uint8_t o2i_building_penetr_loss_ind, uint8_t o2i_car_penetr_loss_ind, uint8_t enable_near_field_effect, uint8_t enable_non_stationarity, float force_los_prob[2], float force_ut_speed[2], float force_indoor_ratio, uint8_t disable_pl_shadowing, uint8_t disable_small_scale_fading, uint8_t enable_per_tti_lsp, uint8_t enable_propagation_delay)
        : scenario(scenario), isd(isd), n_site(n_site), n_sector_per_site(n_sector_per_site), n_ut(n_ut) {
        this->optional_pl_ind = optional_pl_ind;
        this->o2i_building_penetr_loss_ind = o2i_building_penetr_loss_ind;
        this->o2i_car_penetr_loss_ind = o2i_car_penetr_loss_ind;
        this->enable_near_field_effect = enable_near_field_effect;
        this->enable_non_stationarity = enable_non_stationarity;
        this->force_los_prob[0] = force_los_prob[0];
        this->force_los_prob[1] = force_los_prob[1];
        this->force_ut_speed[0] = force_ut_speed[0];
        this->force_ut_speed[1] = force_ut_speed[1];
        this->force_indoor_ratio = force_indoor_ratio;
        this->disable_pl_shadowing = disable_pl_shadowing;
        this->disable_small_scale_fading = disable_small_scale_fading;
        this->enable_per_tti_lsp = enable_per_tti_lsp;
        this->enable_propagation_delay = enable_propagation_delay;
    }
};

// Link-level configuration
struct LinkLevelConfig {
    int fast_fading_type = 0;  // 0: AWGN, 1: TDL, 2: CDL
    char delay_profile = 'A';  // 'A' to 'C'
    float delay_spread = 30.0f;  // in nanoseconds
    float velocity[3] = {0, 0, 0};  // (vx, vy, vz), abs(velocity_direction) = speed in m/s, vz = 0 per 3GPP spec
    int num_ray = 20;  // number of rays to add per path; defualt 48 for TDL, 20 for CDL
    float cfo_hz = 200.0f;  // carrier frequency offset in Hz
    float delay = 0.0f;  // delay in seconds
    
    // Default constructor
    LinkLevelConfig() = default;
    
    // Basic constructor
    LinkLevelConfig(int fast_fading_type, char delay_profile = 'A', float delay_spread = 30.0f)
        : fast_fading_type(fast_fading_type), delay_profile(delay_profile), delay_spread(delay_spread) {}

    // Constructor with velocity
    LinkLevelConfig(int fast_fading_type, char delay_profile, float delay_spread, const float velocity[3])
        : fast_fading_type(fast_fading_type), delay_profile(delay_profile), delay_spread(delay_spread) {
        this->velocity[0] = velocity[0];
        this->velocity[1] = velocity[1];
        this->velocity[2] = velocity[2];
    }
};

// Test configuration
struct SimConfig {
    int link_sim_ind = 0;  // Link simulation indicator
    float center_freq_hz = 3e9f;  // Center frequency
    float bandwidth_hz = 100e6f;  // Bandwidth
    float sc_spacing_hz = 15e3f * 2;  // Subcarrier spacing
    int fft_size = 4096;  // FFT size
    int n_prb = 273;  // Number of PRBs
    int n_prbg = 137;  // Number of PRBGs
    int n_snapshot_per_slot = 1;  // Channel realizations per slot
    int run_mode = 0;  // 0: CIR only, 1: CIR+CFR on PRBG, 2: CIR+CFR on PRB/Sc
    int internal_memory_mode = 0;  // 0: external memory, 1: internal memory
    int freq_convert_type = 1;  // Frequency conversion type
    int sc_sampling = 1;  // Subcarrier sampling
    float * tx_sig_in = nullptr;  // Input signal for transmission
    int proc_sig_freq = 0;  // Signal processing frequency indicator
    int optional_cfr_dim = 0;  // Optional CFR dimension: 0: [nActiveUtForThisCell, n_snapshot_per_slot, nUtAnt, nBsAnt, nPrbg / nSc], 1: [nActiveUtForThisCell, n_snapshot_per_slot, nPrbg / nSc, nUtAnt, nBsAnt]
    int cpu_only_mode = 0;  // 0: GPU mode, 1: CPU only mode
    
    // Default constructor
    SimConfig() = default;
    
    // Constructor with basic parameters  
    SimConfig(float center_freq_hz, float bandwidth_hz, int run_mode = 0)
        : center_freq_hz(center_freq_hz), bandwidth_hz(bandwidth_hz), run_mode(run_mode) {
        // Set reasonable defaults for other fields
        this->n_prb = 273;
        this->n_prbg = 137;
        this->n_snapshot_per_slot = 1;
    }
    
    // Constructor with detailed parameters
    SimConfig(float center_freq_hz, float bandwidth_hz, float sc_spacing_hz, int fft_size, int run_mode = 0)
        : center_freq_hz(center_freq_hz), bandwidth_hz(bandwidth_hz), sc_spacing_hz(sc_spacing_hz), 
          fft_size(fft_size), run_mode(run_mode) {
        // Set reasonable defaults for other fields
        this->n_prb = 273;
        this->n_prbg = 137;
        this->n_snapshot_per_slot = 1;
    }
    
    // Constructor with full parameters (fixed member initialization order)
    SimConfig(int link_sim_ind, float center_freq_hz, float bandwidth_hz, float sc_spacing_hz, int fft_size, 
        int n_prb, int n_prbg, int n_snapshot_per_slot, int run_mode, int internal_memory_mode, int freq_convert_type, int sc_sampling, float * tx_sig_in = nullptr, 
        int proc_sig_freq = 0, int optional_cfr_dim = 0, int cpu_only_mode = 0)
        : link_sim_ind(link_sim_ind), center_freq_hz(center_freq_hz), bandwidth_hz(bandwidth_hz), 
          sc_spacing_hz(sc_spacing_hz), fft_size(fft_size), n_prb(n_prb), n_prbg(n_prbg), 
          n_snapshot_per_slot(n_snapshot_per_slot), run_mode(run_mode), internal_memory_mode(internal_memory_mode), 
          freq_convert_type(freq_convert_type), sc_sampling(sc_sampling), tx_sig_in(tx_sig_in), proc_sig_freq(proc_sig_freq), optional_cfr_dim(optional_cfr_dim), cpu_only_mode(cpu_only_mode) {
    }
};

// External configuration (public API - uses UtParamCfg for user interface)
struct ExternalConfig {
    std::vector<CellParam> cell_config;  // Cell configuration
    std::vector<UtParamCfg> ut_config;  // UT configuration (public API struct)
    std::vector<AntPanelConfig> ant_panel_config;  // Antenna panel configurations

    // Default constructor
    ExternalConfig() = default;

    // Constructor with parameters
    ExternalConfig(const std::vector<CellParam>& cell_config, const std::vector<UtParamCfg>& ut_config, const std::vector<AntPanelConfig>& ant_panel_config)
        : cell_config(cell_config), ut_config(ut_config), ant_panel_config(ant_panel_config) {}
};

#endif // CHAN_MODELS_DATASET_HPP