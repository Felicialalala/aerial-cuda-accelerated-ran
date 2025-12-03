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
#include "sls_table.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand.h>
#include <cmath>
#include <cassert>
#include <cstdint>
#include <random>
#include <string>
#include <vector>
#include <algorithm> // For std::min
#include "antenna_config_reader.hpp"

// Correlation matrices
extern const float sqrtCorrMatUmaLos[LOS_MATRIX_SIZE][LOS_MATRIX_SIZE];
extern const float sqrtCorrMatUmaNlos[NLOS_MATRIX_SIZE][NLOS_MATRIX_SIZE];
extern const float sqrtCorrMatUmaO2i[O2I_MATRIX_SIZE][O2I_MATRIX_SIZE];
extern const float sqrtCorrMatUmiLos[LOS_MATRIX_SIZE][LOS_MATRIX_SIZE];
extern const float sqrtCorrMatUmiNlos[NLOS_MATRIX_SIZE][NLOS_MATRIX_SIZE];
extern const float sqrtCorrMatUmiO2i[O2I_MATRIX_SIZE][O2I_MATRIX_SIZE];
extern const float sqrtCorrMatRmaLos[LOS_MATRIX_SIZE][LOS_MATRIX_SIZE];
extern const float sqrtCorrMatRmaNlos[NLOS_MATRIX_SIZE][NLOS_MATRIX_SIZE];
extern const float sqrtCorrMatRmaO2i[O2I_MATRIX_SIZE][O2I_MATRIX_SIZE];

// Correlation distances
extern const corrDist_t corrDistUmaLos;
extern const corrDist_t corrDistUmaNlos;
extern const corrDist_t corrDistUmiLos;
extern const corrDist_t corrDistUmiNlos;
extern const corrDist_t corrDistRmaLos;
extern const corrDist_t corrDistRmaNlos;


// Path loss calculations
inline float calculateUMaLosPathloss(float d_2d, float d_3d, float h_bs, float h_ut, float fc,
                                   std::mt19937& gen, std::uniform_real_distribution<float>& uniformDist) {
#ifdef CALIBRATION_CFG_
    if (!(d_2d >= 0.0f && d_2d <= 5000.0f)) {
        printf("UMa LOS d_2d out of range: d_2d=%.3f, d_3d=%.3f, h_bs=%.3f, h_ut=%.3f, fc=%.3f\n", d_2d, d_3d, h_bs, h_ut, fc);
    }
    assert(d_2d >= 0.0f && d_2d <= 5000.0f && "2D distance must be between 0m and 5000m");
#else
    if (!(d_2d >= 10.0f && d_2d <= 5000.0f)) {
        printf("UMa LOS d_2d out of range: d_2d=%.3f, d_3d=%.3f, h_bs=%.3f, h_ut=%.3f, fc=%.3f\n", d_2d, d_3d, h_bs, h_ut, fc);
    }
    assert(d_2d >= 10.0f && d_2d <= 5000.0f && "2D distance must be between 10m and 5000m");
#endif
    float d_2d_valid = std::max(d_2d, 10.0f);
    float g_d2d = d_2d_valid <= 18.0f ? 0.0f : 5.0f/4.0f * powf(d_2d_valid / 100.0f, 3.0f) * expf(-d_2d_valid / 150.0f);
    float c_d2d_hut = h_ut < 13.0f ? 0.0f : powf((h_ut - 13.0f) / 10.0f, 1.5f) * g_d2d;
    float prob_h_e = 1.0f / (1.0f + c_d2d_hut);
    
    float h_e;
    if (uniformDist(gen) <= prob_h_e) {
        h_e = 1.0f;  // With probability 1/(1+C(d2D, hUT))
    } else {
        // Generate from discrete uniform distribution uniform(12,15,...,(hUT-1.5))
        float max_h_e = h_ut - 1.5f;
        int n_steps = static_cast<int>((max_h_e - 12.0f) / 3.0f) + 1;
        int step = static_cast<int>(uniformDist(gen) * n_steps);
        h_e = 12.0f + step * 3.0f;
    }
    
    float d_bp_prime = 4.0f * (h_bs - h_e) * (h_ut - h_e) * fc * 10.0f/ 3.0f;  // fc is in GHz, fc*1e9/3e8 = d_bp_prime
    float pl1 = 28.0f + 22.0f * log10f(d_3d) + 20.0f * log10f(fc);
    float pl2 = 28.0f + 40.0f * log10f(d_3d) + 20.0f * log10f(fc) - 9.0f * log10f(d_bp_prime * d_bp_prime + powf(h_bs - h_ut, 2));
    return (d_2d <= d_bp_prime) ? pl1 : pl2;
}

inline float calculateUMiLosPathloss(float d_2d, float d_3d, float h_bs, float h_ut, float fc) {
    float d_bp_prime = 4.0f * h_bs * h_ut * fc * 10.0f/ 3.0f;  // fc is in GHz, fc*1e9/3e8 = d_bp_prime
    float pl1 = 32.4f + 21.0f * log10f(d_3d) + 20.0f * log10f(fc);
    float pl2 = 32.4f + 40.0f * log10f(d_3d) + 20.0f * log10f(fc) - 9.5f * log10f(d_bp_prime * d_bp_prime + powf(h_bs - h_ut, 2));
#ifdef CALIBRATION_CFG_
    if (!(d_2d >= 0.0f && d_2d <= 5000.0f)) {
        printf("UMi LOS d_2d out of range: d_2d=%.3f, d_3d=%.3f, h_bs=%.3f, h_ut=%.3f, fc=%.3f\n", d_2d, d_3d, h_bs, h_ut, fc);
    }
    assert(d_2d >= 0.0f && d_2d <= 5000.0f && "2D distance must be between 0m and 5000m");
#else
    if (!(d_2d >= 10.0f && d_2d <= 5000.0f)) {
        printf("UMi LOS d_2d out of range: d_2d=%.3f, d_3d=%.3f, h_bs=%.3f, h_ut=%.3f, fc=%.3f\n", d_2d, d_3d, h_bs, h_ut, fc);
    }
    assert(d_2d >= 10.0f && d_2d <= 5000.0f && "2D distance must be between 10m and 5000m");
#endif
    return (d_2d <= d_bp_prime) ? pl1 : pl2;
}

inline float calculateRMaLosPathloss(float d_2d, float d_3d, float h_bs, float h_ut, float fc) {
    float d_bp = 2.0f * M_PI * h_bs * h_ut * fc * 10.0f/ 3.0f;  // Breakpoint distance
    constexpr float h = 5.0f;  // Average building height
    float pl1 = 20.0f * log10f(40.0f * M_PI * d_3d * fc / 3.0f) + 
                std::min(0.03f * powf(h, 1.72f), 10.0f) * log10f(d_3d) - 
                std::min(0.044f * powf(h, 1.72f), 14.77f) + 
                0.002f * log10f(h) * d_3d;
    float pl = d_2d <= d_bp ? pl1 : pl1 + 40.0f * log10f(d_3d / d_bp);
#ifdef CALIBRATION_CFG_
    if (!(d_2d >= 0.0f && d_2d <= 10000.0f)) {
        printf("RMa LOS d_2d out of range: d_2d=%.3f, d_3d=%.3f, h_bs=%.3f, h_ut=%.3f, fc=%.3f\n", d_2d, d_3d, h_bs, h_ut, fc);
    }
    assert(d_2d >= 0.0f && d_2d <= 10000.0f && "2D distance must be between 0m and 10000m");
#else
    if (!(d_2d >= 10.0f && d_2d <= 10000.0f)) {
        printf("RMa LOS d_2d out of range: d_2d=%.3f, d_3d=%.3f, h_bs=%.3f, h_ut=%.3f, fc=%.3f\n", d_2d, d_3d, h_bs, h_ut, fc);
    }
    assert(d_2d >= 10.0f && d_2d <= 10000.0f && "2D distance must be between 10m and 10000m");
#endif
    return pl;
}

// Helper functions for LSP calculation
inline void calDist(const CellParam& bs, const UtParam& ut, float& d_2d, float& d_3d, float& d_2d_in, float& d_2d_out, float& d_3d_in, float& d_3d_out) {
    // Calculate total 2D distance
    d_2d = sqrtf(powf(bs.loc.x - ut.loc.x, 2) + powf(bs.loc.y - ut.loc.y, 2));
    
    // Use the pre-calculated indoor distance from UT parameters
    d_2d_in = ut.d_2d_in;
    
    // Calculate outdoor 2D distance
    d_2d_out = d_2d - d_2d_in;
    
    // Calculate vertical distance
    float vertical_dist = bs.loc.z - ut.loc.z;
    
    // Calculate all 3D distances
    d_3d = sqrtf(d_2d * d_2d + vertical_dist * vertical_dist);
    
    // Guard against division by zero when calculating d_3d_in
    if (d_2d > 0.0f) {
        d_3d_in = d_3d * d_2d_in / d_2d;
    } else {
        // When BS and UT are at same (x,y) coordinates, no horizontal indoor distance
        d_3d_in = 0.0f;
    }
    d_3d_out = d_3d - d_3d_in;
}

inline float calLosProb(scenario_t scenario, float d_2d_out, float h_ut, const float force_los_prob[2], uint8_t outdoor_ind) {
    // Check if force_los_prob should be used instead of 3GPP calculations
    // force_los_prob[0] for indoor UTs, force_los_prob[1] for outdoor UEs
    float forced_prob = outdoor_ind ? force_los_prob[1] : force_los_prob[0];
    if (forced_prob >= 0.0f && forced_prob <= 1.0f) {
        return forced_prob;  // Use forced value instead of 3GPP calculation
    }
    
    // Use standard 3GPP LOS probability calculations
    float losProb = 0.0f;
    switch (scenario) {
        case scenario_t::UMa:
            assert(h_ut <= 23.0f && "UE height must be less than 23m");
            if (d_2d_out <= 18.0f) {
                losProb = 1.0f;
            } else {
                float c_prime = h_ut <= 13.0f ? 0.0f : std::pow((h_ut - 13.0f) / 10.0f, 1.5f);
                losProb = ((18.0f / d_2d_out) + std::exp(-d_2d_out / 63.0f) * (1.0f - 18.0f / d_2d_out)) *
                         (1.0f + c_prime * 5.0f / 4.0f * std::pow(d_2d_out / 100.0f, 3.0f) * std::exp(-d_2d_out / 150.0f));
            }
            break;
        case scenario_t::UMi:
            if (d_2d_out <= 18.0f) {
                losProb = 1.0f;
            } else {
                losProb = (18.0f / d_2d_out) + std::exp(-d_2d_out / 36.0f) * (1.0f - 18.0f / d_2d_out);
            }
            break;
        case scenario_t::RMa:
            if (d_2d_out <= 10.0f) {
                losProb = 1.0f;
            } else {
                losProb = std::exp(-(d_2d_out - 10.0f) / 1000.0f);
            }
            break;
        default:
            assert(false && "Unknown scenario");
            break;
    }
    return losProb;
}


inline float calSfStd(scenario_t scenario, bool isLos, float fc, bool optionalPlInd, float d_2d, float h_bs, float h_ut) {
    float sf_std = 0.0f;
    if (isLos) {
        switch (scenario) {
            case scenario_t::UMa:
            case scenario_t::UMi:
                sf_std = (fc < 6e9f ? 7.0f : 4.0f);
                break;
            case scenario_t::RMa: {
                float d_bp = 2 * M_PI * h_bs * h_ut * fc / 3.0e8f;
                sf_std = d_2d <= d_bp ? 4.0f : 6.0f;
                break;
            }
            default:
                assert(false && "Unknown scenario");
                break;
        }
    } else {
        switch (scenario) {
            case scenario_t::UMa:
                sf_std = (fc < 6e9f ? 7.0f : (optionalPlInd ? 7.8f : 6.0f));
                break;
            case scenario_t::UMi:
                sf_std = (fc < 6e9f ? 7.0f : (optionalPlInd ? 8.2f : 7.82f));
                break;
            case scenario_t::RMa:
                sf_std = 8.0f;
                break;
            default:
                assert(false && "Unknown scenario");
                break;
        }
    }
    return sf_std;
}

float calPenetrLos(scenario_t scenario, bool outdoor_ind, float fc, float d_2d_in, 
                   std::uint8_t o2iBuildingPenetrLossInd, std::uint8_t o2iCarPenetrLossInd,
                   std::mt19937& gen, 
                   std::uniform_real_distribution<float>& uniformDist,
                   std::normal_distribution<float>& normalDist) {
    float pl_pen = 0.0f;
    float L_glass, L_concreate, L_IRRglass;
    float pl_tw;
    // building penetration loss
    if (!outdoor_ind) {
        switch (scenario) {
            case scenario_t::UMa:
            case scenario_t::UMi: {
                // Building penetration loss according to 7.4.3.1
                if (fc < 6e9f) {
                    // Use Table 7.4.3-3 for frequencies below 6 GHz
                    if (o2iBuildingPenetrLossInd) {
                        float pl_tw = 20.0f;
                        float pl_in = 0.5f * 25.0f * uniformDist(gen);
                        pl_pen = pl_tw + pl_in;  // sigma_p = 0, no need to generate additional random variable
                    }
                } else {
                    // Use Table 7.4.3-2 for frequencies above 6 GHz
                    switch (o2iBuildingPenetrLossInd) {
                        case 0:  // No penetration loss
                            pl_pen = 0.0f;
                            break;
                        case 1:  // Low-loss building
                            L_glass = 2.0f + 0.2f * fc / 1e9f;
                            L_concreate = 5.0f + 4.0f * fc / 1e9f;
                            pl_tw = 5.0f - 10.0f * std::log10(0.3f * std::pow(10.0f, -0.1f * L_glass) + 0.7f * std::pow(10.0f, -0.1f * L_concreate));
                            pl_pen = pl_tw + 0.5f * d_2d_in + normalDist(gen) * 4.4f;
                            break;
                        case 2:  // 50% low-loss, 50% high-loss building
                            if (uniformDist(gen) < 0.5f) {
                                // Low-loss building
                                L_glass = 2.0f + 0.2f * fc / 1e9f;
                                L_concreate = 5.0f + 4.0f * fc / 1e9f;
                                pl_tw = 5.0f - 10.0f * std::log10(0.3f * std::pow(10.0f, -0.1f * L_glass) + 0.7f * std::pow(10.0f, -0.1f * L_concreate));
                                pl_pen = pl_tw + 0.5f * d_2d_in + normalDist(gen) * 4.4f;
                            } else {
                                // High-loss building
                                L_IRRglass = 23.0f + 0.3f * fc / 1e9f;
                                L_concreate = 5.0f + 4.0f * fc / 1e9f;    
                                pl_tw = 5.0f - 10.0f * std::log10(0.7f * std::pow(10.0f, -0.1f * L_IRRglass) + 0.3f * std::pow(10.0f, -0.1f * L_concreate));
                                pl_pen = pl_tw + 0.5f * d_2d_in + normalDist(gen) * 6.5f;
                            }
                            break;
                        case 3:  // 100% high-loss building
                            L_IRRglass = 23.0f + 0.3f * fc / 1e9f;
                            L_concreate = 5.0f + 4.0f * fc / 1e9f;    
                            pl_tw = 5.0f - 10.0f * std::log10(0.7f * std::pow(10.0f, -0.1f * L_IRRglass) + 0.3f * std::pow(10.0f, -0.1f * L_concreate));
                            pl_pen = pl_tw + 0.5f * d_2d_in + normalDist(gen) * 6.5f;
                            break;
                        default:
                            assert(false && "Unknown penetration loss index for UMa/UMi");
                            break;
                    }
                }
                break;
            }
            case scenario_t::RMa: {
                // Car penetration loss according to 7.4.3.2
                switch (o2iCarPenetrLossInd) {
                    case 0:  // No penetration loss
                        pl_pen = 0.0f;
                        break;
                    case 1:  // Low-loss building
                        L_glass = 2.0f + 0.2f * fc / 1e9f;
                        L_concreate = 5.0f + 4.0f * fc / 1e9f;
                        pl_tw = 5.0f - 10.0f * std::log10(0.3f * std::pow(10.0f, -0.1f * L_glass) + 0.7f * std::pow(10.0f, -0.1f * L_concreate));
                        pl_pen = pl_tw + 0.5f * d_2d_in + normalDist(gen) * 4.4f;
                        break;
                    default:
                        assert(false && "Unknown penetration loss index for RMa");
                        break;
                }
                break;
            }
            default:
                assert(false && "Unknown scenario");
                break;
        }
    }
    else if (scenario == scenario_t::RMa) {
        switch (o2iCarPenetrLossInd) {
            case 0:
                pl_pen = 0.0f;
                break;
            case 1:  // basic car penetration loss
                pl_pen = normalDist(gen) * 5.0f + 9.0f;
                assert(fc > 0.6e9f && fc <= 60e9f && "Frequency must be between 0.6 GHz and 60 GHz for RMa");
                break;
            case 2:  // 50% basic, 50% metallized car penetration loss
                if (uniformDist(gen) < 0.5f) {
                    // Basic car penetration loss
                    pl_pen = normalDist(gen) * 5.0f + 9.0f;
                } else {
                    // Metallized car window penetration loss
                    pl_pen = normalDist(gen) * 20.0f + 9.0f;
                }
                assert(fc > 0.6e9f && fc <= 60e9f && "Frequency must be between 0.6 GHz and 60 GHz for RMa");
                break;
            case 3:  // 100% metallized car window penetration loss
                pl_pen = normalDist(gen) * 20.0f + 9.0f;
                assert(fc > 0.6e9f && fc <= 60e9f && "Frequency must be between 0.6 GHz and 60 GHz for RMa");
                break;
            default:
                assert(false && "Unknown penetration loss index for RMa");
                break;
        }
    }
    
    return pl_pen;
}

inline float calPL(scenario_t scenario, bool isLos, float d_2d, float d_3d, float h_bs, float h_ut, float fc, 
                  bool optionalPlInd, std::mt19937& gen, std::uniform_real_distribution<float>& uniformDist) {
    float pl = 0.0f;
    if (isLos) {
        switch (scenario) {
            case scenario_t::UMa:
                pl = calculateUMaLosPathloss(d_2d, d_3d, h_bs, h_ut, fc, gen, uniformDist);
                break;
            case scenario_t::UMi:
                pl = calculateUMiLosPathloss(d_2d, d_3d, h_bs, h_ut, fc);
                break;
            case scenario_t::RMa:
                pl = calculateRMaLosPathloss(d_2d, d_3d, h_bs, h_ut, fc);
                break;
            default:
                assert(false && "Unknown scenario");
                break;
        }
    } else {
        if (optionalPlInd) {
            switch (scenario) {
                case scenario_t::UMa:
                    pl = 32.4f + 20.0f * std::log10(fc) + 30.0f * std::log10(d_3d);
                    break;
                case scenario_t::UMi:
                    pl = 32.4f + 20.0f * std::log10(fc) + 31.9f * std::log10(d_3d);
                    break;
                case scenario_t::RMa:
                    assert(false && "RMa does not support optional pathloss model");
                    break;
                default:
                    assert(false && "Unknown scenario");
                    break;
            }
        } else {
            switch (scenario) {
                case scenario_t::UMa: {
                    float los_pl = calculateUMaLosPathloss(d_2d, d_3d, h_bs, h_ut, fc, gen, uniformDist);
                    pl = std::max(los_pl, 13.54f + 39.08f * std::log10(d_3d) + 20.0f * std::log10(fc) - 0.6f * (h_ut - 1.5f));
                    break;
                }
                case scenario_t::UMi: {
                    float los_pl = calculateUMiLosPathloss(d_2d, d_3d, h_bs, h_ut, fc);
                    pl = std::max(los_pl, 35.3f * std::log10(d_3d) + 22.4f + 21.3f * std::log10(fc) - 0.3f * (h_ut - 1.5f));
                    break;
                }
                case scenario_t::RMa: {
                    float los_pl = calculateRMaLosPathloss(d_2d, d_3d, h_bs, h_ut, fc);
                    constexpr float W = 20.0f;  // Average street width
                    constexpr float h = 5.0f;  // Average building height
                    float nlos_pl = 161.04f - 7.1f * std::log10(W) + 7.5f * std::log10(h) - 
                                  (24.37f - 3.7f * std::pow(h/h_bs, 2)) * std::log10(h_bs) + 
                                  (43.42f - 3.1f * std::log10(h_bs)) * (std::log10(d_3d) - 3.0f) + 
                                  20.0f * std::log10(fc) - (3.2f * std::pow(std::log10(11.75f * h_ut), 2) - 4.97f);
                    pl = std::max(los_pl, nlos_pl);
                    break;
                }
                default:
                    assert(false && "Unknown scenario");
                    break;
            }
        }
    }
    return pl;
}

inline void calLosAngle(const CellParam& bs, const UtParam& ut, float d_3d, 
                       float& phi_los_aod, float& phi_los_aoa, 
                       float& theta_los_zod, float& theta_los_zoa) {
    // Calculate the vector from BS to UT in the horizontal plane
    float site2ut_x = ut.loc.x - bs.loc.x;
    float site2ut_y = ut.loc.y - bs.loc.y;
    
    // Calculate LOS AOD and AOA (azimuth angles)
    phi_los_aod = atan2(site2ut_y, site2ut_x) * 180.0f / M_PI;  // Convert to degrees
    phi_los_aoa = phi_los_aod + 180.0f;  // AOA is opposite to AOD
    
    // Normalize angles to [-180, 180] range
    if (phi_los_aoa > 180.0f) {
        phi_los_aoa -= 360.0f;
    }
    
    // Calculate LOS ZOD and ZOA (zenith angles)
    float h_diff = bs.loc.z - ut.loc.z;
    theta_los_zod = (M_PI - acos(h_diff / d_3d)) * 180.0f / M_PI;  // Convert to degrees
    theta_los_zoa = 180.0f - theta_los_zod;  // ZOA is complementary to ZOD
}

/**
 * Get scaling factor for AOA/AOD generation based on number of clusters (Table 7.5-2)
 * 
 * @param[in] nCluster Number of clusters
 * @return Scaling factor C_phi^NLOS, or 1.0f if cluster count not in table
 */
inline float getScalingFactorAoaAod(const uint16_t nCluster) {
    for (int i = 0; i < nScalingFactorsAoaAod; i++) {
        if (clusterCountsAoaAod[i] == nCluster) {
            return scalingFactorsAoaAod[i];
        }
    }
    // If not found in table, throw an error
    throw std::runtime_error("Cluster count not found in table for AOA/AOD scaling");
}

/**
 * Get scaling factor for ZOA/ZOD generation based on number of clusters (Table 7.5-4)
 * 
 * @param[in] nCluster Number of clusters
 * @return Scaling factor C_theta^NLOS, or 1.0f if cluster count not in table
 */
inline float getScalingFactorZoaZod(const uint16_t nCluster) {
    for (int i = 0; i < nScalingFactorsZoaZod; i++) {
        if (clusterCountsZoaZod[i] == nCluster) {
            return scalingFactorsZoaZod[i];
        }
    }
    // If not found in table, throw an error
    throw std::runtime_error("Cluster count not found in table for ZOA/ZOD scaling");
}

// Function to generate spatially correlated random numbers
inline void genCRN(float maxX, float minX, float maxY, float minY, 
                  float corrDist, std::mt19937& gen,
                  std::normal_distribution<float>& normalDist,
                  std::vector<std::vector<float>>& crn) {
    // Calculate padding based on correlation distance (3*CorrDist as in MATLAB)
    float D = 3.0f * corrDist;
    
    // Calculate grid dimensions with padding
    int nX = static_cast<int>(std::round(maxX - minX + 1.0f + 2.0f * D));
    int nY = static_cast<int>(std::round(maxY - minY + 1.0f + 2.0f * D));
    
    // Create exponential correlation filter
    std::vector<float> h;
    if (corrDist == 0.0f) {
        h = {1.0f};
    } else {
        h.resize(2 * static_cast<int>(D) + 1);
        for (int i = 0; i < h.size(); i++) {
            h[i] = std::exp(-std::abs(static_cast<float>(i - static_cast<int>(D))) / corrDist);
        }
    }
    
    // Generate Gaussian noise
    std::vector<std::vector<float>> gn(nX, std::vector<float>(nY));
    for (int i = 0; i < nX; i++) {
        for (int j = 0; j < nY; j++) {
            gn[i][j] = normalDist(gen);
        }
    }
    
    // Valid convolution: only compute the non-padded region
    int L = h.size();
    // Convolve rows (valid)
    std::vector<std::vector<float>> temp(nX, std::vector<float>(nY - L + 1, 0.0f));
    for (int i = 0; i < nX; ++i)
        for (int j = 0; j <= nY - L; ++j)
            for (int k = 0; k < L; ++k)
                temp[i][j] += gn[i][j + k] * h[k];

    // Convolve columns (valid)
    crn.resize(nX - L + 1, std::vector<float>(nY - L + 1, 0.0f));
    for (int j = 0; j < temp[0].size(); ++j)
        for (int i = 0; i <= nX - L; ++i)
            for (int k = 0; k < L; ++k)
                crn[i][j] += temp[i + k][j] * h[k];
    
    // Power normalization
    float power = 0.0f;
    for (const auto& row : crn) {
        for (float val : row) {
            power += val * val;
        }
    }
    power = std::sqrt(power / (crn.size() * crn[0].size()));
    
    for (auto& row : crn) {
        for (float& val : row) {
            val /= power;
        }
    }
}

// Function to get LSP value at a specific location
inline float getLspAtLocation(float x, float y, float maxX, float minX, float maxY, float minY, const std::vector<std::vector<float>>& crn) {
    // Calculate the normalized position within the grid
    float normX = (x - minX) / (maxX - minX);
    float normY = (y - minY) / (maxY - minY);

    // Clamp normalized coordinates to [0, 1] to match GPU boundary handling
    if (normX < 0.0f) normX = 0.0f;
    if (normX > 1.0f) normX = 1.0f;
    if (normY < 0.0f) normY = 0.0f;
    if (normY > 1.0f) normY = 1.0f;
    
    // Map to grid indices
    int nX = crn.size();
    int nY = crn[0].size();
    
    float gridX = normX * (nX - 1);
    float gridY = normY * (nY - 1);
    
    // Get the four nearest grid points
    int x0 = static_cast<int>(std::floor(gridX));
    int y0 = static_cast<int>(std::floor(gridY));
    int x1 = std::min(x0 + 1, nX - 1);
    int y1 = std::min(y0 + 1, nY - 1);
    
    // Get the fractional parts for interpolation
    float dx = gridX - x0;
    float dy = gridY - y0;
    
    // Perform bilinear interpolation
    float v00 = crn[x0][y0];
    float v10 = crn[x1][y0];
    float v01 = crn[x0][y1];
    float v11 = crn[x1][y1];
    
    float v0 = v00 * (1.0f - dx) + v10 * dx;
    float v1 = v01 * (1.0f - dx) + v11 * dx;
    
    return v0 * (1.0f - dy) + v1 * dy;
}


template <typename Tscalar, typename Tcomplex>
void slsChan<Tscalar, Tcomplex>::calCmnLinkParams()
{
    // Initialize common link parameters based on scenario
    float fc = m_simConfig->center_freq_hz;
    float lgfc = std::log10(fc / 1e9);
    float lg1fc = std::log10(1.0f + fc / 1e9);

    switch (m_sysConfig->scenario) {
        case scenario_t::UMa:
            // Note 6: fc < 6 GHz, fc = 6 GHz
            if (fc < 6e9) {
                fc = 6e9;
            }
            lgfc = std::log10(fc / 1e9);
            lg1fc = std::log10(1.0f + fc / 1e9);
            // Table 7.5-6 parameters
            m_cmnLinkParams.mu_lgDS[2] = -6.62f; // O2I
            m_cmnLinkParams.mu_lgDS[1] = -6.955f - 0.0963f * lgfc;  // LOS
            m_cmnLinkParams.mu_lgDS[0] = -6.28f - 0.204f * lgfc;    // NLOS
            m_cmnLinkParams.sigma_lgDS[2] = 0.32f;
            m_cmnLinkParams.sigma_lgDS[1] = 0.66f;
            m_cmnLinkParams.sigma_lgDS[0] = 0.39f;

            m_cmnLinkParams.mu_lgASD[2] = 1.25f;
            m_cmnLinkParams.mu_lgASD[1] = 1.06f + 0.1114f * lgfc;
            m_cmnLinkParams.mu_lgASD[0] = 1.5f - 0.1144f * lgfc;
            m_cmnLinkParams.sigma_lgASD[2] = 0.42f;
            m_cmnLinkParams.sigma_lgASD[1] = 0.28f;
            m_cmnLinkParams.sigma_lgASD[0] = 0.28f;
            
            m_cmnLinkParams.mu_lgASA[2] = 1.76f;
            m_cmnLinkParams.mu_lgASA[1] = 1.81f;
            m_cmnLinkParams.mu_lgASA[0] = 2.08f - 0.27f * lgfc;
            m_cmnLinkParams.sigma_lgASA[2] = 0.16f;
            m_cmnLinkParams.sigma_lgASA[1] = 0.2f;
            m_cmnLinkParams.sigma_lgASA[0] = 0.11f;
            
            m_cmnLinkParams.mu_lgZSA[2] = 1.01f;
            m_cmnLinkParams.mu_lgZSA[1] = 0.95f;
            m_cmnLinkParams.mu_lgZSA[0] = 1.512f - 0.3236f * lgfc;
            m_cmnLinkParams.sigma_lgZSA[2] = 0.43f;
            m_cmnLinkParams.sigma_lgZSA[1] = 0.16f;
            m_cmnLinkParams.sigma_lgZSA[0] = 0.16f;
            
            m_cmnLinkParams.mu_K[2] = 0.0f;
            m_cmnLinkParams.mu_K[1] = 9.0f;
            m_cmnLinkParams.mu_K[0] = 0.0f;
            m_cmnLinkParams.sigma_K[2] = 0.0f;
            m_cmnLinkParams.sigma_K[1] = 3.5f;
            m_cmnLinkParams.sigma_K[0] = 0.0f;
            
            m_cmnLinkParams.r_tao[2] = 2.2f;
            m_cmnLinkParams.r_tao[1] = 2.5f;
            m_cmnLinkParams.r_tao[0] = 2.3f;
            
            m_cmnLinkParams.mu_XPR[2] = 9.0f;
            m_cmnLinkParams.mu_XPR[1] = 8.0f;
            m_cmnLinkParams.mu_XPR[0] = 7.0f;
            m_cmnLinkParams.sigma_XPR[2] = 5.0f;
            m_cmnLinkParams.sigma_XPR[1] = 4.0f;
            m_cmnLinkParams.sigma_XPR[0] = 3.0f;
            
            m_cmnLinkParams.nCluster[2] = 12;
            m_cmnLinkParams.nCluster[1] = 12;
            m_cmnLinkParams.nCluster[0] = 20;
            m_cmnLinkParams.nRayPerCluster[2] = 20;
            m_cmnLinkParams.nRayPerCluster[1] = 20;
            m_cmnLinkParams.nRayPerCluster[0] = 20;
            
            m_cmnLinkParams.C_DS[2] = 11.0f;
            m_cmnLinkParams.C_DS[1] = std::max(0.25f, 6.5622f - 3.4084f * lgfc);
            m_cmnLinkParams.C_DS[0] = std::max(0.25f, 6.5622f - 3.4084f * lgfc);
            m_cmnLinkParams.C_ASD[2] = 5.0f;
            m_cmnLinkParams.C_ASD[1] = 5.0f;
            m_cmnLinkParams.C_ASD[0] = 2.0f;
            m_cmnLinkParams.C_ASA[2] = 8.0f;
            m_cmnLinkParams.C_ASA[1] = 11.0f;
            m_cmnLinkParams.C_ASA[0] = 15.0f;
            m_cmnLinkParams.C_ZSA[2] = 3.0f;
            m_cmnLinkParams.C_ZSA[1] = 7.0f;
            m_cmnLinkParams.C_ZSA[0] = 7.0f;
            m_cmnLinkParams.xi[2] = 4.0f;
            m_cmnLinkParams.xi[1] = 3.0f;
            m_cmnLinkParams.xi[0] = 3.0f;

            break;
            
        case scenario_t::UMi:
            // Note 7: fc < 2 GHz, fc = 2 GHz
            if (fc < 2e9) {
                fc = 2e9;
            }
            lgfc = std::log10(fc / 1e9);
            lg1fc = std::log10(1.0f + fc / 1e9);
            // Table 7.5-6 parameters for UMi
            m_cmnLinkParams.mu_lgDS[2] = -6.62f; // O2I
            m_cmnLinkParams.mu_lgDS[1] = -7.14f - 0.24f * lg1fc;  // LOS
            m_cmnLinkParams.mu_lgDS[0] = -6.83f - 0.24f * lg1fc;  // NLOS
            m_cmnLinkParams.sigma_lgDS[2] = 0.32f;
            m_cmnLinkParams.sigma_lgDS[1] = 0.38f;
            m_cmnLinkParams.sigma_lgDS[0] = 0.28f + 0.16f * lg1fc;
            
            m_cmnLinkParams.mu_lgASD[2] = 1.25f;
            m_cmnLinkParams.mu_lgASD[1] = 1.21f - 0.05f * lg1fc;
            m_cmnLinkParams.mu_lgASD[0] = 1.53f - 0.23f * lg1fc;
            m_cmnLinkParams.sigma_lgASD[2] = 0.42f;
            m_cmnLinkParams.sigma_lgASD[1] = 0.41f;
            m_cmnLinkParams.sigma_lgASD[0] = 0.33f + 0.11f * lg1fc;
            
            m_cmnLinkParams.mu_lgASA[2] = 1.76f;
            m_cmnLinkParams.mu_lgASA[1] = 1.73f - 0.08f * lg1fc;
            m_cmnLinkParams.mu_lgASA[0] = 1.81f - 0.08f * lg1fc;
            m_cmnLinkParams.sigma_lgASA[2] = 0.16f;
            m_cmnLinkParams.sigma_lgASA[1] = 0.28f + 0.014f * lg1fc;
            m_cmnLinkParams.sigma_lgASA[0] = 0.3f + 0.05f * lg1fc;
            
            m_cmnLinkParams.mu_lgZSA[2] = 1.01f;
            m_cmnLinkParams.mu_lgZSA[1] = 0.73f - 0.1f * lg1fc;
            m_cmnLinkParams.mu_lgZSA[0] = 0.92f - 0.04f * lg1fc;
            m_cmnLinkParams.sigma_lgZSA[2] = 0.43f;
            m_cmnLinkParams.sigma_lgZSA[1] = 0.34f - 0.04f * lg1fc;
            m_cmnLinkParams.sigma_lgZSA[0] = 0.41f - 0.07f * lg1fc;
            
            m_cmnLinkParams.mu_K[2] = 0.0f;
            m_cmnLinkParams.mu_K[1] = 9.0f;
            m_cmnLinkParams.mu_K[0] = 0.0f;
            m_cmnLinkParams.sigma_K[2] = 0.0f;
            m_cmnLinkParams.sigma_K[1] = 5.0f;
            m_cmnLinkParams.sigma_K[0] = 0.0f;
            
            m_cmnLinkParams.r_tao[2] = 2.2f;
            m_cmnLinkParams.r_tao[1] = 3.0f;
            m_cmnLinkParams.r_tao[0] = 2.1f;
            
            m_cmnLinkParams.mu_XPR[2] = 9.0f;
            m_cmnLinkParams.mu_XPR[1] = 9.0f;
            m_cmnLinkParams.mu_XPR[0] = 8.0f;
            m_cmnLinkParams.sigma_XPR[2] = 5.0f;
            m_cmnLinkParams.sigma_XPR[1] = 3.0f;
            m_cmnLinkParams.sigma_XPR[0] = 3.0f;
            
            m_cmnLinkParams.nCluster[2] = 12;
            m_cmnLinkParams.nCluster[1] = 12;
            m_cmnLinkParams.nCluster[0] = 19;
            m_cmnLinkParams.nRayPerCluster[2] = 20;
            m_cmnLinkParams.nRayPerCluster[1] = 20;
            m_cmnLinkParams.nRayPerCluster[0] = 20;
            
            m_cmnLinkParams.C_DS[2] = 11.0f;
            m_cmnLinkParams.C_DS[1] = 5.0f;
            m_cmnLinkParams.C_DS[0] = 11.0f;
            m_cmnLinkParams.C_ASD[2] = 5.0f;
            m_cmnLinkParams.C_ASD[1] = 3.0f;
            m_cmnLinkParams.C_ASD[0] = 10.0f;
            m_cmnLinkParams.C_ASA[2] = 8.0f;
            m_cmnLinkParams.C_ASA[1] = 17.0f;
            m_cmnLinkParams.C_ASA[0] = 22.0f;
            m_cmnLinkParams.C_ZSA[2] = 3.0f;
            m_cmnLinkParams.C_ZSA[1] = 7.0f;
            m_cmnLinkParams.C_ZSA[0] = 7.0f;
            m_cmnLinkParams.xi[2] = 4.0f;
            m_cmnLinkParams.xi[1] = 3.0f;
            m_cmnLinkParams.xi[0] = 3.0f;

            break;
            
        case scenario_t::RMa:
            // Table 7.5-6 parameters for RMa
            // TO Be updated with O2I parameters
            m_cmnLinkParams.mu_lgDS[2] = -7.47f;
            m_cmnLinkParams.mu_lgDS[1] = -7.49f;
            m_cmnLinkParams.mu_lgDS[0] = -7.43f;
            m_cmnLinkParams.sigma_lgDS[2] = 0.24f;
            m_cmnLinkParams.sigma_lgDS[1] = 0.55f;
            m_cmnLinkParams.sigma_lgDS[0] = 0.48f;
            
            m_cmnLinkParams.mu_lgASD[2] = 0.67f;
            m_cmnLinkParams.mu_lgASD[1] = 0.9f;
            m_cmnLinkParams.mu_lgASD[0] = 0.95f;
            m_cmnLinkParams.sigma_lgASD[2] = 0.18f;
            m_cmnLinkParams.sigma_lgASD[1] = 0.38f;
            m_cmnLinkParams.sigma_lgASD[0] = 0.45f;
            
            m_cmnLinkParams.mu_lgASA[2] = 1.66f;
            m_cmnLinkParams.mu_lgASA[1] = 1.52f;
            m_cmnLinkParams.mu_lgASA[0] = 1.52f;
            m_cmnLinkParams.sigma_lgASA[2] = 0.21f;
            m_cmnLinkParams.sigma_lgASA[1] = 0.24f;
            m_cmnLinkParams.sigma_lgASA[0] = 0.13f;
            
            m_cmnLinkParams.mu_lgZSA[2] = 0.93f;
            m_cmnLinkParams.mu_lgZSA[1] = 0.47f;
            m_cmnLinkParams.mu_lgZSA[0] = 0.58f;
            m_cmnLinkParams.sigma_lgZSA[2] = 0.22f;
            m_cmnLinkParams.sigma_lgZSA[1] = 0.4f;
            m_cmnLinkParams.sigma_lgZSA[0] = 0.37f;
            
            m_cmnLinkParams.mu_K[2] = 0.0f;
            m_cmnLinkParams.mu_K[1] = 7.0f;
            m_cmnLinkParams.mu_K[0] = 0.0f;
            m_cmnLinkParams.sigma_K[2] = 0.0f;
            m_cmnLinkParams.sigma_K[1] = 4.0f;
            m_cmnLinkParams.sigma_K[0] = 0.0f;
            
            m_cmnLinkParams.r_tao[2] = 1.7f;
            m_cmnLinkParams.r_tao[1] = 3.8f;
            m_cmnLinkParams.r_tao[0] = 1.7f;
            
            m_cmnLinkParams.mu_XPR[2] = 7.0f;
            m_cmnLinkParams.mu_XPR[1] = 12.0f;
            m_cmnLinkParams.mu_XPR[0] = 7.0f;
            m_cmnLinkParams.sigma_XPR[2] = 3.0f;
            m_cmnLinkParams.sigma_XPR[1] = 4.0f;
            m_cmnLinkParams.sigma_XPR[0] = 3.0f;
            
            m_cmnLinkParams.nCluster[2] = 10;
            m_cmnLinkParams.nCluster[1] = 11;
            m_cmnLinkParams.nCluster[0] = 10;
            m_cmnLinkParams.nRayPerCluster[2] = 20;
            m_cmnLinkParams.nRayPerCluster[1] = 20;
            m_cmnLinkParams.nRayPerCluster[0] = 20;
            
            m_cmnLinkParams.C_DS[2] = 0.0f;
            m_cmnLinkParams.C_DS[1] = 0.0f;
            m_cmnLinkParams.C_DS[0] = 0.0f;
            m_cmnLinkParams.C_ASD[2] = 2.0f;
            m_cmnLinkParams.C_ASD[1] = 2.0f;
            m_cmnLinkParams.C_ASD[0] = 2.0f;
            m_cmnLinkParams.C_ASA[2] = 3.0f;
            m_cmnLinkParams.C_ASA[1] = 3.0f;
            m_cmnLinkParams.C_ASA[0] = 3.0f;
            m_cmnLinkParams.C_ZSA[2] = 3.0f;
            m_cmnLinkParams.C_ZSA[1] = 3.0f;
            m_cmnLinkParams.C_ZSA[0] = 3.0f;
            m_cmnLinkParams.xi[2] = 3.0f;
            m_cmnLinkParams.xi[1] = 3.0f;
            m_cmnLinkParams.xi[0] = 3.0f;

            break;
            
        default:
            assert(false && "Unknown scenario");
            break;
    }

    
    // get scaling factors for AOA/AOD and ZOA/ZOD
    m_cmnLinkParams.C_phi_NLOS = getScalingFactorAoaAod(m_cmnLinkParams.nCluster[0]);
    m_cmnLinkParams.C_phi_LOS = getScalingFactorAoaAod(m_cmnLinkParams.nCluster[1]);
    m_cmnLinkParams.C_phi_O2I = getScalingFactorAoaAod(m_cmnLinkParams.nCluster[2]);

    m_cmnLinkParams.C_theta_NLOS = getScalingFactorZoaZod(m_cmnLinkParams.nCluster[0]);
    m_cmnLinkParams.C_theta_LOS = getScalingFactorZoaZod(m_cmnLinkParams.nCluster[1]);
    m_cmnLinkParams.C_theta_O2I = getScalingFactorZoaZod(m_cmnLinkParams.nCluster[2]);

    // save fc, lgfc, lg1fc
    m_cmnLinkParams.lgfc = lgfc;  // after fc is updated in the switch statement

    // no need to set ray offset angles, they are constant for all scenarios
    // m_cmnLinkParams.RayOffsetAngles[0] = 0.0447f;
    // m_cmnLinkParams.RayOffsetAngles[1] = -0.0447f;
    // m_cmnLinkParams.RayOffsetAngles[2] = 0.1413f;
    // m_cmnLinkParams.RayOffsetAngles[3] = -0.1413f;
    // m_cmnLinkParams.RayOffsetAngles[4] = 0.2492f;
    // m_cmnLinkParams.RayOffsetAngles[5] = -0.2492f;
    // m_cmnLinkParams.RayOffsetAngles[6] = 0.3715f;
    // m_cmnLinkParams.RayOffsetAngles[7] = -0.3715f;
    // m_cmnLinkParams.RayOffsetAngles[8] = 0.5129f;
    // m_cmnLinkParams.RayOffsetAngles[9] = -0.5129f;
    // m_cmnLinkParams.RayOffsetAngles[10] = 0.6797f;
    // m_cmnLinkParams.RayOffsetAngles[11] = -0.6797f;
    // m_cmnLinkParams.RayOffsetAngles[12] = 0.8844f;
    // m_cmnLinkParams.RayOffsetAngles[13] = -0.8844f;
    // m_cmnLinkParams.RayOffsetAngles[14] = 1.1481f;
    // m_cmnLinkParams.RayOffsetAngles[15] = -1.1481f;
    // m_cmnLinkParams.RayOffsetAngles[16] = 1.5195f;
    // m_cmnLinkParams.RayOffsetAngles[17] = -1.5195f;
    // m_cmnLinkParams.RayOffsetAngles[18] = 2.1551f;
    // m_cmnLinkParams.RayOffsetAngles[19] = -2.1551f;
    
    // Get the appropriate correlation matrix based on scenario and LOS/NLOS
    switch (m_sysConfig->scenario) {
        case scenario_t::UMa:
            for (int i = 0; i < LOS_MATRIX_SIZE; i++) {
                for (int j = 0; j < LOS_MATRIX_SIZE; j++) {
                    m_cmnLinkParams.sqrtCorrMatLos[i * LOS_MATRIX_SIZE + j] = sqrtCorrMatUmaLos[i][j];
                }
            }
            for (int i = 0; i < NLOS_MATRIX_SIZE; i++) {
                for (int j = 0; j < NLOS_MATRIX_SIZE; j++) {
                    m_cmnLinkParams.sqrtCorrMatNlos[i * NLOS_MATRIX_SIZE + j] = sqrtCorrMatUmaNlos[i][j];
                }
            }
            for (int i = 0; i < O2I_MATRIX_SIZE; i++) {
                for (int j = 0; j < O2I_MATRIX_SIZE; j++) {
                    m_cmnLinkParams.sqrtCorrMatO2i[i * O2I_MATRIX_SIZE + j] = sqrtCorrMatUmaO2i[i][j];
                }
            }
            break;
        case scenario_t::UMi:
            for (int i = 0; i < LOS_MATRIX_SIZE; i++) {
                for (int j = 0; j < LOS_MATRIX_SIZE; j++) {
                    m_cmnLinkParams.sqrtCorrMatLos[i * LOS_MATRIX_SIZE + j] = sqrtCorrMatUmiLos[i][j];
                }
            }
            for (int i = 0; i < NLOS_MATRIX_SIZE; i++) {
                for (int j = 0; j < NLOS_MATRIX_SIZE; j++) {
                    m_cmnLinkParams.sqrtCorrMatNlos[i * NLOS_MATRIX_SIZE + j] = sqrtCorrMatUmiNlos[i][j];
                }
            }
            for (int i = 0; i < O2I_MATRIX_SIZE; i++) {
                for (int j = 0; j < O2I_MATRIX_SIZE; j++) {
                    m_cmnLinkParams.sqrtCorrMatO2i[i * O2I_MATRIX_SIZE + j] = sqrtCorrMatUmiO2i[i][j];
                }
            }
            break;
        case scenario_t::RMa:
            for (int i = 0; i < LOS_MATRIX_SIZE; i++) {
                for (int j = 0; j < LOS_MATRIX_SIZE; j++) {
                    m_cmnLinkParams.sqrtCorrMatLos[i * LOS_MATRIX_SIZE + j] = sqrtCorrMatRmaLos[i][j];
                }
            }
            for (int i = 0; i < NLOS_MATRIX_SIZE; i++) {
                for (int j = 0; j < NLOS_MATRIX_SIZE; j++) {
                    m_cmnLinkParams.sqrtCorrMatNlos[i * NLOS_MATRIX_SIZE + j] = sqrtCorrMatRmaNlos[i][j];
                }
            }
            for (int i = 0; i < O2I_MATRIX_SIZE; i++) {
                for (int j = 0; j < O2I_MATRIX_SIZE; j++) {
                    m_cmnLinkParams.sqrtCorrMatO2i[i * O2I_MATRIX_SIZE + j] = sqrtCorrMatRmaO2i[i][j];
                }
            }
            break;
        default:
            throw std::runtime_error("Unknown scenario");
    }

    m_cmnLinkParams.nLink = m_topology.nSite * m_topology.nUT;
    // Get max number of UE antennas
    uint16_t nUeAnt = 0;
    for (uint32_t ueIdx = 0; ueIdx < m_topology.nUT; ueIdx++) {
        nUeAnt = std::max(nUeAnt, m_antPanelConfig->at(m_topology.utParams[ueIdx].antPanelIdx).nAnt);
    }
    m_cmnLinkParams.nUeAnt = nUeAnt;

    // Get max number of BS antennas
    uint16_t nBsAnt = 0;
    for (uint32_t siteIdx = 0; siteIdx < m_topology.nSite; siteIdx++) {
        nBsAnt = std::max(nBsAnt, m_antPanelConfig->at(m_topology.cellParams[siteIdx].antPanelIdx).nAnt);
    }
    m_cmnLinkParams.nBsAnt = nBsAnt;

    // wavelength in meters
    m_cmnLinkParams.lambda_0 = 3.0e8f / m_simConfig->center_freq_hz;
    
    // Initialize subcluster ray arrays (3GPP Table 7.5-5)
    m_cmnLinkParams.raysInSubClusterSizes[0] = 10;
    m_cmnLinkParams.raysInSubClusterSizes[1] = 6;
    m_cmnLinkParams.raysInSubClusterSizes[2] = 4;
    
    // SubCluster 0: {0, 1, 2, 3, 4, 5, 6, 7, 18, 19}
    m_cmnLinkParams.raysInSubCluster0[0] = 0;  m_cmnLinkParams.raysInSubCluster0[1] = 1;
    m_cmnLinkParams.raysInSubCluster0[2] = 2;  m_cmnLinkParams.raysInSubCluster0[3] = 3;
    m_cmnLinkParams.raysInSubCluster0[4] = 4;  m_cmnLinkParams.raysInSubCluster0[5] = 5;
    m_cmnLinkParams.raysInSubCluster0[6] = 6;  m_cmnLinkParams.raysInSubCluster0[7] = 7;
    m_cmnLinkParams.raysInSubCluster0[8] = 18; m_cmnLinkParams.raysInSubCluster0[9] = 19;
    
    // SubCluster 1: {8, 9, 10, 11, 16, 17}  
    m_cmnLinkParams.raysInSubCluster1[0] = 8;  m_cmnLinkParams.raysInSubCluster1[1] = 9;
    m_cmnLinkParams.raysInSubCluster1[2] = 10; m_cmnLinkParams.raysInSubCluster1[3] = 11;
    m_cmnLinkParams.raysInSubCluster1[4] = 16; m_cmnLinkParams.raysInSubCluster1[5] = 17;
    
    // SubCluster 2: {12, 13, 14, 15}
    m_cmnLinkParams.raysInSubCluster2[0] = 12; m_cmnLinkParams.raysInSubCluster2[1] = 13;
    m_cmnLinkParams.raysInSubCluster2[2] = 14; m_cmnLinkParams.raysInSubCluster2[3] = 15;
}

template <typename Tscalar, typename Tcomplex>
void slsChan<Tscalar, Tcomplex>::generateCRN()
{
    // Generate spatially correlated random numbers for LOS, NLOS, and O2I cases
    // Create independent CRN fields for each site to model independent shadow fading between sites
    const uint16_t nSite = m_topology.nSite;
    
    m_crnLos.resize(nSite);
    m_crnNlos.resize(nSite);
    m_crnO2i.resize(nSite);

    // Select appropriate correlation distances based on scenario
    const corrDist_t& corrDistLos = (m_sysConfig->scenario == scenario_t::UMa) ? corrDistUmaLos : 
                                (m_sysConfig->scenario == scenario_t::UMi) ? corrDistUmiLos : corrDistRmaLos;
    const corrDist_t& corrDistNlos = (m_sysConfig->scenario == scenario_t::UMa) ? corrDistUmaNlos : 
                                    (m_sysConfig->scenario == scenario_t::UMi) ? corrDistUmiNlos : corrDistRmaNlos;
    const corrDist_t& corrDistO2i = (m_sysConfig->scenario == scenario_t::UMa) ? corrDistUmaO2i : 
                                    (m_sysConfig->scenario == scenario_t::UMi) ? corrDistUmiO2i : corrDistRmaO2i;

    // Generate independent CRN fields for each site
    for (uint16_t siteIdx = 0; siteIdx < nSite; siteIdx++) {
        m_crnLos[siteIdx].resize(7);
        m_crnNlos[siteIdx].resize(6);
        m_crnO2i[siteIdx].resize(6);
        
        // Generate CRN for LOS case for this site
        genCRN(m_maxX, m_minX, m_maxY, m_minY, corrDistLos.SF, m_gen, m_normalDist, m_crnLos[siteIdx][0]);
        genCRN(m_maxX, m_minX, m_maxY, m_minY, corrDistLos.K, m_gen, m_normalDist, m_crnLos[siteIdx][1]);
        genCRN(m_maxX, m_minX, m_maxY, m_minY, corrDistLos.DS, m_gen, m_normalDist, m_crnLos[siteIdx][2]);
        genCRN(m_maxX, m_minX, m_maxY, m_minY, corrDistLos.ASD, m_gen, m_normalDist, m_crnLos[siteIdx][3]);
        genCRN(m_maxX, m_minX, m_maxY, m_minY, corrDistLos.ASA, m_gen, m_normalDist, m_crnLos[siteIdx][4]);
        genCRN(m_maxX, m_minX, m_maxY, m_minY, corrDistLos.ZSD, m_gen, m_normalDist, m_crnLos[siteIdx][5]);
        genCRN(m_maxX, m_minX, m_maxY, m_minY, corrDistLos.ZSA, m_gen, m_normalDist, m_crnLos[siteIdx][6]);

        // Generate CRN for NLOS case for this site
        genCRN(m_maxX, m_minX, m_maxY, m_minY, corrDistNlos.SF, m_gen, m_normalDist, m_crnNlos[siteIdx][0]);
        genCRN(m_maxX, m_minX, m_maxY, m_minY, corrDistNlos.DS, m_gen, m_normalDist, m_crnNlos[siteIdx][1]);
        genCRN(m_maxX, m_minX, m_maxY, m_minY, corrDistNlos.ASD, m_gen, m_normalDist, m_crnNlos[siteIdx][2]);
        genCRN(m_maxX, m_minX, m_maxY, m_minY, corrDistNlos.ASA, m_gen, m_normalDist, m_crnNlos[siteIdx][3]);
        genCRN(m_maxX, m_minX, m_maxY, m_minY, corrDistNlos.ZSD, m_gen, m_normalDist, m_crnNlos[siteIdx][4]);
        genCRN(m_maxX, m_minX, m_maxY, m_minY, corrDistNlos.ZSA, m_gen, m_normalDist, m_crnNlos[siteIdx][5]);

        // Generate CRN for O2I case for this site
        genCRN(m_maxX, m_minX, m_maxY, m_minY, corrDistO2i.SF, m_gen, m_normalDist, m_crnO2i[siteIdx][0]);
        genCRN(m_maxX, m_minX, m_maxY, m_minY, corrDistO2i.DS, m_gen, m_normalDist, m_crnO2i[siteIdx][1]);
        genCRN(m_maxX, m_minX, m_maxY, m_minY, corrDistO2i.ASD, m_gen, m_normalDist, m_crnO2i[siteIdx][2]);
        genCRN(m_maxX, m_minX, m_maxY, m_minY, corrDistO2i.ASA, m_gen, m_normalDist, m_crnO2i[siteIdx][3]);
        genCRN(m_maxX, m_minX, m_maxY, m_minY, corrDistO2i.ZSD, m_gen, m_normalDist, m_crnO2i[siteIdx][4]);
        genCRN(m_maxX, m_minX, m_maxY, m_minY, corrDistO2i.ZSA, m_gen, m_normalDist, m_crnO2i[siteIdx][5]);
    }
}

template <typename Tscalar, typename Tcomplex>
void slsChan<Tscalar, Tcomplex>::calLinkParam()
{

    // Calculate link parameters for site-UT pairs (co-sited sectors share link parameters)
    uint32_t nSiteUeLink = m_topology.nSite * m_topology.nUT;  // not including sectors
    m_linkParams.resize(nSiteUeLink);

    // Calculate link parameters for each site-UT combination  
    for(uint16_t siteIdx = 0; siteIdx < m_topology.nSite; siteIdx++) {
        for(uint16_t ueIdx = 0; ueIdx < m_topology.nUT; ueIdx++) {
            uint32_t linkIdx = siteIdx * m_topology.nUT + ueIdx;
            
            // Calculate distances using the site's first sector (sector 0) for co-sited calculation
            float d_2d, d_3d, d_2d_in, d_2d_out, d_3d_in, d_3d_out;
            calDist(m_topology.cellParams[siteIdx * m_topology.n_sector_per_site], m_topology.utParams[ueIdx], 
                   d_2d, d_3d, d_2d_in, d_2d_out, d_3d_in, d_3d_out);

            // Store distances in link parameters
            m_linkParams[linkIdx].d2d = d_2d;
            m_linkParams[linkIdx].d2d_in = d_2d_in;
            m_linkParams[linkIdx].d2d_out = d_2d_out;
            m_linkParams[linkIdx].d3d = d_3d;
            m_linkParams[linkIdx].d3d_in = d_3d_in;
            m_linkParams[linkIdx].d3d_out = d_3d_out;

            // Calculate LOS angles
            float phi_los_aod, phi_los_aoa, theta_los_zod, theta_los_zoa;
            calLosAngle(m_topology.cellParams[siteIdx * m_topology.n_sector_per_site], m_topology.utParams[ueIdx], d_3d,
                       phi_los_aod, phi_los_aoa, theta_los_zod, theta_los_zoa);

            // Store LOS angles in link parameters
            m_linkParams[linkIdx].phi_LOS_AOD = phi_los_aod;
            m_linkParams[linkIdx].phi_LOS_AOA = phi_los_aoa;
            m_linkParams[linkIdx].theta_LOS_ZOD = theta_los_zod;
            m_linkParams[linkIdx].theta_LOS_ZOA = theta_los_zoa;

            // Calculate LOS probability and determine LOS/NLOS
            // Only regenerate LOS indicator when m_updateLosState is true (at start or after reset)
            // According to 3GPP TR 38.901, LOS/NLOS state should remain constant during a drop
            if (m_updateLosState) {
                float losProb = calLosProb(m_sysConfig->scenario, d_2d_out, m_topology.utParams[ueIdx].loc.z, m_sysConfig->force_los_prob, m_topology.utParams[ueIdx].outdoor_ind);
                m_linkParams[linkIdx].losInd = (m_uniformDist(m_gen) <= losProb) ? 1 : 0;
            }

            // Calculate path loss (always needed for mode 1 and 2)
            if (m_updatePLAndPenetrationLoss) {
                float pl = calPL(m_sysConfig->scenario, m_linkParams[linkIdx].losInd, d_2d, d_3d, 
                            m_topology.cellParams[siteIdx * m_topology.n_sector_per_site].loc.z, m_topology.utParams[ueIdx].loc.z, 
                            m_simConfig->center_freq_hz / 1e9, m_sysConfig->optional_pl_ind, m_gen, m_uniformDist);

                // Use pre-calculated O2I penetration loss from UE parameters
                // Per 3GPP TR 38.901 Section 7.4.3: O2I is UT-specifically generated, same for ALL BSs
                const float pl_pen = m_topology.utParams[ueIdx].o2i_penetration_loss;
                
                // Add penetration loss to path loss
                pl += pl_pen;
                m_linkParams[linkIdx].pathloss = pl;
            }

            // Generate LSPs (DS, ASD, ASA, SF, K, ZSD, ZSA)            
            // Get spatially correlated random numbers for each LSP
            float & utX = m_topology.utParams[ueIdx].loc.x;
            float & utY = m_topology.utParams[ueIdx].loc.y;
            uint8_t & isLos = m_linkParams[linkIdx].losInd;
            bool isIndoor = (m_topology.utParams[ueIdx].outdoor_ind == 0);
            
            float r_SF, r_K, r_DS, r_ASD, r_ASA, r_ZSD, r_ZSA;
            
            if (isIndoor) {
                // For indoor UEs, always use O2I correlation regardless of LOS/NLOS
                // Use site-specific CRN to get independent shadow fading per site
                r_SF = getLspAtLocation(utX, utY, m_maxX, m_minX, m_maxY, m_minY, m_crnO2i[siteIdx][0]);
                r_K = 0.0f;  // K-factor not applicable for O2I
                r_DS = getLspAtLocation(utX, utY, m_maxX, m_minX, m_maxY, m_minY, m_crnO2i[siteIdx][1]);
                r_ASD = getLspAtLocation(utX, utY, m_maxX, m_minX, m_maxY, m_minY, m_crnO2i[siteIdx][2]);
                r_ASA = getLspAtLocation(utX, utY, m_maxX, m_minX, m_maxY, m_minY, m_crnO2i[siteIdx][3]);
                r_ZSD = getLspAtLocation(utX, utY, m_maxX, m_minX, m_maxY, m_minY, m_crnO2i[siteIdx][4]);
                r_ZSA = getLspAtLocation(utX, utY, m_maxX, m_minX, m_maxY, m_minY, m_crnO2i[siteIdx][5]);
            } else {
                // For outdoor UEs, use LOS/NLOS correlation
                // Use site-specific CRN to get independent shadow fading per site
                r_SF = getLspAtLocation(utX, utY, m_maxX, m_minX, m_maxY, m_minY, isLos ? m_crnLos[siteIdx][0] : m_crnNlos[siteIdx][0]);
                r_K = isLos ? getLspAtLocation(utX, utY, m_maxX, m_minX, m_maxY, m_minY, m_crnLos[siteIdx][1]) : 0.0f;
                r_DS = getLspAtLocation(utX, utY, m_maxX, m_minX, m_maxY, m_minY, isLos ? m_crnLos[siteIdx][2] : m_crnNlos[siteIdx][1]);
                r_ASD = getLspAtLocation(utX, utY, m_maxX, m_minX, m_maxY, m_minY, isLos ? m_crnLos[siteIdx][3] : m_crnNlos[siteIdx][2]);
                r_ASA = getLspAtLocation(utX, utY, m_maxX, m_minX, m_maxY, m_minY, isLos ? m_crnLos[siteIdx][4] : m_crnNlos[siteIdx][3]);
                r_ZSD = getLspAtLocation(utX, utY, m_maxX, m_minX, m_maxY, m_minY, isLos ? m_crnLos[siteIdx][5] : m_crnNlos[siteIdx][4]);
                r_ZSA = getLspAtLocation(utX, utY, m_maxX, m_minX, m_maxY, m_minY, isLos ? m_crnLos[siteIdx][6] : m_crnNlos[siteIdx][5]);
            }
            
            // Create array of uncorrelated variables
            float uncorrVars[LOS_MATRIX_SIZE] = {r_SF, r_K, r_DS, r_ASD, r_ASA, r_ZSD, r_ZSA};
            
            // Perform matrix-vector multiplication to get correlated variables
            float corrVars[LOS_MATRIX_SIZE] = {0};
            if (isIndoor) {
                // For indoor UEs, use O2I correlation matrix (6x6, no K-factor correlation)
                for (int i = 0; i < O2I_MATRIX_SIZE; i++) {
                    for (int j = 0; j <= i; j++) {  // sqrtCorrMatrix is lower triangular matrix
                        // Map indices to skip K-factor
                        const int src_i = (i >= K_IDX) ? i + 1 : i;
                        const int src_j = (j >= K_IDX) ? j + 1 : j;
                        corrVars[src_i] += m_cmnLinkParams.sqrtCorrMatO2i[i * O2I_MATRIX_SIZE + j] * uncorrVars[src_j];
                    }
                }
            } else if (isLos) {
                // For outdoor LOS case, use all 7 variables
                for (int i = 0; i < LOS_MATRIX_SIZE; i++) {
                    for (int j = 0; j <= i; j++) {  // sqrtCorrMatrix is lower triangular matrix
                        corrVars[i] += m_cmnLinkParams.sqrtCorrMatLos[i * LOS_MATRIX_SIZE + j] * uncorrVars[j];
                    }
                }
            } else {
                // For outdoor NLOS case, skip the K-factor (index 1)
                for (int i = 0; i < NLOS_MATRIX_SIZE; i++) {
                    for (int j = 0; j <= i; j++) {  // sqrtCorrMatrix is lower triangular matrix
                        // Map indices to skip K-factor
                        const int src_i = (i >= K_IDX) ? i + 1 : i;
                        const int src_j = (j >= K_IDX) ? j + 1 : j;
                        corrVars[src_i] += m_cmnLinkParams.sqrtCorrMatNlos[i * NLOS_MATRIX_SIZE + j] * uncorrVars[src_j];
                    }
                }
                // Set K-factor to 0 for NLOS
                corrVars[K_IDX] = 0.0f;
            }
            
            // Map to actual LSP values based on scenario and LOS/NLOS
            
            // Set ZSD parameters based on scenario and LOS/NLOS condition
            float & h_ut = m_topology.utParams[ueIdx].loc.z;
            float & h_bs = m_topology.cellParams[siteIdx * m_topology.n_sector_per_site].loc.z;
            uint8_t & losInd = m_linkParams[linkIdx].losInd;
            uint8_t lspIdx = isIndoor ? 2 : losInd;
            float lgfc = m_cmnLinkParams.lgfc;
            float mu, sigma;
            
            if (m_updatePLAndPenetrationLoss) {
                // 1. Shadow Fading (SF) - update for mode 1 and 2
                m_linkParams[linkIdx].SF = corrVars[SF_IDX] * calSfStd(m_sysConfig->scenario, losInd, m_simConfig->center_freq_hz, m_sysConfig->optional_pl_ind, d_2d, h_bs, h_ut);
            }
            
            if (m_updateAllLSPs) {
                // 2. Ricean K-factor (K) - update only for mode 2 (update everything)
                mu = m_cmnLinkParams.mu_K[lspIdx];
                sigma = m_cmnLinkParams.sigma_K[lspIdx];
                m_linkParams[linkIdx].K = lspIdx == 1 ? corrVars[K_IDX] * sigma + mu : 0.0f;
            
                // 3. Delay Spread (DS) - update only for mode 2 (update everything)
                mu = m_cmnLinkParams.mu_lgDS[lspIdx];
                sigma = m_cmnLinkParams.sigma_lgDS[lspIdx];
                m_linkParams[linkIdx].DS = std::pow(10.0f, corrVars[DS_IDX] * sigma + mu + 9.0f);  // add 9.0f to convert from s to ns
            
                // 4. Azimuth Spread of Departure (ASD) - update only for mode 2 (update everything)
                mu = m_cmnLinkParams.mu_lgASD[lspIdx];
                sigma = m_cmnLinkParams.sigma_lgASD[lspIdx];
                float asd_temp = std::pow(10.0f, corrVars[ASD_IDX] * sigma + mu);
                m_linkParams[linkIdx].ASD = std::min(asd_temp, 104.0f);  // Limit to 104 degrees
            
                // 5. Azimuth Spread of Arrival (ASA) - update only for mode 2 (update everything)
                mu = m_cmnLinkParams.mu_lgASA[lspIdx];
                sigma = m_cmnLinkParams.sigma_lgASA[lspIdx];
                float asa_temp = std::pow(10.0f, corrVars[ASA_IDX] * sigma + mu);
                m_linkParams[linkIdx].ASA = std::min(asa_temp, 104.0f);  // Limit to 104 degrees
            
                // 6. Zenith Spread of Departure (ZSD) - update only for mode 2 (update everything)
                switch (m_sysConfig->scenario) {
                    case scenario_t::UMa:
                        if (losInd) {  // LOS
                            m_linkParams[linkIdx].mu_lgZSD = std::max(-0.5f, -2.1f * (d_2d/1000.0f) - 0.01f * (h_ut - 1.5f) + 0.75f);
                            m_linkParams[linkIdx].sigma_lgZSD = 0.4f;
                            m_linkParams[linkIdx].mu_offset_ZOD = 0.0f;
                        } else {  // NLOS
                            m_linkParams[linkIdx].mu_lgZSD = std::max(-0.5f, -2.1f * (d_2d/1000.0f) - 0.01f * (h_ut - 1.5f) + 0.9f);
                            m_linkParams[linkIdx].sigma_lgZSD = 0.49f;
                            m_linkParams[linkIdx].mu_offset_ZOD = 7.66f * lgfc - 5.96f - 
                                std::pow(10.0f, (0.208f * lgfc - 0.782f) * std::log10(std::max(25.0f, d_2d)) + 
                                (2.03f - 0.13f * lgfc) - 0.07f * (h_ut - 1.5f));
                        }
                        break;
                    case scenario_t::UMi:
                        if (losInd) {  // LOS
                            m_linkParams[linkIdx].mu_lgZSD = std::max(-0.21f, -14.8f * (d_2d/1000.0f) - 0.01f * std::abs(h_ut - h_bs) + 0.83f);
                            m_linkParams[linkIdx].sigma_lgZSD = 0.35f;
                            m_linkParams[linkIdx].mu_offset_ZOD = 0.0f;
                        } else {  // NLOS
                            m_linkParams[linkIdx].mu_lgZSD = std::max(-0.5f, -3.1f * (d_2d/1000.0f) + 0.01f * std::max(h_ut - h_bs, 0.0f) + 0.2f);
                            m_linkParams[linkIdx].sigma_lgZSD = 0.35f;
                            m_linkParams[linkIdx].mu_offset_ZOD = -std::pow(10.0f, -1.5f * std::log10(std::max(10.0f, d_2d)) + 3.3f);
                        }
                        break;
                    case scenario_t::RMa:
                        if (losInd) {  // LOS
                            m_linkParams[linkIdx].mu_lgZSD = std::max(-1.0f, -0.17f * (d_2d/1000.0f) - 0.01f * (h_ut - 1.5f) + 0.22f);
                            m_linkParams[linkIdx].sigma_lgZSD = 0.34f;
                            m_linkParams[linkIdx].mu_offset_ZOD = 0.0f;
                        } else {  // NLOS
                            m_linkParams[linkIdx].mu_lgZSD = std::max(-1.0f, -0.19f * (d_2d/1000.0f) - 0.01f * (h_ut - 1.5f) + 0.28f);
                            m_linkParams[linkIdx].sigma_lgZSD = 0.30f;
                            m_linkParams[linkIdx].mu_offset_ZOD = std::atan((35.0f - 3.5f)/d_2d) - std::atan((35.0f - 1.5f)/d_2d);
                        }
                        break;
                    default:
                        assert(false && "Unknown scenario");
                }
                mu = m_linkParams[linkIdx].mu_lgZSD;
                sigma = m_linkParams[linkIdx].sigma_lgZSD;
                float zsd_temp = std::pow(10.0f, corrVars[ZSD_IDX] * sigma + mu);
                m_linkParams[linkIdx].ZSD = std::min(zsd_temp, 52.0f);  // Limit to 52 degrees
            
                // 7. Zenith Spread of Arrival (ZSA) - update only for mode 2 (update everything)
                mu = m_cmnLinkParams.mu_lgZSA[lspIdx];
                sigma = m_cmnLinkParams.sigma_lgZSA[lspIdx];
                float zsa_temp = std::pow(10.0f, corrVars[ZSA_IDX] * sigma + mu);
                m_linkParams[linkIdx].ZSA = std::min(zsa_temp, 52.0f);  // Limit to 52 degrees
            }
        }
    }
    
    // After first call with LOS state generation, set flag to false
    // This ensures LOS state remains constant during the simulation run
    m_updateLosState = false;
}

// Explicit template instantiations
template class slsChan<float, float2>;