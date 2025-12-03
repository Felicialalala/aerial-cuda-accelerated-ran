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

// This file contains the non-template implementations for the slsChan class
// related to network topology and visualization. Template implementations
// have been moved to sls_chan.cuh.

#include <string>
#include <vector>
#include <random>
#include <cmath>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include "sls_chan.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand.h>
#include <fstream>

// Forward declaration for O2I penetration loss calculation (defined in sls_chan_large_scale.cu)
float calPenetrLos(scenario_t scenario, bool outdoor_ind, float fc, float d_2d_in, 
                   std::uint8_t o2iBuildingPenetrLossInd, std::uint8_t o2iCarPenetrLossInd,
                   std::mt19937& gen, 
                   std::uniform_real_distribution<float>& uniformDist,
                   std::normal_distribution<float>& normalDist);

template <typename Tscalar, typename Tcomplex>
void slsChan<Tscalar, Tcomplex>::bsUeDropping()
{
    // Clear existing topology
    m_topology.cellParams.clear();
    m_topology.utParams.clear();
    m_topology.n_sector_per_site = m_sysConfig->n_sector_per_site;
    m_topology.nSite = m_sysConfig->n_site;
    m_topology.nSector = m_sysConfig->n_site * m_topology.n_sector_per_site;
    m_topology.nUT = m_sysConfig->n_ut;
    
    assert(m_topology.n_sector_per_site == 3 && "Currently only support 3 sectors per site");

    // Set scenario-specific parameters
    switch (m_sysConfig->scenario) {
        case Scenario::UMa:
            m_topology.ISD = 500.0f;
            m_topology.bsHeight = 25.0f;
            m_topology.minBsUeDist2d = 35.0f;
            m_topology.maxBsUeDist2dIndoor = 25.0f;
            m_topology.indoorUtPercent = 0.8f;
            break;
        case Scenario::UMi:
            m_topology.ISD = 200.0f;
            m_topology.bsHeight = 10.0f;
            m_topology.minBsUeDist2d = 10.0f;
            m_topology.maxBsUeDist2dIndoor = 25.0f;
            m_topology.indoorUtPercent = 0.8f;
            break;
        case Scenario::RMa:
            assert(m_sysConfig->isd == 1732.0f || m_sysConfig->isd == 5000.0f);
            m_topology.ISD = m_sysConfig->isd;
            m_topology.bsHeight = 35.0f;
            m_topology.minBsUeDist2d = 35.0f;
            m_topology.maxBsUeDist2dIndoor = 10.0f;
            m_topology.indoorUtPercent = 0.5f;
            break;
        default:
            assert(false && "Unknown scenario");
            break;
    }

    // force indoor percentage
    if (m_sysConfig->force_indoor_ratio >= 0 && m_sysConfig->force_indoor_ratio <= 1) {
        m_topology.indoorUtPercent = m_sysConfig->force_indoor_ratio;
    }

    float cellRadius = m_topology.ISD / std::sqrt(3.0f); // cell radius is ISD/sqrt(3)
    float sectorOrien[3] = {30.0f, 150.0f, 270.0f};  // Sector orientations
    
    // Generate BS positions in hexagonal layout
    for (uint32_t siteIdx = 0; siteIdx < m_sysConfig->n_site; siteIdx++) {
        float coorX = 0.0f;
        float coorY = 0.0f;
        
        // Calculate cell center position
        if (siteIdx < 1) {
            // Center site
            coorX = 0.0f;
            coorY = 0.0f;
        }
        else if (siteIdx < 7) {
            // First ring (6 sites)
            float angle = (siteIdx - 1) * M_PI / 3.0f + M_PI / 6.0f;
            coorX = cos(angle) * m_topology.ISD;
            coorY = sin(angle) * m_topology.ISD;
        }
        else if (siteIdx < 19) {
            float angle = (siteIdx - 7) * M_PI / 6.0f;
            if (siteIdx % 2 == 1) {
                coorX = cos(angle) * 3.0f * cellRadius;
                coorY = sin(angle) * 3.0f * cellRadius;
            }
            else {
                coorX = cos(angle) * 2.0f * m_topology.ISD;
                coorY = sin(angle) * 2.0f * m_topology.ISD;
            }
        }
        else {
            fprintf(stderr, "unsupported number of sites: %d\n", m_topology.nSite);
            exit(-1);
        }
        
        // Create BS parameters
        CellParam cell;
        cell.siteId = siteIdx;
        cell.loc = {coorX, coorY, m_topology.bsHeight};
        cell.antPanelIdx = 0;
        if (m_sysConfig->scenario == Scenario::UMa || m_sysConfig->scenario == Scenario::UMi) {
            cell.antPanelOrientation[0] = 12.0f;  // default value for UMa/UMi
        }
        else {
            cell.antPanelOrientation[0] = 10.0f;  // default value for RMa
        }
        cell.antPanelOrientation[1] = 0.0f;
        cell.antPanelOrientation[2] = 0.0f;
        
        for (int i = 0; i < m_topology.n_sector_per_site; i++) {
            cell.cid = siteIdx * m_topology.n_sector_per_site + i;
            cell.antPanelOrientation[1] = sectorOrien[i];
            m_topology.cellParams.push_back(cell);
        }
    }
    assert(m_topology.cellParams.size() == m_topology.nSector);

    // Generate UEs
    for (uint32_t uIdx = 0; uIdx < m_topology.nUT; uIdx++) {

        // create a new UT parameter
        UtParam ut;

        // Set indoor/outdoor status based on scenario probability
        if (m_sysConfig->force_indoor_ratio >= 0 && m_sysConfig->force_indoor_ratio <= 1) {
            m_topology.indoorUtPercent = m_sysConfig->force_indoor_ratio;
        }
        ut.outdoor_ind = (m_uniformDist(m_gen) > m_topology.indoorUtPercent) ? 1 : 0;
        
        // Randomly select a sector for each UE
        uint32_t secIdx = static_cast<uint32_t>(m_uniformDist(m_gen) * m_topology.nSector);
        
        // Generate UE location
        float randomDistance = 0.0f;
        float randomAngle = 0.0f;
        const int maxIterations = 1000;  // Maximum attempts to find valid position
        int iterationCount = 0;
        bool positionFound = false;
        
        while(iterationCount < maxIterations) {
            // Generate random angle within sector coverage (-60° to +60° from sector orientation) 
            randomAngle = 2.0f * M_PI * m_uniformDist(m_gen) / 3.0f - M_PI/3.0f;
            // Generate random distance between minBsUeDist2d and cell radius
#ifdef CALIBRATION_CFG_
            if (ut.outdoor_ind == 0) {  // no min BS-UT distance for indoor UEs in calibration config
                randomDistance = cellRadius * sqrt(m_uniformDist(m_gen));
            }
            else {  // min BS-UT distance for outdoor UEs in calibration config
                randomDistance = (cellRadius - m_topology.minBsUeDist2d) * sqrt(m_uniformDist(m_gen)) + m_topology.minBsUeDist2d;
            }
#else
            // min BS-UT distance for all UEs in non-calibration config
            randomDistance = (cellRadius - m_topology.minBsUeDist2d) * sqrt(m_uniformDist(m_gen)) + m_topology.minBsUeDist2d;
#endif
            // check if the UE is within the cell
            float tempAngle = std::abs(randomAngle);
            tempAngle = tempAngle > M_PI / 6.0f ? M_PI / 3.0f - tempAngle : tempAngle;  // convert to angle inside a right triangle
            float maxDistanceAngle = m_topology.ISD / 2.0f / cos(tempAngle);
            if (randomDistance <= maxDistanceAngle) {
                randomAngle += (m_topology.cellParams[secIdx].antPanelOrientation[1]) * M_PI / 180.0f;
                positionFound = true;
                break;
            }
            iterationCount++;
        }
        
        // Fallback: assign deterministic position if valid position not found
        if (!positionFound) {
            // Use deterministic fallback position at 2 * minimum distance from sector center
            randomAngle = (m_topology.cellParams[secIdx].antPanelOrientation[1]) * M_PI / 180.0f;
            randomDistance = m_topology.minBsUeDist2d * 1.1f * 2.0f;  // 2 * minimum distance
            printf("Warning: UE %u positioned using fallback after %d iterations\n", uIdx, maxIterations);
        }
        
        // Calculate UE position
        ut.uid = uIdx;
        ut.loc.x = cos(randomAngle) * randomDistance + m_topology.cellParams[secIdx].loc.x;
        ut.loc.y = sin(randomAngle) * randomDistance + m_topology.cellParams[secIdx].loc.y;

        // Calculate floor height based on indoor/outdoor status
        if (m_sysConfig->scenario == Scenario::UMa || m_sysConfig->scenario == Scenario::UMi) {
            // Generate random number of total floors (N_fl) between 4 and 8 (inclusive)
            uint16_t N_fl = 4 + static_cast<uint16_t>(m_uniformDist(m_gen) * 5);  // uniformDist[4,8] - multiply by 5 to include 8
            // Generate random floor number (n_fl) between 1 and N_fl (inclusive)
            uint16_t n_floor = ut.outdoor_ind ? 1 : 1 + static_cast<uint16_t>(m_uniformDist(m_gen) * N_fl);  // uniformDist(1,N_fl) for indoor, 1 for outdoor
            ut.loc.z = 3.0f * (n_floor - 1) + 1.5f; // 3m per floor + 1.5m base height
        }
        else {
            ut.loc.z = 1.5f;
        }
        
        ut.antPanelIdx = 1;
        
        // Generate velocity direction first
        float direction = 2.0f * M_PI * m_uniformDist(m_gen);   
        float speed = 10.0f;
        if (ut.outdoor_ind == 0) {
            if (m_sysConfig->force_ut_speed[0] >= 0) {
                speed = m_sysConfig->force_ut_speed[0];
            }
            else {
                speed = (m_sysConfig->scenario == Scenario::RMa) ? 60.0f : 3.0f;
            }
        }
        if (ut.outdoor_ind == 1) {
            if (m_sysConfig->force_ut_speed[1] >= 0) {
                speed = m_sysConfig->force_ut_speed[1];
            }
            else {
                speed = 3.0f;  
            }
        }
        speed /= 3.6f;  // convert to m/s
        direction = 2.0f * M_PI * m_uniformDist(m_gen); 
        ut.velocity[0] = speed * cos(direction);
        ut.velocity[1] = speed * sin(direction);
        ut.velocity[2] = 0.0f;
        
        // UE antenna orientation per 3GPP TR 38.901:
        // Align azimuth with movement direction (physical assumption: handheld device orientation)
        // [0] = β (zenith/downtilt): fixed at 90° (horizontal)
        // [1] = α (azimuth/bearing): aligned with velocity direction (or random if stationary)
        // [2] = γ (slant): fixed at 0°
        float orientation_deg = (speed > 0.0f) 
            ? std::atan2(ut.velocity[1], ut.velocity[0]) * 180.0f / M_PI
            : 360.0f * m_uniformDist(m_gen);  // Random orientation for stationary UEs
        // Wrap to [0, 360) range
        if (orientation_deg < 0.0f) {
            orientation_deg += 360.0f;
        }
        ut.antPanelOrientation[0] = 90.0f;           // Fixed zenith at 90°
        ut.antPanelOrientation[1] = orientation_deg;  // Azimuth aligned with movement
        ut.antPanelOrientation[2] = 0.0f;            // Fixed slant at 0°
        
        // Generate indoor distance for indoor UEs
        // must be within closest cell distance
        if (ut.outdoor_ind == 1) {
            ut.d_2d_in = 0.0f;
        }
        else {
            const int siteIdx = secIdx / m_topology.n_sector_per_site;
            const float closestCellDistance = sqrt(pow(m_topology.cellParams[siteIdx].loc.x - ut.loc.x, 2) + pow(m_topology.cellParams[siteIdx].loc.y - ut.loc.y, 2));
            
            // Try to generate indoor distance with iteration limit to prevent infinite loop
            int iterationCount = 0;
            while (iterationCount < maxIterations) {
#ifdef CALIBRATION_CFG_
                ut.d_2d_in = m_uniformDist(m_gen) * m_topology.maxBsUeDist2dIndoor;
#else
                ut.d_2d_in = std::min(m_uniformDist(m_gen), m_uniformDist(m_gen)) * m_topology.maxBsUeDist2dIndoor;
#endif
                // check if the indoor distance is within closest cell distance
                if (ut.d_2d_in <= closestCellDistance) {
                    break;
                }
                iterationCount++;
            }
            
            // Fallback: if max iterations reached, use uniform random within closestCellDistance
            if (iterationCount >= maxIterations) {
                ut.d_2d_in = m_uniformDist(m_gen) * closestCellDistance;
                std::fprintf(stderr, "WARNING: Max iterations (%d) reached for indoor distance generation (UE %u, closest cell distance: %.2f m, maxBsUeDist2dIndoor: %.2f m). Using fallback: %.2f m\n", 
                            maxIterations, ut.uid, closestCellDistance, m_topology.maxBsUeDist2dIndoor, ut.d_2d_in);
            }
        }

        // Generate O2I penetration loss ONCE per UE (UT-specifically generated per 3GPP TR 38.901 Section 7.4.3)
        // This value is the SAME for all BSs connecting to this UE (building characteristic, not link characteristic)
        ut.o2i_penetration_loss = calPenetrLos(
            m_sysConfig->scenario,
            ut.outdoor_ind,
            m_simConfig->center_freq_hz,
            ut.d_2d_in,
            m_sysConfig->o2i_building_penetr_loss_ind,
            m_sysConfig->o2i_car_penetr_loss_ind,
            m_gen,
            m_uniformDist,
            m_normalDist
        );

        // Add UE to topology
        m_topology.utParams.push_back(ut);
    }
}

template <typename Tscalar, typename Tcomplex>
void slsChan<Tscalar, Tcomplex>::dumpTopologyToYaml(const std::string& filename) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    // Write topology parameters
    outfile << "topology:\n";
    outfile << "  nSite: " << m_topology.nSite << "\n";
    outfile << "  nSector: " << m_topology.nSector << "\n";
    outfile << "  n_sector_per_site: " << m_topology.n_sector_per_site << "\n";
    outfile << "  nUT: " << m_topology.nUT << "\n";
    outfile << "  ISD: " << m_topology.ISD << "\n";
    outfile << "  bsHeight: " << m_topology.bsHeight << "\n";
    outfile << "  minBsUeDist2d: " << m_topology.minBsUeDist2d << "\n";
    outfile << "  maxBsUeDist2dIndoor: " << m_topology.maxBsUeDist2dIndoor << "\n";
    outfile << "  indoorUtPercent: " << m_topology.indoorUtPercent << "\n\n";

    // Write BS parameters
    outfile << "base_stations:\n";
    for (const auto& bs : m_topology.cellParams) {
        outfile << "  - cid: " << bs.cid << "\n";
        outfile << "    siteId: " << bs.siteId << "\n";
        outfile << "    location:\n";
        outfile << "      x: " << bs.loc.x << "\n";
        outfile << "      y: " << bs.loc.y << "\n";
        outfile << "      z: " << bs.loc.z << "\n";
        outfile << "    antPanelIdx: " << bs.antPanelIdx << "\n";
        outfile << "    antPanelOrientation: [" 
                << bs.antPanelOrientation[0] << ", "
                << bs.antPanelOrientation[1] << ", "
                << bs.antPanelOrientation[2] << "]\n";
    }
    outfile << "\n";

    // Write UE parameters
    outfile << "user_equipment:\n";
    for (const auto& ut : m_topology.utParams) {
        outfile << "  - uid: " << ut.uid << "\n";
        outfile << "    location:\n";
        outfile << "      x: " << ut.loc.x << "\n";
        outfile << "      y: " << ut.loc.y << "\n";
        outfile << "      z: " << ut.loc.z << "\n";
        outfile << "    outdoor_ind: " << static_cast<int>(ut.outdoor_ind) << "\n";
        outfile << "    d_2d_in: " << ut.d_2d_in << "\n";
        outfile << "    antPanelIdx: " << ut.antPanelIdx << "\n";
        outfile << "    antPanelOrientation: ["
                << ut.antPanelOrientation[0] << ", "
                << ut.antPanelOrientation[1] << ", "
                << ut.antPanelOrientation[2] << "]\n";
        outfile << "    velocity: ["
                << ut.velocity[0] << ", "
                << ut.velocity[1] << ", "
                << ut.velocity[2] << "]\n";
    }

    outfile.close();
}

// Explicit template instantiations
template class slsChan<float, float2>;
