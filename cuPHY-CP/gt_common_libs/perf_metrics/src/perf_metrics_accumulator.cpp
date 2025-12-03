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

#include "perf_metrics/perf_metrics_accumulator.hpp"
#include "perf_metrics/perf_metrics_utils.hpp"
#include <cstdio>
#include <cstring>
#include <cstdint>

namespace perf_metrics {

// Constructors

PerfMetricsAccumulator::PerfMetricsAccumulator(const std::initializer_list<const char*>& sectionNames)
{
    if (sectionNames.size() > MAX_SECTIONS) {
        printf("PerfMetricsAccumulator: Error - too many sections (%zu), maximum is %zu\n", 
               sectionNames.size(), MAX_SECTIONS);
        return;
    }
    
    // Pre-register all sections to avoid any allocation during timing
    size_t index = 0;
    for (const char* sectionName : sectionNames) {
        m_sections[index] = {sectionName, {}, true};
        ++index;
    }
    m_sectionCount = index;
}

PerfMetricsAccumulator::PerfMetricsAccumulator() = default;

PerfMetricsAccumulator::~PerfMetricsAccumulator() = default;

// Move constructor and assignment
PerfMetricsAccumulator::PerfMetricsAccumulator(PerfMetricsAccumulator&&) = default;
PerfMetricsAccumulator& PerfMetricsAccumulator::operator=(PerfMetricsAccumulator&&) = default;

// Timing Methods

void PerfMetricsAccumulator::startSection(const char* sectionName)
{
    Section* section = findSection(sectionName);
    
    // Error if section doesn't exist - no dynamic allocation allowed
    if (section == nullptr) {
        printf("PerfMetricsAccumulator: Error - section '%s' not found. Pre-register all sections in constructor.\n", sectionName);
        return;
    }
    
    auto& sectionData = section->data;
    
    // Check if timing is already active
    if (sectionData.isActive) {
        printf("PerfMetricsAccumulator: Error - timing already active for section '%s'\n", sectionName);
        return;
    }
    
    // Capture time after bookkeeping to exclude profiler overhead
    sectionData.startTicks = monotonicNowRaw();
    sectionData.isActive = true;
}

void PerfMetricsAccumulator::stopSection(const char* sectionName)
{
    // Capture time first before bookkeeping to exclude profiler overhead
    const t_raw endTicks = monotonicNowRaw();
    
    Section* section = findSection(sectionName);
    if (section == nullptr) {
        printf("PerfMetricsAccumulator: Error - section '%s' not found\n", sectionName);
        return;
    }
    
    auto& sectionData = section->data;
    if (!sectionData.isActive) {
        printf("PerfMetricsAccumulator: Error - timing not active for section '%s'\n", sectionName);
        return;
    }
    
    // Calculate duration in raw ticks and accumulate
    sectionData.totalRawTicks += (endTicks - sectionData.startTicks);
    sectionData.callCount++;
    sectionData.isActive = false;
}

void PerfMetricsAccumulator::addSectionDuration(const char* sectionName, const std::uint64_t durationNs)
{
    Section* section = findSection(sectionName);
    if (section == nullptr) {
        printf("PerfMetricsAccumulator: Error - section '%s' not found. Pre-register all sections in constructor.\n", sectionName);
        return;
    }
    
    auto& sectionData = section->data;
    // Convert nanoseconds to raw ticks to maintain consistent units
    sectionData.totalRawTicks += nsToRaw(durationNs);
    sectionData.callCount++;
}

void PerfMetricsAccumulator::reset()
{
    for (auto& section : m_sections) {
        if (!section.used) continue;
        
        auto& sectionData = section.data;
        sectionData.totalRawTicks = 0;
        sectionData.callCount = 0;
        sectionData.isActive = false;
        // Note: startTicks doesn't need to be reset since isActive=false
    }
}

PerfMetricsAccumulator::Section* PerfMetricsAccumulator::findSection(const char* name)
{
    // Fast path: pointer comparison for string literals (most common case)
    // Try this first - if it succeeds, we never call strcmp at all
    for (auto& section : m_sections) {
        if (section.used && section.name == name) {
            return &section;
        }
    }
    
    // Slow path: strcmp fallback for dynamically created strings
    // Only reached if pointer comparison failed for all sections
    for (auto& section : m_sections) {
        if (section.used && strcmp(section.name, name) == 0) {
            return &section;
        }
    }
    
    return nullptr;
}

} // namespace perf_metrics