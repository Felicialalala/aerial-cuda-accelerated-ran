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

#ifndef PERF_METRICS_ACCUMULATOR_HPP
#define PERF_METRICS_ACCUMULATOR_HPP

#include <cstdint>
#include <initializer_list>
#include <array>
#include <chrono>
#include <cstring>
#include <cstdio>

#include "perf_metrics_logging.hpp"
#include "perf_metrics_utils.hpp"

namespace perf_metrics {

/**
 * Performance metrics accumulator for measuring and aggregating execution times
 * 
 * Simple interface with only 4 public methods:
 * - startSection(name) / stopSection(name): Time measurement
 * - logDurations<TAG>(): Log accumulated results  
 * - reset(): Clear all accumulated data
 * 
 * IMPORTANT: All sections must be pre-registered in constructor to avoid 
 * dynamic allocation during timing operations.
 * 
 * PERFORMANCE NOTE: For best performance, use string literals for section names.
 * The implementation uses pointer comparison optimization for string literals,
 * falling back to strcmp for dynamic strings. String literals are ~3-4x faster.
 * 
 * Usage:
 * @code
 * #define TAG_MY_PERF (NVLOG_TAG_BASE_RU_EMULATOR + 10)
 * 
 * perf_metrics::PerfMetricsAccumulator pma{"Section 1", "Section 2"};
 * 
 * for(int ii = 0; ii < 100; ii++) {
 *     pma.startSection("Section 1");  // String literal - fast!
 *     // ... code for Section 1
 *     pma.stopSection("Section 1");
 *     
 *     pma.startSection("Section 2");
 *     // ... code for Section 2
 *     pma.stopSection("Section 2");
 * }
 * 
 * pma.logDurations<TAG_MY_PERF>();                    // Log with INFO level
 * pma.logDurations<TAG_MY_PERF, LogLevel::DEBUG>();   // Log with DEBUG level
 * pma.reset();                                        // Reset all accumulated results to 0
 * @endcode
 */
class PerfMetricsAccumulator {
public:
    /**
     * Construct with predefined section names (C-string literals only)
     * 
     * @param[in] sectionNames Initializer list of C-string literals to pre-register
     */
    explicit PerfMetricsAccumulator(const std::initializer_list<const char*>& sectionNames);
    
    /**
     * Default constructor - creates empty accumulator (no sections pre-registered)
     */
    PerfMetricsAccumulator();
    
    /**
     * Destructor
     */
    ~PerfMetricsAccumulator();
    
    // Non-copyable but movable
    PerfMetricsAccumulator(const PerfMetricsAccumulator&) = delete;
    PerfMetricsAccumulator& operator=(const PerfMetricsAccumulator&) = delete;
    PerfMetricsAccumulator(PerfMetricsAccumulator&&);
    PerfMetricsAccumulator& operator=(PerfMetricsAccumulator&&);
    
    /**
     * Start timing measurement for a section
     * 
     * @param[in] sectionName Name of the section to start timing (C-string literal)
     */
    void startSection(const char* sectionName);
    
    /**
     * Stop timing measurement for a section and accumulate the duration
     * 
     * @param[in] sectionName Name of the section to stop timing (C-string literal)
     */
    void stopSection(const char* sectionName);
    
    /**
     * Add a pre-calculated duration to a section's accumulated metrics
     * 
     * Useful when timing is performed externally (e.g., by a called function)
     * and you want to include those metrics in the accumulator.
     * 
     * @param[in] sectionName Name of the section (C-string literal, must be pre-registered)
     * @param[in] durationNs Duration in nanoseconds to add to the section
     */
    void addSectionDuration(const char* sectionName, std::uint64_t durationNs);
    
    /**
     * Reset all accumulated metrics to zero
     */
    void reset();
    
    /**
     * Log accumulated durations using NVLOG
     * 
     * Outputs total duration in microseconds for each section.
     * Template parameters allow customization of NVLOG tag and log level.
     * 
     * @tparam NvlogTag Compile-time NVLOG tag (e.g., TAG_MY_PERF)
     * @tparam Level Log level to use (defaults to INFO)
     */
    template<int NvlogTag, LogLevel Level = LogLevel::INFO>
    void logDurations() const;

    /**
     * Log accumulated durations for all sections with a custom prefix
     * 
     * Outputs total duration in microseconds for each section with a custom prefix.
     * Template parameters allow customization of NVLOG tag and log level.
     * 
     * @tparam NvlogTag Compile-time NVLOG tag (e.g., TAG_MY_PERF)
     * @tparam Level Log level to use (defaults to INFO)
     * @param prefix Custom prefix string to prepend to the log message
     */
    template<int NvlogTag, LogLevel Level = LogLevel::INFO>
    void logDurations(const char* prefix) const;

private:
    static constexpr size_t MAX_SECTIONS = 32; //!< Maximum number of sections supported
    
    /**
     * Performance data for a single section
     */
    struct SectionData {
        std::uint64_t totalRawTicks{};  //!< Total accumulated time in raw counter ticks
        std::uint64_t callCount{};      //!< Number of times this section was measured
        t_raw startTicks{};             //!< Current measurement start time (raw ticks)
        bool isActive{};                //!< Whether timing is currently active for this section
    };
    
    /**
     * Section entry with name and data
     */
    struct Section {
        const char* name{};            //!< Section name (points to string literal)
        SectionData data{};            //!< Performance data for this section
        bool used{};                   //!< Whether this slot is in use
    };
    
    std::array<Section, MAX_SECTIONS> m_sections{}; //!< Fixed array of sections
    size_t m_sectionCount{};                        //!< Number of sections currently used
    
    /**
     * Find section by name
     * 
     * @param[in] name Section name to find
     * @return Pointer to section if found, nullptr otherwise
     */
    Section* findSection(const char* name);
};

// Template method implementations (inline for interface-only library)

/**
 * Log accumulated durations in compact format
 * 
 * Calls the overloaded version with nullptr prefix for default behavior.
 * 
 * @tparam NvlogTag Compile-time NVLOG tag
 * @tparam Level Log level to use
 */
template<int NvlogTag, LogLevel Level>
void PerfMetricsAccumulator::logDurations() const {
    logDurations<NvlogTag, Level>(nullptr);
}

/**
 * Log accumulated durations in compact format with custom prefix
 * 
 * Uses fixed-size buffer to avoid dynamic allocation.
 * Format: "[prefix] - Section1:count:duration_us,Section2:count:duration_us"
 * 
 * @tparam NvlogTag Compile-time NVLOG tag
 * @tparam Level Log level to use
 * @param prefix Custom prefix string to prepend to the log message
 */
template<int NvlogTag, LogLevel Level>
void PerfMetricsAccumulator::logDurations(const char* prefix) const {
    if (m_sectionCount == 0) {
        PERF_METRICS_LOG(NvlogTag, Level, "{} - No sections", prefix ? prefix : "PerfMetricsAccumulator");
        return;
    }
    
    // Use fixed-size buffer to avoid dynamic allocation
    constexpr size_t BUFFER_SIZE = 1024;
    char buffer[BUFFER_SIZE];
    size_t pos = 0;
    
    // Start with custom prefix or default
    const char* actualPrefix = prefix ? prefix : "PerfMetricsAccumulator";
    const size_t prefixLen = strlen(actualPrefix);
    if (pos + prefixLen + 3 < BUFFER_SIZE - 1) { // +3 for " - ", -1 for null terminator
        memcpy(buffer + pos, actualPrefix, prefixLen);
        pos += prefixLen;
        memcpy(buffer + pos, " - ", 3);
        pos += 3;
    }
    
    bool first = true;
    for (const auto& section : m_sections) {
        if (!section.used) continue;
        
        const auto& sectionData = section.data;
        const char* sectionName = section.name;
        
        // Convert raw ticks to nanoseconds, then to microseconds
        const std::uint64_t durationNs = rawToNs(sectionData.totalRawTicks);
        const std::uint64_t durationUs = durationNs / 1000;
        
        // Add comma separator if not first
        if (!first && pos + 1 < BUFFER_SIZE - 1) { // -1 for null terminator
            buffer[pos++] = ',';
        }
        first = false;
        
        // Add "SectionName:count:duration"
        const size_t remaining = BUFFER_SIZE - pos - 1; // -1 for null terminator
        const int written = snprintf(buffer + pos, remaining, "%s:%llu:%llu", 
                                     sectionName, 
                                     static_cast<unsigned long long>(sectionData.callCount),
                                     static_cast<unsigned long long>(durationUs));
        
        if (written > 0 && static_cast<size_t>(written) < remaining) {
            pos += written;
        } else {
            // Buffer full, truncate gracefully
            break;
        }
    }
    
    // Null terminate
    buffer[pos] = '\0';
    
    PERF_METRICS_LOG(NvlogTag, Level, "{}", buffer);
}

} // namespace perf_metrics

#endif // PERF_METRICS_ACCUMULATOR_HPP
