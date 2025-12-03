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

#include <iostream>
#include <fstream>
#include <string>
#include <stdexcept>
#include <cstdlib>
#include <iomanip>

#include "nv_utils.h"
#include "nvlog_fmt.hpp"

#include "ptp_service_status_checking.hpp"
#include "nvlog.hpp"

using namespace AppUtils;

#define TAG (NVLOG_TAG_BASE_APP_CFG_UTILS + 2) // "APP.UTILS"


ServiceStatus AppUtils::checkPtpServiceStatus(const std::string& syslogPath, double rmsThreshold, const std::string& serviceName) {
    try {
        // Validate service name
        if (serviceName != "ptp4l" && serviceName != "phc2sys") {
            throw std::runtime_error("Invalid service name: " + serviceName);
        }

        // Step 1: Check service status
        std::string cmd = "systemctl is-active " + serviceName + ".service > /dev/null 2>&1";
        int systemctlResult = std::system(cmd.c_str());
        bool isRunning = (systemctlResult == 0);

        // if (!isRunning) {
        //     cmd = "busctl get-property org.freedesktop.systemd1 /org/freedesktop/systemd1/unit/" + 
        //           serviceName + "_2eservice org.freedesktop.systemd1.Unit ActiveState 2>/dev/null | grep -q 'active'";
        //     systemctlResult = std::system(cmd.c_str());
        //     isRunning = (systemctlResult == 0);
        //     if (!isRunning) {
        //         NVLOGW_FMT(TAG, "Warning: systemctl and busctl failed for {} , assuming stopped", serviceName);
        //     }
        // }

        if (!isRunning) {
            return ServiceStatus::STOPPED;
        }

        // Step 2: Open syslog file
        std::ifstream file(syslogPath, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Unable to open syslog file: " + syslogPath);
        }

        // Check file size
        file.seekg(0, std::ios::end);
        long long fileSize = file.tellg();
        if (fileSize == 0) {
            file.close();
            throw std::runtime_error("Syslog file is empty: " + syslogPath);
        }

        // Step 3: Parse syslog for the latest service RMS using reverse reading
        std::string lastLine;
        long long pos = fileSize - 1;
        std::string line;
        bool found = false;
        std::string servicePrefix = serviceName + ":";

        while (pos >= 0 && !found) {
            file.seekg(pos, std::ios::beg);
            char c;
            if (!file.get(c)) {
                file.clear();
                --pos;
                continue;
            }

            if (c == '\n' || pos == 0) {
                long long lineStart = (pos == 0 ? 0 : pos + 1);
                file.seekg(lineStart, std::ios::beg);
                file.clear();
                if (std::getline(file, line)) {
                    if (line.find(servicePrefix) != std::string::npos && line.find("rms") != std::string::npos) {
                        lastLine = line;
                        found = true;
                    }
                } else {
                    file.clear();
                }
                pos = (pos == 0 ? -1 : pos - 1);
            } else {
                --pos;
            }
        }

        file.close();

        // Step 4: Extract RMS from the last service line
        double rmsValue = -1.0;
        if (!lastLine.empty()) {
            // Parse timestamp (format: "May 14 10:15:47")
            std::string timestampStr;
            try {
                size_t pos = 0;
                // Extract first three fields (month, day, time)
                for (int i = 0; i < 3 && pos < lastLine.length(); ++i) {
                    while (pos < lastLine.length() && std::isspace(lastLine[pos])) ++pos;
                    while (pos < lastLine.length() && !std::isspace(lastLine[pos])) {
                        timestampStr += lastLine[pos++];
                    }
                    if (i < 2) timestampStr += " ";
                }

                std::tm tm = {};
                std::istringstream ss(timestampStr);
		//std::cout << "log timestamp===>" << timestampStr << std::endl;
                ss >> std::get_time(&tm, "%b %d %H:%M:%S");
                if (ss.fail()) {
                    throw std::runtime_error("Failed to parse timestamp: " + timestampStr);
                }

                time_t now = time(nullptr);
                std::tm* local = localtime(&now);
                tm.tm_year = local->tm_year;
                tm.tm_isdst = local->tm_isdst;

                time_t logTime = mktime(&tm);
                time_t currentTime = time(nullptr);

                double timeDiff = std::difftime(currentTime, logTime);
                if (timeDiff > 1.0) {
                    throw std::runtime_error("Log timestamp (" + timestampStr + ") is more than 1 second old: " + std::to_string(timeDiff) + " seconds");
                }
            } catch (const std::exception& e) {
                throw std::runtime_error("Error parsing timestamp in " + serviceName + " line: (" + lastLine + ") : " + e.what());
            }


            size_t rmsPos = lastLine.find("rms");
            try {
                size_t pos = rmsPos + 3;
                while (pos < lastLine.length() && std::isspace(lastLine[pos])) {
                    ++pos;
                }
                std::string numStr;
                while (pos < lastLine.length() && (std::isdigit(lastLine[pos]) || lastLine[pos] == '.' || lastLine[pos] == '-')) {
                    numStr += lastLine[pos++];
                }
                if (numStr.empty()) {
                    throw std::runtime_error("No number found after 'rms' in " + serviceName + " line: " + lastLine);
                }
                rmsValue = std::stod(numStr);
            } catch (const std::exception& e) {
                throw std::runtime_error("Error parsing " + serviceName + " RMS in line: " + lastLine + ": " + e.what());
            }
        } else {
            NVLOGW_FMT(TAG, "Warning: No {} lines with 'rms' found in {}", serviceName, syslogPath);
            throw std::runtime_error("No " + serviceName + " synchronization data available");
        }

        // Step 5: Determine status
        if (rmsValue < rmsThreshold) {
            return ServiceStatus::RUNNING_SYNCED;
        }
        NVLOGI_FMT(TAG, "{}.service: current rms: {} ns, rmsThreshold {} ns)", serviceName, rmsValue, rmsThreshold);
        return ServiceStatus::RUNNING_UNSYNCED;

    } catch (const std::exception& e) {
        NVLOGW_FMT(TAG, "Error: {} ", e.what());
        return ServiceStatus::ERROR;
    }
}

int AppUtils::checkPtpServiceStatus(const std::string& syslogPath, double ptp4lRmsThreshold, double phc2sysRmsThreshold) {
    int ret1 = -1;
    // Check ptp4l.service
    //ServiceStatus ptp4lStatus = checkPtp4lStatus(syslogPath, ptp4lRmsThreshold);
    ServiceStatus ptp4lStatus = checkPtpServiceStatus(syslogPath, ptp4lRmsThreshold, "ptp4l");
    switch (ptp4lStatus)
    {
    case ServiceStatus::RUNNING_SYNCED:
        NVLOGI_FMT(TAG, "ptp4l.service: running and synchronized (RMS < {} ns)", ptp4lRmsThreshold);
        ret1 = 0;
        break;
    case ServiceStatus::RUNNING_UNSYNCED:
        NVLOGW_FMT(TAG, "ptp4l.service: running but rms larger than threshold (RMS >= {} ns)", ptp4lRmsThreshold);
        break;
    case ServiceStatus::STOPPED:
        NVLOGW_FMT(TAG, "ptp4l.service: stopped");
        break;
    case ServiceStatus::ERROR:
        NVLOGW_FMT(TAG, "ptp4l.service: error checking status");
        break;
    }

    int ret2 = -1;
    // Check phc2sys.service
    // ServiceStatus phc2sysStatus = checkPhc2sysStatus(syslogPath, phc2sysRmsThreshold);
    ServiceStatus phc2sysStatus = checkPtpServiceStatus(syslogPath, phc2sysRmsThreshold, "phc2sys");
    switch (phc2sysStatus)
    {
    case ServiceStatus::RUNNING_SYNCED:
        NVLOGI_FMT(TAG, "phc2sys.service: running and synchronized (RMS < {} ns)", phc2sysRmsThreshold);
        ret2 = 0;
        break;
    case ServiceStatus::RUNNING_UNSYNCED:
        NVLOGW_FMT(TAG, "phc2sys.service: running but rms larger than threshold (RMS >= {} ns)", phc2sysRmsThreshold);
        break;
    case ServiceStatus::STOPPED:
        NVLOGW_FMT(TAG, "phc2sys.service: stopped");
        break;
    case ServiceStatus::ERROR:
        NVLOGW_FMT(TAG, "phc2sys.service: error checking status");
        break;
    }

    return (ret1 || ret2) ? -1 : 0;
}
