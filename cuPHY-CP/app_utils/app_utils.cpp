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

#include "app_utils.hpp"
#include "nvlog.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include <stdexcept>
#include <time.h>
#include <sys/ioctl.h>
#include <linux/ethtool.h>
#include <linux/sockios.h>
#include <linux/if.h>
#include <sys/timex.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

using namespace AppUtils;

#define TAG (NVLOG_TAG_BASE_APP_CFG_UTILS + 2) // "APP.UTILS"
#define FD_TO_CLOCKID(fd) ((~(clockid_t)(fd) << 3) | 3)

void AppUtils::set_tai_offset(int tai_offset)
{
    struct timex tmx;
    memset(&tmx, 0, sizeof(tmx));

    // int tai_offset_value = 37; // Example offset value
    int tai_offset_value = tai_offset;

    // Set the tai field in the timex structure
    tmx.modes = ADJ_TAI;
    // tmx.tai = tai_offset_value;
    tmx.constant = tai_offset_value;

    // Call adjtimex to set the TAI offset
    if (adjtimex(&tmx) == -1)
    {
        NVLOGE_FMT(TAG, AERIAL_CLOCK_API_EVENT, "adjtimex failed: {}", strerror(errno));
        return;
    }
}

uint64_t AppUtils::get_ns_clock_tai(void)
{
    struct timespec t;
    int ret;
    ret = clock_gettime(CLOCK_TAI, &t);
    if (ret != 0)
    {
        NVLOGE_FMT(TAG, AERIAL_CLOCK_API_EVENT, "clock_gettime CLOCK_TAI failed");
    }
    return (uint64_t)t.tv_nsec + (uint64_t)t.tv_sec * 1000 * 1000 * 1000;
}

namespace fs = std::filesystem;

std::string AppUtils::get_nic_name(const std::string &pcie_address)
{
    std::string base_path = "/sys/bus/pci/devices/";
    std::string full_path = base_path + pcie_address + "/net/";

    try
    {
        for (const auto &entry : fs::directory_iterator(full_path))
        {
            if (fs::is_directory(entry))
            {
                return entry.path().filename().string();
            }
        }
    }
    catch (const fs::filesystem_error &e)
    {
        throw std::runtime_error("Error accessing directory: " + std::string(e.what()));
    }

    throw std::runtime_error("No network interface found for PCIe address: " + pcie_address);
}

int get_phc_index(const char *ifname)
{
    struct ifreq ifr;
    struct ethtool_ts_info info;
    int sockfd;

    memset(&ifr, 0, sizeof(ifr));
    memset(&info, 0, sizeof(info));
    info.cmd = ETHTOOL_GET_TS_INFO;
    strncpy(ifr.ifr_name, ifname, IFNAMSIZ - 1);
    ifr.ifr_data = (caddr_t)&info;

    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0)
    {
        NVLOGE_FMT(TAG, AERIAL_CLOCK_API_EVENT, "socket");
        return -1;
    }

    if (ioctl(sockfd, SIOCETHTOOL, &ifr) < 0)
    {
        NVLOGE_FMT(TAG, AERIAL_CLOCK_API_EVENT, "ioctl SIOCETHTOOL");
        close(sockfd);
        return -1;
    }

    close(sockfd);

    if (info.phc_index < 0)
    {
        NVLOGE_FMT(TAG, AERIAL_CLOCK_API_EVENT, "No PHC device associated with the network interface");
        return -1;
    }

    return info.phc_index;
}

int open_phc_clock(int phc_index)
{
    char device[32];
    snprintf(device, sizeof(device), "/dev/ptp%d", phc_index);
    return open(device, O_RDONLY);
}

void AppUtils::get_phc_clock(const char *ifname)
{
    struct timespec phc_time;
    int phc_index, phc_fd;
    clockid_t phc_clockid;

    phc_index = get_phc_index(ifname);
    if (phc_index < 0)
    {
        NVLOGE_FMT(TAG, AERIAL_CLOCK_API_EVENT, "get_phc_index {}", phc_index);
        return;
    }

    phc_fd = open_phc_clock(phc_index);
    if (phc_fd < 0)
    {
        NVLOGE_FMT(TAG, AERIAL_CLOCK_API_EVENT, "open_phc_clock");
        return;
    }

    phc_clockid = FD_TO_CLOCKID(phc_fd);

    if (clock_gettime(phc_clockid, &phc_time) == -1)
    {
        NVLOGE_FMT(TAG, AERIAL_CLOCK_API_EVENT, "clock_gettime PHC");
        close(phc_fd);
        return;
    }

    NVLOGC_FMT(TAG, "          PHC clock: {}.{}", phc_time.tv_sec, phc_time.tv_nsec);
}

void AppUtils::clock_sanity_check()
{
    struct timespec realtime, tai;
    struct timex tmx;

    if (clock_gettime(CLOCK_REALTIME, &realtime) == -1)
    {
        NVLOGE_FMT(TAG, AERIAL_CLOCK_API_EVENT, "clock_gettime CLOCK_REALTIME failed");
        return;
    }

    if (clock_gettime(CLOCK_TAI, &tai) == -1)
    {
        NVLOGE_FMT(TAG, AERIAL_CLOCK_API_EVENT, "clock_gettime CLOCK_TAI failed");
        return;
    }

    memset(&tmx, 0, sizeof(tmx));
    if (adjtimex(&tmx) == -1)
    {
        NVLOGE_FMT(TAG, AERIAL_CLOCK_API_EVENT, "adjtimex read tai offset failed");
        return;
    }

    NVLOGC_FMT(TAG, "          CLOCK_TAI: {}.{}", tai.tv_sec, tai.tv_nsec);
    NVLOGC_FMT(TAG, "     CLOCK_REALTIME: {}.{}", realtime.tv_sec, realtime.tv_nsec);
    NVLOGC_FMT(TAG, "TAI/REALTIME offset: {} seconds", tmx.tai);
}
