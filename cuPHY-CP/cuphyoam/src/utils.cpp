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

#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <cstring>
#include <stdio.h>
#include <unistd.h>
#include <ifaddrs.h>
#include <sys/types.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <linux/ethtool.h>
#include <linux/sockios.h>
#include <errno.h>
#include <net/if_arp.h>
#include <net/if.h>
#include <cstdint>
#include <optional>
#include "utils.hpp"

uint64_t encodePCIeAddr(std::string &pcieAddr)
{
    uint64_t encoded_address = 0;
    uint64_t domain = stoi(pcieAddr.substr(0, 4), 0, 16);
    uint64_t bus = stoi(pcieAddr.substr(5, 2), 0, 16);
    uint8_t device = stoi(pcieAddr.substr(8, 2), 0, 16);
    uint8_t function = stoi(pcieAddr.substr(11, 1), 0, 16);

    encoded_address |= domain << 20;
    encoded_address |= bus << 12;
    encoded_address |= device << 4;
    encoded_address |= function;

    return encoded_address;
}

uint64_t encodeMACAddr(std::string &macAddr)
{
    uint64_t mac = 0;
    mac = (mac << 8) | stoi(macAddr.substr(0, 2), 0, 16);
    mac = (mac << 8) | stoi(macAddr.substr(3, 2), 0, 16);
    mac = (mac << 8) | stoi(macAddr.substr(6, 2), 0, 16);
    mac = (mac << 8) | stoi(macAddr.substr(9, 2), 0, 16);
    mac = (mac << 8) | stoi(macAddr.substr(12, 2), 0, 16);
    mac = (mac << 8) | stoi(macAddr.substr(15, 2), 0, 16);
    return mac;
}

std::optional<std::string> convertMacAddrToPCIeAddr(std::string &macAddr)
{
    std::optional<std::string> pcieAddr = std::nullopt;
    struct ifaddrs *ifaddr, *ifa;
    struct ifreq sif;
    struct ethtool_drvinfo d;
    int ret;
    int sd;

    if (getifaddrs(&ifaddr) == -1)
    {
        perror("getifaddrs");
        return std::nullopt;
    }

    sd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sd < 0)
    {
        // printf("Error socket\n");
        return std::nullopt;
    }

    /* Walk through linked list, maintaining head pointer so we
       can free list later */
    for (ifa = ifaddr; ifa != NULL; ifa = ifa->ifa_next)
    {
        if (ifa->ifa_addr == NULL || ifa->ifa_addr->sa_family != AF_PACKET)
            continue;

        const char *devname = ifa->ifa_name;
        memset(&sif, 0, sizeof(struct ifreq));
        strncpy(sif.ifr_name, devname, strlen(devname));

        ret = ioctl(sd, SIOCGIFHWADDR, &sif);
        if (ret == -1)
        {
            perror("ioctl");
            goto end;
        }
        if (sif.ifr_hwaddr.sa_family != ARPHRD_ETHER)
        {
            // printf("not an Ethernet interface");
            continue;
        }

        const unsigned char *mac = (unsigned char *)sif.ifr_hwaddr.sa_data;
        char buffer[20];
        sprintf(buffer, "%02X:%02X:%02X:%02X:%02X:%02X\n", mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
        if (strncasecmp(macAddr.c_str(), buffer, 17))
        {
            continue;
        }
        // printf("%02X:%02X:%02X:%02X:%02X:%02X\n", mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);

        d.cmd = ETHTOOL_GDRVINFO;
        sif.ifr_data = (caddr_t)&d;
        if (ioctl(sd, SIOCETHTOOL, &sif) == -1)
        {
            perror("ioctl");
            goto end;
        }
        // printf("bus-info: %s\n", d.bus_info);
        pcieAddr = d.bus_info;
        break;
    }

end:
    close(sd);
    freeifaddrs(ifaddr);
    return pcieAddr;
}

#endif // UTILS_H
