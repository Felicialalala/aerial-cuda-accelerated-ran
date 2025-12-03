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

#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <stdio.h>
#include <ifaddrs.h>
#include <sys/types.h>
#include <sys/ioctl.h>
#include <linux/ethtool.h>
#include <linux/sockios.h>
#include <errno.h>
#include <net/if_arp.h>
#include <net/if.h>
#include <linux/if.h>
#include <linux/netlink.h>
#include <linux/rtnetlink.h>
#include <sys/socket.h>

#include "nvlog.hpp"
#include "app_config.hpp"
#include "cuphyoam.hpp"
#include "utils.hpp"

#define TAG (NVLOG_TAG_BASE_CUPHY_OAM + 3) // "OAM.CUSConnMgr"

#define MSG_BUF_SIZE 8192

// std::string ptpCfgFilePath = "/etc/ptp.conf";
std::string ptpCfgFilePath = "/host/etc/ptp.conf"; // mapped into container from host
std::string if_names[] = {"aerial00", "aerial01"};

std::unordered_map<int, std::string> ifIdx2Name;
std::unordered_map<int, bool> ifIdx2Status;
std::unordered_map<int, std::string> ifIdx2PCIeAddr;

static int cur_if_idx = -1;

void getPCIeAddr()
{
    struct ifaddrs *ifaddr, *ifa;
    struct ifreq sif;
    struct ethtool_drvinfo d;
    int ret;
    int sd;

    if (getifaddrs(&ifaddr) == -1)
    {
        perror("getifaddrs");
        return;
    }

    sd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sd < 0)
    {
        // printf("Error socket\n");
        return;
    }

    /* Walk through linked list, maintaining head pointer so we
       can free list later */
    for (ifa = ifaddr; ifa != NULL; ifa = ifa->ifa_next)
    {
        if (ifa->ifa_addr == NULL || ifa->ifa_addr->sa_family != AF_PACKET)
            continue;

        int ifIdx = -1;
        for (auto &[idx, if_name] : ifIdx2Name)
        {
            if (strncmp(if_name.c_str(), ifa->ifa_name, strlen(ifa->ifa_name)) == 0)
            {
                ifIdx = idx;
            }
        }

        if (ifIdx == -1)
        {
            continue;
        }

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

        d.cmd = ETHTOOL_GDRVINFO;
        sif.ifr_data = (caddr_t)&d;
        if (ioctl(sd, SIOCETHTOOL, &sif) == -1)
        {
            perror("ioctl");
            goto end;
        }
        // printf("bus-info: %s\n", d.bus_info);
        ifIdx2PCIeAddr[ifIdx] = d.bus_info;
        NVLOGC_FMT(TAG, "Interface idx {} PCIe address is: {} ", ifIdx, ifIdx2PCIeAddr[ifIdx]);
    }

end:
    close(sd);
    freeifaddrs(ifaddr);
}

int switchCUPlanePort(std::string &pcie_addr)
{
    CuphyOAM *oam = CuphyOAM::getInstance();
    auto new_cell_config = new CuphyOAMCellConfig();
    new_cell_config->multi_attrs_cfg = true;
    new_cell_config->cell_id = 0xFF;
    new_cell_config->attrs["nic"] = encodePCIeAddr(pcie_addr);
    oam->cell_multi_attri_update_callback(new_cell_config->cell_id, new_cell_config->attrs, new_cell_config->res);

    auto &res = new_cell_config->res;
    // Set the reply message
    if (res.size() == 0)
    {
        return 0;
    }
    else
    {
        return -1;
    }
    delete new_cell_config;
    return 0;
}

void fetchIfIdx()
{
    unsigned int port_index;
    for (auto &if_name : if_names)
    {
        if ((port_index = if_nametoindex(if_name.c_str())) == 0)
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYOAM_EVENT, "No index found for interface: '{}'", if_name);
            continue;
        }
        ifIdx2Name[port_index] = if_name;
        if (if_name == if_names[0])
        {
            cur_if_idx = port_index;
        }
        NVLOGC_FMT(TAG, "Interface '{}' index is: {}", if_name, port_index);
    }
    NVLOGC_FMT(TAG, "Default CUS port index is: {}", cur_if_idx);
}

// Function to replace the network interface in the PTP configuration file
bool replaceInterfaceInConfig(const std::string &ptpCfgFilePath, std::string &if_name)
{
    std::ifstream configFile(ptpCfgFilePath);
    if (!configFile.is_open())
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYOAM_EVENT, "Error opening configuration file.");
        return false;
    }

    std::vector<std::string> lines;
    std::string line;
    bool interfaceReplaced = false;

    // Read the configuration file line by line
    while (std::getline(configFile, line))
    {
        // Check if the line contains the network interface configuration
        if (line.find(ifIdx2Name[cur_if_idx]) != std::string::npos)
        {
            // Replace the existing interface with the new one
            std::stringstream ss;
            ss << "[" << if_name << "]";
            line = ss.str();
            interfaceReplaced = true;
        }
        lines.push_back(line);
    }

    configFile.close();

    if (!interfaceReplaced)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYOAM_EVENT, "Failed to replace interface in ptp config file.");
        return false;
    }

    // Write the updated configuration back to the file
    std::ofstream outFile(ptpCfgFilePath);
    if (!outFile.is_open())
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYOAM_EVENT, "Error opening configuration file for writing.");
        return false;
    }

    for (const auto &outputLine : lines)
    {
        outFile << outputLine << std::endl;
    }

    outFile.close();
    return true;
}

// Function to restart the ptp4l service
void restartPtp4lService()
{
    // Use the system call to restart the ptp4l service
    int result = std::system("sudo systemctl restart ptp4l");
    if (result != 0)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYOAM_EVENT, "Failed to restart ptp4l service.");
    }
    else
    {
        NVLOGC_FMT(TAG, "ptp4l service restarted successfully.");
    }
}

int switchSPlanePort(std::string &if_name)
{
    if (replaceInterfaceInConfig(ptpCfgFilePath, if_name))
    {
        restartPtp4lService();
        NVLOGC_FMT(TAG, "Successfully switch CUS port to interface {}", if_name);
        return 0;
    }
    NVLOGE_FMT(TAG, AERIAL_CUPHYOAM_EVENT, "Failed to switch CUS port to interface {}", if_name);
    return -1;
}

// Function to handle the link up event
void handleLinkUpEvent(const struct ifinfomsg *ifi)
{
    NVLOGC_FMT(TAG, "Link up event detected on interface index: {}", ifi->ifi_index);
    ifIdx2Status[ifi->ifi_index] = true;
}

// Function to handle the link down event
void handleLinkDownEvent(const struct ifinfomsg *ifi)
{
    auto start = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch());
    NVLOGC_FMT(TAG, "Link down event detected on interface index: {}", ifi->ifi_index);
    bool init_stage = ifIdx2Status.size() != ifIdx2Name.size();
    ifIdx2Status[ifi->ifi_index] = false;
    if (init_stage || ifi->ifi_index != cur_if_idx)
    {
        return;
    }
    for (auto &[idx, status] : ifIdx2Status)
    {
        if (status)
        {
#if 1
            if (switchCUPlanePort(ifIdx2PCIeAddr[idx]) != 0)
            {
                NVLOGC_FMT(TAG, "C/U Plane port failover failed.. ");
            }
            auto t0_end = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch());
#endif
            if (switchSPlanePort(ifIdx2Name[idx]) != 0)
            {
                NVLOGC_FMT(TAG, "S Plane port failover failed..");
            }
            auto t1_end = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch());
            auto t0_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(t0_end - start);
            auto t1_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(t1_end - start);
            NVLOGC_FMT(TAG, "CU Plane port failover took {} nanoseconds.", t0_duration.count());
            NVLOGC_FMT(TAG, "CUS Plane port failover took {} nanoseconds.", t1_duration.count());
            cur_if_idx = idx;
            break;
        }
    }
}

// Function to process netlink messages
void processNetlinkMessage(int sock)
{
    struct
    {
        struct nlmsghdr nlh;
        struct rtgenmsg g;
    } req;

    memset(&req, 0, sizeof(req));
    req.nlh.nlmsg_len = NLMSG_LENGTH(sizeof(struct rtgenmsg));
    req.nlh.nlmsg_type = RTM_GETLINK;
    req.nlh.nlmsg_flags = NLM_F_REQUEST | NLM_F_DUMP;
    req.nlh.nlmsg_seq = 1;
    req.g.rtgen_family = AF_PACKET;

    if (send(sock, &req, req.nlh.nlmsg_len, 0) < 0)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYOAM_EVENT, "Failed to send init poll request.");
        close(sock);
        return;
    }

    char buffer[MSG_BUF_SIZE];
    int len;
    struct nlmsghdr *nlh;

    while ((len = recv(sock, buffer, sizeof(buffer), 0)) > 0)
    {
        for (nlh = (struct nlmsghdr *)buffer; NLMSG_OK(nlh, len); nlh = NLMSG_NEXT(nlh, len))
        {
            if (nlh->nlmsg_type == NLMSG_DONE)
            {
                break;
            }

            if (nlh->nlmsg_type == RTM_NEWLINK)
            {
                struct ifinfomsg *ifi = (struct ifinfomsg *)NLMSG_DATA(nlh);
                if (ifIdx2Name.find(ifi->ifi_index) == ifIdx2Name.end())
                {
                    // filter out the ones we don't care
                    continue;
                }
                if (ifi->ifi_flags & IFF_LOWER_UP)
                {
                    // Interface is up
                    handleLinkUpEvent(ifi);
                }
                else if (!(ifi->ifi_flags & IFF_RUNNING))
                {
                    // Interface is down
                    handleLinkDownEvent(ifi);
                }
            }
        }
    }
}

void* cus_conn_mgr_thread_func(void *arg)
{
    // Switch to a low_priority_core to avoid blocking time critical thread
    auto& appConfig = AppConfig::getInstance();
    auto low_priority_core = appConfig.getLowPriorityCore();
    NVLOGD_FMT(TAG, "{}: AerialCUSConnMgr thread affinity set to cpu core {}", __func__, low_priority_core);
    nv_assign_thread_cpu_core(low_priority_core);

    if(pthread_setname_np(pthread_self(), "AerialCUSConnMgr") != 0)
    {
        NVLOGW_FMT(TAG, "{}: set thread name failed", __func__);
    }


    NVLOGC_FMT(TAG, "AerialCUSConnMgr started...");

    NVLOGC_FMT(TAG, "Default port: '{}',  backup port: '{}'", if_names[0], if_names[1]);

    int netlinkSocket = socket(AF_NETLINK, SOCK_RAW, NETLINK_ROUTE);
    if (netlinkSocket < 0)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYOAM_EVENT, "Failed to create netlink socket.");
        return nullptr;
    }

    struct sockaddr_nl addr;
    memset(&addr, 0, sizeof(addr));
    addr.nl_family = AF_NETLINK;
    addr.nl_groups = RTMGRP_LINK;

    if (bind(netlinkSocket, (struct sockaddr *)&addr, sizeof(addr)) < 0)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYOAM_EVENT, "Failed to bind netlink socket.");
        close(netlinkSocket);
        return nullptr;
    }

    fetchIfIdx();
    getPCIeAddr();

    NVLOGC_FMT(TAG, "Listening for link down events...");
    processNetlinkMessage(netlinkSocket);

    close(netlinkSocket);
    return nullptr;
}
