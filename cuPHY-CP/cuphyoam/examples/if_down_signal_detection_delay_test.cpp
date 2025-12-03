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
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>
#include <fstream>
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
// #include <grpcpp/grpcpp.h>

std::string if_names[] = {"aerial00", "aerial01"};

std::unordered_map<int, std::string> ifIdx2Name;
std::unordered_map<int, bool> ifIdx2Status;
std::unordered_map<int, std::string> ifIdx2PCIeAddr;

static int cur_if_idx = -1;

void fetchIfIdx()
{
    unsigned int port_index;
    for (auto &if_name : if_names)
    {
        if ((port_index = if_nametoindex(if_name.c_str())) == 0)
        {
            // NVLOGE_FMT(TAG, AERIAL_CUPHYOAM_EVENT, "No index found for interface: '{}'", if_name);
            continue;
        }
        ifIdx2Name[port_index] = if_name;
        if (if_name == if_names[0])
        {
            cur_if_idx = port_index;
        }
        //NVLOGC_FMT(TAG, "Interface '{}' index is: {}", if_name, port_index);
    }
    //NVLOGC_FMT(TAG, "Default CUS port index is: {}", cur_if_idx);
}

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
        // NVLOGC_FMT(TAG, "Interface idx {} PCIe address is: {} ", ifIdx, ifIdx2PCIeAddr[ifIdx]);
    }

end:
    close(sd);
    freeifaddrs(ifaddr);
}

std::string exec(const char *cmd)
{
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe)
    {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr)
    {
        result += buffer.data();
    }
    return result;
}

int exec_str(std::string cmd)
{
    return std::system(cmd.c_str());
}

int checkInterfaceStatus(const std::string &interfaceName)
{
    std::string command = "ethtool " + interfaceName + " | grep 'Link detected'";
    std::string output = exec(command.c_str());
    int res = -1;
    if (output.find("yes") != std::string::npos)
    {
        // std::cout << "Interface " << interfaceName << " is up." << std::endl;
    }
    else
    {
        res = 0;
        // std::cout << "Interface " << interfaceName << " is down." << std::endl;
    }
    return res;
}

int checkInterfaceStatusViaMlxlink(const std::string &interfaceName)
{
    std::string command = "mlxlink -d " + ifIdx2PCIeAddr[cur_if_idx] + " | grep 'Physical state'";
    std::string output = exec(command.c_str());
    int res = -1;
    //if (output.find("LinkDown") != std::string::npos)
    if (output.find("LinkUp") != std::string::npos)
    {
        // std::cout << "Interface " << interfaceName << " is up." << std::endl;
    }
    else
    {
        res = 0;
        // std::cout << "Interface " << interfaceName << " is down." << std::endl;
    }
    return res;
}

void ethtool_delay()
{
    try
    {
        exec_str("sudo ifconfig aerial00 down");
        auto start = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch());
        while (checkInterfaceStatus("aerial00") != 0)
            ; // Replace 'aerial00' with your network interface name
        auto end = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch());
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        printf("ethtool delay %ld  nanoseconds.\n", duration.count());
        exec_str("sudo ifconfig aerial00 up");
        while (checkInterfaceStatus("aerial00") == 0)
            ; // Replace 'aerial00' with your network interface name
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        // return 1;
    }
}

void mlxlink_delay()
{
    try
    {
        exec_str("sudo ifconfig aerial00 down");
        auto start = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch());
        while (checkInterfaceStatusViaMlxlink("aerial00") != 0)
            ; // Replace 'aerial00' with your network interface name
        auto end = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch());
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        printf("mlxlink delay %ld  nanoseconds.\n", duration.count());
        exec_str("sudo ifconfig aerial00 up");
        while (checkInterfaceStatus("aerial00") == 0)
            ; // Replace 'aerial00' with your network interface name
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}

// Function to process netlink messages
void processNetlinkMessage(int sock)
{
    char buffer[4096];
    int len;
    struct nlmsghdr *nlh;

    exec_str("sudo ifconfig aerial00 down");
    auto start = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch());

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
                    // handleLinkUpEvent(ifi);
                }
                else if (!(ifi->ifi_flags & IFF_RUNNING))
                {
                    // Interface is down
                    // handleLinkDownEvent(ifi);
                    auto end = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch());
                    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
                    printf("netlink delay %ld  nanoseconds.\n", duration.count());
                    exec_str("sudo ifconfig aerial00 up");
                    while (checkInterfaceStatus("aerial00") == 0)
                        ; // Replace 'aerial00' with your network interface name
                    return;
                }
            }
        }
    }
}

void netlink_delay()
{
    int netlinkSocket = socket(AF_NETLINK, SOCK_RAW, NETLINK_ROUTE);
    if (netlinkSocket < 0)
    {
        //NVLOGE_FMT(TAG, AERIAL_CUPHYOAM_EVENT, "Failed to create netlink socket.");
        // return;
    }

    struct sockaddr_nl addr;
    memset(&addr, 0, sizeof(addr));
    addr.nl_family = AF_NETLINK;
    addr.nl_groups = RTMGRP_LINK;

    if (bind(netlinkSocket, (struct sockaddr *)&addr, sizeof(addr)) < 0)
    {
        //NVLOGE_FMT(TAG, AERIAL_CUPHYOAM_EVENT, "Failed to bind netlink socket.");
        close(netlinkSocket);
        //return;
    }

    fetchIfIdx();
    getPCIeAddr();

    //NVLOGC_FMT(TAG, "Listening for link down events...");
    processNetlinkMessage(netlinkSocket);
}

int main()
{
    ethtool_delay();
    sleep(2);
    netlink_delay();
    sleep(2);
    mlxlink_delay();
    return 0;
}