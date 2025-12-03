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

#include "pdump_client.hpp"

#include "flow.hpp"
#include "nic.hpp"
#include "peer.hpp"
#include "queue.hpp"

#define TAG "FH.PDUMP"

namespace aerial_fh
{
static void pcap_client_worker(PdumpClient* pdump_client)
{
    auto fhi = pdump_client->get_fhi();
    do {
        for(auto nic : fhi->nics())
        {
            auto rxq = nic->get_pcap_rxq();
            rxq->receive();
        }
        std::this_thread::sleep_for(std::chrono::microseconds(kPdumpWorkerSpinDelayUs));
    } while(!pdump_client->signal_exit());
}

PdumpClient::PdumpClient(Fronthaul* fhi, int32_t cpu_core) :
    fhi_{fhi},
    signal_exit_{false}
{
    NVLOGI_FMT(TAG, "Initializing pdump client on core {}", cpu_core);

    auto ret = rte_pdump_init();
    if(ret)
    {
        THROW_FH(ret, "Failed to initialize packet capturing handling");
    }

    thread_ = std::make_unique<std::thread>(pcap_client_worker, this);

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_core, &cpuset);

    ret = pthread_setaffinity_np(thread_->native_handle(), sizeof(cpuset), &cpuset);
    if(ret)
    {
        THROW_FH(ret, "Failed to set affinity for pdump client worker");
    }
}

PdumpClient::~PdumpClient()
{
    signal_exit_ = true;
    thread_->join();

    if(rte_pdump_uninit())
    {
        NVLOGE_FMT(TAG, AERIAL_DPDK_API_EVENT, "Failed to uninitialize packet capturing handling");
    }
}

Fronthaul* PdumpClient::get_fhi() const
{
    return fhi_;
}

bool PdumpClient::signal_exit()
{
    return signal_exit_.load();
}

} // namespace aerial_fh
