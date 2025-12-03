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

#include "fh_stats_dump.hpp"
#include "nic.hpp"
#include "cuphyoam.hpp"
#include "app_config.hpp"

#define TAG "FH.STATS"

namespace aerial_fh
{
static void fh_stats_polling_worker(FHStatsDump* fh_stats_dump)
{
    nvlog_fmtlog_thread_init("fh_stats");
    sleep(1);//Add a delay here, nic might haven't been initialized
    auto fhi = fh_stats_dump->get_fhi();
    do {
        for(auto nic : fhi->nics())
        {
            nic->fh_extended_stats_retrieval();

            {
                auto cur_t        = Time::now_ns();
                auto port_id      = nic->get_port_id();
                if(fh_stats_dump->nic_tx_polling_info.find(port_id) != fh_stats_dump->nic_tx_polling_info.end() && cur_t - fh_stats_dump->nic_tx_polling_info[port_id].first < kNicTputPollingInterval)
                {
                    continue;
                }

                auto cur_tx_bytes = nic->get_tx_bytes_phy();

                if(fh_stats_dump->nic_tx_polling_info.find(port_id) != fh_stats_dump->nic_tx_polling_info.end())
                {
                    auto prev_t        = fh_stats_dump->nic_tx_polling_info[port_id].first;
                    auto prev_tx_bytes = fh_stats_dump->nic_tx_polling_info[port_id].second;

                    auto diff_t = cur_t - prev_t;
                    auto tput = (double)((cur_tx_bytes - prev_tx_bytes) * 8 * 1000) / ((double)diff_t);

                    auto& appConfig                = AppConfig::getInstance();
                    auto  nic_tput_alert_threshold = appConfig.getNicTputAlertThreshold();
                    if(tput > nic_tput_alert_threshold)
                    {
                        std::string msg = "Nic " + nic->get_if_name() + " tput: " + std::to_string(tput) + " Mbps exceeds threshold " + std::to_string(nic_tput_alert_threshold);
                        auto        oam = CuphyOAM::getInstance();
                        oam->notifyAll(msg);
                    }
                }
                fh_stats_dump->nic_tx_polling_info[port_id] = {cur_t, cur_tx_bytes};
            }
        }
        std::this_thread::sleep_for(std::chrono::microseconds(kFHStatsPollingDelayUs));
    } while(!fh_stats_dump->signal_exit());
}

FHStatsDump::FHStatsDump(Fronthaul* fhi, int32_t cpu_core) :
    fhi_{fhi},
    signal_exit_{false}
{
    thread_ = std::make_unique<std::thread>(fh_stats_polling_worker, this);
    if(cpu_core >= 0)
    {
        NVLOGI_FMT(TAG, "Initializing fh stats dump thread on core {}", cpu_core);
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(cpu_core, &cpuset);

        auto ret = pthread_setaffinity_np(thread_->native_handle(), sizeof(cpuset), &cpuset);
        if(ret)
        {
            THROW_FH(ret, "Failed to set affinity for fh stats dump thread");
        }

        ret = pthread_setname_np(thread_->native_handle(), "fh_stats");
        if(ret != 0)
        {
            THROW_FH(ret, "Failed to set fh_stats polling worker thread name");
        }
    }
}

FHStatsDump::~FHStatsDump()
{
    signal_exit_ = true;
    if(thread_.get() != nullptr)
    {
        thread_->join();
    }
}

Fronthaul* FHStatsDump::get_fhi() const
{
    return fhi_;
}

bool FHStatsDump::signal_exit()
{
    return signal_exit_.load();
}

} // namespace aerial_fh
