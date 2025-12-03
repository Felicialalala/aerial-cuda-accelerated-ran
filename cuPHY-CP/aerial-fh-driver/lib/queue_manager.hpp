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

#ifndef AERIAL_FH_QUEUE_MANAGER_HPP__
#define AERIAL_FH_QUEUE_MANAGER_HPP__

#include "queue.hpp"
#include "utils.hpp"
#include "aerial-fh-driver/fh_mutex.hpp"

namespace aerial_fh
{
class Nic;

using TxqsUnique = std::vector<std::unique_ptr<Txq>>;
using RxqsUnique = std::vector<std::unique_ptr<Rxq>>;

class QueueManager {
public:
    QueueManager(Nic* nic);

    void add(Txq* txq);
    void add(Rxq* rxq);
    void add(RxqPcap* rxq_pcap);

    Txq*     assign_txq(bool is_gpu);
    Rxq*     assign_rxq();
    RxqPcap* get_pcap_rxq() const;

    void reclaim(Txq* txq);
    void reclaim(Rxq* rxq);

    void assign_shared_txqs(Iterator<Txq*>& txqs, size_t num, bool is_gpu);

    void warm_up_txqs();
    void init_gpu_txqs();

protected:
    Nic*       nic_;
    aerial_fh::FHMutex mtx_;

    TxqsUnique               txqs_;
    TxqsUnique               txqs_gpu_;
    RxqsUnique               rxqs_;
    std::vector<Txq*>        txqs_free_;
    std::vector<Txq*>        txqs_free_gpu_;
    std::vector<Rxq*>        rxqs_free_;
    std::unique_ptr<RxqPcap> rxq_pcap_;
};
} // namespace aerial_fh

#endif //ifndef AERIAL_FH_QUEUE_MANAGER_HPP__
