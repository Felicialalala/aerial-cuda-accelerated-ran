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

#include "queue_manager.hpp"
#include "nic.hpp"
#include "gpu_comm.hpp"

#define TAG "FH.QMANAGER"

namespace aerial_fh
{
QueueManager::QueueManager(Nic* nic) :
    nic_{nic}
{
}

void QueueManager::add(Txq* txq)
{
    const std::lock_guard<aerial_fh::FHMutex> lock(mtx_);
    if (txq->is_gpu()) {
        txqs_gpu_.emplace_back(txq);
        txqs_free_gpu_.push_back(txq);
    } else {
        txqs_.emplace_back(txq);
        txqs_free_.push_back(txq);
    }
}

void QueueManager::add(Rxq* rxq)
{
    const std::lock_guard<aerial_fh::FHMutex> lock(mtx_);
    rxqs_.emplace_back(rxq);
    rxqs_free_.push_back(rxq);
}

void QueueManager::add(RxqPcap* rxq_pcap)
{
    const std::lock_guard<aerial_fh::FHMutex> lock(mtx_);
    rxq_pcap_.reset(rxq_pcap);
}

Txq* QueueManager::assign_txq(bool is_gpu)
{
    const std::lock_guard<aerial_fh::FHMutex> lock(mtx_);

    Txq* txq = nullptr;

    if (is_gpu) {
        if (txqs_free_gpu_.empty()) {
            THROW_FH(ENOMEM, StringBuilder() << "Ran out of TXQs GPU. Please increase TXQ GPU count for NIC " << nic_->get_name());
        }

        txq = txqs_free_gpu_.back();
        txqs_free_gpu_.pop_back();
    } else {
        if (txqs_free_.empty()) {
            THROW_FH(ENOMEM, StringBuilder() << "Ran out of TXQs. Please increase TXQ count for NIC " << nic_->get_name());
        }

        txq = txqs_free_.back();
        txqs_free_.pop_back();
    }

    return txq;
}

void QueueManager::assign_shared_txqs(Iterator<Txq*>& txqs, size_t num, bool is_gpu)
{
    const std::lock_guard<aerial_fh::FHMutex> lock(mtx_);
    if (is_gpu)
    {
        if(num > txqs_free_gpu_.size())
        {
            THROW_FH(ENOMEM, StringBuilder() << "Not enough GPU TXQs . Please increase TXQ count for NIC " << nic_->get_name());
        }

        for(int i = 0; i < num; ++i)
        {
            txqs.add(txqs_free_gpu_[i]);
        }
    }
    else
    {
        if(num > txqs_free_.size())
        {
            THROW_FH(ENOMEM, StringBuilder() << "Not enough CPU TXQs. Please increase TXQ count for NIC " << nic_->get_name());
        }

        for(int i = 0; i < num; ++i)
        {
            txqs.add(txqs_free_[i]);
        }
    }
}

Rxq* QueueManager::assign_rxq()
{
    const std::lock_guard<aerial_fh::FHMutex> lock(mtx_);

    if(rxqs_free_.empty())
    {
        THROW_FH(ENOMEM, StringBuilder() << "Ran out of RXQs. Please increase RXQ count for NIC " << nic_->get_name());
    }

    auto rxq = rxqs_free_.back();
    rxqs_free_.pop_back();
    return rxq;
}

RxqPcap* QueueManager::get_pcap_rxq() const
{
    return rxq_pcap_.get();
}

void QueueManager::reclaim(Txq* txq)
{
    const std::lock_guard<aerial_fh::FHMutex> lock(mtx_);
    if(txq->is_gpu())
        txqs_free_gpu_.push_back(txq);
    else
        txqs_free_.push_back(txq);
}

void QueueManager::reclaim(Rxq* rxq)
{
    const std::lock_guard<aerial_fh::FHMutex> lock(mtx_);
    rxqs_free_.push_back(rxq);
}

void QueueManager::warm_up_txqs()
{
    for(auto txq : txqs_free_)
    {
        txq->warm_up();
    }
}

void QueueManager::init_gpu_txqs()
{
    GpuComm * gcomm = nic_->get_gpu_comm();

    for(auto txq : txqs_free_gpu_)
    {
        if (txq->is_gpu())
            gcomm->txq_init(txq);
    }
}

} // namespace aerial_fh
