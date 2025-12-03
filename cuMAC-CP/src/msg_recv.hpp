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

#ifndef _MSG_RECV_HPP_
#define _MSG_RECV_HPP_

#include <string.h>
#include <sys/time.h>
#include <semaphore.h>

#include <vector>

#include "cumac.h"
#include "nvlog.hpp"
#include "nv_utils.h"
#include "nv_ipc_utils.h"
#include "nv_phy_utils.hpp"
#include "cumac_app.hpp"
#include "cumac_cp_configs.hpp"

#include "nv_phy_mac_transport.hpp"
#include "nv_phy_epoll_context.hpp"
#include "nvlog.hpp"
#include "nv_lockfree.hpp"
#include "cumac_task.hpp"
#include "cumac_cp_handler.hpp"
#include "cumac.h"
#include "api.h"

/**
 * Worker thread arguments structure
 *
 * Contains configuration and resources for cuMAC worker threads
 * that process scheduled tasks from the task ring
 */
typedef struct {
    pthread_t pthread_id; //!< Thread ID from pthread_create
    sem_t* task_sem; //!< Semaphore for task synchronization
    nv::lock_free_ring_pool<cumac_task>* task_ring; //!< Lock-free ring pool for task queuing
    int cpu_core; //!< CPU core affinity for this thread
    char thread_name[16]; //!< Thread name for debugging (max 15 chars + null terminator)
} work_thread_arg_t;

/**
 * cuMAC message receiver and coordinator
 *
 * Manages IPC message reception from PHY layer, worker thread pool,
 * and task scheduling for cuMAC CP operations. Coordinates multi-cell
 * message handling and task distribution to worker threads.
 */
class cumac_receiver
{

public:
    /**
     * Construct cuMAC receiver
     *
     * @param[in] yaml_node YAML configuration node
     * @param[in] _configs Configuration parameters
     */
    cumac_receiver(yaml::node& yaml_node, cumac_cp_configs& _configs);

    /**
     * Destructor cleans up resources and stops threads
     */
    ~cumac_receiver();

    /**
     * Start receiver and worker threads
     */
    void start();

    /**
     * Stop receiver and worker threads
     */
    void stop();

    /**
     * Wait for receiver thread to complete
     */
    void join();

    /**
     * Start epoll event loop for message reception
     */
    void start_epoll_loop();

    /**
     * Message callback handler
     *
     * @param[in,out] msg Message descriptor from IPC transport
     *
     * @return true on success, false on error
     */
    bool on_msg(nv::phy_mac_msg_desc& msg);

    /**
     * Receive and process incoming messages
     */
    void recv_msg();

    /**
     * Get PHY-MAC transport for specific cell
     *
     * @param[in] cell_id Cell identifier
     *
     * @return Reference to transport instance
     */
    nv::phy_mac_transport& transport(int cell_id) { return transp_wrapper.get_transport(cell_id); }

    /**
     * Get PHY-MAC transport wrapper
     *
     * @return Reference to transport wrapper instance
     */
    nv::phy_mac_transport_wrapper& transport_wrapper() { return transp_wrapper; }

    /**
     * Get cuMAC CP handler instance
     *
     * @return Reference to handler
     */
    cumac_cp_handler& get_cumac_handler()
    {
        return _handler;
    }


private:

    int cell_num{}; //!< Number of cells being managed

    cumac_cp_configs& configs; //!< Configuration parameters reference

    nv::phy_mac_transport_wrapper transp_wrapper; //!< PHY-MAC IPC transport wrapper

    cumac_cp_handler _handler; //!< cuMAC CP message handler

    std::vector<std::vector<work_thread_arg_t>> worker_thread_args{}; //!< Worker thread arguments: [core][thread_per_core]

    sem_t _task_sem{}; //!< Semaphore for task synchronization
    nv::lock_free_ring_pool<cumac_task>* task_ring{}; //!< Lock-free ring pool for cuMAC tasks

    std::unique_ptr<nv::member_event_callback<cumac_receiver>> msg_mcb_p{}; //!< Message callback pointer
    std::unique_ptr<nv::phy_epoll_context> epoll_ctx_p{}; //!< Epoll context for event handling

    pthread_t recv_thread_id{}; //!< Receiver thread ID
};

#endif // _MSG_RECV_HPP_