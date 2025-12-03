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

#ifndef AERIAL_RIVERMAX_HPP__
#define AERIAL_RIVERMAX_HPP__

#include "aerial-fh-driver/api.hpp"
#include "fronthaul.hpp"

#ifdef ENABLE_RIVERMAX
    #include "rivermax_api.h"
#else
    struct rmax_init_config{};
    struct rmax_in_stream_type{};
    struct rmax_in_buffer_attr_flags_t{};
    struct rmax_in_timestamp_format{};
    struct rmax_in_buffer_attr{};
    typedef int rmax_stream_id;
    struct rmax_in_memblock{};
    struct rmax_in_flow_attr_ex{};

    typedef enum {
        RMAX_OK                                  =   0,
        RMAX_ERR_NO_DEVICE                       =   8,
        RMAX_ERR_UNKNOWN_ISSUE                   =  13,
        RMAX_ERR_LAST                            = 100
    } rmax_status_t;

#endif

#define ECPRI_NUMBER_OF_PRB_IN_SYMBOL 273
#define ECPRI_NUMBER_OF_PRB_IN_SYMBOL_ALIGNED 512
#define ECPRI_NUMBER_OF_SYMBOLS_IN_SLOT 14
#define ECPRI_NUMBER_OF_SYMBOLS_IN_SLOT_ALIGNED 16
#define ECPRI_PRB_SIZE_UNCOMPRESSED 48
#define ECPRI_PRB_SIZE_COMPRESSED_LEV_1 43
#define ECPRI_NOF_SLOTS 4
#define ECPRI_HEADER_SIZE 16

namespace aerial_fh
{

class RivermaxPrx {
public:
    RivermaxPrx(Fronthaul* fhi);
    ~RivermaxPrx();
    Fronthaul*  get_fronthaul() const;
    int         init_nic(std::string ip_address, socket_handle * local_socket_);
    int         query_buffer_size(int buffer_elements, int payload_unit_size, socket_handle* sockh,
                                                                size_t * payload_len, size_t * header_len);
    int         create_stream(void* addr, socket_handle* sockh, rmax_stream_id* stream_id);
    int         destroy_stream(rmax_stream_id stream_id);
    int         attach_flow(int flow_id, std::string destination_mac, std::string source_mac, uint64_t vlan_id,
                            uint16_t pc_id, int idx_slot, int sectionMask, int sectionId, rmax_stream_id stream_id,
                            socket_handle* sockh, rmax_in_flow_attr_ex* in_flow);
    int         detach_flow(rmax_stream_id stream_id, rmax_in_flow_attr_ex* in_flow);
    int         get_next_chunk(rmax_stream_id stream_id, int timeout, uint64_t* rx_bytes);

protected:
    Fronthaul*          fhi_;
    unsigned int        id_;
    rmax_in_stream_type  rx_type_;
    rmax_in_buffer_attr_flags_t attr_flags_;
    rmax_in_timestamp_format timestamp_format_;
    rmax_in_buffer_attr m_buffer; //< The payload attributes for the stream.
    rmax_stream_id stream_id; //< ID for the Rivermax stream object.
    struct rmax_in_memblock m_data;   //< Memory block used for the data (payloads).
    struct rmax_in_memblock m_header; //< Memory block used for the headers.
    size_t header_size_;
};

} // namespace aerial_fh

#endif //ifndef AERIAL_RIVERMAX_HPP__