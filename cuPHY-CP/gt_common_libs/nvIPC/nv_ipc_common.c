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

#include <stdlib.h>

#include "nv_ipc.h"
#include "nv_ipc_utils.h"
#include "nv_ipc_shm.h"
#include "nv_ipc_debug.h"

#define TAG (NVLOG_TAG_BASE_NVIPC + 8) //!< Log tag "NVIPC.IPC"


#define NVIPC_DEFAULT_PREFIX "nvipc"

int shm_ipc_set_reset_callback(nv_ipc_t *ipc, int (*callback)(void *), void *cb_args);
nv_ipc_t* create_shm_nv_ipc_interface(const nv_ipc_config_t* cfg);
nv_ipc_t* create_udp_nv_ipc_interface(const nv_ipc_config_t* cfg);
#ifdef NVIPC_DPDK_ENABLE
nv_ipc_t* create_dpdk_nv_ipc_interface(const nv_ipc_config_t* cfg);
#endif
#ifdef NVIPC_DOCA_ENABLE
nv_ipc_t* create_doca_nv_ipc_interface(const nv_ipc_config_t* cfg);
#endif

/** IPC instance linked list node */
typedef struct nv_ipc_link_node_t nv_ipc_link_node_t;
struct nv_ipc_link_node_t
{
    nv_ipc_t*           ipc;                    //!< IPC instance pointer
    nv_ipc_link_node_t* next;                   //!< Next node in list
    char                prefix[NV_NAME_MAX_LEN]; //!< Instance name prefix
    nv_ipc_config_t     config;                 //!< Instance configuration
};

static nv_ipc_link_node_t* ipc_list_head = NULL;

static void nv_ipc_add_instance(nv_ipc_t* ipc, const char* prefix, const nv_ipc_config_t* cfg)
{
    nv_ipc_link_node_t* pnode = malloc(sizeof(nv_ipc_link_node_t));
    if (pnode == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "%s: malloc failed", __func__);
        return;
    }

    pnode->ipc                = ipc;
    pnode->next               = NULL;
    nvlog_safe_strncpy(pnode->prefix, prefix, NV_NAME_MAX_LEN);
    memcpy(&pnode->config, cfg, sizeof(nv_ipc_config_t));

    if(ipc_list_head == NULL)
    {
        ipc_list_head = pnode;
    }
    else
    {
        nv_ipc_link_node_t* curr = ipc_list_head;
        while(curr->next != NULL)
        {
            curr = curr->next;
        }
        curr->next = pnode;
    }
}

nv_ipc_config_t* nv_ipc_get_config_instance(const nv_ipc_t* ipc)
{
    nv_ipc_config_t* cfg = NULL;

    if(ipc == NULL)
    {
        return ipc_list_head == NULL ? NULL : &ipc_list_head->config;
    }
    else
    {
        nv_ipc_link_node_t* curr;
        for(curr = ipc_list_head; curr != NULL; curr = curr->next)
        {
            if(ipc == curr->ipc)
            {
                break;
            }
        }
        return curr == NULL ? NULL : &curr->config;
    }
}

nv_ipc_config_t* nv_ipc_get_config_by_name(const char* prefix)
{
    nv_ipc_config_t* cfg = NULL;

    if(prefix == NULL)
    {
        return ipc_list_head == NULL ? NULL : &ipc_list_head->config;
    }
    else
    {
        nv_ipc_link_node_t* curr;
        for(curr = ipc_list_head; curr != NULL; curr = curr->next)
        {
            if(strncmp(prefix, curr->prefix, NV_NAME_MAX_LEN - 1))
            {
                break;
            }
        }
        return curr == NULL ? NULL : &curr->config;
    }
}

nv_ipc_t* nv_ipc_get_instance(const char* prefix)
{
    if(prefix == NULL)
    {
        return ipc_list_head == NULL ? NULL : ipc_list_head->ipc;
    }
    else
    {
        nv_ipc_link_node_t* curr;
        for(curr = ipc_list_head; curr != NULL; curr = curr->next)
        {
            if(strncmp(prefix, curr->prefix, NV_NAME_MAX_LEN - 1))
            {
                break;
            }
        }
        return curr == NULL ? NULL : curr->ipc;
    }
}

int is_module_primary(nv_ipc_module_t module_type)
{
    switch (module_type)
    {
    case NV_IPC_MODULE_PHY:
    case NV_IPC_MODULE_PRIMARY:
        return 1;
    case NV_IPC_MODULE_MAC:
    case NV_IPC_MODULE_SECONDARY:
    case NV_IPC_MODULE_IPC_DUMP:
        return 0;
    default: // Invalid module type
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s unknown module_type: %d", __func__, module_type);
        return 0;
    }
}

int nv_ipc_get_buf_size(const nv_ipc_config_t* cfg, nv_ipc_mempool_id_t pool_id)
{
    if (cfg->ipc_transport == NV_IPC_TRANSPORT_SHM)
    {
        return cfg->transport_config.shm.mempool_size[pool_id].buf_size;
    }
    else if (cfg->ipc_transport == NV_IPC_TRANSPORT_DPDK)
    {
        return cfg->transport_config.dpdk.mempool_size[pool_id].buf_size;
    }
    else if (cfg->ipc_transport == NV_IPC_TRANSPORT_DOCA)
    {
        return cfg->transport_config.doca.mempool_size[pool_id].buf_size;
    }
    else
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s unknown transport type %d", __func__, cfg->ipc_transport);
        return -1;
    }
}

int nv_ipc_get_pool_len(const nv_ipc_config_t* cfg, nv_ipc_mempool_id_t pool_id)
{
    if (cfg->ipc_transport == NV_IPC_TRANSPORT_SHM)
    {
        return cfg->transport_config.shm.mempool_size[pool_id].pool_len;
    }
    else if (cfg->ipc_transport == NV_IPC_TRANSPORT_DPDK)
    {
        return cfg->transport_config.dpdk.mempool_size[pool_id].pool_len;
    }
    else if (cfg->ipc_transport == NV_IPC_TRANSPORT_DOCA)
    {
        return cfg->transport_config.doca.mempool_size[pool_id].pool_len;
    }
    else
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s unknown transport type %d", __func__, cfg->ipc_transport);
        return -1;
    }
}

int nv_ipc_dump(nv_ipc_t* ipc)
{
    nv_ipc_config_t* cfg = nv_ipc_get_config_instance(ipc);
    if (cfg == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s instance not exist: ipc=0x%p", __func__, ipc);
        return -1;
    }

    if(cfg->ipc_transport == NV_IPC_TRANSPORT_SHM)
    {
        return shm_ipc_dump(ipc);
    }
    else if(cfg->ipc_transport == NV_IPC_TRANSPORT_DPDK)
    {
        NVLOGI(TAG, "%s: TODO module_type=%d, transport=%d", __func__, cfg->module_type,
                cfg->ipc_transport);
        return -1;
    }
#ifdef NVIPC_DOCA_ENABLE
    else if(cfg->ipc_transport == NV_IPC_TRANSPORT_DOCA)
    {
        return doca_ipc_dump(ipc);
    }
#endif
    else
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s unsupported transport: %d", __func__, cfg->ipc_transport);
        return -1;
    }
}

int nv_ipc_set_reset_callback(nv_ipc_t *ipc, int (*callback)(void *), void *cb_args)
{
    nv_ipc_config_t* cfg = nv_ipc_get_config_instance(ipc);
    if (cfg == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s instance not exist: ipc=0x%p", __func__, ipc);
        return -1;
    }

    if (cfg->ipc_transport == NV_IPC_TRANSPORT_SHM)
    {
        return shm_ipc_set_reset_callback(ipc, callback, cb_args);
    }

    NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s unsupported transport: %d", __func__, cfg->ipc_transport);
    return -1;
}

nv_ipc_t* create_nv_ipc_interface(nv_ipc_config_t* cfg)
{
    NVLOGI(TAG, "%s: module_type=%d, transport=%d", __func__, cfg->module_type, cfg->ipc_transport);

    if(cfg->module_type < 0 || cfg->module_type >= NV_IPC_MODULE_MAX)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s unsupported module_type: %d", __func__, cfg->module_type);
        return NULL;
    }

    nv_ipc_t* ipc = NULL;
    if(cfg->ipc_transport == NV_IPC_TRANSPORT_SHM)
    {
        ipc = create_shm_nv_ipc_interface(cfg);
        nv_ipc_add_instance(ipc, cfg->transport_config.shm.prefix, cfg);
        if((cfg->transport_config.shm.mempool_size[NV_IPC_MEMPOOL_GPU_DATA].pool_len > 0) && (cfg->transport_config.shm.mempool_size[NV_IPC_MEMPOOL_GPU_DATA].buf_size > 0))
        {
            /* If we are here, that means GPU_DATA mempool has been successfully created */
            NVLOGI(TAG, "%s: NV_IPC_CFG_FAPI_TB_LOC set to 3", __func__);
        }
    }
    else if(cfg->ipc_transport == NV_IPC_TRANSPORT_UDP)
    {
        ipc = create_udp_nv_ipc_interface(cfg);
        nv_ipc_add_instance(ipc, NVIPC_DEFAULT_PREFIX, cfg);
    }
#ifdef NVIPC_DPDK_ENABLE
    else if(cfg->ipc_transport == NV_IPC_TRANSPORT_DPDK)
    {
        ipc = create_dpdk_nv_ipc_interface(cfg);
        nv_ipc_add_instance(ipc, cfg->transport_config.dpdk.prefix, cfg);
    }
#endif
#ifdef NVIPC_DOCA_ENABLE
    else if(cfg->ipc_transport == NV_IPC_TRANSPORT_DOCA)
    {
        ipc = create_doca_nv_ipc_interface(cfg);
        nv_ipc_add_instance(ipc, cfg->transport_config.doca.prefix, cfg);
    }
#endif
    else
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s unsupported transport: %d", __func__, cfg->ipc_transport);
        return NULL;
    }

    return ipc;
}
