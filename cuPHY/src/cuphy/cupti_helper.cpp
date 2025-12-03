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

#include <cupti.h>
#include <stdio.h>
#include <unistd.h>
#include <mutex>
#include <pthread.h>

#include "nvlog.hpp"
#include "cupti_helper.hpp"


#define TAG "CUPHY.CUPTI"

#define CHECK_CUDA(expr_to_check) do {            \
    cudaError_t result = expr_to_check;           \
    if(result != cudaSuccess)                     \
    {                                             \
        NVLOGF_FMT(TAG,                           \
                AERIAL_INTERNAL_EVENT,            \
                "CUDA Runtime Error: {}:{}:{}",   \
                __FILE__,                         \
                __LINE__,                         \
                cudaGetErrorString(result));      \
    }                                             \
} while (0)

#define CUPTI_EXTERNAL_CORRELATION_KIND_AERIAL CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM2

// 8-byte alignment for the buffers
#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                                 \
  (((uintptr_t) (buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t) (buffer) & ((align)-1))) : (buffer))

// For typical workloads, it's suggested to choose a size between 1 and 10 MB. " from https://docs.nvidia.com/cupti/r_main.html
// We choose 1GB to reduce BufferRequest() callback frequency in the high-priority threads
#define BUF_SIZE (1 * 1024 * 1024 * 1024)
constexpr int MAX_BUFFERS = 8;

#define CUPTI_API_CALL(apiFunctionCall)                                                                 \
do                                                                                                      \
{                                                                                                       \
    CUptiResult _status = apiFunctionCall;                                                              \
    if ((_status != CUPTI_SUCCESS) && (_status != CUPTI_ERROR_MAX_LIMIT_REACHED))                       \
    {                                                                                                   \
        const char *pErrorString;                                                                       \
        cuptiGetResultString(_status, &pErrorString);                                                   \
                                                                                                        \
        NVLOGF_FMT(TAG, AERIAL_CUPHY_EVENT, "{}:{}: Error: Function {} failed with error: {}.",         \
                __FILE__, __LINE__, #apiFunctionCall, pErrorString);                                    \
                                                                                                        \
        exit(EXIT_FAILURE);                                                                             \
    }                                                                                                   \
} while (0)

static const char *
GetName(
    const char *pName)
{
    if (pName == NULL)
    {
        return "<null>";
    }

    return pName;
}

static const char *
GetMemoryKindString(
    CUpti_ActivityMemoryKind memoryKind)
{
    switch (memoryKind)
    {
        case CUPTI_ACTIVITY_MEMORY_KIND_UNKNOWN:
            return "UNKNOWN";
        case CUPTI_ACTIVITY_MEMORY_KIND_PAGEABLE:
            return "PAGEABLE";
        case CUPTI_ACTIVITY_MEMORY_KIND_PINNED:
            return "PINNED";
        case CUPTI_ACTIVITY_MEMORY_KIND_DEVICE:
            return "DEVICE";
        case CUPTI_ACTIVITY_MEMORY_KIND_ARRAY:
            return "ARRAY";
        case CUPTI_ACTIVITY_MEMORY_KIND_MANAGED:
            return "MANAGED";
        case CUPTI_ACTIVITY_MEMORY_KIND_DEVICE_STATIC:
            return "DEVICE_STATIC";
        case CUPTI_ACTIVITY_MEMORY_KIND_MANAGED_STATIC:
            return "MANAGED_STATIC";
        default:
            return "<unknown>";
    }
}

static const char *
GetMemcpyKindString(
    CUpti_ActivityMemcpyKind memcpyKind)
{
    switch (memcpyKind)
    {
        case CUPTI_ACTIVITY_MEMCPY_KIND_UNKNOWN:
            return "UNKNOWN";
        case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
            return "HtoD";
        case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
            return "DtoH";
        case CUPTI_ACTIVITY_MEMCPY_KIND_HTOA:
            return "HtoA";
        case CUPTI_ACTIVITY_MEMCPY_KIND_ATOH:
            return "AtoH";
        case CUPTI_ACTIVITY_MEMCPY_KIND_ATOA:
            return "AtoA";
        case CUPTI_ACTIVITY_MEMCPY_KIND_ATOD:
            return "AtoD";
        case CUPTI_ACTIVITY_MEMCPY_KIND_DTOA:
            return "DtoA";
        case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD:
            return "DtoD";
        case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH:
            return "HtoH";
        case CUPTI_ACTIVITY_MEMCPY_KIND_PTOP:
            return "PtoP";
        default:
            return "<unknown>";
    }
}


static const char *
GetChannelType(
    CUpti_ChannelType channelType)
{
    switch (channelType)
    {
        case CUPTI_CHANNEL_TYPE_INVALID:
            return "INVALID";
        case CUPTI_CHANNEL_TYPE_COMPUTE:
            return "COMPUTE";
        case CUPTI_CHANNEL_TYPE_ASYNC_MEMCPY:
            return "ASYNC_MEMCPY";
        default:
            return "<unknown>";
    }
}

void PrintActivity(CUpti_Activity *pRecord)
{
  CUpti_ActivityKind activityKind = pRecord->kind;

    switch (activityKind)
    {
        case CUPTI_ACTIVITY_KIND_MEMCPY:
        {
            CUpti_ActivityMemcpy5 *pMemcpyRecord = (CUpti_ActivityMemcpy5 *)pRecord;

            NVLOGI_FMT(TAG, "MEMCPY \"{}\" [ {}, {} ] duration {}, size {}, srcKind {}, dstKind {}, correlationId {},"
                    "deviceId {}, contextId {}, streamId {}, graphId {}, graphNodeId {}, channelId {}, channelType {}",
                    GetMemcpyKindString((CUpti_ActivityMemcpyKind)pMemcpyRecord->copyKind),
                    (unsigned long long)pMemcpyRecord->start,
                    (unsigned long long)pMemcpyRecord->end,
                    (unsigned long long)(pMemcpyRecord->end - pMemcpyRecord->start),
                    (unsigned long long)pMemcpyRecord->bytes,
                    GetMemoryKindString((CUpti_ActivityMemoryKind)pMemcpyRecord->srcKind),
                    GetMemoryKindString((CUpti_ActivityMemoryKind)pMemcpyRecord->dstKind),
                    static_cast<uint32_t>(pMemcpyRecord->correlationId),
                    static_cast<uint32_t>(pMemcpyRecord->deviceId),
                    static_cast<uint32_t>(pMemcpyRecord->contextId),
                    static_cast<uint32_t>(pMemcpyRecord->streamId),
                    static_cast<uint32_t>(pMemcpyRecord->graphId),
                    (unsigned long long)pMemcpyRecord->graphNodeId,
                    static_cast<uint32_t>(pMemcpyRecord->channelID),
                    GetChannelType(pMemcpyRecord->channelType));

            break;
        }

        case CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION:
        {
            CUpti_ActivityExternalCorrelation *pExternalCorrelationRecord = (CUpti_ActivityExternalCorrelation *)pRecord;

            NVLOGI_FMT(TAG, "EXTERNAL_CORRELATION externalKind {}, correlationId {}, externalId {}",
                    static_cast<uint32_t>(pExternalCorrelationRecord->externalKind),
                    static_cast<uint32_t>(pExternalCorrelationRecord->correlationId),
                    static_cast<uint64_t>(pExternalCorrelationRecord->externalId));

            break;
        }

        case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
        {
            CUpti_ActivityKernel8 *pKernelRecord = (CUpti_ActivityKernel8 *)pRecord;

            NVLOGI_FMT(TAG, "CONCURRENT_KERNEL [ {}, {} ] duration {}, \"{}\", correlationId {}, "
                    "grid [ {}, {}, {} ], block [ {}, {}, {} ], sharedMemory (static {}, dynamic {}), "
                    "deviceId {}, contextId {}, streamId {}, graphId {}, graphNodeId {}, channelId {}",
                    static_cast<uint64_t>(pKernelRecord->start),
                    static_cast<uint64_t>(pKernelRecord->end),
                    static_cast<int64_t>(pKernelRecord->end - pKernelRecord->start),
                    GetName(pKernelRecord->name),
                    static_cast<uint32_t>(pKernelRecord->correlationId),
                    static_cast<int32_t>(pKernelRecord->gridX),
                    static_cast<int32_t>(pKernelRecord->gridY),
                    static_cast<int32_t>(pKernelRecord->gridZ),
                    static_cast<int32_t>(pKernelRecord->blockX),
                    static_cast<int32_t>(pKernelRecord->blockY),
                    static_cast<int32_t>(pKernelRecord->blockZ),
                    static_cast<int32_t>(pKernelRecord->staticSharedMemory),
                    static_cast<int32_t>(pKernelRecord->dynamicSharedMemory),
                    static_cast<uint32_t>(pKernelRecord->deviceId),
                    static_cast<uint32_t>(pKernelRecord->contextId),
                    static_cast<uint32_t>(pKernelRecord->streamId),
                    static_cast<uint32_t>(pKernelRecord->graphId),
                    static_cast<uint64_t>(pKernelRecord->graphNodeId),
                    static_cast<uint32_t>(pKernelRecord->channelID));

            break;
        }

#if 0
        case CUPTI_ACTIVITY_KIND_RUNTIME:
        {
            // intentionally don't log this type
            NVLOGI_FMT(TAG, "CUPTI Activity CUPTI_ACTIVITY_KIND_RUNTIME not logging any info intentionally");
            break;
        }
#else
        case CUPTI_ACTIVITY_KIND_DRIVER:
        case CUPTI_ACTIVITY_KIND_RUNTIME:
        case CUPTI_ACTIVITY_KIND_INTERNAL_LAUNCH_API:
        if (1) {
            CUpti_ActivityAPI *pApiRecord = (CUpti_ActivityAPI *)pRecord;
            const char* pName = NULL;
            const char* activity = NULL;

            if (pApiRecord->kind == CUPTI_ACTIVITY_KIND_DRIVER)
            {
                cuptiGetCallbackName(CUPTI_CB_DOMAIN_DRIVER_API, pApiRecord->cbid, &pName);
                activity = "CUPTI_ACTIVITY_KIND_DRIVER";
            }
            else if (pApiRecord->kind == CUPTI_ACTIVITY_KIND_RUNTIME)
            {
                cuptiGetCallbackName(CUPTI_CB_DOMAIN_RUNTIME_API, pApiRecord->cbid, &pName);
                activity = "CUPTI_ACTIVITY_KIND_RUNTIME";
            }
            else
            {
                activity = "CUPTI_ACTIVITY_KIND_INTERNAL_LAUNCH_API";
            }

            NVLOGI_FMT(TAG, "{} [ {}, {} ] duration {}, \"{}\", cbid {}, processId {}, threadId {}, correlationId {}",
                    activity,
                    (unsigned long long)pApiRecord->start,
                    (unsigned long long)pApiRecord->end,
                    (unsigned long long)(pApiRecord->end - pApiRecord->start),
                    GetName(pName),
                    static_cast<uint32_t>(pApiRecord->cbid),
                    static_cast<uint32_t>(pApiRecord->processId),
                    static_cast<uint32_t>(pApiRecord->threadId),
                    static_cast<uint32_t>(pApiRecord->correlationId));

        }
        break;
#endif

        case CUPTI_ACTIVITY_KIND_MEMORY:
        {
            CUpti_ActivityMemory *pMemoryRecord = (CUpti_ActivityMemory *)(void *)pRecord;

            NVLOGI_FMT(TAG, "MEMORY [ {}, {} ] duration {}, size {} bytes, address {}, memoryKind {}, deviceId {}, contextId {}, processId {}",
                    (unsigned long long)pMemoryRecord->start,
                    (unsigned long long)pMemoryRecord->end,
                    (unsigned long long)(pMemoryRecord->end - pMemoryRecord->start),
                    (unsigned long long)pMemoryRecord->bytes,
                    (unsigned long long)pMemoryRecord->address,
                    GetMemoryKindString(pMemoryRecord->memoryKind),
                    static_cast<uint32_t>(pMemoryRecord->deviceId),
                    static_cast<uint32_t>(pMemoryRecord->contextId),
                    static_cast<uint32_t>(pMemoryRecord->processId));

            break;
        }

        case CUPTI_ACTIVITY_KIND_GRAPH_TRACE:
        {
            CUpti_ActivityGraphTrace2 *pMemoryRecord = (CUpti_ActivityGraphTrace2 *)(void *)pRecord;

            NVLOGI_FMT(TAG, "GRAPH_TRACE [ {}, {} ] duration {}, graphId {}, streamId {}, deviceId {} {}, contextId {} {}",
                    (unsigned long long)pMemoryRecord->start,
                    (unsigned long long)pMemoryRecord->end,
                    (unsigned long long)(pMemoryRecord->end - pMemoryRecord->start),
                    static_cast<uint32_t>(pMemoryRecord->graphId),
                    static_cast<uint32_t>(pMemoryRecord->streamId),
                    static_cast<uint32_t>(pMemoryRecord->deviceId),
                    static_cast<uint32_t>(pMemoryRecord->endDeviceId),
                    static_cast<uint32_t>(pMemoryRecord->contextId),
                    static_cast<uint32_t>(pMemoryRecord->endContextId));

            break;
        }

        default:
            NVLOGW_FMT(TAG, "CUPTI Activity {} printing not implemented",activityKind);
            break;
    }
}

static std::mutex g_cupti_helper_mutex;
static uint8_t *pBufferEmpty[MAX_BUFFERS] {nullptr};
static uint8_t *pBufferReady[MAX_BUFFERS] {nullptr};
static size_t bufferReadyValidSize[MAX_BUFFERS] {0};
static std::thread* cupti_polling_thread;
static bool cupti_polling_thread_done = false;

// Buffer Management Functions
static void CUPTIAPI
BufferRequested(
    uint8_t **ppBuffer,
    size_t *pSize,
    size_t *pMaxNumRecords)
{
    NVLOGI_FMT(TAG,"BufferRequested() entered");
    const std::lock_guard<std::mutex> lock(g_cupti_helper_mutex);

    // This callback will be called from whatever thread made the CUDA API call


    // use previously allocated buffer to speedup high priority thread
    uint8_t *pBuffer = nullptr;
    int buf_idx = -1;
    for (int k=0; k<MAX_BUFFERS; k++)
    {
        if (pBufferEmpty[k] != nullptr)
        {
            pBuffer = pBufferEmpty[k];
            pBufferEmpty[k] = nullptr;
            buf_idx = k;
            break; // out of for loop
        }
    }
    if (pBuffer == nullptr)
    {
        NVLOGF_FMT(TAG,AERIAL_MEMORY_EVENT,"Unable to find pre-allocated cupti buffer");
    }

    *pSize = BUF_SIZE;
    *ppBuffer = pBuffer;
    *pMaxNumRecords = 0;
    NVLOGI_FMT(TAG,"BufferRequested: pBuffer={} [{}]",reinterpret_cast<void*>(pBuffer),buf_idx);
}


void PrintActivityBuffer(uint8_t *pBuffer, size_t validBytes)
{
    CUpti_Activity *pRecord = NULL;
    CUptiResult status = CUPTI_SUCCESS;

    do {
        status = cuptiActivityGetNextRecord(pBuffer, validBytes, &pRecord);
        if (status == CUPTI_SUCCESS)
        {
            PrintActivity(pRecord);
        }
        else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED)
        {
            break;
        }
        else
        {
            CUPTI_API_CALL(status);
        }
    } while (1);
}


static void CUPTIAPI
BufferCompleted(
    CUcontext context,
    uint32_t streamId,
    uint8_t *pBuffer,
    size_t size,
    size_t validSize)
{
    static int first_time = 1;
    if (first_time)
    {
        pthread_t my_pthread = pthread_self();
        pthread_setname_np(my_pthread,"cupti_cb");
        nvlog_fmtlog_thread_init("cupti_cb");
        first_time = 0;
    }

    NVLOGI_FMT(TAG, "BufferCompleted: pBuffer={} size={} validSize={}",reinterpret_cast<void*>(pBuffer),size,validSize);

    // Place buffer on ready list
    bool bufferMoved = false;
    if (1)
    {
        const std::lock_guard<std::mutex> lock(g_cupti_helper_mutex);
        for (int k=0; k<MAX_BUFFERS; k++)
        {
            if (pBufferReady[k] == nullptr)
            {
                pBufferReady[k] = pBuffer;
                bufferReadyValidSize[k] = validSize;
                bufferMoved = true;
                break; //out of for loop
            }
        }
    }

    if (bufferMoved)
    {
        NVLOGI_FMT(TAG, "BufferCompleted return");
    }
    else
    {
        NVLOGF_FMT(TAG,AERIAL_MEMORY_EVENT,"Unable to allocate replacement cupti buffer");
    }
}

static CUpti_SubscriberHandle subscriber;
static void CuptiCallbackHandler(void* pUserData, CUpti_CallbackDomain domain, CUpti_CallbackId callbackId, const void *pCallbackData)
{
    NVLOGI_FMT(TAG,"CuptiCallbackHandler: pUserData={} domain={} callbackId={}",pUserData,domain,callbackId);
}

void printReadyBuffers()
{
    for (int k=0; k<MAX_BUFFERS; k++)
    {
        uint8_t *pBuffer = nullptr;
        size_t validSize {0};
        if (pBufferReady[k] != nullptr)
        {
            const std::lock_guard<std::mutex> lock(g_cupti_helper_mutex);
            pBuffer = pBufferReady[k];
            validSize = bufferReadyValidSize[k];
            bufferReadyValidSize[k] = 0;
            pBufferReady[k] = nullptr;
        }
        if (validSize > 0)
        {
            NVLOGI_FMT(TAG,"PrintActivityBuffer: pBuffer={} [{}]",reinterpret_cast<void*>(pBuffer),k);
            PrintActivityBuffer(pBuffer, validSize);
        }
        if (pBuffer)
        {
            int buf_idx = -1;
            memset(pBuffer,0,BUF_SIZE);
            const std::lock_guard<std::mutex> lock(g_cupti_helper_mutex);

            for (int n=0; n<MAX_BUFFERS; n++)
            {
                if (pBufferEmpty[n] == nullptr)
                {
                    pBufferEmpty[n] = pBuffer;
                    buf_idx = n;
                    break; // out of for loop
                }
            }
            if (buf_idx == -1)
            {
                NVLOGF_FMT(TAG,AERIAL_MEMORY_EVENT,"Unable to place activity buffer on empty list");
            }
        }
    }
}


#if defined(__cplusplus)
extern "C" {
#endif /* defined(__cplusplus) */

void cupti_stats_polling_worker(void* unused)
{
    nvlog_fmtlog_thread_init("cupti_stats");
    while (cupti_polling_thread_done == false)
    {
        printReadyBuffers();
    }
    printReadyBuffers();
}

void launch_cupti_stats_polling_worker(int32_t cpu_core)
{
    cupti_polling_thread = new std::thread(cupti_stats_polling_worker, nullptr);
    if(cpu_core >= 0)
    {
        NVLOGI_FMT(TAG, "Initializing cupti stats polling thread on core {}", cpu_core);
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(cpu_core, &cpuset);

        auto ret = pthread_setaffinity_np(cupti_polling_thread->native_handle(), sizeof(cpuset), &cpuset);
        if(ret)
        {
            NVLOGF_FMT(TAG, AERIAL_THREAD_API_EVENT, "Failed to set affinity for cupti stats polling thread: ret={}",ret);
        }
    }

    auto ret = pthread_setname_np(cupti_polling_thread->native_handle(), "cupti_stats");
    if(ret != 0)
    {
        NVLOGF_FMT(TAG, AERIAL_THREAD_API_EVENT, "Failed to set cupti_stats_polling_worker thread name: ret={}",ret);
    }
}

void cuphy_cupti_helper_init()
{
    const std::lock_guard<std::mutex> lock(g_cupti_helper_mutex);

    // allocate initial buffers (in the low priority thread)
    for (int k=0; k<MAX_BUFFERS; k++)
    {
        uint8_t *pBuffer;
        CHECK_CUDA(cudaHostAlloc(reinterpret_cast<void**>(&pBuffer), BUF_SIZE+ALIGN_SIZE, cudaHostAllocPortable));
        if (pBuffer == nullptr)
        {
            NVLOGF_FMT(TAG,AERIAL_MEMORY_EVENT,"Unable to allocate initial cupti buffer");
        }
        pBufferEmpty[k] = ALIGN_BUFFER(pBuffer, ALIGN_SIZE);
        memset(pBufferEmpty[k],0,BUF_SIZE);
    }

    if (1)
    {
        size_t valueSize = sizeof(uint8_t);
        uint8_t value = 1;
        CUPTI_API_CALL(cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_ZEROED_OUT_ACTIVITY_BUFFER, &valueSize, (void *)&value));
    }

    if (1)
    {
        size_t valueSize = sizeof(size_t);
        size_t value = 0;

        value=10;
        CUPTI_API_CALL(cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_PRE_ALLOCATE_VALUE, &valueSize, (void *)&value));

        value=16*1024*1024;
        CUPTI_API_CALL(cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE, &valueSize, (void *)&value));

        CUPTI_API_CALL(cuptiActivityGetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE, &valueSize, (void *)&value));
        printf("CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE: %ld\n",value);

        CUPTI_API_CALL(cuptiActivityGetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT, &valueSize, (void *)&value));
        printf("CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT: %ld\n",value);

        CUPTI_API_CALL(cuptiActivityGetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_PRE_ALLOCATE_VALUE, &valueSize, (void *)&value));
        printf("CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_PRE_ALLOCATE_VALUE: %ld\n",value);
    }

    CUPTI_API_CALL(cuptiSubscribe(&subscriber, CuptiCallbackHandler, NULL));
    CUPTI_API_CALL(cuptiActivityRegisterCallbacks(BufferRequested, BufferCompleted));
    CUPTI_API_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION));
    CUPTI_API_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
    CUPTI_API_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
    CUPTI_API_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER));
    // CUPTI_API_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_GRAPH_TRACE));

    //CUPTI_API_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMORY));
    //CUPTI_API_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_ENVIRONMENT));
    //CUPTI_API_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONTEXT));
    //CUPTI_API_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));

    //CUPTI_API_CALL(cuptiActivityFlushPeriod(10)); // time in ms

    //Disable specific activity inherent in Aerial that overwhelms the output
    {
        //There are many Aerial calls to cudaEventQuery
        CUPTI_API_CALL(cuptiActivityEnableRuntimeApi(CUPTI_RUNTIME_TRACE_CBID_cudaEventQuery_v3020, 0));
        CUPTI_API_CALL(cuptiActivityEnableDriverApi(CUPTI_DRIVER_TRACE_CBID_cuEventQuery, 0));

        //Enable only specific runtime API activity
        for (int k=1; k<CUPTI_RUNTIME_TRACE_CBID_SIZE; k++)
        {
            CUPTI_API_CALL(cuptiActivityEnableRuntimeApi(k, 0));
        }
        CUPTI_API_CALL(cuptiActivityEnableRuntimeApi(CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000, 1));

        //Enable only specific driver API activity
        for (int k=1; k<CUPTI_DRIVER_TRACE_CBID_SIZE; k++)
        {
            CUPTI_API_CALL(cuptiActivityEnableDriverApi(k, 0));
        }
        CUPTI_API_CALL(cuptiActivityEnableDriverApi(CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel, 1));
        CUPTI_API_CALL(cuptiActivityEnableDriverApi(CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch, 1));
    }


    NVLOGW_FMT(TAG,"Intialized cuPHY CUPTI");
    launch_cupti_stats_polling_worker(-1); 
}

void cuphy_cupti_helper_flush()
{
    CUPTI_API_CALL(cuptiGetLastError());
    CUPTI_API_CALL(cuptiActivityFlushAll(1));
}

void cuphy_cupti_helper_stop()
{
    cuphy_cupti_helper_flush();
    cupti_polling_thread_done = true;
    int ret = pthread_join(cupti_polling_thread->native_handle(), NULL);
}

void cuphy_cupti_helper_push_external_id(uint64_t id)
{
    CUPTI_API_CALL(cuptiActivityPushExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_AERIAL, id));
}

void cuphy_cupti_helper_pop_external_id()
{
    uint64_t id;
    CUPTI_API_CALL(cuptiActivityPopExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_AERIAL, &id));
}

#if defined(__cplusplus)
} /* extern "C" */
#endif /* defined(__cplusplus) */