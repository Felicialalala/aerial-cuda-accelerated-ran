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

#include "testbench_common.hpp"

testWorker::testWorker(std::string const& name, uint32_t workerId, int cpuId, int gpuId, int cpuThrdSchdPolicy, int cpuThrdPrio, uint32_t mpsSubctxSmCount, std::shared_ptr<testWrkrCmdQ>& cmdQ, std::shared_ptr<testWrkrRspQ>& rspQ, int uldlMode, uint32_t debugMessageLevel, bool useGreenContexts):
    m_name(name),
    m_wrkrId(workerId),
    m_cpuId(cpuId),
    m_gpuId(gpuId),
    m_schdPolicy(cpuThrdSchdPolicy),
    m_prio(cpuThrdPrio),
    m_mpsSubctxSmCount(mpsSubctxSmCount),
    m_shPtrCmdQ(cmdQ),
    m_shPtrRspQ(rspQ),
    m_uldlMode(uldlMode),
    m_dbgMsgLevel(debugMessageLevel),
    m_thrdId(0),
    m_longPattern(0),
    m_useGreenContexts(useGreenContexts),
    m_greenCtxId(0) {}

testWorker::~testWorker()
{
    // Send termination message to worker thread
    auto shPtrPayload = std::make_shared<commnTestExitMsgPayload>();
    shPtrPayload->rsp = true;  

    auto shPtrMsg = std::make_shared<testWrkrCmdMsg>(COMMON_TEST_WRKR_CMD_MSG_EXIT, m_wrkrId, shPtrPayload);
    DEBUG_TRACE("MainThread [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: Sending message: %s\n", getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId(), TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);

    m_shPtrCmdQ->send(shPtrMsg);

    std::shared_ptr<testWrkrRspMsg> shPtrRsp;
    m_shPtrRspQ->receive(shPtrRsp, m_wrkrId);

    if(m_thrd.joinable())
    {
        try
        {
            m_thrd.join();
            DEBUG_TRACE("MainThread [tid %s]: Joining worker thread\n", getThreadIdStr().c_str());
        }
        catch(const std::exception& e)
        {
            NVLOGE_FMT(NVLOG_TESTBENCH, AERIAL_TESTBENCH_EVENT,  "EXCEPTION: {} while joining worker thread id {}", e.what(), m_wrkrId);
        }
    }
    DEBUG_TRACE("MainThread [tid %s]: Destructor completed\n", getThreadIdStr().c_str());
}

void testWorker::createCuCtx()
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: createCuCtx pre context creation (mpsSubctxSmCount %u)\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId(), m_mpsSubctxSmCount);

#if CUDART_VERSION < 11040 // min CUDA version for MPS programmatic API
    printf("MPS programmatic API support requires CUDA 11.4 or higher\n");
    exit(EXIT_FAILURE);
#endif

    static std::mutex ctxPartitionCfgMutex;
    {
        std::lock_guard<std::mutex> ctxPartitionCfgMutexLock(ctxPartitionCfgMutex);
        printf("Make request m_wrkrId %d smCountRequested %d\n", m_wrkrId, m_mpsSubctxSmCount);
        m_cuCtx.create(m_gpuId, m_mpsSubctxSmCount, &m_smCount);
        printf("m_wrkrId %d smCountRequested %d smCountApplied %d\n", m_wrkrId, m_mpsSubctxSmCount, m_smCount);
        m_cuCtx.bind();
    }
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: createCuCtx post context creation\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    // CUDA_CHECK(cudaSetDevice(m_gpuId));

    int maxThreadsPerBlock = 0;
    CU_CHECK(cuDeviceGetAttribute(&maxThreadsPerBlock, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, m_gpuId));

    int maxThreadsPerMultiProcessor = 0;
    CU_CHECK(cuDeviceGetAttribute(&maxThreadsPerMultiProcessor, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, m_gpuId));

    int maxThreadBlocksPerMultiProcessor = maxThreadsPerMultiProcessor / maxThreadsPerBlock;

    // Now allocate resources
    // printf("smIdBufSize = %d\n", m_smCount*maxThreadBlocksPerMultiProcessor);

    m_uqPtrWrkrCuStrm           = std::make_unique<cuphy::stream>(cudaStreamNonBlocking);
    m_shPtrStopEvent            = std::make_shared<cuphy::event>(cudaEventDisableTiming);
    m_shPtrRdSmIdWaitEvent      = std::make_shared<cuphy::event>(cudaEventDisableTiming);

    //  1 SM id per thread block
    m_nSmIds   = m_smCount * maxThreadBlocksPerMultiProcessor;
    m_smIdsGpu = std::move(cuphy::buffer<uint32_t, cuphy::device_alloc>(m_nSmIds));
    m_smIdsCpu = std::move(cuphy::buffer<uint32_t, cuphy::pinned_alloc>(m_nSmIds));

    // print setup:
    printf("%s worker %d :", m_name.c_str(), m_wrkrId);
    printf("\n--> assigned %d SMs", m_smCount);
}

void testWorker::createCuGreenCtx(const cuphy::cudaGreenContext& my_green_context)
{
#if CUDA_VERSION >= 12040
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: createCuCtx pre context creation (mpsSubctxSmCount %u)\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId(), m_mpsSubctxSmCount);
    // The reason we just bind the context and not create one is that depending on the channel this thread is running,
    // we need to know if we'll do a split of the original GPU resources or a resplit. If it's a resplit then we'd need to have access to the green context/resouce
    // So for now we create all green contexts from the main process before these threads have been spawned.
    my_green_context.bind();
    m_smCount = my_green_context.getSmCount();
    m_greenCtxId = reinterpret_cast<uint64_t>(my_green_context.handle()); //
    //printf("%s createCuGreenCtx %d has SM count %d with ctx %p and primary handle %p\n", m_name.c_str(), (int) getCuCtxId(), m_smCount, my_green_context.handle(), my_green_context.primary_handle());
    NVLOGC_FMT(NVLOG_TESTBENCH, "{} createCuGreenCtx has SM count {}", m_name.c_str(), m_smCount); // could also change NVLOG tag based on caller

    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: createCuCtx post context creation\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    // CUDA_CHECK(cudaSetDevice(m_gpuId));

    int maxThreadsPerBlock = 0;
    CU_CHECK(cuDeviceGetAttribute(&maxThreadsPerBlock, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, m_gpuId));

    int maxThreadsPerMultiProcessor = 0;
    CU_CHECK(cuDeviceGetAttribute(&maxThreadsPerMultiProcessor, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, m_gpuId));

    int maxThreadBlocksPerMultiProcessor = maxThreadsPerMultiProcessor / maxThreadsPerBlock;

    // Now allocate resources
    // printf("smIdBufSize = %d\n", m_smCount*maxThreadBlocksPerMultiProcessor);

    m_uqPtrWrkrCuStrm           = std::make_unique<cuphy::stream>(cudaStreamNonBlocking);
    m_shPtrStopEvent            = std::make_shared<cuphy::event>(cudaEventDisableTiming);
    m_shPtrRdSmIdWaitEvent      = std::make_shared<cuphy::event>(cudaEventDisableTiming);

    //  1 SM id per thread block

    m_nSmIds   = m_smCount * maxThreadBlocksPerMultiProcessor;
    m_smIdsGpu = std::move(cuphy::buffer<uint32_t, cuphy::device_alloc>(m_nSmIds));
    m_smIdsCpu = std::move(cuphy::buffer<uint32_t, cuphy::pinned_alloc>(m_nSmIds));

    // print setup:
    printf("%s worker %d :", m_name.c_str(), m_wrkrId);
    printf("\n--> assigned %d SMs", m_smCount);
#endif
}

void testWorker::setThrdProps()
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: setThrdProps\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    //------------------------------------------------------------------
    // Bump up CPU thread prio
    pid_t       pid = (pid_t)syscall(SYS_gettid);
    sched_param schdPrm;
    schdPrm.sched_priority = m_prio;

    // pid_t pid = (pid_t) syscall(SYS_gettid);
    /*
    int schdSetRet = sched_setscheduler(pid, m_schdPolicy, &schdPrm);

    if(0 == schdSetRet)
    {
        DEBUG_TRACE("%s id %d [tid %s]: pid %d policy %d prio %d\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), pid, m_schdPolicy, m_prio);
    }
    else
    {
        DEBUG_TRACE("%s id %d [tid %s]: Failed to set scheduling algo pid %d, prio %d, return code %d: err %s\n",
                    m_name.c_str(),
                    m_wrkrId,
                    getThreadIdStr().c_str(),
                    pid,
                    m_prio,
                    schdSetRet,
                    strerror(errno));
    }
    */

    //------------------------------------------------------------------
    // Set thread affinity to specified CPU
    cpu_set_t cpuSet;
    CPU_ZERO(&cpuSet);
    CPU_SET(m_cpuId, &cpuSet);
    // DEBUG_TRACE("%s Pipeline[%d]: setting affinity of pipeline %d (pid %d) to CPU Id %d\n", m_name.c_str(), instIdx, pid, cpuId);

    int affinitySetRet = sched_setaffinity(pid, sizeof(cpuSet), &cpuSet);

    if(0 == affinitySetRet)
    {
        DEBUG_TRACE("%s id %d [tid %s]: pid %d set affinity to CPU %d\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), pid, m_cpuId);
    }
    else
    {
        DEBUG_TRACE("%s id %d [tid %s]: failed to set affinity pid %d to CPU %d, return code %d err %s\n",
                    m_name.c_str(),
                    m_wrkrId,
                    getThreadIdStr().c_str(),
                    pid,
                    m_cpuId,
                    affinitySetRet,
                    strerror(errno));
    }
}

void testWorker::print(bool cbErrors, bool isPschTxRx, bool waitRsp)
{
    // pack message
    auto shPtrPayload        = std::make_shared<commnTestPrintMsgPayload>();
    shPtrPayload->rsp        = waitRsp;
    shPtrPayload->cbErrors   = cbErrors;
    shPtrPayload->isPschTxRx = isPschTxRx;

    // send message
    auto shPtrMsg = std::make_shared<testWrkrCmdMsg>(COMMON_TEST_WRKR_CMD_MSG_PRINT, m_wrkrId, shPtrPayload);
    DEBUG_TRACE("MainThread [tid %s][currCtxId 0x%0lx]: Sending message: %s\n", getThreadIdStr().c_str(), getCurrCuCtxId(), TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);

    m_shPtrCmdQ->send(shPtrMsg);

    // wait for response
    if(waitRsp)
    {
        std::shared_ptr<testWrkrRspMsg> shPtrRsp;
        m_shPtrRspQ->receive(shPtrRsp, COMMON_TEST_WRKR_RSP_MSG_PRINT, m_wrkrId);
    }
}

void testWorker::eval(bool cbErrors, bool isPschTxRx, bool waitRsp)
{
    // pack message
    auto shPtrPayload        = std::make_shared<commnTestEvalMsgPayload>();
    shPtrPayload->rsp        = waitRsp;
    shPtrPayload->cbErrors   = cbErrors;
    shPtrPayload->isPschTxRx = isPschTxRx;

    // send message to worker
    auto shPtrMsg = std::make_shared<testWrkrCmdMsg>(COMMON_TEST_WRKR_CMD_MSG_EVAL, m_wrkrId, shPtrPayload);
    DEBUG_TRACE("MainThread [tid %s][currCtxId 0x%0lx]: Sending message: %s\n", getThreadIdStr().c_str(), getCurrCuCtxId(), TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);
    m_shPtrCmdQ->send(shPtrMsg);

    // wait for response
    if(waitRsp)
    {
        std::shared_ptr<testWrkrRspMsg> shPtrRsp;
        m_shPtrRspQ->receive(shPtrRsp, COMMON_TEST_WRKR_RSP_MSG_EVAL, m_wrkrId);
    }
}

void testWorker::resetEvalBuffers(bool cbErrors, bool waitRsp)
{
    // pack message
    auto shPtrPayload      = std::make_shared<commnTestResetEvalMsgPayload>();
    shPtrPayload->rsp      = waitRsp;
    shPtrPayload->cbErrors = cbErrors;

    // send message
    auto shPtrMsg = std::make_shared<testWrkrCmdMsg>(COMMON_TEST_WRKR_CMD_MSG_RESET_EVAL, m_wrkrId, shPtrPayload);
    DEBUG_TRACE("MainThread [tid %s][currCtxId 0x%0lx]: Sending message: %s\n", getThreadIdStr().c_str(), getCurrCuCtxId(), TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);

    m_shPtrCmdQ->send(shPtrMsg);

    // wait for response
    if(waitRsp)
    {
        std::shared_ptr<testWrkrRspMsg> shPtrRsp;
        m_shPtrRspQ->receive(shPtrRsp, COMMON_TEST_WRKR_RSP_MSG_RESET_EVAL, m_wrkrId);
    }
}

void testWorker::setWaitVal(uint32_t syncFlagVal, bool waitRsp)
{
    auto shPtrPayload         = std::make_shared<commnTestSetWaitValCmdMsgPayload>();
    shPtrPayload->rsp         = waitRsp;
    shPtrPayload->syncFlagVal = syncFlagVal;

    auto shPtrMsg = std::make_shared<testWrkrCmdMsg>(COMMON_TEST_WRKR_CMD_MSG_SET_WAIT_VAL, m_wrkrId, shPtrPayload);
    DEBUG_TRACE("MainThread [tid %s][currCtxId 0x%0lx]: Sending message: %s\n", getThreadIdStr().c_str(), getCurrCuCtxId(), TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);

    m_shPtrCmdQ->send(shPtrMsg);

    if(waitRsp)
    {
        std::shared_ptr<testWrkrRspMsg> shPtrRsp;
        m_shPtrRspQ->receive(shPtrRsp, COMMON_TEST_WRKR_RSP_MSG_SET_WAIT_VAL, m_wrkrId);
    }
}

void testWorker::readSmIds()
{
    get_sm_ids(m_gpuId, m_smIdsGpu.addr(), m_smCount, m_uqPtrWrkrCuStrm->handle());

    CUDA_CHECK(cudaEventRecord(m_shPtrRdSmIdWaitEvent->handle(), m_uqPtrWrkrCuStrm->handle()));

#if 0
    m_uqPtrWrkrCuStrm->synchronize();
    m_smIds.resize(smCount);
    uint32_t* pSmIds = m_smIdsCpu.addr();
    std::copy(pSmIds, pSmIds + smCount, m_smIds.begin());
    for(int i = 0; i < smCount; ++i)
    {
        DEBUG_TRACE("Worker[%d]: SM id %d\n", m_wrkrId, pSmIds[i]);
    }
#endif
}

void testWorker::readSmIds(std::shared_ptr<cuphy::event>& shPtrRdSmIdWaitEvent, bool waitRsp)
{
    auto shPtrPayload = std::make_shared<commnTestReadSmIdsCmdMsgPayload>();
    shPtrPayload->rsp = waitRsp;

    auto shPtrMsg = std::make_shared<testWrkrCmdMsg>(COMMON_TEST_WRKR_CMD_MSG_READ_SM_IDS, m_wrkrId, shPtrPayload);
    DEBUG_TRACE("MainThread [tid %s][currCtxId 0x%0lx]: Sending message: %s\n", getThreadIdStr().c_str(), getCurrCuCtxId(), TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);

    m_shPtrCmdQ->send(shPtrMsg);

    if(waitRsp)
    {
        std::shared_ptr<testWrkrRspMsg> shPtrRsp;
        m_shPtrRspQ->receive(shPtrRsp, COMMON_TEST_WRKR_RSP_MSG_READ_SM_IDS, m_wrkrId);

        commnTestReadSmIdsRspMsgPayload& readSmIdsRspMsgPayload = *std::static_pointer_cast<commnTestReadSmIdsRspMsgPayload>(shPtrRsp->payload);
        shPtrRdSmIdWaitEvent                                    = readSmIdsRspMsgPayload.shPtrWaitEvent;
    }
}
