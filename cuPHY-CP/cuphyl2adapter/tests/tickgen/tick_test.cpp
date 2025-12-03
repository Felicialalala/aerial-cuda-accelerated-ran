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

#include "tick_test.hpp"
#include "nvlog.hpp"
#include "aerial_metrics.hpp"

#define TAG (NVLOG_TAG_BASE_L2_ADAPTER + 7) // "L2A.TICK"

using namespace aerial_metrics;

#define dbg
#define SLOT_INDICATION_POLLING

#define ABS(N) (((N) < 0) ? (-(N)) : (N))
#define allowed_offset_nsec 100000
#define test_time_sec (20)
#define pre_window 200

int32_t  debug_count = 0;

MetricGaugePtr m_tick_interval_gauge;
MetricCounterPtr m_tick_counter;

class tti_gen {
    std::thread timer_thread; // timer thread

#ifdef dbg
    //Measure slot indication interval
    chrono::high_resolution_clock::time_point prev_tp;
#endif

    unique_ptr<thread_config> timer_thread_cfg;
    uint32_t                  window_nsec;
    volatile bool             has_thread_cfg = true;
    volatile bool             started        = false;

    //phy_epoll_context                                epoll_ctx;
    //unique_ptr<timer_fd>                             timer_fd_p;
    //unique_ptr<member_event_callback<tti_gen>> timer_mcb_p;
    //std::atomic_int tti_gen_ref;
    nanoseconds current_ts;
    nanoseconds current_scheduled_ts;
    //PHY_module* module_;
    pthread_t thread_id;

    std::atomic_int tti_gen_ref;

public:
    void start_slot_indication();
    void stop_slot_indication();

private:
    void     slot_indication_thread_poll_method();
    uint64_t sys_clock_time_handler();

    void slot_indication_thread_timer_fd_method();
    void slot_indication_handler();
    void send_slot_indication();
};

static std::once_flag start_stop_flag;
uint32_t              get_mu_highest() { return 1; }

void tti_gen::start_slot_indication()
{
    NVLOGC_FMT(TAG, "{}: line {}\n", __func__, __LINE__);
    // tti_gen_ref++;
    //    std::call_once(start_stop_flag, [this] ()
    //    {

    timer_thread_cfg.reset(new thread_config);
    timer_thread_cfg->name           = "test_polling";
    timer_thread_cfg->cpu_affinity   = 19;
    timer_thread_cfg->sched_priority = 95;
#ifdef SLOT_INDICATION_POLLING
    std::thread t(&tti_gen::slot_indication_thread_poll_method, this);
#else
    std::thread t(&tti_gen::slot_indication_thread_timer_fd_method, this);
#endif
    thread_id = t.native_handle();
    timer_thread.swap(t);

    int name_st = pthread_setname_np(timer_thread.native_handle(), "timer_thread");

    if(name_st != 0)
    {
        NVLOGW_FMT(TAG, "Timer Thread pthread_setname_np failed with status: {}\n", std::strerror(name_st));
    }

    sched_param sch;
    int         policy;
    int         status = 0;
    //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
    // Set thread priority
    status = pthread_getschedparam(timer_thread.native_handle(), &policy, &sch);
    if(status != 0)
    {
        NVLOGW_FMT(TAG, "timer_thread pthread_getschedparam failed with status : {}\n", std::strerror(status));
    }
    sch.sched_priority = timer_thread_cfg->sched_priority;

    status = pthread_setschedparam(timer_thread.native_handle(), SCHED_FIFO, &sch);
    if(status != 0)
    {
        NVLOGW_FMT(TAG, "timer_thread setschedparam failed with status : {}\n", std::strerror(status));
    }

    //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
    // Set thread CPU affinity
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(timer_thread_cfg->cpu_affinity, &cpuset);
    status = pthread_setaffinity_np(timer_thread.native_handle(), sizeof(cpu_set_t), &cpuset);
    if(status)
    {
        NVLOGW_FMT(TAG, "timer_thread setaffinity_np  failed with status : {}\n", std::strerror(status));
    }
    //       });
}

void tti_gen::stop_slot_indication()
{
    NVLOGD_FMT(TAG, "{}: line {}\n", __func__, __LINE__);
    tti_gen_ref--;
    if(tti_gen_ref == 0)
    {
        std::call_once(start_stop_flag, [this]() {
            pthread_cancel(thread_id);
        });
    }
}

inline uint64_t tti_gen::sys_clock_time_handler()
{
#if 1
    struct timespec ts;
    //	if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
    if(clock_gettime(CLOCK_REALTIME, &ts) != 0)
    {
        NVLOGE_FMT(TAG, AERIAL_CLOCK_API_EVENT, "Read CLOCK failed");
    }
    uint64_t ns = ts.tv_sec;
    ns          = ns * 1000000000L + ts.tv_nsec;
    return ns;
#else
    using namespace std::chrono;
    current_ts = duration_cast<nanoseconds>(system_clock::now().time_since_epoch());
    // current_ts = duration_cast<nanoseconds>(steady_clock::now().time_since_epoch());
    return current_ts.count();
#endif
}

void tti_gen::slot_indication_thread_poll_method()
{
#ifdef dbg
    if(has_thread_cfg)
    {
        sched_param sch;
        int         policy;
        int         status = 0;
        status             = pthread_getschedparam(pthread_self(), &policy, &sch);
        if(status == 0)
        {
            NVLOGD_FMT(TAG, "slot_indication_thread_poll_method sched_priority {}\n", sch.sched_priority);
            NVLOGD_FMT(TAG, "slot_indication_thread_poll_method on CPU {}\n", sched_getcpu());
        }
        else
        {
            NVLOGD_FMT(TAG, "pthread_getschedparam failed with status: {}\n", std::strerror(status));
        }
    }
#endif

    assign_thread_cpu_core(timer_thread_cfg->cpu_affinity);

    bool     first = true;
    uint64_t last_actual, next_expected, first_time;
    uint64_t runtime;
    uint64_t count          = 0;
    uint64_t sum_abs_offset = 0;
    int32_t  min_offset     = 0;
    int32_t  max_offset     = 0;
    int32_t  max_abs_offset = 0;

    //window_nsec  = mu_to_ns(1);
    window_nsec = mu_to_ns(get_mu_highest());

    while(1)
    {
        uint64_t curr = sys_clock_time_handler();
        if(first)
        {
            first_time  = curr;
            last_actual = curr;
            first       = false;

            // next expected tick - round up curr to next 10.24 sec frame boundary
            next_expected = curr;
            next_expected /= (10000000ULL * 1024ULL);
            next_expected++;
            next_expected *= (1024ULL * 10000000ULL);
            next_expected += (364ULL * 10000000ULL); // adjust to SFN = 0, accounting for GPS vs TIA conversion

            NVLOGI_FMT(TAG, "Start time: tick={}, seconds since epoch = {}, nanoseconds = {}\n",
                   curr,
                   curr / 1000000000ULL,
                   curr % 1000000000ULL);

            NVLOGI_FMT(TAG, "FIRST tick scheduled for: tick={}, seconds since epoch = {}, nanoseconds = {}\n",
                   next_expected,
                   next_expected / 1000000000ULL,
                   next_expected % 1000000000ULL);

            continue;
        }

        if(curr < last_actual)
        {
            NVLOGW_FMT(TAG, "error curr {} last {}\n", curr, last_actual);
        }

        int64_t diff = next_expected - curr;
        if(diff >= pre_window)
        {
            continue;
        }

        current_scheduled_ts = nanoseconds(next_expected);
        slot_indication_handler();

        int32_t offset     = (int32_t)((int64_t)curr - (int64_t)next_expected);
        int32_t abs_offset = ABS(offset);
        // stats
        if(offset < min_offset)
            min_offset = offset;
        if(offset > max_offset)
            max_offset = offset;
        if(abs_offset > max_abs_offset)
            max_abs_offset = abs_offset;
        count += 1;
        sum_abs_offset += (uint64_t)abs_offset;

#ifdef dbg
        NVLOGD_FMT(TAG, "{} {} {} {} {}\n",count , offset , abs_offset, sum_abs_offset, sum_abs_offset / count);
#endif
        runtime = curr - first_time;

        if(abs_offset > allowed_offset_nsec)
        {
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "jitter error, offset {}\n", offset);
        }

#ifdef dbg
        NVLOGI_FMT(TAG, "curr: {}, diff: {}, offset: {}\n", curr , diff, offset);
#endif
        next_expected += window_nsec;
        last_actual = curr;

    }

    // print stats
    NVLOGD_FMT(TAG, "total run time: {} sec\n", runtime / (1000000000));
    NVLOGD_FMT(TAG, "event count:    {}\n", count);
    NVLOGD_FMT(TAG, "sum:            {}\n", sum_abs_offset);
    NVLOGD_FMT(TAG, "min offset:     {}\n", min_offset);
    NVLOGD_FMT(TAG, "max offset:     {}\n", max_offset);
    NVLOGD_FMT(TAG, "max abs offset: {}\n", max_abs_offset);
    NVLOGD_FMT(TAG, "avg abs offset: {}\n", sum_abs_offset / count);
}

void tti_gen::slot_indication_handler()
{
//#ifdef dbg
    auto   cur_tp = chrono::high_resolution_clock::now();
    double delay  = chrono::duration_cast<chrono::nanoseconds>(cur_tp - prev_tp).count();
    prev_tp = cur_tp;
    if(delay > 600000) {
       std::cout << "slot indication interval is : " << delay << " micro secs" << std::endl;
    }

    if(++debug_count == 2000) {
        debug_count = 0;
	delay = 750000000;
        m_tick_interval_gauge->Set(delay);
       std::cout << "slot indication interval is : " << delay << " micro secs" << std::endl;
    } else {
        m_tick_interval_gauge->Set(delay);
    }
    m_tick_counter->Increment();
    send_slot_indication();
}

void tti_gen::send_slot_indication()
{
}



int main(int argc, const char* argv[])
{
    int return_value = 0;

    NVLOGC_FMT(TAG, "main open\n");

    try
    {
        m_tick_interval_gauge = AerialMetricsRegistrationManager::getInstance().addSimpleGauge("tick_interval", "Duration of ticker interval");
        m_tick_counter = AerialMetricsRegistrationManager::getInstance().addSimpleCounter("tick_counter", "Number of ticks");
    }
    catch (const std::system_error& f) {
        std::cerr << f.what() << '\n';
        return -1;
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return -1;
    }

    // Relative path of this process is $cuBB_SDK/build/cuPHY-CP/cuphyl2adapter/tests/tickgen/
    char        yaml_file[1024];
    std::string relative_path = std::string("../../../../../").append(NVLOG_DEFAULT_CONFIG_FILE);

    try {
        nv_get_absolute_path(yaml_file, relative_path.c_str());
        nvlog_fmtlog_init(yaml_file, "tick_test",NULL);
        nvlog_fmtlog_thread_init();
    } catch(std::exception& e) {
        return -1;
    }

    NVLOGC_FMT(TAG, "nvlog started\n");

    tti_gen ttigen;
    ttigen.start_slot_indication();

    while(1)
    {
        usleep(1000 * 1000);
        NVLOGI_FMT(TAG, "WORKING ...\n");
    }
}
