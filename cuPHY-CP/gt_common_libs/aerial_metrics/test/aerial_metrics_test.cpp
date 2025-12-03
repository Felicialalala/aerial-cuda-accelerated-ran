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

#include "aerial_metrics.hpp"
#include <chrono>
#include <iostream>
#include <chrono>
#include <thread>
using namespace std;
using namespace std::chrono;
using namespace aerial_metrics;

int main(int argc, const char* argv[])
{
    MetricGaugePtr m_test_gauge;
    MetricCounterPtr m_test_counter;
    m_test_gauge = AerialMetricsRegistrationManager::getInstance().addSimpleGauge("aerial_test_gauge", "Aerial test gauge");
    m_test_counter = AerialMetricsRegistrationManager::getInstance().addSimpleCounter("aerial_test_counter", "Aerial test counter");


    prometheus::Family<prometheus::Histogram>* histogram_family = AerialMetricsRegistrationManager::getInstance()
			.createHistogramFamily("aerial_test_histogram", "Aerial test histogram");
    auto& histogram = histogram_family->Add({{"name", "test_histogram_1"}}, prometheus::Histogram::BucketBoundaries{0, 1, 2});

    prometheus::Family<prometheus::Summary>* summary_family = AerialMetricsRegistrationManager::getInstance()
			.createSummaryFamily("aerial_test_summary", "Aerial test summary");
    auto& summary = summary_family->Add({{"name", "test_summary_1"}}, prometheus::Summary::Quantiles{{0.5, 0.05}, {0.9, 0.01}, {0.99, 0.001}});


    auto& counter_family = AerialMetricsRegistrationManager::getInstance().createFamily<prometheus::Counter>("aerial_test_counter_family", "Aerial test counter family");
    auto& counter = counter_family.Add({{"aerial_test_counter_template", "test_counter_1"}});

    auto& counter_2 = AerialMetricsRegistrationManager::getInstance().addMetric<prometheus::Counter>("aerial_test_counter_family", "Aerial test counter family", {{"aerial_test_counter_template_2", "test_counter_2"}});

    auto& gauge_2 = AerialMetricsRegistrationManager::getInstance().addMetric<prometheus::Gauge>("aerial_test_gauge", "Aerial test gauge", {{"aerial_test_gauge_template_2", "test_gauge_2"}});

    auto& summary_2 = AerialMetricsRegistrationManager::getInstance().addMetric<prometheus::Summary>("aerial_test_summary", "Aerial test summary", {{"name", "test_summary_2"}}, prometheus::Summary::Quantiles{{0.5, 0.05}, {0.9, 0.01}, {0.99, 0.001}});

    auto& histogram_2 = AerialMetricsRegistrationManager::getInstance().addMetric<prometheus::Histogram>("aerial_test_histogram_2", "Aerial test histogram_2", {{"name", "test_histogram_2"}}, prometheus::Histogram::BucketBoundaries{0, 1, 2});

    int debug_count = 0;
    chrono::high_resolution_clock::time_point prev_tp;

    while(1)
    {
        auto   cur_tp = chrono::high_resolution_clock::now();
        double delay  = chrono::duration_cast<chrono::microseconds>(cur_tp - prev_tp).count();
//        std::cout << "Delay is : " << delay << " micro secs" << '\n';
        prev_tp = cur_tp;
        m_test_gauge->Set(delay);
        gauge_2.Set(delay);
        m_test_counter->Increment();

        counter.Increment();
        counter_2.Increment();

        histogram.Observe(0);
        histogram.Observe(0.5);
        histogram.Observe(1);
        histogram.Observe(1.5);
        histogram.Observe(1.5);
        histogram.Observe(2);
        histogram.Observe(3);

        histogram_2.Observe(0);
        histogram_2.Observe(0.5);
        histogram_2.Observe(1);
        histogram_2.Observe(1.5);
        histogram_2.Observe(1.5);
        histogram_2.Observe(2);
        histogram_2.Observe(3);

	summary.Observe(++debug_count%1000000);
	summary_2.Observe(++debug_count%1000000);
	std::this_thread::sleep_for(std::chrono::nanoseconds(500000));
    }
    return 0;
}
