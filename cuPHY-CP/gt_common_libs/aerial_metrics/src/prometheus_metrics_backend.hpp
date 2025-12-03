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

#if !defined(PROMETHEUS_HPP_INCLUDED_)
#define PROMETHEUS_HPP_INCLUDED_

#include <memory>
#include <vector>
#include <array>
#include <string>
#include <tuple>

#include "prometheus/detail/core_export.h"
#include "prometheus/client_metric.h"
#include "prometheus/counter.h"
#include "prometheus/histogram.h"
#include "prometheus/summary.h"
#include "prometheus/exposer.h"
#include "prometheus/family.h"
#include "prometheus/registry.h"

#define ep_addr "0.0.0.0:0"
////////////////////////////////////////////////////////////////////////
// aerial_metrics
namespace aerial_metrics
{

class PrometheusMetricGauge
{
public:
    PrometheusMetricGauge() = delete;
    //------------------------------------------------------------------
    // Constructor
    PrometheusMetricGauge(const std::string &name, const std::string &help_str,
            std::shared_ptr<prometheus::Registry> registry) :
            m_family(prometheus::BuildGauge()
                            .Name(name)
                            .Help(help_str)
                            .Register(*registry)),
            m_gauge(m_family.Add({}))
    {
    }
    //------------------------------------------------------------------
    // Destructor
    ~PrometheusMetricGauge() {}
    //------------------------------------------------------------------
    // Increment()
    void Increment(double number) { m_gauge.Increment(number); }
    //------------------------------------------------------------------
    // Decrement()
    void Decrement(double number) { m_gauge.Decrement(number); }
    //------------------------------------------------------------------
    // Set()
    void Set(double number) { m_gauge.Set(number); }
    //------------------------------------------------------------------
    // Get()
    double Get() const { return m_gauge.Value(); }

private:
    //------------------------------------------------------------------
    // Data
    prometheus::Family<prometheus::Gauge> &m_family;
    prometheus::Gauge &m_gauge;
};

class PrometheusMetricCounter
{
public:
	PrometheusMetricCounter() = delete;

	PrometheusMetricCounter(const std::string &name, const std::string &help_str,
			std::shared_ptr<prometheus::Registry> registry) :
			m_family(prometheus::BuildCounter()
							.Name(name)
							.Help(help_str)
							.Register(*registry)),
			m_counter(m_family.Add({}))
	{
	}

	~PrometheusMetricCounter() {}
    //------------------------------------------------------------------
    // Increment()
	void Increment() { m_counter.Increment(); }
    //------------------------------------------------------------------
    // Increment(double number)
	void Increment(double number) { m_counter.Increment(number); }
    //------------------------------------------------------------------
    // Get()
	double Get() const { return m_counter.Value(); }

private:
    //------------------------------------------------------------------
    // Data
	prometheus::Family<prometheus::Counter> &m_family;
	prometheus::Counter &m_counter;
};

typedef std::shared_ptr<PrometheusMetricCounter> MetricCounterPtr;
typedef std::shared_ptr<PrometheusMetricGauge> MetricGaugePtr;

class PrometheusMetricsBackend
{
public:
    PrometheusMetricsBackend(const std::string &addr) :
            m_exposer(std::unique_ptr<prometheus::Exposer>(
                              new prometheus::Exposer(addr))),
            m_registry(std::make_shared<prometheus::Registry>())
    {
        m_exposer->RegisterCollectable(m_registry);
    }

    ~PrometheusMetricsBackend() {}

    //------------------------------------------------------------------
    // changeBackendAddress()
    void changeBackendAddress(const std::string &addr);

    //------------------------------------------------------------------
    // addSimpleCounter()
    MetricCounterPtr addSimpleCounter(
            const std::string &name, const std::string &help_str);
    //------------------------------------------------------------------
    // addSimpleGauge()
    MetricGaugePtr addSimpleGauge(
            const std::string &name, const std::string &help_str);

    //------------------------------------------------------------------
    // createCounterFamily()
    prometheus::Family<prometheus::Counter>* createCounterFamily(
            const std::string &name, const std::string &help_str);

    //------------------------------------------------------------------
    // createGaugeFamily()
    prometheus::Family<prometheus::Gauge>* createGaugeFamily(
            const std::string &name, const std::string &help_str);

    //------------------------------------------------------------------
    // createSummaryFamily()
    prometheus::Family<prometheus::Summary>* createSummaryFamily(
            const std::string &name, const std::string &help_str);

    //------------------------------------------------------------------
    // createHistogramFamily()
    prometheus::Family<prometheus::Histogram>* createHistogramFamily(
            const std::string &name, const std::string &help_str);

    //------------------------------------------------------------------
    // createFamily()
    template <typename T>
    prometheus::Family<T>& createFamily(
            const std::string &name, const std::string &help_str)
    {
         prometheus::detail::Builder<T> builder;
         return builder.Name(name).Help(help_str).Register(*m_registry);
    }

    //------------------------------------------------------------------
    // addMetric()
    template <typename T, typename... Args>
    T& addMetric(const std::string &name,
                 const std::string &help_str,
                 const std::map<std::string,
                 std::string>& labels,
                 Args&&... args) {
          auto& family = createFamily<T>(name, help_str);
          return family.Add(labels, args...);
    }
private:
    //------------------------------------------------------------------
    // Data
    std::unique_ptr<prometheus::Exposer> m_exposer;
    std::shared_ptr<prometheus::Registry> m_registry;
};

}// namespace aerial_metrics
#endif


