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
#define AERIAL_METRICS_HPP_INCLUDED_

#include "prometheus_metrics_backend.hpp"
#include <iostream>

////////////////////////////////////////////////////////////////////////
/// aerial_metrics
namespace aerial_metrics
{

////////////////////////////////////////////////////////////////////////
/// AerialMetricsRegistrationManager
/// Wrapper class for PrometheusMetricsBackend object
class AerialMetricsRegistrationManager
{
public:
    ///------------------------------------------------------------------
    /// getInstance()
	static AerialMetricsRegistrationManager &getInstance();

    ///------------------------------------------------------------------
    /// changeBackendAddress()
    /// \brief Change Prometheus metrics backend address
    ///
    /// \param addr The new address, e.g. 127.0.0.1:8081
    void changeBackendAddress(std::string &addr);

    ///------------------------------------------------------------------
    /// addSimpleCounter()
    /// \brief Create a simple Counter metric.
    ///
    /// \param name Set the Counter metric name.
    /// \param help Set an additional description.
    /// \return A simple Counter Ptr
    /// \throw std::runtime_exception on invalid metric names.
    MetricCounterPtr addSimpleCounter(
            const std::string &name, const std::string &help_str);
    ///------------------------------------------------------------------
    /// addSimpleGauge()
    /// \brief Create a simple Gauge metric.
    ///
    /// \param name Set the Gauge metric name.
    /// \param help Set an additional description.
    /// \return A simple Gauge Ptr
    /// \throw std::runtime_exception on invalid metric names.
    MetricGaugePtr addSimpleGauge(
            const std::string &name, const std::string &help_str);

    ///------------------------------------------------------------------
    /// createCounterFamily()
    /// \brief Create a new Counter metric family.
    ///
    /// \param name Set the Counter family name.
    /// \param help Set an additional description.
    /// \return An Counter Family object
    /// \throw std::runtime_exception on invalid metric family names.
    prometheus::Family<prometheus::Counter>* createCounterFamily(
            const std::string &name, const std::string &help_str);

    ///------------------------------------------------------------------
    /// createGaugeFamily()
    /// \brief Create a new Gauge metric family.
    ///
    /// \param name Set the Gauge family name.
    /// \param help Set an additional description.
    /// \return An Gauge Family object
    /// \throw std::runtime_exception on invalid metric family names.
    prometheus::Family<prometheus::Gauge>* createGaugeFamily(
            const std::string &name, const std::string &help_str);

    ///------------------------------------------------------------------
    /// createSummaryFamily()
    /// \brief Create a new Summary metric family.
    ///
    /// \param name Set the Summary family name.
    /// \param help Set an additional description.
    /// \return An SummaryFamily object
    /// \throw std::runtime_exception on invalid metric family names.
    prometheus::Family<prometheus::Summary>* createSummaryFamily(
            const std::string &name, const std::string &help_str);

    //------------------------------------------------------------------
    // createHistogramFamily()
    prometheus::Family<prometheus::Histogram>* createHistogramFamily(
            const std::string &name, const std::string &help_str);

    //------------------------------------------------------------------
    // createFamily()
    /// \brief Create a new metric family.
    ///
    /// \param name Set the metric family name.
    /// \param help Set an additional description.
    /// \return An Family object of unspecified type T(Counter, Gauge, Histogram or Summary)
    /// \throw std::runtime_exception on invalid metric family names.
    template <typename T>
    prometheus::Family<T>& createFamily(
            const std::string &name, const std::string &help_str)
    {
        return m_prometheus_metrics_backend->createFamily<T>(name, help_str);
    }

    ///------------------------------------------------------------------
    /// addMetric()
    /// \brief Add a new metric.
    ///
    /// Every metric is uniquely identified by its name and a set of key-value
    /// pairs, also known as labels. Each new set of labels adds a new dimensional data and is exposed in
    /// Prometheus as a time series.
    /// \param name Set the metric name.
    /// \param help Set an additional description.
    /// \param labels Assign a set of key-value pairs (= labels) to the
    /// dimensional data. Each new set of labels adds a new dimensional data and is exposed in
    /// Prometheus as a time series. The function does nothing, if the same set of labels
    /// already exists.
    /// \param args Arguments are passed to the constructor of metric type T. See
    /// Counter, Gauge, Histogram or Summary for required constructor arguments.
    /// \return Return the newly created dimensional data or - if a same set of
    /// labels already exists - the already existing dimensional data.
    /// \return An object of unspecified type T(Counter, Gauge, Histogram or Summary)
    /// \throw std::runtime_exception on invalid metric or label names.
    template <typename T, typename... Args>
    T& addMetric(const std::string &name,
                 const std::string &help_str,
		 const std::map<std::string,
		 std::string>& labels,
		 Args&&... args) {
          return m_prometheus_metrics_backend->addMetric<T>(name, help_str, labels, args...);
    }
private:
    ///------------------------------------------------------------------
    /// Constructor
	AerialMetricsRegistrationManager();
    ///------------------------------------------------------------------
    /// Destructor
	~AerialMetricsRegistrationManager();
    ///------------------------------------------------------------------
    /// Data
    std::unique_ptr<PrometheusMetricsBackend> m_prometheus_metrics_backend;
};

} // namespace aerial_metrics
#endif
