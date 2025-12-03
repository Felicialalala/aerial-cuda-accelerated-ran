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

#include "prometheus_metrics_backend.hpp"

////////////////////////////////////////////////////////////////////////
// aerial_metrics
namespace aerial_metrics
{

void PrometheusMetricsBackend::changeBackendAddress(const std::string &addr)
{
        m_exposer.reset(new prometheus::Exposer(addr));
        m_exposer->RegisterCollectable(m_registry);
}

MetricCounterPtr PrometheusMetricsBackend::addSimpleCounter(
        const std::string &name, const std::string &help_str)
{
    return std::make_shared<PrometheusMetricCounter>(name, help_str, m_registry);
}

MetricGaugePtr PrometheusMetricsBackend::addSimpleGauge(
        const std::string &name, const std::string &help_str)
{
    return std::make_shared<PrometheusMetricGauge>(name, help_str, m_registry);
}

prometheus::Family<prometheus::Counter>* PrometheusMetricsBackend::createCounterFamily(
        const std::string &name, const std::string &help_str)
{
	return &(prometheus::BuildCounter().Name(name).Help(help_str).Register(*m_registry));
}

prometheus::Family<prometheus::Gauge>* PrometheusMetricsBackend::createGaugeFamily(
        const std::string &name, const std::string &help_str)
{
	return &(prometheus::BuildGauge().Name(name).Help(help_str).Register(*m_registry));
}
prometheus::Family<prometheus::Summary>* PrometheusMetricsBackend::createSummaryFamily(
        const std::string &name, const std::string &help_str)
{
	return &(prometheus::BuildSummary().Name(name).Help(help_str).Register(*m_registry));
}

prometheus::Family<prometheus::Histogram>* PrometheusMetricsBackend::createHistogramFamily(
        const std::string &name, const std::string &help_str)
{
	return &(prometheus::BuildHistogram().Name(name).Help(help_str).Register(*m_registry));
}

} // namespace aerial_metrics
