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

////////////////////////////////////////////////////////////////////////
// aerial_metrics
namespace aerial_metrics
{

AerialMetricsRegistrationManager::AerialMetricsRegistrationManager()
{
    std::string addr(ep_addr);
    m_prometheus_metrics_backend = std::unique_ptr<PrometheusMetricsBackend>(new PrometheusMetricsBackend(addr));
}

AerialMetricsRegistrationManager::~AerialMetricsRegistrationManager()
{
}

void AerialMetricsRegistrationManager::changeBackendAddress(std::string &addr)
{
    std::cout<<"Aerial metrics backend address: "<<addr<<std::endl;
    m_prometheus_metrics_backend->changeBackendAddress(addr);
}

AerialMetricsRegistrationManager &AerialMetricsRegistrationManager::getInstance()
{
	static AerialMetricsRegistrationManager aerialMetricsRegistrationManager;
	return aerialMetricsRegistrationManager;
}

MetricCounterPtr AerialMetricsRegistrationManager::addSimpleCounter(
        const std::string &name, const std::string &help_str)
{
    return m_prometheus_metrics_backend->addSimpleCounter(name, help_str);
}

MetricGaugePtr AerialMetricsRegistrationManager::addSimpleGauge(
        const std::string &name, const std::string &help_str)
{
    return m_prometheus_metrics_backend->addSimpleGauge(name, help_str);
}

prometheus::Family<prometheus::Counter>* AerialMetricsRegistrationManager::createCounterFamily(
        const std::string &name, const std::string &help_str)
{
	return m_prometheus_metrics_backend->createCounterFamily(name, help_str);
}

prometheus::Family<prometheus::Gauge>* AerialMetricsRegistrationManager::createGaugeFamily(
        const std::string &name, const std::string &help_str)
{
	return m_prometheus_metrics_backend->createGaugeFamily(name, help_str);
}
prometheus::Family<prometheus::Summary>* AerialMetricsRegistrationManager::createSummaryFamily(
        const std::string &name, const std::string &help_str)
{
	return m_prometheus_metrics_backend->createSummaryFamily(name, help_str);
}

prometheus::Family<prometheus::Histogram>* AerialMetricsRegistrationManager::createHistogramFamily(
        const std::string &name, const std::string &help_str)
{
	return m_prometheus_metrics_backend->createHistogramFamily(name, help_str);
}

} // namespace aerial_metrics
