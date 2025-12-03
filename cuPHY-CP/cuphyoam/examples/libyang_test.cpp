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

#include <iostream>
#include <libyang-cpp/Context.hpp>
#include <libyang-cpp/Utils.hpp>
#include <libyang-cpp/Context.hpp>
#include <libyang-cpp/Type.hpp>
#include <libyang-cpp/Utils.hpp>
#include "nvlog.hpp"


using namespace std::literals;

auto origXML3 = R"|(<config xmlns="http://tail-f.com/ns/config/1.0">
  <delay-management xmlns="urn:xran:delay:2.0">
    <bandwidth-scs-delay-state>
      <bandwidth>20000</bandwidth>
      <subcarrier-spacing>15000</subcarrier-spacing>
    </bandwidth-scs-delay-state>
    <bandwidth-scs-delay-state>
      <bandwidth>50000</bandwidth>
      <subcarrier-spacing>30000</subcarrier-spacing>
    </bandwidth-scs-delay-state>
    <bandwidth-scs-delay-state>
      <bandwidth>100000</bandwidth>
      <subcarrier-spacing>60000</subcarrier-spacing>
    </bandwidth-scs-delay-state>
  </delay-management>
  <ecpri-delay-message xmlns="urn:xran:message5:1.0">
    <ru-compensation>
      <tcv2>9</tcv2>
      <tcv1>19</tcv1>
    </ru-compensation>
    <enable-message5>true</enable-message5>
    <message5-sessions>
      <session-parameters>
        <session-id>0</session-id>
        <processing-element-name>element0</processing-element-name>
        <flow-state>
          <responses-transmitted>236536</responses-transmitted>
          <requests-transmitted>96734</requests-transmitted>
          <followups-transmitted>0</followups-transmitted>
        </flow-state>
      </session-parameters>
      <session-parameters>
        <session-id>1</session-id>
        <processing-element-name>element1</processing-element-name>
        <flow-state>
          <responses-transmitted>12525</responses-transmitted>
          <requests-transmitted>74513</requests-transmitted>
          <followups-transmitted>99</followups-transmitted>
        </flow-state>
      </session-parameters>
    </message5-sessions>
  </ecpri-delay-message>
</config>)|"s;


auto origXML4 = R"|(<config xmlns="http://tail-f.com/ns/config/1.0">
  <xran-users xmlns="urn:xran:user-mgmt:2.0">
    <user>
      <name>fmpmuser</name>
      <password>hashedpassword</password>
      <enabled>true</enabled>
    </user>
    <user>
      <name>nmsuser</name>
      <password>hashedpassword</password>
      <enabled>true</enabled>
    </user>
    <user>
      <name>swmuser</name>
      <password>hashedpassword</password>
      <enabled>true</enabled>
    </user>
    <user>
      <name>xranuser</name>
      <password>hashedpassword</password>
      <enabled>true</enabled>
    </user>
  </xran-users>
</config>)|"s;

auto origXML2 = R"|(<ax xmlns="http://example.com/coze"><x>1</x><x>2</x><x>3</x></ax>)|"s;

auto origXML = R"|(  <data xmlns="urn:ietf:params:xml:ns:netconf:base:1.0">
    <user-plane-configuration xmlns="urn:o-ran:uplane-conf:1.0">
      <low-level-tx-endpoints>
        <name>llte_0_0_0_1</name>
        <compression>
          <compression-type>STATIC</compression-type>
          <bitwidth>3</bitwidth>
          <iq-bitwidth>0</iq-bitwidth>
          <exponent>4</exponent>
        </compression>
        <frame-structure>160</frame-structure>
        <cp-type>NORMAL</cp-type>
        <cp-length>160</cp-length>
        <cp-length-other>144</cp-length-other>
        <offset-to-absolute-frequency-center>-624</offset-to-absolute-frequency-center>
        <number-of-prb-per-scs>
          <scs>KHZ_15</scs>
          <number-of-prb>52</number-of-prb>
        </number-of-prb-per-scs>
        <e-axcid>
          <o-du-port-bitmask>65408</o-du-port-bitmask>
          <band-sector-bitmask>64</band-sector-bitmask>
          <ccid-bitmask>48</ccid-bitmask>
          <ru-port-bitmask>15</ru-port-bitmask>
          <eaxc-id>384</eaxc-id>
        </e-axcid>
      </low-level-tx-endpoints>
      <tx-array-carriers>
        <name>tac_0_0_0_1</name>
        <absolute-frequency-center>175800</absolute-frequency-center>
        <center-of-channel-bandwidth>879000000</center-of-channel-bandwidth>
        <channel-bandwidth>10000000</channel-bandwidth>
        <active>ACTIVE</active>
        <gain>7.3481</gain>
        <downlink-radio-frame-offset>0</downlink-radio-frame-offset>
        <downlink-sfn-offset>0</downlink-sfn-offset>
        <state>READY</state>
      </tx-array-carriers>
      <low-level-tx-links>
        <name>lltl_0_0_0_1</name>
        <processing-element>re_0_0_0_501</processing-element>
        <tx-array-carrier>tac_0_0_0_1</tx-array-carrier>
        <low-level-tx-endpoint>llte_0_0_0_1</low-level-tx-endpoint>
      </low-level-tx-links>
      <low-level-tx-endpoints>
        <name>llte_0_0_0_2</name>
        <compression>
          <compression-type>STATIC</compression-type>
          <bitwidth>3</bitwidth>
          <iq-bitwidth>0</iq-bitwidth>
          <exponent>4</exponent>
        </compression>
        <frame-structure>160</frame-structure>
        <cp-type>NORMAL</cp-type>
        <cp-length>160</cp-length>
        <cp-length-other>144</cp-length-other>
        <offset-to-absolute-frequency-center>-624</offset-to-absolute-frequency-center>
        <number-of-prb-per-scs>
          <scs>KHZ_15</scs>
          <number-of-prb>52</number-of-prb>
        </number-of-prb-per-scs>
        <e-axcid>
          <o-du-port-bitmask>65408</o-du-port-bitmask>
          <band-sector-bitmask>64</band-sector-bitmask>
          <ccid-bitmask>48</ccid-bitmask>
          <ru-port-bitmask>15</ru-port-bitmask>
          <eaxc-id>385</eaxc-id>
        </e-axcid>
      </low-level-tx-endpoints>
      <tx-array-carriers>
        <name>tac_0_0_0_2</name>
        <absolute-frequency-center>175800</absolute-frequency-center>
        <center-of-channel-bandwidth>879000000</center-of-channel-bandwidth>
        <channel-bandwidth>10000000</channel-bandwidth>
        <active>ACTIVE</active>
        <gain>7.3481</gain>
        <downlink-radio-frame-offset>0</downlink-radio-frame-offset>
        <downlink-sfn-offset>0</downlink-sfn-offset>
        <state>READY</state>
      </tx-array-carriers>
      <low-level-tx-links>
        <name>lltl_0_0_0_2</name>
        <processing-element>re_0_0_0_501</processing-element>
        <tx-array-carrier>tac_0_0_0_2</tx-array-carrier>
        <low-level-tx-endpoint>llte_0_0_0_2</low-level-tx-endpoint>
      </low-level-tx-links>
      <low-level-rx-endpoints>
        <name>llre_0_0_0_1</name>
        <compression>
          <compression-type>STATIC</compression-type>
          <bitwidth>3</bitwidth>
          <iq-bitwidth>0</iq-bitwidth>
          <exponent>4</exponent>
        </compression>
        <frame-structure>160</frame-structure>
        <cp-type>NORMAL</cp-type>
        <cp-length>160</cp-length>
        <cp-length-other>144</cp-length-other>
        <offset-to-absolute-frequency-center>-624</offset-to-absolute-frequency-center>
        <number-of-prb-per-scs>
          <scs>KHZ_15</scs>
          <number-of-prb>52</number-of-prb>
        </number-of-prb-per-scs>
        <ul-fft-sampling-offsets>
          <scs>KHZ_15</scs>
          <ul-fft-sampling-offset>36</ul-fft-sampling-offset>
        </ul-fft-sampling-offsets>
        <e-axcid>
          <o-du-port-bitmask>65408</o-du-port-bitmask>
          <band-sector-bitmask>64</band-sector-bitmask>
          <ccid-bitmask>48</ccid-bitmask>
          <ru-port-bitmask>15</ru-port-bitmask>
          <eaxc-id>384</eaxc-id>
        </e-axcid>
      </low-level-rx-endpoints>
      <rx-array-carriers>
        <name>rac_0_0_0_1</name>
        <absolute-frequency-center>166800</absolute-frequency-center>
        <center-of-channel-bandwidth>834000000</center-of-channel-bandwidth>
        <channel-bandwidth>10000000</channel-bandwidth>
        <active>ACTIVE</active>
        <n-ta-offset>13792</n-ta-offset>
        <gain-correction>0.0</gain-correction>
        <downlink-radio-frame-offset>0</downlink-radio-frame-offset>
        <downlink-sfn-offset>0</downlink-sfn-offset>
        <state>READY</state>
      </rx-array-carriers>
      <low-level-rx-links>
        <name>llrl_0_0_0_1</name>
        <processing-element>re_0_0_0_501</processing-element>
        <rx-array-carrier>rac_0_0_0_1</rx-array-carrier>
        <low-level-rx-endpoint>llre_0_0_0_1</low-level-rx-endpoint>
      </low-level-rx-links>
      <low-level-rx-endpoints>
        <name>llre_0_0_0_2</name>
        <compression>
          <compression-type>STATIC</compression-type>
          <bitwidth>3</bitwidth>
          <iq-bitwidth>0</iq-bitwidth>
          <exponent>4</exponent>
        </compression>
        <frame-structure>160</frame-structure>
        <cp-type>NORMAL</cp-type>
        <cp-length>160</cp-length>
        <cp-length-other>144</cp-length-other>
        <offset-to-absolute-frequency-center>-624</offset-to-absolute-frequency-center>
        <number-of-prb-per-scs>
          <scs>KHZ_15</scs>
          <number-of-prb>52</number-of-prb>
        </number-of-prb-per-scs>
        <ul-fft-sampling-offsets>
          <scs>KHZ_15</scs>
          <ul-fft-sampling-offset>36</ul-fft-sampling-offset>
        </ul-fft-sampling-offsets>
        <e-axcid>
          <o-du-port-bitmask>65408</o-du-port-bitmask>
          <band-sector-bitmask>64</band-sector-bitmask>
          <ccid-bitmask>48</ccid-bitmask>
          <ru-port-bitmask>15</ru-port-bitmask>
          <eaxc-id>385</eaxc-id>
        </e-axcid>
      </low-level-rx-endpoints>
      <rx-array-carriers>
        <name>rac_0_0_0_2</name>
        <absolute-frequency-center>166800</absolute-frequency-center>
        <center-of-channel-bandwidth>834000000</center-of-channel-bandwidth>
        <channel-bandwidth>10000000</channel-bandwidth>
        <active>ACTIVE</active>
        <n-ta-offset>13792</n-ta-offset>
        <gain-correction>0.0</gain-correction>
        <downlink-radio-frame-offset>0</downlink-radio-frame-offset>
        <downlink-sfn-offset>0</downlink-sfn-offset>
        <state>READY</state>
      </rx-array-carriers>
      <low-level-rx-links>
        <name>llrl_0_0_0_2</name>
        <processing-element>re_0_0_0_501</processing-element>
        <rx-array-carrier>rac_0_0_0_2</rx-array-carrier>
        <low-level-rx-endpoint>llre_0_0_0_2</low-level-rx-endpoint>
      </low-level-rx-links>
      <low-level-tx-endpoints>
        <name>llte_0_1_0_3</name>
        <compression>
          <compression-type>STATIC</compression-type>
          <bitwidth>3</bitwidth>
          <iq-bitwidth>0</iq-bitwidth>
          <exponent>4</exponent>
        </compression>
        <frame-structure>160</frame-structure>
        <cp-type>NORMAL</cp-type>
        <cp-length>160</cp-length>
        <cp-length-other>144</cp-length-other>
        <offset-to-absolute-frequency-center>-624</offset-to-absolute-frequency-center>
        <number-of-prb-per-scs>
          <scs>KHZ_15</scs>
          <number-of-prb>52</number-of-prb>
        </number-of-prb-per-scs>
        <e-axcid>
          <o-du-port-bitmask>65408</o-du-port-bitmask>
          <band-sector-bitmask>64</band-sector-bitmask>
          <ccid-bitmask>48</ccid-bitmask>
          <ru-port-bitmask>15</ru-port-bitmask>
          <eaxc-id>578</eaxc-id>
        </e-axcid>
      </low-level-tx-endpoints>
      <tx-array-carriers>
        <name>tac_0_1_0_3</name>
        <absolute-frequency-center>387000</absolute-frequency-center>
        <center-of-channel-bandwidth>1935000000</center-of-channel-bandwidth>
        <channel-bandwidth>10000000</channel-bandwidth>
        <active>ACTIVE</active>
        <gain>7.3481</gain>
        <downlink-radio-frame-offset>0</downlink-radio-frame-offset>
        <downlink-sfn-offset>0</downlink-sfn-offset>
        <state>READY</state>
      </tx-array-carriers>
      <low-level-tx-links>
        <name>lltl_0_1_0_3</name>
        <processing-element>re_0_0_0_501</processing-element>
        <tx-array-carrier>tac_0_1_0_3</tx-array-carrier>
        <low-level-tx-endpoint>llte_0_1_0_3</low-level-tx-endpoint>
      </low-level-tx-links>
      <low-level-tx-endpoints>
        <name>llte_0_1_0_4</name>
        <compression>
          <compression-type>STATIC</compression-type>
          <bitwidth>3</bitwidth>
          <iq-bitwidth>0</iq-bitwidth>
          <exponent>4</exponent>
        </compression>
        <frame-structure>160</frame-structure>
        <cp-type>NORMAL</cp-type>
        <cp-length>160</cp-length>
        <cp-length-other>144</cp-length-other>
        <offset-to-absolute-frequency-center>-624</offset-to-absolute-frequency-center>
        <number-of-prb-per-scs>
          <scs>KHZ_15</scs>
          <number-of-prb>52</number-of-prb>
        </number-of-prb-per-scs>
        <e-axcid>
          <o-du-port-bitmask>65408</o-du-port-bitmask>
          <band-sector-bitmask>64</band-sector-bitmask>
          <ccid-bitmask>48</ccid-bitmask>
          <ru-port-bitmask>15</ru-port-bitmask>
          <eaxc-id>579</eaxc-id>
        </e-axcid>
      </low-level-tx-endpoints>
      <tx-array-carriers>
        <name>tac_0_1_0_4</name>
        <absolute-frequency-center>387000</absolute-frequency-center>
        <center-of-channel-bandwidth>1935000000</center-of-channel-bandwidth>
        <channel-bandwidth>10000000</channel-bandwidth>
        <active>ACTIVE</active>
        <gain>7.3481</gain>
        <downlink-radio-frame-offset>0</downlink-radio-frame-offset>
        <downlink-sfn-offset>0</downlink-sfn-offset>
        <state>READY</state>
      </tx-array-carriers>
      <low-level-tx-links>
        <name>lltl_0_1_0_4</name>
        <processing-element>re_0_0_0_501</processing-element>
        <tx-array-carrier>tac_0_1_0_4</tx-array-carrier>
        <low-level-tx-endpoint>llte_0_1_0_4</low-level-tx-endpoint>
      </low-level-tx-links>
      <low-level-rx-endpoints>
        <name>llre_0_1_0_3</name>
        <compression>
          <compression-type>STATIC</compression-type>
          <bitwidth>3</bitwidth>
          <iq-bitwidth>0</iq-bitwidth>
          <exponent>4</exponent>
        </compression>
        <frame-structure>160</frame-structure>
        <cp-type>NORMAL</cp-type>
        <cp-length>160</cp-length>
        <cp-length-other>144</cp-length-other>
        <offset-to-absolute-frequency-center>-624</offset-to-absolute-frequency-center>
        <number-of-prb-per-scs>
          <scs>KHZ_15</scs>
          <number-of-prb>52</number-of-prb>
        </number-of-prb-per-scs>
        <ul-fft-sampling-offsets>
          <scs>KHZ_15</scs>
          <ul-fft-sampling-offset>36</ul-fft-sampling-offset>
        </ul-fft-sampling-offsets>
        <e-axcid>
          <o-du-port-bitmask>65408</o-du-port-bitmask>
          <band-sector-bitmask>64</band-sector-bitmask>
          <ccid-bitmask>48</ccid-bitmask>
          <ru-port-bitmask>15</ru-port-bitmask>
          <eaxc-id>578</eaxc-id>
        </e-axcid>
      </low-level-rx-endpoints>
      <rx-array-carriers>
        <name>rac_0_1_0_3</name>
        <absolute-frequency-center>371000</absolute-frequency-center>
        <center-of-channel-bandwidth>1855000000</center-of-channel-bandwidth>
        <channel-bandwidth>10000000</channel-bandwidth>
        <active>ACTIVE</active>
        <n-ta-offset>13792</n-ta-offset>
        <gain-correction>0.0</gain-correction>
        <downlink-radio-frame-offset>0</downlink-radio-frame-offset>
        <downlink-sfn-offset>0</downlink-sfn-offset>
        <state>READY</state>
      </rx-array-carriers>
      <low-level-rx-links>
        <name>llrl_0_1_0_3</name>
        <processing-element>re_0_0_0_501</processing-element>
        <rx-array-carrier>rac_0_1_0_3</rx-array-carrier>
        <low-level-rx-endpoint>llre_0_1_0_3</low-level-rx-endpoint>
      </low-level-rx-links>
      <low-level-rx-endpoints>
        <name>llre_0_1_0_4</name>
        <compression>
          <compression-type>STATIC</compression-type>
          <bitwidth>3</bitwidth>
          <iq-bitwidth>0</iq-bitwidth>
          <exponent>4</exponent>
        </compression>
        <frame-structure>160</frame-structure>
        <cp-type>NORMAL</cp-type>
        <cp-length>160</cp-length>
        <cp-length-other>144</cp-length-other>
        <offset-to-absolute-frequency-center>-624</offset-to-absolute-frequency-center>
        <number-of-prb-per-scs>
          <scs>KHZ_15</scs>
          <number-of-prb>52</number-of-prb>
        </number-of-prb-per-scs>
        <ul-fft-sampling-offsets>
          <scs>KHZ_15</scs>
          <ul-fft-sampling-offset>36</ul-fft-sampling-offset>
        </ul-fft-sampling-offsets>
        <e-axcid>
          <o-du-port-bitmask>65408</o-du-port-bitmask>
          <band-sector-bitmask>64</band-sector-bitmask>
          <ccid-bitmask>48</ccid-bitmask>
          <ru-port-bitmask>15</ru-port-bitmask>
          <eaxc-id>579</eaxc-id>
        </e-axcid>
      </low-level-rx-endpoints>
      <rx-array-carriers>
        <name>rac_0_1_0_4</name>
        <absolute-frequency-center>371000</absolute-frequency-center>
        <center-of-channel-bandwidth>1855000000</center-of-channel-bandwidth>
        <channel-bandwidth>10000000</channel-bandwidth>
        <active>ACTIVE</active>
        <n-ta-offset>13792</n-ta-offset>
        <gain-correction>0.0</gain-correction>
        <downlink-radio-frame-offset>0</downlink-radio-frame-offset>
        <downlink-sfn-offset>0</downlink-sfn-offset>
        <state>READY</state>
      </rx-array-carriers>
      <low-level-rx-links>
        <name>llrl_0_1_0_4</name>
        <processing-element>re_0_0_0_501</processing-element>
        <rx-array-carrier>rac_0_1_0_4</rx-array-carrier>
        <low-level-rx-endpoint>llre_0_1_0_4</low-level-rx-endpoint>
      </low-level-rx-links>
      <endpoint-types>
        <id>0</id>
        <supported-frame-structures>144</supported-frame-structures>
        <supported-frame-structures>160</supported-frame-structures>
        <supported-frame-structures>172</supported-frame-structures>
        <supported-frame-structures>176</supported-frame-structures>
        <supported-frame-structures>208</supported-frame-structures>
        <managed-delay-support>MANAGED</managed-delay-support>
        <multiple-numerology-supported>false</multiple-numerology-supported>
        <max-control-sections-per-data-section>1</max-control-sections-per-data-section>
        <max-sections-per-symbol>18</max-sections-per-symbol>
        <max-sections-per-slot>72</max-sections-per-slot>
        <max-remasks-per-section-id>1</max-remasks-per-section-id>
        <max-beams-per-symbol>1</max-beams-per-symbol>
        <max-beams-per-slot>1</max-beams-per-slot>
        <max-prb-per-symbol>106</max-prb-per-symbol>
        <prb-capacity-allocation-granularity>1</prb-capacity-allocation-granularity>
        <max-numerologies-per-symbol>1</max-numerologies-per-symbol>
      </endpoint-types>
      <endpoint-types>
        <id>1</id>
        <supported-frame-structures>144</supported-frame-structures>
        <supported-frame-structures>160</supported-frame-structures>
        <supported-frame-structures>172</supported-frame-structures>
        <managed-delay-support>MANAGED</managed-delay-support>
        <multiple-numerology-supported>false</multiple-numerology-supported>
        <max-control-sections-per-data-section>1</max-control-sections-per-data-section>
        <max-sections-per-symbol>18</max-sections-per-symbol>
        <max-sections-per-slot>72</max-sections-per-slot>
        <max-remasks-per-section-id>1</max-remasks-per-section-id>
        <max-beams-per-symbol>1</max-beams-per-symbol>
        <max-beams-per-slot>1</max-beams-per-slot>
        <max-prb-per-symbol>52</max-prb-per-symbol>
        <prb-capacity-allocation-granularity>1</prb-capacity-allocation-granularity>
        <max-numerologies-per-symbol>1</max-numerologies-per-symbol>
      </endpoint-types>
      <static-low-level-tx-endpoints>
        <name>llte_0_0_0_1</name>
        <array>ta_1</array>
        <endpoint-type>0</endpoint-type>
      </static-low-level-tx-endpoints>
      <static-low-level-tx-endpoints>
        <name>llte_0_0_0_2</name>
        <array>ta_2</array>
        <endpoint-type>0</endpoint-type>
      </static-low-level-tx-endpoints>
      <static-low-level-tx-endpoints>
        <name>llte_0_0_1_1</name>
        <array>ta_1</array>
        <endpoint-type>1</endpoint-type>
      </static-low-level-tx-endpoints>
      <static-low-level-tx-endpoints>
        <name>llte_0_0_1_2</name>
        <array>ta_2</array>
        <endpoint-type>1</endpoint-type>
      </static-low-level-tx-endpoints>
      <static-low-level-tx-endpoints>
        <name>llte_0_1_0_3</name>
        <array>ta_3</array>
        <endpoint-type>0</endpoint-type>
      </static-low-level-tx-endpoints>
      <static-low-level-tx-endpoints>
        <name>llte_0_1_0_4</name>
        <array>ta_4</array>
        <endpoint-type>0</endpoint-type>
      </static-low-level-tx-endpoints>
      <static-low-level-tx-endpoints>
        <name>llte_0_1_1_3</name>
        <array>ta_3</array>
        <endpoint-type>1</endpoint-type>
      </static-low-level-tx-endpoints>
      <static-low-level-tx-endpoints>
        <name>llte_0_1_1_4</name>
        <array>ta_4</array>
        <endpoint-type>1</endpoint-type>
      </static-low-level-tx-endpoints>
      <static-low-level-rx-endpoints>
        <name>llre_0_0_0_1</name>
        <array>ra_1</array>
        <endpoint-type>0</endpoint-type>
      </static-low-level-rx-endpoints>
      <static-low-level-rx-endpoints>
        <name>llre_0_0_0_2</name>
        <array>ra_2</array>
        <endpoint-type>0</endpoint-type>
      </static-low-level-rx-endpoints>
      <static-low-level-rx-endpoints>
        <name>llre_0_0_1_1</name>
        <array>ra_1</array>
        <endpoint-type>1</endpoint-type>
      </static-low-level-rx-endpoints>
      <static-low-level-rx-endpoints>
        <name>llre_0_0_1_2</name>
        <array>ra_2</array>
        <endpoint-type>1</endpoint-type>
      </static-low-level-rx-endpoints>
      <static-low-level-rx-endpoints>
        <name>llre_0_1_0_3</name>
        <array>ra_3</array>
        <endpoint-type>0</endpoint-type>
      </static-low-level-rx-endpoints>
      <static-low-level-rx-endpoints>
        <name>llre_0_1_0_4</name>
        <array>ra_4</array>
        <endpoint-type>0</endpoint-type>
      </static-low-level-rx-endpoints>
      <static-low-level-rx-endpoints>
        <name>llre_0_1_1_3</name>
        <array>ra_3</array>
        <endpoint-type>1</endpoint-type>
      </static-low-level-rx-endpoints>
      <static-low-level-rx-endpoints>
        <name>llre_0_1_1_4</name>
        <array>ra_4</array>
        <endpoint-type>1</endpoint-type>
      </static-low-level-rx-endpoints>
      <tx-arrays>
        <name>ta_1</name>
        <number-of-rows>1</number-of-rows>
        <number-of-columns>1</number-of-columns>
        <number-of-array-layers>1</number-of-array-layers>
        <band-number>5</band-number>
        <max-gain>0.0</max-gain>
        <independent-power-budget>false</independent-power-budget>
        <polarisations>
          <p>0</p>
          <polarisation>ZERO</polarisation>
        </polarisations>
        <capabilities>
          <max-supported-frequency-dl>894000000</max-supported-frequency-dl>
          <min-supported-frequency-dl>869000000</min-supported-frequency-dl>
          <max-supported-bandwidth-dl>25000000</max-supported-bandwidth-dl>
          <max-num-carriers-dl>2</max-num-carriers-dl>
          <max-carrier-bandwidth-dl>20000000</max-carrier-bandwidth-dl>
          <min-carrier-bandwidth-dl>5000000</min-carrier-bandwidth-dl>
          <supported-technology-dl>NR</supported-technology-dl>
        </capabilities>
      </tx-arrays>
      <tx-arrays>
        <name>ta_2</name>
        <number-of-rows>1</number-of-rows>
        <number-of-columns>1</number-of-columns>
        <number-of-array-layers>1</number-of-array-layers>
        <band-number>5</band-number>
        <max-gain>0.0</max-gain>
        <independent-power-budget>false</independent-power-budget>
        <polarisations>
          <p>0</p>
          <polarisation>ZERO</polarisation>
        </polarisations>
        <capabilities>
          <max-supported-frequency-dl>894000000</max-supported-frequency-dl>
          <min-supported-frequency-dl>869000000</min-supported-frequency-dl>
          <max-supported-bandwidth-dl>25000000</max-supported-bandwidth-dl>
          <max-num-carriers-dl>2</max-num-carriers-dl>
          <max-carrier-bandwidth-dl>20000000</max-carrier-bandwidth-dl>
          <min-carrier-bandwidth-dl>5000000</min-carrier-bandwidth-dl>
          <supported-technology-dl>NR</supported-technology-dl>
        </capabilities>
      </tx-arrays>
      <tx-arrays>
        <name>ta_3</name>
        <number-of-rows>1</number-of-rows>
        <number-of-columns>1</number-of-columns>
        <number-of-array-layers>1</number-of-array-layers>
        <band-number>2</band-number>
        <max-gain>0.0</max-gain>
        <independent-power-budget>false</independent-power-budget>
        <polarisations>
          <p>0</p>
          <polarisation>ZERO</polarisation>
        </polarisations>
        <capabilities>
          <max-supported-frequency-dl>1999000000</max-supported-frequency-dl>
          <min-supported-frequency-dl>1930000000</min-supported-frequency-dl>
          <max-supported-bandwidth-dl>60000000</max-supported-bandwidth-dl>
          <max-num-carriers-dl>2</max-num-carriers-dl>
          <max-carrier-bandwidth-dl>20000000</max-carrier-bandwidth-dl>
          <min-carrier-bandwidth-dl>5000000</min-carrier-bandwidth-dl>
          <supported-technology-dl>NR</supported-technology-dl>
        </capabilities>
      </tx-arrays>
      <tx-arrays>
        <name>ta_4</name>
        <number-of-rows>1</number-of-rows>
        <number-of-columns>1</number-of-columns>
        <number-of-array-layers>1</number-of-array-layers>
        <band-number>2</band-number>
        <max-gain>0.0</max-gain>
        <independent-power-budget>false</independent-power-budget>
        <polarisations>
          <p>0</p>
          <polarisation>ZERO</polarisation>
        </polarisations>
        <capabilities>
          <max-supported-frequency-dl>1999000000</max-supported-frequency-dl>
          <min-supported-frequency-dl>1930000000</min-supported-frequency-dl>
          <max-supported-bandwidth-dl>60000000</max-supported-bandwidth-dl>
          <max-num-carriers-dl>2</max-num-carriers-dl>
          <max-carrier-bandwidth-dl>20000000</max-carrier-bandwidth-dl>
          <min-carrier-bandwidth-dl>5000000</min-carrier-bandwidth-dl>
          <supported-technology-dl>NR</supported-technology-dl>
        </capabilities>
      </tx-arrays>
      <rx-arrays>
        <name>ra_1</name>
        <number-of-rows>1</number-of-rows>
        <number-of-columns>1</number-of-columns>
        <number-of-array-layers>1</number-of-array-layers>
        <band-number>5</band-number>
        <polarisations>
          <p>0</p>
          <polarisation>ZERO</polarisation>
        </polarisations>
        <gain-correction-range>
          <max>0.0</max>
          <min>0.0</min>
        </gain-correction-range>
        <capabilities>
          <max-supported-frequency-ul>849000000</max-supported-frequency-ul>
          <min-supported-frequency-ul>824000000</min-supported-frequency-ul>
          <max-supported-bandwidth-ul>25000000</max-supported-bandwidth-ul>
          <max-num-carriers-ul>2</max-num-carriers-ul>
          <max-carrier-bandwidth-ul>20000000</max-carrier-bandwidth-ul>
          <min-carrier-bandwidth-ul>5000000</min-carrier-bandwidth-ul>
          <supported-technology-ul>NR</supported-technology-ul>
        </capabilities>
      </rx-arrays>
      <rx-arrays>
        <name>ra_2</name>
        <number-of-rows>1</number-of-rows>
        <number-of-columns>1</number-of-columns>
        <number-of-array-layers>1</number-of-array-layers>
        <band-number>5</band-number>
        <polarisations>
          <p>0</p>
          <polarisation>ZERO</polarisation>
        </polarisations>
        <gain-correction-range>
          <max>0.0</max>
          <min>0.0</min>
        </gain-correction-range>
        <capabilities>
          <max-supported-frequency-ul>849000000</max-supported-frequency-ul>
          <min-supported-frequency-ul>824000000</min-supported-frequency-ul>
          <max-supported-bandwidth-ul>25000000</max-supported-bandwidth-ul>
          <max-num-carriers-ul>2</max-num-carriers-ul>
          <max-carrier-bandwidth-ul>20000000</max-carrier-bandwidth-ul>
          <min-carrier-bandwidth-ul>5000000</min-carrier-bandwidth-ul>
          <supported-technology-ul>NR</supported-technology-ul>
        </capabilities>
      </rx-arrays>
      <rx-arrays>
        <name>ra_3</name>
        <number-of-rows>1</number-of-rows>
        <number-of-columns>1</number-of-columns>
        <number-of-array-layers>1</number-of-array-layers>
        <band-number>2</band-number>
        <polarisations>
          <p>0</p>
          <polarisation>ZERO</polarisation>
        </polarisations>
        <gain-correction-range>
          <max>0.0</max>
          <min>0.0</min>
        </gain-correction-range>
        <capabilities>
          <max-supported-frequency-ul>1910000000</max-supported-frequency-ul>
          <min-supported-frequency-ul>1850000000</min-supported-frequency-ul>
          <max-supported-bandwidth-ul>6000000</max-supported-bandwidth-ul>
          <max-num-carriers-ul>2</max-num-carriers-ul>
          <max-carrier-bandwidth-ul>20000000</max-carrier-bandwidth-ul>
          <min-carrier-bandwidth-ul>5000000</min-carrier-bandwidth-ul>
          <supported-technology-ul>NR</supported-technology-ul>
        </capabilities>
      </rx-arrays>
      <rx-arrays>
        <name>ra_4</name>
        <number-of-rows>1</number-of-rows>
        <number-of-columns>1</number-of-columns>
        <number-of-array-layers>1</number-of-array-layers>
        <band-number>2</band-number>
        <polarisations>
          <p>0</p>
          <polarisation>ZERO</polarisation>
        </polarisations>
        <gain-correction-range>
          <max>0.0</max>
          <min>0.0</min>
        </gain-correction-range>
        <capabilities>
          <max-supported-frequency-ul>1910000000</max-supported-frequency-ul>
          <min-supported-frequency-ul>1850000000</min-supported-frequency-ul>
          <max-supported-bandwidth-ul>6000000</max-supported-bandwidth-ul>
          <max-num-carriers-ul>2</max-num-carriers-ul>
          <max-carrier-bandwidth-ul>20000000</max-carrier-bandwidth-ul>
          <min-carrier-bandwidth-ul>5000000</min-carrier-bandwidth-ul>
          <supported-technology-ul>NR</supported-technology-ul>
        </capabilities>
      </rx-arrays>
    </user-plane-configuration>
  </data>
)|"s;

const auto example_schema2 = R"(
module example-schema2 {
    yang-version 1.1;
    namespace "http://example2.com/";
    prefix lol;
    container contWithTwoNodes {
        presence true;
        leaf one {
            type int32;
        }

        leaf two {
            type int32;
        }
    }
}
)"s;


auto ecpri_delay = R"(<data xmlns="urn:ietf:params:xml:ns:netconf:base:1.0">
  <ecpri-delay-message xmlns="urn:o-ran:message5:1.0">
    <ru-compensation>
      <tcv2/>
      <tcv1/>
    </ru-compensation>
    <enable-message5>false</enable-message5>
    <one-step-t34-supported/>
    <two-step-t34-supported/>
    <message5-sessions>
      <session-parameters>
        <session-id/>
        <processing-element-name/>
        <transport-session-type/>
        <transport-qualified-processing-element-name/>
        <flow-state>
          <responses-transmitted/>
          <requests-transmitted/>
          <followups-transmitted/>
        </flow-state>
      </session-parameters>
    </message5-sessions>
  </ecpri-delay-message>
</data>)"s;



//<?xml version='1.0' encoding='UTF-8'?>
//auto delay = R"(<data xmlns="urn:ietf:params:xml:ns:netconf:base:1.0">
auto delay = R"(<delay-management xmlns="urn:o-ran:delay:1.0">
    <bandwidth-scs-delay-state>
      <bandwidth>100000</bandwidth>
      <subcarrier-spacing>1000</subcarrier-spacing>
      <ru-delay-profile>
        <t2a-min-up>1000000000</t2a-min-up>
        <t2a-max-up>1000000000</t2a-max-up>
        <t2a-min-cp-dl>1000000000</t2a-min-cp-dl>
        <t2a-max-cp-dl>1000000000</t2a-max-cp-dl>
        <tcp-adv-dl>1000000000</tcp-adv-dl>
        <ta3-min>1000000000</ta3-min>
        <ta3-max>1000000000</ta3-max>
        <t2a-min-cp-ul>1000000000</t2a-min-cp-ul>
        <t2a-max-cp-ul>1000000000</t2a-max-cp-ul>
        <ta3-min-ack>1000000000</ta3-min-ack>
        <ta3-max-ack>1000000000</ta3-max-ack>
      </ru-delay-profile>
    </bandwidth-scs-delay-state>
    <adaptive-delay-configuration>
      <bandwidth-scs-delay-state>
        <bandwidth>100000</bandwidth>
        <subcarrier-spacing>1000</subcarrier-spacing>
        <o-du-delay-profile>
          <t1a-max-up>1000000000</t1a-max-up>
          <tx-max>1000000000</tx-max>
          <ta4-max>1000000000</ta4-max>
          <rx-max>1000000000</rx-max>
          <t1a-max-cp-dl>1000000000</t1a-max-cp-dl>
        </o-du-delay-profile>
      </bandwidth-scs-delay-state>
      <transport-delay>
        <t12-min>1000000000</t12-min>
        <t12-max>1000000000</t12-max>
        <t34-min>1000000000</t34-min>
        <t34-max>1000000000</t34-max>
      </transport-delay>
    </adaptive-delay-configuration>
    <beam-context-gap-period>333</beam-context-gap-period>
  </delay-management>)"s;
//</data>)"s;

constexpr const char *CONFIG_YANG_MODEL_PATH = "/opt/modeling/data-model/yang/published/o-ran/ru-fh/";

int main()
{
  const char *file_path = CONFIG_YANG_MODEL_PATH;
  std::optional<libyang::Context> ctx{std::in_place, std::nullopt, libyang::ContextOptions::NoYangLibrary | libyang::ContextOptions::DisableSearchCwd};
  ctx->setSearchDir(file_path);

  //	auto mod = ctx->loadModule("o-ran-usermgmt", std::nullopt);
  //  mod = ctx->loadModule("o-ran-uplane-conf", std::nullopt);
  auto mod = ctx->loadModule("o-ran-delay-management", std::nullopt);
  // mod = ctx->loadModule("o-ran-ecpri-delay", std::nullopt);

  if (mod.implemented())
  {
    std::cout << "mod.implemented" << std::endl;
  }

  std::optional<libyang::ChildInstanstiables> children;
  children = mod.childInstantiables();
  for (const auto &child : *children)
  {
    // actualPaths.emplace_back(child.path());

    std::cout << child.path() << std::endl;
    std::cout << child.nodeType() << std::endl;

    std::vector<std::string> actualPaths;
    for (const auto &it : ctx->findPath(child.path()).childrenDfs())
    {
      std::cout << "child path ==>" << it.path() << std::endl;
    }
  }

  //	auto node = ctx->newPath2("/o-ran-uplane-conf:duplex-scheme", "TDD");

  auto modules = ctx->modules();
  for (int i = 0; i < modules.size(); i++)
  {
    std::cout << "loaded module ==> " << modules.at(i).name() << std::endl;
  }

  ctx->parseModule(example_schema2, libyang::SchemaFormat::YANG);
  auto nodes = ctx->newPath2("/example-schema2:contWithTwoNodes/one", "1");
  std::cout << nodes.createdNode->path() << std::endl;
  std::cout << nodes.createdParent->path() << std::endl;

  //	auto node1 = ctx->newPath2("/o-ran-uplane-conf:user-plane-configuration/endpoint-types[id='0']/max-prb-per-symbol", "8");
  //	auto node2 = ctx->newPath2("/o-ran-uplane-conf:user-plane-configuration/endpoint-types[id='1']/max-prb-per-symbol", "273");
  //	//node1.createdNode->merge(*node2.createdNode);
  //	std::cout << node2.createdNode->findPath("/o-ran-uplane-conf:user-plane-configuration/endpoint-types[id='1']/max-prb-per-symbol")->asTerm().valueStr() << std::endl;

  auto root = ctx->parseData(delay, libyang::DataFormat::XML);

  auto str = *root->printStr(libyang::DataFormat::XML, libyang::PrintFlags::WithSiblings | libyang::PrintFlags::KeepEmptyCont);
  std::cout << str << std::endl;

  std::string path = "/o-ran-delay-management:delay-management/bandwidth-scs-delay-state[bandwidth='100000'][subcarrier-spacing='1000']/ru-delay-profile/ta3-max-ack";
  // std::string path = "/o-ran-delay-management:delay-management/bandwidth-scs-delay-state/ru-delay-profile/t2a-max-up";
  std::cout << "Read node t2a-max-up==>" << root->findPath(path.c_str())->asTerm().valueStr() << std::endl;

  auto leaf = std::optional{ctx->newPath("/o-ran-delay-management:delay-management/bandwidth-scs-delay-state[bandwidth='100000'][subcarrier-spacing='1000']/ru-delay-profile/ta3-max-ack", "6666")};
  root->merge(*leaf);

  std::cout << "Read again node t2a-max-up==>" << root->findPath(path.c_str())->asTerm().valueStr() << std::endl;
  return 0;
}
