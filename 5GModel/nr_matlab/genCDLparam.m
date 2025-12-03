% SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
% SPDX-License-Identifier: Apache-2.0
%
% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
%
% http://www.apache.org/licenses/LICENSE-2.0
%
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.

cdl = nrCDLChannel(DelayProfile='CDL-A');
swapTransmitAndReceive(cdl);
CDLparam.DPA_A = [cdl.info.PathDelays/cdl.DelaySpread; cdl.info.AveragePathGains; cdl.info.AnglesAoD; cdl.info.AnglesAoA; cdl.info.AnglesZoD; cdl.info.AnglesZoA]';
CDLparam.PCP_A = [cdl.info.ClusterAngleSpreads, cdl.info.XPR];

cdl = nrCDLChannel(DelayProfile='CDL-B');
swapTransmitAndReceive(cdl);
CDLparam.DPA_B = [cdl.info.PathDelays/cdl.DelaySpread; cdl.info.AveragePathGains; cdl.info.AnglesAoD; cdl.info.AnglesAoA; cdl.info.AnglesZoD; cdl.info.AnglesZoA]';
CDLparam.PCP_B = [cdl.info.ClusterAngleSpreads, cdl.info.XPR];

cdl = nrCDLChannel(DelayProfile='CDL-C');
swapTransmitAndReceive(cdl);
CDLparam.DPA_C = [cdl.info.PathDelays/cdl.DelaySpread; cdl.info.AveragePathGains; cdl.info.AnglesAoD; cdl.info.AnglesAoA; cdl.info.AnglesZoD; cdl.info.AnglesZoA]';
CDLparam.PCP_C = [cdl.info.ClusterAngleSpreads, cdl.info.XPR];

cdl = nrCDLChannel(DelayProfile='CDL-D');
swapTransmitAndReceive(cdl);
CDLparam.DPA_D = [cdl.info.PathDelays/cdl.DelaySpread; cdl.info.AveragePathGains; cdl.info.AnglesAoD; cdl.info.AnglesAoA; cdl.info.AnglesZoD; cdl.info.AnglesZoA]';
CDLparam.PCP_D = [cdl.info.ClusterAngleSpreads, cdl.info.XPR];

cdl = nrCDLChannel(DelayProfile='CDL-E');
swapTransmitAndReceive(cdl);
CDLparam.DPA_E = [cdl.info.PathDelays/cdl.DelaySpread; cdl.info.AveragePathGains; cdl.info.AnglesAoD; cdl.info.AnglesAoA; cdl.info.AnglesZoD; cdl.info.AnglesZoA]';
CDLparam.PCP_E = [cdl.info.ClusterAngleSpreads, cdl.info.XPR];

save CDLparam CDLparam