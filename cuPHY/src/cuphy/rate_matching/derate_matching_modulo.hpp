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

#pragma once

// Calculate the derate match output index per algorithm in 38.212 5.4.2.1 using fast non-iterative method
inline __host__ __device__ int derate_match_fast_calc_modulo(int inIdx, int Kd, int F, int k0, int Ncb)
{
    int outIdx;
    int K = Kd + F;

    // layout: linear view of the circular buffer
    // 0                          Kd          K=Kd+F                   Ncb-1
    // |--------------------------|===========|----------------------------|
    //     Systematic bits            HOLE          Parity bits
    //         (keep)                (skip)            (keep)
    //=============================================================================================
    // Scenario 0: We start before the hole
    //        k0
    //        v
    //    |--->----->-----|===========|----------------------------|
    //                       ^ if you land here (≥Kd and <K), skip ahead +F
    //    wrap: outIdx -= Ncb; then if in hole, +F again
    if (k0 < Kd)
    {
        outIdx = k0 + inIdx;

        while (outIdx >= Ncb)
        {
            outIdx -= Ncb;
            outIdx += F;
        }
        if (outIdx >= Kd)
        {
            outIdx += F;
        }
        if (outIdx >= Ncb)
        {
            outIdx -= Ncb;
        }

    }

     // Scenario 1: We start after the hole
    //                                           k0 is greater than K
    //                                           v
    // |--------------------------|===========|--->----->---------|
    // wrap : (outIdx -= Ncb); if that lands in hole, +F to skip it
     else if (k0 > K)
     {
        outIdx = k0 + inIdx;

        while (outIdx >= Ncb)
        {
            outIdx -= Ncb;
            if (outIdx >= Kd)
            {
                outIdx += F;
            }
        }
    }
    // Scenario 2: We start in the hole
    //                       k0 (in hole)
    //                       v
    // |----------------|====k0======|----------------------------|
    //                      |<-Fmin->|
    // jump out first: outIdx += Fmin, then proceed like Scenario 1
    else //if ((k0 >= Kd) && (k0 <= K))
    {
        int Fmin = K-k0;
        outIdx = k0 + inIdx + Fmin;

        while (outIdx >= Ncb)
        {
            outIdx -= Ncb;
            if (outIdx >= Kd)
            {
                outIdx += F;
            }
        }
    }

    return outIdx;
}
