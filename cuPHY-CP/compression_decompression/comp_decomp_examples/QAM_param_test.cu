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

#include <stdio.h>
#include <assert.h>

#include "QAM_param.cuh"

void test_list_shift()
{
    QamListParam p;
    p.set(QamListParam::MODCOMP_QAM256, 1, 0);
    assert(p.get_shift() == 0.0625f);
    assert(p.need_shift<0>() == true);
    assert(p.need_shift<1>() == false);
    assert(p.get_shift_mask(0xaaa, 0x555) == 0xaaa);
    assert(p.get_shift_mask(0xaaa, 0) == 0xaaa);
    assert(p.get_shift_mask(0, 0x555) == 0);
    assert(p.get_f2i_fact() == 8.0f);
    assert(p.get_i2f_fact() == 0.125f);

    p.set(QamListParam::MODCOMP_QAM64, 0, 1);
    assert(p.get_shift() == 0.125f);
    assert(p.need_shift<0>() == false);
    assert(p.need_shift<1>() == true);
    assert(p.get_shift_mask(0xaaa, 0x555) == 0x555);
    assert(p.get_shift_mask(0, 0x555) == 0x555);
    assert(p.get_shift_mask(0xaaa, 0) == 0);
    assert(p.get_f2i_fact() == 4.0f);
    assert(p.get_i2f_fact() == 0.25f);

    p.set(QamListParam::MODCOMP_QAM16, 1, 1);
    assert(p.get_shift() == 0.25f);
    assert(p.need_shift<0>() == true);
    assert(p.need_shift<1>() == true);
    assert(p.get_shift_mask(0xaaa, 0x555) == 0xfff);
    assert(p.get_shift_mask(0, 0x555) == 0x555);
    assert(p.get_shift_mask(0xaaa, 0) == 0xaaa);
    assert(p.get_f2i_fact() == 2.0f);
    assert(p.get_i2f_fact() == 0.5f);

    p.set(QamListParam::MODCOMP_QPSK, 0, 0);
    assert(p.get_shift() == 0.5f);
    assert(p.need_shift<0>() == false);
    assert(p.need_shift<1>() == false);
    assert(p.get_shift_mask(0xaaa, 0x555) == 0);
    assert(p.get_shift_mask(0, 0x555) == 0);
    assert(p.get_shift_mask(0xaaa, 0) == 0);
    assert(p.get_f2i_fact() == 1.0f);
    assert(p.get_i2f_fact() == 1.0f);

    p.set(QamListParam::MODCOMP_BPSK, 1, 0);
    assert(p.get_shift() == 0.0009765625f);
    assert(p.need_shift<0>() == true);
    assert(p.need_shift<1>() == false);
    assert(p.get_f2i_fact() == 2.0f);
    assert(p.get_i2f_fact() == 0.5f);
}

void test_list_width()
{
    QamListParam p;
    p.set(QamListParam::MODCOMP_QAM256, 1, 0);
    assert(p.get_qam_width() == QamListParam::MODCOMP_QAM256);

    p.set(QamListParam::MODCOMP_QAM64, 0, 1);
    assert(p.get_qam_width() == QamListParam::MODCOMP_QAM64);

    p.set(QamListParam::MODCOMP_QAM16, 1, 1);
    assert(p.get_qam_width() == QamListParam::MODCOMP_QAM16);

    p.set(QamListParam::MODCOMP_QPSK, 0, 0);
    assert(p.get_qam_width() == QamListParam::MODCOMP_QPSK);

    p.set(QamListParam::MODCOMP_BPSK, 1, 0);
    assert(p.get_qam_width() == QamListParam::MODCOMP_BPSK);
}

void test_list_bits()
{
    QamListParam p;
    p.set(QamListParam::MODCOMP_QAM256, 1, 0);
    assert(p.get_bits_per_element() == 8);

    p.set(QamListParam::MODCOMP_QAM64, 0, 1);
    assert(p.get_bits_per_element() == 6);

    p.set(QamListParam::MODCOMP_QAM16, 1, 1);
    assert(p.get_bits_per_element() == 4);

    p.set(QamListParam::MODCOMP_QPSK, 0, 0);
    assert(p.get_bits_per_element() == 2);

    p.set(QamListParam::MODCOMP_BPSK, 1, 0);
    assert(p.get_bits_per_element() == 4);
}

template <bool selective_sending = false>
void test_prb_bytes()
{
    QamListParam lp;
    QamPrbParam p;

    lp.set(QamListParam::MODCOMP_QAM256, 0, 0);
    p.set(0xfff, 0);
    assert(p.comp_bytes<selective_sending>(lp.get_bits_per_element()) == 12);
    p.set(0, 0xfff);
    assert(p.comp_bytes<selective_sending>(lp.get_bits_per_element()) == 12);
    p.set(0xaaa, 0x555);
    assert(p.comp_bytes<selective_sending>(lp.get_bits_per_element()) == 12);
    p.set(0xaaa, 0);
    assert(p.comp_bytes<selective_sending>(lp.get_bits_per_element()) == (selective_sending ? 6 : 12));
    p.set(0, 0xaaa);
    assert(p.comp_bytes<selective_sending>(lp.get_bits_per_element()) == (selective_sending ? 6 : 12));
    p.set(1, 0);
    assert(p.comp_bytes<selective_sending>(lp.get_bits_per_element()) == (selective_sending ? 1 : 12));
    p.set(0, 1);
    assert(p.comp_bytes<selective_sending>(lp.get_bits_per_element()) == (selective_sending ? 1 : 12));

    lp.set(QamListParam::MODCOMP_QAM64, 0, 0);
    p.set(0xfff, 0);
    assert(p.comp_bytes<selective_sending>(lp.get_bits_per_element()) == 9);
    p.set(0, 0xfff);
    assert(p.comp_bytes<selective_sending>(lp.get_bits_per_element()) == 9);
    p.set(0xaaa, 0x555);
    assert(p.comp_bytes<selective_sending>(lp.get_bits_per_element()) == 9);
    p.set(0xaaa, 0);
    assert(p.comp_bytes<selective_sending>(lp.get_bits_per_element()) == (selective_sending ? 5 : 9));
    p.set(0, 0xaaa);
    assert(p.comp_bytes<selective_sending>(lp.get_bits_per_element()) == (selective_sending ? 5 : 9));
    p.set(1, 0);
    assert(p.comp_bytes<selective_sending>(lp.get_bits_per_element()) == (selective_sending ? 1 : 9));
    p.set(0, 1);
    assert(p.comp_bytes<selective_sending>(lp.get_bits_per_element()) == (selective_sending ? 1 : 9));

    lp.set(QamListParam::MODCOMP_QAM16, 0, 0);
    p.set(0xfff, 0);
    assert(p.comp_bytes<selective_sending>(lp.get_bits_per_element()) == 6);
    p.set(0, 0xfff);
    assert(p.comp_bytes<selective_sending>(lp.get_bits_per_element()) == 6);
    p.set(0xaaa, 0x555);
    assert(p.comp_bytes<selective_sending>(lp.get_bits_per_element()) == 6);
    p.set(0xaaa, 0);
    assert(p.comp_bytes<selective_sending>(lp.get_bits_per_element()) == (selective_sending ? 3 : 6));
    p.set(0, 0xaaa);
    assert(p.comp_bytes<selective_sending>(lp.get_bits_per_element()) == (selective_sending ? 3 : 6));
    p.set(1, 0);
    assert(p.comp_bytes<selective_sending>(lp.get_bits_per_element()) == (selective_sending ? 1 : 6));
    p.set(0, 1);
    assert(p.comp_bytes<selective_sending>(lp.get_bits_per_element()) == (selective_sending ? 1 : 6));

    lp.set(QamListParam::MODCOMP_QPSK, 0, 0);
    p.set(0xfff, 0);
    assert(p.comp_bytes<selective_sending>(lp.get_bits_per_element()) == 3);
    p.set(0, 0xfff);
    assert(p.comp_bytes<selective_sending>(lp.get_bits_per_element()) == 3);
    p.set(0xaaa, 0x555);
    assert(p.comp_bytes<selective_sending>(lp.get_bits_per_element()) == 3);
    p.set(0xaaa, 0);
    assert(p.comp_bytes<selective_sending>(lp.get_bits_per_element()) == (selective_sending ? 2 : 3));
    p.set(0, 0xaaa);
    assert(p.comp_bytes<selective_sending>(lp.get_bits_per_element()) == (selective_sending ? 2 : 3));
    p.set(1, 0);
    assert(p.comp_bytes<selective_sending>(lp.get_bits_per_element()) == (selective_sending ? 1 : 3));
    p.set(0, 1);
    assert(p.comp_bytes<selective_sending>(lp.get_bits_per_element()) == (selective_sending ? 1 : 3));

    lp.set(QamListParam::MODCOMP_BPSK, 0, 0);
    p.set(0xfff, 0);
    assert(p.comp_bytes<selective_sending>(lp.get_bits_per_element()) == 6);
    p.set(0, 0xfff);
    assert(p.comp_bytes<selective_sending>(lp.get_bits_per_element()) == 6);
    p.set(0xaaa, 0x555);
    assert(p.comp_bytes<selective_sending>(lp.get_bits_per_element()) == 6);
    p.set(0xaaa, 0);
    assert(p.comp_bytes<selective_sending>(lp.get_bits_per_element()) == (selective_sending ? 3 : 6));
    p.set(0, 0xaaa);
    assert(p.comp_bytes<selective_sending>(lp.get_bits_per_element()) == (selective_sending ? 3 : 6));
    p.set(1, 0);
    assert(p.comp_bytes<selective_sending>(lp.get_bits_per_element()) == (selective_sending ? 1 : 6));
    p.set(0, 1);
    assert(p.comp_bytes<selective_sending>(lp.get_bits_per_element()) == (selective_sending ? 1 : 6));
}

template <int m>
void set_prb_mask_m(QamPrbParam &p, uint32_t mask0, uint32_t mask1)
{
    if constexpr (m)
        p.set(mask1, mask0);
    else
        p.set(mask0, mask1);
}

template <int m>
void test_prb_masks()
{
    static_assert(m == 0 || m == 1);
    QamPrbParam p;

    set_prb_mask_m<m>(p, 0, 0);
    for (int i = 0; i < 12; i++)
    {
        assert(p.mask_on<0>(i) == false);
        assert(p.mask_on<1>(i) == false);
    }
    assert(p.get_mask<m>() == 0);
    assert(p.get_mask<m ^ 1>() == 0);
    assert(p.get_mask() == 0);

    set_prb_mask_m<m>(p, 0xfff, 0);
    for (int i = 0; i < 12; i++)
    {
        assert(p.mask_on<m>(i) == true);
        assert(p.mask_on<m ^ 1>(i) == false);
    }
    assert(p.get_mask<m>() == 0xfff);
    assert(p.get_mask<m ^ 1>() == 0);
    assert(p.get_mask() == 0xfff);

    set_prb_mask_m<m>(p, 0xaaa, 0);
    for (int i = 0; i < 12; i += 2)
    {
        assert(p.mask_on<m>(i) == false);
        assert(p.mask_on<m>(i + 1) == true);
        assert(p.mask_on<m ^ 1>(i) == false);
        assert(p.mask_on<m ^ 1>(i + 1) == false);
    }
    assert(p.get_mask<m>() == 0xaaa);
    assert(p.get_mask<m ^ 1>() == 0);
    assert(p.get_mask() == 0xaaa);

    set_prb_mask_m<m>(p, 0x555, 0);
    for (int i = 0; i < 12; i += 2)
    {
        assert(p.mask_on<m>(i) == true);
        assert(p.mask_on<m>(i + 1) == false);
        assert(p.mask_on<m ^ 1>(i) == false);
        assert(p.mask_on<m ^ 1>(i + 1) == false);
    }
    assert(p.get_mask<m>() == 0x555);
    assert(p.get_mask<m ^ 1>() == 0);
    assert(p.get_mask() == 0x555);

    set_prb_mask_m<m>(p, 0x555, 0xaaa);
    for (int i = 0; i < 12; i += 2)
    {
        assert(p.mask_on<m>(i) == true);
        assert(p.mask_on<m>(i + 1) == false);
        assert(p.mask_on<m ^ 1>(i) == false);
        assert(p.mask_on<m ^ 1>(i + 1) == true);
    }
    assert(p.get_mask<m>() == 0x555);
    assert(p.get_mask<m ^ 1>() == 0xaaa);
    assert(p.get_mask() == 0xfff);
}

// void test_shift()
// {
//     QamParam p;
//     p.set(QamParam::MODCOMP_QAM256, 1, 0, 0xaaa);
//     assert(p.get_shift() == -0.0625f);
//     assert(p.need_shift<0>() == true);
//     assert(p.need_shift<1>() == false);
//     assert(p.get_shift_mask() == 0xaaa);

//     p.set(QamParam::MODCOMP_QAM64, 0, 0, 0xaaa, 1, 0, 0x555);
//     assert(p.get_shift() == -0.125f);
//     assert(p.need_shift<0>() == false);
//     assert(p.need_shift<1>() == true);
//     assert(p.get_shift_mask() == 0x555);

//     p.set(QamParam::MODCOMP_QAM16, 1, 0, 0xaaa, 0, 0, 0x555);
//     assert(p.get_shift() == -0.25f);
//     assert(p.need_shift<0>() == true);
//     assert(p.need_shift<1>() == false);
//     assert(p.get_shift_mask() == 0xaaa);

//     p.set(QamParam::MODCOMP_QPSK, 1, 0, 0xa00, 1, 0, 0x005);
//     assert(p.get_shift() == -0.5f);
//     assert(p.need_shift<0>() == true);
//     assert(p.need_shift<1>() == true);
//     assert(p.get_shift_mask() == 0xa05);

//     p.set(QamParam::MODCOMP_BPSK, 1, 0, 0xa00, 0, 0, 0x005);
//     assert(p.get_shift() == -0.0009765625f);
//     assert(p.need_shift<0>() == true);
//     assert(p.need_shift<1>() == false);
//     assert(p.get_shift_mask() == 0xa00);
// }

// template <int m>
// void set_mask_m(QamParam &p, uint mask0, uint mask1)
// {
//     if constexpr (m == 0)
//         p.set(QamParam::MODCOMP_QAM16, 0, 0, mask0, 0, 0, mask1);
//     else
//         p.set(QamParam::MODCOMP_QAM16, 0, 0, mask1, 0, 0, mask0);
// }

// template <int m>
// void test_masks()
// {
//     static_assert(m == 0 || m == 1);
//     QamParam p;

//     set_mask_m<m>(p, 0, 0);
//     for (int i = 0; i < 12; i++)
//     {
//         assert(p.mask_on<0>(i) == false);
//         assert(p.mask_on<1>(i) == false);
//     }
//     assert(p.get_mask<m>() == 0);
//     assert(p.get_mask<m ^ 1>() == 0);
//     assert(p.get_mask() == 0);

//     set_mask_m<m>(p, 0xfff, 0);
//     for (int i = 0; i < 12; i++)
//     {
//         assert(p.mask_on<m>(i) == true);
//         assert(p.mask_on<m ^ 1>(i) == false);
//     }
//     assert(p.get_mask<m>() == 0xfff);
//     assert(p.get_mask<m ^ 1>() == 0);
//     assert(p.get_mask() == 0xfff);

//     set_mask_m<m>(p, 0xaaa, 0);
//     for (int i = 0; i < 12; i += 2)
//     {
//         assert(p.mask_on<m>(i) == false);
//         assert(p.mask_on<m>(i + 1) == true);
//         assert(p.mask_on<m ^ 1>(i) == false);
//         assert(p.mask_on<m ^ 1>(i + 1) == false);
//     }
//     assert(p.get_mask<m>() == 0xaaa);
//     assert(p.get_mask<m ^ 1>() == 0);
//     assert(p.get_mask() == 0xaaa);

//     set_mask_m<m>(p, 0x555, 0);
//     for (int i = 0; i < 12; i += 2)
//     {
//         assert(p.mask_on<m>(i) == true);
//         assert(p.mask_on<m>(i + 1) == false);
//         assert(p.mask_on<m ^ 1>(i) == false);
//         assert(p.mask_on<m ^ 1>(i + 1) == false);
//     }
//     assert(p.get_mask<m>() == 0x555);
//     assert(p.get_mask<m ^ 1>() == 0);
//     assert(p.get_mask() == 0x555);

//     set_mask_m<m>(p, 0x555, 0xaaa);
//     for (int i = 0; i < 12; i += 2)
//     {
//         assert(p.mask_on<m>(i) == true);
//         assert(p.mask_on<m>(i + 1) == false);
//         assert(p.mask_on<m ^ 1>(i) == false);
//         assert(p.mask_on<m ^ 1>(i + 1) == true);
//     }
//     assert(p.get_mask<m>() == 0x555);
//     assert(p.get_mask<m ^ 1>() == 0xaaa);
//     assert(p.get_mask() == 0xfff);
// }

// void test_bytes()
// {
//     QamParam p;
//     p.set(QamParam::MODCOMP_QAM256, 1, 0, 0xfff);
//     assert(p.comp_bytes() == 12);
//     p.set(QamParam::MODCOMP_QAM256, 1, 0, 0, 0, 0, 0xfff);
//     assert(p.comp_bytes() == 12);
//     p.set(QamParam::MODCOMP_QAM256, 1, 0, 0xaaa, 0, 0, 0x555);
//     assert(p.comp_bytes() == 12);
//     p.set(QamParam::MODCOMP_QAM256, 1, 0, 0xaaa);
//     assert(p.comp_bytes() == 6);
//     p.set(QamParam::MODCOMP_QAM256, 1, 0, 0x010, 0, 0, 0);
//     assert(p.comp_bytes() == 1);
//     p.set(QamParam::MODCOMP_QAM256, 1, 0, 0, 0, 0, 0x010);
//     assert(p.comp_bytes() == 1);

//     p.set(QamParam::MODCOMP_QAM64, 1, 0, 0xfff);
//     assert(p.comp_bytes() == 9);
//     p.set(QamParam::MODCOMP_QAM64, 1, 0, 0, 0, 0, 0xfff);
//     assert(p.comp_bytes() == 9);
//     p.set(QamParam::MODCOMP_QAM64, 1, 0, 0xaaa, 0, 0, 0x555);
//     assert(p.comp_bytes() == 9);
//     p.set(QamParam::MODCOMP_QAM64, 1, 0, 0xaaa);
//     assert(p.comp_bytes() == 5);
//     p.set(QamParam::MODCOMP_QAM64, 1, 0, 0x010, 0, 0, 0);
//     assert(p.comp_bytes() == 1);
//     p.set(QamParam::MODCOMP_QAM64, 1, 0, 0, 0, 0, 0x010);
//     assert(p.comp_bytes() == 1);

//     p.set(QamParam::MODCOMP_QAM16, 1, 0, 0xfff);
//     assert(p.comp_bytes() == 6);
//     p.set(QamParam::MODCOMP_QAM16, 1, 0, 0, 0, 0, 0xfff);
//     assert(p.comp_bytes() == 6);
//     p.set(QamParam::MODCOMP_QAM16, 1, 0, 0xaaa, 0, 0, 0x555);
//     assert(p.comp_bytes() == 6);
//     p.set(QamParam::MODCOMP_QAM16, 1, 0, 0xaaa);
//     assert(p.comp_bytes() == 3);
//     p.set(QamParam::MODCOMP_QAM16, 1, 0, 0x010, 0, 0, 0);
//     assert(p.comp_bytes() == 1);
//     p.set(QamParam::MODCOMP_QAM16, 1, 0, 0, 0, 0, 0x010);
//     assert(p.comp_bytes() == 1);

//     p.set(QamParam::MODCOMP_QPSK, 1, 0, 0xfff);
//     assert(p.comp_bytes() == 3);
//     p.set(QamParam::MODCOMP_QPSK, 1, 0, 0, 0, 0, 0xfff);
//     assert(p.comp_bytes() == 3);
//     p.set(QamParam::MODCOMP_QPSK, 1, 0, 0xaaa, 0, 0, 0x555);
//     assert(p.comp_bytes() == 3);
//     p.set(QamParam::MODCOMP_QPSK, 1, 0, 0xaaa);
//     assert(p.comp_bytes() == 2);
//     p.set(QamParam::MODCOMP_QPSK, 1, 0, 0x010, 0, 0, 0);
//     assert(p.comp_bytes() == 1);
//     p.set(QamParam::MODCOMP_QPSK, 1, 0, 0, 0, 0, 0x010);
//     assert(p.comp_bytes() == 1);

//     p.set(QamParam::MODCOMP_BPSK, 1, 0, 0xfff);
//     assert(p.comp_bytes() == 6);
//     p.set(QamParam::MODCOMP_BPSK, 1, 0, 0, 0, 0, 0xfff);
//     assert(p.comp_bytes() == 6);
//     p.set(QamParam::MODCOMP_BPSK, 1, 0, 0xaaa, 0, 0, 0x555);
//     assert(p.comp_bytes() == 6);
//     p.set(QamParam::MODCOMP_BPSK, 1, 0, 0xaaa);
//     assert(p.comp_bytes() == 3);
//     p.set(QamParam::MODCOMP_BPSK, 1, 0, 0x010, 0, 0, 0);
//     assert(p.comp_bytes() == 1);
//     p.set(QamParam::MODCOMP_BPSK, 1, 0, 0, 0, 0, 0x010);
//     assert(p.comp_bytes() == 1);
// }

// template <int m>
// void set_modscaler_m(QamParam &p, uint scaler)
// {
//     if constexpr (m == 0)
//         p.set(QamParam::MODCOMP_QAM256, 0, scaler);
//     else
//         p.set(QamParam::MODCOMP_QAM256, 0, 0, 0, 0, scaler);
// }

// template <int m>
// void test_scaler()
// {
//     QamParam p;
//     set_modscaler_m<m>(p, 0);
//     assert(p.get_modscaler_as_float<m>() == 0.0f); // 2^0 x 0
//     set_modscaler_m<m>(p, 0x4000);
//     assert(p.get_modscaler_as_float<m>() == 0.0f); // 2^-8 x 0
//     set_modscaler_m<m>(p, 0x400);
//     assert(p.get_modscaler_as_float<m>() == 0.5f); // 2^0 x 0.5
//     set_modscaler_m<m>(p, 0x200);
//     assert(p.get_modscaler_as_float<m>() == 0.25f); // 2^0 x 0.25
//     set_modscaler_m<m>(p, 0xc00);
//     assert(p.get_modscaler_as_float<m>() == 0.25f); // 2^-1 x 0.5
//     set_modscaler_m<m>(p, 1);
//     assert(p.get_modscaler_as_float<m>() == .00048828125f); // 2^-11
//     set_modscaler_m<m>(p, 0x801);
//     assert(p.get_modscaler_as_float<m>() == .000244140625f); // 2^-12
//     set_modscaler_m<m>(p, 0x7ff);
//     assert(p.get_modscaler_as_float<m>() == 1.0f - .00048828125f); // 1 - 2^-11
//     set_modscaler_m<m>(p, 0xfff);
//     assert(p.get_modscaler_as_float<m>() == .5f - .000244140625f); // .5 - 2^-12
// }

int main()
{
    test_list_shift();
    test_list_width();
    test_list_bits();
    test_prb_bytes<false>();
    test_prb_bytes<true>();
    test_prb_masks<0>();
    test_prb_masks<1>();

    // test_shift();
    // test_masks<0>();
    // test_masks<1>();
    // test_bytes();
    // test_scaler<0>();
    // test_scaler<1>();
    return 0;
}