// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#ifndef CK_AMD_WMMA_HPP
#define CK_AMD_WMMA_HPP

#include "data_type.hpp"

namespace ck {

// wave32 only
// src: fp16, dst: fp32
template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x16_f16_w32;

template <>
struct intrin_wmma_f32_16x16x16_f16_w32<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const half16_t& reg_a, const half16_t& reg_b, FloatC& reg_c)
    {
        reg_c.template AsType<float8_t>()(Number<0>{}) = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
            reg_a, reg_b, reg_c.template AsType<float8_t>()[Number<0>{}]);
    }
};

// src: bf16, dst: fp32
template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x16_bf16_w32;

template <>
struct intrin_wmma_f32_16x16x16_bf16_w32<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const bhalf16_t& reg_a, const bhalf16_t& reg_b, FloatC& reg_c)
    {
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
                reg_a, reg_b, reg_c.template AsType<float8_t>()[Number<0>{}]);
    }
};

// src: fp16, dst: fp16
template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f16_16x16x16_f16_w32;

template <>
struct intrin_wmma_f16_16x16x16_f16_w32<16, 16>
{
    template <class FloatC>
    __device__ static void
    Run(const half16_t& reg_a, const half16_t& reg_b, FloatC& reg_c, const bool opsel)
    {
        // opsel usage
        // false: D0.[0:15] = result
        // true : D0.[16:31]= result
        reg_c.template AsType<half16_t>()(Number<0>{}) = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(
            reg_a, reg_b, reg_c.template AsType<half16_t>()[Number<0>{}], opsel);
    }
};

// src: bf16, dst: bf32
template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_bf16_16x16x16_bf16_w32;

template <>
struct intrin_wmma_bf16_16x16x16_bf16_w32<16, 16>
{
    template <class FloatC>
    __device__ static void
    Run(const bhalf16_t& reg_a, const bhalf16_t& reg_b, FloatC& reg_c, const bool opsel)
    {
        // opsel usage
        // false: D0.[0:15] = result
        // true : D0.[16:31]= result
        reg_c.template AsType<bhalf16_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32(
                reg_a, reg_b, reg_c.template AsType<bhalf16_t>()[Number<0>{}], opsel);
    }
};

// src: iu8, dst: i32
template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_i32_16x16x16_iu8_w32;

template <>
struct intrin_wmma_i32_16x16x16_iu8_w32<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const bool neg_a,
                               const int8x16_t& reg_a,
                               const bool neg_b,
                               const int8x16_t& reg_b,
                               FloatC& reg_c,
                               const bool clamp)
    {
        reg_c.template AsType<int32x8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
                neg_a,
                bit_cast<int32x4_t>(reg_a),
                neg_b,
                bit_cast<int32x4_t>(reg_b),
                reg_c.template AsType<int32x8_t>()[Number<0>{}],
                clamp);
    }
};

#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
// src: iu4, dst: i32
template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_i32_16x16x16_iu4_w32;

template <>
struct intrin_wmma_i32_16x16x16_iu4_w32<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const bool neg_a,
                               const int4x16_t& reg_a,
                               const bool neg_b,
                               const int4x16_t& reg_b,
                               FloatC& reg_c,
                               const bool clamp)
    {
        reg_c.template AsType<int32x8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
                neg_a,
                bit_cast<int32x4_t>(reg_a),
                neg_b,
                bit_cast<int32x4_t>(reg_b),
                reg_c.template AsType<int32x8_t>()[Number<0>{}],
                clamp);
    }
};
#endif
} // namespace ck
#endif
