// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_batched_mha_bwd_xdl_cshuffle_qloop_light_v2.hpp"
#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using F16   = ck::half_t;
using BF16  = ck::bhalf_t;
using F32   = float;
using U16   = unsigned short;
using INT32 = int32_t;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using Scale       = ck::tensor_operation::element_wise::Scale;

using QKVElementOp = PassThrough;
using YElementOp   = PassThrough;

using InputDataType    = F16;
using OutputDataType   = F16;
using GemmDataType     = F16;
using AccDataType      = F32;
using ShuffleDataType  = F32;
using LSEDataType      = F32;
using ZDataType        = U16; // INT32
using Acc0BiasDataType = ck::Tuple<>;
using Acc1BiasDataType = ck::Tuple<>;
using DDataType        = F32;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using Acc0BiasDataType = ck::Tuple<>;
using Acc1BiasDataType = ck::Tuple<>;

// static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;
static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::MNKOPadding;

// static constexpr auto TensorDefault =
// ck::tensor_operation::device::TensorSpecialization::Default;
static constexpr auto TensorSpecQ   = ck::tensor_operation::device::TensorSpecialization::Default;
static constexpr auto TensorSpecK   = ck::tensor_operation::device::TensorSpecialization::Default;
static constexpr auto TensorSpecV   = ck::tensor_operation::device::TensorSpecialization::Default;
static constexpr auto TensorSpecY   = ck::tensor_operation::device::TensorSpecialization::Default;
static constexpr bool Deterministic = false;

static constexpr ck::index_t CShuffleBlockTransferScalarPerVector_NPerBlock = 8;

template <index_t NumDimG,
          index_t NumDimM,
          index_t NumDimN,
          index_t NumDimK,
          index_t NumDimO,
          MaskingSpecialization MaskingSpec>
using device_batched_mha_bwd_qloop_light_v2_f16_f16_instances =
    std::tuple<
        // clang-format off
    // ##############################################################################################|  NumDimG| NumDimM| NumDimN| NumDimK| NumDimO| InputDataType| OutputDataType| GemmDataType| ZDataType| LSEDataType| DDataType| Acc0BiasDataType| Acc1BiasDataType|     GemmAcc|        CShuffle|            A|            B|         Acc|           B1|           C|           GEMM|   ATensorSpec|  B0TensorSpec|  B1TensorSpec|   CTensorSpec| NumGemmK| Block| Gemm01| Gemm0| Gemm0| Gemm1| Gemm1| AK1| BK1| B1K1| MPer| NPer| Gemm0| Gemm0| Gemm1| Gemm2|YDotYGrad|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  B0BlockTransfer| B0BlockTransfer| B0BlockTransfer| B0BlockTransfer| B0BlockTransfer| B0BlockTransfer| B0BlockLds|  B1BlockTransfer| B1BlockTransfer| B1BlockTransfer| B1BlockTransfer| B1BlockTransfer| B1BlockTransfer| B1BlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths| CShuffleBlockTransferScalarPerVector_NPerBlock| MaskingSpec| Deterministic|
    // ##############################################################################################|         |        |        |        |        |              |               |             |          |            |          |                 |                 |    DataType|        DataType|  Elementwise|  Elementwise| Elementwise|  Elementwise| Elementwise| Specialization|              |              |              |              | Prefetch|  Size|   MPer|  NPer|  KPer|  NPer|  KPer|    |    |     |  XDL|  XDL|  MXdl|  NXdl|  NXdl|  NXdl|     KPer|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|    ThreadCluster|   ThreadCluster|  SrcAccessOrder|    SrcVectorDim|       SrcScalar|       DstScalar|  AddExtraN|    ThreadCluster|   ThreadCluster|  SrcAccessOrder|    SrcVectorDim|       SrcScalar|       DstScalar|  AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl|                                               |            |              |
    // ##############################################################################################|         |        |        |        |        |              |               |             |          |            |          |                 |                 |            |                |    Operation|    Operation|   Operation|    Operation|   Operation|               |              |              |              |              |    Stage|      |  Block| Block| Block| Block| Block|    |    |     |     |     |   Per|   Per|   Per|   Per|    Block| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          |  Lengths_K0_N_K1|    ArrangeOrder|                |                |       PerVector|    PerVector_K1|           |  Lengths_K0_N_K1|    ArrangeOrder|                |                |       PerVector|    PerVector_K1|           |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|                                               |            |              |
    // ##############################################################################################|         |        |        |        |        |              |               |             |          |            |          |                 |                 |            |                |             |             |            |             |            |               |              |              |              |              |         |      |       |      |      |      |      |    |    |     |     |     |  Wave|  Wave|  Wave|  Wave|         |                |               |               |               |               |               |          |                 |                |                |                |                |                |           |                 |                |                |                |                |                |           |            |            |                             |                                               |            |              |
    // ck::tensor_operation::device::DeviceBatchedMultiheadAttentionBackward_Qloop_Xdl_CShuffle_Light_V2<  NumDimG, NumDimM, NumDimN, NumDimK, NumDimO, InputDataType, OutputDataType, GemmDataType, ZDataType, LSEDataType, DDataType, Acc0BiasDataType, Acc1BiasDataType, AccDataType, ShuffleDataType, QKVElementOp, QKVElementOp,       Scale, QKVElementOp,  YElementOp,       GemmSpec,   TensorSpecQ,   TensorSpecK,   TensorSpecV,   TensorSpecY,        1,   256,    128,   128,    64,   128,    32,   8,   8,    2,   32,   32,     4,     1,     4,     2,       64,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true,      S<4, 64, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               8,               8,       true,      S<8, 32, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false,           1,           4,               S<1, 32, 1, 8>, CShuffleBlockTransferScalarPerVector_NPerBlock, MaskingSpec, Deterministic>;               
    // ck::tensor_operation::device::DeviceBatchedMultiheadAttentionBackward_Qloop_Xdl_CShuffle_Light_V2<  NumDimG, NumDimM, NumDimN, NumDimK, NumDimO, InputDataType, OutputDataType, GemmDataType, ZDataType, LSEDataType, DDataType, Acc0BiasDataType, Acc1BiasDataType, AccDataType, ShuffleDataType, QKVElementOp, QKVElementOp,       Scale, QKVElementOp,  YElementOp,       GemmSpec,   TensorSpecQ,   TensorSpecK,   TensorSpecV,   TensorSpecY,        1,   256,    128,   128,    64,   128,    32,   8,   8,    2,   32,   32,     4,     1,     4,     1,       64,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true,      S<4, 64, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               8,               8,       true,      S<8, 32, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false,           1,           4,               S<1, 32, 1, 8>, CShuffleBlockTransferScalarPerVector_NPerBlock, MaskingSpec, Deterministic>;               
    // ck::tensor_operation::device::DeviceBatchedMultiheadAttentionBackward_Qloop_Xdl_CShuffle_Light_V2<  NumDimG, NumDimM, NumDimN, NumDimK, NumDimO, InputDataType, OutputDataType, GemmDataType, ZDataType, LSEDataType, DDataType, Acc0BiasDataType, Acc1BiasDataType, AccDataType, ShuffleDataType, QKVElementOp, QKVElementOp,       Scale, QKVElementOp,  YElementOp,       GemmSpec,   TensorSpecQ,   TensorSpecK,   TensorSpecV,   TensorSpecY,        1,   256,    128,   128,    32,   128,    32,   8,   8,    2,   32,   32,     4,     1,     4,     2,       64,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true,      S<4, 64, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               8,               8,       true,      S<8, 32, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false,           1,           4,               S<1, 32, 1, 8>, CShuffleBlockTransferScalarPerVector_NPerBlock, MaskingSpec, Deterministic>;               
    // ck::tensor_operation::device::DeviceBatchedMultiheadAttentionBackward_Qloop_Xdl_CShuffle_Light_V2<  NumDimG, NumDimM, NumDimN, NumDimK, NumDimO, InputDataType, OutputDataType, GemmDataType, ZDataType, LSEDataType, DDataType, Acc0BiasDataType, Acc1BiasDataType, AccDataType, ShuffleDataType, QKVElementOp, QKVElementOp,       Scale, QKVElementOp,  YElementOp,       GemmSpec,   TensorSpecQ,   TensorSpecK,   TensorSpecV,   TensorSpecY,        1,   256,    128,   128,    32,   128,    32,   8,   8,    2,   32,   32,     4,     1,     4,     4,       64,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true,      S<4, 64, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               8,               8,       true,      S<8, 32, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false,           1,           4,               S<1, 32, 1, 8>, CShuffleBlockTransferScalarPerVector_NPerBlock, MaskingSpec, Deterministic>;               
    // ck::tensor_operation::device::DeviceBatchedMultiheadAttentionBackward_Qloop_Xdl_CShuffle_Light_V2<  NumDimG, NumDimM, NumDimN, NumDimK, NumDimO, InputDataType, OutputDataType, GemmDataType, ZDataType, LSEDataType, DDataType, Acc0BiasDataType, Acc1BiasDataType, AccDataType, ShuffleDataType, QKVElementOp, QKVElementOp,       Scale, QKVElementOp,  YElementOp,       GemmSpec,   TensorSpecQ,   TensorSpecK,   TensorSpecV,   TensorSpecY,        1,   256,    128,   128,    32,   128,    32,   8,   8,    2,   32,   32,     4,     1,     4,     1,       64,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true,      S<4, 64, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               8,               8,       true,      S<8, 32, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false,           1,           4,               S<1, 32, 1, 8>, CShuffleBlockTransferScalarPerVector_NPerBlock, MaskingSpec, Deterministic>;               

    // ck::tensor_operation::device::DeviceBatchedMultiheadAttentionBackward_Qloop_Xdl_CShuffle_Light_V2<  NumDimG, NumDimM, NumDimN, NumDimK, NumDimO, InputDataType, OutputDataType, GemmDataType, ZDataType, LSEDataType, DDataType, Acc0BiasDataType, Acc1BiasDataType, AccDataType, ShuffleDataType, QKVElementOp, QKVElementOp,       Scale, QKVElementOp,  YElementOp,       GemmSpec,   TensorSpecQ,   TensorSpecK,   TensorSpecV,   TensorSpecY,        1,   256,     64,   128,    64,   128,    32,   8,   8,    2,   32,   32,     2,     1,     4,     1,       64,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true,      S<4, 64, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               8,               8,       true,      S<8, 32, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false,           1,           4,               S<1, 32, 1, 8>, CShuffleBlockTransferScalarPerVector_NPerBlock, MaskingSpec, Deterministic>;               
    // ck::tensor_operation::device::DeviceBatchedMultiheadAttentionBackward_Qloop_Xdl_CShuffle_Light_V2<  NumDimG, NumDimM, NumDimN, NumDimK, NumDimO, InputDataType, OutputDataType, GemmDataType, ZDataType, LSEDataType, DDataType, Acc0BiasDataType, Acc1BiasDataType, AccDataType, ShuffleDataType, QKVElementOp, QKVElementOp,       Scale, QKVElementOp,  YElementOp,       GemmSpec,   TensorSpecQ,   TensorSpecK,   TensorSpecV,   TensorSpecY,        1,   256,     64,   128,    64,   128,    32,   8,   8,    2,   32,   32,     2,     1,     4,     2,       64,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true,      S<4, 64, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               8,               8,       true,      S<8, 32, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false,           1,           4,               S<1, 32, 1, 8>, CShuffleBlockTransferScalarPerVector_NPerBlock, MaskingSpec, Deterministic>;               
    // ck::tensor_operation::device::DeviceBatchedMultiheadAttentionBackward_Qloop_Xdl_CShuffle_Light_V2<  NumDimG, NumDimM, NumDimN, NumDimK, NumDimO, InputDataType, OutputDataType, GemmDataType, ZDataType, LSEDataType, DDataType, Acc0BiasDataType, Acc1BiasDataType, AccDataType, ShuffleDataType, QKVElementOp, QKVElementOp,       Scale, QKVElementOp,  YElementOp,       GemmSpec,   TensorSpecQ,   TensorSpecK,   TensorSpecV,   TensorSpecY,        1,   256,     64,   128,    32,   128,    32,   8,   8,    2,   32,   32,     2,     1,     4,     1,       64,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true,      S<4, 64, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               8,               8,       true,      S<8, 32, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false,           1,           4,               S<1, 32, 1, 8>, CShuffleBlockTransferScalarPerVector_NPerBlock, MaskingSpec, Deterministic>;               
    ck::tensor_operation::device::DeviceBatchedMultiheadAttentionBackward_Qloop_Xdl_CShuffle_Light_V2<  NumDimG, NumDimM, NumDimN, NumDimK, NumDimO, InputDataType, OutputDataType, GemmDataType, ZDataType, LSEDataType, DDataType, Acc0BiasDataType, Acc1BiasDataType, AccDataType, ShuffleDataType, QKVElementOp, QKVElementOp,       Scale, QKVElementOp,  YElementOp,       GemmSpec,   TensorSpecQ,   TensorSpecK,   TensorSpecV,   TensorSpecY,        1,   256,     64,   128,   128,   128,    32,   8,   8,    2,   32,   32,     2,     1,     4,     1,       64,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true,      S<4, 64, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               8,               8,       true,      S<8, 32, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false,           1,           4,               S<1, 32, 1, 8>, CShuffleBlockTransferScalarPerVector_NPerBlock, MaskingSpec, Deterministic>         
    // ck::tensor_operation::device::DeviceBatchedMultiheadAttentionBackward_Qloop_Xdl_CShuffle_Light_V2<  NumDimG, NumDimM, NumDimN, NumDimK, NumDimO, InputDataType, OutputDataType, GemmDataType, ZDataType, LSEDataType, DDataType, Acc0BiasDataType, Acc1BiasDataType, AccDataType, ShuffleDataType, QKVElementOp, QKVElementOp,       Scale, QKVElementOp,  YElementOp,       GemmSpec,   TensorSpecQ,   TensorSpecK,   TensorSpecV,   TensorSpecY,        1,   256,     32,   128,   128,   128,    32,   8,   8,    2,   32,   32,     1,     1,     4,     1,       64,     S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true,      S<4, 64, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               8,               8,       true,      S<8, 32, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false,           1,           4,               S<1, 32, 1, 8>, CShuffleBlockTransferScalarPerVector_NPerBlock, MaskingSpec, Deterministic>;               
    // ck::tensor_operation::device::DeviceBatchedMultiheadAttentionBackward_Qloop_Xdl_CShuffle_Light_V2<  NumDimG, NumDimM, NumDimN, NumDimK, NumDimO, InputDataType, OutputDataType, GemmDataType, ZDataType, LSEDataType, DDataType, Acc0BiasDataType, Acc1BiasDataType, AccDataType, ShuffleDataType, QKVElementOp, QKVElementOp,       Scale, QKVElementOp,  YElementOp,       GemmSpec,   TensorSpecQ,   TensorSpecK,   TensorSpecV,   TensorSpecY,        1,   256,     32,   128,    64,   128,    32,   8,   8,    2,   32,   32,     1,     1,     4,     1,       64,     S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true,      S<4, 64, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               8,               8,       true,      S<8, 32, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false,           1,           4,               S<1, 32, 1, 8>, CShuffleBlockTransferScalarPerVector_NPerBlock, MaskingSpec, Deterministic>;
        // clang-format on
        >;

void add_device_batched_mha_bwd_qloop_light_v2_casual_f16_f16_instances(
    std::vector<std::unique_ptr<DeviceBatchedMultiheadAttentionBackwardQloopLightV2<
        2,
        1,
        1,
        1,
        1,
        F16,
        F16,
        unsigned short,
        F32,
        F32,
        ck::Tuple<>,
        ck::Tuple<>,
        PassThrough,
        PassThrough,
        Scale,
        PassThrough,
        PassThrough,
        MaskingSpecialization::MaskUpperTriangleFromTopLeft>>>& instances)
{
    add_device_operation_instances(instances,
                                   device_batched_mha_bwd_qloop_light_v2_f16_f16_instances<
                                       2,
                                       1,
                                       1,
                                       1,
                                       1,
                                       MaskingSpecialization::MaskUpperTriangleFromTopLeft>{});
}

void add_device_batched_mha_bwd_qloop_light_v2_noncasual_f16_f16_instances(
    std::vector<std::unique_ptr<
        DeviceBatchedMultiheadAttentionBackwardQloopLightV2<2,
                                                            1,
                                                            1,
                                                            1,
                                                            1,
                                                            F16,
                                                            F16,
                                                            unsigned short,
                                                            F32,
                                                            F32,
                                                            ck::Tuple<>,
                                                            ck::Tuple<>,
                                                            PassThrough,
                                                            PassThrough,
                                                            Scale,
                                                            PassThrough,
                                                            PassThrough,
                                                            MaskingSpecialization::MaskDisabled>>>&
        instances)
{
    add_device_operation_instances(instances,
                                   device_batched_mha_bwd_qloop_light_v2_f16_f16_instances<
                                       2,
                                       1,
                                       1,
                                       1,
                                       1,
                                       MaskingSpecialization::MaskDisabled>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck