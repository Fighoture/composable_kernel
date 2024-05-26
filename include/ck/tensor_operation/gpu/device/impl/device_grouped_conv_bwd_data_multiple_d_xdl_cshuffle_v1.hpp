// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_conv_bwd_data_multiple_d.hpp"
#include "ck/tensor_operation/gpu/device/convolution_backward_data_specialization.hpp"
#include "ck/tensor_operation/operator_transform/transform_conv_bwd_data_to_gemm_v1.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_multiple_d_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v3_multi_abd.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_utils.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/host_utility/flush_cache.hpp"
#include "ck/host_utility/io.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

namespace {

/*
 * \brief Wrapper function of GridwiseGemm::Run to realize BatchedGEMM.
 *
 * \tparam ComputePtrOffsetOfBatch Class that computes the base pointer offsets of A, B, C matrix
 * given the batch. For example, ComputePtrOffsetOfStridedBatch() computes the offsets of evenly
 * strided batched, but we can easily extend to other layouts. The returned offset can be either \p
 * index_t or \p long_index_t. If it returns \p long_index_t, we are not subject to the 2GB
 * limitations.
 *
 * \tparam Block2ETileMap Block2ETileMap::CalculateBottomIndex() takes in id of a workgroup and
 * returns the 2D index of the tile that it computes. \see
 * GridwiseGemm_k0mk1_k0nk1_mn_xdlops_v2r3::Run().
 *
 * \note Using \p ComputePtrOffsetOfBatch gives us the flexibility that 2 workgroups can compute 2
 * tiles from different matrices. Keep in mind that these 2 matrices can share the same grid
 * descriptor (like in BatchedGEMM), or use their own grid descriptors (in GroupedGemm). \link
 * device_conv3d_fwd_xdl_ndhwc_kzyxc_ndhwk.hpp kernel_gemm_xdlops_v2r3_for_conv3d \endlink for \link
 * DeviceConv3d \endlink uses the same concept, but currently does NOT encapsulate the computing of
 * pointer offset into \p ComputePtrOffsetOfStridedBatch.
 *
 * \note \p Block2ETileMap allows customized mapping between a workgroup and the C-tile it computes.
 * Together with \p ComputePtrOffsetOfBatch, we can reuse GridwiseGemm (and GridwiseGemm fusion ) to
 * realize BatchedGemm and GroupedGemm (and the corresponding GEMM fusion).
 *
 */

template <typename GridwiseGemm,
    typename AGridDesc_AK0_M_K1,
    typename BGridDesc_BK0_N_K1,
    typename DsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
    typename CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
    typename ComputePtrOffsetOfBatch,
          bool HasMainKBlockLoop,
          InMemoryDataOperationEnum CGlobalMemoryDataOperation,
          index_t MinimumOccupancy = 1,
          TailNumber TailNum       = TailNumber::Full>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, MinimumOccupancy)
#endif
    // __attribute__((amdgpu_waves_per_eu(1, 1)))
    kernel_grouped_conv_bwd_data_xdl_cshuffle_v3(typename GridwiseGemm::Argument karg,
                                   const AGridDesc_AK0_M_K1 a_grid_desc_ak0_m_ak1,
                                const BGridDesc_BK0_N_K1 b_grid_desc_bk0_n_bk1,
                                const DsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
                                    ds_grid_desc_mblock_mperblock_nblock_nperblock,
                                const CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock c_grid_desc_mblock_mperblock_nblock_nperblock,
                                const ComputePtrOffsetOfBatch compute_ptr_offset_of_batch,
                                const index_t batch_count)
{
#if(!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx908__) || defined(__gfx90a__) || \
    defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__))
    // offset base pointer for each work-group
    const index_t num_blocks_per_batch =
        __builtin_amdgcn_readfirstlane(gridDim.y / batch_count);
        (void)(num_blocks_per_batch);
    const index_t g_idx = 0;// __builtin_amdgcn_readfirstlane(blockIdx.y / num_blocks_per_batch);

    const long_index_t a_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_ptr_offset_of_batch.GetAPtrOffset(g_idx)));
    const long_index_t b_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_ptr_offset_of_batch.GetBPtrOffset(g_idx)));
    const long_index_t e_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_ptr_offset_of_batch.GetEPtrOffset(g_idx)));
    const auto ds_batch_offset = compute_ptr_offset_of_batch.GetDsPtrOffset(g_idx);

    __shared__ char p_shared[GridwiseGemm::GetSharedMemoryNumberOfByte()];

    using AsPointer = decltype(karg.p_as_grid);
    using BsPointer = decltype(karg.p_bs_grid);
    using DsPointer = decltype(karg.p_ds_grid);
    AsPointer p_as_grid_grp;
    BsPointer p_bs_grid_grp;
    DsPointer p_ds_grid_grp;

    static constexpr index_t NumATensor =
        AsPointer::Size();
    static constexpr index_t NumBTensor =
        BsPointer::Size();
    static constexpr index_t NumDTensor =
        DsPointer::Size();

    static_for<0, NumATensor, 1>{}(
        [&](auto i) { p_as_grid_grp(i) =  karg.p_as_grid(i) + a_batch_offset; });
    static_for<0, NumBTensor, 1>{}(
        [&](auto i) { p_bs_grid_grp(i) = karg.p_bs_grid(i) + b_batch_offset; });
    static_for<0, NumDTensor, 1>{}(
        [&](auto i) { p_ds_grid_grp(i) = karg.p_ds_grid[i] + ds_batch_offset[i]; });

    GridwiseGemm::template Run<AGridDesc_AK0_M_K1,
     BGridDesc_BK0_N_K1, DsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
    CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock, HasMainKBlockLoop, CGlobalMemoryDataOperation, TailNum>(
        p_as_grid_grp,
        p_bs_grid_grp,
        p_ds_grid_grp,
        karg.p_c_grid + e_batch_offset,
        p_shared,
        karg, 
        a_grid_desc_ak0_m_ak1,
        b_grid_desc_bk0_n_bk1,
        ds_grid_desc_mblock_mperblock_nblock_nperblock,
        c_grid_desc_mblock_mperblock_nblock_nperblock,
        karg.a_element_op,
        karg.b_element_op,
        karg.c_element_op);
#else
    ignore = karg;
#endif // end of if (defined(__gfx908__) || defined(__gfx90a__))
}

template <typename GridwiseGemm,
    typename AGridDesc_AK0_M_K1,
    typename BGridDesc_BK0_N_K1,
    typename DsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
    typename CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
    typename ComputePtrOffsetOfBatch,
          bool HasMainKBlockLoop,
          InMemoryDataOperationEnum CGlobalMemoryDataOperation,
          index_t MinimumOccupancy = 1,
          TailNumber TailNum       = TailNumber::Full>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, MinimumOccupancy)
#endif
    // __attribute__((amdgpu_waves_per_eu(1, 1)))
    kernel_grouped_conv_bwd_data_xdl_cshuffle_v3_2lds(typename GridwiseGemm::Argument karg,
                                   const AGridDesc_AK0_M_K1 a_grid_desc_ak0_m_ak1,
                                const BGridDesc_BK0_N_K1 b_grid_desc_bk0_n_bk1,
                                const DsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
                                    ds_grid_desc_mblock_mperblock_nblock_nperblock,
                                const CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock c_grid_desc_mblock_mperblock_nblock_nperblock,
                                const ComputePtrOffsetOfBatch compute_ptr_offset_of_batch,
                                const index_t batch_count)
{
#if(!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx908__) || defined(__gfx90a__) || \
    defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__))
    // offset base pointer for each work-group
    const index_t num_blocks_per_batch =
        __builtin_amdgcn_readfirstlane(gridDim.y / batch_count);
    const index_t g_idx = __builtin_amdgcn_readfirstlane(blockIdx.y / num_blocks_per_batch);

    const long_index_t a_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_ptr_offset_of_batch.GetAPtrOffset(g_idx)));
    const long_index_t b_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_ptr_offset_of_batch.GetBPtrOffset(g_idx)));
    const long_index_t e_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_ptr_offset_of_batch.GetEPtrOffset(g_idx)));
    const auto ds_batch_offset = compute_ptr_offset_of_batch.GetDsPtrOffset(g_idx);

    // Pass two lds pointer is the key to tell compiler that ds_read/write
    // operate on different lds chunk at same time without order dependecy
    __shared__ char p_shared_0[GridwiseGemm::GetSharedMemoryNumberOfByte()];
    __shared__ char p_shared_1[GridwiseGemm::GetSharedMemoryNumberOfByte()];

    using AsPointer = decltype(karg.p_as_grid);
    using BsPointer = decltype(karg.p_bs_grid);
    using DsPointer = decltype(karg.p_ds_grid);
    AsPointer p_as_grid_grp;
    BsPointer p_bs_grid_grp;
    DsPointer p_ds_grid_grp;

    static constexpr index_t NumATensor =
        AsPointer::Size();
    static constexpr index_t NumBTensor =
        BsPointer::Size();
    static constexpr index_t NumDTensor =
        DsPointer::Size();

    static_for<0, NumATensor, 1>{}(
        [&](auto i) { p_as_grid_grp(i) =  karg.p_as_grid(i) + a_batch_offset; });
    static_for<0, NumBTensor, 1>{}(
        [&](auto i) { p_bs_grid_grp(i) = karg.p_bs_grid(i) + b_batch_offset; });
    static_for<0, NumDTensor, 1>{}(
        [&](auto i) { p_ds_grid_grp(i) = karg.p_ds_grid[i] + ds_batch_offset[i]; });

    GridwiseGemm::template Run_2Lds<AGridDesc_AK0_M_K1,
     BGridDesc_BK0_N_K1, DsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
    CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock, HasMainKBlockLoop, CGlobalMemoryDataOperation, TailNum>(
        p_as_grid_grp,
        p_bs_grid_grp,
        p_ds_grid_grp,
        karg.p_c_grid + e_batch_offset,
        p_shared_0,
        p_shared_1,
        karg, 
        a_grid_desc_ak0_m_ak1,
        b_grid_desc_bk0_n_bk1,
        ds_grid_desc_mblock_mperblock_nblock_nperblock,
        c_grid_desc_mblock_mperblock_nblock_nperblock,
        karg.a_element_op,
        karg.b_element_op,
        karg.c_element_op);
#else
    ignore = karg;
#endif // end of if (defined(__gfx908__) || defined(__gfx90a__))
}

} // namespace

// Conv backward data multiple D:
//   input : output image A: [G, N, K, Ho, Wo]
//   input : weight B: [G, K, C, Y, X],
//   input : D0, D1, ... : [G, N, K, Ho, Wo]
//   output : input image E: [G, N, C, Hi, Wi]
//   C = a_op(A) * b_op(B)
//   E = cde_op(C, D0, D1, ...)
template <index_t NDimSpatial,
          typename ALayout,   // output image
          typename BLayout,   // weight
          typename DsLayout,  // bias
          typename ELayout,   // input image
          typename ADataType, // output image
          typename BDataType, // weight
          typename AccDataType,
          typename CShuffleDataType,
          typename DsDataType,       // bias
          typename EDataType,        // input image
          typename AElementwiseOp,   // output image
          typename BElementwiseOp,   // weight
          typename CDEElementwiseOp, // C, bias, and input image
          ConvolutionBackwardDataSpecialization ConvBackwardDataSpecialization,
          bool DoPadGemmM,
          bool DoPadGemmN,
          index_t NumGemmKPrefetchStage,
          index_t BlockSize,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t AK1,
          index_t BK1,
          index_t MPerXDL,
          index_t NPerXDL,
          index_t MXdlPerWave,
          index_t NXdlPerWave,
          typename ABlockTransferThreadClusterLengths_AK0_M_AK1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_AK1,
          index_t ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_BK1,
          index_t BBlockLdsExtraN,
          index_t CShuffleMXdlPerWavePerShuffle,
          index_t CShuffleNXdlPerWavePerShuffle,
          typename CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CDEBlockTransferScalarPerVector_NPerBlock,
          LoopScheduler LoopSched = make_default_loop_scheduler(),
          BlockGemmPipelineScheduler BlkGemmPipeSched = BlockGemmPipelineScheduler::Intrawave,
          BlockGemmPipelineVersion BlkGemmPipelineVer = BlockGemmPipelineVersion::v1,
          typename AComputeType   = ADataType,
          typename BComputeType   = AComputeType>
struct DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1
    : public DeviceGroupedConvBwdDataMultipleD<NDimSpatial,
                                               ALayout,    // output image
                                               BLayout,    // weight
                                               DsLayout,   // bias
                                               ELayout,    // input image
                                               ADataType,  // output image
                                               BDataType,  // weight
                                               DsDataType, // bias
                                               EDataType,  // input image
                                               AElementwiseOp,
                                               BElementwiseOp,
                                               CDEElementwiseOp,
                                               AComputeType,
                                               BComputeType>
{
    // TODO: Extend support for more spatial dimensions.
    static_assert(NDimSpatial == 2 || NDimSpatial == 3,
                  "wrong! only implemented for 2D and 3D now");

    using DeviceOp = DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1;

    static constexpr index_t NumDTensor = DsDataType::Size();
    static constexpr GemmSpecialization GemmSpec = GemmSpecialization::Default;

    // TODO: Add support for different A and B data types.
    using ABDataType = ADataType;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    static constexpr auto transform_conv_to_gemm =
        TransformConvBwdDataToGemm_v1<NDimSpatial,
                                      ConvBackwardDataSpecialization,
                                      AK1,
                                      BK1,
                                      MPerBlock,
                                      NPerBlock,
                                      KPerBlock,
                                      DoPadGemmM,
                                      DoPadGemmN>{};

    static auto GetDummyABDsEGridDescriptor()
    {
        const std::array<index_t, NDimSpatial + 3> dummy_tensor_lengths = {1};
        const std::array<index_t, NDimSpatial + 3> dummy_tensor_strides = {1};
        const std::array<index_t, NDimSpatial> dummy_spatial_lengths    = {1};

        const auto a_grid_desc_ak0_m_ak1 =
            transform_conv_to_gemm.template MakeADescriptor_AK0_M_AK1<ALayout>(
                dummy_tensor_lengths,
                dummy_tensor_strides,
                dummy_tensor_lengths,
                dummy_tensor_strides,
                dummy_tensor_lengths,
                dummy_tensor_strides,
                dummy_spatial_lengths,
                dummy_spatial_lengths,
                dummy_spatial_lengths,
                dummy_spatial_lengths,
                dummy_spatial_lengths);

        const auto b_grid_desc_bk0_n_bk1 =
            transform_conv_to_gemm.template MakeBDescriptor_BK0_N_BK1<BLayout>(
                dummy_tensor_lengths,
                dummy_tensor_strides,
                dummy_tensor_lengths,
                dummy_tensor_strides,
                dummy_tensor_lengths,
                dummy_tensor_strides,
                dummy_spatial_lengths,
                dummy_spatial_lengths,
                dummy_spatial_lengths,
                dummy_spatial_lengths,
                dummy_spatial_lengths);

        const auto ds_grid_desc_m_n = generate_tuple(
            [&](auto i) {
                using DLayout = remove_cvref_t<tuple_element_t<i.value, DsLayout>>;

                return transform_conv_to_gemm.template MakeCDescriptor_M_N<DLayout>(
                    dummy_tensor_lengths,
                    dummy_tensor_strides,
                    dummy_tensor_lengths,
                    dummy_tensor_strides,
                    dummy_tensor_lengths,
                    dummy_tensor_strides,
                    dummy_spatial_lengths,
                    dummy_spatial_lengths,
                    dummy_spatial_lengths,
                    dummy_spatial_lengths,
                    dummy_spatial_lengths);
            },
            Number<NumDTensor>{});

        const auto e_grid_desc_m_n =
            transform_conv_to_gemm.template MakeCDescriptor_M_N<ELayout>(dummy_tensor_lengths,
                                                                         dummy_tensor_strides,
                                                                         dummy_tensor_lengths,
                                                                         dummy_tensor_strides,
                                                                         dummy_tensor_lengths,
                                                                         dummy_tensor_strides,
                                                                         dummy_spatial_lengths,
                                                                         dummy_spatial_lengths,
                                                                         dummy_spatial_lengths,
                                                                         dummy_spatial_lengths,
                                                                         dummy_spatial_lengths);

        return make_tuple(a_grid_desc_ak0_m_ak1, b_grid_desc_bk0_n_bk1, ds_grid_desc_m_n, e_grid_desc_m_n);
    }

    #define GridwiseGemmTemplateParams \
        tensor_layout::gemm::RowMajor, \
        tensor_layout::gemm::ColumnMajor, \
        tensor_layout::gemm::RowMajor, \
        Tuple<ADataType>, \
        Tuple<BDataType>, \
        AccDataType, \
        CShuffleDataType, \
        DsDataType, \
        EDataType, \
        AElementwiseOp, \
        BElementwiseOp, \
        CDEElementwiseOp, \
        GemmSpec, \
        BlockSize, \
        MPerBlock, \
        NPerBlock, \
        KPerBlock, \
        AK1, \
        BK1, \
        MPerXDL, \
        NPerXDL, \
        MXdlPerWave, \
        NXdlPerWave, \
        ABlockTransferThreadClusterLengths_AK0_M_AK1, \
        ABlockTransferThreadClusterArrangeOrder, \
        ABlockTransferSrcAccessOrder, \
        ABlockTransferSrcVectorDim, \
        ABlockTransferSrcScalarPerVector, \
        ABlockTransferDstScalarPerVector_AK1, \
        false, \
        ABlockLdsExtraM, \
        BBlockTransferThreadClusterLengths_BK0_N_BK1, \
        BBlockTransferThreadClusterArrangeOrder, \
        BBlockTransferSrcAccessOrder, \
        BBlockTransferSrcVectorDim, \
        BBlockTransferSrcScalarPerVector, \
        BBlockTransferDstScalarPerVector_BK1, \
        false, \
        BBlockLdsExtraN, \
        CShuffleMXdlPerWavePerShuffle, \
        CShuffleNXdlPerWavePerShuffle, \
        CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock, \
        CDEBlockTransferScalarPerVector_NPerBlock, \
        BlkGemmPipeSched, \
        BlkGemmPipelineVer, \
        AComputeType, \
        BComputeType

    using GridwiseGemm = GridwiseGemm_xdl_cshuffle_v3<GridwiseGemmTemplateParams>;

    // desc
    using ABDsEGridDesc = decltype(GetDummyABDsEGridDescriptor());

    using AGridDesc_AK0_M_AK1 = remove_cvref_t<tuple_element_t<0, ABDsEGridDesc>>;
    using BGridDesc_BK0_N_BK1 = remove_cvref_t<tuple_element_t<1, ABDsEGridDesc>>;
    using DsGridDesc_M_N      = remove_cvref_t<tuple_element_t<2, ABDsEGridDesc>>;
    using EGridDesc_M_N       = remove_cvref_t<tuple_element_t<3, ABDsEGridDesc>>;

    // Empty tuple if no multiD
    using DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock = decltype(GridwiseGemm::MakeDsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(DsGridDesc_M_N{}, 1, 1));
    using EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock =
        decltype(GridwiseGemm::MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
            EGridDesc_M_N{}, 1, 1));

    // Argument
    struct Argument : public BaseArgument
    {
        Argument(const void* p_a,                                 // output image
                 const void* p_b,                                 // weight
                 const std::array<const void*, NumDTensor>& p_ds, // bias
                 void* p_e,                                       // input image
                 const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_lengths,
                 const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_strides,
                 const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,
                 const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_strides,
                 const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>&
                     ds_g_n_c_wis_lengths,
                 const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>&
                     ds_g_n_c_wis_strides,
                 const std::array<index_t, NDimSpatial + 3>& e_g_n_c_wis_lengths,
                 const std::array<index_t, NDimSpatial + 3>& e_g_n_c_wis_strides,
                 const std::array<index_t, NDimSpatial>& conv_filter_strides,
                 const std::array<index_t, NDimSpatial>& conv_filter_dilations,
                 const std::array<index_t, NDimSpatial>& input_left_pads,
                 const std::array<index_t, NDimSpatial>& input_right_pads,
                 const AElementwiseOp& a_element_op,
                 const BElementwiseOp& b_element_op,
                 const CDEElementwiseOp& cde_element_op)
            : p_a_grid_{p_a},
              p_b_grid_{p_b},
              p_ds_grid_{},
              p_e_grid_{p_e},
              num_group_{a_g_n_k_wos_lengths[0]},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              cde_element_op_{cde_element_op},
              a_g_n_k_wos_lengths_{a_g_n_k_wos_lengths},
              a_g_n_k_wos_strides_{a_g_n_k_wos_strides},
              b_g_k_c_xs_lengths_{b_g_k_c_xs_lengths},
              b_g_k_c_xs_strides_{b_g_k_c_xs_strides},
              ds_g_n_c_wis_lengths_{ds_g_n_c_wis_lengths},
              ds_g_n_c_wis_strides_{ds_g_n_c_wis_strides},
              e_g_n_c_wis_lengths_{e_g_n_c_wis_lengths},
              e_g_n_c_wis_strides_{e_g_n_c_wis_strides},
              conv_filter_strides_{conv_filter_strides},
              conv_filter_dilations_{conv_filter_dilations},
              input_left_pads_{input_left_pads},
              input_right_pads_{input_right_pads}
        {
            // populate Ds pointer
            static_for<0, NumDTensor, 1>{}([&](auto i) {
                p_ds_grid_(i) = p_ds[i];
            });

            // A/B/Ds/E Batch Stride
            compute_ptr_offset_of_batch_.BatchStrideA_ = a_g_n_k_wos_strides[0];
            compute_ptr_offset_of_batch_.BatchStrideB_ = b_g_k_c_xs_strides[0];
            compute_ptr_offset_of_batch_.BatchStrideE_ = e_g_n_c_wis_strides[0];

            static_for<0, NumDTensor, 1>{}([&](auto i) {
                compute_ptr_offset_of_batch_.BatchStrideDs_(i) = ds_g_n_c_wis_strides[i][0];
            });

            static constexpr auto NonSpatialDimsNum = Number<3>{};

            static constexpr auto DIdx = Number<NonSpatialDimsNum>{};
            static constexpr auto HIdx =
                NDimSpatial == 2 ? Number<NonSpatialDimsNum>{} : Number<NonSpatialDimsNum + 1>{};
            static constexpr auto WIdx = NDimSpatial == 2 ? Number<NonSpatialDimsNum + 1>{}
                                                          : Number<NonSpatialDimsNum + 2>{};

            static constexpr auto ZIdx = Number<NonSpatialDimsNum>{};
            static constexpr auto YIdx =
                NDimSpatial == 2 ? Number<NonSpatialDimsNum>{} : Number<NonSpatialDimsNum + 1>{};
            static constexpr auto XIdx = NDimSpatial == 2 ? Number<NonSpatialDimsNum + 1>{}
                                                          : Number<NonSpatialDimsNum + 2>{};

            // problem definition
            const index_t Z = b_g_k_c_xs_lengths[ZIdx];
            const index_t Y = b_g_k_c_xs_lengths[YIdx];
            const index_t X = b_g_k_c_xs_lengths[XIdx];

            const index_t ConvStrideD = conv_filter_strides[DIdx - NonSpatialDimsNum];
            const index_t ConvStrideH = conv_filter_strides[HIdx - NonSpatialDimsNum];
            const index_t ConvStrideW = conv_filter_strides[WIdx - NonSpatialDimsNum];

            const index_t ConvDilationD = conv_filter_dilations[DIdx - NonSpatialDimsNum];
            const index_t ConvDilationH = conv_filter_dilations[HIdx - NonSpatialDimsNum];
            const index_t ConvDilationW = conv_filter_dilations[WIdx - NonSpatialDimsNum];

            const auto GcdStrideDilationD = math::gcd(ConvStrideD, ConvDilationD);
            const auto GcdStrideDilationH = math::gcd(ConvStrideH, ConvDilationH);
            const auto GcdStrideDilationW = math::gcd(ConvStrideW, ConvDilationW);

            const auto ZTilde = NDimSpatial == 3 ? ConvStrideD / GcdStrideDilationD : 1;
            const auto YTilde = ConvStrideH / GcdStrideDilationH;
            const auto XTilde = ConvStrideW / GcdStrideDilationW;

            for(index_t i_ztilde = 0; i_ztilde < ZTilde; ++i_ztilde)
            {

                for(index_t i_ytilde = 0; i_ytilde < YTilde; ++i_ytilde)
                {
                    for(index_t i_xtilde = 0; i_xtilde < XTilde; ++i_xtilde)
                    {
                        // check slice is valid
                        const auto ZDotSlice =
                            NDimSpatial == 3 ? math::integer_divide_ceil(Z - i_ztilde, ZTilde) : 1;
                        const auto YDotSlice = math::integer_divide_ceil(Y - i_ytilde, YTilde);
                        const auto XDotSlice = math::integer_divide_ceil(X - i_xtilde, XTilde);

                        if(YDotSlice * XDotSlice * ZDotSlice <= 0)
                        {
                            continue;
                        }

                        std::array<index_t, NDimSpatial> tildes;
                        if constexpr(NDimSpatial == 2)
                        {
                            tildes = {i_ytilde, i_xtilde};
                        }
                        else if constexpr(NDimSpatial == 3)
                        {
                            tildes = {i_ztilde, i_ytilde, i_xtilde};
                        }
                        else
                        {
                            throw std::runtime_error("wrong! only implemented for 2D and 3D now");
                        }

                        const auto a_grid_desc_ak0_m_ak1 =
                            transform_conv_to_gemm.template MakeADescriptor_AK0_M_AK1<ALayout>(
                                a_g_n_k_wos_lengths,
                                a_g_n_k_wos_strides,
                                b_g_k_c_xs_lengths,
                                b_g_k_c_xs_strides,
                                e_g_n_c_wis_lengths,
                                e_g_n_c_wis_strides,
                                conv_filter_strides,
                                conv_filter_dilations,
                                input_left_pads,
                                input_right_pads,
                                tildes);

                        const auto b_grid_desc_bk0_n_bk1 =
                            transform_conv_to_gemm.template MakeBDescriptor_BK0_N_BK1<BLayout>(
                                a_g_n_k_wos_lengths,
                                a_g_n_k_wos_strides,
                                b_g_k_c_xs_lengths,
                                b_g_k_c_xs_strides,
                                e_g_n_c_wis_lengths,
                                e_g_n_c_wis_strides,
                                conv_filter_strides,
                                conv_filter_dilations,
                                input_left_pads,
                                input_right_pads,
                                tildes);

                        const auto e_grid_desc_m_n =
                            transform_conv_to_gemm.template MakeCDescriptor_M_N<ELayout>(
                                a_g_n_k_wos_lengths,
                                a_g_n_k_wos_strides,
                                b_g_k_c_xs_lengths,
                                b_g_k_c_xs_strides,
                                e_g_n_c_wis_lengths,
                                e_g_n_c_wis_strides,
                                conv_filter_strides,
                                conv_filter_dilations,
                                input_left_pads,
                                input_right_pads,
                                tildes);

                        const index_t MBlock = GridwiseGemm::CalculateMBlock(e_grid_desc_m_n.GetLength(I0)); 
                        const index_t NBlock = GridwiseGemm::CalculateNBlock(e_grid_desc_m_n.GetLength(I1));

                        DsGridDesc_M_N ds_grid_desc_m_n;

                        // populate Ds desc
                        static_for<0, NumDTensor, 1>{}([&](auto i) {
                            using DLayout = remove_cvref_t<tuple_element_t<i.value, DsLayout>>;

                            ds_grid_desc_m_n(i) =
                                transform_conv_to_gemm.template MakeCDescriptor_M_N<DLayout>(
                                    a_g_n_k_wos_lengths,
                                    a_g_n_k_wos_strides,
                                    b_g_k_c_xs_lengths,
                                    b_g_k_c_xs_strides,
                                    ds_g_n_c_wis_lengths[i],
                                    ds_g_n_c_wis_strides[i],
                                    conv_filter_strides,
                                    conv_filter_dilations,
                                    input_left_pads,
                                    input_right_pads,
                                    tildes);
                        });

                        const auto ds_grid_desc_mblock_mperblock_nblock_nperblock = GridwiseGemm::MakeDsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(ds_grid_desc_m_n, MBlock, NBlock);
                        ds_grid_desc_mblock_mperblock_nblock_nperblock_container_.push_back(ds_grid_desc_mblock_mperblock_nblock_nperblock);

                        // desc for blockwise copy
                        a_grid_desc_ak0_m_ak1_container_.push_back(a_grid_desc_ak0_m_ak1);
                        b_grid_desc_bk0_n_bk1_container_.push_back(b_grid_desc_bk0_n_bk1);
                        // there is no need to check since M, N, K are padded
                        e_grid_desc_mblock_mperblock_nblock_nperblock_container_.push_back(
                            GridwiseGemm::MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                                e_grid_desc_m_n, MBlock, NBlock));
                    }
                }
            }
        }

        void Print() const
        {
            for(std::size_t i = 0; i < a_grid_desc_ak0_m_ak1_container_.size(); i++)
            {
                std::cout << "a_grid_desc_ak0_m_ak1_container_"
                          << a_grid_desc_ak0_m_ak1_container_[i] << std::endl;

                std::cout << "b_grid_desc_bk0_n_bk1_container_"
                          << b_grid_desc_bk0_n_bk1_container_[i] << std::endl;

                static_for<0, NumDTensor, 1>{}([&](auto j) {
                    std::cout << "ds_grid_desc_mblock_mperblock_nblock_nperblock_container_"
                              << ds_grid_desc_mblock_mperblock_nblock_nperblock_container_[i][j]
                              << std::endl;
                });

                std::cout << "e_grid_desc_mblock_mperblock_nblock_nperblock_container_"
                          << e_grid_desc_mblock_mperblock_nblock_nperblock_container_[i]
                          << std::endl;
            }
        }

        // pointers
        std::array<const void*, I1> p_a_grid_;
        std::array<const void*, I1> p_b_grid_;
        std::array<const void*, NumDTensor>  p_ds_grid_;
        void* p_e_grid_;

        // tensor descriptor for problem definition
        index_t num_group_;

        // tensor descriptor for block-wise copy
        std::vector<AGridDesc_AK0_M_AK1> a_grid_desc_ak0_m_ak1_container_;
        std::vector<BGridDesc_BK0_N_BK1> b_grid_desc_bk0_n_bk1_container_;
        std::vector<DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>
            ds_grid_desc_mblock_mperblock_nblock_nperblock_container_;
        std::vector<EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>
            e_grid_desc_mblock_mperblock_nblock_nperblock_container_;

        // for computing batch offset
        ComputePtrOffsetOfStridedBatch<I1, I1, NumDTensor> compute_ptr_offset_of_batch_;

        // element-wise op
        AElementwiseOp a_element_op_;
        BElementwiseOp b_element_op_;
        CDEElementwiseOp cde_element_op_;

        // for checking IsSupportedArgument()
        std::array<index_t, NDimSpatial + 3> a_g_n_k_wos_lengths_;
        std::array<index_t, NDimSpatial + 3> a_g_n_k_wos_strides_;
        std::array<index_t, NDimSpatial + 3> b_g_k_c_xs_lengths_;
        std::array<index_t, NDimSpatial + 3> b_g_k_c_xs_strides_;
        std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor> ds_g_n_c_wis_lengths_;
        std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor> ds_g_n_c_wis_strides_;
        std::array<index_t, NDimSpatial + 3> e_g_n_c_wis_lengths_;
        std::array<index_t, NDimSpatial + 3> e_g_n_c_wis_strides_;
        std::array<index_t, NDimSpatial> conv_filter_strides_;
        std::array<index_t, NDimSpatial> conv_filter_dilations_;
        std::array<index_t, NDimSpatial> input_left_pads_;
        std::array<index_t, NDimSpatial> input_right_pads_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceOp::Argument;

        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            if(stream_config.log_level_ > 0)
            {
                arg.Print();
            }

            float ave_time = 0;

            constexpr index_t minimum_occupancy =
                BlkGemmPipeSched == BlockGemmPipelineScheduler::Intrawave ? 1 : 2;

            for(std::size_t i = 0; i < arg.a_grid_desc_ak0_m_ak1_container_.size(); i++)
            {
                const index_t GemmM = arg.a_grid_desc_ak0_m_ak1_container_[i].GetLength(I1);
                const index_t GemmN = arg.b_grid_desc_bk0_n_bk1_container_[i].GetLength(I1);
                const index_t GemmK = arg.a_grid_desc_ak0_m_ak1_container_[i].GetLength(I0) * arg.a_grid_desc_ak0_m_ak1_container_[i].GetLength(I2);

                index_t gdx, gdy, gdz;
                std::tie(gdx, gdy, gdz) = GridwiseGemm::CalculateGridSize(GemmM, GemmN, I1 /*arg.KBatch*/, arg.num_group_);

                index_t K_split = (GemmK + KPerBlock - 1) / KPerBlock * KPerBlock;
                const bool has_main_k_block_loop = GridwiseGemm::CalculateHasMainKBlockLoop(K_split);

                typename GridwiseGemm::Argument gemm_arg{arg.p_a_grid_, arg.p_b_grid_, arg.p_ds_grid_, arg.p_e_grid_, GemmM, GemmN, GemmK, {I0}, {I0}, arg.compute_ptr_offset_of_batch_.BatchStrideDs_, I0, I1, arg.a_element_op_,
                                            arg.b_element_op_, arg.cde_element_op_};

                const auto Run = [&](const auto& kernel) {
                    ave_time = launch_and_time_kernel(
                        stream_config, kernel, dim3(gdx, gdy, gdz), dim3(BlockSize), 0, gemm_arg,
                    make_tuple(arg.a_grid_desc_ak0_m_ak1_container_[i]),
                    make_tuple(arg.b_grid_desc_bk0_n_bk1_container_[i]),
                    arg.ds_grid_desc_mblock_mperblock_nblock_nperblock_container_[i],
                    arg.e_grid_desc_mblock_mperblock_nblock_nperblock_container_[i],
                    arg.compute_ptr_offset_of_batch_,
                    arg.a_g_n_k_wos_lengths_[0]);
                };

                if(has_main_k_block_loop)
                {
                    // Tail number always full
                    if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v1 ||
                                BlkGemmPipelineVer == BlockGemmPipelineVersion::v3)
                    {
                            const auto kernel =
                                kernel_grouped_conv_bwd_data_xdl_cshuffle_v3<GridwiseGemm, Tuple<DeviceOp::AGridDesc_AK0_M_AK1>,
                        Tuple<DeviceOp::BGridDesc_BK0_N_BK1>, DeviceOp::DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                        DeviceOp::EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock, ComputePtrOffsetOfStridedBatch<I1, I1, NumDTensor>,
                                                            true,
                                                            InMemoryDataOperationEnum::Set,
                                                            minimum_occupancy>;
                            Run(kernel);
                    }
                    // Tail number could be One to Seven
                    else if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v2)
                    {
                            if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::One)
                            {
                                const auto kernel =
                                    kernel_grouped_conv_bwd_data_xdl_cshuffle_v3<GridwiseGemm,                         Tuple<DeviceOp::AGridDesc_AK0_M_AK1>,
                        Tuple<DeviceOp::BGridDesc_BK0_N_BK1>, DeviceOp::DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                        DeviceOp::EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock, ComputePtrOffsetOfStridedBatch<I1, I1, NumDTensor>,
                                                                true,
                                                                InMemoryDataOperationEnum::Set,
                                                                minimum_occupancy,
                                                                TailNumber::One>;
                                Run(kernel);
                            }
                            else if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) ==
                                    TailNumber::Full)
                            {
                                const auto kernel =
                                    kernel_grouped_conv_bwd_data_xdl_cshuffle_v3<GridwiseGemm,                         Tuple<DeviceOp::AGridDesc_AK0_M_AK1>,
                        Tuple<DeviceOp::BGridDesc_BK0_N_BK1>, DeviceOp::DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                        DeviceOp::EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock, ComputePtrOffsetOfStridedBatch<I1, I1, NumDTensor>,
                                                                true,
                                                                InMemoryDataOperationEnum::Set,
                                                                minimum_occupancy,
                                                                TailNumber::Full>;
                                Run(kernel);
                            }

                            if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 2)
                            {
                                if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Two)
                                {
                                    const auto kernel =
                                        kernel_grouped_conv_bwd_data_xdl_cshuffle_v3<GridwiseGemm,                         Tuple<DeviceOp::AGridDesc_AK0_M_AK1>,
                        Tuple<DeviceOp::BGridDesc_BK0_N_BK1>, DeviceOp::DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                        DeviceOp::EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock, ComputePtrOffsetOfStridedBatch<I1, I1, NumDTensor>,
                                                                    true,
                                                                    InMemoryDataOperationEnum::Set,
                                                                    minimum_occupancy,
                                                                    TailNumber::Two>;
                                    Run(kernel);
                                }
                            }

                            if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 3)
                            {
                                if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) ==
                                TailNumber::Three)
                                {
                                    const auto kernel =
                                        kernel_grouped_conv_bwd_data_xdl_cshuffle_v3<GridwiseGemm,                         Tuple<DeviceOp::AGridDesc_AK0_M_AK1>,
                        Tuple<DeviceOp::BGridDesc_BK0_N_BK1>, DeviceOp::DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                        DeviceOp::EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock, ComputePtrOffsetOfStridedBatch<I1, I1, NumDTensor>,
                                                                    true,
                                                                    InMemoryDataOperationEnum::Set,
                                                                    minimum_occupancy,
                                                                    TailNumber::Three>;
                                    Run(kernel);
                                }
                            }

                            if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 4)
                            {
                                if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) ==
                                TailNumber::Four)
                                {
                                    const auto kernel =
                                        kernel_grouped_conv_bwd_data_xdl_cshuffle_v3<GridwiseGemm,                         Tuple<DeviceOp::AGridDesc_AK0_M_AK1>,
                        Tuple<DeviceOp::BGridDesc_BK0_N_BK1>, DeviceOp::DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                        DeviceOp::EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock, ComputePtrOffsetOfStridedBatch<I1, I1, NumDTensor>,
                                                                    true,
                                                                    InMemoryDataOperationEnum::Set,
                                                                    minimum_occupancy,
                                                                    TailNumber::Four>;
                                    Run(kernel);
                                }
                            }

                            if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 5)
                            {
                                if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) ==
                                TailNumber::Five)
                                {
                                    const auto kernel =
                                        kernel_grouped_conv_bwd_data_xdl_cshuffle_v3<GridwiseGemm,                         Tuple<DeviceOp::AGridDesc_AK0_M_AK1>,
                        Tuple<DeviceOp::BGridDesc_BK0_N_BK1>, DeviceOp::DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                        DeviceOp::EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock, ComputePtrOffsetOfStridedBatch<I1, I1, NumDTensor>,
                                                                    true,
                                                                    InMemoryDataOperationEnum::Set,
                                                                    minimum_occupancy,
                                                                    TailNumber::Five>;
                                    Run(kernel);
                                }
                            }

                            if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 6)
                            {
                                if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Six)
                                {
                                    const auto kernel =
                                        kernel_grouped_conv_bwd_data_xdl_cshuffle_v3<GridwiseGemm,                         Tuple<DeviceOp::AGridDesc_AK0_M_AK1>,
                        Tuple<DeviceOp::BGridDesc_BK0_N_BK1>, DeviceOp::DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                        DeviceOp::EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock, ComputePtrOffsetOfStridedBatch<I1, I1, NumDTensor>,
                                                                    true,
                                                                    InMemoryDataOperationEnum::Set,
                                                                    minimum_occupancy,
                                                                    TailNumber::Six>;
                                    Run(kernel);
                                }
                            }

                            if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 7)
                            {
                                if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) ==
                                TailNumber::Seven)
                                {
                                    const auto kernel =
                                        kernel_grouped_conv_bwd_data_xdl_cshuffle_v3<GridwiseGemm,                         Tuple<DeviceOp::AGridDesc_AK0_M_AK1>,
                        Tuple<DeviceOp::BGridDesc_BK0_N_BK1>, DeviceOp::DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                        DeviceOp::EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock, ComputePtrOffsetOfStridedBatch<I1, I1, NumDTensor>,
                                                                    true,
                                                                    InMemoryDataOperationEnum::Set,
                                                                    minimum_occupancy,
                                                                    TailNumber::Seven>;
                                    Run(kernel);
                                }
                            }
                    }
                    // Tail number could be Odd or Even
                    else if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v4)
                    {
                            if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Odd)
                            {
                                const auto kernel =
                                    kernel_grouped_conv_bwd_data_xdl_cshuffle_v3_2lds<GridwiseGemm,                         Tuple<DeviceOp::AGridDesc_AK0_M_AK1>,
                        Tuple<DeviceOp::BGridDesc_BK0_N_BK1>, DeviceOp::DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                        DeviceOp::EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock, ComputePtrOffsetOfStridedBatch<I1, I1, NumDTensor>,
                                                                    true,
                                                                    InMemoryDataOperationEnum::Set,
                                                                    minimum_occupancy,
                                                                    TailNumber::Odd>;
                                Run(kernel);
                            }
                            else
                            {
                                const auto kernel =
                                    kernel_grouped_conv_bwd_data_xdl_cshuffle_v3_2lds<GridwiseGemm,                         Tuple<DeviceOp::AGridDesc_AK0_M_AK1>,
                        Tuple<DeviceOp::BGridDesc_BK0_N_BK1>, DeviceOp::DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                        DeviceOp::EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock, ComputePtrOffsetOfStridedBatch<I1, I1, NumDTensor>,
                                                                    true,
                                                                    InMemoryDataOperationEnum::Set,
                                                                    minimum_occupancy,
                                                                    TailNumber::Even>;
                                Run(kernel);
                            }
                    }
                    else
                    {
                            if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Odd)
                            {
                                const auto kernel =
                                    kernel_grouped_conv_bwd_data_xdl_cshuffle_v3<GridwiseGemm,                         Tuple<DeviceOp::AGridDesc_AK0_M_AK1>,
                        Tuple<DeviceOp::BGridDesc_BK0_N_BK1>, DeviceOp::DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                        DeviceOp::EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock, ComputePtrOffsetOfStridedBatch<I1, I1, NumDTensor>,
                                                                true,
                                                                InMemoryDataOperationEnum::Set,
                                                                minimum_occupancy,
                                                                TailNumber::Odd>;
                                Run(kernel);
                            }
                            else
                            {
                                const auto kernel =
                                    kernel_grouped_conv_bwd_data_xdl_cshuffle_v3<GridwiseGemm,                         Tuple<DeviceOp::AGridDesc_AK0_M_AK1>,
                        Tuple<DeviceOp::BGridDesc_BK0_N_BK1>, DeviceOp::DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                        DeviceOp::EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock, ComputePtrOffsetOfStridedBatch<I1, I1, NumDTensor>,
                                                                true,
                                                                InMemoryDataOperationEnum::Set,
                                                                minimum_occupancy,
                                                                TailNumber::Even>;
                                Run(kernel);
                            }
                    }
                }
                else
                {
                    // Tail number always 1
                    if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v1)
                    {
                            const auto kernel =
                                kernel_grouped_conv_bwd_data_xdl_cshuffle_v3<GridwiseGemm,                         Tuple<DeviceOp::AGridDesc_AK0_M_AK1>,
                        Tuple<DeviceOp::BGridDesc_BK0_N_BK1>, DeviceOp::DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                        DeviceOp::EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock, ComputePtrOffsetOfStridedBatch<I1, I1, NumDTensor>,
                                                            false,
                                                            InMemoryDataOperationEnum::Set,
                                                            minimum_occupancy>;
                            Run(kernel);
                    }
                }
            }

            return ave_time;
        }

        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    static bool IsSupportedArgument(const Argument& arg)
    {
        if(!ck::is_xdl_supported())
        {
            return false;
        }

        const index_t ConvK = arg.b_g_k_c_xs_lengths_[1];
        const index_t ConvC = arg.b_g_k_c_xs_lengths_[2];

        // Specifialization
        if constexpr(ConvBackwardDataSpecialization ==
                     ConvolutionBackwardDataSpecialization::Filter1x1Stride1Pad0)
        {
            // check if it's 1x1, stride=1 pad = 0 conv
            for(int i = 0; i < NDimSpatial; i++)
            {
                if(!(arg.b_g_k_c_xs_lengths_[3 + i] == 1 && arg.conv_filter_strides_[i] == 1 &&
                     arg.input_left_pads_[i] == 0 && arg.input_right_pads_[i] == 0))
                {
                    return false;
                }
            }
        }

        // vector load for A matrix from global memory to LDS
        if constexpr(is_same_v<ALayout, tensor_layout::convolution::GNHWK> ||
                     is_same_v<ALayout, tensor_layout::convolution::GNDHWK> ||
                     is_same_v<ALayout, tensor_layout::convolution::NHWGK> ||
                     is_same_v<ALayout, tensor_layout::convolution::NDHWGK>)
        {
            if(!(ABlockTransferSrcVectorDim == 2 && ConvK % ABlockTransferSrcScalarPerVector == 0))
            {
                return false;
            }
        }
        else
        {
            return false;
        }

        // vector load for B matrix from global memory to LDS
        if constexpr(is_same_v<BLayout, tensor_layout::convolution::GKYXC> ||
                     is_same_v<BLayout, tensor_layout::convolution::GKZYXC>)
        {
            if(!(BBlockTransferSrcVectorDim == 1 && ConvC % BBlockTransferSrcScalarPerVector == 0))
            {
                return false;
            }
        }
        else
        {
            return false;
        }

        // vector store for Ds
        bool ds_valid = true;

        static_for<0, NumDTensor, 1>{}([&](auto i) {
            using DLayout = remove_cvref_t<tuple_element_t<i.value, DsLayout>>;

            if constexpr(is_same_v<DLayout, tensor_layout::convolution::GNHWC> ||
                         is_same_v<DLayout, tensor_layout::convolution::GNDHWC> ||
                         is_same_v<DLayout, tensor_layout::convolution::NHWGC> ||
                         is_same_v<DLayout, tensor_layout::convolution::NDHWGC> ||
                         is_same_v<DLayout, tensor_layout::convolution::G_NHW_C> ||
                         is_same_v<DLayout, tensor_layout::convolution::GC> ||
                         is_same_v<DLayout, tensor_layout::convolution::G_C>)
            {
                // vector load D matrix from global memory
                if(!(ConvC % CDEBlockTransferScalarPerVector_NPerBlock == 0))
                {
                    ds_valid = false;
                }
            }
            else
            {
                ds_valid = false;
            }
        });

        if(!ds_valid)
        {
            return false;
        }

        // vector store for E
        if constexpr(is_same_v<ELayout, tensor_layout::convolution::GNHWC> ||
                     is_same_v<ELayout, tensor_layout::convolution::GNDHWC> ||
                     is_same_v<ELayout, tensor_layout::convolution::NHWGC> ||
                     is_same_v<ELayout, tensor_layout::convolution::NDHWGC>)
        {
            // vector store C matrix into global memory
            if(!(ConvC % CDEBlockTransferScalarPerVector_NPerBlock == 0))
            {
                return false;
            }
        }
        else
        {
            return false;
        }

        for(std::size_t i = 0; i < arg.a_grid_desc_ak0_m_ak1_container_.size(); i++)
        {
            const index_t GemmK = arg.a_grid_desc_ak0_m_ak1_container_[i].GetLength(I0) * arg.a_grid_desc_ak0_m_ak1_container_[i].GetLength(I2);
            const auto num_k_loop = GridwiseGemm::CalculateAK0Padded(GemmK, I1) / (KPerBlock / AK1);

            if constexpr(BlkGemmPipelineVer != BlockGemmPipelineVersion::v1)
            {
                if(num_k_loop <= GridwiseGemm::BlockwiseGemmPipe::PrefetchStages)
                {
                    return false;
                }
            }
        }


        return true;
    }

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto
    MakeArgument(const void* p_a,                                                 // output image
                 const void* p_b,                                                 // weight
                 const std::array<const void*, NumDTensor>& p_ds,                 // bias
                 void* p_e,                                                       // input image
                 const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_lengths, // output image
                 const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_strides, // output image
                 const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,  // weight
                 const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_strides,  // weight
                 const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>&
                     ds_g_n_c_wis_lengths, // bias
                 const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>&
                     ds_g_n_c_wis_strides,                                        // bias
                 const std::array<index_t, NDimSpatial + 3>& e_g_n_c_wis_lengths, // input image
                 const std::array<index_t, NDimSpatial + 3>& e_g_n_c_wis_strides, // input image
                 const std::array<index_t, NDimSpatial>& conv_filter_strides,
                 const std::array<index_t, NDimSpatial>& conv_filter_dilations,
                 const std::array<index_t, NDimSpatial>& input_left_pads,
                 const std::array<index_t, NDimSpatial>& input_right_pads,
                 const AElementwiseOp& a_element_op,
                 const BElementwiseOp& b_element_op,
                 const CDEElementwiseOp& cde_element_op)
    {
        return Argument{p_a,
                        p_b,
                        p_ds,
                        p_e,
                        a_g_n_k_wos_lengths,
                        a_g_n_k_wos_strides,
                        b_g_k_c_xs_lengths,
                        b_g_k_c_xs_strides,
                        ds_g_n_c_wis_lengths,
                        ds_g_n_c_wis_strides,
                        e_g_n_c_wis_lengths,
                        e_g_n_c_wis_strides,
                        conv_filter_strides,
                        conv_filter_dilations,
                        input_left_pads,
                        input_right_pads,
                        a_element_op,
                        b_element_op,
                        cde_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    std::unique_ptr<BaseArgument> MakeArgumentPointer(
        const void* p_a,                                                 // output image
        const void* p_b,                                                 // weight
        const std::array<const void*, NumDTensor>& p_ds,                 // bias
        void* p_e,                                                       // input image
        const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_lengths, // output image
        const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_strides, // output image
        const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,  // weight
        const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_strides,  // weight
        const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>&
            ds_g_n_c_wis_lengths, // bias
        const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>&
            ds_g_n_c_wis_strides,                                        // bias
        const std::array<index_t, NDimSpatial + 3>& e_g_n_c_wis_lengths, // input image
        const std::array<index_t, NDimSpatial + 3>& e_g_n_c_wis_strides, // input image
        const std::array<index_t, NDimSpatial>& conv_filter_strides,
        const std::array<index_t, NDimSpatial>& conv_filter_dilations,
        const std::array<index_t, NDimSpatial>& input_left_pads,
        const std::array<index_t, NDimSpatial>& input_right_pads,
        const AElementwiseOp& a_element_op,
        const BElementwiseOp& b_element_op,
        const CDEElementwiseOp& cde_element_op) override
    {
        return std::make_unique<Argument>(p_a,
                                          p_b,
                                          p_ds,
                                          p_e,
                                          a_g_n_k_wos_lengths,
                                          a_g_n_k_wos_strides,
                                          b_g_k_c_xs_lengths,
                                          b_g_k_c_xs_strides,
                                          ds_g_n_c_wis_lengths,
                                          ds_g_n_c_wis_strides,
                                          e_g_n_c_wis_lengths,
                                          e_g_n_c_wis_strides,
                                          conv_filter_strides,
                                          conv_filter_dilations,
                                          input_left_pads,
                                          input_right_pads,
                                          a_element_op,
                                          b_element_op,
                                          cde_element_op);
    }

    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1"
            << "<"
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << KPerBlock << ", "
            << AK1 << ", "
            << BK1 << ", "
            << getConvBackwardDataSpecializationString(ConvBackwardDataSpecialization) << ", "
            << MPerXDL << ", "
            << NPerXDL << ", "
            << MXdlPerWave << ", "
            << NXdlPerWave << ", "
            << ABlockTransferSrcScalarPerVector << ", "
            << BBlockTransferSrcScalarPerVector << ", "
            << CShuffleMXdlPerWavePerShuffle << ", "
            << CShuffleNXdlPerWavePerShuffle
            << ">";

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
