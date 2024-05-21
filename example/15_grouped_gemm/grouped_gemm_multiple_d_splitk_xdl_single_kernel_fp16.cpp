// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_gemm_multiple_d_splitk_xdl_cshuffle_tile_loop.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_gemm.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/fill.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F16 = ck::half_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using ADataType        = F16;
using BDataType        = F16;
using AccDataType      = F32;
using CShuffleDataType = F32;
using DsDataType       = ck::Tuple<>;
using EDataType        = F16;

using ALayout = Row;
using BLayout = Col;
// using BLayout  = Row;
using DsLayout = ck::Tuple<>;
using ELayout  = Row;

using AElementOp   = PassThrough;
using BElementOp   = PassThrough;
using CDEElementOp = PassThrough;

static constexpr auto GemmMNKPadding = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

using DeviceGemmInstance = ck::tensor_operation::device::DeviceGroupedGemmMultipleDSplitKXdlCShuffle
    // clang-format off
//######| ALayout| BLayout| DsLayout| ELayout|     AData|     BData|     AccData|         CShuffle|     DsData|     EData|           A|           B|          CDE|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
//######|        |        |         |        |      Type|      Type|        Type|         DataType|       Type|      Type| Elementwise| Elementwise|  Elementwise| Spacialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
//######|        |        |         |        |          |          |            |                 |           |          |   Operation|   Operation|    Operation|               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
//######|        |        |         |        |          |          |            |                 |           |          |            |            |             |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
        < ALayout, BLayout, DsLayout, ELayout, ADataType, BDataType, AccDataType, CShuffleDataType, DsDataType, EDataType,  AElementOp,  BElementOp, CDEElementOp, GemmMNKPadding,        1,   128,    64,    64,    32,   8,   8,   32,   32,    2,    1,  S<4, 32, 1>,  S<1, 0, 2>,  S<1, 0, 2>,              2,              8,              8,         1,  S<4, 32, 1>,  S<1, 0, 2>,  S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 16, 1, 4>,               8, ck::PipelineVersion::v1>;
        // < ALayout, BLayout, DsLayout, ELayout, ADataType, BDataType, AccDataType, CShuffleDataType, DsDataType, EDataType,  AElementOp,  BElementOp, CDEElementOp, GemmMNKPadding,        1,   256,    64,   128,    32,   8,   8,   32,   32,    1,    2,  S<4, 64, 1>,  S<1, 0, 2>,  S<1, 0, 2>,              2,              8,              8,         1,  S<4, 64, 1>,  S<1, 0, 2>,  S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 32, 1, 8>,              4>;
        // < ALayout, BLayout, DsLayout, ELayout, ADataType, BDataType, AccDataType, CShuffleDataType, DsDataType, EDataType,  AElementOp,  BElementOp, CDEElementOp, GemmMNKPadding,        1,   128,   192,    32,    32,   8,   8,   32,   32,    3,    1,  S<4, 32, 1>,  S<1, 0, 2>,  S<1, 0, 2>,              2,              8,              8,         1,  S<4, 32, 1>,  S<0, 2, 1>,  S<0, 2, 1>,             1,              1,              8,         1,           1,           1,               S<1, 32, 1, 4>,              8>;
// clang-format on

struct ProblemSize final
{
    std::vector<ck::index_t> Ms;
    std::vector<ck::index_t> Ns;
    std::vector<ck::index_t> Ks;

    std::vector<ck::index_t> stride_As;
    std::vector<ck::index_t> stride_Bs;
    std::vector<ck::index_t> stride_Cs;

    ck::index_t group_count;
};

struct ExecutionConfig final
{
    bool do_verification = true;
    int init_method      = 1;
    // int k_batch          = 128;
    int k_batch      = 1;
    bool time_kernel = false;
};

bool run_grouped_gemm(const ProblemSize& problem_size, const ExecutionConfig& config)
{
    auto group_count = problem_size.group_count;

    // GEMM shape
    std::vector<ck::tensor_operation::device::GemmDesc> gemm_descs;
    std::vector<void*> p_Cs;
    std::vector<const void*> p_As;
    std::vector<const void*> p_Bs;

    gemm_descs.reserve(group_count);
    p_As.reserve(group_count);
    p_Bs.reserve(group_count);

    auto f_host_tensor_descriptor =
        [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
            using namespace ck::literals;

            if(std::is_same<decltype(layout), ck::tensor_layout::gemm::RowMajor>::value)
            {
                return HostTensorDescriptor({row, col}, {stride, 1_uz});
            }
            else
            {
                return HostTensorDescriptor({row, col}, {1_uz, stride});
            }
        };

    std::vector<Tensor<ADataType>> a_tensors;
    std::vector<Tensor<BDataType>> b_tensors;
    std::vector<Tensor<EDataType>> c_host_tensors;
    std::vector<Tensor<EDataType>> c_device_tensors;

    a_tensors.reserve(group_count);
    b_tensors.reserve(group_count);
    c_host_tensors.reserve(group_count);
    c_device_tensors.reserve(group_count);

    using DeviceMemPtr = std::unique_ptr<DeviceMem>;

    std::vector<DeviceMemPtr> a_tensors_device, b_tensors_device, c_tensors_device;

    a_tensors_device.reserve(group_count);
    b_tensors_device.reserve(group_count);
    c_tensors_device.reserve(group_count);

    std::size_t flop = 0, num_btype = 0;

    for(int i = 0; i < group_count; i++)
    {
        a_tensors.push_back(Tensor<ADataType>(f_host_tensor_descriptor(
            problem_size.Ms[i], problem_size.Ks[i], problem_size.stride_As[i], ALayout{})));
        b_tensors.push_back(Tensor<BDataType>(f_host_tensor_descriptor(
            problem_size.Ks[i], problem_size.Ns[i], problem_size.stride_Bs[i], BLayout{})));
        c_host_tensors.push_back(Tensor<EDataType>(f_host_tensor_descriptor(
            problem_size.Ms[i], problem_size.Ns[i], problem_size.stride_Cs[i], ELayout{})));
        c_device_tensors.push_back(Tensor<EDataType>(f_host_tensor_descriptor(
            problem_size.Ms[i], problem_size.Ns[i], problem_size.stride_Cs[i], ELayout{})));
        std::cout << "gemm[" << i << "] a_m_k: " << a_tensors[i].mDesc
                  << " b_k_n: " << b_tensors[i].mDesc << " c_m_n: " << c_device_tensors[i].mDesc
                  << std::endl;

        flop += std::size_t(2) * problem_size.Ms[i] * problem_size.Ks[i] * problem_size.Ns[i];
        num_btype += sizeof(ADataType) * a_tensors[i].mDesc.GetElementSize() +
                     sizeof(BDataType) * b_tensors[i].mDesc.GetElementSize() +
                     sizeof(EDataType) * c_device_tensors[i].mDesc.GetElementSize();

        switch(config.init_method)
        {
        case 0: break;
        case 1:
            a_tensors[i].GenerateTensorValue(GeneratorTensor_2<ADataType>{-5, 5});
            b_tensors[i].GenerateTensorValue(GeneratorTensor_2<BDataType>{-5, 5});
            break;
        case 2:
            a_tensors[i].GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});
            b_tensors[i].GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5});
            break;
        case 3:
            ck::utils::FillConstant<ADataType>{1}(a_tensors[i]);
            ck::utils::FillConstant<BDataType>{1}(b_tensors[i]);
            break;
        default:
            // a_tensors[i].GenerateTensorValue(GeneratorTensor_Sequential<0>{});
            // b_tensors[i].GenerateTensorValue(GeneratorTensor_Sequential<1>{});
            ck::utils::FillMonotonicSeq<ADataType>{0, 1}(a_tensors[i]);
            ck::utils::FillMonotonicSeq<BDataType>{1, 1}(b_tensors[i]);
        }
    }

    using GroupedGemmKernelArgument =
        ck::tensor_operation::device::GroupedGemmMultipleDKernelArguments<>;

    std::vector<GroupedGemmKernelArgument> grouped_gemm_kernel_args_;
    grouped_gemm_kernel_args_.reserve(group_count);

    for(int i = 0; i < group_count; i++)
    {
        a_tensors_device.emplace_back(std::make_unique<DeviceMem>(
            a_tensors[i].mDesc.GetElementSpaceSize() * sizeof(ADataType)));

        b_tensors_device.emplace_back(std::make_unique<DeviceMem>(
            b_tensors[i].mDesc.GetElementSpaceSize() * sizeof(BDataType)));

        c_tensors_device.emplace_back(std::make_unique<DeviceMem>(
            c_device_tensors[i].mDesc.GetElementSpaceSize() * sizeof(EDataType)));

        a_tensors_device[i]->ToDevice(a_tensors[i].mData.data());
        b_tensors_device[i]->ToDevice(b_tensors[i].mData.data());
        c_tensors_device[i]->SetZero();

        p_As.push_back(a_tensors_device[i]->GetDeviceBuffer());
        p_Bs.push_back(b_tensors_device[i]->GetDeviceBuffer());
        p_Cs.push_back(c_tensors_device[i]->GetDeviceBuffer());

        gemm_descs.push_back({problem_size.Ms[i],
                              problem_size.Ns[i],
                              problem_size.Ks[i],
                              problem_size.stride_As[i],
                              problem_size.stride_Bs[i],
                              problem_size.stride_Cs[i],
                              {}});

        grouped_gemm_kernel_args_.push_back({a_tensors_device[i]->GetDeviceBuffer(),
                                             b_tensors_device[i]->GetDeviceBuffer(),
                                             {},
                                             c_tensors_device[i]->GetDeviceBuffer(),
                                             problem_size.Ms[i],
                                             problem_size.Ns[i],
                                             problem_size.Ks[i],
                                             problem_size.stride_As[i],
                                             problem_size.stride_Bs[i],
                                             {},
                                             problem_size.stride_Cs[i]});
    }

    auto a_element_op = AElementOp{};
    auto b_element_op = BElementOp{};
    auto c_element_op = CDEElementOp{};

    auto gemm    = DeviceGemmInstance{};
    auto invoker = gemm.MakeInvoker();

    std::vector<std::array<const void*, 0>> p_Ds = {};

    // do GEMM
    auto argument = gemm.MakeArgument(
        p_As, p_Bs, p_Ds, p_Cs, gemm_descs, a_element_op, b_element_op, c_element_op);

    DeviceMem gemm_arg_dev_mem(gemm.GetDeviceKernelArgSize(&argument));

    if(!gemm.IsSupportedArgument(argument))
    {
        throw std::runtime_error(
            "wrong! device_gemm with the specified compilation parameters does "
            "not support this GEMM problem");
    }

    gemm.SetDeviceKernelArgs(argument, gemm_arg_dev_mem.GetDeviceBuffer());
    gemm.SetKBatchSize(argument, config.k_batch);

    DeviceMem gemm_workspace_dev(gemm.GetWorkSpaceSize(&argument));
    gemm.SetWorkSpacePointer(&argument, gemm_workspace_dev.GetDeviceBuffer());

    invoker.Run(argument, StreamConfig{nullptr, false, 1});

    if(config.time_kernel)
    {
        float ave_time   = invoker.Run(argument, StreamConfig{nullptr, config.time_kernel});
        float tflops     = static_cast<float>(flop) / 1.E9 / ave_time;
        float gb_per_sec = num_btype / 1.E6 / ave_time;

        std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec
                  << " GB/s, " << gemm.GetTypeString() << std::endl;
    }

    bool pass = true;
    if(config.do_verification)
    {
        using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceGemm<ADataType,
                                                                                BDataType,
                                                                                EDataType,
                                                                                AccDataType,
                                                                                AElementOp,
                                                                                BElementOp,
                                                                                CDEElementOp>;

        for(std::size_t i = 0; i < gemm_descs.size(); i++)
        {
            c_tensors_device[i]->FromDevice(c_device_tensors[i].mData.data(),
                                            c_device_tensors[i].mDesc.GetElementSize() *
                                                sizeof(EDataType));
            auto ref_gemm    = ReferenceGemmInstance{};
            auto ref_invoker = ref_gemm.MakeInvoker();

            auto ref_argument = ref_gemm.MakeArgument(a_tensors[i],
                                                      b_tensors[i],
                                                      c_host_tensors[i],
                                                      a_element_op,
                                                      b_element_op,
                                                      c_element_op);

            ref_invoker.Run(ref_argument);
            pass &= ck::utils::check_err(c_device_tensors[i], c_host_tensors[i]);
        }

        std::cout << "Verification: " << (pass ? "SUCCESS" : "FAILURE") << "!" << std::endl;
    }

    return pass;
}

std::vector<int> argToIntArray(char* input)
{
    std::vector<int> out;

    std::istringstream in(input);

    std::string item;

    while(std::getline(in, item, ','))
    {
        out.push_back(std::stoi(item));
    }

    return out;
}

int main(int argc, char* argv[])
{
    ProblemSize problem_size;
    ExecutionConfig config;

    if(argc < 11)
    {
        // std::vector<ck::index_t> Ms{64, 127, 255, 129, 260, 190, 77};
        std::vector<ck::index_t> Ms{64};
        problem_size.group_count = Ms.size();

        for(int i = 0; i < problem_size.group_count; i++)
        {
            problem_size.Ms.push_back(Ms[i]);
            // problem_size.Ns.push_back(252);
            problem_size.Ns.push_back(256);
            problem_size.Ks.push_back(4608);

            problem_size.stride_As.push_back(problem_size.Ks[i]);
            problem_size.stride_Bs.push_back(problem_size.Ks[i]);
            // problem_size.stride_Bs.push_back(problem_size.Ns[i]);
            problem_size.stride_Cs.push_back(problem_size.Ns[i]);
        }

        std::cout
            << "Usage:\n"
            << "arg1: verification (0=no, 1=yes)\n"
            << "arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n"
            << "arg3: time kernel (0=n0, 1=yes)\n"
            << "arg4 to 9: Ms, Ns, Ks, StrideAs, StrideBs, StrideCs (e.g., 256,256 128,128 64,64 "
               "64,64 64,64 128,128)\n"
            << "arg10: k_batch (> 0)\n"
            << "... setting default values." << std::endl;
    }
    else
    {
        config.do_verification = std::stoi(argv[1]);
        config.init_method     = std::stoi(argv[2]);
        config.time_kernel     = std::stoi(argv[3]);
        config.k_batch         = std::stoi(argv[10]);

        problem_size.Ms = argToIntArray(argv[4]);
        problem_size.Ns = argToIntArray(argv[5]);
        problem_size.Ks = argToIntArray(argv[6]);

        problem_size.stride_As = argToIntArray(argv[7]);
        problem_size.stride_Bs = argToIntArray(argv[8]);
        problem_size.stride_Cs = argToIntArray(argv[9]);

        problem_size.group_count = problem_size.Ms.size();
    }

    return !run_grouped_gemm(problem_size, config);
}