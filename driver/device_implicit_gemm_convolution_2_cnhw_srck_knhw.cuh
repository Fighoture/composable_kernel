#pragma once
#include "gridwise_implicit_gemm_convolution_2_cnhw_srck_knhw.cuh"
#include "gridwise_implicit_gemm_convolution_2_cnhw_srck_knhw_lds_pipeline.cuh"
#include <unistd.h>

template <class T, class InDesc, class WeiDesc, class OutDesc>
void device_implicit_gemm_convolution_2_cnhw_srck_knhw(InDesc,
                                                       const Tensor<T>& in_nchw,
                                                       WeiDesc,
                                                       const Tensor<T>& wei_kcsr,
                                                       OutDesc,
                                                       Tensor<T>& out_nkhw,
                                                       unsigned nrepeat)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    constexpr auto in_nchw_desc  = InDesc{};
    constexpr auto wei_kcsr_desc = WeiDesc{};
    constexpr auto out_nkhw_desc = OutDesc{};

    constexpr unsigned N  = in_nchw_desc.GetLength(I0);
    constexpr unsigned Hi = in_nchw_desc.GetLength(I2);
    constexpr unsigned Wi = in_nchw_desc.GetLength(I3);

    constexpr unsigned Ho = out_nkhw_desc.GetLength(I2);
    constexpr unsigned Wo = out_nkhw_desc.GetLength(I3);

    constexpr unsigned K = wei_kcsr_desc.GetLength(I0);
    constexpr unsigned C = wei_kcsr_desc.GetLength(I1);
    constexpr unsigned S = wei_kcsr_desc.GetLength(I2);
    constexpr unsigned R = wei_kcsr_desc.GetLength(I3);

    constexpr unsigned BGhostRead = (S - 1) * Wi + (R - 1);

    // convert in_nchw to in_cnhw
    auto in_cnhw_desc = make_ConstantTensorDescriptor(Sequence<C, N, Hi, Wi>{});
    ostream_ConstantTensorDescriptor(in_cnhw_desc, std::cout << "in_cnhw_desc: ");

    Tensor<T> in_cnhw(make_TensorDescriptor(in_cnhw_desc));

    auto f_reorder_nchw2cnhw = [&](auto n, auto c, auto hi, auto wi) {
        in_cnhw(c, n, hi, wi) = in_nchw(n, c, hi, wi);
    };

    make_ParallelTensorFunctor(f_reorder_nchw2cnhw, N, C, Hi, Wi)(
        std::thread::hardware_concurrency());

    // convert wei_kcsr to wei_srck
    auto wei_srck_desc = make_ConstantTensorDescriptor(Sequence<S, R, C, K>{});
    ostream_ConstantTensorDescriptor(wei_srck_desc, std::cout << "wei_srck_desc: ");

    Tensor<T> wei_srck(make_TensorDescriptor(wei_srck_desc));

    auto f_reorder_kcsr2srck = [&](auto k, auto c, auto s, auto r) {
        wei_srck(s, r, c, k) = wei_kcsr(k, c, s, r);
    };

    make_ParallelTensorFunctor(f_reorder_kcsr2srck, K, C, S, R)(
        std::thread::hardware_concurrency());

    // conver out_nkhw to out_knhw
    auto out_knhw_desc = make_ConstantTensorDescriptor(Sequence<K, N, Ho, Wo>{});
    ostream_ConstantTensorDescriptor(out_knhw_desc, std::cout << "out_knhw_desc: ");

    Tensor<T> out_knhw(make_TensorDescriptor(out_knhw_desc));

#if 0
    constexpr unsigned BPerBlock = 256;
    constexpr unsigned KPerBlock = 1;
    constexpr unsigned CPerBlock = 1;

    constexpr unsigned BPerThread = 8;
    constexpr unsigned KPerThread = 1;
    constexpr unsigned CPerThread = 1;

    constexpr unsigned GemmThreadPerColumnPerCluster    = 1;
    constexpr unsigned GemmThreadPerRowPerCluster = 4;

    constexpr unsigned BlockSize = 32;
#elif 0
    constexpr unsigned BPerBlock = 128;
    constexpr unsigned KPerBlock = 64;
    constexpr unsigned CPerBlock = 2;

    constexpr unsigned BPerThread = 8;
    constexpr unsigned KPerThread = 8;
    constexpr unsigned CPerThread = 1;

    constexpr unsigned GemmThreadPerColumnPerCluster = 4;
    constexpr unsigned GemmThreadPerRowPerCluster    = 4;

    constexpr unsigned BlockSize = 128;
#elif 0
    constexpr unsigned BPerBlock = 128;
    constexpr unsigned KPerBlock = 64;
    constexpr unsigned CPerBlock = 2;

    constexpr unsigned BPerThread = 8;
    constexpr unsigned KPerThread = 8;
    constexpr unsigned CPerThread = 1;

    constexpr unsigned GemmThreadPerColumnPerCluster = 4;
    constexpr unsigned GemmThreadPerRowPerCluster    = 4;

    constexpr unsigned InBlockCopyThreadPerDim0 = 2;
    constexpr unsigned InBlockCopyThreadPerDim1 = 64;

    constexpr unsigned BlockSize = 128;
#elif 1
    // 1x1, 28x28
    constexpr unsigned BPerBlock = 64;
    constexpr unsigned KPerBlock = 128;
    constexpr unsigned CPerBlock = 8;

    constexpr unsigned BPerThread = 4;
    constexpr unsigned KPerThread = 16;
    constexpr unsigned CPerThread = 2;

    constexpr unsigned GemmThreadPerColumnPerCluster = 8;
    constexpr unsigned GemmThreadPerRowPerCluster    = 8;

    constexpr unsigned InBlockCopyThreadPerDim0 = 8;
    constexpr unsigned InBlockCopyThreadPerDim1 = 16;

    constexpr unsigned BlockSize = 128;
#endif

    constexpr unsigned GridSize =
        ((N * Hi * Wi + BPerBlock - 1) / BPerBlock) * ((K + KPerBlock - 1) / KPerBlock);

    dim3 block_dim(BlockSize);
    dim3 grid_dim(GridSize);

    printf("%s: BlockSize %u, GridSize %u \n", __func__, BlockSize, GridSize);

    // mem
    std::size_t data_sz = sizeof(T);
    DeviceMem in_cnhw_device_buf(data_sz * (in_cnhw.mDesc.GetElementSpace() + BGhostRead +
                                            BPerBlock)); // reserve extra space for BGhostRead
    DeviceMem wei_srck_device_buf(data_sz * wei_srck.mDesc.GetElementSpace());
    DeviceMem out_knhw_device_buf(data_sz * out_knhw.mDesc.GetElementSpace());

    in_cnhw_device_buf.ToDevice(in_cnhw.mData.data());
    wei_srck_device_buf.ToDevice(wei_srck.mData.data());
    out_knhw_device_buf.ToDevice(out_knhw.mData.data());

    for(unsigned i = 0; i < nrepeat; ++i)
    {
        cudaEvent_t start, stop;
        float elapsedTime;
        cudaEventCreate(&start);
        cudaEventRecord(start, 0);

#if 0
        gridwise_implicit_gemm_convolution_2_cnhw_srck_knhw
#else
        gridwise_implicit_gemm_convolution_2_cnhw_srck_knhw_lds_pipeline
#endif
        <GridSize,
         BlockSize,
         T,
         decltype(in_cnhw_desc),
         decltype(wei_srck_desc),
         decltype(out_knhw_desc),
         BPerBlock,
         KPerBlock,
         CPerBlock,
         BPerThread,
         KPerThread,
         CPerThread,
         GemmThreadPerColumnPerCluster,
         GemmThreadPerRowPerCluster,
         InBlockCopyThreadPerDim0,
         InBlockCopyThreadPerDim1>
            <<<grid_dim, block_dim>>>(in_cnhw_desc,
                                      static_cast<T*>(in_cnhw_device_buf.GetDeviceBuffer()),
                                      wei_srck_desc,
                                      static_cast<T*>(wei_srck_device_buf.GetDeviceBuffer()),
                                      out_knhw_desc,
                                      static_cast<T*>(out_knhw_device_buf.GetDeviceBuffer()));

        cudaEventCreate(&stop);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&elapsedTime, start, stop);
        printf("Elapsed time : %f ms\n", elapsedTime);

        usleep(std::min(elapsedTime * 1000, float(10000)));
    }

    checkCudaErrors(cudaGetLastError());
    out_knhw_device_buf.FromDevice(out_knhw.mData.data());

    // convert out_knhw to out_nkhw
    auto f_reorder_knhw2nkhw = [&](auto n, auto k, auto ho, auto wo) {
        out_nkhw(n, k, ho, wo) = out_knhw(k, n, ho, wo);
    };

    make_ParallelTensorFunctor(f_reorder_knhw2nkhw, N, K, Ho, Wo)(
        std::thread::hardware_concurrency());
}
