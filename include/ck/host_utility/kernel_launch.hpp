// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <hip/hip_runtime.h>

#include "ck/utility/common_header.hpp"
#include "ck/stream_config.hpp"
#include "ck/host_utility/hip_check_error.hpp"
#include "ck/host_utility/device_prop.hpp"

template <typename... Args, typename F>
float launch_and_time_kernel(const StreamConfig& stream_config,
                             F kernel,
                             dim3 grid_dim,
                             dim3 block_dim,
                             std::size_t lds_byte,
                             Args... args)
{
#if CK_TIME_KERNEL
    if(stream_config.time_kernel_)
    {
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
        if(ck::get_device_name() == "gfx940" || ck::get_device_name() == "gfx941" ||
           ck::get_device_name() == "gfx942")
        {
            hipEvent_t start, stop;

            hip_check_error(hipEventCreate(&start));
            hip_check_error(hipEventCreate(&stop));

            hip_check_error(hipDeviceSynchronize());
            hip_check_error(hipEventRecord(start, stream_config.stream_id_));

            for(int i = 0; i < stream_config.nrepeat_; ++i)
            {
                kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args...);
                hip_check_error(hipGetLastError());
            }

            hip_check_error(hipEventRecord(stop, stream_config.stream_id_));
            hip_check_error(hipEventSynchronize(stop));

            float total_time = 0;

            hip_check_error(hipEventElapsedTime(&total_time, start, stop));
            total_time /= stream_config.nrepeat_;
            stream_config.cold_niters_ =
                (stream_config.time_limit_ms /
                 total_time); // we need longer runtime to ramp up the clk on MI300s
            stream_config.nrepeat_ = stream_config.cold_niters_;
        }
#endif
#if DEBUG_LOG
        printf("%s: grid_dim {%d, %d, %d}, block_dim {%d, %d, %d} \n",
               __func__,
               grid_dim.x,
               grid_dim.y,
               grid_dim.z,
               block_dim.x,
               block_dim.y,
               block_dim.z);

        printf("Warm up %d times\n", stream_config.cold_niters_);
#endif
        // warm up
        for(int i = 0; i < stream_config.cold_niters_; ++i)
        {
            kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args...);
            hip_check_error(hipGetLastError());
        }

        const int nrepeat = stream_config.nrepeat_;
#if DEBUG_LOG
        printf("Start running %d times...\n", nrepeat);
#endif
        hipEvent_t start, stop;

        hip_check_error(hipEventCreate(&start));
        hip_check_error(hipEventCreate(&stop));

        hip_check_error(hipDeviceSynchronize());
        hip_check_error(hipEventRecord(start, stream_config.stream_id_));

        for(int i = 0; i < nrepeat; ++i)
        {
            kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args...);
            hip_check_error(hipGetLastError());
        }

        hip_check_error(hipEventRecord(stop, stream_config.stream_id_));
        hip_check_error(hipEventSynchronize(stop));

        float total_time = 0;

        hip_check_error(hipEventElapsedTime(&total_time, start, stop));

        return total_time / nrepeat;
    }
    else
    {
        kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args...);
        hip_check_error(hipGetLastError());

        return 0;
    }
#else
    kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args...);
    hip_check_error(hipGetLastError());

    return 0;
#endif
}

template <typename... Args, typename F, typename PreProcessFunc>
float launch_and_time_kernel_with_preprocess(const StreamConfig& stream_config,
                                             PreProcessFunc preprocess,
                                             F kernel,
                                             dim3 grid_dim,
                                             dim3 block_dim,
                                             std::size_t lds_byte,
                                             Args... args)
{
#if CK_TIME_KERNEL
    if(stream_config.time_kernel_)
    {
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
        if(ck::get_device_name() == "gfx940" || ck::get_device_name() == "gfx941" ||
           ck::get_device_name() == "gfx942")
        {
            hipEvent_t start, stop;

            hip_check_error(hipEventCreate(&start));
            hip_check_error(hipEventCreate(&stop));

            hip_check_error(hipDeviceSynchronize());
            hip_check_error(hipEventRecord(start, stream_config.stream_id_));

            for(int i = 0; i < stream_config.nrepeat_; ++i)
            {
                kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args...);
                hip_check_error(hipGetLastError());
            }

            hip_check_error(hipEventRecord(stop, stream_config.stream_id_));
            hip_check_error(hipEventSynchronize(stop));

            float total_time = 0;

            hip_check_error(hipEventElapsedTime(&total_time, start, stop));
            total_time /= stream_config.nrepeat_;
            stream_config.cold_niters_ =
                (stream_config.nrepeat_ /
                 total_time); // we need longer runtime to ramp up the clk on MI300s
            stream_config.nrepeat_ = stream_config.cold_niters_;
        }
#endif
#if DEBUG_LOG
        printf("%s: grid_dim {%d, %d, %d}, block_dim {%d, %d, %d} \n",
               __func__,
               grid_dim.x,
               grid_dim.y,
               grid_dim.z,
               block_dim.x,
               block_dim.y,
               block_dim.z);

        printf("Warm up %d times\n", stream_config.cold_niters_);
#endif
        // warm up
        preprocess();
        for(int i = 0; i < stream_config.cold_niters_; ++i)
        {
            kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args...);
            hip_check_error(hipGetLastError());
        }

        const int nrepeat = stream_config.nrepeat_;
#if DEBUG_LOG
        printf("Start running %d times...\n", nrepeat);
#endif
        hipEvent_t start, stop;

        hip_check_error(hipEventCreate(&start));
        hip_check_error(hipEventCreate(&stop));

        hip_check_error(hipDeviceSynchronize());
        hip_check_error(hipEventRecord(start, stream_config.stream_id_));

        for(int i = 0; i < nrepeat; ++i)
        {
            preprocess();
            kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args...);
            hip_check_error(hipGetLastError());
        }

        hip_check_error(hipEventRecord(stop, stream_config.stream_id_));
        hip_check_error(hipEventSynchronize(stop));

        float total_time = 0;

        hip_check_error(hipEventElapsedTime(&total_time, start, stop));

        return total_time / nrepeat;
    }
    else
    {
        preprocess();
        kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args...);
        hip_check_error(hipGetLastError());

        return 0;
    }
#else
    kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args...);
    hip_check_error(hipGetLastError());

    return 0;
#endif
}
