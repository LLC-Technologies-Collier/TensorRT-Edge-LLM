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

#include <thread>
#include <chrono>
#include <atomic>
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <malloc.h>
#include <cstdio>

namespace trt_edgellm
{
namespace rt
{

/*!
 * @brief Background thread that monitors GPU and System memory and triggers reclamation
 */
class MemoryMonitor
{
public:
    MemoryMonitor(int intervalSeconds = 5)
        : mInterval(intervalSeconds)
        , mStop(false)
    {
    }

    ~MemoryMonitor()
    {
        stop();
    }

    void start()
    {
        mThread = std::thread(&MemoryMonitor::monitorLoop, this);
    }

    void stop()
    {
        mStop = true;
        if (mThread.joinable())
        {
            mThread.join();
        }
    }

private:
    void monitorLoop()
    {
        while (!mStop)
        {
            size_t free_mem, total_mem;
            if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess)
            {
                double used_gb = static_cast<double>(total_mem - free_mem) / (1024.0 * 1024.0 * 1024.0);
                double total_gb = static_cast<double>(total_mem) / (1024.0 * 1024.0 * 1024.0);

                // Get system memory from /proc/meminfo
                size_t sys_avail_kb = 0;
                std::ifstream meminfo("/proc/meminfo");
                std::string line;
                if (meminfo.is_open())
                {
                    while (std::getline(meminfo, line))
                    {
                        if (line.compare(0, 13, "MemAvailable:") == 0)
                        {
                            sscanf(line.c_str(), "MemAvailable: %zu", &sys_avail_kb);
                            break;
                        }
                    }
                    meminfo.close();
                }

                printf("[C++ Monitor] GPU Used: %.2f/%.2f GB | Sys Avail: %.2f GB\n", used_gb, total_gb,
                    static_cast<double>(sys_avail_kb) / (1024.0 * 1024.0));
                fflush(stdout);

                // Proactive release if pressure is high (< 4GB avail)
                if (sys_avail_kb > 0 && sys_avail_kb < 4 * 1024 * 1024)
                {
                    printf("[C++ Monitor] High memory pressure detected! Reclaiming spare memory...\n");
                    fflush(stdout);
                    
                    cudaDeviceSynchronize();
                    int device = 0;
                    if (cudaGetDevice(&device) == cudaSuccess)
                    {
                        cudaMemPool_t memPool;
                        if (cudaDeviceGetDefaultMemPool(&memPool, device) == cudaSuccess)
                        {
                            cudaMemPoolTrimTo(memPool, 0);
                        }
                    }
                    malloc_trim(0);
                }
            }

            // Sleep in small increments to respond quickly to stop signal
            for (int i = 0; i < mInterval * 10 && !mStop; ++i)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
    }

    int mInterval;
    std::atomic<bool> mStop;
    std::thread mThread;
};

} // namespace rt
} // namespace trt_edgellm
