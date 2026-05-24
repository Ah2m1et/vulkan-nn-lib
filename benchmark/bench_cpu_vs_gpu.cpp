#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

#include "vk_context.h"
#include "vk_buffer.h"
#include "vk_pipeline.h"

#ifndef SHADER_DIR
#define SHADER_DIR "."
#endif

using Clock = std::chrono::high_resolution_clock;
using ms_d  = std::chrono::duration<double, std::milli>;

static void cpu_matmul(const float* A, const float* B, float* C,
                       uint32_t M, uint32_t N, uint32_t K) {
    for (uint32_t i = 0; i < M; ++i)
        for (uint32_t j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (uint32_t k = 0; k < K; ++k)
                acc += A[i * K + k] * B[k * N + j];
            C[i * N + j] = acc;
        }
}

struct PushConstants { uint32_t M, N, K; };

// Returns total wall time: upload + dispatch + download (ms).
static double bench_gpu_once(const VkContext& ctx, const ComputePipeline& pipe,
                             const std::vector<float>& A, const std::vector<float>& B,
                             uint32_t M, uint32_t N, uint32_t K) {
    VkDeviceSize sA = M * K * sizeof(float);
    VkDeviceSize sB = K * N * sizeof(float);
    VkDeviceSize sC = M * N * sizeof(float);

    auto t0 = Clock::now();

    GpuBuffer bufA = createBuffer(ctx, sA);
    GpuBuffer bufB = createBuffer(ctx, sB);
    GpuBuffer bufC = createBuffer(ctx, sC);

    uploadData(ctx, bufA, A.data(), sA);
    uploadData(ctx, bufB, B.data(), sB);

    PushConstants pc{M, N, K};
    constexpr uint32_t TILE = 16;
    dispatchPipeline(ctx, pipe,
                     {&bufA, &bufB, &bufC},
                     &pc, sizeof(pc),
                     (N + TILE - 1) / TILE,
                     (M + TILE - 1) / TILE);

    std::vector<float> C_out(M * N);
    downloadData(ctx, bufC, C_out.data(), sC);

    auto t1 = Clock::now();

    destroyBuffer(ctx, bufA);
    destroyBuffer(ctx, bufB);
    destroyBuffer(ctx, bufC);

    return ms_d(t1 - t0).count();
}

int main() {
    constexpr int RUNS = 5;
    const uint32_t sizes[] = {128, 256, 512};

    VkContext ctx = createContext();
    auto spirv = loadSPIRV(std::string(SHADER_DIR) + "/matmul.spv");
    ComputePipeline pipe = createComputePipeline(ctx, spirv,
                                                 /*bindingCount=*/3,
                                                 /*pushConstantSize=*/sizeof(PushConstants));

    printf("\n%-8s  %-14s  %-14s  %-8s\n",
           "Size", "CPU avg (ms)", "GPU avg (ms)", "Speedup");
    printf("%-8s  %-14s  %-14s  %-8s\n",
           "--------", "--------------", "--------------", "--------");

    for (uint32_t sz : sizes) {
        uint32_t M = sz, N = sz, K = sz;

        std::vector<float> A(M * K), B(K * N), C(M * N);
        srand(42);
        for (auto& v : A) v = (rand() % 200 - 100) / 50.0f;
        for (auto& v : B) v = (rand() % 200 - 100) / 50.0f;

        // --- CPU benchmark ---
        std::vector<double> cpu_times(RUNS);
        for (int r = 0; r < RUNS; ++r) {
            auto t0 = Clock::now();
            cpu_matmul(A.data(), B.data(), C.data(), M, N, K);
            auto t1 = Clock::now();
            cpu_times[r] = ms_d(t1 - t0).count();
        }
        double cpu_avg = std::accumulate(cpu_times.begin(), cpu_times.end(), 0.0) / RUNS;

        // --- GPU benchmark (1 warm-up, sonra RUNS ölçüm) ---
        bench_gpu_once(ctx, pipe, A, B, M, N, K); // warm-up
        std::vector<double> gpu_times(RUNS);
        for (int r = 0; r < RUNS; ++r)
            gpu_times[r] = bench_gpu_once(ctx, pipe, A, B, M, N, K);
        double gpu_avg = std::accumulate(gpu_times.begin(), gpu_times.end(), 0.0) / RUNS;

        printf("%-8u  %-14.3f  %-14.3f  %-8.2f\n",
               sz, cpu_avg, gpu_avg, cpu_avg / gpu_avg);
    }

    destroyComputePipeline(ctx, pipe);
    destroyContext(ctx);

    return 0;
}
