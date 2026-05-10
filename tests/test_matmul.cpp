#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "vk_context.h"
#include "vk_buffer.h"
#include "vk_pipeline.h"

#ifndef SHADER_DIR
#define SHADER_DIR "."
#endif

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

static void run_gpu_matmul(const VkContext& ctx,
                           const ComputePipeline& pipe,
                           const float* A, const float* B, float* C_gpu,
                           uint32_t M, uint32_t N, uint32_t K) {
    VkDeviceSize sizeA = M * K * sizeof(float);
    VkDeviceSize sizeB = K * N * sizeof(float);
    VkDeviceSize sizeC = M * N * sizeof(float);

    GpuBuffer bufA = createBuffer(ctx, sizeA);
    GpuBuffer bufB = createBuffer(ctx, sizeB);
    GpuBuffer bufC = createBuffer(ctx, sizeC);

    uploadData(ctx, bufA, A, sizeA);
    uploadData(ctx, bufB, B, sizeB);

    PushConstants pc{M, N, K};
    constexpr uint32_t TILE = 16;
    dispatchPipeline(ctx, pipe,
                     {&bufA, &bufB, &bufC},
                     &pc, sizeof(pc),
                     (N + TILE - 1) / TILE,
                     (M + TILE - 1) / TILE);

    downloadData(ctx, bufC, C_gpu, sizeC);

    destroyBuffer(ctx, bufA);
    destroyBuffer(ctx, bufB);
    destroyBuffer(ctx, bufC);
}

int main() {
    VkContext ctx = createContext();

    auto spirv = loadSPIRV(std::string(SHADER_DIR) + "/matmul.spv");
    ComputePipeline pipe = createComputePipeline(ctx, spirv,
                                                 /*bindingCount=*/3,
                                                 /*pushConstantSize=*/sizeof(PushConstants));

    // --- Test 1: 2×3 * 3×2 = 2×2 (known values) ---
    {
        const uint32_t M = 2, K = 3, N = 2;
        float A[] = { 1,2,3, 4,5,6 };
        float B[] = { 7,8, 9,10, 11,12 };
        float C_ref[M * N]{};
        float C_gpu[M * N]{};

        cpu_matmul(A, B, C_ref, M, N, K);
        run_gpu_matmul(ctx, pipe, A, B, C_gpu, M, N, K);

        for (uint32_t i = 0; i < M * N; ++i) {
            if (std::fabs(C_gpu[i] - C_ref[i]) >= 1e-3f) {
                printf("Test1 MISMATCH at [%u]: GPU=%.4f expected=%.4f\n",
                       i, C_gpu[i], C_ref[i]);
                return 1;
            }
        }
        printf("Test 1 (2x3 * 3x2) PASSED\n");
    }

    // --- Test 2: 64×64 * 64×64 (multi-tile) ---
    {
        const uint32_t M = 64, K = 64, N = 64;
        std::vector<float> A(M * K), B(K * N), C_ref(M * N), C_gpu(M * N);

        srand(42);
        for (auto& v : A) v = (rand() % 200 - 100) / 10.0f;
        for (auto& v : B) v = (rand() % 200 - 100) / 10.0f;

        cpu_matmul(A.data(), B.data(), C_ref.data(), M, N, K);
        run_gpu_matmul(ctx, pipe, A.data(), B.data(), C_gpu.data(), M, N, K);

        for (uint32_t i = 0; i < M * N; ++i) {
            if (std::fabs(C_gpu[i] - C_ref[i]) >= 1e-1f) {
                printf("Test2 MISMATCH at [%u]: GPU=%.4f expected=%.4f\n",
                       i, C_gpu[i], C_ref[i]);
                return 1;
            }
        }
        printf("Test 2 (64x64 * 64x64) PASSED\n");
    }

    // --- Test 3: non-power-of-2 boyutlar (tile sınır durumu) ---
    {
        const uint32_t M = 17, K = 33, N = 19;
        std::vector<float> A(M * K), B(K * N), C_ref(M * N), C_gpu(M * N);

        srand(7);
        for (auto& v : A) v = (rand() % 20 - 10) / 5.0f;
        for (auto& v : B) v = (rand() % 20 - 10) / 5.0f;

        cpu_matmul(A.data(), B.data(), C_ref.data(), M, N, K);
        run_gpu_matmul(ctx, pipe, A.data(), B.data(), C_gpu.data(), M, N, K);

        for (uint32_t i = 0; i < M * N; ++i) {
            if (std::fabs(C_gpu[i] - C_ref[i]) >= 1e-1f) {
                printf("Test3 MISMATCH at [%u]: GPU=%.4f expected=%.4f\n",
                       i, C_gpu[i], C_ref[i]);
                return 1;
            }
        }
        printf("Test 3 (17x33 * 33x19, non-power-of-2) PASSED\n");
    }

    destroyComputePipeline(ctx, pipe);
    destroyContext(ctx);

    printf("test_matmul PASSED (CPU reference + GPU dispatch, 3 test cases)\n");
    return 0;
}
