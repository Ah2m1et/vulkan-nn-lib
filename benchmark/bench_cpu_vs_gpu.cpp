#include <chrono>
#include <cstdio>
#include <vector>
#include <numeric>

// CPU matmul (row-major, no optimisation)
static void cpu_matmul(const float* A, const float* B, float* C,
                       int M, int N, int K) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k)
                acc += A[i * K + k] * B[k * N + j];
            C[i * N + j] = acc;
        }
}

int main() {
    constexpr int M = 256, N = 256, K = 256;
    constexpr int RUNS = 5;

    std::vector<float> A(M * K, 1.0f);
    std::vector<float> B(K * N, 1.0f);
    std::vector<float> C(M * N, 0.0f);

    // --- CPU benchmark ---
    std::vector<double> cpu_ms(RUNS);
    for (int r = 0; r < RUNS; ++r) {
        auto t0 = std::chrono::high_resolution_clock::now();
        cpu_matmul(A.data(), B.data(), C.data(), M, N, K);
        auto t1 = std::chrono::high_resolution_clock::now();
        cpu_ms[r] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    double cpu_avg = std::accumulate(cpu_ms.begin(), cpu_ms.end(), 0.0) / RUNS;
    printf("[CPU] %dx%d matmul avg over %d runs: %.3f ms\n", M, N, RUNS, cpu_avg);

    // --- GPU benchmark ---
    // TODO (Week 5): create VkContext, allocate GpuBuffers,
    //   upload A/B, dispatch matmul.spv, download C, measure total time.
    printf("[GPU] TODO (Week 5)\n");

    return 0;
}
