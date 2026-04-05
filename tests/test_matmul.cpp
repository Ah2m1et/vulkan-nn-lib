#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

// CPU reference matmul: C = A * B
// A: M×K, B: K×N, C: M×N (row-major)
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
    // 2×3 * 3×2 = 2×2
    const int M = 2, K = 3, N = 2;
    float A[M * K] = { 1,2,3, 4,5,6 };
    float B[K * N] = { 7,8, 9,10, 11,12 };
    float C[M * N] = {};

    cpu_matmul(A, B, C, M, N, K);

    // C[0][0] = 1*7+2*9+3*11 = 7+18+33 = 58
    assert(std::fabs(C[0] -  58.0f) < 1e-4f);
    // C[0][1] = 1*8+2*10+3*12 = 8+20+36 = 64
    assert(std::fabs(C[1] -  64.0f) < 1e-4f);
    // C[1][0] = 4*7+5*9+6*11 = 28+45+66 = 139
    assert(std::fabs(C[2] - 139.0f) < 1e-4f);
    // C[1][1] = 4*8+5*10+6*12 = 32+50+72 = 154
    assert(std::fabs(C[3] - 154.0f) < 1e-4f);

    // TODO (Week 4): dispatch matmul.spv on GPU, compare with C[]

    printf("test_matmul PASSED (CPU reference)\n");
    return 0;
}
