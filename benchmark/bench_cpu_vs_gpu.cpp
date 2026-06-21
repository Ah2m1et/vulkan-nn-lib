// ---------------------------------------------------------------------------
// bench_cpu_vs_gpu.cpp
//
// ADİL kıyas tabanı (danışman geri bildirimi #3).
//
// Eski sürüm GPU'yu yalnızca OPTİMİZE EDİLMEMİŞ, TEK İŞ PARÇACIKLI naif bir CPU
// referansıyla kıyaslıyordu — bu, GPU lehine yanıltıcı hızlanma sayıları üretir.
// Bu sürüm aynı GEMM'i dört farklı temelle ölçer:
//
//   1. CPU naif    : tek-thread üçlü döngü (referans / taban)
//   2. CPU çok-thread: std::thread ile satır-paralel (donanımın tüm çekirdekleri)
//   3. CPU OpenBLAS : cblas_sgemm — vektörize + çok-thread, endüstri standardı
//                     (HAVE_OPENBLAS tanımlıysa)
//   4. GPU Vulkan  : upload + tiled GEMM dispatch + download (toplam duvar saati)
//
// Hızlanma sütunları HER ZAMAN en güçlü CPU temeline (varsa OpenBLAS) görelidir,
// böylece "GPU 11× hızlı" gibi yanıltıcı iddialardan kaçınılır.
// ---------------------------------------------------------------------------

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <thread>
#include <vector>

#include "vk_context.h"
#include "vk_buffer.h"
#include "vk_pipeline.h"

#ifdef HAVE_OPENBLAS
#include <cblas.h>
#endif

#ifndef SHADER_DIR
#define SHADER_DIR "."
#endif

using Clock = std::chrono::high_resolution_clock;
using ms_d  = std::chrono::duration<double, std::milli>;

// --- 1. Naif tek-thread GEMM (C = A*B) ---
static void cpu_matmul_naive(const float* A, const float* B, float* C,
                             uint32_t M, uint32_t N, uint32_t K) {
    for (uint32_t i = 0; i < M; ++i)
        for (uint32_t j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (uint32_t k = 0; k < K; ++k)
                acc += A[i * K + k] * B[k * N + j];
            C[i * N + j] = acc;
        }
}

// --- 2. Çok-thread GEMM: M satırını çekirdeklere böl ---
static void cpu_matmul_threaded(const float* A, const float* B, float* C,
                                uint32_t M, uint32_t N, uint32_t K,
                                unsigned nThreads) {
    auto worker = [&](uint32_t rowBegin, uint32_t rowEnd) {
        for (uint32_t i = rowBegin; i < rowEnd; ++i)
            for (uint32_t j = 0; j < N; ++j) {
                float acc = 0.0f;
                for (uint32_t k = 0; k < K; ++k)
                    acc += A[i * K + k] * B[k * N + j];
                C[i * N + j] = acc;
            }
    };
    std::vector<std::thread> pool;
    uint32_t chunk = (M + nThreads - 1) / nThreads;
    for (unsigned t = 0; t < nThreads; ++t) {
        uint32_t b = t * chunk, e = std::min(M, b + chunk);
        if (b < e) pool.emplace_back(worker, b, e);
    }
    for (auto& th : pool) th.join();
}

#ifdef HAVE_OPENBLAS
// --- 3. OpenBLAS sgemm ---
static void cpu_matmul_blas(const float* A, const float* B, float* C,
                            uint32_t M, uint32_t N, uint32_t K) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                (int)M, (int)N, (int)K,
                1.0f, A, (int)K, B, (int)N,
                0.0f, C, (int)N);
}
#endif

struct PushConstants { uint32_t M, N, K; };

// --- 4. GPU: toplam duvar saati (upload + dispatch + download) ---
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
    dispatchPipeline(ctx, pipe, {&bufA, &bufB, &bufC}, &pc, sizeof(pc),
                     (N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    std::vector<float> C_out(M * N);
    downloadData(ctx, bufC, C_out.data(), sC);

    auto t1 = Clock::now();
    destroyBuffer(ctx, bufA);
    destroyBuffer(ctx, bufB);
    destroyBuffer(ctx, bufC);
    return ms_d(t1 - t0).count();
}

template <typename F>
static double timeAvg(F&& fn, int runs) {
    std::vector<double> t(runs);
    for (int r = 0; r < runs; ++r) {
        auto t0 = Clock::now();
        fn();
        t[r] = ms_d(Clock::now() - t0).count();
    }
    return std::accumulate(t.begin(), t.end(), 0.0) / runs;
}

int main() {
    constexpr int RUNS = 5;
    const uint32_t sizes[] = {128, 256, 512};

    VkContext ctx = createContext();

    // Hangi GPU/backend ölçülüyor? Dürüst raporlama için cihaz adını yazdır.
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(ctx.physDevice, &props);
    unsigned hw = std::max(1u, std::thread::hardware_concurrency());

    printf("\n=== Adil GEMM Kiyas Tabani ===\n");
    printf("GPU/backend : %s\n", props.deviceName);
    printf("CPU threads : %u\n", hw);
#ifdef HAVE_OPENBLAS
    printf("OpenBLAS    : etkin (cblas_sgemm)\n");
#else
    printf("OpenBLAS    : YOK (yeniden derleyin: cmake ... ile libopenblas-dev kurulu)\n");
#endif
    printf("Not: GPU suresi upload+dispatch+download dahil toplam duvar saatidir.\n\n");

    printf("%-7s  %-12s  %-12s  %-12s  %-12s  %-16s\n",
           "Boyut", "Naif(ms)", "Cok-th(ms)", "OpenBLAS(ms)", "GPU(ms)",
           "GPU vs en-iyi-CPU");
    printf("%-7s  %-12s  %-12s  %-12s  %-12s  %-16s\n",
           "-------", "------------", "------------", "------------",
           "------------", "----------------");

    auto spirv = loadSPIRV(std::string(SHADER_DIR) + "/matmul.spv");
    ComputePipeline pipe = createComputePipeline(ctx, spirv, 3, sizeof(PushConstants));

    for (uint32_t sz : sizes) {
        uint32_t M = sz, N = sz, K = sz;
        std::vector<float> A(M * K), B(K * N), C(M * N);
        srand(42);
        for (auto& v : A) v = (rand() % 200 - 100) / 50.0f;
        for (auto& v : B) v = (rand() % 200 - 100) / 50.0f;

        double naiveMs = timeAvg([&] {
            cpu_matmul_naive(A.data(), B.data(), C.data(), M, N, K); }, RUNS);

        double threadMs = timeAvg([&] {
            cpu_matmul_threaded(A.data(), B.data(), C.data(), M, N, K, hw); }, RUNS);

        double blasMs = -1.0;
#ifdef HAVE_OPENBLAS
        blasMs = timeAvg([&] {
            cpu_matmul_blas(A.data(), B.data(), C.data(), M, N, K); }, RUNS);
#endif

        bench_gpu_once(ctx, pipe, A, B, M, N, K); // warm-up
        double gpuMs = timeAvg([&] {
            bench_gpu_once(ctx, pipe, A, B, M, N, K); }, RUNS);

        // En güçlü mevcut CPU temeli
        double bestCpu = threadMs;
        if (blasMs > 0.0) bestCpu = std::min(bestCpu, blasMs);
        double ratio = bestCpu / gpuMs;

        char blasCol[16];
        if (blasMs > 0.0) snprintf(blasCol, sizeof(blasCol), "%.3f", blasMs);
        else              snprintf(blasCol, sizeof(blasCol), "-");

        printf("%-7u  %-12.3f  %-12.3f  %-12s  %-12.3f  %-.2fx %s\n",
               sz, naiveMs, threadMs, blasCol, gpuMs, ratio,
               ratio > 1.0 ? "(GPU hizli)" : "(CPU hizli)");
    }

    destroyComputePipeline(ctx, pipe);
    destroyContext(ctx);
    return 0;
}
