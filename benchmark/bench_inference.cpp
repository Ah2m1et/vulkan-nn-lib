#include "vk_context.h"
#include "inference.h"

#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

using Ms = std::chrono::duration<double, std::milli>;
using Clock = std::chrono::steady_clock;

static std::vector<float> randomVec(size_t n, std::mt19937& rng) {
    std::normal_distribution<float> dist(0.0f, 0.1f);
    std::vector<float> v(n);
    for (auto& x : v) x = dist(rng);
    return v;
}

int main() {
    std::mt19937 rng(42);

    VkContext ctx = createContext();

    // 784 → 128 → 10  (MNIST-sized MLP)
    constexpr uint32_t IN = 784, H = 128, OUT = 10;

    auto W0 = randomVec((size_t)H   * IN,  rng);
    auto b0 = randomVec(H,                 rng);
    auto W1 = randomVec((size_t)OUT * H,   rng);
    auto b1 = randomVec(OUT,               rng);

    std::vector<float> input = randomVec(IN, rng);

    {
        VulkanMLP mlp(ctx, SHADER_DIR);
        mlp.addLayer(IN, H);
        mlp.addLayer(H, OUT);
        mlp.loadWeights(0, W0, b0);
        mlp.loadWeights(1, W1, b1);

        std::cout << "=== VulkanMLP Inference Benchmark (784->128->10) ===\n\n";

        // --- First call: triggers buildCommandBuffer() ---
        auto t0 = Clock::now();
        auto out = mlp.forward(input);
        double firstMs = Ms(Clock::now() - t0).count();
        std::cout << "First call (includes buildCommandBuffer): "
                  << firstMs << " ms\n\n";

        // --- Warm-up: 10 calls ---
        for (int i = 0; i < 10; ++i) mlp.forward(input);

        // --- Measurement: 1000 calls ---
        constexpr int REPS = 1000;
        auto tStart = Clock::now();
        for (int i = 0; i < REPS; ++i) mlp.forward(input);
        double totalMs = Ms(Clock::now() - tStart).count();

        double perCallMs = totalMs / REPS;
        double throughput = 1000.0 / perCallMs; // inferences per second

        std::cout << "Repeated inference (" << REPS << " calls):\n";
        std::cout << "  Total    : " << totalMs    << " ms\n";
        std::cout << "  Per call : " << perCallMs  << " ms\n";
        std::cout << "  Throughput: " << (int)throughput << " inf/sec\n\n";

        std::cout << "Output logits: [";
        for (size_t i = 0; i < out.size(); ++i)
            std::cout << out[i] << (i + 1 < out.size() ? ", " : "");
        std::cout << "]\n";
    }

    destroyContext(ctx);
    return 0;
}
