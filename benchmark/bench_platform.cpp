#include "vk_context.h"
#include "inference.h"

#include <chrono>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

using Ms    = std::chrono::duration<double, std::milli>;
using Clock = std::chrono::steady_clock;

// ---------------------------------------------------------------------------
// Network configuration
// ---------------------------------------------------------------------------
struct NetConfig {
    const char*             label;
    std::vector<uint32_t>   layers; // e.g. {784, 128, 10}
};

static const NetConfig NETS[] = {
    {"784→128→10   (MNIST small)",  {784, 128, 10}},
    {"784→512→256→10 (MNIST large)", {784, 512, 256, 10}},
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static std::vector<float> randVec(size_t n, std::mt19937& rng) {
    std::normal_distribution<float> d(0.0f, 0.1f);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}

struct BenchResult {
    std::string deviceName;
    std::string deviceType;
    std::string netLabel;
    double      firstCallMs;
    double      perCallMs;
    int         throughput;
};

static BenchResult runBench(const DeviceInfo& dev,
                             const NetConfig&  net,
                             std::mt19937&     rng) {
    constexpr int WARMUP = 5;
    constexpr int REPS   = 200;

    VkContext ctx = createContextForDevice(dev.index);

    BenchResult res{};
    res.deviceName = dev.name;
    res.deviceType = dev.typeName;
    res.netLabel   = net.label;

    {
        VulkanMLP mlp(ctx, SHADER_DIR);

        // Build network
        for (size_t i = 1; i < net.layers.size(); ++i)
            mlp.addLayer(net.layers[i - 1], net.layers[i]);

        // Load random weights
        for (size_t i = 1; i < net.layers.size(); ++i) {
            uint32_t in  = net.layers[i - 1];
            uint32_t out = net.layers[i];
            mlp.loadWeights(i - 1, randVec((size_t)in * out, rng), randVec(out, rng));
        }

        std::vector<float> input = randVec(net.layers[0], rng);

        // First call (triggers buildCommandBuffer)
        auto t0 = Clock::now();
        mlp.forward(input);
        res.firstCallMs = Ms(Clock::now() - t0).count();

        // Warm-up
        for (int i = 0; i < WARMUP; ++i) mlp.forward(input);

        // Measurement
        auto tStart = Clock::now();
        for (int i = 0; i < REPS; ++i) mlp.forward(input);
        double totalMs = Ms(Clock::now() - tStart).count();

        res.perCallMs  = totalMs / REPS;
        res.throughput = static_cast<int>(1000.0 / res.perCallMs);
    }

    destroyContext(ctx);
    return res;
}

// ---------------------------------------------------------------------------
int main() {
    std::mt19937 rng(42);

    auto devices = enumerateDevices();
    if (devices.empty()) {
        std::cerr << "No Vulkan compute devices found.\n";
        return 1;
    }

    std::cout << "=== Platform Independence Benchmark ===\n\n";
    std::cout << "Available Vulkan devices:\n";
    for (auto& d : devices) {
        uint32_t maj = VK_VERSION_MAJOR(d.apiVersion);
        uint32_t min = VK_VERSION_MINOR(d.apiVersion);
        uint32_t pat = VK_VERSION_PATCH(d.apiVersion);
        std::cout << "  [" << d.index << "] " << d.name
                  << "  (" << d.typeName
                  << ", Vulkan " << maj << "." << min << "." << pat << ")\n";
    }
    std::cout << "\n";

    std::vector<BenchResult> results;
    for (auto& net : NETS) {
        for (auto& dev : devices) {
            std::cout << "Running: " << net.label
                      << "  on  " << dev.name << " ...\n";
            results.push_back(runBench(dev, net, rng));
        }
    }

    // ---------------------------------------------------------------------------
    // Print table
    // ---------------------------------------------------------------------------
    std::cout << "\n";
    std::cout << "=== Results ===\n\n";

    // Column widths
    const int W_DEV  = 38;
    const int W_TYPE = 16;
    const int W_NET  = 26;
    const int W_NUM  = 12;

    auto sep = [&]() {
        std::cout << std::string(W_DEV + W_TYPE + W_NET + W_NUM * 3 + 13, '-') << "\n";
    };

    sep();
    std::cout << std::left
              << std::setw(W_DEV)  << "Device"
              << std::setw(W_TYPE) << "Type"
              << std::setw(W_NET)  << "Network"
              << std::right
              << std::setw(W_NUM)  << "1st call(ms)"
              << std::setw(W_NUM)  << "per call(ms)"
              << std::setw(W_NUM)  << "inf/sec"
              << "\n";
    sep();

    for (auto& r : results) {
        std::cout << std::left
                  << std::setw(W_DEV)  << r.deviceName.substr(0, W_DEV - 1)
                  << std::setw(W_TYPE) << r.deviceType.substr(0, W_TYPE - 1)
                  << std::setw(W_NET)  << r.netLabel
                  << std::right
                  << std::setw(W_NUM)  << std::fixed << std::setprecision(3) << r.firstCallMs
                  << std::setw(W_NUM)  << std::fixed << std::setprecision(3) << r.perCallMs
                  << std::setw(W_NUM)  << r.throughput
                  << "\n";
    }
    sep();
    std::cout << "\n";
    std::cout << "Note: 'per call' = steady-state after warm-up ("
              << 200 << " reps).\n";

    return 0;
}
