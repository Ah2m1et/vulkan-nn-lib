#include "vk_context.h"
#include "inference.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

static float relu(float x) { return x > 0.0f ? x : 0.0f; }

// CPU reference: ReLU after every layer except the last.
static std::vector<float> cpuForward(
    const std::vector<float>& input,
    const std::vector<std::vector<float>>& allWeights,
    const std::vector<std::vector<float>>& allBiases)
{
    std::vector<float> act = input;
    for (size_t l = 0; l < allWeights.size(); ++l) {
        uint32_t outSize = (uint32_t)allBiases[l].size();
        uint32_t inSize  = (uint32_t)act.size();
        std::vector<float> out(outSize);
        bool applyReLU = (l + 1 < allWeights.size());
        for (uint32_t i = 0; i < outSize; ++i) {
            float acc = allBiases[l][i];
            for (uint32_t j = 0; j < inSize; ++j)
                acc += allWeights[l][i * inSize + j] * act[j];
            out[i] = applyReLU ? relu(acc) : acc;
        }
        act = std::move(out);
    }
    return act;
}

int main() {
    VkContext ctx = createContext();

    // Network: 4 → 3 → 2
    constexpr uint32_t L0_IN = 4, L0_OUT = 3;
    constexpr uint32_t L1_OUT = 2;

    std::vector<float> W0(L0_OUT * L0_IN), b0(L0_OUT);
    std::vector<float> W1(L1_OUT * L0_OUT), b1(L1_OUT);

    for (size_t i = 0; i < W0.size(); ++i) W0[i] = 0.1f * (float)(i + 1);
    for (size_t i = 0; i < b0.size(); ++i) b0[i] = 0.01f * (float)(i + 1);
    for (size_t i = 0; i < W1.size(); ++i) W1[i] = 0.05f * (float)(i + 1);
    for (size_t i = 0; i < b1.size(); ++i) b1[i] = 0.005f * (float)(i + 1);

    std::vector<float> input = {1.0f, -2.0f, 3.0f, -1.5f};

    // CPU reference
    auto expected = cpuForward(input, {W0, W1}, {b0, b1});

    // GPU — scope ensures mlp is destroyed before ctx
    bool pass = false;
    {
        VulkanMLP mlp(ctx, SHADER_DIR);
        mlp.addLayer(L0_IN, L0_OUT);
        mlp.addLayer(L0_OUT, L1_OUT);
        mlp.loadWeights(0, W0, b0);
        mlp.loadWeights(1, W1, b1);
        auto result = mlp.forward(input);

        assert(result.size() == expected.size() && "output size mismatch");

        pass = true;
        for (size_t i = 0; i < result.size(); ++i) {
            float diff = std::abs(result[i] - expected[i]);
            if (diff > 1e-4f) {
                std::cerr << "FAIL output[" << i << "]: got " << result[i]
                          << ", expected " << expected[i] << " (diff=" << diff << ")\n";
                pass = false;
            }
        }

        if (pass) {
            std::cout << "PASS: VulkanMLP 4->3->2 matches CPU reference\n";
            std::cout << "  result:   [";
            for (size_t i = 0; i < result.size(); ++i)
                std::cout << result[i] << (i + 1 < result.size() ? ", " : "");
            std::cout << "]\n";
            std::cout << "  expected: [";
            for (size_t i = 0; i < expected.size(); ++i)
                std::cout << expected[i] << (i + 1 < expected.size() ? ", " : "");
            std::cout << "]\n";
        }
    } // mlp destroyed here — context still valid

    destroyContext(ctx);
    return pass ? 0 : 1;
}
