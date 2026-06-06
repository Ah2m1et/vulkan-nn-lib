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

static bool approxEqual(const std::vector<float>& a, const std::vector<float>& b,
                        float tol = 1e-4f) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (std::abs(a[i] - b[i]) > tol) return false;
    return true;
}

static bool notEqual(const std::vector<float>& a, const std::vector<float>& b,
                     float tol = 1e-4f) {
    if (a.size() != b.size()) return true;
    for (size_t i = 0; i < a.size(); ++i)
        if (std::abs(a[i] - b[i]) > tol) return true;
    return false;
}

int main() {
    VkContext ctx = createContext();
    bool allPass = true;

    // ----------------------------------------------------------------
    // Test 1: basic forward pass — GPU vs CPU reference
    // ----------------------------------------------------------------
    {
        constexpr uint32_t L0_IN = 4, L0_OUT = 3, L1_OUT = 2;

        std::vector<float> W0(L0_OUT * L0_IN), b0(L0_OUT);
        std::vector<float> W1(L1_OUT * L0_OUT), b1(L1_OUT);
        for (size_t i = 0; i < W0.size(); ++i) W0[i] = 0.1f * (float)(i + 1);
        for (size_t i = 0; i < b0.size(); ++i) b0[i] = 0.01f * (float)(i + 1);
        for (size_t i = 0; i < W1.size(); ++i) W1[i] = 0.05f * (float)(i + 1);
        for (size_t i = 0; i < b1.size(); ++i) b1[i] = 0.005f * (float)(i + 1);

        std::vector<float> input = {1.0f, -2.0f, 3.0f, -1.5f};
        auto expected = cpuForward(input, {W0, W1}, {b0, b1});

        VulkanMLP mlp(ctx, SHADER_DIR);
        mlp.addLayer(L0_IN, L0_OUT);
        mlp.addLayer(L0_OUT, L1_OUT);
        mlp.loadWeights(0, W0, b0);
        mlp.loadWeights(1, W1, b1);

        mlp.printTopology();

        auto result = mlp.forward(input);
        bool pass = approxEqual(result, expected);
        std::cout << (pass ? "PASS" : "FAIL") << " Test 1: GPU forward == CPU reference\n";
        allPass &= pass;

        // ----------------------------------------------------------------
        // Test 2: freeze layer 0
        // After freezing, running forward(inputA) again should give the same
        // result (layer 0's output buffer still holds the values from the
        // last active pass with inputA).
        // ----------------------------------------------------------------
        mlp.freeze(0);
        mlp.printTopology();

        auto resultFrozen = mlp.forward(input); // layer 0 skipped; uses cached output
        bool pass2 = approxEqual(resultFrozen, result);
        std::cout << (pass2 ? "PASS" : "FAIL")
                  << " Test 2: frozen layer 0 reuses cached output (same input)\n";
        allPass &= pass2;

        // ----------------------------------------------------------------
        // Test 3: different input while layer 0 is frozen
        // Layer 0 is skipped → its output buffer still holds values from
        // inputA → result equals cpuForward(inputA, W0, W1) regardless of
        // the new input.
        // ----------------------------------------------------------------
        std::vector<float> inputB = {-1.0f, 0.5f, 2.0f, -3.0f};
        auto resultB = mlp.forward(inputB);

        // layer 0 is frozen: its cached output is from inputA → same as result
        bool pass3 = approxEqual(resultB, result);
        std::cout << (pass3 ? "PASS" : "FAIL")
                  << " Test 3: frozen layer 0 ignores new input\n";
        allPass &= pass3;

        // ----------------------------------------------------------------
        // Test 4: unfreeze and verify recomputation
        // After unfreeze, forward(inputA) should match the original result
        // and forward(inputB) should differ from it.
        // ----------------------------------------------------------------
        mlp.unfreeze(0);
        mlp.printTopology();

        auto resultA2 = mlp.forward(input);
        bool pass4a = approxEqual(resultA2, expected);

        auto resultB2  = mlp.forward(inputB);
        auto expectedB = cpuForward(inputB, {W0, W1}, {b0, b1});
        bool pass4b    = approxEqual(resultB2, expectedB) && notEqual(resultB2, result);

        bool pass4 = pass4a && pass4b;
        std::cout << (pass4 ? "PASS" : "FAIL")
                  << " Test 4: after unfreeze, layer 0 recomputes correctly\n";
        allPass &= pass4;
    }

    destroyContext(ctx);
    return allPass ? 0 : 1;
}
