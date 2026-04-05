#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

// CPU reference ReLU (mirrors the shader logic)
static float relu(float x) { return x > 0.0f ? x : 0.0f; }

int main() {
    std::vector<float> input  = { -3.0f, -1.0f, 0.0f, 1.0f, 2.5f, -0.001f, 100.0f };
    std::vector<float> expected(input.size());
    for (size_t i = 0; i < input.size(); ++i)
        expected[i] = relu(input[i]);

    // Verify reference values
    assert(expected[0] == 0.0f);   // -3   -> 0
    assert(expected[1] == 0.0f);   // -1   -> 0
    assert(expected[2] == 0.0f);   //  0   -> 0
    assert(expected[3] == 1.0f);   //  1   -> 1
    assert(std::fabs(expected[4] - 2.5f)   < 1e-6f);
    assert(expected[5] == 0.0f);   // -0.001 -> 0
    assert(expected[6] == 100.0f); // 100  -> 100

    // TODO (Week 3): upload input to GPU, dispatch relu.spv, download and compare

    printf("test_relu PASSED (CPU reference)\n");
    return 0;
}
