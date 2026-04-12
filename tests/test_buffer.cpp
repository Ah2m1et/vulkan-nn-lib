#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

#ifndef VULKAN_AVAILABLE
int main() {
    printf("test_buffer SKIPPED (Vulkan not available at compile time)\n");
    return 0;
}
#else

#include "vk_context.h"
#include "vk_buffer.h"

int main() {
    VkContext ctx = createContext();

    // --- Test 1: float roundtrip ---
    const std::vector<float> input = { 1.0f, -2.5f, 3.14f, 0.0f, -100.0f, 42.0f };
    const size_t bytes = input.size() * sizeof(float);

    GpuBuffer buf = createBuffer(ctx, bytes);
    uploadData(ctx, buf, input.data(), bytes);

    std::vector<float> output(input.size(), 0.0f);
    downloadData(ctx, buf, output.data(), bytes);

    for (size_t i = 0; i < input.size(); ++i)
        assert(std::fabs(output[i] - input[i]) < 1e-7f);

    destroyBuffer(ctx, buf);
    printf("  [PASS] float roundtrip\n");

    // --- Test 2: zero-fill ---
    GpuBuffer zeroBuf = createBuffer(ctx, 32 * sizeof(float));
    std::vector<float> zeros(32, 0.0f);
    uploadData(ctx, zeroBuf, zeros.data(), zeros.size() * sizeof(float));

    std::vector<float> readBack(32, 1.0f);
    downloadData(ctx, zeroBuf, readBack.data(), readBack.size() * sizeof(float));
    for (float v : readBack)
        assert(v == 0.0f);

    destroyBuffer(ctx, zeroBuf);
    printf("  [PASS] zero-fill roundtrip\n");

    // --- Test 3: command pool is valid ---
    assert(ctx.commandPool != VK_NULL_HANDLE);
    printf("  [PASS] command pool created\n");

    destroyContext(ctx);
    printf("test_buffer PASSED\n");
    return 0;
}
#endif
