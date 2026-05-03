#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

#include "vk_context.h"
#include "vk_buffer.h"
#include "vk_pipeline.h"

#ifndef SHADER_DIR
#define SHADER_DIR "."
#endif

// CPU reference — mirrors the shader logic
static float relu_ref(float x) { return x > 0.0f ? x : 0.0f; }

int main() {
    const std::vector<float> input = {
        -3.0f, -1.0f, 0.0f, 1.0f, 2.5f, -0.001f, 100.0f
    };
    const uint32_t N = static_cast<uint32_t>(input.size());

    // --- CPU reference ---
    std::vector<float> expected(N);
    for (uint32_t i = 0; i < N; ++i)
        expected[i] = relu_ref(input[i]);

    assert(expected[0] == 0.0f);
    assert(expected[1] == 0.0f);
    assert(expected[2] == 0.0f);
    assert(expected[3] == 1.0f);
    assert(std::fabs(expected[4] - 2.5f)   < 1e-6f);
    assert(expected[5] == 0.0f);
    assert(expected[6] == 100.0f);

    // --- GPU dispatch ---
    VkContext ctx = createContext();

    const VkDeviceSize bufSize = N * sizeof(float);
    GpuBuffer inBuf  = createBuffer(ctx, bufSize);
    GpuBuffer outBuf = createBuffer(ctx, bufSize);

    uploadData(ctx, inBuf, input.data(), bufSize);

    auto spirv = loadSPIRV(std::string(SHADER_DIR) + "/relu.spv");
    ComputePipeline pipe = createComputePipeline(ctx, spirv,
                                                 /*bindingCount=*/2,
                                                 /*pushConstantSize=*/sizeof(uint32_t));

    uint32_t pc = N;
    dispatchPipeline(ctx, pipe,
                     {&inBuf, &outBuf},
                     &pc, sizeof(pc),
                     /*groupCountX=*/(N + 63) / 64);

    std::vector<float> gpuOutput(N);
    downloadData(ctx, outBuf, gpuOutput.data(), bufSize);

    // --- Verify GPU output against CPU reference ---
    for (uint32_t i = 0; i < N; ++i) {
        if (std::fabs(gpuOutput[i] - expected[i]) >= 1e-6f) {
            printf("MISMATCH at index %u: GPU=%.6f  expected=%.6f\n",
                   i, gpuOutput[i], expected[i]);
            return 1;
        }
    }

    destroyComputePipeline(ctx, pipe);
    destroyBuffer(ctx, inBuf);
    destroyBuffer(ctx, outBuf);
    destroyContext(ctx);

    printf("test_relu PASSED (CPU reference + GPU dispatch)\n");
    return 0;
}
