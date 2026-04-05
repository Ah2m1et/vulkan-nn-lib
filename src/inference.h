#pragma once
#include "vk_context.h"
#include "vk_buffer.h"

#include <vector>
#include <string>

// Skeleton for a fully-connected MLP that runs inference on the GPU.
// Implementation is spread across weekly milestones (Weeks 6–8).
class VulkanMLP {
public:
    // ctx must outlive this object.
    explicit VulkanMLP(const VkContext& ctx);
    ~VulkanMLP();

    // Add a fully-connected layer: weight matrix (inSize x outSize) + bias (outSize).
    // TODO (Week 7): load weights from file / raw pointer
    void addLayer(uint32_t inSize, uint32_t outSize);

    // Run forward pass; returns output activations.
    // TODO (Week 7): implement descriptor sets, command recording, dispatch
    std::vector<float> forward(const std::vector<float>& input);

private:
    const VkContext& ctx_;

    // One pipeline per layer (matmul + activation)
    // TODO (Week 6): create compute pipelines
    std::vector<VkPipeline>       pipelines_;
    std::vector<VkPipelineLayout> pipelineLayouts_;

    // Weight and bias buffers per layer
    // TODO (Week 7): populate from model file
    std::vector<GpuBuffer> weightBuffers_;
    std::vector<GpuBuffer> biasBuffers_;
};
