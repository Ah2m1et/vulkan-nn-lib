#include "inference.h"
#include <stdexcept>

VulkanMLP::VulkanMLP(const VkContext& ctx)
    : ctx_(ctx) {
    // TODO (Week 6): initialize descriptor pool, command pool
}

VulkanMLP::~VulkanMLP() {
    // TODO (Week 6/7): destroy pipelines, pipeline layouts, buffers
    for (auto& buf : weightBuffers_) destroyBuffer(const_cast<VkContext&>(ctx_), buf);
    for (auto& buf : biasBuffers_)   destroyBuffer(const_cast<VkContext&>(ctx_), buf);
    for (auto pl : pipelines_)       vkDestroyPipeline(ctx_.device, pl, nullptr);
    for (auto pl : pipelineLayouts_) vkDestroyPipelineLayout(ctx_.device, pl, nullptr);
}

void VulkanMLP::addLayer(uint32_t inSize, uint32_t outSize) {
    // TODO (Week 6): create pipeline for matmul + bias + ReLU for this layer
    // TODO (Week 7): allocate weightBuffers_[i] and biasBuffers_[i]
    (void)inSize;
    (void)outSize;
}

std::vector<float> VulkanMLP::forward(const std::vector<float>& input) {
    // TODO (Week 7): record and submit command buffers for each layer
    // TODO (Week 8): pre-record command buffers for repeated inference
    (void)input;
    return {};
}
