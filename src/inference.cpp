#include "inference.h"
#include <stdexcept>
#include <algorithm>

namespace {

struct FcPC {
    uint32_t inSize;
    uint32_t outSize;
    uint32_t useReLU;
};

} // namespace

VulkanMLP::VulkanMLP(const VkContext& ctx, std::string shaderDir)
    : ctx_(ctx), shaderDir_(std::move(shaderDir)) {}

VulkanMLP::~VulkanMLP() {
    if (prerecordedCmd_ != VK_NULL_HANDLE)
        vkFreeCommandBuffers(ctx_.device, ctx_.commandPool, 1, &prerecordedCmd_);
    if (descPool_ != VK_NULL_HANDLE)
        vkDestroyDescriptorPool(ctx_.device, descPool_, nullptr);

    for (auto& layer : layers_) {
        destroyBuffer(ctx_, layer.weightBuf);
        destroyBuffer(ctx_, layer.biasBuf);
        destroyBuffer(ctx_, layer.outputBuf);
        destroyComputePipeline(ctx_, layer.pipeline);
    }
    if (inputBufReady_)
        destroyBuffer(ctx_, inputBuf_);
}

void VulkanMLP::addLayer(uint32_t inSize, uint32_t outSize) {
    if (!inputBufReady_) {
        inputBuf_      = createBuffer(ctx_, inSize * sizeof(float));
        inputBufReady_ = true;
    }

    auto spirv = loadSPIRV(shaderDir_ + "/fc_layer.spv");

    Layer layer;
    layer.inSize   = inSize;
    layer.outSize  = outSize;
    layer.pipeline = createComputePipeline(ctx_, spirv,
                                           /*bindingCount=*/4,
                                           /*pushConstantSize=*/sizeof(FcPC));
    layer.weightBuf = createBuffer(ctx_, (VkDeviceSize)inSize * outSize * sizeof(float));
    layer.biasBuf   = createBuffer(ctx_, outSize * sizeof(float));
    layer.outputBuf = createBuffer(ctx_, outSize * sizeof(float));

    std::vector<float> zeros(std::max(inSize * outSize, outSize), 0.0f);
    uploadData(ctx_, layer.weightBuf, zeros.data(), (size_t)inSize * outSize * sizeof(float));
    uploadData(ctx_, layer.biasBuf,   zeros.data(), outSize * sizeof(float));

    layers_.push_back(std::move(layer));
    prepared_ = false; // invalidate pre-recorded cmd on topology change
}

void VulkanMLP::loadWeights(size_t idx,
                             const std::vector<float>& weights,
                             const std::vector<float>& biases) {
    if (idx >= layers_.size())
        throw std::out_of_range("VulkanMLP::loadWeights: layer index out of range");
    auto& layer = layers_[idx];
    uploadData(ctx_, layer.weightBuf, weights.data(), weights.size() * sizeof(float));
    uploadData(ctx_, layer.biasBuf,   biases.data(),  biases.size()  * sizeof(float));
    // Buffer handles are unchanged; the pre-recorded cmd remains valid.
}

// ---------------------------------------------------------------------------
// Pre-record: build one command buffer covering all layers with barriers.
// Called lazily on the first forward() after addLayer().
// ---------------------------------------------------------------------------
void VulkanMLP::buildCommandBuffer() {
    // Release any previous resources.
    if (prerecordedCmd_ != VK_NULL_HANDLE) {
        vkFreeCommandBuffers(ctx_.device, ctx_.commandPool, 1, &prerecordedCmd_);
        prerecordedCmd_ = VK_NULL_HANDLE;
    }
    if (descPool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(ctx_.device, descPool_, nullptr);
        descPool_ = VK_NULL_HANDLE;
        descSets_.clear();
    }

    if (layers_.empty()) { prepared_ = true; return; }

    // Create persistent descriptor pool (4 storage-buffer bindings per layer).
    VkDescriptorPoolSize poolSize{};
    poolSize.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = 4 * (uint32_t)layers_.size();

    VkDescriptorPoolCreateInfo poolCI{};
    poolCI.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolCI.maxSets       = (uint32_t)layers_.size();
    poolCI.poolSizeCount = 1;
    poolCI.pPoolSizes    = &poolSize;
    VK_CHECK(vkCreateDescriptorPool(ctx_.device, &poolCI, nullptr, &descPool_));

    // Allocate one descriptor set per layer.
    std::vector<VkDescriptorSetLayout> layouts(layers_.size());
    for (size_t i = 0; i < layers_.size(); ++i)
        layouts[i] = layers_[i].pipeline.descriptorSetLayout;

    descSets_.resize(layers_.size());
    VkDescriptorSetAllocateInfo dsAlloc{};
    dsAlloc.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsAlloc.descriptorPool     = descPool_;
    dsAlloc.descriptorSetCount = (uint32_t)layers_.size();
    dsAlloc.pSetLayouts        = layouts.data();
    VK_CHECK(vkAllocateDescriptorSets(ctx_.device, &dsAlloc, descSets_.data()));

    // Bind buffers into each descriptor set.
    for (size_t i = 0; i < layers_.size(); ++i) {
        auto& layer    = layers_[i];
        GpuBuffer* prev = (i == 0) ? &inputBuf_ : &layers_[i - 1].outputBuf;

        VkDescriptorBufferInfo bufInfos[4] = {
            {prev->buffer,              0, VK_WHOLE_SIZE},
            {layer.weightBuf.buffer,    0, VK_WHOLE_SIZE},
            {layer.biasBuf.buffer,      0, VK_WHOLE_SIZE},
            {layer.outputBuf.buffer,    0, VK_WHOLE_SIZE},
        };
        VkWriteDescriptorSet writes[4] = {};
        for (uint32_t b = 0; b < 4; ++b) {
            writes[b].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[b].dstSet          = descSets_[i];
            writes[b].dstBinding      = b;
            writes[b].descriptorCount = 1;
            writes[b].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[b].pBufferInfo     = &bufInfos[b];
        }
        vkUpdateDescriptorSets(ctx_.device, 4, writes, 0, nullptr);
    }

    // Allocate and record the command buffer (reusable — no ONE_TIME_SUBMIT).
    VkCommandBufferAllocateInfo cbAlloc{};
    cbAlloc.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbAlloc.commandPool        = ctx_.commandPool;
    cbAlloc.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbAlloc.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(ctx_.device, &cbAlloc, &prerecordedCmd_));

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = 0; // reusable
    VK_CHECK(vkBeginCommandBuffer(prerecordedCmd_, &beginInfo));

    for (size_t i = 0; i < layers_.size(); ++i) {
        auto& layer    = layers_[i];
        uint32_t useReLU = (i + 1 < layers_.size()) ? 1u : 0u;
        FcPC pc{layer.inSize, layer.outSize, useReLU};
        uint32_t groupX = (layer.outSize + 63) / 64;

        vkCmdBindPipeline(prerecordedCmd_, VK_PIPELINE_BIND_POINT_COMPUTE,
                          layer.pipeline.pipeline);
        vkCmdBindDescriptorSets(prerecordedCmd_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                layer.pipeline.pipelineLayout, 0, 1, &descSets_[i],
                                0, nullptr);
        vkCmdPushConstants(prerecordedCmd_, layer.pipeline.pipelineLayout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(FcPC), &pc);
        vkCmdDispatch(prerecordedCmd_, groupX, 1, 1);

        // Barrier: this layer's output must be fully written before next layer reads.
        if (i + 1 < layers_.size()) {
            VkBufferMemoryBarrier barrier{};
            barrier.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            barrier.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.buffer              = layer.outputBuf.buffer;
            barrier.offset              = 0;
            barrier.size                = VK_WHOLE_SIZE;
            vkCmdPipelineBarrier(prerecordedCmd_,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 0, nullptr, 1, &barrier, 0, nullptr);
        }
    }

    VK_CHECK(vkEndCommandBuffer(prerecordedCmd_));
    prepared_ = true;
}

// ---------------------------------------------------------------------------
// Forward pass: upload input, submit pre-recorded cmd once, download output.
// ---------------------------------------------------------------------------
std::vector<float> VulkanMLP::forward(const std::vector<float>& input) {
    if (layers_.empty()) return {};

    if (!prepared_)
        buildCommandBuffer();

    uploadData(ctx_, inputBuf_, input.data(), input.size() * sizeof(float));

    VkSubmitInfo submitInfo{};
    submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers    = &prerecordedCmd_;
    VK_CHECK(vkQueueSubmit(ctx_.computeQueue, 1, &submitInfo, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(ctx_.computeQueue));

    auto& last = layers_.back();
    std::vector<float> result(last.outSize);
    downloadData(ctx_, last.outputBuf, result.data(), result.size() * sizeof(float));
    return result;
}
