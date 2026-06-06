#include "inference.h"
#include <iostream>
#include <stdexcept>
#include <algorithm>

namespace {

struct FcPC { uint32_t inSize; uint32_t outSize; uint32_t useReLU; };

} // namespace

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

entt::entity VulkanMLP::entityAt(size_t idx) const {
    if (idx >= layerOrder_.size())
        throw std::out_of_range("VulkanMLP: layer index out of range");
    return layerOrder_[idx];
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

VulkanMLP::VulkanMLP(const VkContext& ctx, std::string shaderDir)
    : ctx_(ctx), shaderDir_(std::move(shaderDir)) {}

VulkanMLP::~VulkanMLP() {
    if (prerecordedCmd_ != VK_NULL_HANDLE)
        vkFreeCommandBuffers(ctx_.device, ctx_.commandPool, 1, &prerecordedCmd_);
    if (descPool_ != VK_NULL_HANDLE)
        vkDestroyDescriptorPool(ctx_.device, descPool_, nullptr);

    // Destroy per-layer Vulkan resources through the registry.
    for (auto entity : layerOrder_) {
        destroyBuffer(ctx_, registry_.get<GpuWeights>(entity).buf);
        destroyBuffer(ctx_, registry_.get<GpuBias>(entity).buf);
        destroyBuffer(ctx_, registry_.get<GpuOutput>(entity).buf);
        destroyComputePipeline(ctx_, registry_.get<LayerPipeline>(entity).pipe);
    }

    if (inputBufReady_)
        destroyBuffer(ctx_, inputBuf_);
}

// ---------------------------------------------------------------------------
// addLayer  — creates an entity and attaches all components
// ---------------------------------------------------------------------------

void VulkanMLP::addLayer(uint32_t inSize, uint32_t outSize) {
    if (!inputBufReady_) {
        inputBuf_      = createBuffer(ctx_, inSize * sizeof(float));
        inputBufReady_ = true;
    }

    auto spirv = loadSPIRV(shaderDir_ + "/fc_layer.spv");

    auto entity = registry_.create();

    registry_.emplace<LayerShape>(entity, inSize, outSize);
    registry_.emplace<LayerOrder>(entity, (uint32_t)layerOrder_.size());
    registry_.emplace<LayerPipeline>(entity,
        LayerPipeline{createComputePipeline(ctx_, spirv, 4, sizeof(FcPC))});

    auto wBuf = createBuffer(ctx_, (VkDeviceSize)inSize * outSize * sizeof(float));
    auto bBuf = createBuffer(ctx_, outSize * sizeof(float));
    auto oBuf = createBuffer(ctx_, outSize * sizeof(float));

    std::vector<float> zeros(std::max(inSize * outSize, outSize), 0.0f);
    uploadData(ctx_, wBuf, zeros.data(), (size_t)inSize * outSize * sizeof(float));
    uploadData(ctx_, bBuf, zeros.data(), outSize * sizeof(float));

    registry_.emplace<GpuWeights>(entity, GpuWeights{wBuf});
    registry_.emplace<GpuBias>(entity,    GpuBias{bBuf});
    registry_.emplace<GpuOutput>(entity,  GpuOutput{oBuf});

    layerOrder_.push_back(entity);
    prepared_ = false;
}

// ---------------------------------------------------------------------------
// loadWeights  — uploads data into an entity's buffer components
// ---------------------------------------------------------------------------

void VulkanMLP::loadWeights(size_t idx,
                             const std::vector<float>& weights,
                             const std::vector<float>& biases) {
    auto entity = entityAt(idx);
    uploadData(ctx_, registry_.get<GpuWeights>(entity).buf,
               weights.data(), weights.size() * sizeof(float));
    uploadData(ctx_, registry_.get<GpuBias>(entity).buf,
               biases.data(),  biases.size()  * sizeof(float));
    // Buffer handles unchanged; pre-recorded cmd remains valid.
}

// ---------------------------------------------------------------------------
// freeze / unfreeze  — add/remove tag component, invalidate cmd
// ---------------------------------------------------------------------------

void VulkanMLP::freeze(size_t layerIdx) {
    auto entity = entityAt(layerIdx);
    if (!registry_.all_of<FrozenLayer>(entity)) {
        registry_.emplace<FrozenLayer>(entity);
        prepared_ = false;
    }
}

void VulkanMLP::unfreeze(size_t layerIdx) {
    auto entity = entityAt(layerIdx);
    if (registry_.all_of<FrozenLayer>(entity)) {
        registry_.remove<FrozenLayer>(entity);
        prepared_ = false;
    }
}

bool VulkanMLP::isFrozen(size_t layerIdx) const {
    return registry_.all_of<FrozenLayer>(entityAt(layerIdx));
}

// ---------------------------------------------------------------------------
// printTopology  — ECS query to inspect the network
// ---------------------------------------------------------------------------

void VulkanMLP::printTopology() const {
    std::cout << "VulkanMLP topology (" << layerOrder_.size() << " layers):\n";
    for (size_t i = 0; i < layerOrder_.size(); ++i) {
        auto entity        = layerOrder_[i];
        const auto& shape  = registry_.get<LayerShape>(entity);
        bool frozen        = registry_.all_of<FrozenLayer>(entity);
        std::cout << "  Layer " << i
                  << ": " << shape.inSize << " \342\206\222 " << shape.outSize
                  << (frozen ? "  [FROZEN]" : "") << "\n";
    }
}

// ---------------------------------------------------------------------------
// buildCommandBuffer  — pre-record: one reusable cmd for all active layers
// ---------------------------------------------------------------------------

void VulkanMLP::buildCommandBuffer() {
    if (prerecordedCmd_ != VK_NULL_HANDLE) {
        vkFreeCommandBuffers(ctx_.device, ctx_.commandPool, 1, &prerecordedCmd_);
        prerecordedCmd_ = VK_NULL_HANDLE;
    }
    if (descPool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(ctx_.device, descPool_, nullptr);
        descPool_ = VK_NULL_HANDLE;
        descSets_.clear();
    }

    if (layerOrder_.empty()) { prepared_ = true; return; }

    // Descriptor pool: 4 storage-buffer slots per layer (incl. frozen — their
    // buffers are still referenced by the next active layer's descriptor set).
    VkDescriptorPoolSize poolSize{};
    poolSize.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = 4 * (uint32_t)layerOrder_.size();

    VkDescriptorPoolCreateInfo poolCI{};
    poolCI.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolCI.maxSets       = (uint32_t)layerOrder_.size();
    poolCI.poolSizeCount = 1;
    poolCI.pPoolSizes    = &poolSize;
    VK_CHECK(vkCreateDescriptorPool(ctx_.device, &poolCI, nullptr, &descPool_));

    // Allocate one descriptor set per layer.
    std::vector<VkDescriptorSetLayout> layouts(layerOrder_.size());
    for (size_t i = 0; i < layerOrder_.size(); ++i)
        layouts[i] = registry_.get<LayerPipeline>(layerOrder_[i]).pipe.descriptorSetLayout;

    descSets_.resize(layerOrder_.size());
    VkDescriptorSetAllocateInfo dsAlloc{};
    dsAlloc.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsAlloc.descriptorPool     = descPool_;
    dsAlloc.descriptorSetCount = (uint32_t)layerOrder_.size();
    dsAlloc.pSetLayouts        = layouts.data();
    VK_CHECK(vkAllocateDescriptorSets(ctx_.device, &dsAlloc, descSets_.data()));

    // Bind buffers into each descriptor set (active AND frozen layers).
    for (size_t i = 0; i < layerOrder_.size(); ++i) {
        auto entity = layerOrder_[i];
        GpuBuffer* prev = (i == 0) ? &inputBuf_
                                    : &registry_.get<GpuOutput>(layerOrder_[i-1]).buf;

        VkDescriptorBufferInfo bufInfos[4] = {
            {prev->buffer,                                             0, VK_WHOLE_SIZE},
            {registry_.get<GpuWeights>(entity).buf.buffer,            0, VK_WHOLE_SIZE},
            {registry_.get<GpuBias>(entity).buf.buffer,               0, VK_WHOLE_SIZE},
            {registry_.get<GpuOutput>(entity).buf.buffer,             0, VK_WHOLE_SIZE},
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

    // Allocate and record the command buffer (reusable).
    VkCommandBufferAllocateInfo cbAlloc{};
    cbAlloc.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbAlloc.commandPool        = ctx_.commandPool;
    cbAlloc.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbAlloc.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(ctx_.device, &cbAlloc, &prerecordedCmd_));

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = 0;
    VK_CHECK(vkBeginCommandBuffer(prerecordedCmd_, &beginInfo));

    for (size_t i = 0; i < layerOrder_.size(); ++i) {
        auto entity = layerOrder_[i];

        // Skip frozen layers — their cached output is reused as-is.
        if (registry_.all_of<FrozenLayer>(entity)) continue;

        const auto& shape  = registry_.get<LayerShape>(entity);
        const auto& pipe   = registry_.get<LayerPipeline>(entity).pipe;

        // Last *active* layer gets no ReLU; earlier active layers do.
        // We detect "last active" by scanning ahead.
        bool isLastActive = true;
        for (size_t j = i + 1; j < layerOrder_.size(); ++j)
            if (!registry_.all_of<FrozenLayer>(layerOrder_[j])) { isLastActive = false; break; }

        FcPC pc{shape.inSize, shape.outSize, isLastActive ? 0u : 1u};
        uint32_t groupX = (shape.outSize + 63) / 64;

        vkCmdBindPipeline(prerecordedCmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipeline);
        vkCmdBindDescriptorSets(prerecordedCmd_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipe.pipelineLayout, 0, 1, &descSets_[i], 0, nullptr);
        vkCmdPushConstants(prerecordedCmd_, pipe.pipelineLayout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(FcPC), &pc);
        vkCmdDispatch(prerecordedCmd_, groupX, 1, 1);

        // Barrier if the immediately next layer is also active (they share a buffer).
        bool nextIsActive = (i + 1 < layerOrder_.size()) &&
                            !registry_.all_of<FrozenLayer>(layerOrder_[i + 1]);
        if (nextIsActive) {
            VkBufferMemoryBarrier barrier{};
            barrier.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            barrier.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.buffer              = registry_.get<GpuOutput>(entity).buf.buffer;
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
// forward  — upload input, submit pre-recorded cmd, download output
// ---------------------------------------------------------------------------

std::vector<float> VulkanMLP::forward(const std::vector<float>& input) {
    if (layerOrder_.empty()) return {};

    if (!prepared_)
        buildCommandBuffer();

    uploadData(ctx_, inputBuf_, input.data(), input.size() * sizeof(float));

    VkSubmitInfo submitInfo{};
    submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers    = &prerecordedCmd_;
    VK_CHECK(vkQueueSubmit(ctx_.computeQueue, 1, &submitInfo, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(ctx_.computeQueue));

    // Return the output of the last layer (frozen or not).
    auto& last = registry_.get<GpuOutput>(layerOrder_.back());
    const auto& lastShape = registry_.get<LayerShape>(layerOrder_.back());
    std::vector<float> result(lastShape.outSize);
    downloadData(ctx_, last.buf, result.data(), result.size() * sizeof(float));
    return result;
}
