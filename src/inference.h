#pragma once
#include "vk_context.h"
#include "vk_buffer.h"
#include "vk_pipeline.h"

#include <entt/entt.hpp>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// ECS Components
// Plain-data structs; no behaviour lives here — only the ForwardPassSystem
// (VulkanMLP::forward) operates on them.
// ---------------------------------------------------------------------------

struct LayerShape    { uint32_t inSize; uint32_t outSize; };
struct LayerOrder    { uint32_t idx; };   // position in forward-pass sequence
struct GpuWeights    { GpuBuffer buf; };
struct GpuBias       { GpuBuffer buf; };
struct GpuOutput     { GpuBuffer buf; };
struct LayerPipeline { ComputePipeline pipe; };

// Tag component: forward pass skips this layer's dispatch and reuses the
// cached output buffer from the last time the layer was active.
struct FrozenLayer {};

// ---------------------------------------------------------------------------
// VulkanMLP
// Wraps an entt::registry; each layer is an entity with the components above.
// ---------------------------------------------------------------------------
class VulkanMLP {
public:
    // ctx must outlive this object; shaderDir must contain fc_layer.spv.
    VulkanMLP(const VkContext& ctx, std::string shaderDir);
    ~VulkanMLP();

    VulkanMLP(const VulkanMLP&)            = delete;
    VulkanMLP& operator=(const VulkanMLP&) = delete;

    // Append a fully-connected layer (inSize → outSize).
    void addLayer(uint32_t inSize, uint32_t outSize);

    // Upload pre-trained weights for layer idx.
    //   weights: outSize * inSize floats, row-major (W[i*inSize + j])
    //   biases:  outSize floats
    void loadWeights(size_t idx,
                     const std::vector<float>& weights,
                     const std::vector<float>& biases);

    // Run one forward pass on the GPU.
    // ReLU is applied after every non-frozen layer except the last.
    // First call after addLayer()/freeze()/unfreeze() rebuilds the command buffer.
    std::vector<float> forward(const std::vector<float>& input);

    // ---- ECS-native operations ----

    // Add FrozenLayer tag: subsequent forward() calls skip this layer's
    // dispatch and keep its output buffer unchanged.
    void freeze(size_t layerIdx);

    // Remove FrozenLayer tag: layer is active again.
    void unfreeze(size_t layerIdx);

    bool   isFrozen(size_t layerIdx) const;
    size_t layerCount() const { return layerOrder_.size(); }

    // Print layer topology to stdout (queries registry).
    void printTopology() const;

    // Read-only registry access for external ECS queries.
    const entt::registry& registry() const { return registry_; }

private:
    const VkContext&          ctx_;
    std::string               shaderDir_;
    entt::registry            registry_;
    std::vector<entt::entity> layerOrder_;   // entities in forward-pass order
    GpuBuffer                 inputBuf_{};
    bool                      inputBufReady_ = false;

    // Pre-recorded command buffer resources
    VkDescriptorPool             descPool_       = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> descSets_;      // one per layer (inc. frozen)
    VkCommandBuffer              prerecordedCmd_ = VK_NULL_HANDLE;
    bool                         prepared_       = false;

    void buildCommandBuffer();
    entt::entity entityAt(size_t idx) const;
};
