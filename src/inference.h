#pragma once
#include "vk_context.h"
#include "vk_buffer.h"
#include "vk_pipeline.h"

#include <string>
#include <vector>

// GPU-accelerated fully-connected MLP.
// Layers are added in order; each layer runs ReLU except the last.
//
// On the first forward() call after addLayer(), a single command buffer is
// pre-recorded covering all layers.  Subsequent forward() calls resubmit it
// without any re-recording, avoiding per-layer descriptor-pool and command-
// buffer overhead.
class VulkanMLP {
public:
    // ctx must outlive this object; shaderDir must contain fc_layer.spv.
    VulkanMLP(const VkContext& ctx, std::string shaderDir);
    ~VulkanMLP();

    // Append a fully-connected layer (inSize → outSize).
    void addLayer(uint32_t inSize, uint32_t outSize);

    // Upload pre-trained weights for layer idx.
    //   weights: outSize * inSize floats, row-major (W[i*inSize + j])
    //   biases:  outSize floats
    void loadWeights(size_t idx,
                     const std::vector<float>& weights,
                     const std::vector<float>& biases);

    // Run one forward pass on the GPU; returns output activations.
    // ReLU is applied after every layer except the last.
    // The first call after addLayer() builds the command buffer (one-time cost).
    std::vector<float> forward(const std::vector<float>& input);

private:
    struct Layer {
        uint32_t        inSize   = 0;
        uint32_t        outSize  = 0;
        ComputePipeline pipeline = {};
        GpuBuffer       weightBuf{};
        GpuBuffer       biasBuf{};
        GpuBuffer       outputBuf{};
    };

    const VkContext& ctx_;
    std::string      shaderDir_;
    std::vector<Layer> layers_;
    GpuBuffer          inputBuf_{};
    bool               inputBufReady_ = false;

    // Pre-recorded command buffer resources
    VkDescriptorPool             descPool_       = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> descSets_;
    VkCommandBuffer              prerecordedCmd_ = VK_NULL_HANDLE;
    bool                         prepared_       = false;

    void buildCommandBuffer();
};
