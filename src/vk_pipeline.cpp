#include "vk_pipeline.h"

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

// -----------------------------------------------------------------------
// SPIR-V loading
// -----------------------------------------------------------------------

std::vector<uint32_t> loadSPIRV(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open())
        throw std::runtime_error("loadSPIRV: cannot open file: " + path);

    std::streamsize sizeBytes = f.tellg();
    if (sizeBytes % 4 != 0)
        throw std::runtime_error("loadSPIRV: file size not a multiple of 4: " + path);

    f.seekg(0, std::ios::beg);
    std::vector<uint32_t> words(static_cast<size_t>(sizeBytes) / 4);
    f.read(reinterpret_cast<char*>(words.data()), sizeBytes);
    return words;
}

// -----------------------------------------------------------------------
// Compute pipeline
// -----------------------------------------------------------------------

ComputePipeline createComputePipeline(const VkContext&              ctx,
                                      const std::vector<uint32_t>& spirv,
                                      uint32_t                      bindingCount,
                                      uint32_t                      pushConstantSize) {
    ComputePipeline pipe{};
    pipe.bindingCount     = bindingCount;
    pipe.pushConstantSize = pushConstantSize;

    // --- Descriptor set layout ---
    std::vector<VkDescriptorSetLayoutBinding> bindings(bindingCount);
    for (uint32_t i = 0; i < bindingCount; ++i) {
        bindings[i].binding            = i;
        bindings[i].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount    = 1;
        bindings[i].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[i].pImmutableSamplers = nullptr;
    }

    VkDescriptorSetLayoutCreateInfo dslCI{};
    dslCI.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslCI.bindingCount = bindingCount;
    dslCI.pBindings    = bindings.data();
    VK_CHECK(vkCreateDescriptorSetLayout(ctx.device, &dslCI, nullptr,
                                         &pipe.descriptorSetLayout));

    // --- Pipeline layout (+ optional push constants) ---
    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcRange.offset     = 0;
    pcRange.size       = pushConstantSize;

    VkPipelineLayoutCreateInfo plCI{};
    plCI.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plCI.setLayoutCount         = 1;
    plCI.pSetLayouts            = &pipe.descriptorSetLayout;
    plCI.pushConstantRangeCount = (pushConstantSize > 0) ? 1 : 0;
    plCI.pPushConstantRanges    = (pushConstantSize > 0) ? &pcRange : nullptr;
    VK_CHECK(vkCreatePipelineLayout(ctx.device, &plCI, nullptr,
                                    &pipe.pipelineLayout));

    // --- Shader module ---
    VkShaderModuleCreateInfo smCI{};
    smCI.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smCI.codeSize = spirv.size() * sizeof(uint32_t);
    smCI.pCode    = spirv.data();

    VkShaderModule shaderModule = VK_NULL_HANDLE;
    VK_CHECK(vkCreateShaderModule(ctx.device, &smCI, nullptr, &shaderModule));

    // --- Compute pipeline ---
    VkPipelineShaderStageCreateInfo stageCI{};
    stageCI.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageCI.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    stageCI.module = shaderModule;
    stageCI.pName  = "main";

    VkComputePipelineCreateInfo pipeCI{};
    pipeCI.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeCI.stage  = stageCI;
    pipeCI.layout = pipe.pipelineLayout;
    VK_CHECK(vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &pipeCI,
                                      nullptr, &pipe.pipeline));

    // Shader module can be destroyed once the pipeline is created.
    vkDestroyShaderModule(ctx.device, shaderModule, nullptr);

    return pipe;
}

void destroyComputePipeline(const VkContext& ctx, ComputePipeline& pipe) {
    if (pipe.pipeline            != VK_NULL_HANDLE)
        vkDestroyPipeline(ctx.device, pipe.pipeline, nullptr);
    if (pipe.pipelineLayout      != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(ctx.device, pipe.pipelineLayout, nullptr);
    if (pipe.descriptorSetLayout != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(ctx.device, pipe.descriptorSetLayout, nullptr);
    pipe = ComputePipeline{};
}

// -----------------------------------------------------------------------
// Dispatch helper
// -----------------------------------------------------------------------

void dispatchPipeline(const VkContext&              ctx,
                      const ComputePipeline&         pipe,
                      const std::vector<GpuBuffer*>& buffers,
                      const void*                    pushData,
                      uint32_t                       pushSize,
                      uint32_t                       groupCountX,
                      uint32_t                       groupCountY,
                      uint32_t                       groupCountZ) {
    // --- Transient descriptor pool ---
    VkDescriptorPoolSize poolSize{};
    poolSize.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = pipe.bindingCount;

    VkDescriptorPoolCreateInfo poolCI{};
    poolCI.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolCI.maxSets       = 1;
    poolCI.poolSizeCount = 1;
    poolCI.pPoolSizes    = &poolSize;

    VkDescriptorPool descPool = VK_NULL_HANDLE;
    VK_CHECK(vkCreateDescriptorPool(ctx.device, &poolCI, nullptr, &descPool));

    // --- Allocate descriptor set ---
    VkDescriptorSetAllocateInfo dsAlloc{};
    dsAlloc.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsAlloc.descriptorPool     = descPool;
    dsAlloc.descriptorSetCount = 1;
    dsAlloc.pSetLayouts        = &pipe.descriptorSetLayout;

    VkDescriptorSet descSet = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateDescriptorSets(ctx.device, &dsAlloc, &descSet));

    // --- Update descriptor set with buffer bindings ---
    std::vector<VkDescriptorBufferInfo> bufInfos(buffers.size());
    std::vector<VkWriteDescriptorSet>   writes(buffers.size());

    for (uint32_t i = 0; i < static_cast<uint32_t>(buffers.size()); ++i) {
        bufInfos[i].buffer = buffers[i]->buffer;
        bufInfos[i].offset = 0;
        bufInfos[i].range  = VK_WHOLE_SIZE;

        writes[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet          = descSet;
        writes[i].dstBinding      = i;
        writes[i].dstArrayElement = 0;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo     = &bufInfos[i];
    }
    vkUpdateDescriptorSets(ctx.device,
                           static_cast<uint32_t>(writes.size()), writes.data(),
                           0, nullptr);

    // --- Allocate + record command buffer ---
    VkCommandBufferAllocateInfo cbAlloc{};
    cbAlloc.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbAlloc.commandPool        = ctx.commandPool;
    cbAlloc.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbAlloc.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateCommandBuffers(ctx.device, &cbAlloc, &cmd));

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipe.pipelineLayout, 0, 1, &descSet, 0, nullptr);

    if (pushData && pushSize > 0)
        vkCmdPushConstants(cmd, pipe.pipelineLayout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, pushSize, pushData);

    vkCmdDispatch(cmd, groupCountX, groupCountY, groupCountZ);

    VK_CHECK(vkEndCommandBuffer(cmd));

    // --- Submit + wait ---
    VkSubmitInfo submitInfo{};
    submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers    = &cmd;

    VK_CHECK(vkQueueSubmit(ctx.computeQueue, 1, &submitInfo, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(ctx.computeQueue));

    // --- Cleanup transient resources ---
    vkFreeCommandBuffers(ctx.device, ctx.commandPool, 1, &cmd);
    vkDestroyDescriptorPool(ctx.device, descPool, nullptr);
}
