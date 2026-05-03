#pragma once
#include "vk_context.h"
#include "vk_buffer.h"

#include <cstdint>
#include <string>
#include <vector>

// -----------------------------------------------------------------------
// SPIR-V loading
// -----------------------------------------------------------------------

// Read a compiled .spv file and return its 32-bit words.
std::vector<uint32_t> loadSPIRV(const std::string& path);

// -----------------------------------------------------------------------
// Compute pipeline
// -----------------------------------------------------------------------

struct ComputePipeline {
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout      pipelineLayout      = VK_NULL_HANDLE;
    VkPipeline            pipeline            = VK_NULL_HANDLE;
    uint32_t              bindingCount        = 0;
    uint32_t              pushConstantSize    = 0;
};

// Build a compute pipeline from `spirv` with:
//   - `bindingCount` storage-buffer bindings at set=0
//   - an optional push-constant block of `pushConstantSize` bytes
ComputePipeline createComputePipeline(const VkContext& ctx,
                                      const std::vector<uint32_t>& spirv,
                                      uint32_t bindingCount,
                                      uint32_t pushConstantSize = 0);

void destroyComputePipeline(const VkContext& ctx, ComputePipeline& pipe);

// -----------------------------------------------------------------------
// Dispatch helper
// -----------------------------------------------------------------------

// Allocate a transient descriptor set, bind `buffers`, record a command
// buffer, submit it, and wait for completion.
//
//   buffers      — pointers to GpuBuffer objects (one per binding)
//   pushData     — pointer to push-constant data (may be nullptr)
//   pushSize     — size of push-constant block in bytes
//   groupCount*  — workgroup counts in X / Y / Z
void dispatchPipeline(const VkContext&              ctx,
                      const ComputePipeline&         pipe,
                      const std::vector<GpuBuffer*>& buffers,
                      const void*                    pushData,
                      uint32_t                       pushSize,
                      uint32_t                       groupCountX,
                      uint32_t                       groupCountY = 1,
                      uint32_t                       groupCountZ = 1);
