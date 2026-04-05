#pragma once
#include "vk_context.h"
#include <cstddef>

struct GpuBuffer {
    VkBuffer       buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize   size   = 0;
};

// Allocate a device-local (host-visible) buffer of `sizeBytes`.
GpuBuffer createBuffer(const VkContext& ctx, VkDeviceSize sizeBytes,
                       VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

void destroyBuffer(const VkContext& ctx, GpuBuffer& buf);

// Copy `sizeBytes` from CPU `data` into the GPU buffer (requires host-visible memory).
void uploadData(const VkContext& ctx, GpuBuffer& buf,
                const void* data, size_t sizeBytes);

// Copy GPU buffer contents back to CPU `out`.
void downloadData(const VkContext& ctx, const GpuBuffer& buf,
                  void* out, size_t sizeBytes);
