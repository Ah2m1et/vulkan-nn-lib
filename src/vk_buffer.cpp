#include "vk_buffer.h"
#include <stdexcept>
#include <cstring>

static uint32_t findMemoryType(VkPhysicalDevice physDev,
                               uint32_t typeFilter,
                               VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProps{};
    vkGetPhysicalDeviceMemoryProperties(physDev, &memProps);

    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((typeFilter & (1u << i)) &&
            (memProps.memoryTypes[i].propertyFlags & properties) == properties)
            return i;
    }
    throw std::runtime_error("No suitable memory type found.");
}

GpuBuffer createBuffer(const VkContext& ctx, VkDeviceSize sizeBytes,
                       VkBufferUsageFlags usage) {
    GpuBuffer buf{};
    buf.size = sizeBytes;

    VkBufferCreateInfo bufCI{};
    bufCI.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufCI.size        = sizeBytes;
    bufCI.usage       = usage;
    bufCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(ctx.device, &bufCI, nullptr, &buf.buffer));

    VkMemoryRequirements memReq{};
    vkGetBufferMemoryRequirements(ctx.device, buf.buffer, &memReq);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize  = memReq.size;
    allocInfo.memoryTypeIndex = findMemoryType(
        ctx.physDevice, memReq.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    VK_CHECK(vkAllocateMemory(ctx.device, &allocInfo, nullptr, &buf.memory));
    VK_CHECK(vkBindBufferMemory(ctx.device, buf.buffer, buf.memory, 0));

    return buf;
}

void destroyBuffer(const VkContext& ctx, GpuBuffer& buf) {
    if (buf.buffer != VK_NULL_HANDLE) vkDestroyBuffer(ctx.device, buf.buffer, nullptr);
    if (buf.memory != VK_NULL_HANDLE) vkFreeMemory(ctx.device, buf.memory, nullptr);
    buf = GpuBuffer{};
}

void uploadData(const VkContext& ctx, GpuBuffer& buf,
                const void* data, size_t sizeBytes) {
    void* mapped = nullptr;
    VK_CHECK(vkMapMemory(ctx.device, buf.memory, 0, sizeBytes, 0, &mapped));
    std::memcpy(mapped, data, sizeBytes);
    vkUnmapMemory(ctx.device, buf.memory);
}

void downloadData(const VkContext& ctx, const GpuBuffer& buf,
                  void* out, size_t sizeBytes) {
    void* mapped = nullptr;
    VK_CHECK(vkMapMemory(ctx.device, buf.memory, 0, sizeBytes, 0, &mapped));
    std::memcpy(out, mapped, sizeBytes);
    vkUnmapMemory(ctx.device, buf.memory);
}
