#pragma once
#include <vulkan/vulkan.h>
#include <cstdint>
#include <stdexcept>
#include <string>

// Abort on any Vulkan error
#define VK_CHECK(call)                                                  \
    do {                                                                \
        VkResult _res = (call);                                         \
        if (_res != VK_SUCCESS) {                                       \
            throw std::runtime_error(                                   \
                std::string("Vulkan error ") + std::to_string(_res) +  \
                " at " __FILE__ ":" + std::to_string(__LINE__));       \
        }                                                               \
    } while (0)

struct VkContext {
    VkInstance       instance         = VK_NULL_HANDLE;
    VkPhysicalDevice physDevice       = VK_NULL_HANDLE;
    VkDevice         device           = VK_NULL_HANDLE;
    VkQueue          computeQueue     = VK_NULL_HANDLE;
    VkCommandPool    commandPool      = VK_NULL_HANDLE; // for compute dispatches
    uint32_t         computeFamilyIdx = UINT32_MAX;
};

// Create a minimal Vulkan context with a compute queue.
VkContext createContext();

// Destroy all Vulkan objects held by the context.
void destroyContext(VkContext& ctx);
