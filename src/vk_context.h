#pragma once
#include <vulkan/vulkan.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

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
    VkCommandPool    commandPool      = VK_NULL_HANDLE;
    uint32_t         computeFamilyIdx = UINT32_MAX;
};

struct DeviceInfo {
    uint32_t    index;      // index into vkEnumeratePhysicalDevices list
    std::string name;
    std::string typeName;   // "Discrete GPU", "Integrated GPU", "CPU", "Virtual GPU", "Other"
    uint32_t    apiVersion;
};

// Return info for every physical device that has a compute queue.
std::vector<DeviceInfo> enumerateDevices();

// Create a context using the device at the given physical-device index.
VkContext createContextForDevice(uint32_t deviceIndex);

// Convenience: create a context using the first available device.
VkContext createContext();

// Destroy all Vulkan objects held by the context.
void destroyContext(VkContext& ctx);
