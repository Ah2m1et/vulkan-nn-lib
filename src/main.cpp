#include "vk_context.h"
#include <cstdio>

int main() {
#ifndef VULKAN_AVAILABLE
    printf("Vulkan SDK not found at compile time — skipping smoke test.\n");
    return 0;
#else
    VkContext ctx = createContext();

    // Print device info (vulkaninfo-style summary)
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(ctx.physDevice, &props);

    uint32_t ver = props.apiVersion;
    printf("Device : %s\n", props.deviceName);
    printf("API    : Vulkan %u.%u.%u\n",
           VK_VERSION_MAJOR(ver),
           VK_VERSION_MINOR(ver),
           VK_VERSION_PATCH(ver));
    printf("Compute queue family: %u\n", ctx.computeFamilyIdx);

    destroyContext(ctx);
    printf("Vulkan context OK\n");
    return 0;
#endif
}
