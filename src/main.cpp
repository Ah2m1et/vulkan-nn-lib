#include "vk_context.h"
#include "vk_buffer.h"
#include <cstdio>
#include <cmath>

int main() {
#ifndef VULKAN_AVAILABLE
    printf("Vulkan SDK not found at compile time — skipping smoke test.\n");
    return 0;
#else
    // --- Context ---
    VkContext ctx = createContext();

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(ctx.physDevice, &props);

    uint32_t ver = props.apiVersion;
    printf("Device      : %s\n", props.deviceName);
    printf("API         : Vulkan %u.%u.%u\n",
           VK_VERSION_MAJOR(ver), VK_VERSION_MINOR(ver), VK_VERSION_PATCH(ver));
    printf("Queue family: %u\n", ctx.computeFamilyIdx);
    printf("Command pool: %s\n", ctx.commandPool != VK_NULL_HANDLE ? "OK" : "FAIL");

    // --- Buffer roundtrip ---
    const float src[4] = { 1.0f, -2.0f, 3.14f, 0.0f };
    float dst[4]       = {};

    GpuBuffer buf = createBuffer(ctx, sizeof(src));
    uploadData(ctx, buf, src, sizeof(src));
    downloadData(ctx, buf, dst, sizeof(dst));

    bool ok = true;
    for (int i = 0; i < 4; ++i)
        ok &= (std::fabs(dst[i] - src[i]) < 1e-7f);

    printf("Buffer test : %s\n", ok ? "OK" : "FAIL");

    destroyBuffer(ctx, buf);
    destroyContext(ctx);
    printf("--- Week 2 smoke test PASSED ---\n");
    return ok ? 0 : 1;
#endif
}
