#include "vk_context.h"

#include <vector>
#include <stdexcept>
#include <cstring>
#include <cstdio>

static uint32_t findComputeQueueFamily(VkPhysicalDevice physDev) {
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physDev, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(physDev, &count, families.data());

    for (uint32_t i = 0; i < count; ++i) {
        if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
            return i;
    }
    throw std::runtime_error("No compute queue family found on this device.");
}

static bool hasComputeQueue(VkPhysicalDevice physDev) {
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physDev, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(physDev, &count, families.data());
    for (auto& f : families)
        if (f.queueFlags & VK_QUEUE_COMPUTE_BIT) return true;
    return false;
}

static const char* deviceTypeName(VkPhysicalDeviceType t) {
    switch (t) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   return "Discrete GPU";
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "Integrated GPU";
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:    return "Virtual GPU";
        case VK_PHYSICAL_DEVICE_TYPE_CPU:            return "CPU";
        default:                                      return "Other";
    }
}

static VkInstance makeInstance() {
    VkApplicationInfo appInfo{};
    appInfo.sType            = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "vulkan-nn-lib";
    appInfo.apiVersion       = VK_API_VERSION_1_3;

    VkInstanceCreateInfo ci{};
    ci.sType            = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo = &appInfo;

    VkInstance inst = VK_NULL_HANDLE;
    VK_CHECK(vkCreateInstance(&ci, nullptr, &inst));
    return inst;
}

// ---------------------------------------------------------------------------
std::vector<DeviceInfo> enumerateDevices() {
    VkInstance inst = makeInstance();

    uint32_t count = 0;
    vkEnumeratePhysicalDevices(inst, &count, nullptr);
    std::vector<VkPhysicalDevice> devs(count);
    vkEnumeratePhysicalDevices(inst, &count, devs.data());

    std::vector<DeviceInfo> result;
    for (uint32_t i = 0; i < count; ++i) {
        if (!hasComputeQueue(devs[i])) continue;

        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(devs[i], &props);

        DeviceInfo info;
        info.index      = i;
        info.name       = props.deviceName;
        info.typeName   = deviceTypeName(props.deviceType);
        info.apiVersion = props.apiVersion;
        result.push_back(std::move(info));
    }

    vkDestroyInstance(inst, nullptr);
    return result;
}

// ---------------------------------------------------------------------------
VkContext createContextForDevice(uint32_t deviceIndex) {
    VkContext ctx{};

    VkApplicationInfo appInfo{};
    appInfo.sType            = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "vulkan-nn-lib";
    appInfo.apiVersion       = VK_API_VERSION_1_3;

    VkInstanceCreateInfo instCI{};
    instCI.sType            = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instCI.pApplicationInfo = &appInfo;
    VK_CHECK(vkCreateInstance(&instCI, nullptr, &ctx.instance));

    uint32_t devCount = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(ctx.instance, &devCount, nullptr));
    if (devCount == 0)
        throw std::runtime_error("No Vulkan-capable GPU found.");
    if (deviceIndex >= devCount)
        throw std::runtime_error("Device index " + std::to_string(deviceIndex) +
                                 " out of range (found " + std::to_string(devCount) + ").");

    std::vector<VkPhysicalDevice> devices(devCount);
    VK_CHECK(vkEnumeratePhysicalDevices(ctx.instance, &devCount, devices.data()));
    ctx.physDevice = devices[deviceIndex];

    ctx.computeFamilyIdx = findComputeQueueFamily(ctx.physDevice);

    float priority = 1.0f;
    VkDeviceQueueCreateInfo queueCI{};
    queueCI.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCI.queueFamilyIndex = ctx.computeFamilyIdx;
    queueCI.queueCount       = 1;
    queueCI.pQueuePriorities = &priority;

    VkDeviceCreateInfo devCI{};
    devCI.sType                = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    devCI.queueCreateInfoCount = 1;
    devCI.pQueueCreateInfos    = &queueCI;
    VK_CHECK(vkCreateDevice(ctx.physDevice, &devCI, nullptr, &ctx.device));

    vkGetDeviceQueue(ctx.device, ctx.computeFamilyIdx, 0, &ctx.computeQueue);

    VkCommandPoolCreateInfo poolCI{};
    poolCI.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolCI.queueFamilyIndex = ctx.computeFamilyIdx;
    poolCI.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK(vkCreateCommandPool(ctx.device, &poolCI, nullptr, &ctx.commandPool));

    return ctx;
}

VkContext createContext() {
    return createContextForDevice(0);
}

void destroyContext(VkContext& ctx) {
    if (ctx.commandPool != VK_NULL_HANDLE) vkDestroyCommandPool(ctx.device, ctx.commandPool, nullptr);
    if (ctx.device      != VK_NULL_HANDLE) vkDestroyDevice(ctx.device, nullptr);
    if (ctx.instance    != VK_NULL_HANDLE) vkDestroyInstance(ctx.instance, nullptr);
    ctx = VkContext{};
}
