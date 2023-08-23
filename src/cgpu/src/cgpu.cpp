//
// Copyright (C) 2023 Pablo Delgado Krämer
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.
//

#include "cgpu.h"
#include "shaderReflection.h"

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

#include <volk.h>

#pragma clang diagnostic push
// Silence nullability log spam on AppleClang
#pragma clang diagnostic ignored "-Wnullability-completeness"
#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>
#pragma clang diagnostic pop

#include <memory>

// TODO: should be in 'gtl/gb' subfolder
#include <smallVector.h>
#include <linearDataStore.h>

using namespace gtl;

#define CGPU_MIN_VK_API_VERSION VK_API_VERSION_1_1

/* Internal structures. */

struct CgpuIDevice
{
  VkDevice                     logicalDevice;
  VkPhysicalDevice             physicalDevice;
  VkQueue                      computeQueue;
  VkCommandPool                commandPool;
  VkQueryPool                  timestamp_pool;
  VolkDeviceTable              table;
  CgpuPhysicalDeviceFeatures   features;
  CgpuPhysicalDeviceProperties properties;
  VmaAllocator                 allocator;
};

struct CgpuIBuffer
{
  VkBuffer       buffer;
  uint64_t       size;
  VmaAllocation  allocation;
};

struct CgpuIImage
{
  VkImage       image;
  VkImageView   imageView;
  VmaAllocation allocation;
  uint64_t      size;
  uint32_t      width;
  uint32_t      height;
  uint32_t      depth;
  VkImageLayout layout;
  VkAccessFlags accessMask;
};

struct CgpuIPipeline
{
  VkPipeline                                       pipeline;
  VkPipelineLayout                                 layout;
  VkDescriptorPool                                 descriptorPool;
  VkDescriptorSet                                  descriptorSet;
  VkDescriptorSetLayout                            descriptorSetLayout;
  GbSmallVector<VkDescriptorSetLayoutBinding, 128> descriptorSetLayoutBindings;
  VkPipelineBindPoint                              bindPoint;
  VkStridedDeviceAddressRegionKHR                  sbtRgen;
  VkStridedDeviceAddressRegionKHR                  sbtMiss;
  VkStridedDeviceAddressRegionKHR                  sbtHit;
  CgpuIBuffer                                      sbt;
};

struct CgpuIShader
{
  VkShaderModule module;
  CgpuShaderReflection reflection;
  VkShaderStageFlagBits stageFlags;
};

struct CgpuIFence
{
  VkFence fence;
};

struct CgpuICommandBuffer
{
  VkCommandBuffer commandBuffer;
  CgpuDevice device;
};

struct CgpuIBlas
{
  VkAccelerationStructureKHR as;
  uint64_t address;
  CgpuIBuffer buffer;
  CgpuIBuffer indices;
  CgpuIBuffer vertices;
  bool isOpaque;
};

struct CgpuITlas
{
  VkAccelerationStructureKHR as;
  CgpuIBuffer buffer;
  CgpuIBuffer instances;
};

struct CgpuISampler
{
  VkSampler sampler;
};

struct CgpuIInstance
{
  VkInstance instance;
  GbLinearDataStore<CgpuIDevice, 32> ideviceStore;
  GbLinearDataStore<CgpuIBuffer, 16> ibufferStore;
  GbLinearDataStore<CgpuIImage, 128> iimageStore;
  GbLinearDataStore<CgpuIShader, 32> ishaderStore;
  GbLinearDataStore<CgpuIPipeline, 8> ipipelineStore;
  GbLinearDataStore<CgpuIFence, 8> ifenceStore;
  GbLinearDataStore<CgpuICommandBuffer, 16> icommandBufferStore;
  GbLinearDataStore<CgpuISampler, 8> isamplerStore;
  GbLinearDataStore<CgpuIBlas, 1024> iblasStore;
  GbLinearDataStore<CgpuITlas, 1> itlasStore;
};

static std::unique_ptr<CgpuIInstance> iinstance = nullptr;

/* Helper defines. */

#if defined(NDEBUG)
  #if defined(__GNUC__)
    #define CGPU_INLINE inline __attribute__((__always_inline__))
  #elif defined(_MSC_VER)
    #define CGPU_INLINE __forceinline
  #else
    #define CGPU_INLINE inline
  #endif
#else
  #define CGPU_INLINE
#endif

#define CGPU_RETURN_ERROR(msg)                                        \
  do {                                                                \
    fprintf(stderr, "error in %s:%d: %s\n", __FILE__, __LINE__, msg); \
    return false;                                                     \
  } while (false)

#define CGPU_RETURN_ERROR_INVALID_HANDLE                              \
  CGPU_RETURN_ERROR("invalid resource handle")

#define CGPU_RETURN_ERROR_HARDCODED_LIMIT_REACHED                     \
  CGPU_RETURN_ERROR("hardcoded limit reached")

#define CGPU_RESOLVE_HANDLE(RESOURCE_NAME, HANDLE_TYPE, IRESOURCE_TYPE, RESOURCE_STORE)   \
  CGPU_INLINE static bool cgpuResolve##RESOURCE_NAME(                                     \
    HANDLE_TYPE handle,                                                                   \
    IRESOURCE_TYPE** idata)                                                               \
  {                                                                                       \
    return iinstance->RESOURCE_STORE.get(handle.handle, idata);                           \
  }

CGPU_RESOLVE_HANDLE(       Device,        CgpuDevice,        CgpuIDevice,         ideviceStore)
CGPU_RESOLVE_HANDLE(       Buffer,        CgpuBuffer,        CgpuIBuffer,         ibufferStore)
CGPU_RESOLVE_HANDLE(        Image,         CgpuImage,         CgpuIImage,          iimageStore)
CGPU_RESOLVE_HANDLE(       Shader,        CgpuShader,        CgpuIShader,         ishaderStore)
CGPU_RESOLVE_HANDLE(     Pipeline,      CgpuPipeline,      CgpuIPipeline,       ipipelineStore)
CGPU_RESOLVE_HANDLE(        Fence,         CgpuFence,         CgpuIFence,          ifenceStore)
CGPU_RESOLVE_HANDLE(CommandBuffer, CgpuCommandBuffer, CgpuICommandBuffer, icommandBufferStore)
CGPU_RESOLVE_HANDLE(      Sampler,       CgpuSampler,       CgpuISampler,        isamplerStore)
CGPU_RESOLVE_HANDLE(         Blas,          CgpuBlas,          CgpuIBlas,           iblasStore)
CGPU_RESOLVE_HANDLE(         Tlas,          CgpuTlas,          CgpuITlas,           itlasStore)

/* Helper methods. */

static CgpuPhysicalDeviceFeatures cgpuTranslatePhysicalDeviceFeatures(const VkPhysicalDeviceFeatures* vkFeatures)
{
  CgpuPhysicalDeviceFeatures features = {};
  features.textureCompressionBC = vkFeatures->textureCompressionBC;
  features.pipelineStatisticsQuery = vkFeatures->pipelineStatisticsQuery;
  features.shaderImageGatherExtended = vkFeatures->shaderImageGatherExtended;
  features.shaderStorageImageExtendedFormats = vkFeatures->shaderStorageImageExtendedFormats;
  features.shaderStorageImageReadWithoutFormat = vkFeatures->shaderStorageImageReadWithoutFormat;
  features.shaderStorageImageWriteWithoutFormat = vkFeatures->shaderStorageImageWriteWithoutFormat;
  features.shaderUniformBufferArrayDynamicIndexing = vkFeatures->shaderUniformBufferArrayDynamicIndexing;
  features.shaderSampledImageArrayDynamicIndexing = vkFeatures->shaderSampledImageArrayDynamicIndexing;
  features.shaderStorageBufferArrayDynamicIndexing = vkFeatures->shaderStorageBufferArrayDynamicIndexing;
  features.shaderStorageImageArrayDynamicIndexing = vkFeatures->shaderStorageImageArrayDynamicIndexing;
  features.shaderFloat64 = vkFeatures->shaderFloat64;
  features.shaderInt64 = vkFeatures->shaderInt64;
  features.shaderInt16 = vkFeatures->shaderInt16;
  features.sparseBinding = vkFeatures->sparseBinding;
  features.sparseResidencyBuffer = vkFeatures->sparseResidencyBuffer;
  features.sparseResidencyImage2D = vkFeatures->sparseResidencyImage2D;
  features.sparseResidencyImage3D = vkFeatures->sparseResidencyImage3D;
  features.sparseResidencyAliased = vkFeatures->sparseResidencyAliased;
  return features;
}

static CgpuPhysicalDeviceProperties cgpuTranslatePhysicalDeviceProperties(const VkPhysicalDeviceLimits* vkLimits,
                                                                          const VkPhysicalDeviceSubgroupProperties* vkSubgroupProps,
                                                                          const VkPhysicalDeviceAccelerationStructurePropertiesKHR* vkAsProps,
                                                                          const VkPhysicalDeviceRayTracingPipelinePropertiesKHR* vkRtPipelineProps)
{
  CgpuPhysicalDeviceProperties properties = {};
  properties.maxImageDimension1D = vkLimits->maxImageDimension1D;
  properties.maxImageDimension2D = vkLimits->maxImageDimension2D;
  properties.maxImageDimension3D = vkLimits->maxImageDimension3D;
  properties.maxImageDimensionCube = vkLimits->maxImageDimensionCube;
  properties.maxImageArrayLayers = vkLimits->maxImageArrayLayers;
  properties.maxUniformBufferRange = vkLimits->maxUniformBufferRange;
  properties.maxStorageBufferRange = vkLimits->maxStorageBufferRange;
  properties.maxPushConstantsSize = vkLimits->maxPushConstantsSize;
  properties.maxMemoryAllocationCount = vkLimits->maxMemoryAllocationCount;
  properties.maxSamplerAllocationCount = vkLimits->maxSamplerAllocationCount;
  properties.bufferImageGranularity = vkLimits->bufferImageGranularity;
  properties.sparseAddressSpaceSize = vkLimits->sparseAddressSpaceSize;
  properties.maxBoundDescriptorSets = vkLimits->maxBoundDescriptorSets;
  properties.maxPerStageDescriptorSamplers = vkLimits->maxPerStageDescriptorSamplers;
  properties.maxPerStageDescriptorUniformBuffers = vkLimits->maxPerStageDescriptorUniformBuffers;
  properties.maxPerStageDescriptorStorageBuffers = vkLimits->maxPerStageDescriptorStorageBuffers;
  properties.maxPerStageDescriptorSampledImages = vkLimits->maxPerStageDescriptorSampledImages;
  properties.maxPerStageDescriptorStorageImages = vkLimits->maxPerStageDescriptorStorageImages;
  properties.maxPerStageDescriptorInputAttachments = vkLimits->maxPerStageDescriptorInputAttachments;
  properties.maxPerStageResources = vkLimits->maxPerStageResources;
  properties.maxDescriptorSetSamplers = vkLimits->maxDescriptorSetSamplers;
  properties.maxDescriptorSetUniformBuffers = vkLimits->maxDescriptorSetUniformBuffers;
  properties.maxDescriptorSetUniformBuffersDynamic = vkLimits->maxDescriptorSetUniformBuffersDynamic;
  properties.maxDescriptorSetStorageBuffers = vkLimits->maxDescriptorSetStorageBuffers;
  properties.maxDescriptorSetStorageBuffersDynamic = vkLimits->maxDescriptorSetStorageBuffersDynamic;
  properties.maxDescriptorSetSampledImages = vkLimits->maxDescriptorSetSampledImages;
  properties.maxDescriptorSetStorageImages = vkLimits->maxDescriptorSetStorageImages;
  properties.maxDescriptorSetInputAttachments = vkLimits->maxDescriptorSetInputAttachments;
  properties.maxComputeSharedMemorySize = vkLimits->maxComputeSharedMemorySize;
  properties.maxComputeWorkGroupCount[0] = vkLimits->maxComputeWorkGroupCount[0];
  properties.maxComputeWorkGroupCount[1] = vkLimits->maxComputeWorkGroupCount[1];
  properties.maxComputeWorkGroupCount[2] = vkLimits->maxComputeWorkGroupCount[2];
  properties.maxComputeWorkGroupInvocations = vkLimits->maxComputeWorkGroupInvocations;
  properties.maxComputeWorkGroupSize[0] = vkLimits->maxComputeWorkGroupSize[0];
  properties.maxComputeWorkGroupSize[1] = vkLimits->maxComputeWorkGroupSize[1];
  properties.maxComputeWorkGroupSize[2] = vkLimits->maxComputeWorkGroupSize[2];
  properties.mipmapPrecisionBits = vkLimits->mipmapPrecisionBits;
  properties.maxSamplerLodBias = vkLimits->maxSamplerLodBias;
  properties.maxSamplerAnisotropy = vkLimits->maxSamplerAnisotropy;
  properties.minMemoryMapAlignment = vkLimits->minMemoryMapAlignment;
  properties.minUniformBufferOffsetAlignment = vkLimits->minUniformBufferOffsetAlignment;
  properties.minStorageBufferOffsetAlignment = vkLimits->minStorageBufferOffsetAlignment;
  properties.minTexelOffset = vkLimits->minTexelOffset;
  properties.maxTexelOffset = vkLimits->maxTexelOffset;
  properties.minTexelGatherOffset = vkLimits->minTexelGatherOffset;
  properties.maxTexelGatherOffset = vkLimits->maxTexelGatherOffset;
  properties.minInterpolationOffset = vkLimits->minInterpolationOffset;
  properties.maxInterpolationOffset = vkLimits->maxInterpolationOffset;
  properties.subPixelInterpolationOffsetBits = vkLimits->subPixelInterpolationOffsetBits;
  properties.maxSampleMaskWords = vkLimits->maxSampleMaskWords;
  properties.timestampComputeAndGraphics = vkLimits->timestampComputeAndGraphics;
  properties.timestampPeriod = vkLimits->timestampPeriod;
  properties.discreteQueuePriorities = vkLimits->discreteQueuePriorities;
  properties.optimalBufferCopyOffsetAlignment = vkLimits->optimalBufferCopyOffsetAlignment;
  properties.optimalBufferCopyRowPitchAlignment = vkLimits->optimalBufferCopyRowPitchAlignment;
  properties.nonCoherentAtomSize = vkLimits->nonCoherentAtomSize;
  properties.subgroupSize = vkSubgroupProps->subgroupSize;
  properties.minAccelerationStructureScratchOffsetAlignment = vkAsProps->minAccelerationStructureScratchOffsetAlignment;
  properties.shaderGroupHandleSize = vkRtPipelineProps->shaderGroupHandleSize;
  properties.maxShaderGroupStride = vkRtPipelineProps->maxShaderGroupStride;
  properties.shaderGroupBaseAlignment = vkRtPipelineProps->shaderGroupBaseAlignment;
  properties.shaderGroupHandleCaptureReplaySize = vkRtPipelineProps->shaderGroupHandleCaptureReplaySize;
  properties.maxRayDispatchInvocationCount = vkRtPipelineProps->maxRayDispatchInvocationCount;
  properties.shaderGroupHandleAlignment = vkRtPipelineProps->shaderGroupHandleAlignment;
  properties.maxRayHitAttributeSize = vkRtPipelineProps->maxRayHitAttributeSize;
  return properties;
}

static VkSamplerAddressMode cgpuTranslateAddressMode(CgpuSamplerAddressMode mode)
{
  switch (mode)
  {
  case CGPU_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  case CGPU_SAMPLER_ADDRESS_MODE_REPEAT: return VK_SAMPLER_ADDRESS_MODE_REPEAT;
  case CGPU_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT: return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
  case CGPU_SAMPLER_ADDRESS_MODE_CLAMP_TO_BLACK: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
  default: return VK_SAMPLER_ADDRESS_MODE_MAX_ENUM;
  }
}

static VkPipelineStageFlags cgpuPipelineStageFlagsFromShaderStageFlags(VkShaderStageFlags shaderStageFlags)
{
  VkPipelineStageFlags pipelineStageFlags = (VkPipelineStageFlags) 0;
  if (shaderStageFlags & VK_SHADER_STAGE_COMPUTE_BIT)
  {
    pipelineStageFlags |= VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
  }
  if ((shaderStageFlags & VK_SHADER_STAGE_RAYGEN_BIT_KHR) |
      (shaderStageFlags & VK_SHADER_STAGE_ANY_HIT_BIT_KHR) |
      (shaderStageFlags & VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR) |
      (shaderStageFlags & VK_SHADER_STAGE_MISS_BIT_KHR) |
      (shaderStageFlags & VK_SHADER_STAGE_INTERSECTION_BIT_KHR))
  {
    pipelineStageFlags |= VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;
  }

  assert(int(pipelineStageFlags) != 0);
  return pipelineStageFlags;
}

/* API method implementation. */

static bool cgpuFindLayer(const char* name, uint32_t layerCount, VkLayerProperties* layers)
{
  for (uint32_t i = 0; i < layerCount; i++)
  {
    if (!strcmp(layers[i].layerName, name))
    {
      return true;
    }
  }
  return false;
}

static bool cgpuFindExtension(const char* name, uint32_t extensionCount, VkExtensionProperties* extensions)
{
  for (uint32_t i = 0; i < extensionCount; ++i)
  {
    if (!strcmp(extensions[i].extensionName, name))
    {
      return true;
    }
  }
  return false;
}

bool cgpuInitialize(const char* appName, uint32_t versionMajor, uint32_t versionMinor, uint32_t versionPatch)
{
  if (volkInitialize() != VK_SUCCESS || volkGetInstanceVersion() < CGPU_MIN_VK_API_VERSION)
  {
    CGPU_RETURN_ERROR("failed to initialize volk");
  }

  GbSmallVector<const char*, 8> enabledLayers;
  GbSmallVector<const char*, 8> enabledExtensions;
#ifndef NDEBUG
  {
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

    GbSmallVector<VkLayerProperties, 16> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    const char* VK_LAYER_KHRONOS_VALIDATION_NAME = "VK_LAYER_KHRONOS_validation";

    if (cgpuFindLayer(VK_LAYER_KHRONOS_VALIDATION_NAME, availableLayers.size(), availableLayers.data()))
    {
      enabledLayers.push_back(VK_LAYER_KHRONOS_VALIDATION_NAME);
    }
  }
  {
    uint32_t extensionCount;
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);

    GbSmallVector<VkExtensionProperties, 512> availableExtensions(extensionCount);
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, availableExtensions.data());

    if (cgpuFindExtension(VK_EXT_DEBUG_UTILS_EXTENSION_NAME, availableExtensions.size(), availableExtensions.data()))
    {
      enabledExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }
  }
#endif

  VkApplicationInfo appInfo;
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pNext = nullptr;
  appInfo.pApplicationName = appName;
  appInfo.applicationVersion = VK_MAKE_VERSION(versionMajor, versionMinor, versionPatch);
  appInfo.pEngineName = appName;
  appInfo.engineVersion = VK_MAKE_VERSION(versionMajor, versionMinor, versionPatch);
  appInfo.apiVersion = CGPU_MIN_VK_API_VERSION;

  VkInstanceCreateInfo createInfo;
  createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  createInfo.pNext = nullptr;
  createInfo.flags = 0;
  createInfo.pApplicationInfo = &appInfo;
  createInfo.enabledLayerCount = enabledLayers.size();
  createInfo.ppEnabledLayerNames = enabledLayers.data();
  createInfo.enabledExtensionCount = enabledExtensions.size();
  createInfo.ppEnabledExtensionNames = enabledExtensions.data();

  VkInstance instance;
  if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS)
  {
    CGPU_RETURN_ERROR("failed to create vulkan instance");
  }

  volkLoadInstanceOnly(instance);

  iinstance = std::make_unique<CgpuIInstance>();
  iinstance->instance = instance;
  return true;
}

void cgpuTerminate()
{
  vkDestroyInstance(iinstance->instance, nullptr);
  iinstance.reset();
}

bool cgpuCreateDevice(CgpuDevice* device)
{
  device->handle = iinstance->ideviceStore.allocate();

  CgpuIDevice* idevice;
  if (!cgpuResolveDevice(*device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  uint32_t physDeviceCount;
  vkEnumeratePhysicalDevices(
    iinstance->instance,
    &physDeviceCount,
    nullptr
  );

  if (physDeviceCount == 0)
  {
    iinstance->ideviceStore.free(device->handle);
    CGPU_RETURN_ERROR("no physical device found");
  }

  GbSmallVector<VkPhysicalDevice, 8> phys_devices;
  phys_devices.resize(physDeviceCount);

  vkEnumeratePhysicalDevices(
    iinstance->instance,
    &physDeviceCount,
    phys_devices.data()
  );

  idevice->physicalDevice = phys_devices[0];

  VkPhysicalDeviceFeatures features;
  vkGetPhysicalDeviceFeatures(idevice->physicalDevice, &features);
  idevice->features = cgpuTranslatePhysicalDeviceFeatures(&features);

  VkPhysicalDeviceAccelerationStructurePropertiesKHR asProperties = {};
  asProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR;
  asProperties.pNext = nullptr;

  VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtPipelineProperties = {};
  rtPipelineProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
  rtPipelineProperties.pNext = &asProperties;

  VkPhysicalDeviceSubgroupProperties subgroupProperties = {};
  subgroupProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
  subgroupProperties.pNext = &rtPipelineProperties;

  VkPhysicalDeviceProperties2 deviceProperties = {};
  deviceProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  deviceProperties.pNext = &subgroupProperties;

  vkGetPhysicalDeviceProperties2(idevice->physicalDevice, &deviceProperties);

  if (deviceProperties.properties.apiVersion < CGPU_MIN_VK_API_VERSION)
  {
    iinstance->ideviceStore.free(device->handle);
    CGPU_RETURN_ERROR("unsupported vulkan version");
  }

  if ((subgroupProperties.supportedStages & VK_QUEUE_COMPUTE_BIT) != VK_QUEUE_COMPUTE_BIT ||
      (subgroupProperties.supportedOperations & VK_SUBGROUP_FEATURE_BASIC_BIT) != VK_SUBGROUP_FEATURE_BASIC_BIT ||
      (subgroupProperties.supportedOperations & VK_SUBGROUP_FEATURE_BALLOT_BIT) != VK_SUBGROUP_FEATURE_BALLOT_BIT)
  {
    iinstance->ideviceStore.free(device->handle);
    CGPU_RETURN_ERROR("subgroup features not supported");
  }

  const VkPhysicalDeviceLimits* limits = &deviceProperties.properties.limits;
  idevice->properties = cgpuTranslatePhysicalDeviceProperties(limits, &subgroupProperties, &asProperties, &rtPipelineProperties);

  uint32_t deviceExtCount;
  vkEnumerateDeviceExtensionProperties(
    idevice->physicalDevice,
    nullptr,
    &deviceExtCount,
    nullptr
  );

  GbSmallVector<VkExtensionProperties, 1024> deviceExtensions;
  deviceExtensions.resize(deviceExtCount);

  vkEnumerateDeviceExtensionProperties(
    idevice->physicalDevice,
    nullptr,
    &deviceExtCount,
    deviceExtensions.data()
  );

  const char* requiredExtensions[] = {
    VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
    VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME, // required by VK_KHR_acceleration_structure
    VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME, // required by VK_KHR_acceleration_structure
    VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME, // required by VK_KHR_acceleration_structure
    VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
    VK_KHR_SPIRV_1_4_EXTENSION_NAME, // required by VK_KHR_ray_tracing_pipeline
    VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME, // required by VK_KHR_spirv_1_4
    VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME
  };
  uint32_t requiredExtensionCount = sizeof(requiredExtensions) / sizeof(requiredExtensions[0]);

  GbSmallVector<const char*, 32> enabledDeviceExtensions;
  for (uint32_t i = 0; i < requiredExtensionCount; i++)
  {
    const char* extension = requiredExtensions[i];

    if (!cgpuFindExtension(extension, deviceExtCount, deviceExtensions.data()))
    {
      iinstance->ideviceStore.free(device->handle);

      fprintf(stderr, "error in %s:%d: extension %s not supported\n", __FILE__, __LINE__, extension);
      return false;
    }

    enabledDeviceExtensions.push_back(extension);
  }

  const char* VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME = "VK_KHR_portability_subset";
  if (cgpuFindExtension(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME, deviceExtCount, deviceExtensions.data()))
  {
    enabledDeviceExtensions.push_back(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME);
  }

#ifndef NDEBUG
  if (cgpuFindExtension(VK_KHR_SHADER_CLOCK_EXTENSION_NAME, deviceExtCount, deviceExtensions.data()) && features.shaderInt64)
  {
    idevice->features.shaderClock = true;
    enabledDeviceExtensions.push_back(VK_KHR_SHADER_CLOCK_EXTENSION_NAME);
  }

#ifndef __APPLE__
  if (cgpuFindExtension(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME, deviceExtCount, deviceExtensions.data()))
  {
    idevice->features.debugPrintf = true;
    enabledDeviceExtensions.push_back(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME);
  }
#endif
#endif
  if (cgpuFindExtension(VK_EXT_MEMORY_PRIORITY_EXTENSION_NAME, deviceExtCount, deviceExtensions.data()) &&
      cgpuFindExtension(VK_EXT_PAGEABLE_DEVICE_LOCAL_MEMORY_EXTENSION_NAME, deviceExtCount, deviceExtensions.data()))
  {
    idevice->features.pageableDeviceLocalMemory = true;
    enabledDeviceExtensions.push_back(VK_EXT_MEMORY_PRIORITY_EXTENSION_NAME);
    enabledDeviceExtensions.push_back(VK_EXT_PAGEABLE_DEVICE_LOCAL_MEMORY_EXTENSION_NAME);
  }

  if (cgpuFindExtension(VK_NV_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME, deviceExtCount, deviceExtensions.data()))
  {
    idevice->features.rayTracingInvocationReorder = true;
    enabledDeviceExtensions.push_back(VK_NV_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME);
  }

  uint32_t queueFamilyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(
    idevice->physicalDevice,
    &queueFamilyCount,
    nullptr
  );

  GbSmallVector<VkQueueFamilyProperties, 32> queueFamilies;
  queueFamilies.resize(queueFamilyCount);

  vkGetPhysicalDeviceQueueFamilyProperties(
    idevice->physicalDevice,
    &queueFamilyCount,
    queueFamilies.data()
  );

  int32_t queueFamilyIndex = -1;
  for (uint32_t i = 0; i < queueFamilyCount; ++i)
  {
    const VkQueueFamilyProperties* queue_family = &queueFamilies[i];

    if ((queue_family->queueFlags & VK_QUEUE_COMPUTE_BIT) && (queue_family->queueFlags & VK_QUEUE_TRANSFER_BIT))
    {
      queueFamilyIndex = i;
    }
  }
  if (queueFamilyIndex == -1) {
    iinstance->ideviceStore.free(device->handle);
    CGPU_RETURN_ERROR("no suitable queue family");
  }

  VkDeviceQueueCreateInfo queueCreateInfo = {};
  queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queueCreateInfo.pNext = nullptr;
  queueCreateInfo.flags = 0;
  queueCreateInfo.queueFamilyIndex = queueFamilyIndex;
  queueCreateInfo.queueCount = 1;
  const float queue_priority = 1.0f;
  queueCreateInfo.pQueuePriorities = &queue_priority;

  void* pNext = nullptr;

  VkPhysicalDevicePageableDeviceLocalMemoryFeaturesEXT pageableMemoryFeatures = {};
  pageableMemoryFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PAGEABLE_DEVICE_LOCAL_MEMORY_FEATURES_EXT;
  pageableMemoryFeatures.pNext = nullptr;
  pageableMemoryFeatures.pageableDeviceLocalMemory = VK_TRUE;

  if (idevice->features.pageableDeviceLocalMemory)
  {
    pNext = &pageableMemoryFeatures;
  }

  VkPhysicalDeviceShaderClockFeaturesKHR shaderClockFeatures = {};
  shaderClockFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR;
  shaderClockFeatures.pNext = pNext;
  shaderClockFeatures.shaderSubgroupClock = VK_TRUE;
  shaderClockFeatures.shaderDeviceClock = VK_FALSE;

  if (idevice->features.shaderClock)
  {
    pNext = &shaderClockFeatures;
  }

  VkPhysicalDeviceRayTracingInvocationReorderFeaturesNV invocationReorderFeatures = {};
  invocationReorderFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_FEATURES_NV;
  invocationReorderFeatures.pNext = pNext;
  invocationReorderFeatures.rayTracingInvocationReorder = VK_TRUE;

  if (idevice->features.rayTracingInvocationReorder)
  {
    pNext = &invocationReorderFeatures;
  }

  VkPhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructureFeatures = {};
  accelerationStructureFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
  accelerationStructureFeatures.pNext = pNext;
  accelerationStructureFeatures.accelerationStructure = VK_TRUE;
  accelerationStructureFeatures.accelerationStructureCaptureReplay = VK_FALSE;
  accelerationStructureFeatures.accelerationStructureIndirectBuild = VK_FALSE;
  accelerationStructureFeatures.accelerationStructureHostCommands = VK_FALSE;
  accelerationStructureFeatures.descriptorBindingAccelerationStructureUpdateAfterBind = VK_FALSE;

  VkPhysicalDeviceRayTracingPipelineFeaturesKHR rayTracingPipelineFeatures = {};
  rayTracingPipelineFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
  rayTracingPipelineFeatures.pNext = &accelerationStructureFeatures;
  rayTracingPipelineFeatures.rayTracingPipeline = VK_TRUE;
  rayTracingPipelineFeatures.rayTracingPipelineShaderGroupHandleCaptureReplay = VK_FALSE;
  rayTracingPipelineFeatures.rayTracingPipelineShaderGroupHandleCaptureReplayMixed = VK_FALSE;
  rayTracingPipelineFeatures.rayTracingPipelineTraceRaysIndirect = VK_FALSE;
  rayTracingPipelineFeatures.rayTraversalPrimitiveCulling = VK_FALSE;

  VkPhysicalDeviceBufferDeviceAddressFeaturesKHR bufferDeviceAddressFeatures = {};
  bufferDeviceAddressFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
  bufferDeviceAddressFeatures.pNext = &rayTracingPipelineFeatures;
  bufferDeviceAddressFeatures.bufferDeviceAddress = VK_TRUE;
  bufferDeviceAddressFeatures.bufferDeviceAddressCaptureReplay = VK_FALSE;
  bufferDeviceAddressFeatures.bufferDeviceAddressMultiDevice = VK_FALSE;

  VkPhysicalDeviceDescriptorIndexingFeaturesEXT descriptorIndexingFeatures = {};
  descriptorIndexingFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES;
  descriptorIndexingFeatures.pNext = &bufferDeviceAddressFeatures;
  descriptorIndexingFeatures.shaderInputAttachmentArrayDynamicIndexing = VK_FALSE;
  descriptorIndexingFeatures.shaderUniformTexelBufferArrayDynamicIndexing = VK_FALSE;
  descriptorIndexingFeatures.shaderStorageTexelBufferArrayDynamicIndexing = VK_FALSE;
  descriptorIndexingFeatures.shaderUniformBufferArrayNonUniformIndexing = VK_FALSE;
  descriptorIndexingFeatures.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
  descriptorIndexingFeatures.shaderStorageBufferArrayNonUniformIndexing = VK_FALSE;
  descriptorIndexingFeatures.shaderStorageImageArrayNonUniformIndexing = VK_TRUE;
  descriptorIndexingFeatures.shaderInputAttachmentArrayNonUniformIndexing = VK_FALSE;
  descriptorIndexingFeatures.shaderUniformTexelBufferArrayNonUniformIndexing = VK_FALSE;
  descriptorIndexingFeatures.shaderStorageTexelBufferArrayNonUniformIndexing = VK_FALSE;
  descriptorIndexingFeatures.descriptorBindingUniformBufferUpdateAfterBind = VK_FALSE;
  descriptorIndexingFeatures.descriptorBindingSampledImageUpdateAfterBind = VK_FALSE;
  descriptorIndexingFeatures.descriptorBindingStorageImageUpdateAfterBind = VK_FALSE;
  descriptorIndexingFeatures.descriptorBindingStorageBufferUpdateAfterBind = VK_FALSE;
  descriptorIndexingFeatures.descriptorBindingUniformTexelBufferUpdateAfterBind = VK_FALSE;
  descriptorIndexingFeatures.descriptorBindingStorageTexelBufferUpdateAfterBind = VK_FALSE;
  descriptorIndexingFeatures.descriptorBindingUpdateUnusedWhilePending = VK_FALSE;
  descriptorIndexingFeatures.descriptorBindingPartiallyBound = VK_FALSE;
  descriptorIndexingFeatures.descriptorBindingVariableDescriptorCount = VK_FALSE;
  descriptorIndexingFeatures.runtimeDescriptorArray = VK_FALSE;

  VkPhysicalDeviceShaderFloat16Int8Features shaderFloat16Int8Features = {};
  shaderFloat16Int8Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
  shaderFloat16Int8Features.pNext = &descriptorIndexingFeatures;
  shaderFloat16Int8Features.shaderFloat16 = VK_TRUE;
  shaderFloat16Int8Features.shaderInt8 = VK_FALSE;

  VkPhysicalDevice16BitStorageFeatures device16bitStorageFeatures = {};
  device16bitStorageFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
  device16bitStorageFeatures.pNext = &shaderFloat16Int8Features;
  device16bitStorageFeatures.storageBuffer16BitAccess = VK_TRUE;
  device16bitStorageFeatures.uniformAndStorageBuffer16BitAccess = VK_TRUE;
  device16bitStorageFeatures.storagePushConstant16 = VK_FALSE;
  device16bitStorageFeatures.storageInputOutput16 = VK_FALSE;

  VkPhysicalDeviceFeatures2 deviceFeatures2;
  deviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  deviceFeatures2.pNext = &device16bitStorageFeatures;
  deviceFeatures2.features.robustBufferAccess = VK_FALSE;
  deviceFeatures2.features.fullDrawIndexUint32 = VK_FALSE;
  deviceFeatures2.features.imageCubeArray = VK_FALSE;
  deviceFeatures2.features.independentBlend = VK_FALSE;
  deviceFeatures2.features.geometryShader = VK_FALSE;
  deviceFeatures2.features.tessellationShader = VK_FALSE;
  deviceFeatures2.features.sampleRateShading = VK_FALSE;
  deviceFeatures2.features.dualSrcBlend = VK_FALSE;
  deviceFeatures2.features.logicOp = VK_FALSE;
  deviceFeatures2.features.multiDrawIndirect = VK_FALSE;
  deviceFeatures2.features.drawIndirectFirstInstance = VK_FALSE;
  deviceFeatures2.features.depthClamp = VK_FALSE;
  deviceFeatures2.features.depthBiasClamp = VK_FALSE;
  deviceFeatures2.features.fillModeNonSolid = VK_FALSE;
  deviceFeatures2.features.depthBounds = VK_FALSE;
  deviceFeatures2.features.wideLines = VK_FALSE;
  deviceFeatures2.features.largePoints = VK_FALSE;
  deviceFeatures2.features.alphaToOne = VK_FALSE;
  deviceFeatures2.features.multiViewport = VK_FALSE;
  deviceFeatures2.features.samplerAnisotropy = VK_TRUE;
  deviceFeatures2.features.textureCompressionETC2 = VK_FALSE;
  deviceFeatures2.features.textureCompressionASTC_LDR = VK_FALSE;
  deviceFeatures2.features.textureCompressionBC = VK_FALSE;
  deviceFeatures2.features.occlusionQueryPrecise = VK_FALSE;
  deviceFeatures2.features.pipelineStatisticsQuery = VK_FALSE;
  deviceFeatures2.features.vertexPipelineStoresAndAtomics = VK_FALSE;
  deviceFeatures2.features.fragmentStoresAndAtomics = VK_FALSE;
  deviceFeatures2.features.shaderTessellationAndGeometryPointSize = VK_FALSE;
  deviceFeatures2.features.shaderImageGatherExtended = VK_TRUE;
  deviceFeatures2.features.shaderStorageImageExtendedFormats = VK_FALSE;
  deviceFeatures2.features.shaderStorageImageMultisample = VK_FALSE;
  deviceFeatures2.features.shaderStorageImageReadWithoutFormat = VK_FALSE;
  deviceFeatures2.features.shaderStorageImageWriteWithoutFormat = VK_FALSE;
  deviceFeatures2.features.shaderUniformBufferArrayDynamicIndexing = VK_FALSE;
  deviceFeatures2.features.shaderSampledImageArrayDynamicIndexing = VK_TRUE;
  deviceFeatures2.features.shaderStorageBufferArrayDynamicIndexing = VK_FALSE;
  deviceFeatures2.features.shaderStorageImageArrayDynamicIndexing = VK_FALSE;
  deviceFeatures2.features.shaderClipDistance = VK_FALSE;
  deviceFeatures2.features.shaderCullDistance = VK_FALSE;
  deviceFeatures2.features.shaderFloat64 = VK_FALSE;
  deviceFeatures2.features.shaderInt64 = idevice->features.shaderClock;
  deviceFeatures2.features.shaderInt16 = VK_TRUE;
  deviceFeatures2.features.shaderResourceResidency = VK_FALSE;
  deviceFeatures2.features.shaderResourceMinLod = VK_FALSE;
  deviceFeatures2.features.sparseBinding = VK_FALSE;
  deviceFeatures2.features.sparseResidencyBuffer = VK_FALSE;
  deviceFeatures2.features.sparseResidencyImage2D = VK_FALSE;
  deviceFeatures2.features.sparseResidencyImage3D = VK_FALSE;
  deviceFeatures2.features.sparseResidency2Samples = VK_FALSE;
  deviceFeatures2.features.sparseResidency4Samples = VK_FALSE;
  deviceFeatures2.features.sparseResidency8Samples = VK_FALSE;
  deviceFeatures2.features.sparseResidency16Samples = VK_FALSE;
  deviceFeatures2.features.sparseResidencyAliased = VK_FALSE;
  deviceFeatures2.features.variableMultisampleRate = VK_FALSE;
  deviceFeatures2.features.inheritedQueries = VK_FALSE;

  VkDeviceCreateInfo deviceCreateInfo;
  deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  deviceCreateInfo.pNext = &deviceFeatures2;
  deviceCreateInfo.flags = 0;
  deviceCreateInfo.queueCreateInfoCount = 1;
  deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
  /* These two fields are ignored by up-to-date implementations since
   * nowadays, there is no difference to instance validation layers. */
  deviceCreateInfo.enabledLayerCount = 0;
  deviceCreateInfo.ppEnabledLayerNames = nullptr;
  deviceCreateInfo.enabledExtensionCount = enabledDeviceExtensions.size();
  deviceCreateInfo.ppEnabledExtensionNames = enabledDeviceExtensions.data();
  deviceCreateInfo.pEnabledFeatures = nullptr;

  VkResult result = vkCreateDevice(
    idevice->physicalDevice,
    &deviceCreateInfo,
    nullptr,
    &idevice->logicalDevice
  );
  if (result != VK_SUCCESS) {
    iinstance->ideviceStore.free(device->handle);
    CGPU_RETURN_ERROR("failed to create device");
  }

  volkLoadDeviceTable(
    &idevice->table,
    idevice->logicalDevice
  );

  idevice->table.vkGetDeviceQueue(
    idevice->logicalDevice,
    queueFamilyIndex,
    0,
    &idevice->computeQueue
  );

  VkCommandPoolCreateInfo poolCreateInfo = {};
  poolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  poolCreateInfo.pNext = nullptr;
  poolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  poolCreateInfo.queueFamilyIndex = queueFamilyIndex;

  result = idevice->table.vkCreateCommandPool(
    idevice->logicalDevice,
    &poolCreateInfo,
    nullptr,
    &idevice->commandPool
  );

  if (result != VK_SUCCESS)
  {
    iinstance->ideviceStore.free(device->handle);

    idevice->table.vkDestroyDevice(
      idevice->logicalDevice,
      nullptr
    );

    CGPU_RETURN_ERROR("failed to create command pool");
  }

  VkQueryPoolCreateInfo timestampPoolCreateInfo = {};
  timestampPoolCreateInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
  timestampPoolCreateInfo.pNext = nullptr;
  timestampPoolCreateInfo.flags = 0;
  timestampPoolCreateInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
  timestampPoolCreateInfo.queryCount = CGPU_MAX_TIMESTAMP_QUERIES;
  timestampPoolCreateInfo.pipelineStatistics = 0;

  result = idevice->table.vkCreateQueryPool(
    idevice->logicalDevice,
    &timestampPoolCreateInfo,
    nullptr,
    &idevice->timestamp_pool
  );

  if (result != VK_SUCCESS)
  {
    iinstance->ideviceStore.free(device->handle);

    idevice->table.vkDestroyCommandPool(
      idevice->logicalDevice,
      idevice->commandPool,
      nullptr
    );
    idevice->table.vkDestroyDevice(
      idevice->logicalDevice,
      nullptr
    );

    CGPU_RETURN_ERROR("failed to create query pool");
  }

  VmaVulkanFunctions vmaVulkanFunctions = {};
  vmaVulkanFunctions.vkGetPhysicalDeviceProperties = vkGetPhysicalDeviceProperties;
  vmaVulkanFunctions.vkGetPhysicalDeviceMemoryProperties = vkGetPhysicalDeviceMemoryProperties;
  vmaVulkanFunctions.vkGetPhysicalDeviceMemoryProperties2KHR = vkGetPhysicalDeviceMemoryProperties2;
  vmaVulkanFunctions.vkAllocateMemory = idevice->table.vkAllocateMemory;
  vmaVulkanFunctions.vkFreeMemory = idevice->table.vkFreeMemory;
  vmaVulkanFunctions.vkMapMemory = idevice->table.vkMapMemory;
  vmaVulkanFunctions.vkUnmapMemory = idevice->table.vkUnmapMemory;
  vmaVulkanFunctions.vkFlushMappedMemoryRanges = idevice->table.vkFlushMappedMemoryRanges;
  vmaVulkanFunctions.vkInvalidateMappedMemoryRanges = idevice->table.vkInvalidateMappedMemoryRanges;
  vmaVulkanFunctions.vkBindBufferMemory = idevice->table.vkBindBufferMemory;
  vmaVulkanFunctions.vkBindImageMemory = idevice->table.vkBindImageMemory;
  vmaVulkanFunctions.vkGetBufferMemoryRequirements = idevice->table.vkGetBufferMemoryRequirements;
  vmaVulkanFunctions.vkGetImageMemoryRequirements = idevice->table.vkGetImageMemoryRequirements;
  vmaVulkanFunctions.vkCreateBuffer = idevice->table.vkCreateBuffer;
  vmaVulkanFunctions.vkDestroyBuffer = idevice->table.vkDestroyBuffer;
  vmaVulkanFunctions.vkCreateImage = idevice->table.vkCreateImage;
  vmaVulkanFunctions.vkDestroyImage = idevice->table.vkDestroyImage;
  vmaVulkanFunctions.vkCmdCopyBuffer = idevice->table.vkCmdCopyBuffer;
  vmaVulkanFunctions.vkGetBufferMemoryRequirements2KHR = idevice->table.vkGetBufferMemoryRequirements2;
  vmaVulkanFunctions.vkGetImageMemoryRequirements2KHR = idevice->table.vkGetImageMemoryRequirements2;
  vmaVulkanFunctions.vkBindBufferMemory2KHR = idevice->table.vkBindBufferMemory2;
  vmaVulkanFunctions.vkBindImageMemory2KHR = idevice->table.vkBindImageMemory2;

  VmaAllocatorCreateInfo allocCreateInfo = {};
  allocCreateInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
  allocCreateInfo.vulkanApiVersion = CGPU_MIN_VK_API_VERSION;
  allocCreateInfo.physicalDevice = idevice->physicalDevice;
  allocCreateInfo.device = idevice->logicalDevice;
  allocCreateInfo.instance = iinstance->instance;
  allocCreateInfo.pVulkanFunctions = &vmaVulkanFunctions;

  result = vmaCreateAllocator(&allocCreateInfo, &idevice->allocator);

  if (result != VK_SUCCESS)
  {
    iinstance->ideviceStore.free(device->handle);

    idevice->table.vkDestroyQueryPool(
      idevice->logicalDevice,
      idevice->timestamp_pool,
      nullptr
    );
    idevice->table.vkDestroyCommandPool(
      idevice->logicalDevice,
      idevice->commandPool,
      nullptr
    );
    idevice->table.vkDestroyDevice(
      idevice->logicalDevice,
      nullptr
    );
    CGPU_RETURN_ERROR("failed to create vma allocator");
  }

  return true;
}

bool cgpuDestroyDevice(CgpuDevice device)
{
  CgpuIDevice* idevice;
  if (!cgpuResolveDevice(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  vmaDestroyAllocator(idevice->allocator);

  idevice->table.vkDestroyQueryPool(
    idevice->logicalDevice,
    idevice->timestamp_pool,
    nullptr
  );
  idevice->table.vkDestroyCommandPool(
    idevice->logicalDevice,
    idevice->commandPool,
    nullptr
  );
  idevice->table.vkDestroyDevice(
    idevice->logicalDevice,
    nullptr
  );

  iinstance->ideviceStore.free(device.handle);
  return true;
}

bool cgpuCreateShader(CgpuDevice device,
                      uint64_t size,
                      const uint8_t* source,
                      CgpuShaderStageFlags stageFlags,
                      CgpuShader* shader)
{
  CgpuIDevice* idevice;
  if (!cgpuResolveDevice(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  shader->handle = iinstance->ishaderStore.allocate();

  CgpuIShader* ishader;
  if (!cgpuResolveShader(*shader, &ishader)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  VkShaderModuleCreateInfo shaderModuleCreateInfo;
  shaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  shaderModuleCreateInfo.pNext = nullptr;
  shaderModuleCreateInfo.flags = 0;
  shaderModuleCreateInfo.codeSize = size;
  shaderModuleCreateInfo.pCode = (uint32_t*) source;

  VkResult result = idevice->table.vkCreateShaderModule(
    idevice->logicalDevice,
    &shaderModuleCreateInfo,
    nullptr,
    &ishader->module
  );
  if (result != VK_SUCCESS) {
    iinstance->ishaderStore.free(shader->handle);
    CGPU_RETURN_ERROR("failed to create shader module");
  }

  if (!cgpuReflectShader((uint32_t*) source, size, &ishader->reflection))
  {
    idevice->table.vkDestroyShaderModule(
      idevice->logicalDevice,
      ishader->module,
      nullptr
    );
    iinstance->ishaderStore.free(shader->handle);
    CGPU_RETURN_ERROR("failed to reflect shader");
  }

  ishader->stageFlags = (VkShaderStageFlagBits) stageFlags;

  return true;
}

bool cgpuDestroyShader(CgpuDevice device, CgpuShader shader)
{
  CgpuIDevice* idevice;
  if (!cgpuResolveDevice(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIShader* ishader;
  if (!cgpuResolveShader(shader, &ishader)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  idevice->table.vkDestroyShaderModule(
    idevice->logicalDevice,
    ishader->module,
    nullptr
  );

  iinstance->ishaderStore.free(shader.handle);

  return true;
}

static bool cgpuCreateIBufferAligned(CgpuIDevice* idevice,
                                     CgpuBufferUsageFlags usage,
                                     CgpuMemoryPropertyFlags memoryProperties,
                                     uint64_t size,
                                     uint64_t alignment,
                                     CgpuIBuffer* ibuffer)
{
  VkBufferCreateInfo buffer_info;
  buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_info.pNext = nullptr;
  buffer_info.flags = 0;
  buffer_info.size = size;
  buffer_info.usage = (VkBufferUsageFlags) usage;
  buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  buffer_info.queueFamilyIndexCount = 0;
  buffer_info.pQueueFamilyIndices = nullptr;

  VmaAllocationCreateInfo vmaAllocCreateInfo = {};
  vmaAllocCreateInfo.requiredFlags = (VkMemoryPropertyFlags) memoryProperties;

  VkResult result;
  if (alignment > 0)
  {
    result = vmaCreateBufferWithAlignment(
      idevice->allocator,
      &buffer_info,
      &vmaAllocCreateInfo,
      alignment,
      &ibuffer->buffer,
      &ibuffer->allocation,
      nullptr
    );
  }
  else
  {
    result = vmaCreateBuffer(
      idevice->allocator,
      &buffer_info,
      &vmaAllocCreateInfo,
      &ibuffer->buffer,
      &ibuffer->allocation,
      nullptr
    );
  }

  if (result != VK_SUCCESS)
  {
    CGPU_RETURN_ERROR("failed to create buffer");
  }

  ibuffer->size = size;

  return true;
}

static bool cgpuCreateBufferAligned(CgpuDevice device,
                                    CgpuBufferUsageFlags usage,
                                    CgpuMemoryPropertyFlags memoryProperties,
                                    uint64_t size,
                                    uint64_t alignment,
                                    CgpuBuffer* buffer)
{
  CgpuIDevice* idevice;
  if (!cgpuResolveDevice(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  buffer->handle = iinstance->ibufferStore.allocate();

  CgpuIBuffer* ibuffer;
  if (!cgpuResolveBuffer(*buffer, &ibuffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  if (!cgpuCreateIBufferAligned(idevice, usage, memoryProperties, size, alignment, ibuffer))
  {
    iinstance->ibufferStore.free(buffer->handle);
    CGPU_RETURN_ERROR("failed to create buffer");
  }

  return true;
}

bool cgpuCreateBuffer(CgpuDevice device,
                      CgpuBufferUsageFlags usage,
                      CgpuMemoryPropertyFlags memoryProperties,
                      uint64_t size,
                      CgpuBuffer* buffer)
{
  uint64_t alignment = 0;

  return cgpuCreateBufferAligned(device, usage, memoryProperties, size, alignment, buffer);
}

static void cgpuDestroyIBuffer(CgpuIDevice* idevice, CgpuIBuffer* ibuffer)
{
  vmaDestroyBuffer(idevice->allocator, ibuffer->buffer, ibuffer->allocation);
}

bool cgpuDestroyBuffer(CgpuDevice device, CgpuBuffer buffer)
{
  CgpuIDevice* idevice;
  if (!cgpuResolveDevice(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIBuffer* ibuffer;
  if (!cgpuResolveBuffer(buffer, &ibuffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  cgpuDestroyIBuffer(idevice, ibuffer);

  iinstance->ibufferStore.free(buffer.handle);

  return true;
}

bool cgpuMapBuffer(CgpuDevice device, CgpuBuffer buffer, void** mappedMem)
{
  CgpuIDevice* idevice;
  if (!cgpuResolveDevice(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIBuffer* ibuffer;
  if (!cgpuResolveBuffer(buffer, &ibuffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  if (vmaMapMemory(idevice->allocator, ibuffer->allocation, mappedMem) != VK_SUCCESS) {
    CGPU_RETURN_ERROR("failed to map buffer memory");
  }
  return true;
}

bool cgpuUnmapBuffer(CgpuDevice device, CgpuBuffer buffer)
{
  CgpuIDevice* idevice;
  if (!cgpuResolveDevice(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIBuffer* ibuffer;
  if (!cgpuResolveBuffer(buffer, &ibuffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  vmaUnmapMemory(idevice->allocator, ibuffer->allocation);
  return true;
}

bool cgpuCreateImage(CgpuDevice device,
                     const CgpuImageDesc* imageDesc,
                     CgpuImage* image)
{
  CgpuIDevice* idevice;
  if (!cgpuResolveDevice(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  image->handle = iinstance->iimageStore.allocate();

  CgpuIImage* iimage;
  if (!cgpuResolveImage(*image, &iimage)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  VkImageTiling vkImageTiling = VK_IMAGE_TILING_OPTIMAL;
  if (!imageDesc->is3d && ((imageDesc->usage & CGPU_IMAGE_USAGE_FLAG_TRANSFER_SRC) | (imageDesc->usage & CGPU_IMAGE_USAGE_FLAG_TRANSFER_DST)))
  {
    vkImageTiling = VK_IMAGE_TILING_LINEAR;
  }

  VkImageCreateInfo imageCreateInfo;
  imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  imageCreateInfo.pNext = nullptr;
  imageCreateInfo.flags = 0;
  imageCreateInfo.imageType = imageDesc->is3d ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;
  imageCreateInfo.format = (VkFormat) imageDesc->format;
  imageCreateInfo.extent.width = imageDesc->width;
  imageCreateInfo.extent.height = imageDesc->height;
  imageCreateInfo.extent.depth = imageDesc->is3d ? imageDesc->depth : 1;
  imageCreateInfo.mipLevels = 1;
  imageCreateInfo.arrayLayers = 1;
  imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
  imageCreateInfo.tiling = vkImageTiling;
  imageCreateInfo.usage = (VkImageUsageFlags) imageDesc->usage;
  imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  imageCreateInfo.queueFamilyIndexCount = 0;
  imageCreateInfo.pQueueFamilyIndices = nullptr;
  imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

  VmaAllocationCreateInfo allocationCreateInfo = {};
  allocationCreateInfo.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

  VkResult result = vmaCreateImage(
    idevice->allocator,
    &imageCreateInfo,
    &allocationCreateInfo,
    &iimage->image,
    &iimage->allocation,
    nullptr
  );

  if (result != VK_SUCCESS) {
    iinstance->iimageStore.free(image->handle);
    CGPU_RETURN_ERROR("failed to create image");
  }

  VmaAllocationInfo allocationInfo;
  vmaGetAllocationInfo(idevice->allocator, iimage->allocation, &allocationInfo);

  iimage->size = allocationInfo.size;

  VkImageViewCreateInfo imageViewCreateInfo;
  imageViewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  imageViewCreateInfo.pNext = nullptr;
  imageViewCreateInfo.flags = 0;
  imageViewCreateInfo.image = iimage->image;
  imageViewCreateInfo.viewType = imageDesc->is3d ? VK_IMAGE_VIEW_TYPE_3D : VK_IMAGE_VIEW_TYPE_2D;
  imageViewCreateInfo.format = (VkFormat) imageDesc->format;
  imageViewCreateInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
  imageViewCreateInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
  imageViewCreateInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
  imageViewCreateInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
  imageViewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  imageViewCreateInfo.subresourceRange.baseMipLevel = 0;
  imageViewCreateInfo.subresourceRange.levelCount = 1;
  imageViewCreateInfo.subresourceRange.baseArrayLayer = 0;
  imageViewCreateInfo.subresourceRange.layerCount = 1;

  result = idevice->table.vkCreateImageView(
    idevice->logicalDevice,
    &imageViewCreateInfo,
    nullptr,
    &iimage->imageView
  );
  if (result != VK_SUCCESS)
  {
    iinstance->iimageStore.free(image->handle);
    vmaDestroyImage(idevice->allocator, iimage->image, iimage->allocation);
    CGPU_RETURN_ERROR("failed to create image view");
  }

  iimage->width = imageDesc->width;
  iimage->height = imageDesc->height;
  iimage->depth = imageDesc->is3d ? imageDesc->depth : 1;
  iimage->layout = imageCreateInfo.initialLayout;
  iimage->accessMask = 0;

  return true;
}

bool cgpuDestroyImage(CgpuDevice device, CgpuImage image)
{
  CgpuIDevice* idevice;
  if (!cgpuResolveDevice(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIImage* iimage;
  if (!cgpuResolveImage(image, &iimage)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  idevice->table.vkDestroyImageView(
    idevice->logicalDevice,
    iimage->imageView,
    nullptr
  );

  vmaDestroyImage(idevice->allocator, iimage->image, iimage->allocation);

  iinstance->iimageStore.free(image.handle);

  return true;
}

bool cgpuMapImage(CgpuDevice device, CgpuImage image, void** mappedMem)
{
  CgpuIDevice* idevice;
  if (!cgpuResolveDevice(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIImage* iimage;
  if (!cgpuResolveImage(image, &iimage)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  if (vmaMapMemory(idevice->allocator, iimage->allocation, mappedMem) != VK_SUCCESS) {
    CGPU_RETURN_ERROR("failed to map image memory");
  }
  return true;
}

bool cgpuUnmapImage(CgpuDevice device, CgpuImage image)
{
  CgpuIDevice* idevice;
  if (!cgpuResolveDevice(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIImage* iimage;
  if (!cgpuResolveImage(image, &iimage)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  vmaUnmapMemory(idevice->allocator, iimage->allocation);
  return true;
}

bool cgpuCreateSampler(CgpuDevice device,
                       CgpuSamplerAddressMode addressModeU,
                       CgpuSamplerAddressMode addressModeV,
                       CgpuSamplerAddressMode addressModeW,
                       CgpuSampler* sampler)
{
  CgpuIDevice* idevice;
  if (!cgpuResolveDevice(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  sampler->handle = iinstance->isamplerStore.allocate();

  CgpuISampler* isampler;
  if (!cgpuResolveSampler(*sampler, &isampler)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  // Emulate MDL's clip wrap mode if necessary; use optimal mode (according to ARM) if not.
  bool clampToBlack = (addressModeU == CGPU_SAMPLER_ADDRESS_MODE_CLAMP_TO_BLACK) ||
                      (addressModeV == CGPU_SAMPLER_ADDRESS_MODE_CLAMP_TO_BLACK) ||
                      (addressModeW == CGPU_SAMPLER_ADDRESS_MODE_CLAMP_TO_BLACK);

  VkSamplerCreateInfo samplerCreateInfo;
  samplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  samplerCreateInfo.pNext = nullptr;
  samplerCreateInfo.flags = 0;
  samplerCreateInfo.magFilter = VK_FILTER_LINEAR;
  samplerCreateInfo.minFilter = VK_FILTER_LINEAR;
  samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  samplerCreateInfo.addressModeU = cgpuTranslateAddressMode(addressModeU);
  samplerCreateInfo.addressModeV = cgpuTranslateAddressMode(addressModeV);
  samplerCreateInfo.addressModeW = cgpuTranslateAddressMode(addressModeW);
  samplerCreateInfo.mipLodBias = 0.0f;
  samplerCreateInfo.anisotropyEnable = VK_FALSE;
  samplerCreateInfo.maxAnisotropy = 1.0f;
  samplerCreateInfo.compareEnable = VK_FALSE;
  samplerCreateInfo.compareOp = VK_COMPARE_OP_NEVER;
  samplerCreateInfo.minLod = 0.0f;
  samplerCreateInfo.maxLod = VK_LOD_CLAMP_NONE;
  samplerCreateInfo.borderColor = clampToBlack ? VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK : VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
  samplerCreateInfo.unnormalizedCoordinates = VK_FALSE;

  VkResult result = idevice->table.vkCreateSampler(
    idevice->logicalDevice,
    &samplerCreateInfo,
    nullptr,
    &isampler->sampler
  );

  if (result != VK_SUCCESS) {
    iinstance->isamplerStore.free(sampler->handle);
    CGPU_RETURN_ERROR("failed to create sampler");
  }

  return true;
}

bool cgpuDestroySampler(CgpuDevice device, CgpuSampler sampler)
{
  CgpuIDevice* idevice;
  if (!cgpuResolveDevice(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuISampler* isampler;
  if (!cgpuResolveSampler(sampler, &isampler)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  idevice->table.vkDestroySampler(idevice->logicalDevice, isampler->sampler, nullptr);

  iinstance->isamplerStore.free(sampler.handle);

  return true;
}

static bool cgpuCreatePipelineLayout(CgpuIDevice* idevice, CgpuIPipeline* ipipeline, CgpuIShader* ishader, VkShaderStageFlags stageFlags)
{
  VkPushConstantRange pushConstRange;
  pushConstRange.stageFlags = stageFlags;
  pushConstRange.offset = 0;
  pushConstRange.size = ishader->reflection.pushConstantsSize;

  VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo;
  pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutCreateInfo.pNext = nullptr;
  pipelineLayoutCreateInfo.flags = 0;
  pipelineLayoutCreateInfo.setLayoutCount = 1;
  pipelineLayoutCreateInfo.pSetLayouts = &ipipeline->descriptorSetLayout;
  pipelineLayoutCreateInfo.pushConstantRangeCount = pushConstRange.size ? 1 : 0;
  pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstRange;

  return idevice->table.vkCreatePipelineLayout(idevice->logicalDevice,
                                               &pipelineLayoutCreateInfo,
                                               nullptr,
                                               &ipipeline->layout) == VK_SUCCESS;
}

static bool cgpuCreatePipelineDescriptors(CgpuIDevice* idevice, CgpuIPipeline* ipipeline, CgpuIShader* ishader, VkShaderStageFlags stageFlags)
{
  const CgpuShaderReflection* shaderReflection = &ishader->reflection;

  for (uint32_t i = 0; i < shaderReflection->bindings.size(); i++)
  {
    const CgpuShaderReflectionBinding* binding_reflection = &shaderReflection->bindings[i];

    VkDescriptorSetLayoutBinding layout_binding;
    layout_binding.binding = binding_reflection->binding;
    layout_binding.descriptorType = (VkDescriptorType) binding_reflection->descriptorType;
    layout_binding.descriptorCount = binding_reflection->count;
    layout_binding.stageFlags = stageFlags;
    layout_binding.pImmutableSamplers = nullptr;

    ipipeline->descriptorSetLayoutBindings.push_back(layout_binding);
  }

  VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo;
  descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  descriptorSetLayoutCreateInfo.pNext = nullptr;
  descriptorSetLayoutCreateInfo.flags = 0;
  descriptorSetLayoutCreateInfo.bindingCount = ipipeline->descriptorSetLayoutBindings.size();
  descriptorSetLayoutCreateInfo.pBindings = ipipeline->descriptorSetLayoutBindings.data();

  VkResult result = idevice->table.vkCreateDescriptorSetLayout(
    idevice->logicalDevice,
    &descriptorSetLayoutCreateInfo,
    nullptr,
    &ipipeline->descriptorSetLayout
  );

  if (result != VK_SUCCESS) {
    CGPU_RETURN_ERROR("failed to create descriptor set layout");
  }

  uint32_t bufferCount = 0;
  uint32_t storageImageCount = 0;
  uint32_t sampledImageCount = 0;
  uint32_t samplerCount = 0;
  uint32_t asCount = 0;

  for (uint32_t i = 0; i < shaderReflection->bindings.size(); i++)
  {
    const CgpuShaderReflectionBinding* binding = &shaderReflection->bindings[i];

    switch (binding->descriptorType)
    {
    case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: bufferCount += binding->count; break;
    case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE: storageImageCount += binding->count; break;
    case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE: sampledImageCount += binding->count; break;
    case VK_DESCRIPTOR_TYPE_SAMPLER: samplerCount += binding->count; break;
    case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR: asCount += binding->count; break;
    default: {
      idevice->table.vkDestroyDescriptorSetLayout(
        idevice->logicalDevice,
        ipipeline->descriptorSetLayout,
        nullptr
      );
      CGPU_RETURN_ERROR("invalid descriptor type");
    }
    }
  }

  uint32_t poolSizeCount = 0;
  VkDescriptorPoolSize poolSizes[16];

  if (bufferCount > 0)
  {
    poolSizes[poolSizeCount].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[poolSizeCount].descriptorCount = bufferCount;
    poolSizeCount++;
  }
  if (storageImageCount > 0)
  {
    poolSizes[poolSizeCount].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes[poolSizeCount].descriptorCount = storageImageCount;
    poolSizeCount++;
  }
  if (sampledImageCount > 0)
  {
    poolSizes[poolSizeCount].type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    poolSizes[poolSizeCount].descriptorCount = sampledImageCount;
    poolSizeCount++;
  }
  if (samplerCount > 0)
  {
    poolSizes[poolSizeCount].type = VK_DESCRIPTOR_TYPE_SAMPLER;
    poolSizes[poolSizeCount].descriptorCount = samplerCount;
    poolSizeCount++;
  }
  if (asCount > 0)
  {
    poolSizes[poolSizeCount].type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    poolSizes[poolSizeCount].descriptorCount = asCount;
    poolSizeCount++;
  }

  VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
  descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  descriptorPoolCreateInfo.pNext = nullptr;
  descriptorPoolCreateInfo.flags = 0;
  descriptorPoolCreateInfo.maxSets = 1;
  descriptorPoolCreateInfo.poolSizeCount = poolSizeCount;
  descriptorPoolCreateInfo.pPoolSizes = poolSizes;

  result = idevice->table.vkCreateDescriptorPool(
    idevice->logicalDevice,
    &descriptorPoolCreateInfo,
    nullptr,
    &ipipeline->descriptorPool
  );
  if (result != VK_SUCCESS) {
    idevice->table.vkDestroyDescriptorSetLayout(
      idevice->logicalDevice,
      ipipeline->descriptorSetLayout,
      nullptr
    );
    CGPU_RETURN_ERROR("failed to create descriptor pool");
  }

  VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {};
  descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  descriptorSetAllocateInfo.pNext = nullptr;
  descriptorSetAllocateInfo.descriptorPool = ipipeline->descriptorPool;
  descriptorSetAllocateInfo.descriptorSetCount = 1;
  descriptorSetAllocateInfo.pSetLayouts = &ipipeline->descriptorSetLayout;

  result = idevice->table.vkAllocateDescriptorSets(
    idevice->logicalDevice,
    &descriptorSetAllocateInfo,
    &ipipeline->descriptorSet
  );
  if (result != VK_SUCCESS) {
    idevice->table.vkDestroyDescriptorPool(
      idevice->logicalDevice,
      ipipeline->descriptorPool,
      nullptr
    );
    idevice->table.vkDestroyDescriptorSetLayout(
      idevice->logicalDevice,
      ipipeline->descriptorSetLayout,
      nullptr
    );
    CGPU_RETURN_ERROR("failed to allocate descriptor set");
  }

  return true;
}

bool cgpuCreateComputePipeline(CgpuDevice device,
                               CgpuShader shader,
                               CgpuPipeline* pipeline)
{
  CgpuIDevice* idevice;
  if (!cgpuResolveDevice(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIShader* ishader;
  if (!cgpuResolveShader(shader, &ishader)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  pipeline->handle = iinstance->ipipelineStore.allocate();

  CgpuIPipeline* ipipeline;
  if (!cgpuResolvePipeline(*pipeline, &ipipeline)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  if (!cgpuCreatePipelineDescriptors(idevice, ipipeline, ishader, VK_SHADER_STAGE_COMPUTE_BIT))
  {
    iinstance->ipipelineStore.free(pipeline->handle);
    CGPU_RETURN_ERROR("failed to create descriptor set layout");
  }

  if (!cgpuCreatePipelineLayout(idevice, ipipeline, ishader, VK_SHADER_STAGE_COMPUTE_BIT))
  {
    iinstance->ipipelineStore.free(pipeline->handle);
    idevice->table.vkDestroyDescriptorSetLayout(
      idevice->logicalDevice,
      ipipeline->descriptorSetLayout,
      nullptr
    );
    idevice->table.vkDestroyDescriptorPool(
      idevice->logicalDevice,
      ipipeline->descriptorPool,
      nullptr
    );
    CGPU_RETURN_ERROR("failed to create pipeline layout");
  }

  VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo = {};
  pipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  pipelineShaderStageCreateInfo.pNext = nullptr;
  pipelineShaderStageCreateInfo.flags = 0;
  pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  pipelineShaderStageCreateInfo.module = ishader->module;
  pipelineShaderStageCreateInfo.pName = "main";
  pipelineShaderStageCreateInfo.pSpecializationInfo = nullptr;

  VkComputePipelineCreateInfo pipelineCreateInfo = {};
  pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipelineCreateInfo.pNext = nullptr;
  pipelineCreateInfo.flags = 0;
  pipelineCreateInfo.stage = pipelineShaderStageCreateInfo;
  pipelineCreateInfo.layout = ipipeline->layout;
  pipelineCreateInfo.basePipelineHandle = VK_NULL_HANDLE;
  pipelineCreateInfo.basePipelineIndex = 0;

  VkResult result = idevice->table.vkCreateComputePipelines(
    idevice->logicalDevice,
    VK_NULL_HANDLE,
    1,
    &pipelineCreateInfo,
    nullptr,
    &ipipeline->pipeline
  );

  if (result != VK_SUCCESS) {
    iinstance->ipipelineStore.free(pipeline->handle);
    idevice->table.vkDestroyPipelineLayout(
      idevice->logicalDevice,
      ipipeline->layout,
      nullptr
    );
    idevice->table.vkDestroyDescriptorSetLayout(
      idevice->logicalDevice,
      ipipeline->descriptorSetLayout,
      nullptr
    );
    idevice->table.vkDestroyDescriptorPool(
      idevice->logicalDevice,
      ipipeline->descriptorPool,
      nullptr
    );
    CGPU_RETURN_ERROR("failed to create compute pipeline");
  }

  ipipeline->bindPoint = VK_PIPELINE_BIND_POINT_COMPUTE;

  return true;
}

static VkDeviceAddress cgpuGetBufferDeviceAddress(CgpuIDevice* idevice, CgpuIBuffer* ibuffer)
{
  VkBufferDeviceAddressInfoKHR addressInfo = {};
  addressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
  addressInfo.pNext = nullptr;
  addressInfo.buffer = ibuffer->buffer;
  return idevice->table.vkGetBufferDeviceAddressKHR(idevice->logicalDevice, &addressInfo);
}

static uint32_t cgpuAlignSize(uint32_t size, uint32_t alignment)
{
    return (size + (alignment - 1)) & ~(alignment - 1);
}

static bool cgpuCreateRtPipelineSbt(CgpuIDevice* idevice,
                                    CgpuIPipeline* ipipeline,
                                    uint32_t groupCount,
                                    uint32_t missShaderCount,
                                    uint32_t hitGroupCount)
{
  uint32_t handleSize = idevice->properties.shaderGroupHandleSize;
  uint32_t alignedHandleSize = cgpuAlignSize(handleSize, idevice->properties.shaderGroupHandleAlignment);

  ipipeline->sbtRgen.stride = cgpuAlignSize(alignedHandleSize, idevice->properties.shaderGroupBaseAlignment);
  ipipeline->sbtRgen.size = ipipeline->sbtRgen.stride; // Special raygen condition: size must be equal to stride
  ipipeline->sbtMiss.stride = alignedHandleSize;
  ipipeline->sbtMiss.size = cgpuAlignSize(missShaderCount * alignedHandleSize, idevice->properties.shaderGroupBaseAlignment);
  ipipeline->sbtHit.stride = alignedHandleSize;
  ipipeline->sbtHit.size = cgpuAlignSize(hitGroupCount * alignedHandleSize, idevice->properties.shaderGroupBaseAlignment);

  uint32_t firstGroup = 0;
  uint32_t dataSize = handleSize * groupCount;

  GbSmallVector<uint8_t, 64> handleData(dataSize);
  if (idevice->table.vkGetRayTracingShaderGroupHandlesKHR(idevice->logicalDevice, ipipeline->pipeline, firstGroup, groupCount, handleData.size(), handleData.data()) != VK_SUCCESS)
  {
    CGPU_RETURN_ERROR("failed to create sbt handles");
  }

  VkDeviceSize sbtSize = ipipeline->sbtRgen.size + ipipeline->sbtMiss.size + ipipeline->sbtHit.size;
  CgpuBufferUsageFlags bufferUsageFlags = CGPU_BUFFER_USAGE_FLAG_TRANSFER_SRC | CGPU_BUFFER_USAGE_FLAG_SHADER_DEVICE_ADDRESS | CGPU_BUFFER_USAGE_FLAG_SHADER_BINDING_TABLE_BIT_KHR;
  CgpuMemoryPropertyFlags bufferMemPropFlags = CGPU_MEMORY_PROPERTY_FLAG_HOST_VISIBLE | CGPU_MEMORY_PROPERTY_FLAG_HOST_CACHED;

  if (!cgpuCreateIBufferAligned(idevice, bufferUsageFlags, bufferMemPropFlags, sbtSize, 0, &ipipeline->sbt))
  {
    CGPU_RETURN_ERROR("failed to create sbt buffer");
  }

  VkDeviceAddress sbtDeviceAddress = cgpuGetBufferDeviceAddress(idevice, &ipipeline->sbt);
  ipipeline->sbtRgen.deviceAddress = sbtDeviceAddress;
  ipipeline->sbtMiss.deviceAddress = sbtDeviceAddress + ipipeline->sbtRgen.size;
  ipipeline->sbtHit.deviceAddress = sbtDeviceAddress + ipipeline->sbtRgen.size + ipipeline->sbtMiss.size;

  uint8_t* sbtMem;
  if (vmaMapMemory(idevice->allocator, ipipeline->sbt.allocation, (void**)&sbtMem) != VK_SUCCESS) {
    CGPU_RETURN_ERROR("failed to map buffer memory");
  }

  uint32_t handleCount = 0;
  uint8_t* sbtMemRgen = &sbtMem[0];
  uint8_t* sbtMemMiss = &sbtMem[ipipeline->sbtRgen.size];
  uint8_t* sbtMemHit = &sbtMem[ipipeline->sbtRgen.size + ipipeline->sbtMiss.size];

  // Rgen
  sbtMem = sbtMemRgen;
  memcpy(sbtMem, &handleData[handleSize * (handleCount++)], handleSize);
  // Miss
  sbtMem = sbtMemMiss;
  for (uint32_t i = 0; i < missShaderCount; i++)
  {
    memcpy(sbtMem, &handleData[handleSize * (handleCount++)], handleSize);
    sbtMem += ipipeline->sbtMiss.stride;
  }
  // Hit
  sbtMem = sbtMemHit;
  for (uint32_t i = 0; i < hitGroupCount; i++)
  {
    memcpy(sbtMem, &handleData[handleSize * (handleCount++)], handleSize);
    sbtMem += ipipeline->sbtHit.stride;
  }

  vmaUnmapMemory(idevice->allocator, ipipeline->sbt.allocation);
  return true;
}

bool cgpuCreateRtPipeline(CgpuDevice device,
                          const CgpuRtPipelineDesc* desc,
                          CgpuPipeline* pipeline)
{
  CgpuIDevice* idevice;
  if (!cgpuResolveDevice(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  pipeline->handle = iinstance->ipipelineStore.allocate();

  CgpuIPipeline* ipipeline;
  if (!cgpuResolvePipeline(*pipeline, &ipipeline)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  // Zero-init for cleanup routine.
  memset(ipipeline, 0, sizeof(CgpuIPipeline));

  // In a ray tracing pipeline, all shaders are expected to have the same descriptor set layouts. Here, we
  // construct the descriptor set layouts and the pipeline layout from only the ray generation shader.
  CgpuIShader* irgenShader;
  if (!cgpuResolveShader(desc->rgenShader, &irgenShader)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  // Set up stages
  GbSmallVector<VkPipelineShaderStageCreateInfo, 128> stages;
  VkShaderStageFlags shaderStageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

  auto pushStage = [&stages](VkShaderStageFlagBits stage, VkShaderModule module) {
    VkPipelineShaderStageCreateInfo pipeline_shader_stage_create_info = {};
    pipeline_shader_stage_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_shader_stage_create_info.pNext = nullptr;
    pipeline_shader_stage_create_info.flags = 0;
    pipeline_shader_stage_create_info.stage = stage;
    pipeline_shader_stage_create_info.module = module;
    pipeline_shader_stage_create_info.pName = "main";
    pipeline_shader_stage_create_info.pSpecializationInfo = nullptr;
    stages.push_back(pipeline_shader_stage_create_info);
  };

  // Ray gen
  pushStage(VK_SHADER_STAGE_RAYGEN_BIT_KHR, irgenShader->module);

  // Miss
  if (desc->missShaderCount > 0)
  {
    shaderStageFlags |= VK_SHADER_STAGE_MISS_BIT_KHR;
  }
  for (uint32_t i = 0; i < desc->missShaderCount; i++)
  {
    CgpuIShader* imiss_shader;
    if (!cgpuResolveShader(desc->missShaders[i], &imiss_shader)) {
      CGPU_RETURN_ERROR_INVALID_HANDLE;
    }
    assert(imiss_shader->module);

    pushStage(VK_SHADER_STAGE_MISS_BIT_KHR, imiss_shader->module);
  }

  // Hit
  for (uint32_t i = 0; i < desc->hitGroupCount; i++)
  {
    const CgpuRtHitGroup* hitGroup = &desc->hitGroups[i];

    // Closest hit (optional)
    if (hitGroup->closestHitShader.handle)
    {
      CgpuIShader* iclosestHitShader;
      if (!cgpuResolveShader(hitGroup->closestHitShader, &iclosestHitShader)) {
        CGPU_RETURN_ERROR_INVALID_HANDLE;
      }
      assert(iclosestHitShader->stageFlags == VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);

      pushStage(iclosestHitShader->stageFlags, iclosestHitShader->module);
      shaderStageFlags |= iclosestHitShader->stageFlags;
    }

    // Any hit (optional)
    if (hitGroup->anyHitShader.handle)
    {
      CgpuIShader* ianyHitShader;
      if (!cgpuResolveShader(hitGroup->anyHitShader, &ianyHitShader)) {
        CGPU_RETURN_ERROR_INVALID_HANDLE;
      }
      assert(ianyHitShader->stageFlags == VK_SHADER_STAGE_ANY_HIT_BIT_KHR);

      pushStage(ianyHitShader->stageFlags, ianyHitShader->module);
      shaderStageFlags |= ianyHitShader->stageFlags;
    }
  }

  // Set up groups
  GbSmallVector<VkRayTracingShaderGroupCreateInfoKHR, 128> groups;
  groups.resize(1/*rgen*/ + desc->missShaderCount + desc->hitGroupCount);

  for (uint32_t i = 0; i < groups.size(); i++)
  {
    VkRayTracingShaderGroupCreateInfoKHR createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    createInfo.pNext = nullptr;
    createInfo.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    createInfo.generalShader = i;
    createInfo.closestHitShader = VK_SHADER_UNUSED_KHR;
    createInfo.anyHitShader = VK_SHADER_UNUSED_KHR;
    createInfo.intersectionShader = VK_SHADER_UNUSED_KHR;
    createInfo.pShaderGroupCaptureReplayHandle = nullptr;
    groups[i] = createInfo;
  }

  bool anyNullClosestHitShader = false;
  bool anyNullAnyHitShader = false;

  uint32_t hitStageAndGroupOffset = 1/*rgen*/ + desc->missShaderCount;
  uint32_t hitShaderStageIndex = hitStageAndGroupOffset;
  for (uint32_t i = 0; i < desc->hitGroupCount; i++)
  {
    const CgpuRtHitGroup* hit_group = &desc->hitGroups[i];

    uint32_t groupIndex = hitStageAndGroupOffset + i;
    groups[groupIndex].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    groups[groupIndex].generalShader = VK_SHADER_UNUSED_KHR;

    if (hit_group->closestHitShader.handle)
    {
      groups[groupIndex].closestHitShader = (hitShaderStageIndex++);
    }
    else
    {
      anyNullClosestHitShader |= true;
    }

    if (hit_group->anyHitShader.handle)
    {
      groups[groupIndex].anyHitShader = (hitShaderStageIndex++);
    }
    else
    {
      anyNullAnyHitShader |= true;
    }
  }

  // Create descriptor and pipeline layout.
  if (!cgpuCreatePipelineDescriptors(idevice, ipipeline, irgenShader, shaderStageFlags))
  {
    goto cleanup_fail;
  }
  if (!cgpuCreatePipelineLayout(idevice, ipipeline, irgenShader, shaderStageFlags))
  {
    goto cleanup_fail;
  }

  // Create pipeline.
  {
    uint32_t groupCount = hitStageAndGroupOffset + desc->hitGroupCount;

    VkPipelineCreateFlags flags = 0;
    if (!anyNullClosestHitShader)
    {
      flags |= VK_PIPELINE_CREATE_RAY_TRACING_NO_NULL_CLOSEST_HIT_SHADERS_BIT_KHR;
    }
    if (!anyNullAnyHitShader)
    {
      flags |= VK_PIPELINE_CREATE_RAY_TRACING_NO_NULL_ANY_HIT_SHADERS_BIT_KHR;
    }

    VkRayTracingPipelineCreateInfoKHR rt_pipeline_create_info = {};
    rt_pipeline_create_info.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
    rt_pipeline_create_info.pNext = nullptr;
    rt_pipeline_create_info.flags = flags;
    rt_pipeline_create_info.stageCount = stages.size();
    rt_pipeline_create_info.pStages = stages.data();
    rt_pipeline_create_info.groupCount = groups.size();
    rt_pipeline_create_info.pGroups = groups.data();
    rt_pipeline_create_info.maxPipelineRayRecursionDepth = 1;
    rt_pipeline_create_info.pLibraryInfo = nullptr;
    rt_pipeline_create_info.pLibraryInterface = nullptr;
    rt_pipeline_create_info.pDynamicState = nullptr;
    rt_pipeline_create_info.layout = ipipeline->layout;
    rt_pipeline_create_info.basePipelineHandle = VK_NULL_HANDLE;
    rt_pipeline_create_info.basePipelineIndex = 0;

    if (idevice->table.vkCreateRayTracingPipelinesKHR(idevice->logicalDevice,
                                                      VK_NULL_HANDLE,
                                                      VK_NULL_HANDLE,
                                                      1,
                                                      &rt_pipeline_create_info,
                                                      nullptr,
                                                      &ipipeline->pipeline) != VK_SUCCESS)
    {
      goto cleanup_fail;
    }

    ipipeline->bindPoint = VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR;

    // Create the SBT.
    if (!cgpuCreateRtPipelineSbt(idevice, ipipeline, groupCount, desc->missShaderCount, desc->hitGroupCount))
    {
      goto cleanup_fail;
    }

    return true;
  }

cleanup_fail:
  idevice->table.vkDestroyPipelineLayout(idevice->logicalDevice, ipipeline->layout, nullptr);
  idevice->table.vkDestroyDescriptorSetLayout(idevice->logicalDevice, ipipeline->descriptorSetLayout, nullptr);
  idevice->table.vkDestroyDescriptorPool(idevice->logicalDevice, ipipeline->descriptorPool, nullptr);
  iinstance->ipipelineStore.free(pipeline->handle);

  CGPU_RETURN_ERROR("failed to create rt pipeline");
}

bool cgpuDestroyPipeline(CgpuDevice device, CgpuPipeline pipeline)
{
  CgpuIDevice* idevice;
  if (!cgpuResolveDevice(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIPipeline* ipipeline;
  if (!cgpuResolvePipeline(pipeline, &ipipeline)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  if (ipipeline->bindPoint == VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR)
  {
    cgpuDestroyIBuffer(idevice, &ipipeline->sbt);
  }

  idevice->table.vkDestroyDescriptorPool(
    idevice->logicalDevice,
    ipipeline->descriptorPool,
    nullptr
  );
  idevice->table.vkDestroyPipeline(
    idevice->logicalDevice,
    ipipeline->pipeline,
    nullptr
  );
  idevice->table.vkDestroyPipelineLayout(
    idevice->logicalDevice,
    ipipeline->layout,
    nullptr
  );
  idevice->table.vkDestroyDescriptorSetLayout(
    idevice->logicalDevice,
    ipipeline->descriptorSetLayout,
    nullptr
  );

  iinstance->ipipelineStore.free(pipeline.handle);

  return true;
}

static bool cgpuCreateTopOrBottomAs(CgpuDevice device,
                                    VkAccelerationStructureTypeKHR asType,
                                    VkAccelerationStructureGeometryKHR* asGeom,
                                    uint32_t primitiveCount,
                                    CgpuIBuffer* iasBuffer,
                                    VkAccelerationStructureKHR* as)
{
  CgpuIDevice* idevice;
  cgpuResolveDevice(device, &idevice);

  // Get AS size
  VkAccelerationStructureBuildGeometryInfoKHR asBuildGeomInfo = {};
  asBuildGeomInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
  asBuildGeomInfo.pNext = nullptr;
  asBuildGeomInfo.type = asType;
  asBuildGeomInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  asBuildGeomInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
  asBuildGeomInfo.srcAccelerationStructure = VK_NULL_HANDLE;
  asBuildGeomInfo.dstAccelerationStructure = VK_NULL_HANDLE; // set in second round
  asBuildGeomInfo.geometryCount = 1;
  asBuildGeomInfo.pGeometries = asGeom;
  asBuildGeomInfo.ppGeometries = nullptr;
  asBuildGeomInfo.scratchData.hostAddress = nullptr;
  asBuildGeomInfo.scratchData.deviceAddress = 0; // set in second round

  VkAccelerationStructureBuildSizesInfoKHR asBuildSizesInfo = {};
  asBuildSizesInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
  asBuildSizesInfo.pNext = nullptr;
  asBuildSizesInfo.accelerationStructureSize = 0; // output
  asBuildSizesInfo.updateScratchSize = 0; // output
  asBuildSizesInfo.buildScratchSize = 0; // output

  idevice->table.vkGetAccelerationStructureBuildSizesKHR(idevice->logicalDevice,
                                                         VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                                         &asBuildGeomInfo,
                                                         &primitiveCount,
                                                         &asBuildSizesInfo);

  // Create AS buffer & AS object
  if (!cgpuCreateIBufferAligned(idevice,
                                CGPU_BUFFER_USAGE_FLAG_SHADER_DEVICE_ADDRESS | CGPU_BUFFER_USAGE_FLAG_ACCELERATION_STRUCTURE_STORAGE,
                                CGPU_MEMORY_PROPERTY_FLAG_DEVICE_LOCAL,
                                asBuildSizesInfo.accelerationStructureSize, 0,
                                iasBuffer))
  {
    CGPU_RETURN_ERROR("failed to create AS buffer");
  }

  VkAccelerationStructureCreateInfoKHR asCreateInfo = {};
  asCreateInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
  asCreateInfo.pNext = nullptr;
  asCreateInfo.createFlags = 0;
  asCreateInfo.buffer = iasBuffer->buffer;
  asCreateInfo.offset = 0;
  asCreateInfo.size = asBuildSizesInfo.accelerationStructureSize;
  asCreateInfo.type = asType;
  asCreateInfo.deviceAddress = 0; // used for capture-replay feature

  if (idevice->table.vkCreateAccelerationStructureKHR(idevice->logicalDevice, &asCreateInfo, nullptr, as) != VK_SUCCESS)
  {
    cgpuDestroyIBuffer(idevice, iasBuffer);
    CGPU_RETURN_ERROR("failed to create Vulkan AS object");
  }

  // Set up device-local scratch buffer
  CgpuIBuffer iscratchBuffer;
  if (!cgpuCreateIBufferAligned(idevice,
                                CGPU_BUFFER_USAGE_FLAG_STORAGE_BUFFER | CGPU_BUFFER_USAGE_FLAG_SHADER_DEVICE_ADDRESS,
                                CGPU_MEMORY_PROPERTY_FLAG_DEVICE_LOCAL,
                                asBuildSizesInfo.buildScratchSize,
                                idevice->properties.minAccelerationStructureScratchOffsetAlignment,
                                &iscratchBuffer))
  {
    cgpuDestroyIBuffer(idevice, iasBuffer);
    idevice->table.vkDestroyAccelerationStructureKHR(idevice->logicalDevice, *as, nullptr);
    CGPU_RETURN_ERROR("failed to create AS scratch buffer");
  }

  asBuildGeomInfo.dstAccelerationStructure = *as;
  asBuildGeomInfo.scratchData.hostAddress = 0;
  asBuildGeomInfo.scratchData.deviceAddress = cgpuGetBufferDeviceAddress(idevice, &iscratchBuffer);

  VkAccelerationStructureBuildRangeInfoKHR asBuildRangeInfo = {};
  asBuildRangeInfo.primitiveCount = primitiveCount;
  asBuildRangeInfo.primitiveOffset = 0;
  asBuildRangeInfo.firstVertex = 0;
  asBuildRangeInfo.transformOffset = 0;

  const VkAccelerationStructureBuildRangeInfoKHR* as_build_range_info_ptr = &asBuildRangeInfo;

  CgpuCommandBuffer commandBuffer;
  if (!cgpuCreateCommandBuffer(device, &commandBuffer))
  {
    cgpuDestroyIBuffer(idevice, iasBuffer);
    cgpuDestroyIBuffer(idevice, &iscratchBuffer);
    idevice->table.vkDestroyAccelerationStructureKHR(idevice->logicalDevice, *as, nullptr);
    CGPU_RETURN_ERROR("failed to create AS build command buffer");
  }

  CgpuICommandBuffer* icommandBuffer;
  cgpuResolveCommandBuffer(commandBuffer, &icommandBuffer);

  // Build AS on device
  cgpuBeginCommandBuffer(commandBuffer);
  idevice->table.vkCmdBuildAccelerationStructuresKHR(icommandBuffer->commandBuffer, 1, &asBuildGeomInfo, &as_build_range_info_ptr);
  cgpuEndCommandBuffer(commandBuffer);

  CgpuFence fence;
  if (!cgpuCreateFence(device, &fence))
  {
    cgpuDestroyIBuffer(idevice, iasBuffer);
    cgpuDestroyIBuffer(idevice, &iscratchBuffer);
    idevice->table.vkDestroyAccelerationStructureKHR(idevice->logicalDevice, *as, nullptr);
    CGPU_RETURN_ERROR("failed to create AS build fence");
  }
  cgpuResetFence(device, fence);
  cgpuSubmitCommandBuffer(device, commandBuffer, fence);
  cgpuWaitForFence(device, fence);

  // Dispose resources
  cgpuDestroyFence(device, fence);
  cgpuDestroyCommandBuffer(device, commandBuffer);
  cgpuDestroyIBuffer(idevice, &iscratchBuffer);

  return true;
}


bool cgpuCreateBlas(CgpuDevice device,
                    uint32_t vertexCount,
                    const CgpuVertex* vertices,
                    uint32_t indexCount,
                    const uint32_t* indices,
                    bool isOpaque,
                    CgpuBlas* blas)
{
  CgpuIDevice* idevice;
  if (!cgpuResolveDevice(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  blas->handle = iinstance->iblasStore.allocate();

  CgpuIBlas* iblas;
  if (!cgpuResolveBlas(*blas, &iblas)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  if ((indexCount % 3) != 0) {
    CGPU_RETURN_ERROR("BLAS indices do not represent triangles");
  }

  // Create index buffer & copy data into it
  uint64_t indexBufferSize = indexCount * sizeof(uint32_t);
  if (!cgpuCreateIBufferAligned(idevice,
                                CGPU_BUFFER_USAGE_FLAG_SHADER_DEVICE_ADDRESS | CGPU_BUFFER_USAGE_FLAG_ACCELERATION_STRUCTURE_BUILD_INPUT,
                                CGPU_MEMORY_PROPERTY_FLAG_HOST_VISIBLE | CGPU_MEMORY_PROPERTY_FLAG_HOST_COHERENT,
                                indexBufferSize, 0,
                                &iblas->indices))
  {
    CGPU_RETURN_ERROR("failed to create BLAS index buffer");
  }

  {
    void* mappedMem;
    if (vmaMapMemory(idevice->allocator, iblas->indices.allocation, (void**) &mappedMem) != VK_SUCCESS) {
      cgpuDestroyIBuffer(idevice, &iblas->indices);
      CGPU_RETURN_ERROR("failed to map buffer memory");
    }
    memcpy(mappedMem, indices, indexBufferSize);
    vmaUnmapMemory(idevice->allocator, iblas->indices.allocation);
  }

  // Create vertex buffer & copy data into it
  uint64_t vertexBufferSize = vertexCount * sizeof(CgpuVertex);
  if (!cgpuCreateIBufferAligned(idevice,
                                CGPU_BUFFER_USAGE_FLAG_SHADER_DEVICE_ADDRESS | CGPU_BUFFER_USAGE_FLAG_ACCELERATION_STRUCTURE_BUILD_INPUT,
                                CGPU_MEMORY_PROPERTY_FLAG_HOST_VISIBLE | CGPU_MEMORY_PROPERTY_FLAG_HOST_COHERENT,
                                vertexBufferSize, 0,
                                &iblas->vertices))
  {
    cgpuDestroyIBuffer(idevice, &iblas->indices);
    CGPU_RETURN_ERROR("failed to create BLAS vertex buffer");
  }

  {
    void* mappedMem;
    if (vmaMapMemory(idevice->allocator, iblas->vertices.allocation, (void**)&mappedMem) != VK_SUCCESS)
    {
      cgpuDestroyIBuffer(idevice, &iblas->indices);
      cgpuDestroyIBuffer(idevice, &iblas->vertices);
      CGPU_RETURN_ERROR("failed to map buffer memory");
    }
    memcpy(mappedMem, vertices, vertexBufferSize);
    vmaUnmapMemory(idevice->allocator, iblas->vertices.allocation);
  }

  // Create BLAS
  VkAccelerationStructureGeometryTrianglesDataKHR asTriangleData = {};
  asTriangleData.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
  asTriangleData.pNext = nullptr;
  asTriangleData.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
  asTriangleData.vertexData.hostAddress = nullptr;
  asTriangleData.vertexData.deviceAddress = cgpuGetBufferDeviceAddress(idevice, &iblas->vertices);
  asTriangleData.vertexStride = sizeof(CgpuVertex);
  asTriangleData.maxVertex = vertexCount;
  asTriangleData.indexType = VK_INDEX_TYPE_UINT32;
  asTriangleData.indexData.hostAddress = nullptr;
  asTriangleData.indexData.deviceAddress = cgpuGetBufferDeviceAddress(idevice, &iblas->indices);
  asTriangleData.transformData.hostAddress = nullptr;
  asTriangleData.transformData.deviceAddress = 0; // optional

  VkAccelerationStructureGeometryDataKHR asGeomData = {};
  asGeomData.triangles = asTriangleData;

  VkAccelerationStructureGeometryKHR asGeom = {};
  asGeom.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
  asGeom.pNext = nullptr;
  asGeom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
  asGeom.geometry = asGeomData;
  asGeom.flags = isOpaque ? VK_GEOMETRY_OPAQUE_BIT_KHR : 0;

  uint32_t triangleCount = indexCount / 3;
  if (!cgpuCreateTopOrBottomAs(device, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, &asGeom, triangleCount, &iblas->buffer, &iblas->as))
  {
    cgpuDestroyIBuffer(idevice, &iblas->indices);
    cgpuDestroyIBuffer(idevice, &iblas->vertices);
    CGPU_RETURN_ERROR("failed to build BLAS");
  }

  VkAccelerationStructureDeviceAddressInfoKHR asAddressInfo = {};
  asAddressInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
  asAddressInfo.pNext = nullptr;
  asAddressInfo.accelerationStructure = iblas->as;
  iblas->address = idevice->table.vkGetAccelerationStructureDeviceAddressKHR(idevice->logicalDevice, &asAddressInfo);

  iblas->isOpaque = isOpaque;

  return true;
}

bool cgpuCreateTlas(CgpuDevice device,
                    uint32_t instanceCount,
                    const CgpuBlasInstance* instances,
                    CgpuTlas* tlas)
{
  CgpuIDevice* idevice;
  if (!cgpuResolveDevice(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  tlas->handle = iinstance->itlasStore.allocate();

  CgpuITlas* itlas;
  if (!cgpuResolveTlas(*tlas, &itlas)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  // Create instance buffer & copy into it
  if (!cgpuCreateIBufferAligned(idevice,
                                CGPU_BUFFER_USAGE_FLAG_SHADER_DEVICE_ADDRESS | CGPU_BUFFER_USAGE_FLAG_ACCELERATION_STRUCTURE_BUILD_INPUT,
                                CGPU_MEMORY_PROPERTY_FLAG_HOST_VISIBLE | CGPU_MEMORY_PROPERTY_FLAG_HOST_COHERENT,
                                instanceCount * sizeof(VkAccelerationStructureInstanceKHR), 0,
                                &itlas->instances))
  {
    CGPU_RETURN_ERROR("failed to create TLAS instances buffer");
  }

  bool areAllBlasOpaque = true;
  {
    uint8_t* mapped_mem;
    if (vmaMapMemory(idevice->allocator, itlas->instances.allocation, (void**) &mapped_mem) != VK_SUCCESS)
    {
      cgpuDestroyIBuffer(idevice, &itlas->instances);
      CGPU_RETURN_ERROR("failed to map buffer memory");
    }

    for (uint32_t i = 0; i < instanceCount; i++)
    {
      CgpuIBlas* iblas;
      if (!cgpuResolveBlas(instances[i].as, &iblas)) {
        cgpuDestroyIBuffer(idevice, &itlas->instances);
        CGPU_RETURN_ERROR_INVALID_HANDLE;
      }

      VkAccelerationStructureInstanceKHR* asInstance = (VkAccelerationStructureInstanceKHR*) &mapped_mem[i * sizeof(VkAccelerationStructureInstanceKHR)];
      memcpy(&asInstance->transform, &instances[i].transform, sizeof(VkTransformMatrixKHR));
      asInstance->instanceCustomIndex = instances[i].faceIndexOffset;
      asInstance->mask = 0xFF;
      asInstance->instanceShaderBindingTableRecordOffset = instances[i].hitGroupIndex;
      asInstance->flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
      asInstance->accelerationStructureReference = iblas->address;

      areAllBlasOpaque &= iblas->isOpaque;
    }

    vmaUnmapMemory(idevice->allocator, itlas->instances.allocation);
  }

  // Create TLAS
  VkAccelerationStructureGeometryKHR asGeom = {};
  asGeom.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
  asGeom.pNext = nullptr;
  asGeom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
  asGeom.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
  asGeom.geometry.instances.pNext = nullptr;
  asGeom.geometry.instances.arrayOfPointers = VK_FALSE;
  asGeom.geometry.instances.data.hostAddress = nullptr;
  asGeom.geometry.instances.data.deviceAddress = cgpuGetBufferDeviceAddress(idevice, &itlas->instances);
  asGeom.flags = areAllBlasOpaque ? VK_GEOMETRY_OPAQUE_BIT_KHR : 0;

  if (!cgpuCreateTopOrBottomAs(device, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, &asGeom, instanceCount, &itlas->buffer, &itlas->as))
  {
    cgpuDestroyIBuffer(idevice, &itlas->instances);
    CGPU_RETURN_ERROR("failed to build TLAS");
  }

  return true;
}

bool cgpuDestroyBlas(CgpuDevice device, CgpuBlas blas)
{
  CgpuIDevice* idevice;
  if (!cgpuResolveDevice(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIBlas* iblas;
  if (!cgpuResolveBlas(blas, &iblas)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  idevice->table.vkDestroyAccelerationStructureKHR(idevice->logicalDevice, iblas->as, nullptr);
  cgpuDestroyIBuffer(idevice, &iblas->buffer);
  cgpuDestroyIBuffer(idevice, &iblas->indices);
  cgpuDestroyIBuffer(idevice, &iblas->vertices);

  iinstance->iblasStore.free(blas.handle);
  return true;
}

bool cgpuDestroyTlas(CgpuDevice device, CgpuTlas tlas)
{
  CgpuIDevice* idevice;
  if (!cgpuResolveDevice(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuITlas* itlas;
  if (!cgpuResolveTlas(tlas, &itlas)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  idevice->table.vkDestroyAccelerationStructureKHR(idevice->logicalDevice, itlas->as, nullptr);
  cgpuDestroyIBuffer(idevice, &itlas->instances);
  cgpuDestroyIBuffer(idevice, &itlas->buffer);

  iinstance->itlasStore.free(tlas.handle);
  return true;
}

bool cgpuCreateCommandBuffer(CgpuDevice device, CgpuCommandBuffer* commandBuffer)
{
  CgpuIDevice* idevice;
  if (!cgpuResolveDevice(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  commandBuffer->handle = iinstance->icommandBufferStore.allocate();

  CgpuICommandBuffer* icommandBuffer;
  if (!cgpuResolveCommandBuffer(*commandBuffer, &icommandBuffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  icommandBuffer->device.handle = device.handle;

  VkCommandBufferAllocateInfo cmdbufAllocInfo = {};
  cmdbufAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cmdbufAllocInfo.pNext = nullptr;
  cmdbufAllocInfo.commandPool = idevice->commandPool;
  cmdbufAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cmdbufAllocInfo.commandBufferCount = 1;

  VkResult result = idevice->table.vkAllocateCommandBuffers(
    idevice->logicalDevice,
    &cmdbufAllocInfo,
    &icommandBuffer->commandBuffer
  );
  if (result != VK_SUCCESS) {
    iinstance->icommandBufferStore.free(commandBuffer->handle);
    CGPU_RETURN_ERROR("failed to allocate command buffer");
  }

  return true;
}

bool cgpuDestroyCommandBuffer(CgpuDevice device, CgpuCommandBuffer commandBuffer)
{
  CgpuIDevice* idevice;
  if (!cgpuResolveDevice(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuICommandBuffer* icommandBuffer;
  if (!cgpuResolveCommandBuffer(commandBuffer, &icommandBuffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  idevice->table.vkFreeCommandBuffers(
    idevice->logicalDevice,
    idevice->commandPool,
    1,
    &icommandBuffer->commandBuffer
  );

  iinstance->icommandBufferStore.free(commandBuffer.handle);
  return true;
}

bool cgpuBeginCommandBuffer(CgpuCommandBuffer commandBuffer)
{
  CgpuICommandBuffer* icommandBuffer;
  if (!cgpuResolveCommandBuffer(commandBuffer, &icommandBuffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIDevice* idevice;
  if (!cgpuResolveDevice(icommandBuffer->device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  VkCommandBufferBeginInfo beginInfo = {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.pNext = nullptr;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
  beginInfo.pInheritanceInfo = nullptr;

  VkResult result = idevice->table.vkBeginCommandBuffer(
    icommandBuffer->commandBuffer,
    &beginInfo
  );

  if (result != VK_SUCCESS) {
    CGPU_RETURN_ERROR("failed to begin command buffer");
  }
  return true;
}

bool cgpuCmdBindPipeline(CgpuCommandBuffer commandBuffer, CgpuPipeline pipeline)
{
  CgpuICommandBuffer* icommandBuffer;
  if (!cgpuResolveCommandBuffer(commandBuffer, &icommandBuffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIDevice* idevice;
  if (!cgpuResolveDevice(icommandBuffer->device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIPipeline* ipipeline;
  if (!cgpuResolvePipeline(pipeline, &ipipeline)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  idevice->table.vkCmdBindPipeline(
    icommandBuffer->commandBuffer,
    ipipeline->bindPoint,
    ipipeline->pipeline
  );
  idevice->table.vkCmdBindDescriptorSets(
    icommandBuffer->commandBuffer,
    ipipeline->bindPoint,
    ipipeline->layout,
    0,
    1,
    &ipipeline->descriptorSet,
    0,
    0
  );

  return true;
}

bool cgpuCmdTransitionShaderImageLayouts(CgpuCommandBuffer commandBuffer,
                                         CgpuShader shader,
                                         uint32_t imageCount,
                                         const CgpuImageBinding* images)
{
  CgpuIShader* ishader;
  if (!cgpuResolveShader(shader, &ishader)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuICommandBuffer* icommandBuffer;
  if (!cgpuResolveCommandBuffer(commandBuffer, &icommandBuffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIDevice* idevice;
  if (!cgpuResolveDevice(icommandBuffer->device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  GbSmallVector<VkImageMemoryBarrier, 64> barriers;

  /* FIXME: this has quadratic complexity */
  const CgpuShaderReflection* reflection = &ishader->reflection;
  for (uint32_t i = 0; i < reflection->bindings.size(); i++)
  {
    const CgpuShaderReflectionBinding* binding = &reflection->bindings[i];

    VkImageLayout newLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (binding->descriptorType == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE)
    {
      newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }
    else if (binding->descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
    {
      newLayout = VK_IMAGE_LAYOUT_GENERAL;
    }
    else
    {
      /* Not an image. */
      continue;
    }

    for (uint32_t j = 0; j < binding->count; j++)
    {
      /* Image layout needs transitioning. */
      const CgpuImageBinding* imageBinding = nullptr;
      for (uint32_t k = 0; k < imageCount; k++)
      {
        if (images[k].binding == binding->binding && images[k].index == j)
        {
          imageBinding = &images[k];
          break;
        }
      }
      if (!imageBinding)
      {
        CGPU_RETURN_ERROR("descriptor set binding mismatch");
      }

      CgpuIImage* iimage;
      if (!cgpuResolveImage(imageBinding->image, &iimage)) {
        CGPU_RETURN_ERROR_INVALID_HANDLE;
      }

      VkImageLayout oldLayout = iimage->layout;
      if (newLayout == oldLayout)
      {
        continue;
      }

      VkAccessFlags accessMask = 0;
      if (binding->readAccess) {
        accessMask = VK_ACCESS_SHADER_READ_BIT;
      }
      if (binding->writeAccess) {
        accessMask = VK_ACCESS_SHADER_WRITE_BIT;
      }

      VkImageMemoryBarrier barrier = {};
      barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
      barrier.pNext = nullptr;
      barrier.srcAccessMask = iimage->accessMask;
      barrier.dstAccessMask = accessMask;
      barrier.oldLayout = oldLayout;
      barrier.newLayout = newLayout;
      barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.image = iimage->image;
      barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      barrier.subresourceRange.baseMipLevel = 0;
      barrier.subresourceRange.levelCount = 1;
      barrier.subresourceRange.baseArrayLayer = 0;
      barrier.subresourceRange.layerCount = 1;
      barriers.push_back(barrier);

      iimage->accessMask = accessMask;
      iimage->layout = newLayout;
    }
  }

  if (barriers.size() > 0)
  {
    VkPipelineStageFlags stageFlags = cgpuPipelineStageFlagsFromShaderStageFlags(ishader->stageFlags);

    idevice->table.vkCmdPipelineBarrier(
      icommandBuffer->commandBuffer,
      stageFlags,
      stageFlags,
      0,
      0,
      nullptr,
      0,
      nullptr,
      barriers.size(),
      barriers.data()
    );
  }

  return true;
}

bool cgpuCmdUpdateBindings(CgpuCommandBuffer commandBuffer,
                           CgpuPipeline pipeline,
                           const CgpuBindings* bindings
)
{
  CgpuICommandBuffer* icommandBuffer;
  if (!cgpuResolveCommandBuffer(commandBuffer, &icommandBuffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIDevice* idevice;
  if (!cgpuResolveDevice(icommandBuffer->device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIPipeline* ipipeline;
  if (!cgpuResolvePipeline(pipeline, &ipipeline)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  GbSmallVector<VkDescriptorBufferInfo, 64> bufferInfos;
  GbSmallVector<VkDescriptorImageInfo, 128> imageInfos;
  GbSmallVector<VkWriteDescriptorSetAccelerationStructureKHR, 1> asInfos;

  bufferInfos.reserve(bindings->bufferCount);
  imageInfos.reserve(bindings->imageCount + bindings->samplerCount);
  asInfos.reserve(bindings->tlasCount);

  GbSmallVector<VkWriteDescriptorSet, 128> writeDescriptorSets;

  /* FIXME: this has a rather high complexity */
  for (uint32_t i = 0; i < ipipeline->descriptorSetLayoutBindings.size(); i++)
  {
    const VkDescriptorSetLayoutBinding* layoutBinding = &ipipeline->descriptorSetLayoutBindings[i];

    VkWriteDescriptorSet writeDescriptorSet = {};
    writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet.pNext = nullptr;
    writeDescriptorSet.dstSet = ipipeline->descriptorSet;
    writeDescriptorSet.dstBinding = layoutBinding->binding;
    writeDescriptorSet.dstArrayElement = 0;
    writeDescriptorSet.descriptorCount = layoutBinding->descriptorCount;
    writeDescriptorSet.descriptorType = layoutBinding->descriptorType;
    writeDescriptorSet.pTexelBufferView = nullptr;
    writeDescriptorSet.pBufferInfo = nullptr;
    writeDescriptorSet.pImageInfo = nullptr;

    for (uint32_t j = 0; j < layoutBinding->descriptorCount; j++)
    {
      bool slotHandled = false;

      if (layoutBinding->descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
      {
        for (uint32_t k = 0; k < bindings->bufferCount; ++k)
        {
          const CgpuBufferBinding* bufferBinding = &bindings->buffers[k];

          if (bufferBinding->binding != layoutBinding->binding || bufferBinding->index != j)
          {
            continue;
          }

          CgpuIBuffer* ibuffer;
          CgpuBuffer buffer = bufferBinding->buffer;
          if (!cgpuResolveBuffer(buffer, &ibuffer)) {
            CGPU_RETURN_ERROR_INVALID_HANDLE;
          }

          if ((bufferBinding->offset % idevice->properties.minStorageBufferOffsetAlignment) != 0) {
            CGPU_RETURN_ERROR("buffer binding offset not aligned");
          }

          VkDescriptorBufferInfo bufferInfo = {};
          bufferInfo.buffer = ibuffer->buffer;
          bufferInfo.offset = bufferBinding->offset;
          bufferInfo.range = (bufferBinding->size == CGPU_WHOLE_SIZE) ? (ibuffer->size - bufferBinding->offset) : bufferBinding->size;
          bufferInfos.push_back(bufferInfo);

          if (j == 0)
          {
            writeDescriptorSet.pBufferInfo = &bufferInfos.back();
          }

          slotHandled = true;
          break;
        }
      }
      else if (layoutBinding->descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE ||
               layoutBinding->descriptorType == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE)
      {
        for (uint32_t k = 0; k < bindings->imageCount; k++)
        {
          const CgpuImageBinding* imageBinding = &bindings->images[k];

          if (imageBinding->binding != layoutBinding->binding || imageBinding->index != j)
          {
            continue;
          }

          CgpuIImage* iimage;
          CgpuImage image = imageBinding->image;
          if (!cgpuResolveImage(image, &iimage)) {
            CGPU_RETURN_ERROR_INVALID_HANDLE;
          }

          VkDescriptorImageInfo imageInfo = {};
          imageInfo.sampler = VK_NULL_HANDLE;
          imageInfo.imageView = iimage->imageView;
          imageInfo.imageLayout = iimage->layout;
          imageInfos.push_back(imageInfo);

          if (j == 0)
          {
            writeDescriptorSet.pImageInfo = &imageInfos.back();
          }

          slotHandled = true;
          break;
        }
      }
      else if (layoutBinding->descriptorType == VK_DESCRIPTOR_TYPE_SAMPLER)
      {
        for (uint32_t k = 0; k < bindings->samplerCount; k++)
        {
          const CgpuSamplerBinding* samplerBinding = &bindings->samplers[k];

          if (samplerBinding->binding != layoutBinding->binding || samplerBinding->index != j)
          {
            continue;
          }

          CgpuISampler* isampler;
          CgpuSampler sampler = samplerBinding->sampler;
          if (!cgpuResolveSampler(sampler, &isampler)) {
            CGPU_RETURN_ERROR_INVALID_HANDLE;
          }

          VkDescriptorImageInfo imageInfo = {};
          imageInfo.sampler = isampler->sampler;
          imageInfo.imageView = VK_NULL_HANDLE;
          imageInfo.imageLayout = VK_IMAGE_LAYOUT_UNDEFINED;
          imageInfos.push_back(imageInfo);

          if (j == 0)
          {
            writeDescriptorSet.pImageInfo = &imageInfos.back();
          }

          slotHandled = true;
          break;
        }
      }
      else if (layoutBinding->descriptorType == VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR)
      {
        for (uint32_t k = 0; k < bindings->tlasCount; ++k)
        {
          const CgpuTlasBinding* asBinding = &bindings->tlases[k];

          if (asBinding->binding != layoutBinding->binding || asBinding->index != j)
          {
            continue;
          }

          CgpuITlas* itlas;
          CgpuTlas tlas = asBinding->as;
          if (!cgpuResolveTlas(tlas, &itlas)) {
            CGPU_RETURN_ERROR_INVALID_HANDLE;
          }

          VkWriteDescriptorSetAccelerationStructureKHR asInfo = {};
          asInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
          asInfo.pNext = nullptr;
          asInfo.accelerationStructureCount = 1;
          asInfo.pAccelerationStructures = &itlas->as;
          asInfos.push_back(asInfo);

          if (j == 0)
          {
            writeDescriptorSet.pNext = &asInfos.back();
          }

          slotHandled = true;
          break;
        }
      }

      if (!slotHandled)
      {
        CGPU_RETURN_ERROR("resource binding mismatch");
      }
    }

    writeDescriptorSets.push_back(writeDescriptorSet);
  }

  idevice->table.vkUpdateDescriptorSets(
    idevice->logicalDevice,
    writeDescriptorSets.size(),
    writeDescriptorSets.data(),
    0,
    nullptr
  );

  return true;
}

bool cgpuCmdUpdateBuffer(CgpuCommandBuffer commandBuffer,
                         const uint8_t* data,
                         uint64_t size,
                         CgpuBuffer dstBuffer,
                         uint64_t dstOffset)
{
  CgpuICommandBuffer* icommandBuffer;
  if (!cgpuResolveCommandBuffer(commandBuffer, &icommandBuffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIDevice* idevice;
  if (!cgpuResolveDevice(icommandBuffer->device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIBuffer* idstBuffer;
  if (!cgpuResolveBuffer(dstBuffer, &idstBuffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  idevice->table.vkCmdUpdateBuffer(
    icommandBuffer->commandBuffer,
    idstBuffer->buffer,
    dstOffset,
    size,
    (const void*) data
  );

  return true;
}

bool cgpuCmdCopyBuffer(CgpuCommandBuffer commandBuffer,
                       CgpuBuffer srcBuffer,
                       uint64_t srcOffset,
                       CgpuBuffer dstBuffer,
                       uint64_t dstOffset,
                       uint64_t size)
{
  CgpuICommandBuffer* icommandBuffer;
  if (!cgpuResolveCommandBuffer(commandBuffer, &icommandBuffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIDevice* idevice;
  if (!cgpuResolveDevice(icommandBuffer->device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIBuffer* isrcBuffer;
  if (!cgpuResolveBuffer(srcBuffer, &isrcBuffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIBuffer* idstBuffer;
  if (!cgpuResolveBuffer(dstBuffer, &idstBuffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  VkBufferCopy region;
  region.srcOffset = srcOffset;
  region.dstOffset = dstOffset;
  region.size = (size == CGPU_WHOLE_SIZE) ? isrcBuffer->size : size;

  idevice->table.vkCmdCopyBuffer(
    icommandBuffer->commandBuffer,
    isrcBuffer->buffer,
    idstBuffer->buffer,
    1,
    &region
  );

  return true;
}

bool cgpuCmdCopyBufferToImage(CgpuCommandBuffer commandBuffer,
                              CgpuBuffer buffer,
                              CgpuImage image,
                              const CgpuBufferImageCopyDesc* desc)
{
  CgpuICommandBuffer* icommandBuffer;
  if (!cgpuResolveCommandBuffer(commandBuffer, &icommandBuffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIDevice* idevice;
  if (!cgpuResolveDevice(icommandBuffer->device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIBuffer* ibuffer;
  if (!cgpuResolveBuffer(buffer, &ibuffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIImage* iimage;
  if (!cgpuResolveImage(image, &iimage)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  if (iimage->layout != VK_IMAGE_LAYOUT_GENERAL)
  {
    VkAccessFlags accessMask = iimage->accessMask | VK_ACCESS_MEMORY_WRITE_BIT;
    VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL;

    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.pNext = nullptr;
    barrier.srcAccessMask = iimage->accessMask;
    barrier.dstAccessMask = accessMask;
    barrier.oldLayout = iimage->layout;
    barrier.newLayout = layout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = iimage->image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    idevice->table.vkCmdPipelineBarrier(
      icommandBuffer->commandBuffer,
      // FIXME: batch this barrier and reduce pipeline flags scope
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
      0,
      0,
      nullptr,
      0,
      nullptr,
      1,
      &barrier
    );

    iimage->layout = layout;
    iimage->accessMask = accessMask;
  }

  VkImageSubresourceLayers layers;
  layers.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  layers.mipLevel = 0;
  layers.baseArrayLayer = 0;
  layers.layerCount = 1;

  VkOffset3D offset;
  offset.x = desc->texelOffsetX;
  offset.y = desc->texelOffsetY;
  offset.z = desc->texelOffsetZ;

  VkExtent3D extent;
  extent.width = desc->texelExtentX;
  extent.height = desc->texelExtentY;
  extent.depth = desc->texelExtentZ;

  VkBufferImageCopy region;
  region.bufferOffset = desc->bufferOffset;
  region.bufferRowLength = 0;
  region.bufferImageHeight = 0;
  region.imageSubresource = layers;
  region.imageOffset = offset;
  region.imageExtent = extent;

  idevice->table.vkCmdCopyBufferToImage(
    icommandBuffer->commandBuffer,
    ibuffer->buffer,
    iimage->image,
    iimage->layout,
    1,
    &region
  );

  return true;
}

bool cgpuCmdPushConstants(CgpuCommandBuffer commandBuffer,
                          CgpuPipeline pipeline,
                          CgpuShaderStageFlags stageFlags,
                          uint32_t size,
                          const void* data)
{
  CgpuICommandBuffer* icommandBuffer;
  if (!cgpuResolveCommandBuffer(commandBuffer, &icommandBuffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIDevice* idevice;
  if (!cgpuResolveDevice(icommandBuffer->device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIPipeline* ipipeline;
  if (!cgpuResolvePipeline(pipeline, &ipipeline)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  idevice->table.vkCmdPushConstants(
    icommandBuffer->commandBuffer,
    ipipeline->layout,
    (VkShaderStageFlags) stageFlags,
    0,
    size,
    data
  );
  return true;
}

bool cgpuCmdDispatch(CgpuCommandBuffer commandBuffer,
                     uint32_t dim_x,
                     uint32_t dim_y,
                     uint32_t dim_z)
{
  CgpuICommandBuffer* icommandBuffer;
  if (!cgpuResolveCommandBuffer(commandBuffer, &icommandBuffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIDevice* idevice;
  if (!cgpuResolveDevice(icommandBuffer->device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  idevice->table.vkCmdDispatch(
    icommandBuffer->commandBuffer,
    dim_x,
    dim_y,
    dim_z
  );
  return true;
}

bool cgpuCmdPipelineBarrier(CgpuCommandBuffer commandBuffer,
                            uint32_t barrierCount,
                            const CgpuMemoryBarrier* barriers,
                            uint32_t bufferBarrierCount,
                            const CgpuBufferMemoryBarrier* bufferBarriers,
                            uint32_t imageBarrierCount,
                            const CgpuImageMemoryBarrier* imageBarriers)
{
  CgpuICommandBuffer* icommandBuffer;
  if (!cgpuResolveCommandBuffer(commandBuffer, &icommandBuffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIDevice* idevice;
  if (!cgpuResolveDevice(icommandBuffer->device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  GbSmallVector<VkMemoryBarrier, 128> vkMemBarriers;

  for (uint32_t i = 0; i < barrierCount; ++i)
  {
    const CgpuMemoryBarrier* bCgpu = &barriers[i];

    VkMemoryBarrier bVk = {};
    bVk.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    bVk.pNext = nullptr;
    bVk.srcAccessMask = (VkAccessFlags) bCgpu->srcAccessFlags;
    bVk.dstAccessMask = (VkAccessFlags) bCgpu->dstAccessFlags;
    vkMemBarriers.push_back(bVk);
  }

  GbSmallVector<VkBufferMemoryBarrier, 32> vkBufferMemBarriers;
  GbSmallVector<VkImageMemoryBarrier, 128> vkImageMemBarriers;

  for (uint32_t i = 0; i < bufferBarrierCount; ++i)
  {
    const CgpuBufferMemoryBarrier* bCgpu = &bufferBarriers[i];

    CgpuIBuffer* ibuffer;
    if (!cgpuResolveBuffer(bCgpu->buffer, &ibuffer)) {
      CGPU_RETURN_ERROR_INVALID_HANDLE;
    }

    VkBufferMemoryBarrier bVk = {};
    bVk.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    bVk.pNext = nullptr;
    bVk.srcAccessMask = (VkAccessFlags) bCgpu->srcAccessFlags;
    bVk.dstAccessMask = (VkAccessFlags) bCgpu->dstAccessFlags;
    bVk.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bVk.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bVk.buffer = ibuffer->buffer;
    bVk.offset = bCgpu->offset;
    bVk.size = (bCgpu->size == CGPU_WHOLE_SIZE) ? VK_WHOLE_SIZE : bCgpu->size;
    vkBufferMemBarriers.push_back(bVk);
  }

  for (uint32_t i = 0; i < imageBarrierCount; ++i)
  {
    const CgpuImageMemoryBarrier* bCgpu = &imageBarriers[i];

    CgpuIImage* iimage;
    if (!cgpuResolveImage(bCgpu->image, &iimage)) {
      CGPU_RETURN_ERROR_INVALID_HANDLE;
    }

    VkAccessFlags accessMask = (VkAccessFlags) bCgpu->accessMask;

    VkImageMemoryBarrier bVk = {};
    bVk.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    bVk.pNext = nullptr;
    bVk.srcAccessMask = iimage->accessMask;
    bVk.dstAccessMask = accessMask;
    bVk.oldLayout = iimage->layout;
    bVk.newLayout = iimage->layout;
    bVk.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bVk.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bVk.image = iimage->image;
    bVk.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    bVk.subresourceRange.baseMipLevel = 0;
    bVk.subresourceRange.levelCount = 1;
    bVk.subresourceRange.baseArrayLayer = 0;
    bVk.subresourceRange.layerCount = 1;
    vkImageMemBarriers.push_back(bVk);

    iimage->accessMask = accessMask;
  }

  idevice->table.vkCmdPipelineBarrier(
    icommandBuffer->commandBuffer,
    // FIXME: expose flags in desc struct
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_HOST_BIT,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_HOST_BIT,
    0,
    vkMemBarriers.size(),
    vkMemBarriers.data(),
    vkBufferMemBarriers.size(),
    vkBufferMemBarriers.data(),
    vkImageMemBarriers.size(),
    vkImageMemBarriers.data()
  );

  return true;
}

bool cgpuCmdResetTimestamps(CgpuCommandBuffer commandBuffer,
                            uint32_t offset,
                            uint32_t count)
{
  CgpuICommandBuffer* icommandBuffer;
  if (!cgpuResolveCommandBuffer(commandBuffer, &icommandBuffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIDevice* idevice;
  if (!cgpuResolveDevice(icommandBuffer->device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  idevice->table.vkCmdResetQueryPool(
    icommandBuffer->commandBuffer,
    idevice->timestamp_pool,
    offset,
    count
  );

  return true;
}

bool cgpuCmdWriteTimestamp(CgpuCommandBuffer commandBuffer,
                           uint32_t timestampIndex)
{
  CgpuICommandBuffer* icommandBuffer;
  if (!cgpuResolveCommandBuffer(commandBuffer, &icommandBuffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIDevice* idevice;
  if (!cgpuResolveDevice(icommandBuffer->device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  idevice->table.vkCmdWriteTimestamp(
    icommandBuffer->commandBuffer,
    // FIXME: use correct pipeline flag bits
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    idevice->timestamp_pool,
    timestampIndex
  );

  return true;
}

bool cgpuCmdCopyTimestamps(CgpuCommandBuffer commandBuffer,
                           CgpuBuffer buffer,
                           uint32_t offset,
                           uint32_t count,
                           bool waitUntilAvailable)
{
  uint32_t lastIndex = offset + count;
  if (lastIndex >= CGPU_MAX_TIMESTAMP_QUERIES) {
    CGPU_RETURN_ERROR_HARDCODED_LIMIT_REACHED;
  }

  CgpuICommandBuffer* icommandBuffer;
  if (!cgpuResolveCommandBuffer(commandBuffer, &icommandBuffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIDevice* idevice;
  if (!cgpuResolveDevice(icommandBuffer->device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIBuffer* ibuffer;
  if (!cgpuResolveBuffer(buffer, &ibuffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  VkQueryResultFlags waitFlag = waitUntilAvailable ? VK_QUERY_RESULT_WAIT_BIT : VK_QUERY_RESULT_WITH_AVAILABILITY_BIT;

  idevice->table.vkCmdCopyQueryPoolResults(
    icommandBuffer->commandBuffer,
    idevice->timestamp_pool,
    offset,
    count,
    ibuffer->buffer,
    0,
    sizeof(uint64_t),
    VK_QUERY_RESULT_64_BIT | waitFlag
  );

  return true;
}

bool cgpuCmdTraceRays(CgpuCommandBuffer commandBuffer, CgpuPipeline rtPipeline, uint32_t width, uint32_t height)
{
  CgpuICommandBuffer* icommandBuffer;
  if (!cgpuResolveCommandBuffer(commandBuffer, &icommandBuffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIDevice* idevice;
  if (!cgpuResolveDevice(icommandBuffer->device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIPipeline* ipipeline;
  if (!cgpuResolvePipeline(rtPipeline, &ipipeline)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  VkStridedDeviceAddressRegionKHR callableSBT = {};
  idevice->table.vkCmdTraceRaysKHR(icommandBuffer->commandBuffer,
                                   &ipipeline->sbtRgen,
                                   &ipipeline->sbtMiss,
                                   &ipipeline->sbtHit,
                                   &callableSBT,
                                   width, height, 1);
  return true;
}

bool cgpuEndCommandBuffer(CgpuCommandBuffer commandBuffer)
{
  CgpuICommandBuffer* icommandBuffer;
  if (!cgpuResolveCommandBuffer(commandBuffer, &icommandBuffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIDevice* idevice;
  if (!cgpuResolveDevice(icommandBuffer->device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  idevice->table.vkEndCommandBuffer(icommandBuffer->commandBuffer);
  return true;
}

bool cgpuCreateFence(CgpuDevice device, CgpuFence* fence)
{
  CgpuIDevice* idevice;
  if (!cgpuResolveDevice(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  fence->handle = iinstance->ifenceStore.allocate();

  CgpuIFence* ifence;
  if (!cgpuResolveFence(*fence, &ifence)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  VkFenceCreateInfo createInfo;
  createInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  createInfo.pNext = nullptr;
  createInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

  VkResult result = idevice->table.vkCreateFence(
    idevice->logicalDevice,
    &createInfo,
    nullptr,
    &ifence->fence
  );

  if (result != VK_SUCCESS) {
    iinstance->ifenceStore.free(fence->handle);
    CGPU_RETURN_ERROR("failed to create fence");
  }
  return true;
}

bool cgpuDestroyFence(CgpuDevice device, CgpuFence fence)
{
  CgpuIDevice* idevice;
  if (!cgpuResolveDevice(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIFence* ifence;
  if (!cgpuResolveFence(fence, &ifence)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  idevice->table.vkDestroyFence(
    idevice->logicalDevice,
    ifence->fence,
    nullptr
  );
  iinstance->ifenceStore.free(fence.handle);
  return true;
}

bool cgpuResetFence(CgpuDevice device, CgpuFence fence)
{
  CgpuIDevice* idevice;
  if (!cgpuResolveDevice(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIFence* ifence;
  if (!cgpuResolveFence(fence, &ifence)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  VkResult result = idevice->table.vkResetFences(
    idevice->logicalDevice,
    1,
    &ifence->fence
  );
  if (result != VK_SUCCESS) {
    CGPU_RETURN_ERROR("failed to reset fence");
  }
  return true;
}

bool cgpuWaitForFence(CgpuDevice device, CgpuFence fence)
{
  CgpuIDevice* idevice;
  if (!cgpuResolveDevice(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIFence* ifence;
  if (!cgpuResolveFence(fence, &ifence)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  VkResult result = idevice->table.vkWaitForFences(
    idevice->logicalDevice,
    1,
    &ifence->fence,
    VK_TRUE,
    UINT64_MAX
  );
  if (result != VK_SUCCESS) {
    CGPU_RETURN_ERROR("failed to wait for fence");
  }
  return true;
}

bool cgpuSubmitCommandBuffer(CgpuDevice device,
                             CgpuCommandBuffer commandBuffer,
                             CgpuFence fence)
{
  CgpuIDevice* idevice;
  if (!cgpuResolveDevice(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuICommandBuffer* icommandBuffer;
  if (!cgpuResolveCommandBuffer(commandBuffer, &icommandBuffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIFence* ifence;
  if (!cgpuResolveFence(fence, &ifence)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  VkSubmitInfo submitInfo;
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.pNext = nullptr;
  submitInfo.waitSemaphoreCount = 0;
  submitInfo.pWaitSemaphores = nullptr;
  submitInfo.pWaitDstStageMask = nullptr;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &icommandBuffer->commandBuffer;
  submitInfo.signalSemaphoreCount = 0;
  submitInfo.pSignalSemaphores = nullptr;

  VkResult result = idevice->table.vkQueueSubmit(
    idevice->computeQueue,
    1,
    &submitInfo,
    ifence->fence
  );

  if (result != VK_SUCCESS) {
    CGPU_RETURN_ERROR("failed to submit command buffer");
  }
  return true;
}

bool cgpuFlushMappedMemory(CgpuDevice device,
                           CgpuBuffer buffer,
                           uint64_t offset,
                           uint64_t size)
{
  CgpuIDevice* idevice;
  if (!cgpuResolveDevice(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIBuffer* ibuffer;
  if (!cgpuResolveBuffer(buffer, &ibuffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  VkResult result = vmaFlushAllocation(
    idevice->allocator,
    ibuffer->allocation,
    offset,
    (size == CGPU_WHOLE_SIZE) ? ibuffer->size : size
  );

  if (result != VK_SUCCESS) {
    CGPU_RETURN_ERROR("failed to flush mapped memory");
  }
  return true;
}

bool cgpuInvalidateMappedMemory(CgpuDevice device,
                                CgpuBuffer buffer,
                                uint64_t offset,
                                uint64_t size)
{
  CgpuIDevice* idevice;
  if (!cgpuResolveDevice(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIBuffer* ibuffer;
  if (!cgpuResolveBuffer(buffer, &ibuffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  VkResult result = vmaInvalidateAllocation(
    idevice->allocator,
    ibuffer->allocation,
    offset,
    (size == CGPU_WHOLE_SIZE) ? ibuffer->size : size
  );

  if (result != VK_SUCCESS) {
    CGPU_RETURN_ERROR("failed to invalidate mapped memory");
  }
  return true;
}

bool cgpuGetPhysicalDeviceFeatures(CgpuDevice device,
                                   CgpuPhysicalDeviceFeatures* features)
{
  CgpuIDevice* idevice;
  if (!cgpuResolveDevice(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  memcpy(features, &idevice->features, sizeof(CgpuPhysicalDeviceFeatures));
  return true;
}

bool cgpuGetPhysicalDeviceProperties(CgpuDevice device,
                                     CgpuPhysicalDeviceProperties* properties)
{
  CgpuIDevice* idevice;
  if (!cgpuResolveDevice(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  memcpy(properties, &idevice->properties, sizeof(CgpuPhysicalDeviceProperties));
  return true;
}
