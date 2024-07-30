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

#include "Cgpu.h"
#include "ShaderReflection.h"

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <memory>

#include <volk.h>

#include <gtl/gb/Data.h>
#include <gtl/gb/Fmt.h>
#include <gtl/gb/Log.h>

#ifdef __clang__
#pragma clang diagnostic push
// Silence nullability log spam on AppleClang
#pragma clang diagnostic ignored "-Wnullability-completeness"
#endif

#ifdef GTL_VERBOSE
#define VMA_DEBUG_LOG_FORMAT(format, ...) do { \
    GB_DEBUG_DYN("[VMA] {}", GB_FMT_SPRINTF((format), __VA_ARGS__)); \
  } while (false)
#endif

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#include <gtl/gb/LinearDataStore.h>
#include <gtl/gb/SmallVector.h>
#include <gtl/gb/Log.h>

namespace gtl
{
  /* Constants. */

  constexpr static const uint32_t CGPU_MIN_VK_API_VERSION = VK_API_VERSION_1_1;

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
    VkPipelineCache              pipelineCache;
  };

  struct CgpuIBuffer
  {
    VkBuffer       buffer;
    uint64_t       size;
    VmaAllocation  allocation;
  };

  struct CgpuIImage
  {
    VkImage           image;
    VkImageView       imageView;
    VmaAllocation     allocation;
    uint64_t          size;
    uint32_t          width;
    uint32_t          height;
    uint32_t          depth;
    VkImageLayout     layout;
    VkAccessFlags2KHR accessMask;
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

  struct CgpuISemaphore
  {
    VkSemaphore semaphore;
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
    GbLinearDataStore<CgpuISemaphore, 16> isemaphoreStore;
    GbLinearDataStore<CgpuICommandBuffer, 16> icommandBufferStore;
    GbLinearDataStore<CgpuISampler, 8> isamplerStore;
    GbLinearDataStore<CgpuIBlas, 1024> iblasStore;
    GbLinearDataStore<CgpuITlas, 1> itlasStore;
    bool debugUtilsEnabled;
  };

  static std::unique_ptr<CgpuIInstance> iinstance = nullptr;

  /* Helper macros. */

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

#define CGPU_RETURN_ERROR(msg)                      \
  do {                                              \
    GB_ERROR("{}:{}: {}", __FILE__, __LINE__, msg); \
    return false;                                   \
  } while (false)

#define CGPU_RETURN_ERROR_INVALID_HANDLE                              \
  CGPU_RETURN_ERROR("invalid resource handle")

#define CGPU_RETURN_ERROR_HARDCODED_LIMIT_REACHED                     \
  CGPU_RETURN_ERROR("hardcoded limit reached")

#define CGPU_RESOLVE_HANDLE(RESOURCE_NAME, HANDLE_TYPE, IRESOURCE_TYPE, RESOURCE_STORE)          \
  CGPU_INLINE static bool cgpuResolve##RESOURCE_NAME(HANDLE_TYPE handle, IRESOURCE_TYPE** idata) \
  {                                                                                              \
    return iinstance->RESOURCE_STORE.get(handle.handle, idata);                                  \
  }

  CGPU_RESOLVE_HANDLE(       Device,        CgpuDevice,        CgpuIDevice,        ideviceStore)
  CGPU_RESOLVE_HANDLE(       Buffer,        CgpuBuffer,        CgpuIBuffer,        ibufferStore)
  CGPU_RESOLVE_HANDLE(        Image,         CgpuImage,         CgpuIImage,         iimageStore)
  CGPU_RESOLVE_HANDLE(       Shader,        CgpuShader,        CgpuIShader,        ishaderStore)
  CGPU_RESOLVE_HANDLE(     Pipeline,      CgpuPipeline,      CgpuIPipeline,      ipipelineStore)
  CGPU_RESOLVE_HANDLE(    Semaphore,     CgpuSemaphore,     CgpuISemaphore,     isemaphoreStore)
  CGPU_RESOLVE_HANDLE(CommandBuffer, CgpuCommandBuffer, CgpuICommandBuffer, icommandBufferStore)
  CGPU_RESOLVE_HANDLE(      Sampler,       CgpuSampler,       CgpuISampler,       isamplerStore)
  CGPU_RESOLVE_HANDLE(         Blas,          CgpuBlas,          CgpuIBlas,          iblasStore)
  CGPU_RESOLVE_HANDLE(         Tlas,          CgpuTlas,          CgpuITlas,          itlasStore)

#define CGPU_RESOLVE_OR_RETURN(HANDLE, VAR_NAME, ITYPE, RESOLVE_FUNC)   \
  ITYPE* VAR_NAME;                                                      \
  if (!RESOLVE_FUNC(HANDLE, &VAR_NAME)) {                               \
    CGPU_RETURN_ERROR_INVALID_HANDLE;                                   \
  }

#define CGPU_RESOLVE_OR_RETURN_DEVICE(HANDLE, VAR_NAME)         CGPU_RESOLVE_OR_RETURN(HANDLE, VAR_NAME, CgpuIDevice, cgpuResolveDevice)
#define CGPU_RESOLVE_OR_RETURN_BUFFER(HANDLE, VAR_NAME)         CGPU_RESOLVE_OR_RETURN(HANDLE, VAR_NAME, CgpuIBuffer, cgpuResolveBuffer)
#define CGPU_RESOLVE_OR_RETURN_IMAGE(HANDLE, VAR_NAME)          CGPU_RESOLVE_OR_RETURN(HANDLE, VAR_NAME, CgpuIImage, cgpuResolveImage)
#define CGPU_RESOLVE_OR_RETURN_SHADER(HANDLE, VAR_NAME)         CGPU_RESOLVE_OR_RETURN(HANDLE, VAR_NAME, CgpuIShader, cgpuResolveShader)
#define CGPU_RESOLVE_OR_RETURN_PIPELINE(HANDLE, VAR_NAME)       CGPU_RESOLVE_OR_RETURN(HANDLE, VAR_NAME, CgpuIPipeline, cgpuResolvePipeline)
#define CGPU_RESOLVE_OR_RETURN_SEMAPHORE(HANDLE, VAR_NAME)      CGPU_RESOLVE_OR_RETURN(HANDLE, VAR_NAME, CgpuISemaphore, cgpuResolveSemaphore)
#define CGPU_RESOLVE_OR_RETURN_COMMAND_BUFFER(HANDLE, VAR_NAME) CGPU_RESOLVE_OR_RETURN(HANDLE, VAR_NAME, CgpuICommandBuffer, cgpuResolveCommandBuffer)
#define CGPU_RESOLVE_OR_RETURN_SAMPLER(HANDLE, VAR_NAME)        CGPU_RESOLVE_OR_RETURN(HANDLE, VAR_NAME, CgpuISampler, cgpuResolveSampler)
#define CGPU_RESOLVE_OR_RETURN_BLAS(HANDLE, VAR_NAME)           CGPU_RESOLVE_OR_RETURN(HANDLE, VAR_NAME, CgpuIBlas, cgpuResolveBlas)
#define CGPU_RESOLVE_OR_RETURN_TLAS(HANDLE, VAR_NAME)           CGPU_RESOLVE_OR_RETURN(HANDLE, VAR_NAME, CgpuITlas, cgpuResolveTlas)

  /* Helper methods. */

  static CgpuPhysicalDeviceFeatures cgpuTranslatePhysicalDeviceFeatures(const VkPhysicalDeviceFeatures* vkFeatures)
  {
    CgpuPhysicalDeviceFeatures features = {
      .textureCompressionBC = bool(vkFeatures->textureCompressionBC),
      .pipelineStatisticsQuery = bool(vkFeatures->pipelineStatisticsQuery),
      .shaderImageGatherExtended = bool(vkFeatures->shaderImageGatherExtended),
      .shaderStorageImageExtendedFormats = bool(vkFeatures->shaderStorageImageExtendedFormats),
      .shaderStorageImageReadWithoutFormat = bool(vkFeatures->shaderStorageImageReadWithoutFormat),
      .shaderStorageImageWriteWithoutFormat = bool(vkFeatures->shaderStorageImageWriteWithoutFormat),
      .shaderUniformBufferArrayDynamicIndexing = bool(vkFeatures->shaderUniformBufferArrayDynamicIndexing),
      .shaderSampledImageArrayDynamicIndexing = bool(vkFeatures->shaderSampledImageArrayDynamicIndexing),
      .shaderStorageBufferArrayDynamicIndexing = bool(vkFeatures->shaderStorageBufferArrayDynamicIndexing),
      .shaderStorageImageArrayDynamicIndexing = bool(vkFeatures->shaderStorageImageArrayDynamicIndexing),
      .shaderFloat64 = bool(vkFeatures->shaderFloat64),
      .shaderInt64 = bool(vkFeatures->shaderInt64),
      .shaderInt16 = bool(vkFeatures->shaderInt16),
      .sparseBinding = bool(vkFeatures->sparseBinding),
      .sparseResidencyBuffer = bool(vkFeatures->sparseResidencyBuffer),
      .sparseResidencyImage2D = bool(vkFeatures->sparseResidencyImage2D),
      .sparseResidencyImage3D = bool(vkFeatures->sparseResidencyImage3D),
      .sparseResidencyAliased = bool(vkFeatures->sparseResidencyAliased),
    };
    return features;
  }

  static CgpuPhysicalDeviceProperties cgpuTranslatePhysicalDeviceProperties(const VkPhysicalDeviceLimits* vkLimits,
                                                                            const VkPhysicalDeviceSubgroupProperties* vkSubgroupProps,
                                                                            const VkPhysicalDeviceAccelerationStructurePropertiesKHR* vkAsProps,
                                                                            const VkPhysicalDeviceRayTracingPipelinePropertiesKHR* vkRtPipelineProps)
  {
    CgpuPhysicalDeviceProperties properties = {
      .maxImageDimension1D = vkLimits->maxImageDimension1D,
      .maxImageDimension2D = vkLimits->maxImageDimension2D,
      .maxImageDimension3D = vkLimits->maxImageDimension3D,
      .maxImageDimensionCube = vkLimits->maxImageDimensionCube,
      .maxImageArrayLayers = vkLimits->maxImageArrayLayers,
      .maxUniformBufferRange = vkLimits->maxUniformBufferRange,
      .maxStorageBufferRange = vkLimits->maxStorageBufferRange,
      .maxPushConstantsSize = vkLimits->maxPushConstantsSize,
      .maxMemoryAllocationCount = vkLimits->maxMemoryAllocationCount,
      .maxSamplerAllocationCount = vkLimits->maxSamplerAllocationCount,
      .bufferImageGranularity = vkLimits->bufferImageGranularity,
      .sparseAddressSpaceSize = vkLimits->sparseAddressSpaceSize,
      .maxBoundDescriptorSets = vkLimits->maxBoundDescriptorSets,
      .maxPerStageDescriptorSamplers = vkLimits->maxPerStageDescriptorSamplers,
      .maxPerStageDescriptorUniformBuffers = vkLimits->maxPerStageDescriptorUniformBuffers,
      .maxPerStageDescriptorStorageBuffers = vkLimits->maxPerStageDescriptorStorageBuffers,
      .maxPerStageDescriptorSampledImages = vkLimits->maxPerStageDescriptorSampledImages,
      .maxPerStageDescriptorStorageImages = vkLimits->maxPerStageDescriptorStorageImages,
      .maxPerStageDescriptorInputAttachments = vkLimits->maxPerStageDescriptorInputAttachments,
      .maxPerStageResources = vkLimits->maxPerStageResources,
      .maxDescriptorSetSamplers = vkLimits->maxDescriptorSetSamplers,
      .maxDescriptorSetUniformBuffers = vkLimits->maxDescriptorSetUniformBuffers,
      .maxDescriptorSetUniformBuffersDynamic = vkLimits->maxDescriptorSetUniformBuffersDynamic,
      .maxDescriptorSetStorageBuffers = vkLimits->maxDescriptorSetStorageBuffers,
      .maxDescriptorSetStorageBuffersDynamic = vkLimits->maxDescriptorSetStorageBuffersDynamic,
      .maxDescriptorSetSampledImages = vkLimits->maxDescriptorSetSampledImages,
      .maxDescriptorSetStorageImages = vkLimits->maxDescriptorSetStorageImages,
      .maxDescriptorSetInputAttachments = vkLimits->maxDescriptorSetInputAttachments,
      .maxComputeSharedMemorySize = vkLimits->maxComputeSharedMemorySize,
      .maxComputeWorkGroupCount = { vkLimits->maxComputeWorkGroupCount[0], vkLimits->maxComputeWorkGroupCount[1], vkLimits->maxComputeWorkGroupCount[2] },
      .maxComputeWorkGroupInvocations = vkLimits->maxComputeWorkGroupInvocations,
      .maxComputeWorkGroupSize = { vkLimits->maxComputeWorkGroupSize[0], vkLimits->maxComputeWorkGroupSize[1], vkLimits->maxComputeWorkGroupSize[2] },
      .mipmapPrecisionBits = vkLimits->mipmapPrecisionBits,
      .maxSamplerLodBias = vkLimits->maxSamplerLodBias,
      .maxSamplerAnisotropy = vkLimits->maxSamplerAnisotropy,
      .minMemoryMapAlignment = vkLimits->minMemoryMapAlignment,
      .minUniformBufferOffsetAlignment = vkLimits->minUniformBufferOffsetAlignment,
      .minStorageBufferOffsetAlignment = vkLimits->minStorageBufferOffsetAlignment,
      .minTexelOffset = vkLimits->minTexelOffset,
      .maxTexelOffset = vkLimits->maxTexelOffset,
      .minTexelGatherOffset = vkLimits->minTexelGatherOffset,
      .maxTexelGatherOffset = vkLimits->maxTexelGatherOffset,
      .minInterpolationOffset = vkLimits->minInterpolationOffset,
      .maxInterpolationOffset = vkLimits->maxInterpolationOffset,
      .subPixelInterpolationOffsetBits = vkLimits->subPixelInterpolationOffsetBits,
      .maxSampleMaskWords = vkLimits->maxSampleMaskWords,
      .timestampComputeAndGraphics = bool(vkLimits->timestampComputeAndGraphics),
      .timestampPeriod = vkLimits->timestampPeriod,
      .discreteQueuePriorities = vkLimits->discreteQueuePriorities,
      .optimalBufferCopyOffsetAlignment = vkLimits->optimalBufferCopyOffsetAlignment,
      .optimalBufferCopyRowPitchAlignment = vkLimits->optimalBufferCopyRowPitchAlignment,
      .nonCoherentAtomSize = vkLimits->nonCoherentAtomSize,
      .subgroupSize = vkSubgroupProps->subgroupSize,
      .minAccelerationStructureScratchOffsetAlignment = vkAsProps->minAccelerationStructureScratchOffsetAlignment,
      .shaderGroupHandleSize = vkRtPipelineProps->shaderGroupHandleSize,
      .maxShaderGroupStride = vkRtPipelineProps->maxShaderGroupStride,
      .shaderGroupBaseAlignment = vkRtPipelineProps->shaderGroupBaseAlignment,
      .shaderGroupHandleCaptureReplaySize = vkRtPipelineProps->shaderGroupHandleCaptureReplaySize,
      .maxRayDispatchInvocationCount = vkRtPipelineProps->maxRayDispatchInvocationCount,
      .shaderGroupHandleAlignment = vkRtPipelineProps->shaderGroupHandleAlignment,
      .maxRayHitAttributeSize = vkRtPipelineProps->maxRayHitAttributeSize,
    };
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

  static VkPipelineStageFlags2KHR cgpuPipelineStageFlagsFromShaderStageFlags(VkShaderStageFlags shaderStageFlags)
  {
    VkPipelineStageFlags2KHR pipelineStageFlags = VK_PIPELINE_STAGE_2_NONE_KHR;

    if (shaderStageFlags & VK_SHADER_STAGE_COMPUTE_BIT)
    {
      pipelineStageFlags |= VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR;
    }

    if ((shaderStageFlags & VK_SHADER_STAGE_RAYGEN_BIT_KHR) |
        (shaderStageFlags & VK_SHADER_STAGE_ANY_HIT_BIT_KHR) |
        (shaderStageFlags & VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR) |
        (shaderStageFlags & VK_SHADER_STAGE_MISS_BIT_KHR) |
        (shaderStageFlags & VK_SHADER_STAGE_INTERSECTION_BIT_KHR))
    {
      pipelineStageFlags |= VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
    }

    assert(pipelineStageFlags != VK_PIPELINE_STAGE_2_NONE_KHR);
    return pipelineStageFlags;
  }

  static const char* cgpuGetVendorName(uint32_t deviceId)
  {
    switch (deviceId)
    {
    case 0x1002:
      return "AMD";
    case 0x10DE:
      return "NVIDIA";
    case 0x8086:
      return "INTEL";
    default:
      return "UNKNOWN";
    }
  }

  /* API method implementation. */

#ifndef NDEBUG
  static bool cgpuFindLayer(const char* name, size_t layerCount, VkLayerProperties* layers)
  {
    for (size_t i = 0; i < layerCount; i++)
    {
      if (!strcmp(layers[i].layerName, name))
      {
        return true;
      }
    }
    return false;
  }
#endif

  static bool cgpuFindExtension(const char* name, size_t extensionCount, VkExtensionProperties* extensions)
  {
    for (size_t i = 0; i < extensionCount; ++i)
    {
      if (!strcmp(extensions[i].extensionName, name))
      {
        return true;
      }
    }
    return false;
  }

  static void cgpuSetObjectName(VkDevice device, VkObjectType type, uint64_t handle, const char* name)
  {
    VkDebugUtilsObjectNameInfoEXT info = {
      .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
      .pNext = nullptr,
      .objectType = type,
      .objectHandle = handle,
      .pObjectName = name
    };

    [[maybe_unused]] VkResult result = vkSetDebugUtilsObjectNameEXT(device, &info);

    assert(result == VK_SUCCESS);
  }

  bool cgpuInitialize(const char* appName, uint32_t versionMajor, uint32_t versionMinor, uint32_t versionPatch)
  {
    if (volkInitialize() != VK_SUCCESS)
    {
      CGPU_RETURN_ERROR("failed to initialize volk");
    }

    uint32_t instanceVersion = volkGetInstanceVersion();
    GB_LOG("Vulkan instance version {}.{}.{}", VK_VERSION_MAJOR(instanceVersion),
      VK_VERSION_MINOR(instanceVersion), VK_VERSION_PATCH(instanceVersion));

    if (instanceVersion < CGPU_MIN_VK_API_VERSION)
    {
      GB_ERROR("Vulkan instance version does match minimum of {}.{}.{}",
        VK_VERSION_MAJOR(CGPU_MIN_VK_API_VERSION), VK_VERSION_MINOR(CGPU_MIN_VK_API_VERSION),
        VK_VERSION_PATCH(CGPU_MIN_VK_API_VERSION));
      return false;
    }

    GbSmallVector<const char*, 8> enabledLayers;
    GbSmallVector<const char*, 8> enabledExtensions;
    bool debugUtilsEnabled = false;
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
        GB_LOG("> enabled layer {}", VK_LAYER_KHRONOS_VALIDATION_NAME);
      }
    }
#endif

    {
      uint32_t extensionCount;
      vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);

      GbSmallVector<VkExtensionProperties, 512> availableExtensions(extensionCount);
      vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, availableExtensions.data());

#ifndef NDEBUG
      if (cgpuFindExtension(VK_EXT_DEBUG_UTILS_EXTENSION_NAME, availableExtensions.size(), availableExtensions.data()))
      {
        enabledExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        GB_LOG("> enabled instance extension {}", VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

        debugUtilsEnabled = true;
      }
#endif

      if (cgpuFindExtension(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME, availableExtensions.size(), availableExtensions.data()))
      {
        enabledExtensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        GB_LOG("> enabled instance extension {}", VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
      }
    }

    uint32_t versionVariant = 0;
    VkApplicationInfo appInfo = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pNext = nullptr,
      .pApplicationName = appName,
      .applicationVersion = VK_MAKE_API_VERSION(versionVariant, versionMajor, versionMinor, versionPatch),
      .pEngineName = appName,
      .engineVersion = VK_MAKE_API_VERSION(versionVariant, versionMajor, versionMinor, versionPatch),
      .apiVersion = CGPU_MIN_VK_API_VERSION,
    };

    VkInstanceCreateInfo createInfo = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pNext = nullptr,
      .flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR,
      .pApplicationInfo = &appInfo,
      .enabledLayerCount = (uint32_t) enabledLayers.size(),
      .ppEnabledLayerNames = enabledLayers.data(),
      .enabledExtensionCount = (uint32_t) enabledExtensions.size(),
      .ppEnabledExtensionNames = enabledExtensions.data(),
    };

    VkInstance instance;
    {
      VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);

      if (result != VK_SUCCESS)
      {
        GB_ERROR("{}:{}: failed to create Vulkan instance (code: {})", __FILE__, __LINE__, int(result));
        return false;
      }
    }

    volkLoadInstanceOnly(instance);

    iinstance = std::make_unique<CgpuIInstance>();
    iinstance->instance = instance;
    iinstance->debugUtilsEnabled = debugUtilsEnabled;
    return true;
  }

  void cgpuTerminate()
  {
    vkDestroyInstance(iinstance->instance, nullptr);
    iinstance.reset();
  }

  bool cgpuCreateDevice(CgpuDevice* device)
  {
    uint64_t handle = iinstance->ideviceStore.allocate();

    CGPU_RESOLVE_OR_RETURN_DEVICE({ handle }, idevice);

    uint32_t physDeviceCount;
    vkEnumeratePhysicalDevices(
      iinstance->instance,
      &physDeviceCount,
      nullptr
    );

    if (physDeviceCount == 0)
    {
      iinstance->ideviceStore.free(handle);
      CGPU_RETURN_ERROR("no physical device found");
    }
    else if (physDeviceCount > 1)
    {
      GB_WARN("more than one device found -- choosing first one");
    }

    GbSmallVector<VkPhysicalDevice, 8> physicalDevices(physDeviceCount);
    vkEnumeratePhysicalDevices(iinstance->instance, &physDeviceCount, physicalDevices.data());

    idevice->physicalDevice = physicalDevices[0];

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

    GB_LOG("Vulkan device properties:");
    uint32_t apiVersion = deviceProperties.properties.apiVersion;
    {
      uint32_t major = VK_VERSION_MAJOR(apiVersion);
      uint32_t minor = VK_VERSION_MINOR(apiVersion);
      uint32_t patch = VK_VERSION_PATCH(apiVersion);
      GB_LOG("> API version: {}.{}.{}", major, minor, patch);
    }

    GB_LOG("> name: {}", deviceProperties.properties.deviceName);
    GB_LOG("> vendor: {} ({})", cgpuGetVendorName(deviceProperties.properties.vendorID), deviceProperties.properties.vendorID);

    if (apiVersion < CGPU_MIN_VK_API_VERSION)
    {
      iinstance->ideviceStore.free(handle);

      GB_ERROR("Vulkan device API version does match minimum of {}.{}.{}",
        VK_VERSION_MAJOR(CGPU_MIN_VK_API_VERSION), VK_VERSION_MINOR(CGPU_MIN_VK_API_VERSION),
        VK_VERSION_PATCH(CGPU_MIN_VK_API_VERSION));

      return false;
    }

    const VkPhysicalDeviceLimits* limits = &deviceProperties.properties.limits;
    idevice->properties = cgpuTranslatePhysicalDeviceProperties(limits, &subgroupProperties, &asProperties, &rtPipelineProperties);

    uint32_t extensionCount;
    vkEnumerateDeviceExtensionProperties(idevice->physicalDevice, nullptr, &extensionCount, nullptr);

    GbSmallVector<VkExtensionProperties, 1024> extensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(idevice->physicalDevice, nullptr, &extensionCount, extensions.data());

    GbSmallVector<const char*, 16> requiredExtensions = {
      VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
      VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME, // required by VK_KHR_acceleration_structure
      VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME, // required by VK_KHR_acceleration_structure
      VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME, // required by VK_KHR_acceleration_structure
      VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
      VK_KHR_SPIRV_1_4_EXTENSION_NAME, // required by VK_KHR_ray_tracing_pipeline
      VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME, // required by VK_KHR_spirv_1_4
      VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME,
      VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
      VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME
    };

    GbSmallVector<const char*, 32> enabledExtensions;
    for (uint32_t i = 0; i < requiredExtensions.size(); i++)
    {
      const char* extension = requiredExtensions[i];

      if (!cgpuFindExtension(extension, extensionCount, extensions.data()))
      {
        iinstance->ideviceStore.free(handle);

        GB_ERROR("extension {} not supported", extension);
        return false;
      }

      enabledExtensions.push_back(extension);
    }

    const auto enableOptionalExtension = [&](const char* extName)
    {
      if (!cgpuFindExtension(extName, extensions.size(), extensions.data()))
      {
        return false;
      }

      enabledExtensions.push_back(extName);

      GB_LOG("extension {} enabled", extName);
      return true;
    };

    if (enableOptionalExtension(VK_EXT_MEMORY_PRIORITY_EXTENSION_NAME) &&
        enableOptionalExtension(VK_EXT_PAGEABLE_DEVICE_LOCAL_MEMORY_EXTENSION_NAME))
    {
      idevice->features.pageableDeviceLocalMemory = true;
    }

    const char* VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME = "VK_KHR_portability_subset";
    enableOptionalExtension(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME);

#ifndef NDEBUG
    if (features.shaderInt64 && enableOptionalExtension(VK_KHR_SHADER_CLOCK_EXTENSION_NAME))
    {
      idevice->features.shaderClock = true;
    }

#ifndef __APPLE__
    if (enableOptionalExtension(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME))
    {
      idevice->features.debugPrintf = true;
    }
#endif
#endif

    if (enableOptionalExtension(VK_NV_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME))
    {
      idevice->features.rayTracingInvocationReorder = true;
    }

#ifndef NDEBUG
    // This feature requires env var NV_ALLOW_RAYTRACING_VALIDATION=1 to be set.
    if (iinstance->debugUtilsEnabled && enableOptionalExtension(VK_NV_RAY_TRACING_VALIDATION_EXTENSION_NAME))
    {
      idevice->features.rayTracingValidation = true;
    }
#endif

    uint32_t queueFamilyCount;
    vkGetPhysicalDeviceQueueFamilyProperties(idevice->physicalDevice, &queueFamilyCount, nullptr);

    GbSmallVector<VkQueueFamilyProperties, 32> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(idevice->physicalDevice, &queueFamilyCount, queueFamilies.data());

    uint32_t queueFamilyIndex = UINT32_MAX;
    for (uint32_t i = 0; i < queueFamilyCount; ++i)
    {
      const VkQueueFamilyProperties* queue_family = &queueFamilies[i];

      if ((queue_family->queueFlags & VK_QUEUE_COMPUTE_BIT) && (queue_family->queueFlags & VK_QUEUE_TRANSFER_BIT))
      {
        queueFamilyIndex = i;
      }
    }
    if (queueFamilyIndex == UINT32_MAX)
    {
      iinstance->ideviceStore.free(handle);
      CGPU_RETURN_ERROR("no suitable queue family");
    }

    void* pNext = nullptr;

    VkPhysicalDevicePageableDeviceLocalMemoryFeaturesEXT pageableMemoryFeatures = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PAGEABLE_DEVICE_LOCAL_MEMORY_FEATURES_EXT,
      .pNext = pNext,
      .pageableDeviceLocalMemory = VK_TRUE,
    };

    if (idevice->features.pageableDeviceLocalMemory)
    {
      pNext = &pageableMemoryFeatures;
    }

    VkPhysicalDeviceShaderClockFeaturesKHR shaderClockFeatures = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR,
      .pNext = pNext,
      .shaderSubgroupClock = VK_TRUE,
      .shaderDeviceClock = VK_FALSE,
    };

    if (idevice->features.shaderClock)
    {
      pNext = &shaderClockFeatures;
    }

    VkPhysicalDeviceRayTracingValidationFeaturesNV rayTracingValidationFeatures = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_VALIDATION_FEATURES_NV,
      .pNext = pNext,
      .rayTracingValidation = VK_TRUE
    };

    if (idevice->features.rayTracingValidation)
    {
      pNext = &rayTracingValidationFeatures;
    }

    VkPhysicalDeviceRayTracingInvocationReorderFeaturesNV invocationReorderFeatures = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_FEATURES_NV,
      .pNext = pNext,
      .rayTracingInvocationReorder = VK_TRUE,
    };

    if (idevice->features.rayTracingInvocationReorder)
    {
      pNext = &invocationReorderFeatures;
    }

    VkPhysicalDeviceTimelineSemaphoreFeaturesKHR timelineSemaphoreFeatures = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES,
      .pNext = pNext,
      .timelineSemaphore = VK_TRUE
    };

    VkPhysicalDeviceSynchronization2FeaturesKHR synchronization2Features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR,
      .pNext = &timelineSemaphoreFeatures,
      .synchronization2 = VK_TRUE
    };

    VkPhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructureFeatures = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
      .pNext = &synchronization2Features,
      .accelerationStructure = VK_TRUE,
      .accelerationStructureCaptureReplay = VK_FALSE,
      .accelerationStructureIndirectBuild = VK_FALSE,
      .accelerationStructureHostCommands = VK_FALSE,
      .descriptorBindingAccelerationStructureUpdateAfterBind = VK_FALSE,
    };

    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rayTracingPipelineFeatures = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR,
      .pNext = &accelerationStructureFeatures,
      .rayTracingPipeline = VK_TRUE,
      .rayTracingPipelineShaderGroupHandleCaptureReplay = VK_FALSE,
      .rayTracingPipelineShaderGroupHandleCaptureReplayMixed = VK_FALSE,
      .rayTracingPipelineTraceRaysIndirect = VK_FALSE,
      .rayTraversalPrimitiveCulling = VK_FALSE,
    };

    VkPhysicalDeviceBufferDeviceAddressFeaturesKHR bufferDeviceAddressFeatures = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES,
      .pNext = &rayTracingPipelineFeatures,
      .bufferDeviceAddress = VK_TRUE,
      .bufferDeviceAddressCaptureReplay = VK_FALSE,
      .bufferDeviceAddressMultiDevice = VK_FALSE,
    };

    VkPhysicalDeviceDescriptorIndexingFeaturesEXT descriptorIndexingFeatures = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES,
      .pNext = &bufferDeviceAddressFeatures,
      .shaderInputAttachmentArrayDynamicIndexing = VK_FALSE,
      .shaderUniformTexelBufferArrayDynamicIndexing = VK_FALSE,
      .shaderStorageTexelBufferArrayDynamicIndexing = VK_FALSE,
      .shaderUniformBufferArrayNonUniformIndexing = VK_FALSE,
      .shaderSampledImageArrayNonUniformIndexing = VK_TRUE,
      .shaderStorageBufferArrayNonUniformIndexing = VK_FALSE,
      .shaderStorageImageArrayNonUniformIndexing = VK_TRUE,
      .shaderInputAttachmentArrayNonUniformIndexing = VK_FALSE,
      .shaderUniformTexelBufferArrayNonUniformIndexing = VK_FALSE,
      .shaderStorageTexelBufferArrayNonUniformIndexing = VK_FALSE,
      .descriptorBindingUniformBufferUpdateAfterBind = VK_FALSE,
      .descriptorBindingSampledImageUpdateAfterBind = VK_FALSE,
      .descriptorBindingStorageImageUpdateAfterBind = VK_FALSE,
      .descriptorBindingStorageBufferUpdateAfterBind = VK_FALSE,
      .descriptorBindingUniformTexelBufferUpdateAfterBind = VK_FALSE,
      .descriptorBindingStorageTexelBufferUpdateAfterBind = VK_FALSE,
      .descriptorBindingUpdateUnusedWhilePending = VK_FALSE,
      .descriptorBindingPartiallyBound = VK_FALSE,
      .descriptorBindingVariableDescriptorCount = VK_FALSE,
      .runtimeDescriptorArray = VK_FALSE,
    };

    VkPhysicalDeviceShaderFloat16Int8Features shaderFloat16Int8Features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES,
      .pNext = &descriptorIndexingFeatures,
      .shaderFloat16 = VK_TRUE,
      .shaderInt8 = VK_FALSE,
    };

    VkPhysicalDevice16BitStorageFeatures device16bitStorageFeatures = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES,
      .pNext = &shaderFloat16Int8Features,
      .storageBuffer16BitAccess = VK_TRUE,
      .uniformAndStorageBuffer16BitAccess = VK_TRUE,
      .storagePushConstant16 = VK_FALSE,
      .storageInputOutput16 = VK_FALSE,
    };

    VkPhysicalDeviceFeatures2 deviceFeatures2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &device16bitStorageFeatures,
      .features = {
        .robustBufferAccess = VK_FALSE,
        .fullDrawIndexUint32 = VK_FALSE,
        .imageCubeArray = VK_FALSE,
        .independentBlend = VK_FALSE,
        .geometryShader = VK_FALSE,
        .tessellationShader = VK_FALSE,
        .sampleRateShading = VK_FALSE,
        .dualSrcBlend = VK_FALSE,
        .logicOp = VK_FALSE,
        .multiDrawIndirect = VK_FALSE,
        .drawIndirectFirstInstance = VK_FALSE,
        .depthClamp = VK_FALSE,
        .depthBiasClamp = VK_FALSE,
        .fillModeNonSolid = VK_FALSE,
        .depthBounds = VK_FALSE,
        .wideLines = VK_FALSE,
        .largePoints = VK_FALSE,
        .alphaToOne = VK_FALSE,
        .multiViewport = VK_FALSE,
        .samplerAnisotropy = VK_TRUE,
        .textureCompressionETC2 = VK_FALSE,
        .textureCompressionASTC_LDR = VK_FALSE,
        .textureCompressionBC = VK_FALSE,
        .occlusionQueryPrecise = VK_FALSE,
        .pipelineStatisticsQuery = VK_FALSE,
        .vertexPipelineStoresAndAtomics = VK_FALSE,
        .fragmentStoresAndAtomics = VK_FALSE,
        .shaderTessellationAndGeometryPointSize = VK_FALSE,
        .shaderImageGatherExtended = VK_TRUE,
        .shaderStorageImageExtendedFormats = VK_FALSE,
        .shaderStorageImageMultisample = VK_FALSE,
        .shaderStorageImageReadWithoutFormat = VK_FALSE,
        .shaderStorageImageWriteWithoutFormat = VK_FALSE,
        .shaderUniformBufferArrayDynamicIndexing = VK_FALSE,
        .shaderSampledImageArrayDynamicIndexing = VK_TRUE,
        .shaderStorageBufferArrayDynamicIndexing = VK_FALSE,
        .shaderStorageImageArrayDynamicIndexing = VK_FALSE,
        .shaderClipDistance = VK_FALSE,
        .shaderCullDistance = VK_FALSE,
        .shaderFloat64 = VK_FALSE,
        .shaderInt64 = idevice->features.shaderClock,
        .shaderInt16 = VK_TRUE,
        .shaderResourceResidency = VK_FALSE,
        .shaderResourceMinLod = VK_FALSE,
        .sparseBinding = VK_FALSE,
        .sparseResidencyBuffer = VK_FALSE,
        .sparseResidencyImage2D = VK_FALSE,
        .sparseResidencyImage3D = VK_FALSE,
        .sparseResidency2Samples = VK_FALSE,
        .sparseResidency4Samples = VK_FALSE,
        .sparseResidency8Samples = VK_FALSE,
        .sparseResidency16Samples = VK_FALSE,
        .sparseResidencyAliased = VK_FALSE,
        .variableMultisampleRate = VK_FALSE,
        .inheritedQueries = VK_FALSE,
      }
    };

    const float queuePriority = 1.0f;

    VkDeviceQueueCreateInfo queueCreateInfo = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .queueFamilyIndex = (uint32_t) queueFamilyIndex,
      .queueCount = 1,
      .pQueuePriorities = &queuePriority,
    };

    VkDeviceCreateInfo deviceCreateInfo = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &deviceFeatures2,
      .flags = 0,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &queueCreateInfo,
      /* These two fields are ignored by up-to-date implementations since
       * nowadays, there is no difference to instance validation layers. */
      .enabledLayerCount = 0,
      .ppEnabledLayerNames = nullptr,
      .enabledExtensionCount = (uint32_t) enabledExtensions.size(),
      .ppEnabledExtensionNames = enabledExtensions.data(),
      .pEnabledFeatures = nullptr,
    };

    VkResult result = vkCreateDevice(
      idevice->physicalDevice,
      &deviceCreateInfo,
      nullptr,
      &idevice->logicalDevice
    );
    if (result != VK_SUCCESS) {
      iinstance->ideviceStore.free(handle);
      CGPU_RETURN_ERROR("failed to create device");
    }

    volkLoadDeviceTable(&idevice->table, idevice->logicalDevice);

    idevice->table.vkGetDeviceQueue(
      idevice->logicalDevice,
      queueFamilyIndex,
      0,
      &idevice->computeQueue
    );

    VkCommandPoolCreateInfo poolCreateInfo = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .pNext = nullptr,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      .queueFamilyIndex = queueFamilyIndex,
    };

    result = idevice->table.vkCreateCommandPool(
      idevice->logicalDevice,
      &poolCreateInfo,
      nullptr,
      &idevice->commandPool
    );

    if (result != VK_SUCCESS)
    {
      iinstance->ideviceStore.free(handle);

      idevice->table.vkDestroyDevice(idevice->logicalDevice, nullptr);

      CGPU_RETURN_ERROR("failed to create command pool");
    }

    VkQueryPoolCreateInfo timestampPoolCreateInfo = {
      .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .queryType = VK_QUERY_TYPE_TIMESTAMP,
      .queryCount = CGPU_MAX_TIMESTAMP_QUERIES,
      .pipelineStatistics = 0,
    };

    result = idevice->table.vkCreateQueryPool(
      idevice->logicalDevice,
      &timestampPoolCreateInfo,
      nullptr,
      &idevice->timestamp_pool
    );

    if (result != VK_SUCCESS)
    {
      iinstance->ideviceStore.free(handle);

      idevice->table.vkDestroyCommandPool(idevice->logicalDevice, idevice->commandPool, nullptr);
      idevice->table.vkDestroyDevice(idevice->logicalDevice, nullptr);

      CGPU_RETURN_ERROR("failed to create query pool");
    }

    VmaVulkanFunctions vmaVulkanFunctions = {
      .vkGetInstanceProcAddr = nullptr,
      .vkGetDeviceProcAddr = nullptr,
      .vkGetPhysicalDeviceProperties = vkGetPhysicalDeviceProperties,
      .vkGetPhysicalDeviceMemoryProperties = vkGetPhysicalDeviceMemoryProperties,
      .vkAllocateMemory = idevice->table.vkAllocateMemory,
      .vkFreeMemory = idevice->table.vkFreeMemory,
      .vkMapMemory = idevice->table.vkMapMemory,
      .vkUnmapMemory = idevice->table.vkUnmapMemory,
      .vkFlushMappedMemoryRanges = idevice->table.vkFlushMappedMemoryRanges,
      .vkInvalidateMappedMemoryRanges = idevice->table.vkInvalidateMappedMemoryRanges,
      .vkBindBufferMemory = idevice->table.vkBindBufferMemory,
      .vkBindImageMemory = idevice->table.vkBindImageMemory,
      .vkGetBufferMemoryRequirements = idevice->table.vkGetBufferMemoryRequirements,
      .vkGetImageMemoryRequirements = idevice->table.vkGetImageMemoryRequirements,
      .vkCreateBuffer = idevice->table.vkCreateBuffer,
      .vkDestroyBuffer = idevice->table.vkDestroyBuffer,
      .vkCreateImage = idevice->table.vkCreateImage,
      .vkDestroyImage = idevice->table.vkDestroyImage,
      .vkCmdCopyBuffer = idevice->table.vkCmdCopyBuffer,
      .vkGetBufferMemoryRequirements2KHR = idevice->table.vkGetBufferMemoryRequirements2,
      .vkGetImageMemoryRequirements2KHR = idevice->table.vkGetImageMemoryRequirements2,
      .vkBindBufferMemory2KHR = idevice->table.vkBindBufferMemory2,
      .vkBindImageMemory2KHR = idevice->table.vkBindImageMemory2,
      .vkGetPhysicalDeviceMemoryProperties2KHR = vkGetPhysicalDeviceMemoryProperties2,
      .vkGetDeviceBufferMemoryRequirements = vkGetDeviceBufferMemoryRequirements,
      .vkGetDeviceImageMemoryRequirements = vkGetDeviceImageMemoryRequirements,
    };

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
      iinstance->ideviceStore.free(handle);

      idevice->table.vkDestroyQueryPool(idevice->logicalDevice, idevice->timestamp_pool, nullptr);
      idevice->table.vkDestroyCommandPool(idevice->logicalDevice, idevice->commandPool, nullptr);
      idevice->table.vkDestroyDevice(idevice->logicalDevice, nullptr);

      CGPU_RETURN_ERROR("failed to create vma allocator");
    }

    VkPipelineCacheCreateInfo cacheCreateInfo = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .initialDataSize = 0,
      .pInitialData = nullptr
    };

    result = idevice->table.vkCreatePipelineCache(
      idevice->logicalDevice,
      &cacheCreateInfo,
      nullptr,
      &idevice->pipelineCache
    );

    if (result != VK_SUCCESS)
    {
      CGPU_RETURN_ERROR("failed to create pipeline cache");

      idevice->pipelineCache = VK_NULL_HANDLE;
    }

    device->handle = handle;
    return true;
  }

  bool cgpuDestroyDevice(CgpuDevice device)
  {
    CGPU_RESOLVE_OR_RETURN_DEVICE(device, idevice);

    if (idevice->pipelineCache != VK_NULL_HANDLE)
    {
      idevice->table.vkDestroyPipelineCache(idevice->logicalDevice, idevice->pipelineCache, nullptr);
    }

    idevice->table.vkDestroyQueryPool(idevice->logicalDevice, idevice->timestamp_pool, nullptr);
    idevice->table.vkDestroyCommandPool(idevice->logicalDevice, idevice->commandPool, nullptr);

    vmaDestroyAllocator(idevice->allocator);

    idevice->table.vkDestroyDevice(idevice->logicalDevice, nullptr);

    iinstance->ideviceStore.free(device.handle);
    return true;
  }

  bool cgpuCreateShader(CgpuDevice device,
                        CgpuShaderCreateInfo createInfo,
                        CgpuShader* shader)
  {
    CGPU_RESOLVE_OR_RETURN_DEVICE(device, idevice);

    uint64_t handle = iinstance->ishaderStore.allocate();

    CGPU_RESOLVE_OR_RETURN_SHADER({ handle }, ishader);

    VkShaderModuleCreateInfo shaderModuleCreateInfo = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .codeSize = createInfo.size,
      .pCode = (uint32_t*) createInfo.source,
    };

    VkResult result = idevice->table.vkCreateShaderModule(
      idevice->logicalDevice,
      &shaderModuleCreateInfo,
      nullptr,
      &ishader->module
    );
    if (result != VK_SUCCESS) {
      iinstance->ishaderStore.free(handle);
      CGPU_RETURN_ERROR("failed to create shader module");
    }

    if (!cgpuReflectShader((uint32_t*) createInfo.source, createInfo.size, &ishader->reflection))
    {
      idevice->table.vkDestroyShaderModule(
        idevice->logicalDevice,
        ishader->module,
        nullptr
      );
      iinstance->ishaderStore.free(handle);
      CGPU_RETURN_ERROR("failed to reflect shader");
    }

    if (iinstance->debugUtilsEnabled && createInfo.debugName)
    {
      cgpuSetObjectName(idevice->logicalDevice, VK_OBJECT_TYPE_SHADER_MODULE, (uint64_t) ishader->module, createInfo.debugName);
    }

    ishader->stageFlags = (VkShaderStageFlagBits) createInfo.stageFlags;

    shader->handle = handle;
    return true;
  }

  bool cgpuDestroyShader(CgpuDevice device, CgpuShader shader)
  {
    CGPU_RESOLVE_OR_RETURN_DEVICE(device, idevice);
    CGPU_RESOLVE_OR_RETURN_SHADER(shader, ishader);

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
                                       CgpuIBuffer* ibuffer,
                                       const char* debugName)
  {
    VkBufferCreateInfo bufferInfo = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .size = size,
      .usage = (VkBufferUsageFlags) usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .queueFamilyIndexCount = 0,
      .pQueueFamilyIndices = nullptr,
    };

    VmaAllocationCreateInfo vmaAllocCreateInfo = {};
    vmaAllocCreateInfo.requiredFlags = (VkMemoryPropertyFlags) memoryProperties;

    VkResult result;
    if (alignment > 0)
    {
      result = vmaCreateBufferWithAlignment(
        idevice->allocator,
        &bufferInfo,
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
        &bufferInfo,
        &vmaAllocCreateInfo,
        &ibuffer->buffer,
        &ibuffer->allocation,
        nullptr
      );
    }

    if (debugName)
    {
      vmaSetAllocationName(idevice->allocator, ibuffer->allocation, debugName);
    }

    if (result != VK_SUCCESS)
    {
      CGPU_RETURN_ERROR("failed to create buffer");
    }

    ibuffer->size = size;

    return true;
  }

  static bool cgpuCreateBufferAligned(CgpuDevice device,
                                      CgpuBufferCreateInfo createInfo,
                                      uint64_t alignment,
                                      CgpuBuffer* buffer)
  {
    CGPU_RESOLVE_OR_RETURN_DEVICE(device, idevice);

    uint64_t handle = iinstance->ibufferStore.allocate();

    CGPU_RESOLVE_OR_RETURN_BUFFER({ handle }, ibuffer);

    assert(createInfo.size > 0);

    if (!cgpuCreateIBufferAligned(idevice, createInfo.usage, createInfo.memoryProperties, createInfo.size, alignment, ibuffer, createInfo.debugName))
    {
      iinstance->ibufferStore.free(handle);
      CGPU_RETURN_ERROR("failed to create buffer");
    }

    if (iinstance->debugUtilsEnabled && createInfo.debugName)
    {
      cgpuSetObjectName(idevice->logicalDevice, VK_OBJECT_TYPE_BUFFER, (uint64_t) ibuffer->buffer, createInfo.debugName);
    }

    buffer->handle = handle;
    return true;
  }

  bool cgpuCreateBuffer(CgpuDevice device,
                        CgpuBufferCreateInfo createInfo,
                        CgpuBuffer* buffer)
  {
    uint64_t alignment = 0;

    return cgpuCreateBufferAligned(device, createInfo, alignment, buffer);
  }

  static void cgpuDestroyIBuffer(CgpuIDevice* idevice, CgpuIBuffer* ibuffer)
  {
    vmaDestroyBuffer(idevice->allocator, ibuffer->buffer, ibuffer->allocation);
  }

  bool cgpuDestroyBuffer(CgpuDevice device, CgpuBuffer buffer)
  {
    CGPU_RESOLVE_OR_RETURN_DEVICE(device, idevice);
    CGPU_RESOLVE_OR_RETURN_BUFFER(buffer, ibuffer);

    cgpuDestroyIBuffer(idevice, ibuffer);

    iinstance->ibufferStore.free(buffer.handle);

    return true;
  }

  bool cgpuMapBuffer(CgpuDevice device, CgpuBuffer buffer, void** mappedMem)
  {
    CGPU_RESOLVE_OR_RETURN_DEVICE(device, idevice);
    CGPU_RESOLVE_OR_RETURN_BUFFER(buffer, ibuffer);

    if (vmaMapMemory(idevice->allocator, ibuffer->allocation, mappedMem) != VK_SUCCESS) {
      CGPU_RETURN_ERROR("failed to map buffer memory");
    }
    return true;
  }

  bool cgpuUnmapBuffer(CgpuDevice device, CgpuBuffer buffer)
  {
    CGPU_RESOLVE_OR_RETURN_DEVICE(device, idevice);
    CGPU_RESOLVE_OR_RETURN_BUFFER(buffer, ibuffer);

    vmaUnmapMemory(idevice->allocator, ibuffer->allocation);
    return true;
  }

  static VkDeviceAddress cgpuGetBufferDeviceAddress(CgpuIDevice* idevice, CgpuIBuffer* ibuffer)
  {
    VkBufferDeviceAddressInfoKHR addressInfo = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
      .pNext = nullptr,
      .buffer = ibuffer->buffer,
    };
    return idevice->table.vkGetBufferDeviceAddressKHR(idevice->logicalDevice, &addressInfo);
  }

  uint64_t cgpuGetBufferAddress(CgpuDevice device, CgpuBuffer buffer)
  {
    CGPU_RESOLVE_OR_RETURN_DEVICE(device, idevice);
    CGPU_RESOLVE_OR_RETURN_BUFFER(buffer, ibuffer);

    static_assert(sizeof(uint64_t) == sizeof(VkDeviceAddress));
    return uint64_t(cgpuGetBufferDeviceAddress(idevice, ibuffer));
  }

  bool cgpuCreateImage(CgpuDevice device,
                       CgpuImageCreateInfo createInfo,
                       CgpuImage* image)
  {
    CGPU_RESOLVE_OR_RETURN_DEVICE(device, idevice);

    uint64_t handle = iinstance->iimageStore.allocate();

    CGPU_RESOLVE_OR_RETURN_IMAGE({ handle }, iimage);

    // FIXME: check device support
    VkImageTiling vkImageTiling = VK_IMAGE_TILING_OPTIMAL;
    if (!createInfo.is3d && ((createInfo.usage & CGPU_IMAGE_USAGE_FLAG_TRANSFER_SRC) | (createInfo.usage & CGPU_IMAGE_USAGE_FLAG_TRANSFER_DST)))
    {
      vkImageTiling = VK_IMAGE_TILING_LINEAR;
    }

    VkImageCreateInfo imageCreateInfo = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .imageType = createInfo.is3d ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D,
      .format = (VkFormat) createInfo.format,
      .extent = {
        .width = createInfo.width,
        .height = createInfo.height,
        .depth = createInfo.is3d ? createInfo.depth : 1,
      },
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = vkImageTiling,
      .usage = (VkImageUsageFlags) createInfo.usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .queueFamilyIndexCount = 0,
      .pQueueFamilyIndices = nullptr,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };

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
      iinstance->iimageStore.free(handle);
      CGPU_RETURN_ERROR("failed to create image");
    }

    if (createInfo.debugName)
    {
      vmaSetAllocationName(idevice->allocator, iimage->allocation, createInfo.debugName);
    }

    VmaAllocationInfo allocationInfo;
    vmaGetAllocationInfo(idevice->allocator, iimage->allocation, &allocationInfo);

    iimage->size = allocationInfo.size;

    VkImageViewCreateInfo imageViewCreateInfo = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .image = iimage->image,
      .viewType = createInfo.is3d ? VK_IMAGE_VIEW_TYPE_3D : VK_IMAGE_VIEW_TYPE_2D,
      .format = (VkFormat) createInfo.format,
      .components = {
        .r = VK_COMPONENT_SWIZZLE_IDENTITY,
        .g = VK_COMPONENT_SWIZZLE_IDENTITY,
        .b = VK_COMPONENT_SWIZZLE_IDENTITY,
        .a = VK_COMPONENT_SWIZZLE_IDENTITY,
      },
      .subresourceRange = {
      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = 1,
      },
    };

    result = idevice->table.vkCreateImageView(
      idevice->logicalDevice,
      &imageViewCreateInfo,
      nullptr,
      &iimage->imageView
    );
    if (result != VK_SUCCESS)
    {
      iinstance->iimageStore.free(handle);
      vmaDestroyImage(idevice->allocator, iimage->image, iimage->allocation);
      CGPU_RETURN_ERROR("failed to create image view");
    }

    if (iinstance->debugUtilsEnabled && createInfo.debugName)
    {
      cgpuSetObjectName(idevice->logicalDevice, VK_OBJECT_TYPE_IMAGE, (uint64_t) iimage->image, createInfo.debugName);
    }

    iimage->width = createInfo.width;
    iimage->height = createInfo.height;
    iimage->depth = createInfo.is3d ? createInfo.depth : 1;
    iimage->layout = imageCreateInfo.initialLayout;
    iimage->accessMask = 0;

    image->handle = handle;
    return true;
  }

  bool cgpuDestroyImage(CgpuDevice device, CgpuImage image)
  {
    CGPU_RESOLVE_OR_RETURN_DEVICE(device, idevice);
    CGPU_RESOLVE_OR_RETURN_IMAGE(image, iimage);

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
    CGPU_RESOLVE_OR_RETURN_DEVICE(device, idevice);
    CGPU_RESOLVE_OR_RETURN_IMAGE(image, iimage);

    if (vmaMapMemory(idevice->allocator, iimage->allocation, mappedMem) != VK_SUCCESS) {
      CGPU_RETURN_ERROR("failed to map image memory");
    }
    return true;
  }

  bool cgpuUnmapImage(CgpuDevice device, CgpuImage image)
  {
    CGPU_RESOLVE_OR_RETURN_DEVICE(device, idevice);
    CGPU_RESOLVE_OR_RETURN_IMAGE(image, iimage);

    vmaUnmapMemory(idevice->allocator, iimage->allocation);
    return true;
  }

  bool cgpuCreateSampler(CgpuDevice device,
                         CgpuSamplerCreateInfo createInfo,
                         CgpuSampler* sampler)
  {
    CGPU_RESOLVE_OR_RETURN_DEVICE(device, idevice);

    uint64_t handle = iinstance->isamplerStore.allocate();

    CGPU_RESOLVE_OR_RETURN_SAMPLER({ handle }, isampler);

    // Emulate MDL's clip wrap mode if necessary; use optimal mode (according to ARM) if not.
    bool clampToBlack = (createInfo.addressModeU == CGPU_SAMPLER_ADDRESS_MODE_CLAMP_TO_BLACK) ||
                        (createInfo.addressModeV == CGPU_SAMPLER_ADDRESS_MODE_CLAMP_TO_BLACK) ||
                        (createInfo.addressModeW == CGPU_SAMPLER_ADDRESS_MODE_CLAMP_TO_BLACK);

    VkSamplerCreateInfo samplerCreateInfo = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .magFilter = VK_FILTER_LINEAR,
      .minFilter = VK_FILTER_LINEAR,
      .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
      .addressModeU = cgpuTranslateAddressMode(createInfo.addressModeU),
      .addressModeV = cgpuTranslateAddressMode(createInfo.addressModeV),
      .addressModeW = cgpuTranslateAddressMode(createInfo.addressModeW),
      .mipLodBias = 0.0f,
      .anisotropyEnable = VK_FALSE,
      .maxAnisotropy = 1.0f,
      .compareEnable = VK_FALSE,
      .compareOp = VK_COMPARE_OP_NEVER,
      .minLod = 0.0f,
      .maxLod = VK_LOD_CLAMP_NONE,
      .borderColor = clampToBlack ? VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK : VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
      .unnormalizedCoordinates = VK_FALSE,
    };

    VkResult result = idevice->table.vkCreateSampler(
      idevice->logicalDevice,
      &samplerCreateInfo,
      nullptr,
      &isampler->sampler
    );

    if (result != VK_SUCCESS) {
      iinstance->isamplerStore.free(handle);
      CGPU_RETURN_ERROR("failed to create sampler");
    }

    sampler->handle = handle;
    return true;
  }

  bool cgpuDestroySampler(CgpuDevice device, CgpuSampler sampler)
  {
    CGPU_RESOLVE_OR_RETURN_DEVICE(device, idevice);
    CGPU_RESOLVE_OR_RETURN_SAMPLER(sampler, isampler);

    idevice->table.vkDestroySampler(idevice->logicalDevice, isampler->sampler, nullptr);

    iinstance->isamplerStore.free(sampler.handle);

    return true;
  }

  static bool cgpuCreatePipelineLayout(CgpuIDevice* idevice, CgpuIPipeline* ipipeline, CgpuIShader* ishader, VkShaderStageFlags stageFlags)
  {
    VkPushConstantRange pushConstRange = {
      .stageFlags = stageFlags,
      .offset = 0,
      .size = ishader->reflection.pushConstantsSize,
    };

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .setLayoutCount = 1,
      .pSetLayouts = &ipipeline->descriptorSetLayout,
      .pushConstantRangeCount = pushConstRange.size ? 1u : 0u,
      .pPushConstantRanges = &pushConstRange,
    };

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

      VkDescriptorSetLayoutBinding layoutBinding = {
        .binding = binding_reflection->binding,
        .descriptorType = (VkDescriptorType) binding_reflection->descriptorType,
        .descriptorCount = binding_reflection->count,
        .stageFlags = stageFlags,
        .pImmutableSamplers = nullptr,
      };

      ipipeline->descriptorSetLayoutBindings.push_back(layoutBinding);
    }

    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .bindingCount = (uint32_t) ipipeline->descriptorSetLayoutBindings.size(),
      .pBindings = ipipeline->descriptorSetLayoutBindings.data(),
    };

    VkResult result = idevice->table.vkCreateDescriptorSetLayout(
      idevice->logicalDevice,
      &descriptorSetLayoutCreateInfo,
      nullptr,
      &ipipeline->descriptorSetLayout
    );

    if (result != VK_SUCCESS) {
      CGPU_RETURN_ERROR("failed to create descriptor set layout");
    }

    uint32_t uniformBufferCount = 0;
    uint32_t storageBufferCount = 0;
    uint32_t storageImageCount = 0;
    uint32_t sampledImageCount = 0;
    uint32_t samplerCount = 0;
    uint32_t asCount = 0;

    for (uint32_t i = 0; i < shaderReflection->bindings.size(); i++)
    {
      const CgpuShaderReflectionBinding* binding = &shaderReflection->bindings[i];

      switch (binding->descriptorType)
      {
      case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER: storageBufferCount += binding->count; break;
      case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: uniformBufferCount += binding->count; break;
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

    if (uniformBufferCount > 0)
    {
      poolSizes[poolSizeCount].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
      poolSizes[poolSizeCount].descriptorCount = uniformBufferCount;
      poolSizeCount++;
    }
    if (storageBufferCount > 0)
    {
      poolSizes[poolSizeCount].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      poolSizes[poolSizeCount].descriptorCount = storageBufferCount;
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

    VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .maxSets = 1,
      .poolSizeCount = poolSizeCount,
      .pPoolSizes = poolSizes,
    };

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

    VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .pNext = nullptr,
      .descriptorPool = ipipeline->descriptorPool,
      .descriptorSetCount = 1,
      .pSetLayouts = &ipipeline->descriptorSetLayout,
    };

    result = idevice->table.vkAllocateDescriptorSets(
      idevice->logicalDevice,
      &descriptorSetAllocateInfo,
      &ipipeline->descriptorSet
    );

    if (result != VK_SUCCESS) {
      idevice->table.vkDestroyDescriptorPool(idevice->logicalDevice, ipipeline->descriptorPool, nullptr);
      idevice->table.vkDestroyDescriptorSetLayout(idevice->logicalDevice, ipipeline->descriptorSetLayout, nullptr);
      CGPU_RETURN_ERROR("failed to allocate descriptor set");
    }

    return true;
  }

  bool cgpuCreateComputePipeline(CgpuDevice device,
                                 CgpuComputePipelineCreateInfo createInfo,
                                 CgpuPipeline* pipeline)
  {
    CGPU_RESOLVE_OR_RETURN_DEVICE(device, idevice);
    CGPU_RESOLVE_OR_RETURN_SHADER(createInfo.shader, ishader);

    uint64_t handle = iinstance->ipipelineStore.allocate();

    CGPU_RESOLVE_OR_RETURN_PIPELINE({ handle }, ipipeline);

    if (!cgpuCreatePipelineDescriptors(idevice, ipipeline, ishader, VK_SHADER_STAGE_COMPUTE_BIT))
    {
      iinstance->ipipelineStore.free(handle);
      CGPU_RETURN_ERROR("failed to create descriptor set layout");
    }

    if (!cgpuCreatePipelineLayout(idevice, ipipeline, ishader, VK_SHADER_STAGE_COMPUTE_BIT))
    {
      iinstance->ipipelineStore.free(handle);
      idevice->table.vkDestroyDescriptorSetLayout(idevice->logicalDevice, ipipeline->descriptorSetLayout, nullptr);
      idevice->table.vkDestroyDescriptorPool(idevice->logicalDevice, ipipeline->descriptorPool, nullptr);
      CGPU_RETURN_ERROR("failed to create pipeline layout");
    }

    VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .stage = VK_SHADER_STAGE_COMPUTE_BIT,
      .module = ishader->module,
      .pName = "main",
      .pSpecializationInfo = nullptr,
    };

    VkComputePipelineCreateInfo pipelineCreateInfo = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .stage = pipelineShaderStageCreateInfo,
      .layout = ipipeline->layout,
      .basePipelineHandle = VK_NULL_HANDLE,
      .basePipelineIndex = -1
    };

    VkResult result = idevice->table.vkCreateComputePipelines(
      idevice->logicalDevice,
      idevice->pipelineCache,
      1,
      &pipelineCreateInfo,
      nullptr,
      &ipipeline->pipeline
    );

    if (result != VK_SUCCESS) {
      iinstance->ipipelineStore.free(handle);
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

    if (iinstance->debugUtilsEnabled && createInfo.debugName)
    {
      cgpuSetObjectName(idevice->logicalDevice, VK_OBJECT_TYPE_PIPELINE, (uint64_t) ipipeline->pipeline, createInfo.debugName);
    }

    ipipeline->bindPoint = VK_PIPELINE_BIND_POINT_COMPUTE;

    pipeline->handle = handle;
    return true;
  }

  static bool cgpuCreateRtPipelineSbt(CgpuIDevice* idevice,
                                      CgpuIPipeline* ipipeline,
                                      uint32_t groupCount,
                                      uint32_t missShaderCount,
                                      uint32_t hitGroupCount)
  {
    uint32_t handleSize = idevice->properties.shaderGroupHandleSize;
    uint32_t alignedHandleSize = gbAlignUpwards(handleSize, idevice->properties.shaderGroupHandleAlignment);

    ipipeline->sbtRgen.stride = gbAlignUpwards(alignedHandleSize, idevice->properties.shaderGroupBaseAlignment);
    ipipeline->sbtRgen.size = ipipeline->sbtRgen.stride; // Special raygen condition: size must be equal to stride
    ipipeline->sbtMiss.stride = alignedHandleSize;
    ipipeline->sbtMiss.size = gbAlignUpwards(missShaderCount * alignedHandleSize, idevice->properties.shaderGroupBaseAlignment);
    ipipeline->sbtHit.stride = alignedHandleSize;
    ipipeline->sbtHit.size = gbAlignUpwards(hitGroupCount * alignedHandleSize, idevice->properties.shaderGroupBaseAlignment);

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

    if (!cgpuCreateIBufferAligned(idevice, bufferUsageFlags, bufferMemPropFlags, sbtSize, 0, &ipipeline->sbt, "[SBT]"))
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
                            CgpuRtPipelineCreateInfo createInfo,
                            CgpuPipeline* pipeline)
  {
    CGPU_RESOLVE_OR_RETURN_DEVICE(device, idevice);

    uint64_t handle = iinstance->ipipelineStore.allocate();

    CGPU_RESOLVE_OR_RETURN_PIPELINE({ handle }, ipipeline);

    // Zero-init for cleanup routine.
    memset(ipipeline, 0, sizeof(CgpuIPipeline));

    // In a ray tracing pipeline, all shaders are expected to have the same descriptor set layouts. Here, we
    // construct the descriptor set layouts and the pipeline layout from only the ray generation shader.
    CGPU_RESOLVE_OR_RETURN_SHADER(createInfo.rgenShader, irgenShader);

    // Set up stages
    GbSmallVector<VkPipelineShaderStageCreateInfo, 128> stages;
    VkShaderStageFlags shaderStageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

    auto pushStage = [&stages](VkShaderStageFlagBits stage, VkShaderModule module) {
      VkPipelineShaderStageCreateInfo pipeline_shader_stage_create_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .stage = stage,
        .module = module,
        .pName = "main",
        .pSpecializationInfo = nullptr,
      };
      stages.push_back(pipeline_shader_stage_create_info);
    };

    // Ray gen
    pushStage(VK_SHADER_STAGE_RAYGEN_BIT_KHR, irgenShader->module);

    // Miss
    if (createInfo.missShaderCount > 0)
    {
      shaderStageFlags |= VK_SHADER_STAGE_MISS_BIT_KHR;
    }
    for (uint32_t i = 0; i < createInfo.missShaderCount; i++)
    {
      CGPU_RESOLVE_OR_RETURN_SHADER(createInfo.missShaders[i], imissShader);

      assert(imissShader->module);
      pushStage(VK_SHADER_STAGE_MISS_BIT_KHR, imissShader->module);
    }

    // Hit
    for (uint32_t i = 0; i < createInfo.hitGroupCount; i++)
    {
      const CgpuRtHitGroup* hitGroup = &createInfo.hitGroups[i];

      // Closest hit (optional)
      if (hitGroup->closestHitShader.handle)
      {
        CGPU_RESOLVE_OR_RETURN_SHADER(hitGroup->closestHitShader, iclosestHitShader);

        assert(iclosestHitShader->stageFlags == VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);

        pushStage(iclosestHitShader->stageFlags, iclosestHitShader->module);
        shaderStageFlags |= iclosestHitShader->stageFlags;
      }

      // Any hit (optional)
      if (hitGroup->anyHitShader.handle)
      {
        CGPU_RESOLVE_OR_RETURN_SHADER(hitGroup->anyHitShader, ianyHitShader);

        assert(ianyHitShader->stageFlags == VK_SHADER_STAGE_ANY_HIT_BIT_KHR);

        pushStage(ianyHitShader->stageFlags, ianyHitShader->module);
        shaderStageFlags |= ianyHitShader->stageFlags;
      }
    }

    // Set up groups
    GbSmallVector<VkRayTracingShaderGroupCreateInfoKHR, 128> groups;
    groups.resize(1/*rgen*/ + createInfo.missShaderCount + createInfo.hitGroupCount);

    for (uint32_t i = 0; i < groups.size(); i++)
    {
      VkRayTracingShaderGroupCreateInfoKHR sgCreateInfo = {
        .sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
        .pNext = nullptr,
        .type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
        .generalShader = i,
        .closestHitShader = VK_SHADER_UNUSED_KHR,
        .anyHitShader = VK_SHADER_UNUSED_KHR,
        .intersectionShader = VK_SHADER_UNUSED_KHR,
        .pShaderGroupCaptureReplayHandle = nullptr,
      };

      groups[i] = sgCreateInfo;
    }

    bool anyNullClosestHitShader = false;
    bool anyNullAnyHitShader = false;

    uint32_t hitStageAndGroupOffset = 1/*rgen*/ + createInfo.missShaderCount;
    uint32_t hitShaderStageIndex = hitStageAndGroupOffset;
    for (uint32_t i = 0; i < createInfo.hitGroupCount; i++)
    {
      const CgpuRtHitGroup* hit_group = &createInfo.hitGroups[i];

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
      uint32_t groupCount = hitStageAndGroupOffset + createInfo.hitGroupCount;

      VkPipelineCreateFlags flags = 0;
      if (!anyNullClosestHitShader && createInfo.hitGroupCount > 0)
      {
        flags |= VK_PIPELINE_CREATE_RAY_TRACING_NO_NULL_CLOSEST_HIT_SHADERS_BIT_KHR;
      }
      if (!anyNullAnyHitShader && createInfo.hitGroupCount > 0)
      {
        flags |= VK_PIPELINE_CREATE_RAY_TRACING_NO_NULL_ANY_HIT_SHADERS_BIT_KHR;
      }

      VkRayTracingPipelineCreateInfoKHR rtPipelineCreateInfo = {
        .sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
        .pNext = nullptr,
        .flags = flags,
        .stageCount = (uint32_t) stages.size(),
        .pStages = stages.data(),
        .groupCount = (uint32_t) groups.size(),
        .pGroups = groups.data(),
        .maxPipelineRayRecursionDepth = 1,
        .pLibraryInfo = nullptr,
        .pLibraryInterface = nullptr,
        .pDynamicState = nullptr,
        .layout = ipipeline->layout,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = -1
      };

      if (idevice->table.vkCreateRayTracingPipelinesKHR(idevice->logicalDevice,
                                                        VK_NULL_HANDLE,
                                                        idevice->pipelineCache,
                                                        1,
                                                        &rtPipelineCreateInfo,
                                                        nullptr,
                                                        &ipipeline->pipeline) != VK_SUCCESS)
      {
        goto cleanup_fail;
      }

      ipipeline->bindPoint = VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR;

      // Create the SBT.
      if (!cgpuCreateRtPipelineSbt(idevice, ipipeline, groupCount, createInfo.missShaderCount, createInfo.hitGroupCount))
      {
        goto cleanup_fail;
      }

      if (iinstance->debugUtilsEnabled && createInfo.debugName)
      {
        cgpuSetObjectName(idevice->logicalDevice, VK_OBJECT_TYPE_PIPELINE, (uint64_t) ipipeline->pipeline, createInfo.debugName);
      }

      pipeline->handle = handle;
      return true;
    }

cleanup_fail:
    idevice->table.vkDestroyPipelineLayout(idevice->logicalDevice, ipipeline->layout, nullptr);
    idevice->table.vkDestroyDescriptorSetLayout(idevice->logicalDevice, ipipeline->descriptorSetLayout, nullptr);
    idevice->table.vkDestroyDescriptorPool(idevice->logicalDevice, ipipeline->descriptorPool, nullptr);
    iinstance->ipipelineStore.free(handle);

    CGPU_RETURN_ERROR("failed to create rt pipeline");
  }

  bool cgpuDestroyPipeline(CgpuDevice device, CgpuPipeline pipeline)
  {
    CGPU_RESOLVE_OR_RETURN_DEVICE(device, idevice);
    CGPU_RESOLVE_OR_RETURN_PIPELINE(pipeline, ipipeline);

    if (ipipeline->bindPoint == VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR)
    {
      cgpuDestroyIBuffer(idevice, &ipipeline->sbt);
    }

    idevice->table.vkDestroyDescriptorPool(idevice->logicalDevice, ipipeline->descriptorPool, nullptr);
    idevice->table.vkDestroyPipeline(idevice->logicalDevice, ipipeline->pipeline, nullptr);
    idevice->table.vkDestroyPipelineLayout(idevice->logicalDevice, ipipeline->layout, nullptr);
    idevice->table.vkDestroyDescriptorSetLayout(idevice->logicalDevice, ipipeline->descriptorSetLayout, nullptr);

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
    VkAccelerationStructureBuildGeometryInfoKHR asBuildGeomInfo = {
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
      .pNext = nullptr,
      .type = asType,
      .flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
      .mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
      .srcAccelerationStructure = VK_NULL_HANDLE,
      .dstAccelerationStructure = VK_NULL_HANDLE, // set in second round
      .geometryCount = 1,
      .pGeometries = asGeom,
      .ppGeometries = nullptr,
      .scratchData = {
        .deviceAddress = 0, // set in second round
      }
    };

    VkAccelerationStructureBuildSizesInfoKHR asBuildSizesInfo = {
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR,
      .pNext = nullptr,
      .accelerationStructureSize = 0, // output
      .updateScratchSize = 0, // output
      .buildScratchSize = 0, // output
    };

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
                                  iasBuffer, "[AS buffer]"))
    {
      CGPU_RETURN_ERROR("failed to create AS buffer");
    }

    VkAccelerationStructureCreateInfoKHR asCreateInfo = {
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
      .pNext = nullptr,
      .createFlags = 0,
      .buffer = iasBuffer->buffer,
      .offset = 0,
      .size = asBuildSizesInfo.accelerationStructureSize,
      .type = asType,
      .deviceAddress = 0, // used for capture-replay feature
    };

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
                                  &iscratchBuffer, "[AS scratch buffer]"))
    {
      cgpuDestroyIBuffer(idevice, iasBuffer);
      idevice->table.vkDestroyAccelerationStructureKHR(idevice->logicalDevice, *as, nullptr);
      CGPU_RETURN_ERROR("failed to create AS scratch buffer");
    }

    asBuildGeomInfo.dstAccelerationStructure = *as;
    asBuildGeomInfo.scratchData.hostAddress = 0;
    asBuildGeomInfo.scratchData.deviceAddress = cgpuGetBufferDeviceAddress(idevice, &iscratchBuffer);

    VkAccelerationStructureBuildRangeInfoKHR asBuildRangeInfo = {
      .primitiveCount = primitiveCount,
      .primitiveOffset = 0,
      .firstVertex = 0,
      .transformOffset = 0,
    };

    const VkAccelerationStructureBuildRangeInfoKHR* asBuildRangeInfoPtr = &asBuildRangeInfo;

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
    idevice->table.vkCmdBuildAccelerationStructuresKHR(icommandBuffer->commandBuffer, 1, &asBuildGeomInfo, &asBuildRangeInfoPtr);
    cgpuEndCommandBuffer(commandBuffer);

    CgpuSemaphore semaphore;
    if (!cgpuCreateSemaphore(device, &semaphore))
    {
      cgpuDestroyIBuffer(idevice, iasBuffer);
      cgpuDestroyIBuffer(idevice, &iscratchBuffer);
      idevice->table.vkDestroyAccelerationStructureKHR(idevice->logicalDevice, *as, nullptr);
      CGPU_RETURN_ERROR("failed to create AS build semaphore");
    }

    CgpuSignalSemaphoreInfo signalSemaphoreInfo{ .semaphore = semaphore, .value = 1 };
    cgpuSubmitCommandBuffer(device, commandBuffer, 1, &signalSemaphoreInfo);
    CgpuWaitSemaphoreInfo waitSemaphoreInfo{ .semaphore = semaphore, .value = 1 };
    cgpuWaitSemaphores(device, 1, &waitSemaphoreInfo);

    // Dispose resources
    cgpuDestroySemaphore(device, semaphore);
    cgpuDestroyCommandBuffer(device, commandBuffer);
    cgpuDestroyIBuffer(idevice, &iscratchBuffer);

    return true;
  }

  bool cgpuCreateBlas(CgpuDevice device,
                      CgpuBlasCreateInfo createInfo,
                      CgpuBlas* blas)
  {
    CGPU_RESOLVE_OR_RETURN_DEVICE(device, idevice);
    CGPU_RESOLVE_OR_RETURN_BUFFER(createInfo.vertexBuffer, ivertexBuffer);
    CGPU_RESOLVE_OR_RETURN_BUFFER(createInfo.indexBuffer, iindexBuffer);

    uint64_t handle = iinstance->iblasStore.allocate();

    CGPU_RESOLVE_OR_RETURN_BLAS({ handle }, iblas);

    VkAccelerationStructureGeometryTrianglesDataKHR asTriangleData = {
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
      .pNext = nullptr,
      .vertexFormat = VK_FORMAT_R32G32B32_SFLOAT,
      .vertexData = {
        .deviceAddress = cgpuGetBufferDeviceAddress(idevice, ivertexBuffer),
      },
      .vertexStride = sizeof(CgpuVertex),
      .maxVertex = createInfo.maxVertex,
      .indexType = VK_INDEX_TYPE_UINT32,
      .indexData = {
        .deviceAddress = cgpuGetBufferDeviceAddress(idevice, iindexBuffer),
      },
      .transformData = {
        .deviceAddress = 0, // optional
      },
    };

    VkAccelerationStructureGeometryDataKHR asGeomData = {
      .triangles = asTriangleData,
    };

    VkAccelerationStructureGeometryKHR asGeom = {
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
      .pNext = nullptr,
      .geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
      .geometry = asGeomData,
      .flags = VkGeometryFlagsKHR(createInfo.isOpaque ? VK_GEOMETRY_OPAQUE_BIT_KHR : 0)
    };

    bool creationSuccessul = cgpuCreateTopOrBottomAs(device, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
        &asGeom, createInfo.triangleCount, &iblas->buffer, &iblas->as);

    if (!creationSuccessul)
    {
      iinstance->iblasStore.free(handle);
      CGPU_RETURN_ERROR("failed to build BLAS");
    }

    if (iinstance->debugUtilsEnabled && createInfo.debugName)
    {
      cgpuSetObjectName(idevice->logicalDevice, VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR, (uint64_t) iblas->as, createInfo.debugName);
    }

    VkAccelerationStructureDeviceAddressInfoKHR asAddressInfo = {
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
      .pNext = nullptr,
      .accelerationStructure = iblas->as,
    };
    iblas->address = idevice->table.vkGetAccelerationStructureDeviceAddressKHR(idevice->logicalDevice, &asAddressInfo);

    iblas->isOpaque = createInfo.isOpaque;

    blas->handle = handle;
    return true;
  }

  bool cgpuCreateTlas(CgpuDevice device,
                      CgpuTlasCreateInfo createInfo,
                      CgpuTlas* tlas)
  {
    CGPU_RESOLVE_OR_RETURN_DEVICE(device, idevice);

    uint64_t handle = iinstance->itlasStore.allocate();

    CGPU_RESOLVE_OR_RETURN_TLAS({ handle }, itlas);

    // Create instance buffer & copy into it
    if (!cgpuCreateIBufferAligned(idevice,
                                  CGPU_BUFFER_USAGE_FLAG_SHADER_DEVICE_ADDRESS | CGPU_BUFFER_USAGE_FLAG_ACCELERATION_STRUCTURE_BUILD_INPUT,
                                  CGPU_MEMORY_PROPERTY_FLAG_HOST_VISIBLE | CGPU_MEMORY_PROPERTY_FLAG_HOST_COHERENT,
                                  (createInfo.instanceCount ? createInfo.instanceCount : 1) * sizeof(VkAccelerationStructureInstanceKHR), 0,
                                  &itlas->instances, createInfo.debugName))
    {
      iinstance->itlasStore.free(handle);
      CGPU_RETURN_ERROR("failed to create TLAS instances buffer");
    }

    bool areAllBlasOpaque = true;
    {
      uint8_t* mapped_mem;
      if (vmaMapMemory(idevice->allocator, itlas->instances.allocation, (void**) &mapped_mem) != VK_SUCCESS)
      {
        iinstance->itlasStore.free(handle);
        cgpuDestroyIBuffer(idevice, &itlas->instances);
        CGPU_RETURN_ERROR("failed to map buffer memory");
      }

      for (uint32_t i = 0; i < createInfo.instanceCount; i++)
      {
        const CgpuBlasInstance& instanceDesc = createInfo.instances[i];

        CgpuIBlas* iblas;
        if (!cgpuResolveBlas(instanceDesc.as, &iblas)) {
          iinstance->itlasStore.free(handle);
          cgpuDestroyIBuffer(idevice, &itlas->instances);
          CGPU_RETURN_ERROR_INVALID_HANDLE;
        }

        uint32_t instanceCustomIndex = instanceDesc.instanceCustomIndex;
        if ((instanceCustomIndex & 0xFF000000u) != 0u)
        {
          iinstance->itlasStore.free(handle);
          cgpuDestroyIBuffer(idevice, &itlas->instances);
          CGPU_RETURN_ERROR("instanceCustomIndex must be equal to or smaller than 2^24");
        }

        VkAccelerationStructureInstanceKHR* asInstance = (VkAccelerationStructureInstanceKHR*) &mapped_mem[i * sizeof(VkAccelerationStructureInstanceKHR)];
        memcpy(&asInstance->transform, &instanceDesc.transform, sizeof(VkTransformMatrixKHR));
        asInstance->instanceCustomIndex = instanceCustomIndex;
        asInstance->mask = 0xFF;
        asInstance->instanceShaderBindingTableRecordOffset = instanceDesc.hitGroupIndex;
        asInstance->flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        asInstance->accelerationStructureReference = iblas->address;

        areAllBlasOpaque &= iblas->isOpaque;
      }

      vmaUnmapMemory(idevice->allocator, itlas->instances.allocation);
    }

    // Create TLAS
    VkAccelerationStructureGeometryKHR asGeom = {
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
      .pNext = nullptr,
      .geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR,
      .geometry = {
        .instances = {
          .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
          .pNext = nullptr,
          .arrayOfPointers = VK_FALSE,
          .data = {
            .deviceAddress = cgpuGetBufferDeviceAddress(idevice, &itlas->instances),
          }
        },
      },
      .flags = VkGeometryFlagsKHR(areAllBlasOpaque ? VK_GEOMETRY_OPAQUE_BIT_KHR : 0)
    };

    if (!cgpuCreateTopOrBottomAs(device, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, &asGeom, createInfo.instanceCount, &itlas->buffer, &itlas->as))
    {
      iinstance->itlasStore.free(handle);
      cgpuDestroyIBuffer(idevice, &itlas->instances);
      CGPU_RETURN_ERROR("failed to build TLAS");
    }

    if (iinstance->debugUtilsEnabled && createInfo.debugName)
    {
      cgpuSetObjectName(idevice->logicalDevice, VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR, (uint64_t) itlas->as, createInfo.debugName);
    }

    tlas->handle = handle;
    return true;
  }

  bool cgpuDestroyBlas(CgpuDevice device, CgpuBlas blas)
  {
    CGPU_RESOLVE_OR_RETURN_DEVICE(device, idevice);
    CGPU_RESOLVE_OR_RETURN_BLAS(blas, iblas);

    idevice->table.vkDestroyAccelerationStructureKHR(idevice->logicalDevice, iblas->as, nullptr);
    cgpuDestroyIBuffer(idevice, &iblas->buffer);

    iinstance->iblasStore.free(blas.handle);
    return true;
  }

  bool cgpuDestroyTlas(CgpuDevice device, CgpuTlas tlas)
  {
    CGPU_RESOLVE_OR_RETURN_DEVICE(device, idevice);
    CGPU_RESOLVE_OR_RETURN_TLAS(tlas, itlas);

    idevice->table.vkDestroyAccelerationStructureKHR(idevice->logicalDevice, itlas->as, nullptr);
    cgpuDestroyIBuffer(idevice, &itlas->instances);
    cgpuDestroyIBuffer(idevice, &itlas->buffer);

    iinstance->itlasStore.free(tlas.handle);
    return true;
  }

  bool cgpuCreateCommandBuffer(CgpuDevice device, CgpuCommandBuffer* commandBuffer)
  {
    CGPU_RESOLVE_OR_RETURN_DEVICE(device, idevice);

    uint64_t handle = iinstance->icommandBufferStore.allocate();

    CGPU_RESOLVE_OR_RETURN_COMMAND_BUFFER({ handle }, icommandBuffer);

    icommandBuffer->device.handle = device.handle;

    VkCommandBufferAllocateInfo cmdbufAllocInfo = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .pNext = nullptr,
      .commandPool = idevice->commandPool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
    };

    VkResult result = idevice->table.vkAllocateCommandBuffers(
      idevice->logicalDevice,
      &cmdbufAllocInfo,
      &icommandBuffer->commandBuffer
    );
    if (result != VK_SUCCESS) {
      iinstance->icommandBufferStore.free(handle);
      CGPU_RETURN_ERROR("failed to allocate command buffer");
    }

    commandBuffer->handle = handle;
    return true;
  }

  bool cgpuDestroyCommandBuffer(CgpuDevice device, CgpuCommandBuffer commandBuffer)
  {
    CGPU_RESOLVE_OR_RETURN_DEVICE(device, idevice);
    CGPU_RESOLVE_OR_RETURN_COMMAND_BUFFER(commandBuffer, icommandBuffer);

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
    CGPU_RESOLVE_OR_RETURN_COMMAND_BUFFER(commandBuffer, icommandBuffer);
    CGPU_RESOLVE_OR_RETURN_DEVICE(icommandBuffer->device, idevice);

    VkCommandBufferBeginInfo beginInfo = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .pNext = nullptr,
      .flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT,
      .pInheritanceInfo = nullptr,
    };

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
    CGPU_RESOLVE_OR_RETURN_COMMAND_BUFFER(commandBuffer, icommandBuffer);
    CGPU_RESOLVE_OR_RETURN_DEVICE(icommandBuffer->device, idevice);
    CGPU_RESOLVE_OR_RETURN_PIPELINE(pipeline, ipipeline);

    idevice->table.vkCmdBindPipeline(
      icommandBuffer->commandBuffer,
      ipipeline->bindPoint,
      ipipeline->pipeline
    );

    uint32_t firstDescriptorSet = 0;
    uint32_t descriptorSetCount = 1;
    uint32_t dynamicOffsetCount = 0;
    const uint32_t* dynamicOffsets = nullptr;

    idevice->table.vkCmdBindDescriptorSets(
      icommandBuffer->commandBuffer,
      ipipeline->bindPoint,
      ipipeline->layout,
      firstDescriptorSet,
      descriptorSetCount,
      &ipipeline->descriptorSet,
      dynamicOffsetCount,
      dynamicOffsets
    );

    return true;
  }

  bool cgpuCmdTransitionShaderImageLayouts(CgpuCommandBuffer commandBuffer,
                                           CgpuShader shader,
                                           uint32_t imageCount,
                                           const CgpuImageBinding* images)
  {
    CGPU_RESOLVE_OR_RETURN_SHADER(shader, ishader);
    CGPU_RESOLVE_OR_RETURN_COMMAND_BUFFER(commandBuffer, icommandBuffer);
    CGPU_RESOLVE_OR_RETURN_DEVICE(icommandBuffer->device, idevice);

    GbSmallVector<VkImageMemoryBarrier2KHR, 64> barriers;

    /* FIXME: this has quadratic complexity */
    const CgpuShaderReflection* reflection = &ishader->reflection;
    for (uint32_t i = 0; i < reflection->bindings.size(); i++)
    {
      const CgpuShaderReflectionBinding* binding = &reflection->bindings[i];

      VkImageLayout newLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      if (binding->descriptorType == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE)
      {
        newLayout = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL_KHR;
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

        CGPU_RESOLVE_OR_RETURN_IMAGE(imageBinding->image, iimage);

        VkImageLayout oldLayout = iimage->layout;
        if (newLayout == oldLayout)
        {
          continue;
        }

        VkAccessFlags2KHR accessMask = VK_ACCESS_2_NONE_KHR;
        if (binding->readAccess) {
          accessMask = VK_ACCESS_2_SHADER_READ_BIT_KHR;
        }
        if (binding->writeAccess) {
          accessMask = VK_ACCESS_2_SHADER_WRITE_BIT_KHR;
        }

        VkImageSubresourceRange range = {
          .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
          .baseMipLevel = 0,
          .levelCount = 1,
          .baseArrayLayer = 0,
          .layerCount = 1
        };

        VkImageMemoryBarrier2KHR barrier = {
          .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2_KHR,
          .pNext = nullptr,
          .srcStageMask = cgpuPipelineStageFlagsFromShaderStageFlags(ishader->stageFlags),
          .srcAccessMask = iimage->accessMask,
          .dstStageMask = cgpuPipelineStageFlagsFromShaderStageFlags(ishader->stageFlags),
          .dstAccessMask = accessMask,
          .oldLayout = oldLayout,
          .newLayout = newLayout,
          .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .image = iimage->image,
          .subresourceRange = range
        };
        barriers.push_back(barrier);

        iimage->accessMask = accessMask;
        iimage->layout = newLayout;
      }
    }

    if (barriers.size() > 0)
    {
      VkDependencyInfoKHR dependencyInfo = {
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR,
        .pNext = nullptr,
        .dependencyFlags = 0,
        .memoryBarrierCount = 0,
        .pMemoryBarriers = nullptr,
        .bufferMemoryBarrierCount = 0,
        .pBufferMemoryBarriers = nullptr,
        .imageMemoryBarrierCount = (uint32_t) barriers.size(),
        .pImageMemoryBarriers = barriers.data()
      };

      idevice->table.vkCmdPipelineBarrier2KHR(icommandBuffer->commandBuffer, &dependencyInfo);
    }

    return true;
  }

  bool cgpuCmdUpdateBindings(CgpuCommandBuffer commandBuffer,
                             CgpuPipeline pipeline,
                             const CgpuBindings* bindings
  )
  {
    CGPU_RESOLVE_OR_RETURN_COMMAND_BUFFER(commandBuffer, icommandBuffer);
    CGPU_RESOLVE_OR_RETURN_DEVICE(icommandBuffer->device, idevice);
    CGPU_RESOLVE_OR_RETURN_PIPELINE(pipeline, ipipeline);

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

      VkWriteDescriptorSet writeDescriptorSet = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .pNext = nullptr,
        .dstSet = ipipeline->descriptorSet,
        .dstBinding = layoutBinding->binding,
        .dstArrayElement = 0,
        .descriptorCount = layoutBinding->descriptorCount,
        .descriptorType = layoutBinding->descriptorType,
        .pImageInfo = nullptr,
        .pBufferInfo = nullptr,
        .pTexelBufferView = nullptr,
      };

      for (uint32_t j = 0; j < layoutBinding->descriptorCount; j++)
      {
        bool slotHandled = false;

        if (layoutBinding->descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER ||
            layoutBinding->descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
        {
          for (uint32_t k = 0; k < bindings->bufferCount; ++k)
          {
            const CgpuBufferBinding* bufferBinding = &bindings->buffers[k];

            if (bufferBinding->binding != layoutBinding->binding || bufferBinding->index != j)
            {
              continue;
            }

            CGPU_RESOLVE_OR_RETURN_BUFFER(bufferBinding->buffer, ibuffer);

            if ((bufferBinding->offset % idevice->properties.minStorageBufferOffsetAlignment) != 0) {
              CGPU_RETURN_ERROR("buffer binding offset not aligned");
            }

            VkDescriptorBufferInfo bufferInfo = {
              .buffer = ibuffer->buffer,
              .offset = bufferBinding->offset,
              .range = (bufferBinding->size == CGPU_WHOLE_SIZE) ? (ibuffer->size - bufferBinding->offset) : bufferBinding->size,
            };
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

            CGPU_RESOLVE_OR_RETURN_IMAGE(imageBinding->image, iimage);

            VkDescriptorImageInfo imageInfo = {
              .sampler = VK_NULL_HANDLE,
              .imageView = iimage->imageView,
              .imageLayout = iimage->layout,
            };
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

            CGPU_RESOLVE_OR_RETURN_SAMPLER(samplerBinding->sampler, isampler);

            VkDescriptorImageInfo imageInfo = {
              .sampler = isampler->sampler,
              .imageView = VK_NULL_HANDLE,
              .imageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            };
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

            CGPU_RESOLVE_OR_RETURN_TLAS(asBinding->as, itlas);

            VkWriteDescriptorSetAccelerationStructureKHR asInfo = {
              .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
              .pNext = nullptr,
              .accelerationStructureCount = 1,
              .pAccelerationStructures = &itlas->as,
            };
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
      (uint32_t) writeDescriptorSets.size(),
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
    CGPU_RESOLVE_OR_RETURN_COMMAND_BUFFER(commandBuffer, icommandBuffer);
    CGPU_RESOLVE_OR_RETURN_DEVICE(icommandBuffer->device, idevice);
    CGPU_RESOLVE_OR_RETURN_BUFFER(dstBuffer, idstBuffer);

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
    CGPU_RESOLVE_OR_RETURN_COMMAND_BUFFER(commandBuffer, icommandBuffer);
    CGPU_RESOLVE_OR_RETURN_DEVICE(icommandBuffer->device, idevice);
    CGPU_RESOLVE_OR_RETURN_BUFFER(srcBuffer, isrcBuffer);
    CGPU_RESOLVE_OR_RETURN_BUFFER(dstBuffer, idstBuffer);

    VkBufferCopy region = {
      .srcOffset = srcOffset,
      .dstOffset = dstOffset,
      .size = (size == CGPU_WHOLE_SIZE) ? isrcBuffer->size : size,
    };

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
    CGPU_RESOLVE_OR_RETURN_COMMAND_BUFFER(commandBuffer, icommandBuffer);
    CGPU_RESOLVE_OR_RETURN_DEVICE(icommandBuffer->device, idevice);
    CGPU_RESOLVE_OR_RETURN_BUFFER(buffer, ibuffer);
    CGPU_RESOLVE_OR_RETURN_IMAGE(image, iimage);

    if (iimage->layout != VK_IMAGE_LAYOUT_GENERAL)
    {
      VkAccessFlags2KHR accessMask = iimage->accessMask | VK_ACCESS_2_MEMORY_WRITE_BIT_KHR;
      VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL;

      VkImageSubresourceRange range = {
        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = 1
      };

      VkImageMemoryBarrier2KHR barrier = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2_KHR,
        .pNext = nullptr,
        .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_TRANSFER_BIT_KHR | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR, // FIXME
        .srcAccessMask = iimage->accessMask,
        .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_TRANSFER_BIT_KHR | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR, // FIXME
        .dstAccessMask = accessMask,
        .oldLayout = iimage->layout,
        .newLayout = layout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = iimage->image,
        .subresourceRange = range
      };

      VkDependencyInfoKHR dependencyInfo = {
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR,
        .pNext = nullptr,
        .dependencyFlags = 0,
        .memoryBarrierCount = 0,
        .pMemoryBarriers = nullptr,
        .bufferMemoryBarrierCount = 0,
        .pBufferMemoryBarriers = nullptr,
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &barrier
      };

      idevice->table.vkCmdPipelineBarrier2KHR(icommandBuffer->commandBuffer, &dependencyInfo);

      iimage->layout = layout;
      iimage->accessMask = accessMask;
    }

    VkImageSubresourceLayers layers = {
      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
      .mipLevel = 0,
      .baseArrayLayer = 0,
      .layerCount = 1,
    };

    VkOffset3D offset = {
      .x = desc->texelOffsetX,
      .y = desc->texelOffsetY,
      .z = desc->texelOffsetZ,
    };

    VkExtent3D extent = {
      .width = desc->texelExtentX,
      .height = desc->texelExtentY,
      .depth = desc->texelExtentZ,
    };

    VkBufferImageCopy region = {
      .bufferOffset = desc->bufferOffset,
      .bufferRowLength = 0,
      .bufferImageHeight = 0,
      .imageSubresource = layers,
      .imageOffset = offset,
      .imageExtent = extent,
    };

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
    CGPU_RESOLVE_OR_RETURN_COMMAND_BUFFER(commandBuffer, icommandBuffer);
    CGPU_RESOLVE_OR_RETURN_DEVICE(icommandBuffer->device, idevice);
    CGPU_RESOLVE_OR_RETURN_PIPELINE(pipeline, ipipeline);

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
    CGPU_RESOLVE_OR_RETURN_COMMAND_BUFFER(commandBuffer, icommandBuffer);
    CGPU_RESOLVE_OR_RETURN_DEVICE(icommandBuffer->device, idevice);

    idevice->table.vkCmdDispatch(
      icommandBuffer->commandBuffer,
      dim_x,
      dim_y,
      dim_z
    );
    return true;
  }

  bool cgpuCmdPipelineBarrier(CgpuCommandBuffer commandBuffer,
                              const CgpuPipelineBarrier* barrier)
  {
    CGPU_RESOLVE_OR_RETURN_COMMAND_BUFFER(commandBuffer, icommandBuffer);
    CGPU_RESOLVE_OR_RETURN_DEVICE(icommandBuffer->device, idevice);

    GbSmallVector<VkMemoryBarrier2KHR, 128> vkMemBarriers;

    for (uint32_t i = 0; i < barrier->memoryBarrierCount; ++i)
    {
      const CgpuMemoryBarrier* bCgpu = &barrier->memoryBarriers[i];

      VkMemoryBarrier2KHR bVk = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2_KHR,
        .pNext = nullptr,
        .srcStageMask = (VkPipelineStageFlagBits2KHR) bCgpu->srcStageMask,
        .srcAccessMask = (VkAccessFlags2KHR) bCgpu->srcAccessMask,
        .dstStageMask = (VkPipelineStageFlagBits2KHR) bCgpu->dstStageMask,
        .dstAccessMask = (VkAccessFlags2KHR) bCgpu->dstAccessMask
      };
      vkMemBarriers.push_back(bVk);
    }

    GbSmallVector<VkBufferMemoryBarrier2KHR, 32> vkBufferMemBarriers;
    GbSmallVector<VkImageMemoryBarrier2KHR, 128> vkImageMemBarriers;

    for (uint32_t i = 0; i < barrier->bufferBarrierCount; ++i)
    {
      const CgpuBufferMemoryBarrier* bCgpu = &barrier->bufferBarriers[i];

      CGPU_RESOLVE_OR_RETURN_BUFFER(bCgpu->buffer, ibuffer);

      VkBufferMemoryBarrier2KHR bVk = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2_KHR,
        .pNext = nullptr,
        .srcStageMask = (VkPipelineStageFlagBits2KHR) bCgpu->srcStageMask,
        .srcAccessMask = (VkAccessFlags2KHR) bCgpu->srcAccessMask,
        .dstStageMask = (VkPipelineStageFlagBits2KHR) bCgpu->dstStageMask,
        .dstAccessMask = (VkAccessFlags2KHR) bCgpu->dstAccessMask,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = ibuffer->buffer,
        .offset = bCgpu->offset,
        .size = (bCgpu->size == CGPU_WHOLE_SIZE) ? VK_WHOLE_SIZE : bCgpu->size
      };
      vkBufferMemBarriers.push_back(bVk);
    }

    for (uint32_t i = 0; i < barrier->imageBarrierCount; ++i)
    {
      const CgpuImageMemoryBarrier* bCgpu = &barrier->imageBarriers[i];

      CGPU_RESOLVE_OR_RETURN_IMAGE(bCgpu->image, iimage);

      VkAccessFlags2KHR accessMask = (VkAccessFlags2KHR) bCgpu->accessMask;

      VkImageSubresourceRange range = {
        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = 1
      };
      VkImageMemoryBarrier2KHR bVk = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2_KHR,
        .pNext = nullptr,
        .srcStageMask = (VkPipelineStageFlagBits2KHR) bCgpu->srcStageMask,
        .srcAccessMask = iimage->accessMask,
        .dstStageMask = (VkPipelineStageFlagBits2KHR) bCgpu->dstStageMask,
        .dstAccessMask = accessMask,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = iimage->image,
        .subresourceRange = range
      };
      vkImageMemBarriers.push_back(bVk);

      iimage->accessMask = accessMask;
    }

    VkDependencyInfoKHR dependencyInfo = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR,
      .pNext = nullptr,
      .dependencyFlags = 0,
      .memoryBarrierCount = (uint32_t) vkMemBarriers.size(),
      .pMemoryBarriers = vkMemBarriers.data(),
      .bufferMemoryBarrierCount = (uint32_t)vkBufferMemBarriers.size(),
      .pBufferMemoryBarriers = vkBufferMemBarriers.data(),
      .imageMemoryBarrierCount = (uint32_t) vkImageMemBarriers.size(),
      .pImageMemoryBarriers = vkImageMemBarriers.data()
    };

    idevice->table.vkCmdPipelineBarrier2KHR(icommandBuffer->commandBuffer, &dependencyInfo);

    return true;
  }

  bool cgpuCmdResetTimestamps(CgpuCommandBuffer commandBuffer,
                              uint32_t offset,
                              uint32_t count)
  {
    CGPU_RESOLVE_OR_RETURN_COMMAND_BUFFER(commandBuffer, icommandBuffer);
    CGPU_RESOLVE_OR_RETURN_DEVICE(icommandBuffer->device, idevice);

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
    CGPU_RESOLVE_OR_RETURN_COMMAND_BUFFER(commandBuffer, icommandBuffer);
    CGPU_RESOLVE_OR_RETURN_DEVICE(icommandBuffer->device, idevice);

    idevice->table.vkCmdWriteTimestamp2KHR(
      icommandBuffer->commandBuffer,
      // FIXME: use correct pipeline flag bits
      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR,
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
    CGPU_RESOLVE_OR_RETURN_COMMAND_BUFFER(commandBuffer, icommandBuffer);
    CGPU_RESOLVE_OR_RETURN_DEVICE(icommandBuffer->device, idevice);
    CGPU_RESOLVE_OR_RETURN_BUFFER(buffer, ibuffer);

    uint32_t lastIndex = offset + count;
    if (lastIndex >= CGPU_MAX_TIMESTAMP_QUERIES) {
      CGPU_RETURN_ERROR_HARDCODED_LIMIT_REACHED;
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
    CGPU_RESOLVE_OR_RETURN_COMMAND_BUFFER(commandBuffer, icommandBuffer);
    CGPU_RESOLVE_OR_RETURN_DEVICE(icommandBuffer->device, idevice);
    CGPU_RESOLVE_OR_RETURN_PIPELINE(rtPipeline, ipipeline);

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
    CGPU_RESOLVE_OR_RETURN_COMMAND_BUFFER(commandBuffer, icommandBuffer);
    CGPU_RESOLVE_OR_RETURN_DEVICE(icommandBuffer->device, idevice);

    idevice->table.vkEndCommandBuffer(icommandBuffer->commandBuffer);
    return true;
  }

  bool cgpuCreateSemaphore(CgpuDevice device, CgpuSemaphore* semaphore, uint64_t initialValue)
  {
    CGPU_RESOLVE_OR_RETURN_DEVICE(device, idevice);

    uint64_t handle = iinstance->isemaphoreStore.allocate();

    CGPU_RESOLVE_OR_RETURN_SEMAPHORE({ handle }, isemaphore);

    VkSemaphoreTypeCreateInfo typeCreateInfo = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
      .pNext = nullptr,
      .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE_KHR,
      .initialValue = initialValue
    };

    VkSemaphoreCreateInfo createInfo = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
      .pNext = &typeCreateInfo,
      .flags = 0 // unused
    };

    VkResult result = idevice->table.vkCreateSemaphore(
      idevice->logicalDevice,
      &createInfo,
      nullptr,
      &isemaphore->semaphore
    );

    if (result != VK_SUCCESS) {
      iinstance->isemaphoreStore.free(handle);
      CGPU_RETURN_ERROR("failed to create semaphore");
    }

    semaphore->handle = handle;
    return true;
  }

  bool cgpuDestroySemaphore(CgpuDevice device, CgpuSemaphore semaphore)
  {
    CGPU_RESOLVE_OR_RETURN_DEVICE(device, idevice);
    CGPU_RESOLVE_OR_RETURN_SEMAPHORE(semaphore, isemaphore);

    idevice->table.vkDestroySemaphore(
      idevice->logicalDevice,
      isemaphore->semaphore,
      nullptr
    );

    iinstance->isemaphoreStore.free(semaphore.handle);
    return true;
  }

  bool cgpuWaitSemaphores(CgpuDevice device,
                          uint32_t semaphoreInfoCount,
                          CgpuWaitSemaphoreInfo* semaphoreInfos,
                          uint64_t timeoutNs)
  {
    CGPU_RESOLVE_OR_RETURN_DEVICE(device, idevice);

    GbSmallVector<VkSemaphore, 8> semaphores(semaphoreInfoCount);
    GbSmallVector<uint64_t, 8> semaphoreValues(semaphoreInfoCount);

    for (uint32_t i = 0; i < semaphoreInfoCount; i++)
    {
      const CgpuWaitSemaphoreInfo& semaphoreInfo = semaphoreInfos[i];

      CGPU_RESOLVE_OR_RETURN_SEMAPHORE(semaphoreInfo.semaphore, isemaphore);

      semaphores[i] = isemaphore->semaphore;
      semaphoreValues[i] = semaphoreInfo.value;
    }
    assert(semaphores.size() == semaphoreValues.size());

    VkSemaphoreWaitInfo waitInfo = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
      .pNext = nullptr,
      .flags = 0, // wait for all semaphores
      .semaphoreCount = (uint32_t) semaphores.size(),
      .pSemaphores = semaphores.data(),
      .pValues = semaphoreValues.data()
    };
    VkResult result = idevice->table.vkWaitSemaphoresKHR(
      idevice->logicalDevice,
      &waitInfo,
      timeoutNs
    );
    if (result != VK_SUCCESS) {
      CGPU_RETURN_ERROR("failed to wait for semaphores");
    }
    return true;
  }

  bool cgpuSubmitCommandBuffer(CgpuDevice device,
                               CgpuCommandBuffer commandBuffer,
                               uint32_t signalSemaphoreInfoCount,
                               CgpuSignalSemaphoreInfo* signalSemaphoreInfos,
                               uint32_t waitSemaphoreInfoCount,
                               CgpuWaitSemaphoreInfo* waitSemaphoreInfos)
  {
    CGPU_RESOLVE_OR_RETURN_DEVICE(device, idevice);
    CGPU_RESOLVE_OR_RETURN_COMMAND_BUFFER(commandBuffer, icommandBuffer);

    GbSmallVector<VkSemaphoreSubmitInfo, 8> signalSubmitInfos(signalSemaphoreInfoCount);
    GbSmallVector<VkSemaphoreSubmitInfo, 8> waitSubmitInfos(waitSemaphoreInfoCount);

    const auto createSubmitInfos = [&](uint32_t infoCount, auto& semaphoreInfos, auto& submitInfos)
    {
      for (uint32_t i = 0; i < infoCount; i++)
      {
        CGPU_RESOLVE_OR_RETURN_SEMAPHORE(semaphoreInfos[i].semaphore, isemaphore);

        submitInfos[i] = VkSemaphoreSubmitInfo {
          .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
          .pNext = nullptr,
          .semaphore = isemaphore->semaphore,
          .value = semaphoreInfos[i].value,
          .stageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
          .deviceIndex = 0, // only relevant if in device group
        };
      }
      return true;
    };

    if (!createSubmitInfos(signalSemaphoreInfoCount, signalSemaphoreInfos, signalSubmitInfos) ||
        !createSubmitInfos(waitSemaphoreInfoCount, waitSemaphoreInfos, waitSubmitInfos))
    {
      CGPU_RETURN_ERROR_INVALID_HANDLE;
    }

    VkCommandBufferSubmitInfo commandBufferSubmitInfo = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
      .pNext = nullptr,
      .commandBuffer = icommandBuffer->commandBuffer,
      .deviceMask = 0
    };

    VkSubmitInfo2KHR submitInfo = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2_KHR,
      .pNext = nullptr,
      .flags = 0,
      .waitSemaphoreInfoCount = (uint32_t) waitSubmitInfos.size(),
      .pWaitSemaphoreInfos = waitSubmitInfos.data(),
      .commandBufferInfoCount = 1,
      .pCommandBufferInfos = &commandBufferSubmitInfo,
      .signalSemaphoreInfoCount = (uint32_t) signalSubmitInfos.size(),
      .pSignalSemaphoreInfos = signalSubmitInfos.data()
    };

    VkResult result = idevice->table.vkQueueSubmit2KHR(
      idevice->computeQueue,
      1,
      &submitInfo,
      VK_NULL_HANDLE
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
    CGPU_RESOLVE_OR_RETURN_DEVICE(device, idevice);
    CGPU_RESOLVE_OR_RETURN_BUFFER(buffer, ibuffer);

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
    CGPU_RESOLVE_OR_RETURN_DEVICE(device, idevice);
    CGPU_RESOLVE_OR_RETURN_BUFFER(buffer, ibuffer);

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
    CGPU_RESOLVE_OR_RETURN_DEVICE(device, idevice);
    memcpy(features, &idevice->features, sizeof(CgpuPhysicalDeviceFeatures));
    return true;
  }

  bool cgpuGetPhysicalDeviceProperties(CgpuDevice device,
                                       CgpuPhysicalDeviceProperties* properties)
  {
    CGPU_RESOLVE_OR_RETURN_DEVICE(device, idevice);
    memcpy(properties, &idevice->properties, sizeof(CgpuPhysicalDeviceProperties));
    return true;
  }
}
