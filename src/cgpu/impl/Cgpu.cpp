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

#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <array>
#include <memory>
#include <atomic>

#include <volk.h>

#include <gtl/gb/Fmt.h>
#include <gtl/gb/Log.h>
#include <gtl/gb/LinearDataStore.h>
#include <gtl/gb/SmallVector.h>

#ifdef __clang__
#pragma clang diagnostic push
// Silence nullability log spam on AppleClang
#pragma clang diagnostic ignored "-Wnullability-completeness"
#endif

#ifdef GTL_VERBOSE
// Uncomment for verbose VMA logging.
//#define VMA_DEBUG_LOG_FORMAT(format, ...) GB_DEBUG_DYN("[VMA] {}", GB_FMT_SPRINTF((format), __VA_ARGS__))
#define VMA_LEAK_LOG_FORMAT(format, ...) do { GB_ERROR_DYN("[VMA] {}", GB_FMT_SPRINTF((format), __VA_ARGS__)); gtl::gbLogFlush(); } while(false)
#endif

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#ifdef __clang__
#pragma clang diagnostic pop
#endif

namespace gtl
{
  /* Constants */

  constexpr static const uint32_t CGPU_MIN_VK_API_VERSION = VK_API_VERSION_1_1;

  constexpr static const uint32_t CGPU_VENDOR_ID_AMD = 0x1002;
  constexpr static const uint32_t CGPU_VENDOR_ID_NVIDIA = 0x10DE;
  constexpr static const uint32_t CGPU_VENDOR_ID_INTEL = 0x8086;
  constexpr static const uint32_t CGPU_VENDOR_ID_MESA = VK_VENDOR_ID_MESA;

  constexpr static const VkShaderStageFlags CGPU_RT_PIPELINE_ACCESS_FLAGS = VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                                                                            VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                                                            VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                                                                            VK_SHADER_STAGE_MISS_BIT_KHR;

  constexpr static const char* CGPU_SHADER_ENTRY_POINT = "main";

  constexpr static const uint32_t CGPU_INITIAL_PHYSICAL_DEVICE_COUNT = 4;

  static const std::array<const char*, 14> CGPU_REQUIRED_EXTENSIONS = {
    VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
    VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME, // required by VK_KHR_acceleration_structure
    VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME, // required by VK_KHR_acceleration_structure
    VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME, // required by VK_KHR_acceleration_structure
    VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
    VK_KHR_SPIRV_1_4_EXTENSION_NAME, // required by VK_KHR_ray_tracing_pipeline
    VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME, // required by VK_KHR_spirv_1_4
    VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME,
    VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
    VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME,
    VK_KHR_MAINTENANCE_5_EXTENSION_NAME,
    VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME, // required by VK_KHR_maintenance5
    VK_KHR_DEPTH_STENCIL_RESOLVE_EXTENSION_NAME, // required by VK_KHR_dynamic_rendering
    VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME // required by VK_KHR_depth_stencil_resolve
  };

  /* Internal structures */

  struct CgpuIDeviceProperties
  {
    VkDriverId  driverID;
    uint32_t minAccelerationStructureScratchOffsetAlignment;
    size_t   minMemoryMapAlignment;
    uint64_t minStorageBufferOffsetAlignment;
    uint64_t minUniformBufferOffsetAlignment;
    uint64_t optimalBufferCopyOffsetAlignment;
    uint64_t optimalBufferCopyRowPitchAlignment;
    uint32_t shaderGroupBaseAlignment;
    uint32_t shaderGroupHandleAlignment;
    uint32_t shaderGroupHandleSize;
  };

  struct CgpuIDeviceFeatures
  {
    bool driverProperties;
    bool maintenance4;
    bool pageableDeviceLocalMemory;
    bool pipelineLibraries;
    bool rayTracingValidation;
  };

  struct CgpuIDevice
  {
    VmaAllocator               allocator;
    VmaPool                    asScratchMemoryPool;
    VkQueue                    computeQueue;
    VkCommandPool              commandPool;
    CgpuDeviceFeatures         features;
    CgpuIDeviceFeatures        internalFeatures;
    CgpuIDeviceProperties      internalProperties;
    VkDevice                   logicalDevice;
    VkPhysicalDevice           physicalDevice;
    VkPipelineCache            pipelineCache;
    CgpuDeviceProperties       properties;
    VolkDeviceTable            table;
  };

  struct CgpuIBuffer
  {
    VmaAllocation allocation;
    VkBuffer      buffer;
    uint64_t      size;

    void*         cpuPtr = nullptr;
    uint64_t      gpuAddress = 0;
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
    VkPipeline                                pipeline;
    VkPipelineLayout                          layout;
    VkDescriptorPool                          descriptorPool;
    uint32_t                                  descriptorSetCount;
    VkDescriptorSetLayout                     descriptorSetLayouts[CGPU_MAX_DESCRIPTOR_SET_COUNT];
    std::vector<VkDescriptorSetLayoutBinding> descriptorSetLayoutBindings[CGPU_MAX_DESCRIPTOR_SET_COUNT];
    VkPipelineBindPoint                       bindPoint;
    VkStridedDeviceAddressRegionKHR           sbtRgen;
    VkStridedDeviceAddressRegionKHR           sbtMiss;
    VkStridedDeviceAddressRegionKHR           sbtHit;
    CgpuBuffer                                sbt;
  };

  struct CgpuIPipelineLibrary
  {
    VkDescriptorPool                          descriptorPool;
    uint32_t                                  descriptorSetCount;
    VkDescriptorSetLayout                     descriptorSetLayouts[CGPU_MAX_DESCRIPTOR_SET_COUNT];
    std::vector<VkDescriptorSetLayoutBinding> descriptorSetLayoutBindings[CGPU_MAX_DESCRIPTOR_SET_COUNT];
    VkPipelineLayout                          layout;
    VkPipeline                                pipeline;
  };

  struct CgpuIShader
  {
    VkShaderModule module = VK_NULL_HANDLE; // null when RT pipeline library is used
    CgpuShaderReflection reflection;
    VkShaderStageFlagBits stageFlags;
    CgpuIPipelineLibrary pipelineLibrary;
  };

  struct CgpuISemaphore
  {
    VkSemaphore semaphore;
  };

  struct CgpuICommandBuffer
  {
    VkCommandBuffer commandBuffer;
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
  };

  struct CgpuISampler
  {
    VkSampler sampler;
  };

  struct CgpuIBindSet
  {
    std::vector<VkDescriptorSetLayoutBinding> descriptorSetLayoutBindings;
    VkDescriptorSet descriptorSet;
  };

  /* Context */

  struct CgpuContext
  {
    VkInstance instance;

    CgpuIDevice idevice;

    GbLinearDataStore<CgpuIBuffer, 16> ibufferStore;
    GbLinearDataStore<CgpuIImage, 128> iimageStore;
    GbLinearDataStore<CgpuIShader, 32> ishaderStore;
    GbLinearDataStore<CgpuIPipeline, 8> ipipelineStore;
    GbLinearDataStore<CgpuISemaphore, 16> isemaphoreStore;
    GbLinearDataStore<CgpuICommandBuffer, 16> icommandBufferStore;
    GbLinearDataStore<CgpuISampler, 8> isamplerStore;
    GbLinearDataStore<CgpuIBlas, 1024> iblasStore;
    GbLinearDataStore<CgpuITlas, 1> itlasStore;
    GbLinearDataStore<CgpuIBindSet, 32> ibindSetStore;

    bool debugUtilsEnabled;
  };

  /* Helper macros */

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

#define CGPU_FATAL(msg)                             \
  do {                                              \
    GB_ERROR("{}:{}: {}", __FILE__, __LINE__, msg); \
    exit(EXIT_FAILURE);                             \
  } while (false)

#define CGPU_RETURN_ERROR_HARDCODED_LIMIT_REACHED   \
  CGPU_RETURN_ERROR("hardcoded limit reached")

#define CGPU_RESOLVE_HANDLE(RESOURCE_NAME, HANDLE_TYPE, IRESOURCE_TYPE, RESOURCE_STORE)                            \
  CGPU_INLINE static bool cgpuResolve##RESOURCE_NAME(CgpuContext* ctx, HANDLE_TYPE handle, IRESOURCE_TYPE** idata) \
  {                                                                                                                \
    return ctx->RESOURCE_STORE.get(handle.handle, idata);                                                          \
  }

  CGPU_RESOLVE_HANDLE(       Buffer,        CgpuBuffer,        CgpuIBuffer,        ibufferStore)
  CGPU_RESOLVE_HANDLE(        Image,         CgpuImage,         CgpuIImage,         iimageStore)
  CGPU_RESOLVE_HANDLE(       Shader,        CgpuShader,        CgpuIShader,        ishaderStore)
  CGPU_RESOLVE_HANDLE(     Pipeline,      CgpuPipeline,      CgpuIPipeline,      ipipelineStore)
  CGPU_RESOLVE_HANDLE(    Semaphore,     CgpuSemaphore,     CgpuISemaphore,     isemaphoreStore)
  CGPU_RESOLVE_HANDLE(CommandBuffer, CgpuCommandBuffer, CgpuICommandBuffer, icommandBufferStore)
  CGPU_RESOLVE_HANDLE(      Sampler,       CgpuSampler,       CgpuISampler,       isamplerStore)
  CGPU_RESOLVE_HANDLE(         Blas,          CgpuBlas,          CgpuIBlas,          iblasStore)
  CGPU_RESOLVE_HANDLE(         Tlas,          CgpuTlas,          CgpuITlas,          itlasStore)
  CGPU_RESOLVE_HANDLE(      BindSet,       CgpuBindSet,       CgpuIBindSet,       ibindSetStore)

#define CGPU_RESOLVE_OR_EXIT(CTX, HANDLE, VAR_NAME, ITYPE, RESOLVE_FUNC) \
  ITYPE* VAR_NAME;                                                       \
  if (!RESOLVE_FUNC(CTX, HANDLE, &VAR_NAME)) [[unlikely]] {                   \
    CGPU_FATAL("invalid handle!");                                       \
  }

#define CGPU_RESOLVE_BUFFER(CTX, HANDLE, VAR_NAME)         CGPU_RESOLVE_OR_EXIT(CTX, HANDLE, VAR_NAME, CgpuIBuffer, cgpuResolveBuffer)
#define CGPU_RESOLVE_IMAGE(CTX, HANDLE, VAR_NAME)          CGPU_RESOLVE_OR_EXIT(CTX, HANDLE, VAR_NAME, CgpuIImage, cgpuResolveImage)
#define CGPU_RESOLVE_SHADER(CTX, HANDLE, VAR_NAME)         CGPU_RESOLVE_OR_EXIT(CTX, HANDLE, VAR_NAME, CgpuIShader, cgpuResolveShader)
#define CGPU_RESOLVE_PIPELINE(CTX, HANDLE, VAR_NAME)       CGPU_RESOLVE_OR_EXIT(CTX, HANDLE, VAR_NAME, CgpuIPipeline, cgpuResolvePipeline)
#define CGPU_RESOLVE_SEMAPHORE(CTX, HANDLE, VAR_NAME)      CGPU_RESOLVE_OR_EXIT(CTX, HANDLE, VAR_NAME, CgpuISemaphore, cgpuResolveSemaphore)
#define CGPU_RESOLVE_COMMAND_BUFFER(CTX, HANDLE, VAR_NAME) CGPU_RESOLVE_OR_EXIT(CTX, HANDLE, VAR_NAME, CgpuICommandBuffer, cgpuResolveCommandBuffer)
#define CGPU_RESOLVE_SAMPLER(CTX, HANDLE, VAR_NAME)        CGPU_RESOLVE_OR_EXIT(CTX, HANDLE, VAR_NAME, CgpuISampler, cgpuResolveSampler)
#define CGPU_RESOLVE_BLAS(CTX, HANDLE, VAR_NAME)           CGPU_RESOLVE_OR_EXIT(CTX, HANDLE, VAR_NAME, CgpuIBlas, cgpuResolveBlas)
#define CGPU_RESOLVE_TLAS(CTX, HANDLE, VAR_NAME)           CGPU_RESOLVE_OR_EXIT(CTX, HANDLE, VAR_NAME, CgpuITlas, cgpuResolveTlas)
#define CGPU_RESOLVE_BIND_SET(CTX, HANDLE, VAR_NAME)       CGPU_RESOLVE_OR_EXIT(CTX, HANDLE, VAR_NAME, CgpuIBindSet, cgpuResolveBindSet)

  /* Helper methods */

  static VkSamplerAddressMode cgpuTranslateAddressMode(CgpuSamplerAddressMode mode)
  {
    switch (mode)
    {
    case CgpuSamplerAddressMode::ClampToEdge: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    case CgpuSamplerAddressMode::Repeat: return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    case CgpuSamplerAddressMode::MirrorRepeat: return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    case CgpuSamplerAddressMode::ClampToBlack: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    default: return VK_SAMPLER_ADDRESS_MODE_MAX_ENUM;
    }
  }

  static VkPipelineStageFlags2KHR CgpuPipelineStageFromShaderStageFlags(VkShaderStageFlags shaderStageFlags)
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
    case CGPU_VENDOR_ID_AMD:
      return "AMD";
    case CGPU_VENDOR_ID_NVIDIA:
      return "NVIDIA";
    case CGPU_VENDOR_ID_INTEL:
      return "Intel";
    case CGPU_VENDOR_ID_MESA:
      return "Mesa";
    default:
      return nullptr;
    }
  }

  template<typename T>
  static T cgpuAlign(T value, T alignment)
  {
      return (value + (alignment - 1)) & ~(alignment - 1);
  }

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

  static bool cgpuFindExtension(const char* name, size_t extensionCount, const VkExtensionProperties* extensions)
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

  /* Static state */

  static bool s_volkInitialized = false;
  static std::mutex s_volkLock;

  /* Implementation */

  static VkResult cgpuCreateMemoryPool(VmaPool& pool,
                                       VmaAllocator allocator,
                                       VkMemoryPropertyFlags memoryProperties,
                                       uint32_t allocationAlignment = 0,
                                       float priority = 0.5f)
  {
    VmaAllocationCreateInfo allocCreateInfo = {};
    allocCreateInfo.requiredFlags = memoryProperties;

    uint32_t memTypeIndex;
    VkResult result = vmaFindMemoryTypeIndex(allocator, UINT32_MAX, &allocCreateInfo, &memTypeIndex);

    if (result != VK_SUCCESS)
    {
      return result;
    }

    VmaPoolCreateInfo poolCreateInfo = {};
    poolCreateInfo.memoryTypeIndex = memTypeIndex;
    poolCreateInfo.priority = priority;
    poolCreateInfo.minAllocationAlignment = allocationAlignment;

    return vmaCreatePool(allocator, &poolCreateInfo, &pool);
  }

  static void cgpuDestroyMemoryPool(VmaAllocator allocator, VmaPool pool)
  {
    vmaDestroyPool(allocator, pool);
  }

  struct CgpuDevicePropertyChain
  {
    VkPhysicalDeviceProperties2 properties2;
    VkPhysicalDeviceDriverPropertiesKHR driver;
    VkPhysicalDeviceAccelerationStructurePropertiesKHR accelerationStructure;
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rayTracingPipeline;
    VkPhysicalDeviceSubgroupProperties subgroup;
  };

  static void cgpuSetupDevicePropertyChain(CgpuDevicePropertyChain& chain,
                                           const VkExtensionProperties* extensions,
                                           uint32_t extensionCount)
  {
    const auto findExtension = [&](const char* name)
    {
      return cgpuFindExtension(name, extensionCount, extensions);
    };

    void* pNext = nullptr;

    chain = {};

    chain.driver = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES_KHR,
      .pNext = pNext
    };

    if (findExtension(VK_KHR_DRIVER_PROPERTIES_EXTENSION_NAME))
    {
      pNext = &chain.driver;
    }

    chain.accelerationStructure = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR,
      .pNext = pNext
    };
    chain.rayTracingPipeline = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR,
      .pNext = &chain.accelerationStructure
    };
    chain.subgroup = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES,
      .pNext = &chain.rayTracingPipeline
    };
    chain.properties2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
      .pNext = &chain.subgroup
    };
  }

  static CgpuDeviceProperties cgpuGetDeviceProperties(const CgpuDevicePropertyChain& chain)
  {
    const VkPhysicalDeviceLimits& limits = chain.properties2.properties.limits;

    return CgpuDeviceProperties {
      .maxComputeSharedMemorySize = limits.maxComputeSharedMemorySize,
      .maxPushConstantsSize = limits.maxPushConstantsSize,
      .maxRayHitAttributeSize = chain.rayTracingPipeline.maxRayHitAttributeSize,
      .subgroupSize = chain.subgroup.subgroupSize
    };
  }

  static CgpuIDeviceProperties cgpuGetInternalDeviceProperties(const CgpuDevicePropertyChain& chain)
  {
    const VkPhysicalDeviceLimits& limits = chain.properties2.properties.limits;

    return CgpuIDeviceProperties {
      .driverID = chain.driver.driverID,
      .minAccelerationStructureScratchOffsetAlignment =
        chain.accelerationStructure.minAccelerationStructureScratchOffsetAlignment,
      .minMemoryMapAlignment = limits.minMemoryMapAlignment,
      .minStorageBufferOffsetAlignment = limits.minStorageBufferOffsetAlignment,
      .minUniformBufferOffsetAlignment = limits.minUniformBufferOffsetAlignment,
      .optimalBufferCopyOffsetAlignment = limits.optimalBufferCopyOffsetAlignment,
      .optimalBufferCopyRowPitchAlignment = limits.optimalBufferCopyRowPitchAlignment,
      .shaderGroupBaseAlignment = chain.rayTracingPipeline.shaderGroupBaseAlignment,
      .shaderGroupHandleAlignment = chain.rayTracingPipeline.shaderGroupHandleAlignment,
      .shaderGroupHandleSize = chain.rayTracingPipeline.shaderGroupHandleSize
    };
  }

  struct CgpuDeviceFeatureChain
  {
    VkPhysicalDeviceFeatures2 features2;
    VkPhysicalDeviceMaintenance4Features maintenance4;
    VkPhysicalDevicePipelineLibraryGroupHandlesFeaturesEXT groupHandles;
    VkPhysicalDeviceMemoryPriorityFeaturesEXT memoryPriority;
    VkPhysicalDevicePageableDeviceLocalMemoryFeaturesEXT pageableDeviceLocalMemory;
    VkPhysicalDeviceShaderClockFeaturesKHR shaderClock;
    VkPhysicalDeviceRayTracingInvocationReorderFeaturesNV rayTracingInvocationReorder;
    VkPhysicalDeviceRayTracingValidationFeaturesNV rayTracingValidation;
    VkPhysicalDeviceMaintenance5FeaturesKHR maintenance5;
    VkPhysicalDeviceTimelineSemaphoreFeaturesKHR timelineSemaphore;
    VkPhysicalDeviceSynchronization2FeaturesKHR synchronization2;
    VkPhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructure;
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rayTracingPipeline;
    VkPhysicalDeviceBufferDeviceAddressFeaturesKHR bufferDeviceAddress;
    VkPhysicalDeviceDescriptorIndexingFeaturesEXT descriptorIndexing;
    VkPhysicalDeviceShaderFloat16Int8Features shaderFloat16Int8;
    VkPhysicalDevice16BitStorageFeatures storage16Bit;
  };

  static void cgpuSetupDeviceFeatureChain(CgpuDeviceFeatureChain& chain,
                                          const VkExtensionProperties* extensions,
                                          uint32_t extensionCount,
                                          bool debugUtilsEnabled,
                                          uint32_t vendorID)
  {
    const auto findExtension = [&](const char* name)
    {
      return cgpuFindExtension(name, extensionCount, extensions);
    };

    void* pNext = nullptr;

    chain = {};

    chain.maintenance4 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES,
      .pNext = pNext
    };

    if (findExtension(VK_KHR_MAINTENANCE_4_EXTENSION_NAME))
    {
      pNext = &chain.maintenance4;
    }

    chain.groupHandles = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_LIBRARY_GROUP_HANDLES_FEATURES_EXT,
      .pNext = pNext
    };

    if (vendorID == CGPU_VENDOR_ID_NVIDIA && // issues on AMD and Intel
        findExtension(VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME) &&
        findExtension(VK_EXT_PIPELINE_LIBRARY_GROUP_HANDLES_EXTENSION_NAME))
    {
      pNext = &chain.groupHandles;
    }

    chain.memoryPriority = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PRIORITY_FEATURES_EXT,
      .pNext = pNext
    };
    chain.pageableDeviceLocalMemory = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PAGEABLE_DEVICE_LOCAL_MEMORY_FEATURES_EXT,
      .pNext = &chain.memoryPriority
    };

    if (findExtension(VK_EXT_MEMORY_PRIORITY_EXTENSION_NAME) &&
        findExtension(VK_EXT_PAGEABLE_DEVICE_LOCAL_MEMORY_EXTENSION_NAME))
    {
      pNext = &chain.pageableDeviceLocalMemory;
    }

    chain.shaderClock = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR,
      .pNext = pNext
    };

#ifndef NDEBUG
    if (findExtension(VK_KHR_SHADER_CLOCK_EXTENSION_NAME))
    {
      pNext = &chain.shaderClock;
    }
#endif

    chain.rayTracingInvocationReorder = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_FEATURES_NV,
      .pNext = pNext
    };

#ifndef NDEBUG
    if (findExtension(VK_NV_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME))
    {
      pNext = &chain.rayTracingInvocationReorder;
    }
#endif

    chain.rayTracingValidation = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_VALIDATION_FEATURES_NV,
      .pNext = pNext
    };

#ifndef NDEBUG
    if (debugUtilsEnabled && findExtension(VK_NV_RAY_TRACING_VALIDATION_EXTENSION_NAME))
    {
      pNext = &chain.rayTracingValidation;
    }
#endif

    chain.maintenance5 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES_KHR,
      .pNext = pNext
    };
    chain.timelineSemaphore = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES,
      .pNext = &chain.maintenance5
    };
    chain.synchronization2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR,
      .pNext = &chain.timelineSemaphore
    };
    chain.accelerationStructure = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
      .pNext = &chain.synchronization2
    };
    chain.rayTracingPipeline = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR,
      .pNext = &chain.accelerationStructure
    };
    chain.bufferDeviceAddress = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES,
      .pNext = &chain.rayTracingPipeline
    };
    chain.descriptorIndexing = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES,
      .pNext = &chain.bufferDeviceAddress
    };
    chain.shaderFloat16Int8 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES,
      .pNext = &chain.descriptorIndexing
    };
    chain.storage16Bit = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES,
      .pNext = &chain.shaderFloat16Int8
    };
    chain.features2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &chain.storage16Bit
    };
  }

  using CgpuExtensionVector = GbSmallVector<const char*, 16>;

  static void cgpuAddFeatureExtensions(const CgpuDeviceFeatures& features,
                                       const CgpuIDeviceFeatures& internalFeatures,
                                       CgpuExtensionVector& extensions)
  {
    if (features.debugPrintf)
    {
      extensions.push_back(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME);
    }
    if (features.rayTracingInvocationReorder)
    {
      extensions.push_back(VK_NV_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME);
    }
    if (features.shaderClock)
    {
      extensions.push_back(VK_KHR_SHADER_CLOCK_EXTENSION_NAME);
    }

    if (internalFeatures.driverProperties)
    {
      extensions.push_back(VK_KHR_DRIVER_PROPERTIES_EXTENSION_NAME);
    }
    if (internalFeatures.maintenance4)
    {
      extensions.push_back(VK_KHR_MAINTENANCE_4_EXTENSION_NAME);
    }
    if (internalFeatures.pageableDeviceLocalMemory)
    {
      extensions.push_back(VK_EXT_MEMORY_PRIORITY_EXTENSION_NAME);
      extensions.push_back(VK_EXT_PAGEABLE_DEVICE_LOCAL_MEMORY_EXTENSION_NAME);
    }
    if (internalFeatures.pipelineLibraries)
    {
      extensions.push_back(VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME);
      extensions.push_back(VK_EXT_PIPELINE_LIBRARY_GROUP_HANDLES_EXTENSION_NAME);
    }
    if (internalFeatures.rayTracingValidation)
    {
      extensions.push_back(VK_NV_RAY_TRACING_VALIDATION_EXTENSION_NAME);
    }
  }

  struct CgpuDeviceCandidate
  {
    VkPhysicalDevice device;
    CgpuExtensionVector enabledExtensions;

    CgpuDevicePropertyChain propertyChain;
    CgpuDeviceFeatureChain featureChain;

    CgpuDeviceFeatures features;
    CgpuIDeviceFeatures internalFeatures;

    uint32_t queueFamilyIndex;
    uint32_t score;
    std::vector<std::string> errorMessages; // if size() > 0: device is unsuitable
  };

  static void cgpuQueryDeviceCandidate(VkPhysicalDevice device, bool debugUtilsEnabled, CgpuDeviceCandidate& c)
  {
    c.device = device;

    // query & check queue
    uint32_t queueFamilyCount;
    vkGetPhysicalDeviceQueueFamilyProperties(c.device, &queueFamilyCount, nullptr);

    GbSmallVector<VkQueueFamilyProperties, 8> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(c.device, &queueFamilyCount, queueFamilies.data());

    c.queueFamilyIndex = UINT32_MAX;
    for (uint32_t i = 0; i < queueFamilyCount; ++i)
    {
      const VkQueueFamilyProperties* queueFamily = &queueFamilies[i];

      if ((queueFamily->queueFlags & VK_QUEUE_COMPUTE_BIT) && (queueFamily->queueFlags & VK_QUEUE_TRANSFER_BIT))
      {
        c.queueFamilyIndex = i;
      }
    }
    if (c.queueFamilyIndex == UINT32_MAX)
    {
      c.errorMessages.push_back("no suitable queue family");
    }

    // query & check memory
    VkPhysicalDeviceMemoryProperties memoryProperties;
    vkGetPhysicalDeviceMemoryProperties(c.device, &memoryProperties);

    VkDeviceSize largestDeviceLocalHeapSize = 0;
    bool isHeapHostAccessible = false;

    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++)
    {
      const auto& memoryType = memoryProperties.memoryTypes[i];
      VkDeviceSize heapSize = memoryProperties.memoryHeaps[memoryType.heapIndex].size;

      if (!bool(memoryType.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) || heapSize < largestDeviceLocalHeapSize)
      {
        continue;
      }

      largestDeviceLocalHeapSize = heapSize;

      if (bool(memoryType.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
      {
        isHeapHostAccessible = true;
      }
      else if (heapSize > largestDeviceLocalHeapSize)
      {
        isHeapHostAccessible = false;
      }
    }

    // query & check extensions
    uint32_t extensionCount;
    vkEnumerateDeviceExtensionProperties(c.device, nullptr, &extensionCount, nullptr);

    std::vector<VkExtensionProperties> extensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(c.device, nullptr, &extensionCount, extensions.data());

    for (uint32_t j = 0; j < CGPU_REQUIRED_EXTENSIONS.size(); j++)
    {
      const char* extension = CGPU_REQUIRED_EXTENSIONS[j];

      if (!cgpuFindExtension(extension, extensionCount, extensions.data()))
      {
        c.errorMessages.push_back(GB_FMT("extension {} missing", extension));
      }

      c.enabledExtensions.push_back(CGPU_REQUIRED_EXTENSIONS[j]);
    }

    const auto findExtension = [&](const char* name)
    {
      return cgpuFindExtension(name, extensions.size(), extensions.data());
    };

    // query & check properties
    cgpuSetupDevicePropertyChain(c.propertyChain, extensions.data(), extensionCount);
    vkGetPhysicalDeviceProperties2(c.device, &c.propertyChain.properties2);

    const VkPhysicalDeviceProperties& properties = c.propertyChain.properties2.properties;

    uint32_t apiVersion = properties.apiVersion;
    if (apiVersion < CGPU_MIN_VK_API_VERSION)
    {
      c.errorMessages.push_back(GB_FMT("outdated Vulkan API {}.{}.{}", VK_API_VERSION_MAJOR(apiVersion),
        VK_API_VERSION_MINOR(apiVersion), VK_API_VERSION_PATCH(apiVersion)));
    }

    // query & check features
    CgpuDeviceFeatureChain tempFeatureChain;
    cgpuSetupDeviceFeatureChain(tempFeatureChain, extensions.data(), extensionCount,
                                debugUtilsEnabled, properties.vendorID);
    vkGetPhysicalDeviceFeatures2(c.device, &tempFeatureChain.features2);

#define CGPU_REQUIRE_FEATURE(STRUCT, FIELD)                              \
      if (tempFeatureChain.STRUCT.FIELD) {                               \
        c.featureChain.STRUCT.FIELD = VK_TRUE;                           \
      } else {                                                           \
        c.errorMessages.push_back(GB_FMT("feature {} missing", #FIELD)); \
      }

    cgpuSetupDeviceFeatureChain(c.featureChain, extensions.data(), extensionCount,
                                debugUtilsEnabled, properties.vendorID);
    CGPU_REQUIRE_FEATURE(maintenance5, maintenance5);
    CGPU_REQUIRE_FEATURE(timelineSemaphore, timelineSemaphore);
    CGPU_REQUIRE_FEATURE(synchronization2, synchronization2);
    CGPU_REQUIRE_FEATURE(accelerationStructure, accelerationStructure);
    CGPU_REQUIRE_FEATURE(rayTracingPipeline, rayTracingPipeline);
    CGPU_REQUIRE_FEATURE(bufferDeviceAddress, bufferDeviceAddress);
    CGPU_REQUIRE_FEATURE(descriptorIndexing, shaderSampledImageArrayNonUniformIndexing);
    CGPU_REQUIRE_FEATURE(descriptorIndexing, descriptorBindingPartiallyBound);
    CGPU_REQUIRE_FEATURE(descriptorIndexing, runtimeDescriptorArray);
    CGPU_REQUIRE_FEATURE(shaderFloat16Int8, shaderFloat16);
    CGPU_REQUIRE_FEATURE(storage16Bit, storageBuffer16BitAccess);
    CGPU_REQUIRE_FEATURE(features2.features, shaderSampledImageArrayDynamicIndexing);
    CGPU_REQUIRE_FEATURE(features2.features, shaderInt16);
    CGPU_REQUIRE_FEATURE(features2.features, shaderInt64);
#undef CGPU_REQUIRE_FEATURE

#define CGPU_ENABLE_FEATURE(STRUCT, FIELD) \
      bool(c.featureChain.STRUCT.FIELD = tempFeatureChain.STRUCT.FIELD)

    bool pageableDeviceLocalMemory = tempFeatureChain.memoryPriority.memoryPriority &&
                                     tempFeatureChain.pageableDeviceLocalMemory.pageableDeviceLocalMemory;

    if (pageableDeviceLocalMemory)
    {
      c.featureChain.memoryPriority.memoryPriority = VK_TRUE;
      c.featureChain.pageableDeviceLocalMemory.pageableDeviceLocalMemory = VK_TRUE;
    }

    c.features = {
      .debugPrintf = findExtension(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME),
      .rayTracingInvocationReorder = CGPU_ENABLE_FEATURE(rayTracingInvocationReorder, rayTracingInvocationReorder),
      .shaderClock = CGPU_ENABLE_FEATURE(shaderClock, shaderSubgroupClock),
      .sharedMemory = isHeapHostAccessible // UMA or ReBAR
    };

    c.internalFeatures =
    {
      .driverProperties = findExtension(VK_KHR_DRIVER_PROPERTIES_EXTENSION_NAME),
      .maintenance4 = CGPU_ENABLE_FEATURE(maintenance4, maintenance4),
      .pageableDeviceLocalMemory = pageableDeviceLocalMemory,
      .pipelineLibraries = CGPU_ENABLE_FEATURE(groupHandles, pipelineLibraryGroupHandles),
      .rayTracingValidation = CGPU_ENABLE_FEATURE(rayTracingValidation, rayTracingValidation)
    };

    cgpuAddFeatureExtensions(c.features, c.internalFeatures, c.enabledExtensions);
#undef CGPU_ENABLE_FEATURE

    // calculate score
    c.score = 0;

    if (!c.errorMessages.empty())
    {
      return;
    }

    if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
    {
      c.score += 10000;
    }
    else if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU)
    {
      c.score += 8000; // can be a masked dGPU
    }

    c.score += int(largestDeviceLocalHeapSize / uint64_t(1024 * 1024 * 1024)); // bytes to gigabytes
  }

  using CgpuCandidateVector = GbSmallVector<CgpuDeviceCandidate, CGPU_INITIAL_PHYSICAL_DEVICE_COUNT>;

  static CgpuCandidateVector cgpuQueryDeviceCandidates(VkInstance instance, bool debugUtilsEnabled)
  {
    uint32_t deviceCount;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

    if (deviceCount == 0)
    {
      return {};
    }

    GbSmallVector<VkPhysicalDevice, CGPU_INITIAL_PHYSICAL_DEVICE_COUNT> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    CgpuCandidateVector candidates(deviceCount);

    for (uint32_t deviceIdx = 0; deviceIdx < deviceCount; deviceIdx++)
    {
      CgpuDeviceCandidate& c = candidates[deviceIdx];

      cgpuQueryDeviceCandidate(devices[deviceIdx], debugUtilsEnabled, c);
    }

    return candidates;
  }

  static void cgpuPrintEnabledFeatures(const CgpuDeviceFeatures& features, const CgpuIDeviceFeatures& internalFeatures)
  {
    GB_LOG("Optional features:");

#define CGPU_PRINT_FEATURE(STRUCT, FIELD) \
    if (STRUCT.FIELD) GB_LOG("- " #FIELD);

    CGPU_PRINT_FEATURE(features,         debugPrintf);
    CGPU_PRINT_FEATURE(internalFeatures, driverProperties);
    CGPU_PRINT_FEATURE(internalFeatures, maintenance4);
    CGPU_PRINT_FEATURE(internalFeatures, pageableDeviceLocalMemory);
    CGPU_PRINT_FEATURE(internalFeatures, pipelineLibraries);
    CGPU_PRINT_FEATURE(features,         rayTracingInvocationReorder);
    CGPU_PRINT_FEATURE(internalFeatures, rayTracingValidation);
    CGPU_PRINT_FEATURE(features,         shaderClock);
    CGPU_PRINT_FEATURE(features,         sharedMemory);

#undef CGPU_PRINT_FEATURE
  }

  static bool cgpuCreateIDevice(VkInstance instance, bool debugUtilsEnabled, CgpuIDevice* idevice)
  {
    // query & sort devices
    CgpuCandidateVector candidates = cgpuQueryDeviceCandidates(instance, debugUtilsEnabled);

    if (candidates.empty())
    {
      GB_ERROR("no GPUs found");
      return false;
    }

    std::sort(candidates.begin(), candidates.end(), [](const CgpuDeviceCandidate& a, const CgpuDeviceCandidate& b) {
      return a.score > b.score;
    });

    uint32_t deviceIndex = 0;
    if (const char* envStr = getenv("GTL_DEVICE_INDEX_OVERRIDE"); envStr)
    {
      int newDeviceIndex = strtol(envStr, nullptr, 10);

      deviceIndex = uint32_t(newDeviceIndex < 0 ? 0 :
        (newDeviceIndex >= candidates.size() ? candidates.size() - 1 : newDeviceIndex));
    }

    GB_LOG("Device list:");
    for (uint32_t i = 0; i < candidates.size(); i++)
    {
      const CgpuDeviceCandidate& candidate = candidates[i];
      const VkPhysicalDeviceProperties& properties = candidate.propertyChain.properties2.properties;

      std::string idxStr = (i == deviceIndex) ? "x" : GB_FMT("{}", i);

      GB_LOG("[{}] ({}) {}", idxStr, candidate.score, properties.deviceName);

      for (const std::string& msg : candidate.errorMessages)
      {
        GB_LOG("  - {}", msg);
      }
    }

    if (candidates[deviceIndex].score == 0)
    {
      GB_ERROR("GPU not suitable");
      return false;
    }

    const CgpuDeviceCandidate& candidate = candidates[deviceIndex];

    // print info
    const VkPhysicalDeviceProperties& properties = candidate.propertyChain.properties2.properties;

    GB_LOG("Selected device {}:", deviceIndex);
    uint32_t apiVersion = properties.apiVersion;
    {
      uint32_t major = VK_VERSION_MAJOR(apiVersion);
      uint32_t minor = VK_VERSION_MINOR(apiVersion);
      uint32_t patch = VK_VERSION_PATCH(apiVersion);
      GB_LOG("> API version: {}.{}.{}", major, minor, patch);
    }

    GB_LOG("> name: {}", properties.deviceName);

    if (const char* vendor = cgpuGetVendorName(properties.vendorID); vendor)
    {
      GB_LOG("> vendor: {}", vendor);
    }
    else
    {
      GB_LOG("> vendor: Unknown ({:#08x})", properties.vendorID);
    }

    if (candidate.internalFeatures.driverProperties)
    {
      GB_LOG("> driver: {} ({})", candidate.propertyChain.driver.driverName,
                                  candidate.propertyChain.driver.driverInfo);
    }

    cgpuPrintEnabledFeatures(candidate.features, candidate.internalFeatures);

    // create device
    *idevice = {};

    idevice->physicalDevice = candidate.device;

    idevice->features = candidate.features;
    idevice->internalFeatures = candidate.internalFeatures;
    idevice->properties = cgpuGetDeviceProperties(candidate.propertyChain);
    idevice->internalProperties = cgpuGetInternalDeviceProperties(candidate.propertyChain);

    const float queuePriority = 1.0f;

    VkDeviceQueueCreateInfo queueCreateInfo = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .queueFamilyIndex = (uint32_t) candidate.queueFamilyIndex,
      .queueCount = 1,
      .pQueuePriorities = &queuePriority,
    };

    VkDeviceCreateInfo deviceCreateInfo = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &candidate.featureChain,
      .flags = 0,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &queueCreateInfo,
      /* These two fields are ignored by up-to-date implementations since
       * nowadays, there is no difference to instance validation layers. */
      .enabledLayerCount = 0,
      .ppEnabledLayerNames = nullptr,
      .enabledExtensionCount = (uint32_t) candidate.enabledExtensions.size(),
      .ppEnabledExtensionNames = candidate.enabledExtensions.data(),
      .pEnabledFeatures = nullptr,
    };

    VkResult result = vkCreateDevice(
      idevice->physicalDevice,
      &deviceCreateInfo,
      nullptr,
      &idevice->logicalDevice
    );
    if (result != VK_SUCCESS)
    {
      CGPU_RETURN_ERROR("failed to create device");
    }

    volkLoadDeviceTable(&idevice->table, idevice->logicalDevice);

    idevice->table.vkGetDeviceQueue(
      idevice->logicalDevice,
      candidate.queueFamilyIndex,
      0,
      &idevice->computeQueue
    );

    VkCommandPoolCreateInfo poolCreateInfo = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .pNext = nullptr,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      .queueFamilyIndex = candidate.queueFamilyIndex,
    };

    result = idevice->table.vkCreateCommandPool(
      idevice->logicalDevice,
      &poolCreateInfo,
      nullptr,
      &idevice->commandPool
    );

    if (result != VK_SUCCESS)
    {
      idevice->table.vkDestroyDevice(idevice->logicalDevice, nullptr);
      CGPU_RETURN_ERROR("failed to create command pool");
    }

    VmaVulkanFunctions vmaVulkanFunctions = {
      .vkGetInstanceProcAddr = vkGetInstanceProcAddr,
      .vkGetDeviceProcAddr = vkGetDeviceProcAddr,
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

    VmaAllocatorCreateFlags allocatorCreateFlags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT |
                                                   VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE5_BIT;
    if (candidate.internalFeatures.pageableDeviceLocalMemory)
    {
      allocatorCreateFlags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_PRIORITY_BIT;
    }
    if (candidate.internalFeatures.maintenance4)
    {
      allocatorCreateFlags |= VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE4_BIT;
    }

    VmaAllocatorCreateInfo allocCreateInfo = {};
    allocCreateInfo.flags = allocatorCreateFlags;
    allocCreateInfo.vulkanApiVersion = CGPU_MIN_VK_API_VERSION;
    allocCreateInfo.physicalDevice = idevice->physicalDevice;
    allocCreateInfo.device = idevice->logicalDevice;
    allocCreateInfo.instance = instance;
    allocCreateInfo.pVulkanFunctions = &vmaVulkanFunctions;

    result = vmaCreateAllocator(&allocCreateInfo, &idevice->allocator);

    if (result != VK_SUCCESS)
    {
      idevice->table.vkDestroyCommandPool(idevice->logicalDevice, idevice->commandPool, nullptr);
      idevice->table.vkDestroyDevice(idevice->logicalDevice, nullptr);

      CGPU_RETURN_ERROR("failed to create vma allocator");
    }

    VkMemoryPropertyFlags asScratchMemoryProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    if (idevice->features.sharedMemory)
    {
      asScratchMemoryProperties |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    }

    float asScratchMemoryPriority = 1.0f;
    result = cgpuCreateMemoryPool(idevice->asScratchMemoryPool, idevice->allocator,
                                  asScratchMemoryProperties,
                                  idevice->internalProperties.minAccelerationStructureScratchOffsetAlignment,
                                  asScratchMemoryPriority);

    if (result != VK_SUCCESS)
    {
      vmaDestroyAllocator(idevice->allocator);
      idevice->table.vkDestroyCommandPool(idevice->logicalDevice, idevice->commandPool, nullptr);
      idevice->table.vkDestroyDevice(idevice->logicalDevice, nullptr);

      CGPU_RETURN_ERROR("failed to create AS scratch memory pool");
    }

    vmaSetPoolName(idevice->allocator, idevice->asScratchMemoryPool, "[AS scratch memory pool]");

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
      GB_WARN("{}:{}: {}", __FILE__, __LINE__, "failed to create pipeline cache");

      idevice->pipelineCache = VK_NULL_HANDLE;
    }

    return true;
  }

  static void cgpuDestroyIDevice(CgpuContext* ctx, CgpuIDevice* idevice)
  {
    cgpuDestroyMemoryPool(idevice->allocator, idevice->asScratchMemoryPool);

    if (idevice->pipelineCache != VK_NULL_HANDLE)
    {
      idevice->table.vkDestroyPipelineCache(idevice->logicalDevice, idevice->pipelineCache, nullptr);
    }

    idevice->table.vkDestroyCommandPool(idevice->logicalDevice, idevice->commandPool, nullptr);

    vmaDestroyAllocator(idevice->allocator);

    idevice->table.vkDestroyDevice(idevice->logicalDevice, nullptr);
  }

  CgpuContext* cgpuCreateContext(const char* appName, uint32_t versionMajor, uint32_t versionMinor, uint32_t versionPatch)
  {
    // refcount Volk initialization
    {
      std::lock_guard guard(s_volkLock);

      if (!s_volkInitialized && volkInitialize() != VK_SUCCESS)
      {
        GB_ERROR("failed to initialize volk");
        return nullptr;
      }

      s_volkInitialized = true;
    }

    uint32_t instanceVersion = volkGetInstanceVersion();
    GB_LOG("Vulkan instance:");
    GB_LOG("> version {}.{}.{}", VK_VERSION_MAJOR(instanceVersion), VK_VERSION_MINOR(instanceVersion), VK_VERSION_PATCH(instanceVersion));

    if (instanceVersion < CGPU_MIN_VK_API_VERSION)
    {
      GB_ERROR("Vulkan instance version does match minimum of {}.{}.{}",
        VK_VERSION_MAJOR(CGPU_MIN_VK_API_VERSION), VK_VERSION_MINOR(CGPU_MIN_VK_API_VERSION),
        VK_VERSION_PATCH(CGPU_MIN_VK_API_VERSION));
      return nullptr;
    }

    GbSmallVector<const char*, 4> enabledLayers;
    GbSmallVector<const char*, 16> enabledExtensions;
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
      }

      if (enabledLayers.size() > 0)
      {
        GB_LOG("> layers: {}", enabledLayers);
      }
    }
#endif

    {
      uint32_t extensionCount;
      vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);

      std::vector<VkExtensionProperties> availableExtensions(extensionCount);
      vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, availableExtensions.data());

#ifndef NDEBUG
      if (cgpuFindExtension(VK_EXT_DEBUG_UTILS_EXTENSION_NAME, availableExtensions.size(), availableExtensions.data()))
      {
        enabledExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        debugUtilsEnabled = true;
      }
#endif

      if (enabledExtensions.size() > 0)
      {
        GB_LOG("> extensions: {}", enabledExtensions);
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
      .flags = 0,
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
        return nullptr;
      }
    }

    volkLoadInstanceOnly(instance);

    CgpuIDevice idevice;
    if (!cgpuCreateIDevice(instance, debugUtilsEnabled, &idevice))
    {
      vkDestroyInstance(instance, nullptr);
      return nullptr;
    }

    CgpuContext* ctx = new CgpuContext {
      .instance = instance,
      .idevice = idevice,
      .debugUtilsEnabled = debugUtilsEnabled
    };

    return ctx;
  }

  void cgpuDestroyContext(CgpuContext* ctx)
  {
    cgpuDestroyIDevice(ctx, &ctx->idevice);
    vkDestroyInstance(ctx->instance, nullptr);
    delete ctx;
  }

  static void cgpuCreatePipelineLayout(CgpuIDevice* idevice,
                                       VkDescriptorSetLayout* descriptorSetLayouts,
                                       uint32_t descriptorSetCount,
                                       CgpuIShader* ishader,
                                       VkShaderStageFlags stageFlags,
                                       VkPipelineLayout* pipelineLayout)
  {
    VkPushConstantRange pushConstRange = {
      .stageFlags = stageFlags,
      .offset = 0,
      .size = ishader->reflection.pushConstantsSize
    };

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .setLayoutCount = descriptorSetCount,
      .pSetLayouts = descriptorSetLayouts,
      .pushConstantRangeCount = pushConstRange.size ? 1u : 0u,
      .pPushConstantRanges = &pushConstRange
    };

    VkResult result = idevice->table.vkCreatePipelineLayout(idevice->logicalDevice,
                                                            &pipelineLayoutCreateInfo,
                                                            nullptr,
                                                            pipelineLayout);

    if (result != VK_SUCCESS)
    {
      CGPU_FATAL("failed to create pipeline layout");
    }
  }

  static void cgpuCreatePipelineDescriptorSet(CgpuIDevice* idevice,
                                              const std::vector<CgpuShaderReflectionBinding>& bindings,
                                              VkShaderStageFlags stageFlags,
                                              VkDescriptorPool& descriptorPool,
                                              VkDescriptorSetLayout& descriptorSetLayout,
                                              std::vector<VkDescriptorSetLayoutBinding>& descriptorSetLayoutBindings)
  {
    size_t bindingCount = bindings.size();

    std::vector<VkDescriptorBindingFlagsEXT> bindingFlags(bindingCount, 0);
    descriptorSetLayoutBindings.resize(bindingCount);

    for (uint32_t i = 0; i < bindingCount; i++)
    {
      const CgpuShaderReflectionBinding& bindingReflection = bindings[i];

      VkDescriptorType descriptorType = (VkDescriptorType) bindingReflection.descriptorType;
      if (descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
      {
        descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
      }

      VkDescriptorSetLayoutBinding layoutBinding = {
        .binding = bindingReflection.binding,
        .descriptorType = descriptorType,
        .descriptorCount = bindingReflection.count,
        .stageFlags = stageFlags,
        .pImmutableSamplers = nullptr,
      };

      if (descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ||
          descriptorType == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE ||
          descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
      {
        bindingFlags[i] = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT_EXT;
      }

      descriptorSetLayoutBindings[i] = layoutBinding;
    }

    VkDescriptorSetLayoutBindingFlagsCreateInfoEXT layoutBindingFlags {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
      .pNext = nullptr,
      .bindingCount = (uint32_t) bindingCount,
      .pBindingFlags = bindingFlags.data()
    };

    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .pNext = &layoutBindingFlags,
      .flags = 0,
      .bindingCount = (uint32_t) bindingCount,
      .pBindings = descriptorSetLayoutBindings.data(),
    };

    VkResult result = idevice->table.vkCreateDescriptorSetLayout(
      idevice->logicalDevice,
      &descriptorSetLayoutCreateInfo,
      nullptr,
      &descriptorSetLayout
    );

    if (result != VK_SUCCESS)
    {
      CGPU_FATAL("failed to create descriptor set layout");
    }
  }

  static void cgpuCreatePipelineDescriptorSets(CgpuIDevice* idevice,
                                               CgpuIShader* ishader,
                                               VkShaderStageFlags stageFlags,
                                               VkDescriptorPool& descriptorPool,
                                               VkDescriptorSetLayout* descriptorSetLayouts,
                                               std::vector<VkDescriptorSetLayoutBinding>* descriptorSetLayoutBindings,
                                               uint32_t& descriptorSetCount)
  {
    const CgpuShaderReflection* shaderReflection = &ishader->reflection;
    const std::vector<CgpuShaderReflectionDescriptorSet>& descriptorSets = shaderReflection->descriptorSets;

    descriptorSetCount = uint32_t(descriptorSets.size());
    if (descriptorSetCount >= CGPU_MAX_DESCRIPTOR_SET_COUNT)
    {
      CGPU_FATAL("max descriptor set count exceeded");
    }

    for (uint32_t i = 0; i < descriptorSetCount; i++)
    {
      const CgpuShaderReflectionDescriptorSet& descriptorSet = descriptorSets[i];
      const std::vector<CgpuShaderReflectionBinding>& bindings = descriptorSet.bindings;

      cgpuCreatePipelineDescriptorSet(idevice, bindings, stageFlags, descriptorPool, descriptorSetLayouts[i], descriptorSetLayoutBindings[i]);
    }
  }

  static void cgpuCreateRtPipelineLibrary(CgpuIDevice* idevice,
                                          const VkShaderModuleCreateInfo& moduleCreateInfo,
                                          CgpuIShader* ishader,
                                          VkShaderStageFlags stageFlags,
                                          uint32_t maxRayPayloadSize,
                                          uint32_t maxRayHitAttributeSize)
  {
    CgpuIPipelineLibrary& library = ishader->pipelineLibrary;

    cgpuCreatePipelineDescriptorSets(
      idevice, ishader, stageFlags,
      library.descriptorPool,
      library.descriptorSetLayouts,
      library.descriptorSetLayoutBindings,
      library.descriptorSetCount
    );

    cgpuCreatePipelineLayout(idevice, library.descriptorSetLayouts, library.descriptorSetCount,
                             ishader, stageFlags, &library.layout);

    VkPipelineShaderStageCreateInfo stageCreateInfo = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .pNext = &moduleCreateInfo,
      .flags = 0,
      .stage = ishader->stageFlags,
      .module = VK_NULL_HANDLE,
      .pName = CGPU_SHADER_ENTRY_POINT,
      .pSpecializationInfo = nullptr
    };

    VkRayTracingPipelineInterfaceCreateInfoKHR interfaceCreateInfo {
      .sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_INTERFACE_CREATE_INFO_KHR,
      .pNext = nullptr,
      .maxPipelineRayPayloadSize = maxRayPayloadSize,
      .maxPipelineRayHitAttributeSize = maxRayHitAttributeSize
    };

    VkRayTracingPipelineCreateInfoKHR rtPipelineCreateInfo = {
      .sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
      .pNext = nullptr,
      .flags = VK_PIPELINE_CREATE_LIBRARY_BIT_KHR,
      .stageCount = 1,
      .pStages = &stageCreateInfo,
      .groupCount = 0,
      .pGroups = nullptr,
      .maxPipelineRayRecursionDepth = 1,
      .pLibraryInfo = nullptr,
      .pLibraryInterface = &interfaceCreateInfo,
      .pDynamicState = nullptr,
      .layout = library.layout,
      .basePipelineHandle = VK_NULL_HANDLE,
      .basePipelineIndex = -1
    };

    if (idevice->table.vkCreateRayTracingPipelinesKHR(idevice->logicalDevice,
                                                      VK_NULL_HANDLE,
                                                      idevice->pipelineCache,
                                                      1,
                                                      &rtPipelineCreateInfo,
                                                      nullptr,
                                                      &library.pipeline) != VK_SUCCESS)
    {
      CGPU_FATAL("failed to create RT pipeline library");
    }
  }

  static void cgpuCreateShader(CgpuContext* ctx,
                               const CgpuShaderCreateInfo& createInfo,
                               CgpuIShader* ishader)
  {
    CgpuIDevice* idevice = &ctx->idevice;

    ishader->stageFlags = (VkShaderStageFlagBits)createInfo.stageFlags;

    if (!cgpuReflectShader((uint32_t*) createInfo.source, createInfo.size, &ishader->reflection))
    {
      CGPU_FATAL("failed to reflect shader");
    }

#ifndef NDEBUG
    if (createInfo.stageFlags != CgpuShaderStage::Compute)
    {
      assert(createInfo.maxRayPayloadSize > 0);
      assert(createInfo.maxRayHitAttributeSize > 0);
      assert(ishader->reflection.maxRayPayloadSize <= createInfo.maxRayPayloadSize);
      assert(ishader->reflection.maxRayHitAttributeSize <= createInfo.maxRayHitAttributeSize);
    }
#endif

    VkShaderModuleCreateInfo moduleCreateInfo = {
     .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
     .pNext = nullptr,
     .flags = 0,
     .codeSize = createInfo.size,
     .pCode = (uint32_t*) createInfo.source,
    };

    if (!idevice->internalFeatures.pipelineLibraries || createInfo.stageFlags == CgpuShaderStage::Compute)
    {
      VkResult result = idevice->table.vkCreateShaderModule(
        idevice->logicalDevice,
        &moduleCreateInfo,
        nullptr,
        &ishader->module
      );

      if (result != VK_SUCCESS)
      {
        CGPU_FATAL("failed to create shader module");
      }

      if (ctx->debugUtilsEnabled && createInfo.debugName)
      {
        cgpuSetObjectName(idevice->logicalDevice, VK_OBJECT_TYPE_SHADER_MODULE,
                          (uint64_t) ishader->module, createInfo.debugName);
      }
    }
    else
    {
      cgpuCreateRtPipelineLibrary(idevice, moduleCreateInfo, ishader, CGPU_RT_PIPELINE_ACCESS_FLAGS,
                                  createInfo.maxRayPayloadSize, createInfo.maxRayHitAttributeSize);
    }
  }

  bool cgpuCreateShader(CgpuContext* ctx,
                        CgpuShaderCreateInfo createInfo,
                        CgpuShader* shader)
  {
    shader->handle = ctx->ishaderStore.allocate();

    CGPU_RESOLVE_SHADER(ctx, *shader, ishader);

    cgpuCreateShader(ctx, createInfo, ishader);

    return true;
  }

  bool cgpuCreateShadersParallel(CgpuContext* ctx,
                         uint32_t shaderCount,
                         CgpuShaderCreateInfo* createInfos,
                         CgpuShader* shaders)
  {
    for (uint32_t i = 0; i < shaderCount; i++)
    {
      shaders[i].handle = ctx->ishaderStore.allocate();
    }

    std::vector<CgpuIShader*> ishaders;
    ishaders.resize(shaderCount, nullptr);

    for (uint32_t i = 0; i < shaderCount; i++)
    {
      CGPU_RESOLVE_SHADER(ctx, shaders[i], ishader);
      ishaders[i] = ishader;
    }

    // TODO: proper error handling
#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < int(shaderCount); i++)
    {
      cgpuCreateShader(ctx, createInfos[i], ishaders[i]);
    }

    return true;
  }

  void cgpuDestroyShader(CgpuContext* ctx, CgpuShader shader)
  {
    CgpuIDevice* idevice = &ctx->idevice;
    CGPU_RESOLVE_SHADER(ctx, shader, ishader);

    const CgpuIPipelineLibrary& library = ishader->pipelineLibrary;
    idevice->table.vkDestroyPipeline(idevice->logicalDevice, library.pipeline, nullptr);
    idevice->table.vkDestroyPipelineLayout(idevice->logicalDevice, library.layout, nullptr);
    for (uint32_t i = 0; i < library.descriptorSetCount; i++)
    {
      idevice->table.vkDestroyDescriptorSetLayout(idevice->logicalDevice, library.descriptorSetLayouts[i], nullptr);
    }
    idevice->table.vkDestroyDescriptorPool(idevice->logicalDevice, library.descriptorPool, nullptr);

    if (ishader->module != VK_NULL_HANDLE)
    {
      idevice->table.vkDestroyShaderModule(idevice->logicalDevice, ishader->module, nullptr);
    }

    ctx->ishaderStore.free(shader.handle);
  }

  static void cgpuDestroyIBuffer(CgpuIDevice* idevice, CgpuIBuffer* ibuffer)
  {
    if (ibuffer->cpuPtr)
    {
      vmaUnmapMemory(idevice->allocator, ibuffer->allocation);
    }
    vmaDestroyBuffer(idevice->allocator, ibuffer->buffer, ibuffer->allocation);
  }

  static bool cgpuCreateIBuffer(CgpuIDevice* idevice,
                                CgpuBufferUsage usage,
                                CgpuMemoryProperties memoryProperties,
                                uint64_t size,
                                uint64_t alignment,
                                CgpuIBuffer* ibuffer,
                                const char* debugName,
                                VmaPool memoryPool = VK_NULL_HANDLE)
  {
    constexpr static uint64_t BASE_ALIGNMENT = 32; // size of largest math primitive (vec4); ensure that
                                                   // compiler can emit wide loads.

    uint64_t newSize = cgpuAlign(size, BASE_ALIGNMENT); // required for vkCmdFillBuffer to clear whole range

    if (idevice->features.sharedMemory)
    {
      memoryProperties |= CgpuMemoryProperties::HostVisible;
    }

    VkBufferCreateInfo bufferInfo = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .size = newSize,
      .usage = (VkBufferUsageFlags) usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .queueFamilyIndexCount = 0,
      .pQueueFamilyIndices = nullptr,
    };

    float priority = 0.5f; // higher than images
    if (bool(usage & (CgpuBufferUsage::AccelerationStructureBuild | CgpuBufferUsage::AccelerationStructureStorage |
                      CgpuBufferUsage::ShaderBindingTable | CgpuBufferUsage::ShaderDeviceAddress)))
    {
      priority = 1.0f;
    }

    VmaAllocationCreateInfo allocCreateInfo = {};
    allocCreateInfo.requiredFlags = (VkMemoryPropertyFlags) memoryProperties;
    allocCreateInfo.pool = memoryPool;
    allocCreateInfo.priority = priority;

    size_t newAlignment = alignment;
    size_t mmapAlign = idevice->internalProperties.minMemoryMapAlignment;

    if (bool(memoryProperties & CgpuMemoryProperties::HostVisible) && alignment < mmapAlign)
    {
      alignment = mmapAlign;
    }

    newAlignment = cgpuAlign(alignment, BASE_ALIGNMENT);

    VkResult result = vmaCreateBufferWithAlignment(
      idevice->allocator,
      &bufferInfo,
      &allocCreateInfo,
      newAlignment,
      &ibuffer->buffer,
      &ibuffer->allocation,
      nullptr
    );

    if (result != VK_SUCCESS)
    {
      CGPU_RETURN_ERROR("failed to create buffer");
    }

    if (debugName)
    {
      vmaSetAllocationName(idevice->allocator, ibuffer->allocation, debugName);
    }
    
    if (bool(usage & CgpuBufferUsage::ShaderDeviceAddress))
    {
      VkBufferDeviceAddressInfoKHR addressInfo = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .pNext = nullptr,
        .buffer = ibuffer->buffer,
      };

      ibuffer->gpuAddress = idevice->table.vkGetBufferDeviceAddressKHR(idevice->logicalDevice, &addressInfo);
    }

    if (bool(memoryProperties & CgpuMemoryProperties::HostVisible) &&
        vmaMapMemory(idevice->allocator, ibuffer->allocation, &ibuffer->cpuPtr) != VK_SUCCESS)
    {
      cgpuDestroyIBuffer(idevice, ibuffer);
      CGPU_RETURN_ERROR("failed to map buffer memory");
    }

    ibuffer->size = newSize;

    return true;
  }

  bool cgpuCreateBuffer(CgpuContext* ctx,
                        CgpuBufferCreateInfo createInfo,
                        CgpuBuffer* buffer)
  {
    CgpuIDevice* idevice = &ctx->idevice;

    uint64_t handle = ctx->ibufferStore.allocate();

    CGPU_RESOLVE_BUFFER(ctx, { handle }, ibuffer);

    assert(createInfo.size > 0);

    if (!cgpuCreateIBuffer(idevice, createInfo.usage, createInfo.memoryProperties, createInfo.size,
                           createInfo.alignment, ibuffer, createInfo.debugName))
    {
      ctx->ibufferStore.free(handle);
      CGPU_RETURN_ERROR("failed to create buffer");
    }

    if (ctx->debugUtilsEnabled && createInfo.debugName)
    {
      cgpuSetObjectName(idevice->logicalDevice, VK_OBJECT_TYPE_BUFFER, (uint64_t) ibuffer->buffer, createInfo.debugName);
    }

    buffer->handle = handle;
    return true;
  }

  void cgpuDestroyBuffer(CgpuContext* ctx, CgpuBuffer buffer)
  {
    CgpuIDevice* idevice = &ctx->idevice;
    CGPU_RESOLVE_BUFFER(ctx, buffer, ibuffer);

    cgpuDestroyIBuffer(idevice, ibuffer);

    ctx->ibufferStore.free(buffer.handle);
  }

  void* cgpuGetBufferCpuPtr(CgpuContext* ctx, CgpuBuffer buffer)
  {
    CGPU_RESOLVE_BUFFER(ctx, buffer, ibuffer);

    return ibuffer->cpuPtr;
  }

  uint64_t cgpuGetBufferGpuAddress(CgpuContext* ctx, CgpuBuffer buffer)
  {
    CGPU_RESOLVE_BUFFER(ctx, buffer, ibuffer);

    return ibuffer->gpuAddress;
  }

  bool cgpuCreateImage(CgpuContext* ctx,
                       CgpuImageCreateInfo createInfo,
                       CgpuImage* image)
  {
    CgpuIDevice* idevice = &ctx->idevice;

    uint64_t handle = ctx->iimageStore.allocate();

    CGPU_RESOLVE_IMAGE(ctx, { handle }, iimage);

    // FIXME: check device support
    VkImageTiling vkImageTiling = VK_IMAGE_TILING_OPTIMAL;
    if (!createInfo.is3d && bool((createInfo.usage & CgpuImageUsage::TransferSrc) | (createInfo.usage & CgpuImageUsage::TransferDst)))
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

    if (result != VK_SUCCESS)
    {
      ctx->iimageStore.free(handle);
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
      ctx->iimageStore.free(handle);
      vmaDestroyImage(idevice->allocator, iimage->image, iimage->allocation);
      CGPU_RETURN_ERROR("failed to create image view");
    }

    if (ctx->debugUtilsEnabled && createInfo.debugName)
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

  void cgpuDestroyImage(CgpuContext* ctx, CgpuImage image)
  {
    CgpuIDevice* idevice = &ctx->idevice;
    CGPU_RESOLVE_IMAGE(ctx, image, iimage);

    idevice->table.vkDestroyImageView(
      idevice->logicalDevice,
      iimage->imageView,
      nullptr
    );

    vmaDestroyImage(idevice->allocator, iimage->image, iimage->allocation);

    ctx->iimageStore.free(image.handle);
  }

  bool cgpuCreateSampler(CgpuContext* ctx,
                         CgpuSamplerCreateInfo createInfo,
                         CgpuSampler* sampler)
  {
    CgpuIDevice* idevice = &ctx->idevice;

    uint64_t handle = ctx->isamplerStore.allocate();

    CGPU_RESOLVE_SAMPLER(ctx, { handle }, isampler);

    // Emulate MDL's clip wrap mode if necessary; use optimal mode (according to ARM) if not.
    bool clampToBlack = (createInfo.addressModeU == CgpuSamplerAddressMode::ClampToBlack) ||
                        (createInfo.addressModeV == CgpuSamplerAddressMode::ClampToBlack) ||
                        (createInfo.addressModeW == CgpuSamplerAddressMode::ClampToBlack);

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

    if (result != VK_SUCCESS)
    {
      ctx->isamplerStore.free(handle);
      CGPU_RETURN_ERROR("failed to create sampler");
    }

    sampler->handle = handle;
    return true;
  }

  void cgpuDestroySampler(CgpuContext* ctx, CgpuSampler sampler)
  {
    CgpuIDevice* idevice = &ctx->idevice;
    CGPU_RESOLVE_SAMPLER(ctx, sampler, isampler);

    idevice->table.vkDestroySampler(idevice->logicalDevice, isampler->sampler, nullptr);

    ctx->isamplerStore.free(sampler.handle);
  }

  static void cgpuCreatePipelineDescriptors(CgpuIDevice* idevice,
                                            CgpuIShader* ishader,
                                            VkShaderStageFlags stageFlags,
                                            CgpuIPipeline* ipipeline)
  {
    cgpuCreatePipelineDescriptorSets(
      idevice, ishader, stageFlags,
      ipipeline->descriptorPool,
      ipipeline->descriptorSetLayouts,
      ipipeline->descriptorSetLayoutBindings,
      ipipeline->descriptorSetCount
    );

    const CgpuShaderReflection* shaderReflection = &ishader->reflection;
    const std::vector<CgpuShaderReflectionDescriptorSet>& descriptorSets = shaderReflection->descriptorSets;

    uint32_t uniformBufferCount = 0;
    uint32_t storageBufferCount = 0;
    uint32_t storageImageCount = 0;
    uint32_t sampledImageCount = 0;
    uint32_t samplerCount = 0;
    uint32_t asCount = 0;

    for (const CgpuShaderReflectionDescriptorSet& descriptorSet : descriptorSets)
    {
      const std::vector<CgpuShaderReflectionBinding>& bindings = descriptorSet.bindings;

      for (uint32_t i = 0; i < bindings.size(); i++)
      {
        const CgpuShaderReflectionBinding* binding = &bindings[i];

        switch (binding->descriptorType)
        {
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER: uniformBufferCount += binding->count; break;
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: storageBufferCount += binding->count; break;
        case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE: storageImageCount += binding->count; break;
        case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE: sampledImageCount += binding->count; break;
        case VK_DESCRIPTOR_TYPE_SAMPLER: samplerCount += binding->count; break;
        case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR: asCount += binding->count; break;
        default: {
          CGPU_FATAL("invalid descriptor type");
        }
        }
      }
    }

    uint32_t poolSizeCount = 0;
    VkDescriptorPoolSize poolSizes[16];

    if (uniformBufferCount > 0)
    {
      // We treat all uniform buffers we encounter as dynamic.
      poolSizes[poolSizeCount].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
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
      .maxSets = CGPU_MAX_DESCRIPTOR_SET_COUNT,
      .poolSizeCount = poolSizeCount,
      .pPoolSizes = poolSizes
    };

    VkResult result = idevice->table.vkCreateDescriptorPool(
      idevice->logicalDevice,
      &descriptorPoolCreateInfo,
      nullptr,
      &ipipeline->descriptorPool
    );
    if (result != VK_SUCCESS)
    {
      CGPU_FATAL("failed to create descriptor pool");
    }
  }

  void cgpuCreateComputePipeline(CgpuContext* ctx,
                                 CgpuComputePipelineCreateInfo createInfo,
                                 CgpuPipeline* pipeline)
  {
    CgpuIDevice* idevice = &ctx->idevice;
    CGPU_RESOLVE_SHADER(ctx, createInfo.shader, ishader);

    uint64_t handle = ctx->ipipelineStore.allocate();

    CGPU_RESOLVE_PIPELINE(ctx, { handle }, ipipeline);

    cgpuCreatePipelineDescriptors(idevice, ishader, VK_SHADER_STAGE_COMPUTE_BIT, ipipeline);

    cgpuCreatePipelineLayout(idevice, ipipeline->descriptorSetLayouts, ipipeline->descriptorSetCount,
                             ishader, VK_SHADER_STAGE_COMPUTE_BIT, &ipipeline->layout);

    VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .stage = VK_SHADER_STAGE_COMPUTE_BIT,
      .module = ishader->module,
      .pName = CGPU_SHADER_ENTRY_POINT,
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

    if (result != VK_SUCCESS)
    {
      CGPU_FATAL("failed to create compute pipeline");
    }

    if (ctx->debugUtilsEnabled && createInfo.debugName)
    {
      cgpuSetObjectName(idevice->logicalDevice, VK_OBJECT_TYPE_PIPELINE, (uint64_t) ipipeline->pipeline, createInfo.debugName);
    }

    ipipeline->bindPoint = VK_PIPELINE_BIND_POINT_COMPUTE;

    pipeline->handle = handle;
  }

  static bool cgpuCopyMemoryToBuffer(CgpuContext* ctx, const uint8_t* data, uint64_t size, CgpuBuffer dst)
  {
    if (size > CGPU_MAX_BUFFER_UPDATE_SIZE)
    {
      CGPU_RETURN_ERROR("buffer size too large!");
    }

    CgpuCommandBuffer commandBuffer;
    cgpuCreateCommandBuffer(ctx, &commandBuffer);

    cgpuBeginCommandBuffer(ctx, commandBuffer);
    cgpuCmdUpdateBuffer(ctx, commandBuffer, data, size, dst);
    cgpuEndCommandBuffer(ctx, commandBuffer);

    CgpuSemaphore semaphore;
    cgpuCreateSemaphore(ctx, &semaphore);
    CgpuSignalSemaphoreInfo signalSemaphoreInfo{ .semaphore = semaphore, .value = 1 };

    cgpuSubmitCommandBuffer(ctx, commandBuffer, 1, &signalSemaphoreInfo);

    CgpuWaitSemaphoreInfo waitSemaphoreInfo{ .semaphore = semaphore, .value = 1 };
    cgpuWaitSemaphores(ctx, 1, &waitSemaphoreInfo);

    cgpuDestroySemaphore(ctx, semaphore);
    cgpuDestroyCommandBuffer(ctx, commandBuffer);

    return true;
  }

  static void cgpuCreateRtPipelineSbt(CgpuContext* ctx,
                                      CgpuIPipeline* ipipeline,
                                      uint32_t groupCount,
                                      uint32_t missShaderCount,
                                      uint32_t hitGroupCount)
  {
    CgpuIDevice* idevice = &ctx->idevice;

    const CgpuIDeviceProperties& properties = idevice->internalProperties;

    uint32_t handleSize = properties.shaderGroupHandleSize;
    uint32_t alignedHandleSize = cgpuAlign(handleSize, properties.shaderGroupHandleAlignment);

    ipipeline->sbtRgen.stride = cgpuAlign(alignedHandleSize, properties.shaderGroupBaseAlignment);
    ipipeline->sbtRgen.size = ipipeline->sbtRgen.stride; // Special raygen condition: size must be equal to stride
    ipipeline->sbtMiss.stride = alignedHandleSize;
    ipipeline->sbtMiss.size = cgpuAlign(missShaderCount * alignedHandleSize, properties.shaderGroupBaseAlignment);
    ipipeline->sbtHit.stride = alignedHandleSize;
    ipipeline->sbtHit.size = cgpuAlign(hitGroupCount * alignedHandleSize, properties.shaderGroupBaseAlignment);

    uint32_t firstGroup = 0;
    uint32_t dataSize = handleSize * groupCount;

    std::vector<uint8_t> handleData(dataSize);
    if (idevice->table.vkGetRayTracingShaderGroupHandlesKHR(idevice->logicalDevice, ipipeline->pipeline, firstGroup, groupCount, handleData.size(), handleData.data()) != VK_SUCCESS)
    {
      CGPU_FATAL("failed to create sbt handles");
    }

    VkDeviceSize sbtSize = ipipeline->sbtRgen.size + ipipeline->sbtMiss.size + ipipeline->sbtHit.size;

    CgpuBufferCreateInfo sbtCreateInfo = {
      .usage = CgpuBufferUsage::TransferDst | CgpuBufferUsage::ShaderDeviceAddress | CgpuBufferUsage::ShaderBindingTable,
      .memoryProperties = CgpuMemoryProperties::DeviceLocal,
      .size = sbtSize,
      .debugName = "[SBT]",
      .alignment = properties.shaderGroupBaseAlignment
    };

    if (!cgpuCreateBuffer(ctx, sbtCreateInfo, &ipipeline->sbt))
    {
      CGPU_FATAL("failed to create sbt buffer");
    }

    CGPU_RESOLVE_BUFFER(ctx, ipipeline->sbt, isbt);

    VkDeviceAddress sbtDeviceAddress = isbt->gpuAddress;
    ipipeline->sbtRgen.deviceAddress = sbtDeviceAddress;
    ipipeline->sbtMiss.deviceAddress = sbtDeviceAddress + ipipeline->sbtRgen.size;
    ipipeline->sbtHit.deviceAddress = sbtDeviceAddress + ipipeline->sbtRgen.size + ipipeline->sbtMiss.size;

    auto sbtMem = std::make_unique<uint8_t[]>(sbtSize);
    uint8_t* sbtMemRgen = &sbtMem[0];
    uint8_t* sbtMemMiss = &sbtMem[ipipeline->sbtRgen.size];
    uint8_t* sbtMemHit = &sbtMem[ipipeline->sbtRgen.size + ipipeline->sbtMiss.size];

    uint32_t handleCount = 0;

    // Rgen
    uint8_t* sbtMemPtr = sbtMemRgen;
    memcpy(sbtMemPtr, &handleData[handleSize * (handleCount++)], handleSize);
    // Miss
    sbtMemPtr = sbtMemMiss;
    for (uint32_t i = 0; i < missShaderCount; i++)
    {
      memcpy(sbtMemPtr, &handleData[handleSize * (handleCount++)], handleSize);
      sbtMemPtr += ipipeline->sbtMiss.stride;
    }
    // Hit
    sbtMemPtr = sbtMemHit;
    for (uint32_t i = 0; i < hitGroupCount; i++)
    {
      memcpy(sbtMemPtr, &handleData[handleSize * (handleCount++)], handleSize);
      sbtMemPtr += ipipeline->sbtHit.stride;
    }

    if (!cgpuCopyMemoryToBuffer(ctx, &sbtMem[0], sbtSize, ipipeline->sbt))
    {
      CGPU_FATAL("failed to copy to sbt buffer");
    }
  }

  void cgpuCreateRtPipeline(CgpuContext* ctx,
                            CgpuRtPipelineCreateInfo createInfo,
                            CgpuPipeline* pipeline)
  {
    CgpuIDevice* idevice = &ctx->idevice;

    uint64_t handle = ctx->ipipelineStore.allocate();

    CGPU_RESOLVE_PIPELINE(ctx, { handle }, ipipeline);

    // Gather groups.
    size_t groupCount = 1/*rgen*/ + createInfo.missShaderCount + createInfo.hitGroupCount;
    std::vector<VkRayTracingShaderGroupCreateInfoKHR> groups(groupCount);

    for (uint32_t i = 0; i < groups.size(); i++)
    {
      VkRayTracingShaderGroupCreateInfoKHR groupCreateInfo = {
        .sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
        .pNext = nullptr,
        .type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
        .generalShader = i,
        .closestHitShader = VK_SHADER_UNUSED_KHR,
        .anyHitShader = VK_SHADER_UNUSED_KHR,
        .intersectionShader = VK_SHADER_UNUSED_KHR,
        .pShaderGroupCaptureReplayHandle = nullptr,
      };

      groups[i] = groupCreateInfo;
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

    // Create descriptor backing and pipeline layout.
    CGPU_RESOLVE_SHADER(ctx, createInfo.rgenShader, irgenShader);

    cgpuCreatePipelineDescriptors(idevice, irgenShader, CGPU_RT_PIPELINE_ACCESS_FLAGS, ipipeline);

    cgpuCreatePipelineLayout(idevice, ipipeline->descriptorSetLayouts, ipipeline->descriptorSetCount,
                             irgenShader, CGPU_RT_PIPELINE_ACCESS_FLAGS, &ipipeline->layout);

    // Collect pipeline libraries OR stages.
    std::vector<VkPipeline> libraries;
    std::vector<VkPipelineShaderStageCreateInfo> stages;
    if (idevice->internalFeatures.pipelineLibraries)
    {
      libraries.reserve(groupCount * 2);

      libraries.push_back(irgenShader->pipelineLibrary.pipeline);

      const auto getShaderPipelineHandle = [ctx](CgpuShader shader) {
        CGPU_RESOLVE_SHADER(ctx, shader, ishader);
        return ishader->pipelineLibrary.pipeline;
      };

      for (uint32_t i = 0; i < createInfo.missShaderCount; i++)
      {
        libraries.push_back(getShaderPipelineHandle(createInfo.missShaders[i]));
      }

      for (uint32_t i = 0; i < createInfo.hitGroupCount; i++)
      {
        CgpuShader closestHitShader = createInfo.hitGroups[i].closestHitShader;
        if (closestHitShader.handle)
        {
          libraries.push_back(getShaderPipelineHandle(closestHitShader));
        }

        CgpuShader anyHitShader = createInfo.hitGroups[i].anyHitShader;
        if (anyHitShader.handle)
        {
          libraries.push_back(getShaderPipelineHandle(anyHitShader));
        }
      }
    }
    else
    {
      stages.reserve(groupCount * 2);

      auto pushStage = [&stages](VkShaderStageFlagBits stage, VkShaderModule module) {
        assert(module);
        VkPipelineShaderStageCreateInfo stageCreateInfo = {
          .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .pNext = nullptr,
          .flags = 0,
          .stage = stage,
          .module = module,
          .pName = CGPU_SHADER_ENTRY_POINT,
          .pSpecializationInfo = nullptr,
        };
        stages.push_back(stageCreateInfo);
      };

      pushStage(VK_SHADER_STAGE_RAYGEN_BIT_KHR, irgenShader->module);

      for (uint32_t i = 0; i < createInfo.missShaderCount; i++)
      {
        CGPU_RESOLVE_SHADER(ctx, createInfo.missShaders[i], imissShader);
        pushStage(VK_SHADER_STAGE_MISS_BIT_KHR, imissShader->module);
      }

      for (uint32_t i = 0; i < createInfo.hitGroupCount; i++)
      {
        const CgpuRtHitGroup* hitGroup = &createInfo.hitGroups[i];

        if (hitGroup->closestHitShader.handle)
        {
          CGPU_RESOLVE_SHADER(ctx, hitGroup->closestHitShader, iclosestHitShader);
          pushStage(iclosestHitShader->stageFlags, iclosestHitShader->module);
        }

        if (hitGroup->anyHitShader.handle)
        {
          CGPU_RESOLVE_SHADER(ctx, hitGroup->anyHitShader, ianyHitShader);
          pushStage(ianyHitShader->stageFlags, ianyHitShader->module);
        }
      }
    }

    // Create pipeline.
    {
      uint32_t groupCount = hitStageAndGroupOffset + createInfo.hitGroupCount;

      VkPipelineLibraryCreateInfoKHR libraryCreateInfo {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR,
        .pNext = nullptr,
        .libraryCount = (uint32_t) libraries.size(),
        .pLibraries = libraries.data()
      };

      assert(createInfo.maxRayPayloadSize > 0);
      assert(createInfo.maxRayHitAttributeSize > 0);

      VkRayTracingPipelineInterfaceCreateInfoKHR interfaceCreateInfo {
        .sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_INTERFACE_CREATE_INFO_KHR,
        .pNext = nullptr,
        .maxPipelineRayPayloadSize = createInfo.maxRayPayloadSize,
        .maxPipelineRayHitAttributeSize = createInfo.maxRayHitAttributeSize
      };

      VkRayTracingPipelineCreateInfoKHR rtPipelineCreateInfo = {
        .sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
        .pNext = nullptr,
        .flags = 0,
        .stageCount = (uint32_t) stages.size(),
        .pStages = stages.data(),
        .groupCount = (uint32_t) groups.size(),
        .pGroups = groups.data(),
        .maxPipelineRayRecursionDepth = 1,
        .pLibraryInfo = idevice->internalFeatures.pipelineLibraries ? &libraryCreateInfo : nullptr,
        .pLibraryInterface = idevice->internalFeatures.pipelineLibraries ? &interfaceCreateInfo : nullptr,
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
        CGPU_FATAL("failed to create RT pipeline");
      }

      ipipeline->bindPoint = VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR;

      // Create the SBT.
      cgpuCreateRtPipelineSbt(ctx, ipipeline, groupCount, createInfo.missShaderCount, createInfo.hitGroupCount);

      if (ctx->debugUtilsEnabled && createInfo.debugName)
      {
        cgpuSetObjectName(idevice->logicalDevice, VK_OBJECT_TYPE_PIPELINE, (uint64_t) ipipeline->pipeline, createInfo.debugName);
      }

      pipeline->handle = handle;
    }
  }

  void cgpuDestroyPipeline(CgpuContext* ctx, CgpuPipeline pipeline)
  {
    CgpuIDevice* idevice = &ctx->idevice;
    CGPU_RESOLVE_PIPELINE(ctx, pipeline, ipipeline);

    if (ipipeline->bindPoint == VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR)
    {
      cgpuDestroyBuffer(ctx, ipipeline->sbt);
    }

    idevice->table.vkDestroyDescriptorPool(idevice->logicalDevice, ipipeline->descriptorPool, nullptr);
    idevice->table.vkDestroyPipeline(idevice->logicalDevice, ipipeline->pipeline, nullptr);
    idevice->table.vkDestroyPipelineLayout(idevice->logicalDevice, ipipeline->layout, nullptr);
    for (uint32_t i = 0; i < ipipeline->descriptorSetCount; i++)
    {
      idevice->table.vkDestroyDescriptorSetLayout(idevice->logicalDevice, ipipeline->descriptorSetLayouts[i], nullptr);
    }
    ctx->ipipelineStore.free(pipeline.handle);
  }

  static bool cgpuCreateTopOrBottomAs(CgpuContext* ctx,
                                      VkAccelerationStructureTypeKHR asType,
                                      VkAccelerationStructureGeometryKHR* asGeom,
                                      uint32_t primitiveCount,
                                      CgpuIBuffer* iasBuffer,
                                      VkAccelerationStructureKHR* as)
  {
    CgpuIDevice* idevice = &ctx->idevice;

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
    if (!cgpuCreateIBuffer(idevice,
                           CgpuBufferUsage::ShaderDeviceAddress | CgpuBufferUsage::AccelerationStructureStorage,
                           CgpuMemoryProperties::DeviceLocal,
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
    if (!cgpuCreateIBuffer(idevice,
                           CgpuBufferUsage::Storage | CgpuBufferUsage::ShaderDeviceAddress,
                           CgpuMemoryProperties::DeviceLocal,
                           asBuildSizesInfo.buildScratchSize,
                           idevice->internalProperties.minAccelerationStructureScratchOffsetAlignment,
                           &iscratchBuffer, "[AS scratch buffer]",
                           idevice->asScratchMemoryPool))
    {
      cgpuDestroyIBuffer(idevice, iasBuffer);
      idevice->table.vkDestroyAccelerationStructureKHR(idevice->logicalDevice, *as, nullptr);
      CGPU_RETURN_ERROR("failed to create AS scratch buffer");
    }

    asBuildGeomInfo.dstAccelerationStructure = *as;
    asBuildGeomInfo.scratchData.hostAddress = 0;
    asBuildGeomInfo.scratchData.deviceAddress = iscratchBuffer.gpuAddress;

    VkAccelerationStructureBuildRangeInfoKHR asBuildRangeInfo = {
      .primitiveCount = primitiveCount,
      .primitiveOffset = 0,
      .firstVertex = 0,
      .transformOffset = 0,
    };

    const VkAccelerationStructureBuildRangeInfoKHR* asBuildRangeInfoPtr = &asBuildRangeInfo;

    CgpuCommandBuffer commandBuffer;
    if (!cgpuCreateCommandBuffer(ctx, &commandBuffer))
    {
      cgpuDestroyIBuffer(idevice, iasBuffer);
      cgpuDestroyIBuffer(idevice, &iscratchBuffer);
      idevice->table.vkDestroyAccelerationStructureKHR(idevice->logicalDevice, *as, nullptr);
      CGPU_RETURN_ERROR("failed to create AS build command buffer");
    }

    CgpuICommandBuffer* icommandBuffer;
    cgpuResolveCommandBuffer(ctx, commandBuffer, &icommandBuffer);

    // Build AS on device
    cgpuBeginCommandBuffer(ctx, commandBuffer);
    idevice->table.vkCmdBuildAccelerationStructuresKHR(icommandBuffer->commandBuffer, 1, &asBuildGeomInfo, &asBuildRangeInfoPtr);
    cgpuEndCommandBuffer(ctx, commandBuffer);

    CgpuSemaphore semaphore;
    if (!cgpuCreateSemaphore(ctx, &semaphore))
    {
      cgpuDestroyIBuffer(idevice, iasBuffer);
      cgpuDestroyIBuffer(idevice, &iscratchBuffer);
      idevice->table.vkDestroyAccelerationStructureKHR(idevice->logicalDevice, *as, nullptr);
      CGPU_RETURN_ERROR("failed to create AS build semaphore");
    }

    CgpuSignalSemaphoreInfo signalSemaphoreInfo{ .semaphore = semaphore, .value = 1 };
    cgpuSubmitCommandBuffer(ctx, commandBuffer, 1, &signalSemaphoreInfo);
    CgpuWaitSemaphoreInfo waitSemaphoreInfo{ .semaphore = semaphore, .value = 1 };
    cgpuWaitSemaphores(ctx, 1, &waitSemaphoreInfo);

    // Dispose resources
    cgpuDestroySemaphore(ctx, semaphore);
    cgpuDestroyCommandBuffer(ctx, commandBuffer);
    cgpuDestroyIBuffer(idevice, &iscratchBuffer);

    return true;
  }

  bool cgpuCreateBlas(CgpuContext* ctx,
                      CgpuBlasCreateInfo createInfo,
                      CgpuBlas* blas)
  {
    CgpuIDevice* idevice = &ctx->idevice;
    CGPU_RESOLVE_BUFFER(ctx, createInfo.vertexPosBuffer, ivertexBuffer);
    CGPU_RESOLVE_BUFFER(ctx, createInfo.indexBuffer, iindexBuffer);

    uint64_t handle = ctx->iblasStore.allocate();

    CGPU_RESOLVE_BLAS(ctx, { handle }, iblas);

    VkAccelerationStructureGeometryTrianglesDataKHR asTriangleData = {
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
      .pNext = nullptr,
      .vertexFormat = VK_FORMAT_R32G32B32_SFLOAT,
      .vertexData = {
        .deviceAddress = ivertexBuffer->gpuAddress,
      },
      .vertexStride = sizeof(float) * 3,
      .maxVertex = createInfo.maxVertex,
      .indexType = VK_INDEX_TYPE_UINT32,
      .indexData = {
        .deviceAddress = iindexBuffer->gpuAddress,
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

    bool creationSuccessul = cgpuCreateTopOrBottomAs(ctx, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
        &asGeom, createInfo.triangleCount, &iblas->buffer, &iblas->as);

    if (!creationSuccessul)
    {
      ctx->iblasStore.free(handle);
      CGPU_RETURN_ERROR("failed to build BLAS");
    }

    if (ctx->debugUtilsEnabled && createInfo.debugName)
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

  bool cgpuCreateTlas(CgpuContext* ctx,
                      CgpuTlasCreateInfo createInfo,
                      CgpuTlas* tlas)
  {
    CgpuIDevice* idevice = &ctx->idevice;

    uint64_t handle = ctx->itlasStore.allocate();

    CGPU_RESOLVE_TLAS(ctx, { handle }, itlas);

    // Create instance buffer & copy into it
    CgpuIBuffer instances;
    if (!cgpuCreateIBuffer(idevice,
                           CgpuBufferUsage::ShaderDeviceAddress | CgpuBufferUsage::AccelerationStructureBuild,
                           CgpuMemoryProperties::HostVisible,
                           (createInfo.instanceCount ? createInfo.instanceCount : 1) * sizeof(VkAccelerationStructureInstanceKHR),
                           16/*required by spec*/, &instances, createInfo.debugName))
    {
      ctx->itlasStore.free(handle);
      CGPU_RETURN_ERROR("failed to create TLAS instances buffer");
    }

    bool areAllBlasOpaque = true;
    {
      uint8_t* mapped_mem;
      if (vmaMapMemory(idevice->allocator, instances.allocation, (void**) &mapped_mem) != VK_SUCCESS)
      {
        CGPU_FATAL("failed to map buffer memory");
      }

      for (uint32_t i = 0; i < createInfo.instanceCount; i++)
      {
        const CgpuBlasInstance& instanceDesc = createInfo.instances[i];
        CGPU_RESOLVE_BLAS(ctx, instanceDesc.as, iblas);

        uint32_t instanceCustomIndex = instanceDesc.instanceCustomIndex;
        if ((instanceCustomIndex & 0xFF000000u) != 0u)
        {
          CGPU_FATAL("instanceCustomIndex must be equal to or smaller than 2^24");
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

      vmaUnmapMemory(idevice->allocator, instances.allocation);
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
            .deviceAddress = instances.gpuAddress
          }
        },
      },
      .flags = VkGeometryFlagsKHR(areAllBlasOpaque ? VK_GEOMETRY_OPAQUE_BIT_KHR : 0)
    };

    bool result = cgpuCreateTopOrBottomAs(ctx, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, &asGeom, createInfo.instanceCount, &itlas->buffer, &itlas->as);

    cgpuDestroyIBuffer(idevice, &instances);

    if (!result)
    {
      ctx->itlasStore.free(handle);
      CGPU_RETURN_ERROR("failed to build TLAS");
    }

    if (ctx->debugUtilsEnabled && createInfo.debugName)
    {
      cgpuSetObjectName(idevice->logicalDevice, VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR, (uint64_t) itlas->as, createInfo.debugName);
    }

    tlas->handle = handle;
    return true;
  }

  void cgpuDestroyBlas(CgpuContext* ctx, CgpuBlas blas)
  {
    CgpuIDevice* idevice = &ctx->idevice;
    CGPU_RESOLVE_BLAS(ctx, blas, iblas);

    idevice->table.vkDestroyAccelerationStructureKHR(idevice->logicalDevice, iblas->as, nullptr);
    cgpuDestroyIBuffer(idevice, &iblas->buffer);

    ctx->iblasStore.free(blas.handle);
  }

  void cgpuDestroyTlas(CgpuContext* ctx, CgpuTlas tlas)
  {
    CgpuIDevice* idevice = &ctx->idevice;
    CGPU_RESOLVE_TLAS(ctx, tlas, itlas);

    idevice->table.vkDestroyAccelerationStructureKHR(idevice->logicalDevice, itlas->as, nullptr);
    cgpuDestroyIBuffer(idevice, &itlas->buffer);

    ctx->itlasStore.free(tlas.handle);
  }

  void cgpuCreateBindSets(CgpuContext* ctx, CgpuPipeline pipeline, CgpuBindSet* bindSets, uint32_t bindSetCount)
  {
    CgpuIDevice* idevice = &ctx->idevice;
    CGPU_RESOLVE_PIPELINE(ctx, pipeline, ipipeline);

    if (ipipeline->descriptorSetCount != bindSetCount)
    {
      CGPU_FATAL("descriptor set count mismatch");
    }

    VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .pNext = nullptr,
      .descriptorPool = ipipeline->descriptorPool,
      .descriptorSetCount = ipipeline->descriptorSetCount,
      .pSetLayouts = ipipeline->descriptorSetLayouts,
    };

    VkDescriptorSet descriptorSets[CGPU_MAX_DESCRIPTOR_SET_COUNT];

    VkResult result = idevice->table.vkAllocateDescriptorSets(
      idevice->logicalDevice,
      &descriptorSetAllocateInfo,
      descriptorSets
    );

    if (result != VK_SUCCESS)
    {
      CGPU_FATAL("failed to allocate descriptor set");
    }

    for (uint32_t i = 0; i < bindSetCount; i++)
    {
      bindSets[i].handle = ctx->ibindSetStore.allocate();

      CGPU_RESOLVE_BIND_SET(ctx, bindSets[i], ibindSet);
      ibindSet->descriptorSetLayoutBindings = ipipeline->descriptorSetLayoutBindings[i];
      ibindSet->descriptorSet = descriptorSets[i];
    }
  }

  void cgpuDestroyBindSets(CgpuContext* ctx, CgpuBindSet* bindSets, uint32_t bindSetCount)
  {
    CgpuIDevice* idevice = &ctx->idevice;

    for (uint32_t i = 0; i < bindSetCount; i++)
    {
      // descriptor set memory is managed by pipeline VkDescriptorPool
      ctx->ibindSetStore.free(bindSets[i].handle);
    }
  }

  void cgpuUpdateBindSet(CgpuContext* ctx,
                         CgpuBindSet bindSet,
                         const CgpuBindings* bindings)
  {
    CGPU_RESOLVE_BIND_SET(ctx, bindSet, ibindSet);
    CgpuIDevice* idevice = &ctx->idevice;

    GbSmallVector<VkDescriptorBufferInfo, 8> bufferInfos;
    GbSmallVector<VkWriteDescriptorSetAccelerationStructureKHR, 1> asInfos;
    std::vector<VkDescriptorImageInfo> imageInfos;
    imageInfos.reserve(128);

    bufferInfos.reserve(bindings->bufferCount);
    imageInfos.reserve(bindings->imageCount + bindings->samplerCount);
    asInfos.reserve(bindings->tlasCount);

    std::vector<VkWriteDescriptorSet> writeDescriptorSets;
    writeDescriptorSets.reserve(512);

    const std::vector<VkDescriptorSetLayoutBinding>& layoutBindings = ibindSet->descriptorSetLayoutBindings;

    for (uint32_t i = 0; i < layoutBindings.size(); i++)
    {
      const VkDescriptorSetLayoutBinding* layoutBinding = &layoutBindings[i];

      VkDescriptorType descriptorType = layoutBinding->descriptorType;
      if (descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
      {
        descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
      }

      VkWriteDescriptorSet writeDescriptorSet = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .pNext = nullptr,
        .dstSet = ibindSet->descriptorSet,
        .dstBinding = layoutBinding->binding,
        .dstArrayElement = 0,
        .descriptorCount = layoutBinding->descriptorCount,
        .descriptorType = descriptorType,
        .pImageInfo = nullptr,
        .pBufferInfo = nullptr,
        .pTexelBufferView = nullptr,
      };

      if (descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC ||
          descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
      {
        for (uint32_t k = 0; k < bindings->bufferCount; ++k)
        {
          const CgpuBufferBinding* bufferBinding = &bindings->buffers[k];

          if (bufferBinding->binding != layoutBinding->binding)
          {
            continue;
          }

          if (bufferBinding->index >= layoutBinding->descriptorCount)
          {
            CGPU_FATAL("descriptor binding out of range");
          }

          CGPU_RESOLVE_BUFFER(ctx, bufferBinding->buffer, ibuffer);

          VkDescriptorBufferInfo bufferInfo = {
            .buffer = ibuffer->buffer,
            .offset = bufferBinding->offset,
            .range = (bufferBinding->size == CGPU_WHOLE_SIZE) ? (ibuffer->size - bufferBinding->offset) : bufferBinding->size,
          };
          bufferInfos.push_back(bufferInfo);

          writeDescriptorSet.pBufferInfo = &bufferInfos.back();
          writeDescriptorSets.push_back(writeDescriptorSet);
        }
      }
      else if (descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE ||
               descriptorType == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE)
      {
        for (uint32_t k = 0; k < bindings->imageCount; k++)
        {
          const CgpuImageBinding* imageBinding = &bindings->images[k];

          if (imageBinding->binding != layoutBinding->binding)
          {
            continue;
          }

          if (imageBinding->index >= layoutBinding->descriptorCount)
          {
            CGPU_FATAL("descriptor binding out of range");
          }

          CGPU_RESOLVE_IMAGE(ctx, imageBinding->image, iimage);

          VkDescriptorImageInfo imageInfo = {
            .sampler = VK_NULL_HANDLE,
            .imageView = iimage->imageView,
            .imageLayout = iimage->layout,
          };
          imageInfos.push_back(imageInfo);

          writeDescriptorSet.dstArrayElement = imageBinding->index;
          writeDescriptorSet.descriptorCount = 1;
          writeDescriptorSet.pImageInfo = &imageInfos.back();
          writeDescriptorSets.push_back(writeDescriptorSet);
        }
      }
      else if (descriptorType == VK_DESCRIPTOR_TYPE_SAMPLER)
      {
        for (uint32_t k = 0; k < bindings->samplerCount; k++)
        {
          const CgpuSamplerBinding* samplerBinding = &bindings->samplers[k];

          if (samplerBinding->binding != layoutBinding->binding)
          {
            continue;
          }

          if (samplerBinding->index >= layoutBinding->descriptorCount)
          {
            CGPU_FATAL("descriptor binding out of range");
          }

          CGPU_RESOLVE_SAMPLER(ctx, samplerBinding->sampler, isampler);

          VkDescriptorImageInfo imageInfo = {
            .sampler = isampler->sampler,
            .imageView = VK_NULL_HANDLE,
            .imageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
          };
          imageInfos.push_back(imageInfo);

          writeDescriptorSet.pImageInfo = &imageInfos.back();
          writeDescriptorSets.push_back(writeDescriptorSet);
        }
      }
      else if (descriptorType == VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR)
      {
        for (uint32_t k = 0; k < bindings->tlasCount; ++k)
        {
          const CgpuTlasBinding* asBinding = &bindings->tlases[k];

          if (asBinding->binding != layoutBinding->binding)
          {
            continue;
          }

          if (asBinding->index >= layoutBinding->descriptorCount)
          {
            CGPU_FATAL("descriptor binding out of range");
          }

          CGPU_RESOLVE_TLAS(ctx, asBinding->as, itlas);

          VkWriteDescriptorSetAccelerationStructureKHR asInfo = {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
            .pNext = nullptr,
            .accelerationStructureCount = 1,
            .pAccelerationStructures = &itlas->as,
          };
          asInfos.push_back(asInfo);

          writeDescriptorSet.pNext = &asInfos.back();
          writeDescriptorSets.push_back(writeDescriptorSet);
        }
      }
    }

    idevice->table.vkUpdateDescriptorSets(
      idevice->logicalDevice,
      (uint32_t) writeDescriptorSets.size(),
      writeDescriptorSets.data(),
      0,
      nullptr
    );
  }

  bool cgpuCreateCommandBuffer(CgpuContext* ctx, CgpuCommandBuffer* commandBuffer)
  {
    CgpuIDevice* idevice = &ctx->idevice;

    uint64_t handle = ctx->icommandBufferStore.allocate();

    CGPU_RESOLVE_COMMAND_BUFFER(ctx, { handle }, icommandBuffer);

    VkCommandBufferAllocateInfo cmdbufAllocInfo = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .pNext = nullptr,
      .commandPool = idevice->commandPool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
    };

    if (idevice->table.vkAllocateCommandBuffers(idevice->logicalDevice,
                                                &cmdbufAllocInfo,
                                                &icommandBuffer->commandBuffer) != VK_SUCCESS)
    {
      CGPU_FATAL("failed to allocate command buffer");
    }

    commandBuffer->handle = handle;
    return true;
  }

  void cgpuDestroyCommandBuffer(CgpuContext* ctx, CgpuCommandBuffer commandBuffer)
  {
    CgpuIDevice* idevice = &ctx->idevice;
    CGPU_RESOLVE_COMMAND_BUFFER(ctx, commandBuffer, icommandBuffer);

    idevice->table.vkFreeCommandBuffers(
      idevice->logicalDevice,
      idevice->commandPool,
      1,
      &icommandBuffer->commandBuffer
    );

    ctx->icommandBufferStore.free(commandBuffer.handle);
  }

  bool cgpuBeginCommandBuffer(CgpuContext* ctx, CgpuCommandBuffer commandBuffer)
  {
    CGPU_RESOLVE_COMMAND_BUFFER(ctx, commandBuffer, icommandBuffer);
    CgpuIDevice* idevice = &ctx->idevice;

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

  void cgpuCmdBindPipeline(CgpuContext* ctx,
                           CgpuCommandBuffer commandBuffer,
                           CgpuPipeline pipeline,
                           const CgpuBindSet* bindSets,
                           uint32_t bindSetCount,
                           uint32_t dynamicOffsetCount,
                           const uint32_t* dynamicOffsets)
  {
    CGPU_RESOLVE_COMMAND_BUFFER(ctx, commandBuffer, icommandBuffer);
    CGPU_RESOLVE_PIPELINE(ctx, pipeline, ipipeline);
    CgpuIDevice* idevice = &ctx->idevice;

    idevice->table.vkCmdBindPipeline(
      icommandBuffer->commandBuffer,
      ipipeline->bindPoint,
      ipipeline->pipeline
    );

    std::array<VkDescriptorSet, CGPU_MAX_DESCRIPTOR_SET_COUNT> descriptorSets;
    for (uint32_t i = 0; i < bindSetCount; i++)
    {
      CGPU_RESOLVE_BIND_SET(ctx, bindSets[i], ibindSet);
      descriptorSets[i] = ibindSet->descriptorSet;
    }

    uint32_t firstDescriptorSet = 0;
    idevice->table.vkCmdBindDescriptorSets(
      icommandBuffer->commandBuffer,
      ipipeline->bindPoint,
      ipipeline->layout,
      firstDescriptorSet,
      ipipeline->descriptorSetCount,
      &descriptorSets[firstDescriptorSet],
      dynamicOffsetCount,
      dynamicOffsets
    );
  }

  void cgpuCmdTransitionShaderImageLayouts(CgpuContext* ctx,
                                           CgpuCommandBuffer commandBuffer,
                                           CgpuShader shader,
                                           uint32_t descriptorSetIndex,
                                           uint32_t imageCount,
                                           const CgpuImageBinding* images)
  {
    CGPU_RESOLVE_SHADER(ctx, shader, ishader);
    CGPU_RESOLVE_COMMAND_BUFFER(ctx, commandBuffer, icommandBuffer);
    CgpuIDevice* idevice = &ctx->idevice;

    std::vector<VkImageMemoryBarrier2KHR> barriers;
    barriers.reserve(64);

    const CgpuShaderReflection* reflection = &ishader->reflection;
    const std::vector<CgpuShaderReflectionDescriptorSet>& descriptorSets = reflection->descriptorSets;
    if (descriptorSetIndex >= descriptorSets.size())
    {
      CGPU_FATAL("descriptor set index out of bounds");
    }

    const CgpuShaderReflectionDescriptorSet& descriptorSet = descriptorSets[descriptorSetIndex];
    const std::vector<CgpuShaderReflectionBinding>& bindings = descriptorSet.bindings;

    /* NOTE: this has quadratic complexity */
    for (uint32_t i = 0; i < bindings.size(); i++)
    {
      const CgpuShaderReflectionBinding* binding = &bindings[i];

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
          continue;
        }

        CGPU_RESOLVE_IMAGE(ctx, imageBinding->image, iimage);

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
          .srcStageMask = CgpuPipelineStageFromShaderStageFlags(ishader->stageFlags),
          .srcAccessMask = iimage->accessMask,
          .dstStageMask = CgpuPipelineStageFromShaderStageFlags(ishader->stageFlags),
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
  }

  void cgpuCmdUpdateBuffer(CgpuContext* ctx,
                           CgpuCommandBuffer commandBuffer,
                           const uint8_t* data,
                           uint64_t size,
                           CgpuBuffer dstBuffer,
                           uint64_t dstOffset)
  {
    CGPU_RESOLVE_COMMAND_BUFFER(ctx, commandBuffer, icommandBuffer);
    CGPU_RESOLVE_BUFFER(ctx, dstBuffer, idstBuffer);
    CgpuIDevice* idevice = &ctx->idevice;

    idevice->table.vkCmdUpdateBuffer(
      icommandBuffer->commandBuffer,
      idstBuffer->buffer,
      dstOffset,
      size,
      (const void*) data
    );
  }

  void cgpuCmdCopyBuffer(CgpuContext* ctx,
                         CgpuCommandBuffer commandBuffer,
                         CgpuBuffer srcBuffer,
                         uint64_t srcOffset,
                         CgpuBuffer dstBuffer,
                         uint64_t dstOffset,
                         uint64_t size)
  {
    CGPU_RESOLVE_COMMAND_BUFFER(ctx, commandBuffer, icommandBuffer);
    CGPU_RESOLVE_BUFFER(ctx, srcBuffer, isrcBuffer);
    CGPU_RESOLVE_BUFFER(ctx, dstBuffer, idstBuffer);
    CgpuIDevice* idevice = &ctx->idevice;

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
  }

  void cgpuCmdCopyBufferToImage(CgpuContext* ctx,
                                CgpuCommandBuffer commandBuffer,
                                CgpuBuffer buffer,
                                CgpuImage image,
                                const CgpuBufferImageCopyDesc* desc)
  {
    CGPU_RESOLVE_COMMAND_BUFFER(ctx, commandBuffer, icommandBuffer);
    CGPU_RESOLVE_BUFFER(ctx, buffer, ibuffer);
    CGPU_RESOLVE_IMAGE(ctx, image, iimage);
    CgpuIDevice* idevice = &ctx->idevice;

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
  }

  void cgpuCmdPushConstants(CgpuContext* ctx,
                            CgpuCommandBuffer commandBuffer,
                            CgpuPipeline pipeline,
                            uint32_t size,
                            const void* data)
  {
    CGPU_RESOLVE_COMMAND_BUFFER(ctx, commandBuffer, icommandBuffer);
    CGPU_RESOLVE_PIPELINE(ctx, pipeline, ipipeline);
    CgpuIDevice* idevice = &ctx->idevice;

    VkShaderStageFlags stageFlags;
    switch (ipipeline->bindPoint)
    {
      case VK_PIPELINE_BIND_POINT_COMPUTE:
        stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        break;
      case VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR:
        stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                     VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                     VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                     VK_SHADER_STAGE_MISS_BIT_KHR;
        break;
      default:
        CGPU_FATAL("unhandled pipeline bind point");
    }

    idevice->table.vkCmdPushConstants(
      icommandBuffer->commandBuffer,
      ipipeline->layout,
      stageFlags,
      0,
      size,
      data
    );
  }

  void cgpuCmdDispatch(CgpuContext* ctx,
                       CgpuCommandBuffer commandBuffer,
                       uint32_t dimX,
                       uint32_t dimY,
                       uint32_t dimZ)
  {
    CGPU_RESOLVE_COMMAND_BUFFER(ctx, commandBuffer, icommandBuffer);
    CgpuIDevice* idevice = &ctx->idevice;

    idevice->table.vkCmdDispatch(
      icommandBuffer->commandBuffer,
      dimX,
      dimY,
      dimZ
    );
  }

  void cgpuCmdPipelineBarrier(CgpuContext* ctx,
                              CgpuCommandBuffer commandBuffer,
                              const CgpuPipelineBarrier* barrier)
  {
    CGPU_RESOLVE_COMMAND_BUFFER(ctx, commandBuffer, icommandBuffer);
    CgpuIDevice* idevice = &ctx->idevice;

    std::vector<VkMemoryBarrier2KHR> vkMemBarriers;
    vkMemBarriers.reserve(128);

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

    std::vector<VkBufferMemoryBarrier2KHR> vkBufferMemBarriers;
    vkBufferMemBarriers.reserve(16);
    std::vector<VkImageMemoryBarrier2KHR> vkImageMemBarriers;
    vkImageMemBarriers.reserve(128);

    for (uint32_t i = 0; i < barrier->bufferBarrierCount; ++i)
    {
      const CgpuBufferMemoryBarrier* bCgpu = &barrier->bufferBarriers[i];

      CGPU_RESOLVE_BUFFER(ctx, bCgpu->buffer, ibuffer);

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

      CGPU_RESOLVE_IMAGE(ctx, bCgpu->image, iimage);

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
  }

  void cgpuCmdTraceRays(CgpuContext* ctx,
                        CgpuCommandBuffer commandBuffer,
                        CgpuPipeline pipeline,
                        uint32_t width,
                        uint32_t height)
  {
    CGPU_RESOLVE_COMMAND_BUFFER(ctx, commandBuffer, icommandBuffer);
    CGPU_RESOLVE_PIPELINE(ctx, pipeline, ipipeline);
    CgpuIDevice* idevice = &ctx->idevice;

    VkStridedDeviceAddressRegionKHR callableSBT = {};
    idevice->table.vkCmdTraceRaysKHR(icommandBuffer->commandBuffer,
                                     &ipipeline->sbtRgen,
                                     &ipipeline->sbtMiss,
                                     &ipipeline->sbtHit,
                                     &callableSBT,
                                     width, height, 1);
  }

  void cgpuCmdFillBuffer(CgpuContext* ctx,
                         CgpuCommandBuffer commandBuffer,
                         CgpuBuffer buffer,
                         uint64_t dstOffset,
                         uint64_t size,
                         uint8_t data)
  {
    CGPU_RESOLVE_COMMAND_BUFFER(ctx, commandBuffer, icommandBuffer);
    CGPU_RESOLVE_BUFFER(ctx, buffer, ibuffer);
    CgpuIDevice* idevice = &ctx->idevice;

    uint64_t rangeSize = (size == CGPU_WHOLE_SIZE) ? ibuffer->size : size;
    idevice->table.vkCmdFillBuffer(icommandBuffer->commandBuffer, ibuffer->buffer, dstOffset, rangeSize, data);
  }

  void cgpuEndCommandBuffer(CgpuContext* ctx, CgpuCommandBuffer commandBuffer)
  {
    CGPU_RESOLVE_COMMAND_BUFFER(ctx, commandBuffer, icommandBuffer);
    CgpuIDevice* idevice = &ctx->idevice;

    idevice->table.vkEndCommandBuffer(icommandBuffer->commandBuffer);
  }

  bool cgpuCreateSemaphore(CgpuContext* ctx, CgpuSemaphore* semaphore, uint64_t initialValue)
  {
    CgpuIDevice* idevice = &ctx->idevice;

    uint64_t handle = ctx->isemaphoreStore.allocate();

    CGPU_RESOLVE_SEMAPHORE(ctx, { handle }, isemaphore);

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

    if (result != VK_SUCCESS)
    {
      ctx->isemaphoreStore.free(handle);
      CGPU_RETURN_ERROR("failed to create semaphore");
    }

    semaphore->handle = handle;
    return true;
  }

  void cgpuDestroySemaphore(CgpuContext* ctx, CgpuSemaphore semaphore)
  {
    CgpuIDevice* idevice = &ctx->idevice;
    CGPU_RESOLVE_SEMAPHORE(ctx, semaphore, isemaphore);

    idevice->table.vkDestroySemaphore(
      idevice->logicalDevice,
      isemaphore->semaphore,
      nullptr
    );

    ctx->isemaphoreStore.free(semaphore.handle);
  }

  bool cgpuWaitSemaphores(CgpuContext* ctx,
                          uint32_t semaphoreInfoCount,
                          CgpuWaitSemaphoreInfo* semaphoreInfos,
                          uint64_t timeoutNs)
  {
    CgpuIDevice* idevice = &ctx->idevice;

    GbSmallVector<VkSemaphore, 8> semaphores(semaphoreInfoCount);
    GbSmallVector<uint64_t, 8> semaphoreValues(semaphoreInfoCount);

    for (uint32_t i = 0; i < semaphoreInfoCount; i++)
    {
      const CgpuWaitSemaphoreInfo& semaphoreInfo = semaphoreInfos[i];

      CGPU_RESOLVE_SEMAPHORE(ctx, semaphoreInfo.semaphore, isemaphore);

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

    if (result != VK_SUCCESS)
    {
      CGPU_RETURN_ERROR("failed to wait for semaphores");
    }
    return true;
  }

  void cgpuSubmitCommandBuffer(CgpuContext* ctx,
                               CgpuCommandBuffer commandBuffer,
                               uint32_t signalSemaphoreInfoCount,
                               CgpuSignalSemaphoreInfo* signalSemaphoreInfos,
                               uint32_t waitSemaphoreInfoCount,
                               CgpuWaitSemaphoreInfo* waitSemaphoreInfos)
  {
    CgpuIDevice* idevice = &ctx->idevice;
    CGPU_RESOLVE_COMMAND_BUFFER(ctx, commandBuffer, icommandBuffer);

    GbSmallVector<VkSemaphoreSubmitInfo, 8> signalSubmitInfos(signalSemaphoreInfoCount);
    GbSmallVector<VkSemaphoreSubmitInfo, 8> waitSubmitInfos(waitSemaphoreInfoCount);

    const auto createSubmitInfos = [&](uint32_t infoCount, auto& semaphoreInfos, auto& submitInfos)
    {
      for (uint32_t i = 0; i < infoCount; i++)
      {
        CGPU_RESOLVE_SEMAPHORE(ctx, semaphoreInfos[i].semaphore, isemaphore);

        submitInfos[i] = VkSemaphoreSubmitInfo {
          .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
          .pNext = nullptr,
          .semaphore = isemaphore->semaphore,
          .value = semaphoreInfos[i].value,
          .stageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
          .deviceIndex = 0 // only relevant if in device group
        };
      }
    };

    createSubmitInfos(signalSemaphoreInfoCount, signalSemaphoreInfos, signalSubmitInfos);
    createSubmitInfos(waitSemaphoreInfoCount, waitSemaphoreInfos, waitSubmitInfos);

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

    if (idevice->table.vkQueueSubmit2KHR(idevice->computeQueue,
                                         1,
                                         &submitInfo,
                                         VK_NULL_HANDLE) != VK_SUCCESS)
    {
      CGPU_FATAL("failed to submit command buffer");
    }
  }

  const CgpuDeviceFeatures& cgpuGetDeviceFeatures(CgpuContext* ctx)
  {
    CgpuIDevice* idevice = &ctx->idevice;
    return idevice->features;
  }

  const CgpuDeviceProperties& cgpuGetDeviceProperties(CgpuContext* ctx)
  {
    CgpuIDevice* idevice = &ctx->idevice;
    return idevice->properties;
  }
}
