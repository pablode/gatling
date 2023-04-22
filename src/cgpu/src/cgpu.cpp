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
#include "dataStoreCpu.h"
#include "shader_reflection.h"

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

#include <volk.h>

#include <vma.h>

#include <memory>

// TODO: should be in 'gtl/gb' subfolder
#include <smallVector.h>

using namespace gtl;

#define CGPU_MIN_VK_API_VERSION VK_API_VERSION_1_1

/* Internal structures. */

struct CgpuIDevice
{
  VkDevice                     logical_device;
  VkPhysicalDevice             physical_device;
  VkQueue                      compute_queue;
  VkCommandPool                command_pool;
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
  VkImageView   image_view;
  VmaAllocation allocation;
  uint64_t      size;
  uint32_t      width;
  uint32_t      height;
  uint32_t      depth;
  VkImageLayout layout;
  VkAccessFlags access_mask;
};

struct CgpuIPipeline
{
  VkPipeline                                       pipeline;
  VkPipelineLayout                                 layout;
  VkDescriptorPool                                 descriptor_pool;
  VkDescriptorSet                                  descriptor_set;
  VkDescriptorSetLayout                            descriptor_set_layout;
  GbSmallVector<VkDescriptorSetLayoutBinding, 128> descriptor_set_layout_bindings;
  VkPipelineBindPoint                              bind_point;
  VkStridedDeviceAddressRegionKHR                  sbtRgen;
  VkStridedDeviceAddressRegionKHR                  sbtMiss;
  VkStridedDeviceAddressRegionKHR                  sbtHit;
  CgpuIBuffer                                      sbt;
};

struct CgpuIShader
{
  VkShaderModule module;
  CgpuShaderReflection reflection;
  VkShaderStageFlagBits stage_flags;
};

struct CgpuIFence
{
  VkFence fence;
};

struct CgpuICommandBuffer
{
  VkCommandBuffer command_buffer;
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
  CgpuDataStoreCpu<CgpuIDevice, 32> idevice_store;
  CgpuDataStoreCpu<CgpuIBuffer, 16> ibuffer_store;
  CgpuDataStoreCpu<CgpuIImage, 128> iimage_store;
  CgpuDataStoreCpu<CgpuIShader, 32> ishader_store;
  CgpuDataStoreCpu<CgpuIPipeline, 8> ipipeline_store;
  CgpuDataStoreCpu<CgpuIFence, 8> ifence_store;
  CgpuDataStoreCpu<CgpuICommandBuffer, 16> icommand_buffer_store;
  CgpuDataStoreCpu<CgpuISampler, 8> isampler_store;
  CgpuDataStoreCpu<CgpuIBlas, 1024> iblas_store;
  CgpuDataStoreCpu<CgpuITlas, 1> itlas_store;
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
  CGPU_INLINE static bool cgpu_resolve_##RESOURCE_NAME(                                   \
    HANDLE_TYPE handle,                                                                   \
    IRESOURCE_TYPE** idata)                                                               \
  {                                                                                       \
    return iinstance->RESOURCE_STORE.get(handle.handle, idata);                           \
  }

CGPU_RESOLVE_HANDLE(        device,        CgpuDevice,        CgpuIDevice,         idevice_store)
CGPU_RESOLVE_HANDLE(        buffer,        CgpuBuffer,        CgpuIBuffer,         ibuffer_store)
CGPU_RESOLVE_HANDLE(         image,         CgpuImage,         CgpuIImage,          iimage_store)
CGPU_RESOLVE_HANDLE(        shader,        CgpuShader,        CgpuIShader,         ishader_store)
CGPU_RESOLVE_HANDLE(      pipeline,      CgpuPipeline,      CgpuIPipeline,       ipipeline_store)
CGPU_RESOLVE_HANDLE(         fence,         CgpuFence,         CgpuIFence,          ifence_store)
CGPU_RESOLVE_HANDLE(command_buffer, CgpuCommandBuffer, CgpuICommandBuffer, icommand_buffer_store)
CGPU_RESOLVE_HANDLE(       sampler,       CgpuSampler,       CgpuISampler,        isampler_store)
CGPU_RESOLVE_HANDLE(          blas,          CgpuBlas,          CgpuIBlas,           iblas_store)
CGPU_RESOLVE_HANDLE(          tlas,          CgpuTlas,          CgpuITlas,           itlas_store)

static CgpuPhysicalDeviceFeatures cgpu_translate_physical_device_features(const VkPhysicalDeviceFeatures* vk_features)
{
  CgpuPhysicalDeviceFeatures features = {};
  features.textureCompressionBC = vk_features->textureCompressionBC;
  features.pipelineStatisticsQuery = vk_features->pipelineStatisticsQuery;
  features.shaderImageGatherExtended = vk_features->shaderImageGatherExtended;
  features.shaderStorageImageExtendedFormats = vk_features->shaderStorageImageExtendedFormats;
  features.shaderStorageImageReadWithoutFormat = vk_features->shaderStorageImageReadWithoutFormat;
  features.shaderStorageImageWriteWithoutFormat = vk_features->shaderStorageImageWriteWithoutFormat;
  features.shaderUniformBufferArrayDynamicIndexing = vk_features->shaderUniformBufferArrayDynamicIndexing;
  features.shaderSampledImageArrayDynamicIndexing = vk_features->shaderSampledImageArrayDynamicIndexing;
  features.shaderStorageBufferArrayDynamicIndexing = vk_features->shaderStorageBufferArrayDynamicIndexing;
  features.shaderStorageImageArrayDynamicIndexing = vk_features->shaderStorageImageArrayDynamicIndexing;
  features.shaderFloat64 = vk_features->shaderFloat64;
  features.shaderInt64 = vk_features->shaderInt64;
  features.shaderInt16 = vk_features->shaderInt16;
  features.sparseBinding = vk_features->sparseBinding;
  features.sparseResidencyBuffer = vk_features->sparseResidencyBuffer;
  features.sparseResidencyImage2D = vk_features->sparseResidencyImage2D;
  features.sparseResidencyImage3D = vk_features->sparseResidencyImage3D;
  features.sparseResidencyAliased = vk_features->sparseResidencyAliased;
  return features;
}

static CgpuPhysicalDeviceProperties cgpu_translate_physical_device_properties(const VkPhysicalDeviceLimits* vk_limits,
                                                                              const VkPhysicalDeviceSubgroupProperties* vk_subgroup_props,
                                                                              const VkPhysicalDeviceAccelerationStructurePropertiesKHR* vk_as_props,
                                                                              const VkPhysicalDeviceRayTracingPipelinePropertiesKHR* vk_rt_pipeline_props)
{
  CgpuPhysicalDeviceProperties properties = {};
  properties.maxImageDimension1D = vk_limits->maxImageDimension1D;
  properties.maxImageDimension2D = vk_limits->maxImageDimension2D;
  properties.maxImageDimension3D = vk_limits->maxImageDimension3D;
  properties.maxImageDimensionCube = vk_limits->maxImageDimensionCube;
  properties.maxImageArrayLayers = vk_limits->maxImageArrayLayers;
  properties.maxUniformBufferRange = vk_limits->maxUniformBufferRange;
  properties.maxStorageBufferRange = vk_limits->maxStorageBufferRange;
  properties.maxPushConstantsSize = vk_limits->maxPushConstantsSize;
  properties.maxMemoryAllocationCount = vk_limits->maxMemoryAllocationCount;
  properties.maxSamplerAllocationCount = vk_limits->maxSamplerAllocationCount;
  properties.bufferImageGranularity = vk_limits->bufferImageGranularity;
  properties.sparseAddressSpaceSize = vk_limits->sparseAddressSpaceSize;
  properties.maxBoundDescriptorSets = vk_limits->maxBoundDescriptorSets;
  properties.maxPerStageDescriptorSamplers = vk_limits->maxPerStageDescriptorSamplers;
  properties.maxPerStageDescriptorUniformBuffers = vk_limits->maxPerStageDescriptorUniformBuffers;
  properties.maxPerStageDescriptorStorageBuffers = vk_limits->maxPerStageDescriptorStorageBuffers;
  properties.maxPerStageDescriptorSampledImages = vk_limits->maxPerStageDescriptorSampledImages;
  properties.maxPerStageDescriptorStorageImages = vk_limits->maxPerStageDescriptorStorageImages;
  properties.maxPerStageDescriptorInputAttachments = vk_limits->maxPerStageDescriptorInputAttachments;
  properties.maxPerStageResources = vk_limits->maxPerStageResources;
  properties.maxDescriptorSetSamplers = vk_limits->maxDescriptorSetSamplers;
  properties.maxDescriptorSetUniformBuffers = vk_limits->maxDescriptorSetUniformBuffers;
  properties.maxDescriptorSetUniformBuffersDynamic = vk_limits->maxDescriptorSetUniformBuffersDynamic;
  properties.maxDescriptorSetStorageBuffers = vk_limits->maxDescriptorSetStorageBuffers;
  properties.maxDescriptorSetStorageBuffersDynamic = vk_limits->maxDescriptorSetStorageBuffersDynamic;
  properties.maxDescriptorSetSampledImages = vk_limits->maxDescriptorSetSampledImages;
  properties.maxDescriptorSetStorageImages = vk_limits->maxDescriptorSetStorageImages;
  properties.maxDescriptorSetInputAttachments = vk_limits->maxDescriptorSetInputAttachments;
  properties.maxComputeSharedMemorySize = vk_limits->maxComputeSharedMemorySize;
  properties.maxComputeWorkGroupCount[0] = vk_limits->maxComputeWorkGroupCount[0];
  properties.maxComputeWorkGroupCount[1] = vk_limits->maxComputeWorkGroupCount[1];
  properties.maxComputeWorkGroupCount[2] = vk_limits->maxComputeWorkGroupCount[2];
  properties.maxComputeWorkGroupInvocations = vk_limits->maxComputeWorkGroupInvocations;
  properties.maxComputeWorkGroupSize[0] = vk_limits->maxComputeWorkGroupSize[0];
  properties.maxComputeWorkGroupSize[1] = vk_limits->maxComputeWorkGroupSize[1];
  properties.maxComputeWorkGroupSize[2] = vk_limits->maxComputeWorkGroupSize[2];
  properties.mipmapPrecisionBits = vk_limits->mipmapPrecisionBits;
  properties.maxSamplerLodBias = vk_limits->maxSamplerLodBias;
  properties.maxSamplerAnisotropy = vk_limits->maxSamplerAnisotropy;
  properties.minMemoryMapAlignment = vk_limits->minMemoryMapAlignment;
  properties.minUniformBufferOffsetAlignment = vk_limits->minUniformBufferOffsetAlignment;
  properties.minStorageBufferOffsetAlignment = vk_limits->minStorageBufferOffsetAlignment;
  properties.minTexelOffset = vk_limits->minTexelOffset;
  properties.maxTexelOffset = vk_limits->maxTexelOffset;
  properties.minTexelGatherOffset = vk_limits->minTexelGatherOffset;
  properties.maxTexelGatherOffset = vk_limits->maxTexelGatherOffset;
  properties.minInterpolationOffset = vk_limits->minInterpolationOffset;
  properties.maxInterpolationOffset = vk_limits->maxInterpolationOffset;
  properties.subPixelInterpolationOffsetBits = vk_limits->subPixelInterpolationOffsetBits;
  properties.maxSampleMaskWords = vk_limits->maxSampleMaskWords;
  properties.timestampComputeAndGraphics = vk_limits->timestampComputeAndGraphics;
  properties.timestampPeriod = vk_limits->timestampPeriod;
  properties.discreteQueuePriorities = vk_limits->discreteQueuePriorities;
  properties.optimalBufferCopyOffsetAlignment = vk_limits->optimalBufferCopyOffsetAlignment;
  properties.optimalBufferCopyRowPitchAlignment = vk_limits->optimalBufferCopyRowPitchAlignment;
  properties.nonCoherentAtomSize = vk_limits->nonCoherentAtomSize;
  properties.subgroupSize = vk_subgroup_props->subgroupSize;
  properties.minAccelerationStructureScratchOffsetAlignment = vk_as_props->minAccelerationStructureScratchOffsetAlignment;
  properties.shaderGroupHandleSize = vk_rt_pipeline_props->shaderGroupHandleSize;
  properties.maxShaderGroupStride = vk_rt_pipeline_props->maxShaderGroupStride;
  properties.shaderGroupBaseAlignment = vk_rt_pipeline_props->shaderGroupBaseAlignment;
  properties.shaderGroupHandleCaptureReplaySize = vk_rt_pipeline_props->shaderGroupHandleCaptureReplaySize;
  properties.maxRayDispatchInvocationCount = vk_rt_pipeline_props->maxRayDispatchInvocationCount;
  properties.shaderGroupHandleAlignment = vk_rt_pipeline_props->shaderGroupHandleAlignment;
  properties.maxRayHitAttributeSize = vk_rt_pipeline_props->maxRayHitAttributeSize;
  return properties;
}

static VkSamplerAddressMode cgpu_translate_address_mode(CgpuSamplerAddressMode mode)
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

/* API method implementation. */

bool cgpu_initialize(const char* p_app_name,
                     uint32_t version_major,
                     uint32_t version_minor,
                     uint32_t version_patch)
{
  if (volkInitialize() != VK_SUCCESS ||
      volkGetInstanceVersion() < CGPU_MIN_VK_API_VERSION)
  {
    CGPU_RETURN_ERROR("failed to initialize volk");
  }

#ifndef NDEBUG
  const char* validation_layers[] = {
      "VK_LAYER_KHRONOS_validation"
  };
  const char* instance_extensions[] = {
      VK_EXT_DEBUG_UTILS_EXTENSION_NAME
  };
  uint32_t validation_layer_count = 1;
  uint32_t instance_extension_count = 1;
#else
  const char** validation_layers = nullptr;
  uint32_t validation_layer_count = 0;
  const char** instance_extensions = nullptr;
  uint32_t instance_extension_count = 0;
#endif

  VkApplicationInfo app_info;
  app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app_info.pNext = nullptr;
  app_info.pApplicationName = p_app_name;
  app_info.applicationVersion = VK_MAKE_VERSION(version_major, version_minor, version_patch);
  app_info.pEngineName = p_app_name;
  app_info.engineVersion = VK_MAKE_VERSION(version_major, version_minor, version_patch);
  app_info.apiVersion = CGPU_MIN_VK_API_VERSION;

  VkInstanceCreateInfo create_info;
  create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  create_info.pNext = nullptr;
  create_info.flags = 0;
  create_info.pApplicationInfo = &app_info;
  create_info.enabledLayerCount = validation_layer_count;
  create_info.ppEnabledLayerNames = validation_layers;
  create_info.enabledExtensionCount = instance_extension_count;
  create_info.ppEnabledExtensionNames = instance_extensions;

  VkInstance instance;
  if (vkCreateInstance(&create_info, nullptr, &instance) != VK_SUCCESS)
  {
    CGPU_RETURN_ERROR("failed to create vulkan instance");
  }

  volkLoadInstanceOnly(instance);

  iinstance = std::make_unique<CgpuIInstance>();
  iinstance->instance = instance;
  return true;
}

void cgpu_terminate()
{
  vkDestroyInstance(iinstance->instance, nullptr);
  iinstance.reset();
}

static bool cgpu_find_device_extension(const char* extension_name,
                                       uint32_t extension_count,
                                       VkExtensionProperties* extensions)
{
  for (uint32_t i = 0; i < extension_count; ++i)
  {
    const VkExtensionProperties* extension = &extensions[i];

    if (strcmp(extension->extensionName, extension_name) == 0)
    {
      return true;
    }
  }
  return false;
}

bool cgpu_create_device(CgpuDevice* p_device)
{
  p_device->handle = iinstance->idevice_store.allocate();

  CgpuIDevice* idevice;
  if (!cgpu_resolve_device(*p_device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  uint32_t phys_device_count;
  vkEnumeratePhysicalDevices(
    iinstance->instance,
    &phys_device_count,
    nullptr
  );

  if (phys_device_count == 0)
  {
    iinstance->idevice_store.free(p_device->handle);
    CGPU_RETURN_ERROR("no physical device found");
  }

  GbSmallVector<VkPhysicalDevice, 8> phys_devices;
  phys_devices.resize(phys_device_count);

  vkEnumeratePhysicalDevices(
    iinstance->instance,
    &phys_device_count,
    phys_devices.data()
  );

  idevice->physical_device = phys_devices[0];

  VkPhysicalDeviceFeatures features;
  vkGetPhysicalDeviceFeatures(idevice->physical_device, &features);
  idevice->features = cgpu_translate_physical_device_features(&features);

  VkPhysicalDeviceAccelerationStructurePropertiesKHR as_properties = {};
  as_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR;
  as_properties.pNext = nullptr;

  VkPhysicalDeviceRayTracingPipelinePropertiesKHR rt_pipeline_properties = {};
  rt_pipeline_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
  rt_pipeline_properties.pNext = &as_properties;

  VkPhysicalDeviceSubgroupProperties subgroup_properties;
  subgroup_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
  subgroup_properties.pNext = &rt_pipeline_properties;

  VkPhysicalDeviceProperties2 device_properties;
  device_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  device_properties.pNext = &subgroup_properties;

  vkGetPhysicalDeviceProperties2(idevice->physical_device, &device_properties);

  if (device_properties.properties.apiVersion < CGPU_MIN_VK_API_VERSION)
  {
    iinstance->idevice_store.free(p_device->handle);
    CGPU_RETURN_ERROR("unsupported vulkan version");
  }

  if ((subgroup_properties.supportedStages & VK_QUEUE_COMPUTE_BIT) != VK_QUEUE_COMPUTE_BIT ||
      (subgroup_properties.supportedOperations & VK_SUBGROUP_FEATURE_BASIC_BIT) != VK_SUBGROUP_FEATURE_BASIC_BIT ||
      (subgroup_properties.supportedOperations & VK_SUBGROUP_FEATURE_BALLOT_BIT) != VK_SUBGROUP_FEATURE_BALLOT_BIT)
  {
    iinstance->idevice_store.free(p_device->handle);
    CGPU_RETURN_ERROR("subgroup features not supported");
  }

  const VkPhysicalDeviceLimits* limits = &device_properties.properties.limits;
  idevice->properties = cgpu_translate_physical_device_properties(limits, &subgroup_properties, &as_properties, &rt_pipeline_properties);

  uint32_t device_ext_count;
  vkEnumerateDeviceExtensionProperties(
    idevice->physical_device,
    nullptr,
    &device_ext_count,
    nullptr
  );

  GbSmallVector<VkExtensionProperties, 1024> device_extensions;
  device_extensions.resize(device_ext_count);

  vkEnumerateDeviceExtensionProperties(
    idevice->physical_device,
    nullptr,
    &device_ext_count,
    device_extensions.data()
  );

  const char* required_extensions[] = {
    VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
    VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME, // required by VK_KHR_acceleration_structure
    VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME, // required by VK_KHR_acceleration_structure
    VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME, // required by VK_KHR_acceleration_structure
    VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
    VK_KHR_SPIRV_1_4_EXTENSION_NAME, // required by VK_KHR_ray_tracing_pipeline
    VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME, // required by VK_KHR_spirv_1_4
    VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME
  };
  uint32_t required_extension_count = sizeof(required_extensions) / sizeof(required_extensions[0]);

  GbSmallVector<const char*, 32> enabled_device_extensions;
  for (uint32_t i = 0; i < required_extension_count; i++)
  {
    const char* extension = required_extensions[i];

    if (!cgpu_find_device_extension(extension, device_ext_count, device_extensions.data()))
    {
      iinstance->idevice_store.free(p_device->handle);

      fprintf(stderr, "error in %s:%d: extension %s not supported\n", __FILE__, __LINE__, extension);
      return false;
    }

    enabled_device_extensions.push_back(extension);
  }

  const char* VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME = "VK_KHR_portability_subset";
  if (cgpu_find_device_extension(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME, device_ext_count, device_extensions.data()))
  {
    enabled_device_extensions.push_back(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME);
  }

#ifndef NDEBUG
  if (cgpu_find_device_extension(VK_KHR_SHADER_CLOCK_EXTENSION_NAME, device_ext_count, device_extensions.data()) && features.shaderInt64)
  {
    idevice->features.shaderClock = true;
    enabled_device_extensions.push_back(VK_KHR_SHADER_CLOCK_EXTENSION_NAME);
  }

#ifndef __APPLE__
  if (cgpu_find_device_extension(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME, device_ext_count, device_extensions.data()))
  {
    idevice->features.debugPrintf = true;
    enabled_device_extensions.push_back(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME);
  }
#endif
#endif
  if (cgpu_find_device_extension(VK_EXT_MEMORY_PRIORITY_EXTENSION_NAME, device_ext_count, device_extensions.data()) &&
      cgpu_find_device_extension(VK_EXT_PAGEABLE_DEVICE_LOCAL_MEMORY_EXTENSION_NAME, device_ext_count, device_extensions.data()))
  {
    idevice->features.pageableDeviceLocalMemory = true;
    enabled_device_extensions.push_back(VK_EXT_MEMORY_PRIORITY_EXTENSION_NAME);
    enabled_device_extensions.push_back(VK_EXT_PAGEABLE_DEVICE_LOCAL_MEMORY_EXTENSION_NAME);
  }

  uint32_t queue_family_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(
    idevice->physical_device,
    &queue_family_count,
    nullptr
  );

  GbSmallVector<VkQueueFamilyProperties, 32> queue_families;
  queue_families.resize(queue_family_count);

  vkGetPhysicalDeviceQueueFamilyProperties(
    idevice->physical_device,
    &queue_family_count,
    queue_families.data()
  );

  int32_t queue_family_index = -1;
  for (uint32_t i = 0; i < queue_family_count; ++i)
  {
    const VkQueueFamilyProperties* queue_family = &queue_families[i];

    if ((queue_family->queueFlags & VK_QUEUE_COMPUTE_BIT) && (queue_family->queueFlags & VK_QUEUE_TRANSFER_BIT))
    {
      queue_family_index = i;
    }
  }
  if (queue_family_index == -1) {
    iinstance->idevice_store.free(p_device->handle);
    CGPU_RETURN_ERROR("no suitable queue family");
  }

  VkDeviceQueueCreateInfo queue_create_info;
  queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queue_create_info.pNext = nullptr;
  queue_create_info.flags = 0;
  queue_create_info.queueFamilyIndex = queue_family_index;
  queue_create_info.queueCount = 1;
  const float queue_priority = 1.0f;
  queue_create_info.pQueuePriorities = &queue_priority;

  void* pNext = nullptr;

  VkPhysicalDevicePageableDeviceLocalMemoryFeaturesEXT pageable_memory_features = {};
  pageable_memory_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PAGEABLE_DEVICE_LOCAL_MEMORY_FEATURES_EXT;
  pageable_memory_features.pNext = nullptr;
  pageable_memory_features.pageableDeviceLocalMemory = VK_TRUE;

  if (idevice->features.pageableDeviceLocalMemory)
  {
    pNext = &pageable_memory_features;
  }

  VkPhysicalDeviceShaderClockFeaturesKHR shader_clock_features = {};
  shader_clock_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR;
  shader_clock_features.pNext = pNext;
  shader_clock_features.shaderSubgroupClock = VK_TRUE;
  shader_clock_features.shaderDeviceClock = VK_FALSE;

  if (idevice->features.shaderClock)
  {
    pNext = &shader_clock_features;
  }

  VkPhysicalDeviceAccelerationStructureFeaturesKHR acceleration_structure_features = {};
  acceleration_structure_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
  acceleration_structure_features.pNext = pNext;
  acceleration_structure_features.accelerationStructure = VK_TRUE;
  acceleration_structure_features.accelerationStructureCaptureReplay = VK_FALSE;
  acceleration_structure_features.accelerationStructureIndirectBuild = VK_FALSE;
  acceleration_structure_features.accelerationStructureHostCommands = VK_FALSE;
  acceleration_structure_features.descriptorBindingAccelerationStructureUpdateAfterBind = VK_FALSE;

  VkPhysicalDeviceRayTracingPipelineFeaturesKHR ray_tracing_pipeline_features = {};
  ray_tracing_pipeline_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
  ray_tracing_pipeline_features.pNext = &acceleration_structure_features;
  ray_tracing_pipeline_features.rayTracingPipeline = VK_TRUE;
  ray_tracing_pipeline_features.rayTracingPipelineShaderGroupHandleCaptureReplay = VK_FALSE;
  ray_tracing_pipeline_features.rayTracingPipelineShaderGroupHandleCaptureReplayMixed = VK_FALSE;
  ray_tracing_pipeline_features.rayTracingPipelineTraceRaysIndirect = VK_FALSE;
  ray_tracing_pipeline_features.rayTraversalPrimitiveCulling = VK_FALSE;

  VkPhysicalDeviceBufferDeviceAddressFeaturesKHR buffer_device_address_features = {};
  buffer_device_address_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
  buffer_device_address_features.pNext = &ray_tracing_pipeline_features;
  buffer_device_address_features.bufferDeviceAddress = VK_TRUE;
  buffer_device_address_features.bufferDeviceAddressCaptureReplay = VK_FALSE;
  buffer_device_address_features.bufferDeviceAddressMultiDevice = VK_FALSE;

  VkPhysicalDeviceDescriptorIndexingFeaturesEXT descriptor_indexing_features = {};
  descriptor_indexing_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES;
  descriptor_indexing_features.pNext = &buffer_device_address_features;
  descriptor_indexing_features.shaderInputAttachmentArrayDynamicIndexing = VK_FALSE;
  descriptor_indexing_features.shaderUniformTexelBufferArrayDynamicIndexing = VK_FALSE;
  descriptor_indexing_features.shaderStorageTexelBufferArrayDynamicIndexing = VK_FALSE;
  descriptor_indexing_features.shaderUniformBufferArrayNonUniformIndexing = VK_FALSE;
  descriptor_indexing_features.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
  descriptor_indexing_features.shaderStorageBufferArrayNonUniformIndexing = VK_FALSE;
  descriptor_indexing_features.shaderStorageImageArrayNonUniformIndexing = VK_TRUE;
  descriptor_indexing_features.shaderInputAttachmentArrayNonUniformIndexing = VK_FALSE;
  descriptor_indexing_features.shaderUniformTexelBufferArrayNonUniformIndexing = VK_FALSE;
  descriptor_indexing_features.shaderStorageTexelBufferArrayNonUniformIndexing = VK_FALSE;
  descriptor_indexing_features.descriptorBindingUniformBufferUpdateAfterBind = VK_FALSE;
  descriptor_indexing_features.descriptorBindingSampledImageUpdateAfterBind = VK_FALSE;
  descriptor_indexing_features.descriptorBindingStorageImageUpdateAfterBind = VK_FALSE;
  descriptor_indexing_features.descriptorBindingStorageBufferUpdateAfterBind = VK_FALSE;
  descriptor_indexing_features.descriptorBindingUniformTexelBufferUpdateAfterBind = VK_FALSE;
  descriptor_indexing_features.descriptorBindingStorageTexelBufferUpdateAfterBind = VK_FALSE;
  descriptor_indexing_features.descriptorBindingUpdateUnusedWhilePending = VK_FALSE;
  descriptor_indexing_features.descriptorBindingPartiallyBound = VK_FALSE;
  descriptor_indexing_features.descriptorBindingVariableDescriptorCount = VK_FALSE;
  descriptor_indexing_features.runtimeDescriptorArray = VK_FALSE;

  VkPhysicalDeviceShaderFloat16Int8Features shader_float16_int8_features = {};
  shader_float16_int8_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
  shader_float16_int8_features.pNext = &descriptor_indexing_features;
  shader_float16_int8_features.shaderFloat16 = VK_TRUE;
  shader_float16_int8_features.shaderInt8 = VK_FALSE;

  VkPhysicalDevice16BitStorageFeatures device_16bit_storage_features = {};
  device_16bit_storage_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
  device_16bit_storage_features.pNext = &shader_float16_int8_features;
  device_16bit_storage_features.storageBuffer16BitAccess = VK_TRUE;
  device_16bit_storage_features.uniformAndStorageBuffer16BitAccess = VK_TRUE;
  device_16bit_storage_features.storagePushConstant16 = VK_FALSE;
  device_16bit_storage_features.storageInputOutput16 = VK_FALSE;

  VkPhysicalDeviceFeatures2 device_features2;
  device_features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  device_features2.pNext = &device_16bit_storage_features;
  device_features2.features.robustBufferAccess = VK_FALSE;
  device_features2.features.fullDrawIndexUint32 = VK_FALSE;
  device_features2.features.imageCubeArray = VK_FALSE;
  device_features2.features.independentBlend = VK_FALSE;
  device_features2.features.geometryShader = VK_FALSE;
  device_features2.features.tessellationShader = VK_FALSE;
  device_features2.features.sampleRateShading = VK_FALSE;
  device_features2.features.dualSrcBlend = VK_FALSE;
  device_features2.features.logicOp = VK_FALSE;
  device_features2.features.multiDrawIndirect = VK_FALSE;
  device_features2.features.drawIndirectFirstInstance = VK_FALSE;
  device_features2.features.depthClamp = VK_FALSE;
  device_features2.features.depthBiasClamp = VK_FALSE;
  device_features2.features.fillModeNonSolid = VK_FALSE;
  device_features2.features.depthBounds = VK_FALSE;
  device_features2.features.wideLines = VK_FALSE;
  device_features2.features.largePoints = VK_FALSE;
  device_features2.features.alphaToOne = VK_FALSE;
  device_features2.features.multiViewport = VK_FALSE;
  device_features2.features.samplerAnisotropy = VK_TRUE;
  device_features2.features.textureCompressionETC2 = VK_FALSE;
  device_features2.features.textureCompressionASTC_LDR = VK_FALSE;
  device_features2.features.textureCompressionBC = VK_FALSE;
  device_features2.features.occlusionQueryPrecise = VK_FALSE;
  device_features2.features.pipelineStatisticsQuery = VK_FALSE;
  device_features2.features.vertexPipelineStoresAndAtomics = VK_FALSE;
  device_features2.features.fragmentStoresAndAtomics = VK_FALSE;
  device_features2.features.shaderTessellationAndGeometryPointSize = VK_FALSE;
  device_features2.features.shaderImageGatherExtended = VK_TRUE;
  device_features2.features.shaderStorageImageExtendedFormats = VK_FALSE;
  device_features2.features.shaderStorageImageMultisample = VK_FALSE;
  device_features2.features.shaderStorageImageReadWithoutFormat = VK_FALSE;
  device_features2.features.shaderStorageImageWriteWithoutFormat = VK_FALSE;
  device_features2.features.shaderUniformBufferArrayDynamicIndexing = VK_FALSE;
  device_features2.features.shaderSampledImageArrayDynamicIndexing = VK_TRUE;
  device_features2.features.shaderStorageBufferArrayDynamicIndexing = VK_FALSE;
  device_features2.features.shaderStorageImageArrayDynamicIndexing = VK_FALSE;
  device_features2.features.shaderClipDistance = VK_FALSE;
  device_features2.features.shaderCullDistance = VK_FALSE;
  device_features2.features.shaderFloat64 = VK_FALSE;
  device_features2.features.shaderInt64 = idevice->features.shaderClock;
  device_features2.features.shaderInt16 = VK_TRUE;
  device_features2.features.shaderResourceResidency = VK_FALSE;
  device_features2.features.shaderResourceMinLod = VK_FALSE;
  device_features2.features.sparseBinding = VK_FALSE;
  device_features2.features.sparseResidencyBuffer = VK_FALSE;
  device_features2.features.sparseResidencyImage2D = VK_FALSE;
  device_features2.features.sparseResidencyImage3D = VK_FALSE;
  device_features2.features.sparseResidency2Samples = VK_FALSE;
  device_features2.features.sparseResidency4Samples = VK_FALSE;
  device_features2.features.sparseResidency8Samples = VK_FALSE;
  device_features2.features.sparseResidency16Samples = VK_FALSE;
  device_features2.features.sparseResidencyAliased = VK_FALSE;
  device_features2.features.variableMultisampleRate = VK_FALSE;
  device_features2.features.inheritedQueries = VK_FALSE;

  VkDeviceCreateInfo device_create_info;
  device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  device_create_info.pNext = &device_features2;
  device_create_info.flags = 0;
  device_create_info.queueCreateInfoCount = 1;
  device_create_info.pQueueCreateInfos = &queue_create_info;
  /* These two fields are ignored by up-to-date implementations since
   * nowadays, there is no difference to instance validation layers. */
  device_create_info.enabledLayerCount = 0;
  device_create_info.ppEnabledLayerNames = nullptr;
  device_create_info.enabledExtensionCount = enabled_device_extensions.size();
  device_create_info.ppEnabledExtensionNames = enabled_device_extensions.data();
  device_create_info.pEnabledFeatures = nullptr;

  VkResult result = vkCreateDevice(
    idevice->physical_device,
    &device_create_info,
    nullptr,
    &idevice->logical_device
  );
  if (result != VK_SUCCESS) {
    iinstance->idevice_store.free(p_device->handle);
    CGPU_RETURN_ERROR("failed to create device");
  }

  volkLoadDeviceTable(
    &idevice->table,
    idevice->logical_device
  );

  idevice->table.vkGetDeviceQueue(
    idevice->logical_device,
    queue_family_index,
    0,
    &idevice->compute_queue
  );

  VkCommandPoolCreateInfo pool_info;
  pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  pool_info.pNext = nullptr;
  pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  pool_info.queueFamilyIndex = queue_family_index;

  result = idevice->table.vkCreateCommandPool(
    idevice->logical_device,
    &pool_info,
    nullptr,
    &idevice->command_pool
  );

  if (result != VK_SUCCESS)
  {
    iinstance->idevice_store.free(p_device->handle);

    idevice->table.vkDestroyDevice(
      idevice->logical_device,
      nullptr
    );

    CGPU_RETURN_ERROR("failed to create command pool");
  }

  VkQueryPoolCreateInfo timestamp_pool_info;
  timestamp_pool_info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
  timestamp_pool_info.pNext = nullptr;
  timestamp_pool_info.flags = 0;
  timestamp_pool_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
  timestamp_pool_info.queryCount = CGPU_MAX_TIMESTAMP_QUERIES;
  timestamp_pool_info.pipelineStatistics = 0;

  result = idevice->table.vkCreateQueryPool(
    idevice->logical_device,
    &timestamp_pool_info,
    nullptr,
    &idevice->timestamp_pool
  );

  if (result != VK_SUCCESS)
  {
    iinstance->idevice_store.free(p_device->handle);

    idevice->table.vkDestroyCommandPool(
      idevice->logical_device,
      idevice->command_pool,
      nullptr
    );
    idevice->table.vkDestroyDevice(
      idevice->logical_device,
      nullptr
    );

    CGPU_RETURN_ERROR("failed to create query pool");
  }

  VmaVulkanFunctions vulkan_functions = {};
  vulkan_functions.vkGetPhysicalDeviceProperties = vkGetPhysicalDeviceProperties;
  vulkan_functions.vkGetPhysicalDeviceMemoryProperties = vkGetPhysicalDeviceMemoryProperties;
  vulkan_functions.vkGetPhysicalDeviceMemoryProperties2KHR = vkGetPhysicalDeviceMemoryProperties2;
  vulkan_functions.vkAllocateMemory = idevice->table.vkAllocateMemory;
  vulkan_functions.vkFreeMemory = idevice->table.vkFreeMemory;
  vulkan_functions.vkMapMemory = idevice->table.vkMapMemory;
  vulkan_functions.vkUnmapMemory = idevice->table.vkUnmapMemory;
  vulkan_functions.vkFlushMappedMemoryRanges = idevice->table.vkFlushMappedMemoryRanges;
  vulkan_functions.vkInvalidateMappedMemoryRanges = idevice->table.vkInvalidateMappedMemoryRanges;
  vulkan_functions.vkBindBufferMemory = idevice->table.vkBindBufferMemory;
  vulkan_functions.vkBindImageMemory = idevice->table.vkBindImageMemory;
  vulkan_functions.vkGetBufferMemoryRequirements = idevice->table.vkGetBufferMemoryRequirements;
  vulkan_functions.vkGetImageMemoryRequirements = idevice->table.vkGetImageMemoryRequirements;
  vulkan_functions.vkCreateBuffer = idevice->table.vkCreateBuffer;
  vulkan_functions.vkDestroyBuffer = idevice->table.vkDestroyBuffer;
  vulkan_functions.vkCreateImage = idevice->table.vkCreateImage;
  vulkan_functions.vkDestroyImage = idevice->table.vkDestroyImage;
  vulkan_functions.vkCmdCopyBuffer = idevice->table.vkCmdCopyBuffer;
  vulkan_functions.vkGetBufferMemoryRequirements2KHR = idevice->table.vkGetBufferMemoryRequirements2;
  vulkan_functions.vkGetImageMemoryRequirements2KHR = idevice->table.vkGetImageMemoryRequirements2;
  vulkan_functions.vkBindBufferMemory2KHR = idevice->table.vkBindBufferMemory2;
  vulkan_functions.vkBindImageMemory2KHR = idevice->table.vkBindImageMemory2;

  VmaAllocatorCreateInfo alloc_create_info = {};
  alloc_create_info.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
  alloc_create_info.vulkanApiVersion = CGPU_MIN_VK_API_VERSION;
  alloc_create_info.physicalDevice = idevice->physical_device;
  alloc_create_info.device = idevice->logical_device;
  alloc_create_info.instance = iinstance->instance;
  alloc_create_info.pVulkanFunctions = &vulkan_functions;

  result = vmaCreateAllocator(&alloc_create_info, &idevice->allocator);

  if (result != VK_SUCCESS)
  {
    iinstance->idevice_store.free(p_device->handle);

    idevice->table.vkDestroyQueryPool(
      idevice->logical_device,
      idevice->timestamp_pool,
      nullptr
    );
    idevice->table.vkDestroyCommandPool(
      idevice->logical_device,
      idevice->command_pool,
      nullptr
    );
    idevice->table.vkDestroyDevice(
      idevice->logical_device,
      nullptr
    );
    CGPU_RETURN_ERROR("failed to create vma allocator");
  }

  return true;
}

bool cgpu_destroy_device(CgpuDevice device)
{
  CgpuIDevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  vmaDestroyAllocator(idevice->allocator);

  idevice->table.vkDestroyQueryPool(
    idevice->logical_device,
    idevice->timestamp_pool,
    nullptr
  );
  idevice->table.vkDestroyCommandPool(
    idevice->logical_device,
    idevice->command_pool,
    nullptr
  );
  idevice->table.vkDestroyDevice(
    idevice->logical_device,
    nullptr
  );

  iinstance->idevice_store.free(device.handle);
  return true;
}

bool cgpu_create_shader(CgpuDevice device,
                        uint64_t size,
                        const uint8_t* p_source,
                        CgpuShaderStageFlags stage_flags,
                        CgpuShader* p_shader)
{
  CgpuIDevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  p_shader->handle = iinstance->ishader_store.allocate();

  CgpuIShader* ishader;
  if (!cgpu_resolve_shader(*p_shader, &ishader)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  VkShaderModuleCreateInfo shader_module_create_info;
  shader_module_create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  shader_module_create_info.pNext = nullptr;
  shader_module_create_info.flags = 0;
  shader_module_create_info.codeSize = size;
  shader_module_create_info.pCode = (uint32_t*) p_source;

  VkResult result = idevice->table.vkCreateShaderModule(
    idevice->logical_device,
    &shader_module_create_info,
    nullptr,
    &ishader->module
  );
  if (result != VK_SUCCESS) {
    iinstance->ishader_store.free(p_shader->handle);
    CGPU_RETURN_ERROR("failed to create shader module");
  }

  if (!cgpu_perform_shader_reflection(size, (uint32_t*) p_source, &ishader->reflection))
  {
    idevice->table.vkDestroyShaderModule(
      idevice->logical_device,
      ishader->module,
      nullptr
    );
    iinstance->ishader_store.free(p_shader->handle);
    CGPU_RETURN_ERROR("failed to reflect shader");
  }

  ishader->stage_flags = (VkShaderStageFlagBits) stage_flags;

  return true;
}

bool cgpu_destroy_shader(CgpuDevice device,
                         CgpuShader shader)
{
  CgpuIDevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIShader* ishader;
  if (!cgpu_resolve_shader(shader, &ishader)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  cgpu_destroy_shader_reflection(&ishader->reflection);

  idevice->table.vkDestroyShaderModule(
    idevice->logical_device,
    ishader->module,
    nullptr
  );

  iinstance->ishader_store.free(shader.handle);

  return true;
}

static bool cgpu_create_ibuffer_aligned(CgpuIDevice* idevice,
                                        CgpuBufferUsageFlags usage,
                                        CgpuMemoryPropertyFlags memory_properties,
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

  VmaAllocationCreateInfo alloc_info = {};
  alloc_info.requiredFlags = (VkMemoryPropertyFlags) memory_properties;

  VkResult result;
  if (alignment > 0)
  {
    result = vmaCreateBufferWithAlignment(
      idevice->allocator,
      &buffer_info,
      &alloc_info,
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
      &alloc_info,
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

static bool cgpu_create_buffer_aligned(CgpuDevice device,
                                       CgpuBufferUsageFlags usage,
                                       CgpuMemoryPropertyFlags memory_properties,
                                       uint64_t size,
                                       uint64_t alignment,
                                       CgpuBuffer* p_buffer)
{
  CgpuIDevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  p_buffer->handle = iinstance->ibuffer_store.allocate();

  CgpuIBuffer* ibuffer;
  if (!cgpu_resolve_buffer(*p_buffer, &ibuffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  if (!cgpu_create_ibuffer_aligned(idevice, usage, memory_properties, size, alignment, ibuffer))
  {
    iinstance->ibuffer_store.free(p_buffer->handle);
    CGPU_RETURN_ERROR("failed to create buffer");
  }

  return true;
}

bool cgpu_create_buffer(CgpuDevice device,
                        CgpuBufferUsageFlags usage,
                        CgpuMemoryPropertyFlags memory_properties,
                        uint64_t size,
                        CgpuBuffer* p_buffer)
{
  uint64_t alignment = 0;

  return cgpu_create_buffer_aligned(device, usage, memory_properties, size, alignment, p_buffer);
}

static void cgpu_destroy_ibuffer(CgpuIDevice* idevice,
                                 CgpuIBuffer* ibuffer)
{
  vmaDestroyBuffer(idevice->allocator, ibuffer->buffer, ibuffer->allocation);
}

bool cgpu_destroy_buffer(CgpuDevice device,
                         CgpuBuffer buffer)
{
  CgpuIDevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIBuffer* ibuffer;
  if (!cgpu_resolve_buffer(buffer, &ibuffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  cgpu_destroy_ibuffer(idevice, ibuffer);

  iinstance->ibuffer_store.free(buffer.handle);

  return true;
}

bool cgpu_map_buffer(CgpuDevice device,
                     CgpuBuffer buffer,
                     void** pp_mapped_mem)
{
  CgpuIDevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIBuffer* ibuffer;
  if (!cgpu_resolve_buffer(buffer, &ibuffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  if (vmaMapMemory(idevice->allocator, ibuffer->allocation, pp_mapped_mem) != VK_SUCCESS) {
    CGPU_RETURN_ERROR("failed to map buffer memory");
  }
  return true;
}

bool cgpu_unmap_buffer(CgpuDevice device,
                       CgpuBuffer buffer)
{
  CgpuIDevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIBuffer* ibuffer;
  if (!cgpu_resolve_buffer(buffer, &ibuffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  vmaUnmapMemory(idevice->allocator, ibuffer->allocation);
  return true;
}

bool cgpu_create_image(CgpuDevice device,
                       const CgpuImageDesc* image_desc,
                       CgpuImage* p_image)
{
  CgpuIDevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  p_image->handle = iinstance->iimage_store.allocate();

  CgpuIImage* iimage;
  if (!cgpu_resolve_image(*p_image, &iimage)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  VkImageTiling vk_image_tiling = VK_IMAGE_TILING_OPTIMAL;
  if (image_desc->usage == CGPU_IMAGE_USAGE_FLAG_TRANSFER_SRC ||
      image_desc->usage == CGPU_IMAGE_USAGE_FLAG_TRANSFER_DST)
  {
    vk_image_tiling = VK_IMAGE_TILING_LINEAR;
  }

  VkImageCreateInfo image_info;
  image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  image_info.pNext = nullptr;
  image_info.flags = 0;
  image_info.imageType = image_desc->is3d ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;
  image_info.format = (VkFormat) image_desc->format;
  image_info.extent.width = image_desc->width;
  image_info.extent.height = image_desc->height;
  image_info.extent.depth = image_desc->is3d ? image_desc->depth : 1;
  image_info.mipLevels = 1;
  image_info.arrayLayers = 1;
  image_info.samples = VK_SAMPLE_COUNT_1_BIT;
  image_info.tiling = vk_image_tiling;
  image_info.usage = (VkImageUsageFlags) image_desc->usage;
  image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  image_info.queueFamilyIndexCount = 0;
  image_info.pQueueFamilyIndices = nullptr;
  image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

  VmaAllocationCreateInfo alloc_info = {};
  alloc_info.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

  VkResult result = vmaCreateImage(
    idevice->allocator,
    &image_info,
    &alloc_info,
    &iimage->image,
    &iimage->allocation,
    nullptr
  );

  if (result != VK_SUCCESS) {
    iinstance->iimage_store.free(p_image->handle);
    CGPU_RETURN_ERROR("failed to create image");
  }

  VmaAllocationInfo allocation_info;
  vmaGetAllocationInfo(
    idevice->allocator,
    iimage->allocation,
    &allocation_info
  );

  iimage->size = allocation_info.size;

  VkImageViewCreateInfo image_view_info;
  image_view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  image_view_info.pNext = nullptr;
  image_view_info.flags = 0;
  image_view_info.image = iimage->image;
  image_view_info.viewType = image_desc->is3d ? VK_IMAGE_VIEW_TYPE_3D : VK_IMAGE_VIEW_TYPE_2D;
  image_view_info.format = (VkFormat) image_desc->format;
  image_view_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
  image_view_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
  image_view_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
  image_view_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
  image_view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  image_view_info.subresourceRange.baseMipLevel = 0;
  image_view_info.subresourceRange.levelCount = 1;
  image_view_info.subresourceRange.baseArrayLayer = 0;
  image_view_info.subresourceRange.layerCount = 1;

  result = idevice->table.vkCreateImageView(
    idevice->logical_device,
    &image_view_info,
    nullptr,
    &iimage->image_view
  );
  if (result != VK_SUCCESS)
  {
    iinstance->iimage_store.free(p_image->handle);
    vmaDestroyImage(idevice->allocator, iimage->image, iimage->allocation);
    CGPU_RETURN_ERROR("failed to create image view");
  }

  iimage->width = image_desc->width;
  iimage->height = image_desc->height;
  iimage->depth = image_desc->is3d ? image_desc->depth : 1;
  iimage->layout = VK_IMAGE_LAYOUT_UNDEFINED;
  iimage->access_mask = 0;

  return true;
}

bool cgpu_destroy_image(CgpuDevice device,
                        CgpuImage image)
{
  CgpuIDevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIImage* iimage;
  if (!cgpu_resolve_image(image, &iimage)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  idevice->table.vkDestroyImageView(
    idevice->logical_device,
    iimage->image_view,
    nullptr
  );

  vmaDestroyImage(idevice->allocator, iimage->image, iimage->allocation);

  iinstance->iimage_store.free(image.handle);

  return true;
}

bool cgpu_map_image(CgpuDevice device,
                    CgpuImage image,
                    void** pp_mapped_mem)
{
  CgpuIDevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIImage* iimage;
  if (!cgpu_resolve_image(image, &iimage)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  if (vmaMapMemory(idevice->allocator, iimage->allocation, pp_mapped_mem) != VK_SUCCESS) {
    CGPU_RETURN_ERROR("failed to map image memory");
  }
  return true;
}

bool cgpu_unmap_image(CgpuDevice device,
                      CgpuImage image)
{
  CgpuIDevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIImage* iimage;
  if (!cgpu_resolve_image(image, &iimage)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  vmaUnmapMemory(idevice->allocator, iimage->allocation);
  return true;
}

bool cgpu_create_sampler(CgpuDevice device,
                         CgpuSamplerAddressMode address_mode_u,
                         CgpuSamplerAddressMode address_mode_v,
                         CgpuSamplerAddressMode address_mode_w,
                         CgpuSampler* p_sampler)
{
  CgpuIDevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  p_sampler->handle = iinstance->isampler_store.allocate();

  CgpuISampler* isampler;
  if (!cgpu_resolve_sampler(*p_sampler, &isampler)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  // Emulate MDL's clip wrap mode if necessary; use optimal mode (according to ARM) if not.
  bool clampToBlack = (address_mode_u == CGPU_SAMPLER_ADDRESS_MODE_CLAMP_TO_BLACK) ||
                      (address_mode_v == CGPU_SAMPLER_ADDRESS_MODE_CLAMP_TO_BLACK) ||
                      (address_mode_w == CGPU_SAMPLER_ADDRESS_MODE_CLAMP_TO_BLACK);

  VkSamplerCreateInfo create_info;
  create_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  create_info.pNext = nullptr;
  create_info.flags = 0;
  create_info.magFilter = VK_FILTER_LINEAR;
  create_info.minFilter = VK_FILTER_LINEAR;
  create_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  create_info.addressModeU = cgpu_translate_address_mode(address_mode_u);
  create_info.addressModeV = cgpu_translate_address_mode(address_mode_v);
  create_info.addressModeW = cgpu_translate_address_mode(address_mode_w);
  create_info.mipLodBias = 0.0f;
  create_info.anisotropyEnable = VK_FALSE;
  create_info.maxAnisotropy = 1.0f;
  create_info.compareEnable = VK_FALSE;
  create_info.compareOp = VK_COMPARE_OP_NEVER;
  create_info.minLod = 0.0f;
  create_info.maxLod = VK_LOD_CLAMP_NONE;
  create_info.borderColor = clampToBlack ? VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK : VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
  create_info.unnormalizedCoordinates = VK_FALSE;

  VkResult result = idevice->table.vkCreateSampler(
    idevice->logical_device,
    &create_info,
    nullptr,
    &isampler->sampler
  );

  if (result != VK_SUCCESS) {
    iinstance->isampler_store.free(p_sampler->handle);
    CGPU_RETURN_ERROR("failed to create sampler");
  }

  return true;
}

bool cgpu_destroy_sampler(CgpuDevice device,
                          CgpuSampler sampler)
{
  CgpuIDevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuISampler* isampler;
  if (!cgpu_resolve_sampler(sampler, &isampler)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  idevice->table.vkDestroySampler(idevice->logical_device, isampler->sampler, nullptr);

  iinstance->isampler_store.free(sampler.handle);

  return true;
}

static bool cgpu_create_pipeline_layout(CgpuIDevice* idevice, CgpuIPipeline* ipipeline, CgpuIShader* ishader, VkShaderStageFlags stageFlags)
{
  VkPushConstantRange push_const_range;
  push_const_range.stageFlags = stageFlags;
  push_const_range.offset = 0;
  push_const_range.size = ishader->reflection.push_constants_size;

  VkPipelineLayoutCreateInfo pipeline_layout_create_info;
  pipeline_layout_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipeline_layout_create_info.pNext = nullptr;
  pipeline_layout_create_info.flags = 0;
  pipeline_layout_create_info.setLayoutCount = 1;
  pipeline_layout_create_info.pSetLayouts = &ipipeline->descriptor_set_layout;
  pipeline_layout_create_info.pushConstantRangeCount = push_const_range.size ? 1 : 0;
  pipeline_layout_create_info.pPushConstantRanges = &push_const_range;

  return idevice->table.vkCreatePipelineLayout(idevice->logical_device,
                                               &pipeline_layout_create_info,
                                               nullptr,
                                               &ipipeline->layout) == VK_SUCCESS;
}

static bool cgpu_create_pipeline_descriptors(CgpuIDevice* idevice, CgpuIPipeline* ipipeline, CgpuIShader* ishader, VkShaderStageFlags stageFlags)
{
  const CgpuShaderReflection* shader_reflection = &ishader->reflection;

  for (uint32_t i = 0; i < shader_reflection->binding_count; i++)
  {
    const CgpuShaderReflectionBinding* binding_reflection = &shader_reflection->bindings[i];

    VkDescriptorSetLayoutBinding layout_binding;
    layout_binding.binding = binding_reflection->binding;
    layout_binding.descriptorType = (VkDescriptorType) binding_reflection->descriptor_type;
    layout_binding.descriptorCount = binding_reflection->count;
    layout_binding.stageFlags = stageFlags;
    layout_binding.pImmutableSamplers = nullptr;

    ipipeline->descriptor_set_layout_bindings.push_back(layout_binding);
  }

  VkDescriptorSetLayoutCreateInfo descriptor_set_layout_create_info;
  descriptor_set_layout_create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  descriptor_set_layout_create_info.pNext = nullptr;
  descriptor_set_layout_create_info.flags = 0;
  descriptor_set_layout_create_info.bindingCount = ipipeline->descriptor_set_layout_bindings.size();
  descriptor_set_layout_create_info.pBindings = ipipeline->descriptor_set_layout_bindings.data();

  VkResult result = idevice->table.vkCreateDescriptorSetLayout(
    idevice->logical_device,
    &descriptor_set_layout_create_info,
    nullptr,
    &ipipeline->descriptor_set_layout
  );

  if (result != VK_SUCCESS) {
    CGPU_RETURN_ERROR("failed to create descriptor set layout");
  }

  uint32_t buffer_count = 0;
  uint32_t storage_image_count = 0;
  uint32_t sampled_image_count = 0;
  uint32_t sampler_count = 0;
  uint32_t as_count = 0;

  for (uint32_t i = 0; i < shader_reflection->binding_count; i++)
  {
    const CgpuShaderReflectionBinding* binding = &shader_reflection->bindings[i];

    switch (binding->descriptor_type)
    {
    case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: buffer_count += binding->count; break;
    case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE: storage_image_count += binding->count; break;
    case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE: sampled_image_count += binding->count; break;
    case VK_DESCRIPTOR_TYPE_SAMPLER: sampler_count += binding->count; break;
    case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR: as_count += binding->count; break;
    default: {
      idevice->table.vkDestroyDescriptorSetLayout(
        idevice->logical_device,
        ipipeline->descriptor_set_layout,
        nullptr
      );
      CGPU_RETURN_ERROR("invalid descriptor type");
    }
    }
  }

  uint32_t pool_size_count = 0;
  VkDescriptorPoolSize pool_sizes[16];

  if (buffer_count > 0)
  {
    pool_sizes[pool_size_count].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_sizes[pool_size_count].descriptorCount = buffer_count;
    pool_size_count++;
  }
  if (storage_image_count > 0)
  {
    pool_sizes[pool_size_count].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    pool_sizes[pool_size_count].descriptorCount = storage_image_count;
    pool_size_count++;
  }
  if (sampled_image_count > 0)
  {
    pool_sizes[pool_size_count].type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    pool_sizes[pool_size_count].descriptorCount = sampled_image_count;
    pool_size_count++;
  }
  if (sampler_count > 0)
  {
    pool_sizes[pool_size_count].type = VK_DESCRIPTOR_TYPE_SAMPLER;
    pool_sizes[pool_size_count].descriptorCount = sampler_count;
    pool_size_count++;
  }
  if (as_count > 0)
  {
    pool_sizes[pool_size_count].type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    pool_sizes[pool_size_count].descriptorCount = as_count;
    pool_size_count++;
  }

  VkDescriptorPoolCreateInfo descriptor_pool_create_info;
  descriptor_pool_create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  descriptor_pool_create_info.pNext = nullptr;
  descriptor_pool_create_info.flags = 0;
  descriptor_pool_create_info.maxSets = 1;
  descriptor_pool_create_info.poolSizeCount = pool_size_count;
  descriptor_pool_create_info.pPoolSizes = pool_sizes;

  result = idevice->table.vkCreateDescriptorPool(
    idevice->logical_device,
    &descriptor_pool_create_info,
    nullptr,
    &ipipeline->descriptor_pool
  );
  if (result != VK_SUCCESS) {
    idevice->table.vkDestroyDescriptorSetLayout(
      idevice->logical_device,
      ipipeline->descriptor_set_layout,
      nullptr
    );
    CGPU_RETURN_ERROR("failed to create descriptor pool");
  }

  VkDescriptorSetAllocateInfo descriptor_set_allocate_info;
  descriptor_set_allocate_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  descriptor_set_allocate_info.pNext = nullptr;
  descriptor_set_allocate_info.descriptorPool = ipipeline->descriptor_pool;
  descriptor_set_allocate_info.descriptorSetCount = 1;
  descriptor_set_allocate_info.pSetLayouts = &ipipeline->descriptor_set_layout;

  result = idevice->table.vkAllocateDescriptorSets(
    idevice->logical_device,
    &descriptor_set_allocate_info,
    &ipipeline->descriptor_set
  );
  if (result != VK_SUCCESS) {
    idevice->table.vkDestroyDescriptorPool(
      idevice->logical_device,
      ipipeline->descriptor_pool,
      nullptr
    );
    idevice->table.vkDestroyDescriptorSetLayout(
      idevice->logical_device,
      ipipeline->descriptor_set_layout,
      nullptr
    );
    CGPU_RETURN_ERROR("failed to allocate descriptor set");
  }

  return true;
}

bool cgpu_create_compute_pipeline(CgpuDevice device,
                                  CgpuShader shader,
                                  CgpuPipeline* p_pipeline)
{
  CgpuIDevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIShader* ishader;
  if (!cgpu_resolve_shader(shader, &ishader)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  p_pipeline->handle = iinstance->ipipeline_store.allocate();

  CgpuIPipeline* ipipeline;
  if (!cgpu_resolve_pipeline(*p_pipeline, &ipipeline)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  if (!cgpu_create_pipeline_descriptors(idevice, ipipeline, ishader, VK_SHADER_STAGE_COMPUTE_BIT))
  {
    iinstance->ipipeline_store.free(p_pipeline->handle);
    CGPU_RETURN_ERROR("failed to create descriptor set layout");
  }

  if (!cgpu_create_pipeline_layout(idevice, ipipeline, ishader, VK_SHADER_STAGE_COMPUTE_BIT))
  {
    iinstance->ipipeline_store.free(p_pipeline->handle);
    idevice->table.vkDestroyDescriptorSetLayout(
      idevice->logical_device,
      ipipeline->descriptor_set_layout,
      nullptr
    );
    idevice->table.vkDestroyDescriptorPool(
      idevice->logical_device,
      ipipeline->descriptor_pool,
      nullptr
    );
    CGPU_RETURN_ERROR("failed to create pipeline layout");
  }

  VkPipelineShaderStageCreateInfo pipeline_shader_stage_create_info;
  pipeline_shader_stage_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  pipeline_shader_stage_create_info.pNext = nullptr;
  pipeline_shader_stage_create_info.flags = 0;
  pipeline_shader_stage_create_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  pipeline_shader_stage_create_info.module = ishader->module;
  pipeline_shader_stage_create_info.pName = "main";
  pipeline_shader_stage_create_info.pSpecializationInfo = nullptr;

  VkComputePipelineCreateInfo pipeline_create_info;
  pipeline_create_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipeline_create_info.pNext = nullptr;
  pipeline_create_info.flags = 0;
  pipeline_create_info.stage = pipeline_shader_stage_create_info;
  pipeline_create_info.layout = ipipeline->layout;
  pipeline_create_info.basePipelineHandle = VK_NULL_HANDLE;
  pipeline_create_info.basePipelineIndex = 0;

  VkResult result = idevice->table.vkCreateComputePipelines(
    idevice->logical_device,
    VK_NULL_HANDLE,
    1,
    &pipeline_create_info,
    nullptr,
    &ipipeline->pipeline
  );

  if (result != VK_SUCCESS) {
    iinstance->ipipeline_store.free(p_pipeline->handle);
    idevice->table.vkDestroyPipelineLayout(
      idevice->logical_device,
      ipipeline->layout,
      nullptr
    );
    idevice->table.vkDestroyDescriptorSetLayout(
      idevice->logical_device,
      ipipeline->descriptor_set_layout,
      nullptr
    );
    idevice->table.vkDestroyDescriptorPool(
      idevice->logical_device,
      ipipeline->descriptor_pool,
      nullptr
    );
    CGPU_RETURN_ERROR("failed to create compute pipeline");
  }

  ipipeline->bind_point = VK_PIPELINE_BIND_POINT_COMPUTE;

  return true;
}

static VkDeviceAddress cgpu_get_buffer_device_address(CgpuIDevice* idevice, CgpuIBuffer* ibuffer)
{
  VkBufferDeviceAddressInfoKHR address_info = {};
  address_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
  address_info.pNext = nullptr;
  address_info.buffer = ibuffer->buffer;
  return idevice->table.vkGetBufferDeviceAddressKHR(idevice->logical_device, &address_info);
}

static uint32_t cgpu_align_size(uint32_t size, uint32_t alignment)
{
    return (size + (alignment - 1)) & ~(alignment - 1);
}

static bool cgpu_create_rt_pipeline_sbt(CgpuIDevice* idevice, CgpuIPipeline* ipipeline, uint32_t groupCount, uint32_t miss_shader_count, uint32_t hit_group_count)
{
  uint32_t handleSize = idevice->properties.shaderGroupHandleSize;
  uint32_t alignedHandleSize = cgpu_align_size(handleSize, idevice->properties.shaderGroupHandleAlignment);

  ipipeline->sbtRgen.stride = cgpu_align_size(alignedHandleSize, idevice->properties.shaderGroupBaseAlignment);
  ipipeline->sbtRgen.size = ipipeline->sbtRgen.stride; // Special raygen condition: size must be equal to stride
  ipipeline->sbtMiss.stride = alignedHandleSize;
  ipipeline->sbtMiss.size = cgpu_align_size(miss_shader_count * alignedHandleSize, idevice->properties.shaderGroupBaseAlignment);
  ipipeline->sbtHit.stride = alignedHandleSize;
  ipipeline->sbtHit.size = cgpu_align_size(hit_group_count * alignedHandleSize, idevice->properties.shaderGroupBaseAlignment);

  uint32_t firstGroup = 0;
  uint32_t dataSize = handleSize * groupCount;

  GbSmallVector<uint8_t, 64> handleData(dataSize);
  if (idevice->table.vkGetRayTracingShaderGroupHandlesKHR(idevice->logical_device, ipipeline->pipeline, firstGroup, groupCount, handleData.size(), handleData.data()) != VK_SUCCESS)
  {
    CGPU_RETURN_ERROR("failed to create sbt handles");
  }

  VkDeviceSize sbtSize = ipipeline->sbtRgen.size + ipipeline->sbtMiss.size + ipipeline->sbtHit.size;
  CgpuBufferUsageFlags bufferUsageFlags = CGPU_BUFFER_USAGE_FLAG_TRANSFER_SRC | CGPU_BUFFER_USAGE_FLAG_SHADER_DEVICE_ADDRESS | CGPU_BUFFER_USAGE_FLAG_SHADER_BINDING_TABLE_BIT_KHR;
  CgpuMemoryPropertyFlags bufferMemPropFlags = CGPU_MEMORY_PROPERTY_FLAG_HOST_VISIBLE | CGPU_MEMORY_PROPERTY_FLAG_HOST_CACHED;

  if (!cgpu_create_ibuffer_aligned(idevice, bufferUsageFlags, bufferMemPropFlags, sbtSize, 0, &ipipeline->sbt))
  {
    CGPU_RETURN_ERROR("failed to create sbt buffer");
  }

  VkDeviceAddress sbtDeviceAddress = cgpu_get_buffer_device_address(idevice, &ipipeline->sbt);
  ipipeline->sbtRgen.deviceAddress = sbtDeviceAddress;
  ipipeline->sbtMiss.deviceAddress = sbtDeviceAddress + ipipeline->sbtRgen.size;
  ipipeline->sbtHit.deviceAddress = sbtDeviceAddress + ipipeline->sbtRgen.size + ipipeline->sbtMiss.size;

  uint8_t* sbt_mem;
  if (vmaMapMemory(idevice->allocator, ipipeline->sbt.allocation, (void**)&sbt_mem) != VK_SUCCESS) {
    CGPU_RETURN_ERROR("failed to map buffer memory");
  }

  uint32_t handle_count = 0;
  uint8_t* sbt_mem_rgen = &sbt_mem[0];
  uint8_t* sbt_mem_miss = &sbt_mem[ipipeline->sbtRgen.size];
  uint8_t* sbt_mem_hit = &sbt_mem[ipipeline->sbtRgen.size + ipipeline->sbtMiss.size];

  // Rgen
  sbt_mem = sbt_mem_rgen;
  memcpy(sbt_mem, &handleData[handleSize * (handle_count++)], handleSize);
  // Miss
  sbt_mem = sbt_mem_miss;
  for (uint32_t i = 0; i < miss_shader_count; i++)
  {
    memcpy(sbt_mem, &handleData[handleSize * (handle_count++)], handleSize);
    sbt_mem += ipipeline->sbtMiss.stride;
  }
  // Hit
  sbt_mem = sbt_mem_hit;
  for (uint32_t i = 0; i < hit_group_count; i++)
  {
    memcpy(sbt_mem, &handleData[handleSize * (handle_count++)], handleSize);
    sbt_mem += ipipeline->sbtHit.stride;
  }

  vmaUnmapMemory(idevice->allocator, ipipeline->sbt.allocation);
  return true;
}

bool cgpu_create_rt_pipeline(CgpuDevice device,
                             const cgpu_rt_pipeline_desc* desc,
                             CgpuPipeline* p_pipeline)
{
  CgpuIDevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  p_pipeline->handle = iinstance->ipipeline_store.allocate();

  CgpuIPipeline* ipipeline;
  if (!cgpu_resolve_pipeline(*p_pipeline, &ipipeline)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  // Zero-init for cleanup routine.
  memset(ipipeline, 0, sizeof(CgpuIPipeline));

  // In a ray tracing pipeline, all shaders are expected to have the same descriptor set layouts. Here, we
  // construct the descriptor set layouts and the pipeline layout from only the ray generation shader.
  CgpuIShader* irgen_shader;
  if (!cgpu_resolve_shader(desc->rgen_shader, &irgen_shader)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  // Set up stages
  GbSmallVector<VkPipelineShaderStageCreateInfo, 128> stages;
  VkShaderStageFlags pipelineStageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

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
  pushStage(VK_SHADER_STAGE_RAYGEN_BIT_KHR, irgen_shader->module);

  // Miss
  if (desc->miss_shader_count > 0)
  {
    pipelineStageFlags |= VK_SHADER_STAGE_MISS_BIT_KHR;
  }
  for (uint32_t i = 0; i < desc->miss_shader_count; i++)
  {
    CgpuIShader* imiss_shader;
    if (!cgpu_resolve_shader(desc->miss_shaders[i], &imiss_shader)) {
      CGPU_RETURN_ERROR_INVALID_HANDLE;
    }
    assert(imiss_shader->module);

    pushStage(VK_SHADER_STAGE_MISS_BIT_KHR, imiss_shader->module);
  }

  // Hit
  for (uint32_t i = 0; i < desc->hit_group_count; i++)
  {
    const CgpuRtHitGroup* hit_group = &desc->hit_groups[i];

    // Closest hit (optional)
    if (hit_group->closestHitShader.handle != CGPU_INVALID_HANDLE)
    {
      CgpuIShader* iclosestHitShader;
      if (!cgpu_resolve_shader(hit_group->closestHitShader, &iclosestHitShader)) {
        CGPU_RETURN_ERROR_INVALID_HANDLE;
      }
      assert(iclosestHitShader->stage_flags == VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);

      pushStage(iclosestHitShader->stage_flags, iclosestHitShader->module);
      pipelineStageFlags |= iclosestHitShader->stage_flags;
    }

    // Any hit (optional)
    if (hit_group->anyHitShader.handle != CGPU_INVALID_HANDLE)
    {
      CgpuIShader* ianyHitShader;
      if (!cgpu_resolve_shader(hit_group->anyHitShader, &ianyHitShader)) {
        CGPU_RETURN_ERROR_INVALID_HANDLE;
      }
      assert(ianyHitShader->stage_flags == VK_SHADER_STAGE_ANY_HIT_BIT_KHR);

      pushStage(ianyHitShader->stage_flags, ianyHitShader->module);
      pipelineStageFlags |= ianyHitShader->stage_flags;
    }
  }

  // Set up groups
  GbSmallVector<VkRayTracingShaderGroupCreateInfoKHR, 128> groups;
  groups.resize(1/*rgen*/ + desc->miss_shader_count + desc->hit_group_count);

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

  uint32_t hitStageAndGroupOffset = 1/*rgen*/ + desc->miss_shader_count;
  uint32_t hitShaderStageIndex = hitStageAndGroupOffset;
  for (uint32_t i = 0; i < desc->hit_group_count; i++)
  {
    const CgpuRtHitGroup* hit_group = &desc->hit_groups[i];

    uint32_t groupIndex = hitStageAndGroupOffset + i;
    groups[groupIndex].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    groups[groupIndex].generalShader = VK_SHADER_UNUSED_KHR;

    if (hit_group->closestHitShader.handle != CGPU_INVALID_HANDLE)
    {
      groups[groupIndex].closestHitShader = (hitShaderStageIndex++);
    }
    else
    {
      anyNullClosestHitShader |= true;
    }

    if (hit_group->anyHitShader.handle != CGPU_INVALID_HANDLE)
    {
      groups[groupIndex].anyHitShader = (hitShaderStageIndex++);
    }
    else
    {
      anyNullAnyHitShader |= true;
    }
  }

  // Create descriptor and pipeline layout.
  if (!cgpu_create_pipeline_descriptors(idevice, ipipeline, irgen_shader, pipelineStageFlags))
  {
    goto cleanup_fail;
  }
  if (!cgpu_create_pipeline_layout(idevice, ipipeline, irgen_shader, pipelineStageFlags))
  {
    goto cleanup_fail;
  }

  // Create pipeline.
  {
    uint32_t groupCount = hitStageAndGroupOffset + desc->hit_group_count;

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

    if (idevice->table.vkCreateRayTracingPipelinesKHR(idevice->logical_device,
                                                      VK_NULL_HANDLE,
                                                      VK_NULL_HANDLE,
                                                      1,
                                                      &rt_pipeline_create_info,
                                                      nullptr,
                                                      &ipipeline->pipeline) != VK_SUCCESS)
    {
      goto cleanup_fail;
    }

    ipipeline->bind_point = VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR;

    // Create the SBT.
    if (!cgpu_create_rt_pipeline_sbt(idevice, ipipeline, groupCount, desc->miss_shader_count, desc->hit_group_count))
    {
      goto cleanup_fail;
    }

    return true;
  }

cleanup_fail:
  idevice->table.vkDestroyPipelineLayout(idevice->logical_device, ipipeline->layout, nullptr);
  idevice->table.vkDestroyDescriptorSetLayout(idevice->logical_device, ipipeline->descriptor_set_layout, nullptr);
  idevice->table.vkDestroyDescriptorPool(idevice->logical_device, ipipeline->descriptor_pool, nullptr);
  iinstance->ipipeline_store.free(p_pipeline->handle);

  CGPU_RETURN_ERROR("failed to create rt pipeline");
}

bool cgpu_destroy_pipeline(CgpuDevice device,
                           CgpuPipeline pipeline)
{
  CgpuIDevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIPipeline* ipipeline;
  if (!cgpu_resolve_pipeline(pipeline, &ipipeline)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  if (ipipeline->bind_point == VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR)
  {
    cgpu_destroy_ibuffer(idevice, &ipipeline->sbt);
  }

  idevice->table.vkDestroyDescriptorPool(
    idevice->logical_device,
    ipipeline->descriptor_pool,
    nullptr
  );
  idevice->table.vkDestroyPipeline(
    idevice->logical_device,
    ipipeline->pipeline,
    nullptr
  );
  idevice->table.vkDestroyPipelineLayout(
    idevice->logical_device,
    ipipeline->layout,
    nullptr
  );
  idevice->table.vkDestroyDescriptorSetLayout(
    idevice->logical_device,
    ipipeline->descriptor_set_layout,
    nullptr
  );

  iinstance->ipipeline_store.free(pipeline.handle);

  return true;
}

static bool cgpu_create_top_or_bottom_as(CgpuDevice device,
                                         VkAccelerationStructureTypeKHR as_type,
                                         VkAccelerationStructureGeometryKHR* as_geom,
                                         uint32_t primitive_count,
                                         CgpuIBuffer* ias_buffer,
                                         VkAccelerationStructureKHR* as)
{
  CgpuIDevice* idevice;
  cgpu_resolve_device(device, &idevice);

  // Get AS size
  VkAccelerationStructureBuildGeometryInfoKHR as_build_geom_info = {};
  as_build_geom_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
  as_build_geom_info.pNext = nullptr;
  as_build_geom_info.type = as_type;
  as_build_geom_info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  as_build_geom_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
  as_build_geom_info.srcAccelerationStructure = VK_NULL_HANDLE;
  as_build_geom_info.dstAccelerationStructure = VK_NULL_HANDLE; // set in second round
  as_build_geom_info.geometryCount = 1;
  as_build_geom_info.pGeometries = as_geom;
  as_build_geom_info.ppGeometries = nullptr;
  as_build_geom_info.scratchData.hostAddress = nullptr;
  as_build_geom_info.scratchData.deviceAddress = 0; // set in second round

  VkAccelerationStructureBuildSizesInfoKHR as_build_sizes_info = {};
  as_build_sizes_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
  as_build_sizes_info.pNext = nullptr;
  as_build_sizes_info.accelerationStructureSize = 0; // output
  as_build_sizes_info.updateScratchSize = 0; // output
  as_build_sizes_info.buildScratchSize = 0; // output

  idevice->table.vkGetAccelerationStructureBuildSizesKHR(idevice->logical_device,
                                                         VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                                         &as_build_geom_info,
                                                         &primitive_count,
                                                         &as_build_sizes_info);

  // Create AS buffer & AS object
  if (!cgpu_create_ibuffer_aligned(idevice,
                                   CGPU_BUFFER_USAGE_FLAG_SHADER_DEVICE_ADDRESS | CGPU_BUFFER_USAGE_FLAG_ACCELERATION_STRUCTURE_STORAGE,
                                   CGPU_MEMORY_PROPERTY_FLAG_DEVICE_LOCAL,
                                   as_build_sizes_info.accelerationStructureSize, 0,
                                   ias_buffer))
  {
    CGPU_RETURN_ERROR("failed to create AS buffer");
  }

  VkAccelerationStructureCreateInfoKHR as_create_info = {};
  as_create_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
  as_create_info.pNext = nullptr;
  as_create_info.createFlags = 0;
  as_create_info.buffer = ias_buffer->buffer;
  as_create_info.offset = 0;
  as_create_info.size = as_build_sizes_info.accelerationStructureSize;
  as_create_info.type = as_type;
  as_create_info.deviceAddress = 0; // used for capture-replay feature

  if (idevice->table.vkCreateAccelerationStructureKHR(idevice->logical_device, &as_create_info, nullptr, as) != VK_SUCCESS)
  {
    cgpu_destroy_ibuffer(idevice, ias_buffer);
    CGPU_RETURN_ERROR("failed to create Vulkan AS object");
  }

  // Set up device-local scratch buffer
  CgpuIBuffer iscratch_buffer;
  if (!cgpu_create_ibuffer_aligned(idevice,
                                   CGPU_BUFFER_USAGE_FLAG_STORAGE_BUFFER | CGPU_BUFFER_USAGE_FLAG_SHADER_DEVICE_ADDRESS,
                                   CGPU_MEMORY_PROPERTY_FLAG_DEVICE_LOCAL,
                                   as_build_sizes_info.buildScratchSize,
                                   idevice->properties.minAccelerationStructureScratchOffsetAlignment,
                                   &iscratch_buffer))
  {
    cgpu_destroy_ibuffer(idevice, ias_buffer);
    idevice->table.vkDestroyAccelerationStructureKHR(idevice->logical_device, *as, nullptr);
    CGPU_RETURN_ERROR("failed to create AS scratch buffer");
  }

  as_build_geom_info.dstAccelerationStructure = *as;
  as_build_geom_info.scratchData.hostAddress = 0;
  as_build_geom_info.scratchData.deviceAddress = cgpu_get_buffer_device_address(idevice, &iscratch_buffer);

  VkAccelerationStructureBuildRangeInfoKHR as_build_range_info;
  as_build_range_info.primitiveCount = primitive_count;
  as_build_range_info.primitiveOffset = 0;
  as_build_range_info.firstVertex = 0;
  as_build_range_info.transformOffset = 0;

  const VkAccelerationStructureBuildRangeInfoKHR* as_build_range_info_ptr = &as_build_range_info;

  CgpuCommandBuffer command_buffer;
  if (!cgpu_create_command_buffer(device, &command_buffer)) {
    cgpu_destroy_ibuffer(idevice, ias_buffer);
    cgpu_destroy_ibuffer(idevice, &iscratch_buffer);
    idevice->table.vkDestroyAccelerationStructureKHR(idevice->logical_device, *as, nullptr);
    CGPU_RETURN_ERROR("failed to create AS build command buffer");
  }

  CgpuICommandBuffer* icommand_buffer;
  cgpu_resolve_command_buffer(command_buffer, &icommand_buffer);

  // Build AS on device
  cgpu_begin_command_buffer(command_buffer);
  idevice->table.vkCmdBuildAccelerationStructuresKHR(icommand_buffer->command_buffer, 1, &as_build_geom_info, &as_build_range_info_ptr);
  cgpu_end_command_buffer(command_buffer);

  CgpuFence fence;
  if (!cgpu_create_fence(device, &fence)) {
    cgpu_destroy_ibuffer(idevice, ias_buffer);
    cgpu_destroy_ibuffer(idevice, &iscratch_buffer);
    idevice->table.vkDestroyAccelerationStructureKHR(idevice->logical_device, *as, nullptr);
    CGPU_RETURN_ERROR("failed to create AS build fence");
  }
  cgpu_reset_fence(device, fence);
  cgpu_submit_command_buffer(device, command_buffer, fence);
  cgpu_wait_for_fence(device, fence);

  // Dispose resources
  cgpu_destroy_fence(device, fence);
  cgpu_destroy_command_buffer(device, command_buffer);
  cgpu_destroy_ibuffer(idevice, &iscratch_buffer);

  return true;
}


bool cgpu_create_blas(CgpuDevice device,
                      uint32_t vertex_count,
                      const CgpuVertex* vertices,
                      uint32_t index_count,
                      const uint32_t* indices,
                      bool isOpaque,
                      CgpuBlas* p_blas)
{
  CgpuIDevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  p_blas->handle = iinstance->iblas_store.allocate();

  CgpuIBlas* iblas;
  if (!cgpu_resolve_blas(*p_blas, &iblas)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  if ((index_count % 3) != 0) {
    CGPU_RETURN_ERROR("BLAS indices do not represent triangles");
  }

  // Create index buffer & copy data into it
  uint64_t index_buffer_size = index_count * sizeof(uint32_t);
  if (!cgpu_create_ibuffer_aligned(idevice,
                                   CGPU_BUFFER_USAGE_FLAG_SHADER_DEVICE_ADDRESS | CGPU_BUFFER_USAGE_FLAG_ACCELERATION_STRUCTURE_BUILD_INPUT,
                                   CGPU_MEMORY_PROPERTY_FLAG_HOST_VISIBLE | CGPU_MEMORY_PROPERTY_FLAG_HOST_COHERENT,
                                   index_buffer_size, 0,
                                   &iblas->indices))
  {
    CGPU_RETURN_ERROR("failed to create BLAS index buffer");
  }

  {
    void* mapped_mem;
    if (vmaMapMemory(idevice->allocator, iblas->indices.allocation, (void**) &mapped_mem) != VK_SUCCESS) {
      cgpu_destroy_ibuffer(idevice, &iblas->indices);
      CGPU_RETURN_ERROR("failed to map buffer memory");
    }
    memcpy(mapped_mem, indices, index_buffer_size);
    vmaUnmapMemory(idevice->allocator, iblas->indices.allocation);
  }

  // Create vertex buffer & copy data into it
  uint64_t vertex_buffer_size = vertex_count * sizeof(CgpuVertex);
  if (!cgpu_create_ibuffer_aligned(idevice,
                                   CGPU_BUFFER_USAGE_FLAG_SHADER_DEVICE_ADDRESS | CGPU_BUFFER_USAGE_FLAG_ACCELERATION_STRUCTURE_BUILD_INPUT,
                                   CGPU_MEMORY_PROPERTY_FLAG_HOST_VISIBLE | CGPU_MEMORY_PROPERTY_FLAG_HOST_COHERENT,
                                   vertex_buffer_size, 0,
                                   &iblas->vertices))
  {
    cgpu_destroy_ibuffer(idevice, &iblas->indices);
    CGPU_RETURN_ERROR("failed to create BLAS vertex buffer");
  }

  {
    void* mapped_mem;
    if (vmaMapMemory(idevice->allocator, iblas->vertices.allocation, (void**)&mapped_mem) != VK_SUCCESS)
    {
      cgpu_destroy_ibuffer(idevice, &iblas->indices);
      cgpu_destroy_ibuffer(idevice, &iblas->vertices);
      CGPU_RETURN_ERROR("failed to map buffer memory");
    }
    memcpy(mapped_mem, vertices, vertex_buffer_size);
    vmaUnmapMemory(idevice->allocator, iblas->vertices.allocation);
  }

  // Create BLAS
  VkAccelerationStructureGeometryTrianglesDataKHR as_triangle_data = {};
  as_triangle_data.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
  as_triangle_data.pNext = nullptr;
  as_triangle_data.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
  as_triangle_data.vertexData.hostAddress = nullptr;
  as_triangle_data.vertexData.deviceAddress = cgpu_get_buffer_device_address(idevice, &iblas->vertices);
  as_triangle_data.vertexStride = sizeof(CgpuVertex);
  as_triangle_data.maxVertex = vertex_count;
  as_triangle_data.indexType = VK_INDEX_TYPE_UINT32;
  as_triangle_data.indexData.hostAddress = nullptr;
  as_triangle_data.indexData.deviceAddress = cgpu_get_buffer_device_address(idevice, &iblas->indices);
  as_triangle_data.transformData.hostAddress = nullptr;
  as_triangle_data.transformData.deviceAddress = 0; // optional

  VkAccelerationStructureGeometryDataKHR as_geom_data = {};
  as_geom_data.triangles = as_triangle_data;

  VkAccelerationStructureGeometryKHR as_geom = {};
  as_geom.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
  as_geom.pNext = nullptr;
  as_geom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
  as_geom.geometry = as_geom_data;
  as_geom.flags = isOpaque ? VK_GEOMETRY_OPAQUE_BIT_KHR : VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR;

  uint32_t triangle_count = index_count / 3;
  if (!cgpu_create_top_or_bottom_as(device, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, &as_geom, triangle_count, &iblas->buffer, &iblas->as))
  {
    cgpu_destroy_ibuffer(idevice, &iblas->indices);
    cgpu_destroy_ibuffer(idevice, &iblas->vertices);
    CGPU_RETURN_ERROR("failed to build BLAS");
  }

  VkAccelerationStructureDeviceAddressInfoKHR as_address_info = {};
  as_address_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
  as_address_info.pNext = nullptr;
  as_address_info.accelerationStructure = iblas->as;
  iblas->address = idevice->table.vkGetAccelerationStructureDeviceAddressKHR(idevice->logical_device, &as_address_info);

  iblas->isOpaque = isOpaque;

  return true;
}

bool cgpu_create_tlas(CgpuDevice device,
                      uint32_t instance_count,
                      const CgpuBlasInstance* instances,
                      CgpuTlas* p_tlas)
{
  CgpuIDevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  p_tlas->handle = iinstance->itlas_store.allocate();

  CgpuITlas* itlas;
  if (!cgpu_resolve_tlas(*p_tlas, &itlas)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  // Create instance buffer & copy into it
  if (!cgpu_create_ibuffer_aligned(idevice,
                                   CGPU_BUFFER_USAGE_FLAG_SHADER_DEVICE_ADDRESS | CGPU_BUFFER_USAGE_FLAG_ACCELERATION_STRUCTURE_BUILD_INPUT,
                                   CGPU_MEMORY_PROPERTY_FLAG_HOST_VISIBLE | CGPU_MEMORY_PROPERTY_FLAG_HOST_COHERENT,
                                   instance_count * sizeof(VkAccelerationStructureInstanceKHR), 0,
                                   &itlas->instances))
  {
    CGPU_RETURN_ERROR("failed to create TLAS instances buffer");
  }

  bool areAllBlasOpaque = true;
  {
    uint8_t* mapped_mem;
    if (vmaMapMemory(idevice->allocator, itlas->instances.allocation, (void**) &mapped_mem) != VK_SUCCESS)
    {
      cgpu_destroy_ibuffer(idevice, &itlas->instances);
      CGPU_RETURN_ERROR("failed to map buffer memory");
    }

    for (uint32_t i = 0; i < instance_count; i++)
    {
      CgpuIBlas* iblas;
      if (!cgpu_resolve_blas(instances[i].as, &iblas)) {
        cgpu_destroy_ibuffer(idevice, &itlas->instances);
        CGPU_RETURN_ERROR_INVALID_HANDLE;
      }

      VkAccelerationStructureInstanceKHR* as_instance = (VkAccelerationStructureInstanceKHR*) &mapped_mem[i * sizeof(VkAccelerationStructureInstanceKHR)];
      memcpy(&as_instance->transform, &instances[i].transform, sizeof(VkTransformMatrixKHR));
      as_instance->instanceCustomIndex = instances[i].faceIndexOffset;
      as_instance->mask = 0xFF;
      as_instance->instanceShaderBindingTableRecordOffset = instances[i].hitGroupIndex;
      as_instance->flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
      as_instance->accelerationStructureReference = iblas->address;

      areAllBlasOpaque &= iblas->isOpaque;
    }

    vmaUnmapMemory(idevice->allocator, itlas->instances.allocation);
  }

  // Create TLAS
  VkAccelerationStructureGeometryKHR as_geom = {};
  as_geom.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
  as_geom.pNext = nullptr;
  as_geom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
  as_geom.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
  as_geom.geometry.instances.pNext = nullptr;
  as_geom.geometry.instances.arrayOfPointers = VK_FALSE;
  as_geom.geometry.instances.data.hostAddress = nullptr;
  as_geom.geometry.instances.data.deviceAddress = cgpu_get_buffer_device_address(idevice, &itlas->instances);
  as_geom.flags = areAllBlasOpaque ? VK_GEOMETRY_OPAQUE_BIT_KHR : 0;

  if (!cgpu_create_top_or_bottom_as(device, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, &as_geom, instance_count, &itlas->buffer, &itlas->as))
  {
    cgpu_destroy_ibuffer(idevice, &itlas->instances);
    CGPU_RETURN_ERROR("failed to build TLAS");
  }

  return true;
}

bool cgpu_destroy_blas(CgpuDevice device, CgpuBlas blas)
{
  CgpuIDevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIBlas* iblas;
  if (!cgpu_resolve_blas(blas, &iblas)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  idevice->table.vkDestroyAccelerationStructureKHR(idevice->logical_device, iblas->as, nullptr);
  cgpu_destroy_ibuffer(idevice, &iblas->buffer);
  cgpu_destroy_ibuffer(idevice, &iblas->indices);
  cgpu_destroy_ibuffer(idevice, &iblas->vertices);

  iinstance->iblas_store.free(blas.handle);
  return true;
}

bool cgpu_destroy_tlas(CgpuDevice device, CgpuTlas tlas)
{
  CgpuIDevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuITlas* itlas;
  if (!cgpu_resolve_tlas(tlas, &itlas)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  idevice->table.vkDestroyAccelerationStructureKHR(idevice->logical_device, itlas->as, nullptr);
  cgpu_destroy_ibuffer(idevice, &itlas->instances);
  cgpu_destroy_ibuffer(idevice, &itlas->buffer);

  iinstance->itlas_store.free(tlas.handle);
  return true;
}

bool cgpu_create_command_buffer(CgpuDevice device,
                                CgpuCommandBuffer* p_command_buffer)
{
  CgpuIDevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  p_command_buffer->handle = iinstance->icommand_buffer_store.allocate();

  CgpuICommandBuffer* icommand_buffer;
  if (!cgpu_resolve_command_buffer(*p_command_buffer, &icommand_buffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  icommand_buffer->device.handle = device.handle;

  VkCommandBufferAllocateInfo cmdbuf_alloc_info;
  cmdbuf_alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cmdbuf_alloc_info.pNext = nullptr;
  cmdbuf_alloc_info.commandPool = idevice->command_pool;
  cmdbuf_alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cmdbuf_alloc_info.commandBufferCount = 1;

  VkResult result = idevice->table.vkAllocateCommandBuffers(
    idevice->logical_device,
    &cmdbuf_alloc_info,
    &icommand_buffer->command_buffer
  );
  if (result != VK_SUCCESS) {
    iinstance->icommand_buffer_store.free(p_command_buffer->handle);
    CGPU_RETURN_ERROR("failed to allocate command buffer");
  }

  return true;
}

bool cgpu_destroy_command_buffer(CgpuDevice device,
                                 CgpuCommandBuffer command_buffer)
{
  CgpuIDevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuICommandBuffer* icommand_buffer;
  if (!cgpu_resolve_command_buffer(command_buffer, &icommand_buffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  idevice->table.vkFreeCommandBuffers(
    idevice->logical_device,
    idevice->command_pool,
    1,
    &icommand_buffer->command_buffer
  );

  iinstance->icommand_buffer_store.free(command_buffer.handle);
  return true;
}

bool cgpu_begin_command_buffer(CgpuCommandBuffer command_buffer)
{
  CgpuICommandBuffer* icommand_buffer;
  if (!cgpu_resolve_command_buffer(command_buffer, &icommand_buffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIDevice* idevice;
  if (!cgpu_resolve_device(icommand_buffer->device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  VkCommandBufferBeginInfo command_buffer_begin_info;
  command_buffer_begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  command_buffer_begin_info.pNext = nullptr;
  command_buffer_begin_info.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
  command_buffer_begin_info.pInheritanceInfo = nullptr;

  VkResult result = idevice->table.vkBeginCommandBuffer(
    icommand_buffer->command_buffer,
    &command_buffer_begin_info
  );

  if (result != VK_SUCCESS) {
    CGPU_RETURN_ERROR("failed to begin command buffer");
  }
  return true;
}

bool cgpu_cmd_bind_pipeline(CgpuCommandBuffer command_buffer,
                            CgpuPipeline pipeline)
{
  CgpuICommandBuffer* icommand_buffer;
  if (!cgpu_resolve_command_buffer(command_buffer, &icommand_buffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIDevice* idevice;
  if (!cgpu_resolve_device(icommand_buffer->device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIPipeline* ipipeline;
  if (!cgpu_resolve_pipeline(pipeline, &ipipeline)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  idevice->table.vkCmdBindPipeline(
    icommand_buffer->command_buffer,
    ipipeline->bind_point,
    ipipeline->pipeline
  );
  idevice->table.vkCmdBindDescriptorSets(
    icommand_buffer->command_buffer,
    ipipeline->bind_point,
    ipipeline->layout,
    0,
    1,
    &ipipeline->descriptor_set,
    0,
    0
  );

  return true;
}

bool cgpu_cmd_transition_shader_image_layouts(CgpuCommandBuffer command_buffer,
                                              CgpuShader shader,
                                              uint32_t image_count,
                                              const CgpuImageBinding* p_images)
{
  CgpuIShader* ishader;
  if (!cgpu_resolve_shader(shader, &ishader)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuICommandBuffer* icommand_buffer;
  if (!cgpu_resolve_command_buffer(command_buffer, &icommand_buffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIDevice* idevice;
  if (!cgpu_resolve_device(icommand_buffer->device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  GbSmallVector<VkImageMemoryBarrier, 64> barriers;

  /* FIXME: this has quadratic complexity */
  const CgpuShaderReflection* reflection = &ishader->reflection;
  for (uint32_t i = 0; i < reflection->binding_count; i++)
  {
    const CgpuShaderReflectionBinding* binding = &reflection->bindings[i];

    VkImageLayout new_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (binding->descriptor_type == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE)
    {
      new_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }
    else if (binding->descriptor_type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
    {
      new_layout = VK_IMAGE_LAYOUT_GENERAL;
    }
    else
    {
      /* Not an image. */
      continue;
    }

    for (uint32_t j = 0; j < binding->count; j++)
    {
      /* Image layout needs transitioning. */
      const CgpuImageBinding* image_binding = nullptr;
      for (uint32_t k = 0; k < image_count; k++)
      {
        if (p_images[k].binding == binding->binding && p_images[k].index == j)
        {
          image_binding = &p_images[k];
          break;
        }
      }
      if (!image_binding)
      {
        CGPU_RETURN_ERROR("descriptor set binding mismatch");
      }

      CgpuIImage* iimage;
      if (!cgpu_resolve_image(image_binding->image, &iimage)) {
        CGPU_RETURN_ERROR_INVALID_HANDLE;
      }

      VkImageLayout old_layout = iimage->layout;
      if (new_layout == old_layout)
      {
        continue;
      }

      VkAccessFlags access_mask = 0;
      if (binding->read_access) {
        access_mask = VK_ACCESS_SHADER_READ_BIT;
      }
      if (binding->write_access) {
        access_mask = VK_ACCESS_SHADER_WRITE_BIT;
      }

      VkImageMemoryBarrier barrier = {};
      barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
      barrier.pNext = nullptr;
      barrier.srcAccessMask = iimage->access_mask;
      barrier.dstAccessMask = access_mask;
      barrier.oldLayout = old_layout;
      barrier.newLayout = new_layout;
      barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.image = iimage->image;
      barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      barrier.subresourceRange.baseMipLevel = 0;
      barrier.subresourceRange.levelCount = 1;
      barrier.subresourceRange.baseArrayLayer = 0;
      barrier.subresourceRange.layerCount = 1;
      barriers.push_back(barrier);

      iimage->access_mask = access_mask;
      iimage->layout = new_layout;
    }
  }

  if (barriers.size() > 0)
  {
    idevice->table.vkCmdPipelineBarrier(
      icommand_buffer->command_buffer,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
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

bool cgpu_cmd_update_bindings(CgpuCommandBuffer command_buffer,
                              CgpuPipeline pipeline,
                              const CgpuBindings* bindings
)
{
  CgpuICommandBuffer* icommand_buffer;
  if (!cgpu_resolve_command_buffer(command_buffer, &icommand_buffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIDevice* idevice;
  if (!cgpu_resolve_device(icommand_buffer->device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIPipeline* ipipeline;
  if (!cgpu_resolve_pipeline(pipeline, &ipipeline)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  GbSmallVector<VkDescriptorBufferInfo, 64> buffer_infos;
  GbSmallVector<VkDescriptorImageInfo, 128> image_infos;
  GbSmallVector<VkWriteDescriptorSetAccelerationStructureKHR, 1> as_infos;

  buffer_infos.reserve(bindings->buffer_count);
  image_infos.reserve(bindings->image_count + bindings->sampler_count);
  as_infos.reserve(bindings->tlas_count);

  GbSmallVector<VkWriteDescriptorSet, 128> write_descriptor_sets;

  /* FIXME: this has a rather high complexity */
  for (uint32_t i = 0; i < ipipeline->descriptor_set_layout_bindings.size(); i++)
  {
    const VkDescriptorSetLayoutBinding* layout_binding = &ipipeline->descriptor_set_layout_bindings[i];

    VkWriteDescriptorSet write_descriptor_set = {};
    write_descriptor_set.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write_descriptor_set.pNext = nullptr;
    write_descriptor_set.dstSet = ipipeline->descriptor_set;
    write_descriptor_set.dstBinding = layout_binding->binding;
    write_descriptor_set.dstArrayElement = 0;
    write_descriptor_set.descriptorCount = layout_binding->descriptorCount;
    write_descriptor_set.descriptorType = layout_binding->descriptorType;
    write_descriptor_set.pTexelBufferView = nullptr;
    write_descriptor_set.pBufferInfo = nullptr;
    write_descriptor_set.pImageInfo = nullptr;

    for (uint32_t j = 0; j < layout_binding->descriptorCount; j++)
    {
      bool slotHandled = false;

      if (layout_binding->descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
      {
        for (uint32_t k = 0; k < bindings->buffer_count; ++k)
        {
          const CgpuBufferBinding* buffer_binding = &bindings->p_buffers[k];

          if (buffer_binding->binding != layout_binding->binding || buffer_binding->index != j)
          {
            continue;
          }

          CgpuIBuffer* ibuffer;
          CgpuBuffer buffer = buffer_binding->buffer;
          if (!cgpu_resolve_buffer(buffer, &ibuffer)) {
            CGPU_RETURN_ERROR_INVALID_HANDLE;
          }

          if ((buffer_binding->offset % idevice->properties.minStorageBufferOffsetAlignment) != 0) {
            CGPU_RETURN_ERROR("buffer binding offset not aligned");
          }

          VkDescriptorBufferInfo buffer_info = {};
          buffer_info.buffer = ibuffer->buffer;
          buffer_info.offset = buffer_binding->offset;
          buffer_info.range = (buffer_binding->size == CGPU_WHOLE_SIZE) ? (ibuffer->size - buffer_binding->offset) : buffer_binding->size;
          buffer_infos.push_back(buffer_info);

          if (j == 0)
          {
            write_descriptor_set.pBufferInfo = &buffer_infos.back();
          }

          slotHandled = true;
          break;
        }
      }
      else if (layout_binding->descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE ||
               layout_binding->descriptorType == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE)
      {
        for (uint32_t k = 0; k < bindings->image_count; k++)
        {
          const CgpuImageBinding* image_binding = &bindings->p_images[k];

          if (image_binding->binding != layout_binding->binding || image_binding->index != j)
          {
            continue;
          }

          CgpuIImage* iimage;
          CgpuImage image = image_binding->image;
          if (!cgpu_resolve_image(image, &iimage)) {
            CGPU_RETURN_ERROR_INVALID_HANDLE;
          }

          VkDescriptorImageInfo image_info = {};
          image_info.sampler = VK_NULL_HANDLE;
          image_info.imageView = iimage->image_view;
          image_info.imageLayout = iimage->layout;
          image_infos.push_back(image_info);

          if (j == 0)
          {
            write_descriptor_set.pImageInfo = &image_infos.back();
          }

          slotHandled = true;
          break;
        }
      }
      else if (layout_binding->descriptorType == VK_DESCRIPTOR_TYPE_SAMPLER)
      {
        for (uint32_t k = 0; k < bindings->sampler_count; k++)
        {
          const CgpuSamplerBinding* sampler_binding = &bindings->p_samplers[k];

          if (sampler_binding->binding != layout_binding->binding || sampler_binding->index != j)
          {
            continue;
          }

          CgpuISampler* isampler;
          CgpuSampler sampler = sampler_binding->sampler;
          if (!cgpu_resolve_sampler(sampler, &isampler)) {
            CGPU_RETURN_ERROR_INVALID_HANDLE;
          }

          VkDescriptorImageInfo image_info = {};
          image_info.sampler = isampler->sampler;
          image_info.imageView = VK_NULL_HANDLE;
          image_info.imageLayout = VK_IMAGE_LAYOUT_UNDEFINED;
          image_infos.push_back(image_info);

          if (j == 0)
          {
            write_descriptor_set.pImageInfo = &image_infos.back();
          }

          slotHandled = true;
          break;
        }
      }
      else if (layout_binding->descriptorType == VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR)
      {
        for (uint32_t k = 0; k < bindings->tlas_count; ++k)
        {
          const CgpuTlasBinding* as_binding = &bindings->p_tlases[k];

          if (as_binding->binding != layout_binding->binding || as_binding->index != j)
          {
            continue;
          }

          CgpuITlas* itlas;
          CgpuTlas tlas = as_binding->as;
          if (!cgpu_resolve_tlas(tlas, &itlas)) {
            CGPU_RETURN_ERROR_INVALID_HANDLE;
          }

          VkWriteDescriptorSetAccelerationStructureKHR as_info = {};
          as_info.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
          as_info.pNext = nullptr;
          as_info.accelerationStructureCount = 1;
          as_info.pAccelerationStructures = &itlas->as;
          as_infos.push_back(as_info);

          if (j == 0)
          {
            write_descriptor_set.pNext = &as_infos.back();
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

    write_descriptor_sets.push_back(write_descriptor_set);
  }

  idevice->table.vkUpdateDescriptorSets(
    idevice->logical_device,
    write_descriptor_sets.size(),
    write_descriptor_sets.data(),
    0,
    nullptr
  );

  return true;
}

bool cgpu_cmd_copy_buffer(CgpuCommandBuffer command_buffer,
                          CgpuBuffer source_buffer,
                          uint64_t source_offset,
                          CgpuBuffer destination_buffer,
                          uint64_t destination_offset,
                          uint64_t size)
{
  CgpuICommandBuffer* icommand_buffer;
  if (!cgpu_resolve_command_buffer(command_buffer, &icommand_buffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIDevice* idevice;
  if (!cgpu_resolve_device(icommand_buffer->device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIBuffer* isource_buffer;
  if (!cgpu_resolve_buffer(source_buffer, &isource_buffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIBuffer* idestination_buffer;
  if (!cgpu_resolve_buffer(destination_buffer, &idestination_buffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  VkBufferCopy region;
  region.srcOffset = source_offset;
  region.dstOffset = destination_offset;
  region.size = (size == CGPU_WHOLE_SIZE) ? isource_buffer->size : size;

  idevice->table.vkCmdCopyBuffer(
    icommand_buffer->command_buffer,
    isource_buffer->buffer,
    idestination_buffer->buffer,
    1,
    &region
  );

  return true;
}

bool cgpu_cmd_copy_buffer_to_image(CgpuCommandBuffer command_buffer,
                                   CgpuBuffer buffer,
                                   uint64_t buffer_offset,
                                   CgpuImage image)
{
  CgpuICommandBuffer* icommand_buffer;
  if (!cgpu_resolve_command_buffer(command_buffer, &icommand_buffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIDevice* idevice;
  if (!cgpu_resolve_device(icommand_buffer->device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIBuffer* ibuffer;
  if (!cgpu_resolve_buffer(buffer, &ibuffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIImage* iimage;
  if (!cgpu_resolve_image(image, &iimage)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  if (iimage->layout != VK_IMAGE_LAYOUT_GENERAL)
  {
    VkAccessFlags access_mask = iimage->access_mask | VK_ACCESS_MEMORY_WRITE_BIT;
    VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL;

    VkImageMemoryBarrier barrier;
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.pNext = nullptr;
    barrier.srcAccessMask = iimage->access_mask;
    barrier.dstAccessMask = access_mask;
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
      icommand_buffer->command_buffer,
      // FIXME: batch this barrier and use correct pipeline flag bits
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      0,
      0,
      nullptr,
      0,
      nullptr,
      1,
      &barrier
    );

    iimage->layout = layout;
    iimage->access_mask = access_mask;
  }

  VkImageSubresourceLayers layers;
  layers.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  layers.mipLevel = 0;
  layers.baseArrayLayer = 0;
  layers.layerCount = 1;

  VkOffset3D offset;
  offset.x = 0;
  offset.y = 0;
  offset.z = 0;

  VkExtent3D extent;
  extent.width = iimage->width;
  extent.height = iimage->height;
  extent.depth = iimage->depth;

  VkBufferImageCopy region;
  region.bufferOffset = buffer_offset;
  region.bufferRowLength = 0;
  region.bufferImageHeight = 0;
  region.imageSubresource = layers;
  region.imageOffset = offset;
  region.imageExtent = extent;

  idevice->table.vkCmdCopyBufferToImage(
    icommand_buffer->command_buffer,
    ibuffer->buffer,
    iimage->image,
    iimage->layout,
    1,
    &region
  );

  return true;
}

bool cgpu_cmd_push_constants(CgpuCommandBuffer command_buffer,
                             CgpuPipeline pipeline,
                             CgpuShaderStageFlags stage_flags,
                             uint32_t size,
                             const void* p_data)
{
  CgpuICommandBuffer* icommand_buffer;
  if (!cgpu_resolve_command_buffer(command_buffer, &icommand_buffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIDevice* idevice;
  if (!cgpu_resolve_device(icommand_buffer->device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIPipeline* ipipeline;
  if (!cgpu_resolve_pipeline(pipeline, &ipipeline)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  idevice->table.vkCmdPushConstants(
    icommand_buffer->command_buffer,
    ipipeline->layout,
    (VkShaderStageFlags) stage_flags,
    0,
    size,
    p_data
  );
  return true;
}

bool cgpu_cmd_dispatch(CgpuCommandBuffer command_buffer,
                       uint32_t dim_x,
                       uint32_t dim_y,
                       uint32_t dim_z)
{
  CgpuICommandBuffer* icommand_buffer;
  if (!cgpu_resolve_command_buffer(command_buffer, &icommand_buffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIDevice* idevice;
  if (!cgpu_resolve_device(icommand_buffer->device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  idevice->table.vkCmdDispatch(
    icommand_buffer->command_buffer,
    dim_x,
    dim_y,
    dim_z
  );
  return true;
}

bool cgpu_cmd_pipeline_barrier(CgpuCommandBuffer command_buffer,
                               uint32_t barrier_count,
                               const CgpuMemoryBarrier* p_barriers,
                               uint32_t buffer_barrier_count,
                               const CgpuBufferMemoryBarrier* p_buffer_barriers,
                               uint32_t image_barrier_count,
                               const CgpuImageMemoryBarrier* p_image_barriers)
{
  CgpuICommandBuffer* icommand_buffer;
  if (!cgpu_resolve_command_buffer(command_buffer, &icommand_buffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIDevice* idevice;
  if (!cgpu_resolve_device(icommand_buffer->device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  GbSmallVector<VkMemoryBarrier, 128> vk_memory_barriers;

  for (uint32_t i = 0; i < barrier_count; ++i)
  {
    const CgpuMemoryBarrier* b_cgpu = &p_barriers[i];

    VkMemoryBarrier b_vk = {};
    b_vk.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    b_vk.pNext = nullptr;
    b_vk.srcAccessMask = (VkAccessFlags) b_cgpu->src_access_flags;
    b_vk.dstAccessMask = (VkAccessFlags) b_cgpu->dst_access_flags;
    vk_memory_barriers.push_back(b_vk);
  }

  GbSmallVector<VkBufferMemoryBarrier, 32> vk_buffer_memory_barriers;
  GbSmallVector<VkImageMemoryBarrier, 128> vk_image_memory_barriers;

  for (uint32_t i = 0; i < buffer_barrier_count; ++i)
  {
    const CgpuBufferMemoryBarrier* b_cgpu = &p_buffer_barriers[i];

    CgpuIBuffer* ibuffer;
    if (!cgpu_resolve_buffer(b_cgpu->buffer, &ibuffer)) {
      CGPU_RETURN_ERROR_INVALID_HANDLE;
    }

    VkBufferMemoryBarrier b_vk = {};
    b_vk.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    b_vk.pNext = nullptr;
    b_vk.srcAccessMask = (VkAccessFlags) b_cgpu->src_access_flags;
    b_vk.dstAccessMask = (VkAccessFlags) b_cgpu->dst_access_flags;
    b_vk.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b_vk.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b_vk.buffer = ibuffer->buffer;
    b_vk.offset = b_cgpu->offset;
    b_vk.size = (b_cgpu->size == CGPU_WHOLE_SIZE) ? VK_WHOLE_SIZE : b_cgpu->size;
    vk_buffer_memory_barriers.push_back(b_vk);
  }

  for (uint32_t i = 0; i < image_barrier_count; ++i)
  {
    const CgpuImageMemoryBarrier* b_cgpu = &p_image_barriers[i];

    CgpuIImage* iimage;
    if (!cgpu_resolve_image(b_cgpu->image, &iimage)) {
      CGPU_RETURN_ERROR_INVALID_HANDLE;
    }

    VkAccessFlags access_mask = (VkAccessFlags) b_cgpu->access_mask;

    VkImageMemoryBarrier b_vk = {};
    b_vk.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b_vk.pNext = nullptr;
    b_vk.srcAccessMask = iimage->access_mask;
    b_vk.dstAccessMask = access_mask;
    b_vk.oldLayout = iimage->layout;
    b_vk.newLayout = iimage->layout;
    b_vk.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b_vk.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b_vk.image = iimage->image;
    b_vk.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    b_vk.subresourceRange.baseMipLevel = 0;
    b_vk.subresourceRange.levelCount = 1;
    b_vk.subresourceRange.baseArrayLayer = 0;
    b_vk.subresourceRange.layerCount = 1;
    vk_image_memory_barriers.push_back(b_vk);

    iimage->access_mask = access_mask;
  }

  idevice->table.vkCmdPipelineBarrier(
    icommand_buffer->command_buffer,
    // FIXME: use correct pipeline flag bits
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
    0,
    vk_memory_barriers.size(),
    vk_memory_barriers.data(),
    vk_buffer_memory_barriers.size(),
    vk_buffer_memory_barriers.data(),
    vk_image_memory_barriers.size(),
    vk_image_memory_barriers.data()
  );

  return true;
}

bool cgpu_cmd_reset_timestamps(CgpuCommandBuffer command_buffer,
                               uint32_t offset,
                               uint32_t count)
{
  CgpuICommandBuffer* icommand_buffer;
  if (!cgpu_resolve_command_buffer(command_buffer, &icommand_buffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIDevice* idevice;
  if (!cgpu_resolve_device(icommand_buffer->device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  idevice->table.vkCmdResetQueryPool(
    icommand_buffer->command_buffer,
    idevice->timestamp_pool,
    offset,
    count
  );

  return true;
}

bool cgpu_cmd_write_timestamp(CgpuCommandBuffer command_buffer,
                              uint32_t timestamp_index)
{
  CgpuICommandBuffer* icommand_buffer;
  if (!cgpu_resolve_command_buffer(command_buffer, &icommand_buffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIDevice* idevice;
  if (!cgpu_resolve_device(icommand_buffer->device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  idevice->table.vkCmdWriteTimestamp(
    icommand_buffer->command_buffer,
    // FIXME: use correct pipeline flag bits
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    idevice->timestamp_pool,
    timestamp_index
  );

  return true;
}

bool cgpu_cmd_copy_timestamps(CgpuCommandBuffer command_buffer,
                              CgpuBuffer buffer,
                              uint32_t offset,
                              uint32_t count,
                              bool wait_until_available)
{
  uint32_t last_index = offset + count;
  if (last_index >= CGPU_MAX_TIMESTAMP_QUERIES) {
    CGPU_RETURN_ERROR_HARDCODED_LIMIT_REACHED;
  }

  CgpuICommandBuffer* icommand_buffer;
  if (!cgpu_resolve_command_buffer(command_buffer, &icommand_buffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIDevice* idevice;
  if (!cgpu_resolve_device(icommand_buffer->device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIBuffer* ibuffer;
  if (!cgpu_resolve_buffer(buffer, &ibuffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  VkQueryResultFlags wait_flag = wait_until_available ? VK_QUERY_RESULT_WAIT_BIT : VK_QUERY_RESULT_WITH_AVAILABILITY_BIT;

  idevice->table.vkCmdCopyQueryPoolResults(
    icommand_buffer->command_buffer,
    idevice->timestamp_pool,
    offset,
    count,
    ibuffer->buffer,
    0,
    sizeof(uint64_t),
    VK_QUERY_RESULT_64_BIT | wait_flag
  );

  return true;
}

bool cgpu_cmd_trace_rays(CgpuCommandBuffer command_buffer, CgpuPipeline rt_pipeline, uint32_t width, uint32_t height)
{
  CgpuICommandBuffer* icommand_buffer;
  if (!cgpu_resolve_command_buffer(command_buffer, &icommand_buffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIDevice* idevice;
  if (!cgpu_resolve_device(icommand_buffer->device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIPipeline* ipipeline;
  if (!cgpu_resolve_pipeline(rt_pipeline, &ipipeline)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  VkStridedDeviceAddressRegionKHR callableSBT = {};
  idevice->table.vkCmdTraceRaysKHR(icommand_buffer->command_buffer,
                                   &ipipeline->sbtRgen,
                                   &ipipeline->sbtMiss,
                                   &ipipeline->sbtHit,
                                   &callableSBT,
                                   width, height, 1);
  return true;
}

bool cgpu_end_command_buffer(CgpuCommandBuffer command_buffer)
{
  CgpuICommandBuffer* icommand_buffer;
  if (!cgpu_resolve_command_buffer(command_buffer, &icommand_buffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIDevice* idevice;
  if (!cgpu_resolve_device(icommand_buffer->device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  idevice->table.vkEndCommandBuffer(icommand_buffer->command_buffer);
  return true;
}

bool cgpu_create_fence(CgpuDevice device,
                       CgpuFence* p_fence)
{
  CgpuIDevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  p_fence->handle = iinstance->ifence_store.allocate();

  CgpuIFence* ifence;
  if (!cgpu_resolve_fence(*p_fence, &ifence)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  VkFenceCreateInfo fence_create_info;
  fence_create_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fence_create_info.pNext = nullptr;
  fence_create_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

  VkResult result = idevice->table.vkCreateFence(
    idevice->logical_device,
    &fence_create_info,
    nullptr,
    &ifence->fence
  );

  if (result != VK_SUCCESS) {
    iinstance->ifence_store.free(p_fence->handle);
    CGPU_RETURN_ERROR("failed to create fence");
  }
  return true;
}

bool cgpu_destroy_fence(CgpuDevice device,
                        CgpuFence fence)
{
  CgpuIDevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIFence* ifence;
  if (!cgpu_resolve_fence(fence, &ifence)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  idevice->table.vkDestroyFence(
    idevice->logical_device,
    ifence->fence,
    nullptr
  );
  iinstance->ifence_store.free(fence.handle);
  return true;
}

bool cgpu_reset_fence(CgpuDevice device,
                      CgpuFence fence)
{
  CgpuIDevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIFence* ifence;
  if (!cgpu_resolve_fence(fence, &ifence)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  VkResult result = idevice->table.vkResetFences(
    idevice->logical_device,
    1,
    &ifence->fence
  );
  if (result != VK_SUCCESS) {
    CGPU_RETURN_ERROR("failed to reset fence");
  }
  return true;
}

bool cgpu_wait_for_fence(CgpuDevice device, CgpuFence fence)
{
  CgpuIDevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIFence* ifence;
  if (!cgpu_resolve_fence(fence, &ifence)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  VkResult result = idevice->table.vkWaitForFences(
    idevice->logical_device,
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

bool cgpu_submit_command_buffer(CgpuDevice device,
                                CgpuCommandBuffer command_buffer,
                                CgpuFence fence)
{
  CgpuIDevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuICommandBuffer* icommand_buffer;
  if (!cgpu_resolve_command_buffer(command_buffer, &icommand_buffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIFence* ifence;
  if (!cgpu_resolve_fence(fence, &ifence)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  VkSubmitInfo submit_info;
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.pNext = nullptr;
  submit_info.waitSemaphoreCount = 0;
  submit_info.pWaitSemaphores = nullptr;
  submit_info.pWaitDstStageMask = nullptr;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &icommand_buffer->command_buffer;
  submit_info.signalSemaphoreCount = 0;
  submit_info.pSignalSemaphores = nullptr;

  VkResult result = idevice->table.vkQueueSubmit(
    idevice->compute_queue,
    1,
    &submit_info,
    ifence->fence
  );

  if (result != VK_SUCCESS) {
    CGPU_RETURN_ERROR("failed to submit command buffer");
  }
  return true;
}

bool cgpu_flush_mapped_memory(CgpuDevice device,
                              CgpuBuffer buffer,
                              uint64_t offset,
                              uint64_t size)
{
  CgpuIDevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIBuffer* ibuffer;
  if (!cgpu_resolve_buffer(buffer, &ibuffer)) {
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

bool cgpu_invalidate_mapped_memory(CgpuDevice device,
                                   CgpuBuffer buffer,
                                   uint64_t offset,
                                   uint64_t size)
{
  CgpuIDevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  CgpuIBuffer* ibuffer;
  if (!cgpu_resolve_buffer(buffer, &ibuffer)) {
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

bool cgpu_get_physical_device_features(CgpuDevice device,
                                       CgpuPhysicalDeviceFeatures* p_features)
{
  CgpuIDevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  memcpy(p_features, &idevice->features, sizeof(CgpuPhysicalDeviceFeatures));
  return true;
}

bool cgpu_get_physical_device_properties(CgpuDevice device,
                                         CgpuPhysicalDeviceProperties* p_properties)
{
  CgpuIDevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  memcpy(p_properties, &idevice->properties, sizeof(CgpuPhysicalDeviceProperties));
  return true;
}
