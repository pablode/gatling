/*
 * Copyright (C) 2019-2022 Pablo Delgado Krämer
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

#include "cgpu.h"
#include "resource_store.h"
#include "shader_reflection.h"

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <volk.h>

#include <vma.h>

#define CGPU_MIN_VK_API_VERSION VK_API_VERSION_1_1

/* Array and pool allocation limits. */

#define CGPU_MAX_PHYSICAL_DEVICES 8
#define CGPU_MAX_DEVICE_EXTENSIONS 1024
#define CGPU_MAX_QUEUE_FAMILIES 64
#define CGPU_MAX_DESCRIPTOR_SET_LAYOUT_BINDINGS 128
#define CGPU_MAX_DESCRIPTOR_BUFFER_INFOS 64
#define CGPU_MAX_DESCRIPTOR_IMAGE_INFOS 2048
#define CGPU_MAX_DESCRIPTOR_AS_INFOS 1
#define CGPU_MAX_WRITE_DESCRIPTOR_SETS 128
#define CGPU_MAX_BUFFER_MEMORY_BARRIERS 64
#define CGPU_MAX_IMAGE_MEMORY_BARRIERS 2048
#define CGPU_MAX_MEMORY_BARRIERS 128
#define CGPU_MAX_RAY_RECURSION_DEPTH 31

/* Internal structures. */

typedef struct cgpu_iinstance {
  VkInstance instance;
} cgpu_iinstance;

typedef struct cgpu_idevice {
  VkDevice                        logical_device;
  VkPhysicalDevice                physical_device;
  VkQueue                         compute_queue;
  VkCommandPool                   command_pool;
  VkQueryPool                     timestamp_pool;
  struct VolkDeviceTable          table;
  cgpu_physical_device_features   features;
  cgpu_physical_device_properties properties;
  VmaAllocator                    allocator;
} cgpu_idevice;

typedef struct cgpu_ibuffer {
  VkBuffer       buffer;
  uint64_t       size;
  VmaAllocation  allocation;
} cgpu_ibuffer;

typedef struct cgpu_iimage {
  VkImage       image;
  VkImageView   image_view;
  VmaAllocation allocation;
  uint64_t      size;
  uint32_t      width;
  uint32_t      height;
  uint32_t      depth;
  VkImageLayout layout;
  VkAccessFlags access_mask;
} cgpu_iimage;

typedef struct cgpu_ipipeline {
  VkPipeline                   pipeline;
  VkPipelineLayout             layout;
  VkDescriptorPool             descriptor_pool;
  VkDescriptorSet              descriptor_set;
  VkDescriptorSetLayout        descriptor_set_layout;
  VkDescriptorSetLayoutBinding descriptor_set_layout_bindings[CGPU_MAX_DESCRIPTOR_SET_LAYOUT_BINDINGS];
  uint32_t                     descriptor_set_layout_binding_count;
  cgpu_shader                  shader;
  VkPipelineBindPoint          bind_point;
} cgpu_ipipeline;

typedef struct cgpu_ishader {
  VkShaderModule module;
  cgpu_shader_reflection reflection;
  VkShaderStageFlagBits stage_flags;
} cgpu_ishader;

typedef struct cgpu_ifence {
  VkFence fence;
} cgpu_ifence;

typedef struct cgpu_icommand_buffer {
  VkCommandBuffer command_buffer;
  cgpu_device     device;
} cgpu_icommand_buffer;

typedef struct cgpu_iacceleration_structure {
  VkAccelerationStructureKHR blas;
  VkAccelerationStructureKHR tlas;
  uint64_t blas_address;
  cgpu_buffer blas_buffer;
  cgpu_buffer tlas_buffer;
  cgpu_buffer index_buffer; // blas
  cgpu_buffer vertex_buffer; // blas
  cgpu_buffer instances_buffer; // tlas
} cgpu_iacceleration_structure;

typedef struct cgpu_isampler {
  VkSampler sampler;
} cgpu_isampler;

/* Handle and structure storage. */

static cgpu_iinstance iinstance;
static resource_store idevice_store;
static resource_store ibuffer_store;
static resource_store iimage_store;
static resource_store ishader_store;
static resource_store ipipeline_store;
static resource_store ifence_store;
static resource_store icommand_buffer_store;
static resource_store isampler_store;
static resource_store iacceleration_structure_store;

/* Helper functions. */

#define CGPU_RETURN_ERROR(msg)                                        \
  do {                                                                \
    fprintf(stderr, "error in %s:%d: %s\n", __FILE__, __LINE__, msg); \
    return false;                                                     \
  } while (false)

#define CGPU_RETURN_ERROR_INVALID_HANDLE                              \
  CGPU_RETURN_ERROR("invalid resource handle");

#define CGPU_RETURN_ERROR_HARDCODED_LIMIT_REACHED                     \
  CGPU_RETURN_ERROR("hardcoded limit reached");

#define CGPU_RESOLVE_HANDLE(RESOURCE_NAME, HANDLE_TYPE, IRESOURCE_TYPE, RESOURCE_STORE)   \
  CGPU_INLINE static bool cgpu_resolve_##RESOURCE_NAME(                                   \
    HANDLE_TYPE handle,                                                                   \
    IRESOURCE_TYPE** idata)                                                               \
  {                                                                                       \
    return resource_store_get(&RESOURCE_STORE, handle.handle, (void**) idata);            \
  }

CGPU_RESOLVE_HANDLE(                device,                 cgpu_device,                 cgpu_idevice,                 idevice_store)
CGPU_RESOLVE_HANDLE(                buffer,                 cgpu_buffer,                 cgpu_ibuffer,                 ibuffer_store)
CGPU_RESOLVE_HANDLE(                 image,                  cgpu_image,                  cgpu_iimage,                  iimage_store)
CGPU_RESOLVE_HANDLE(                shader,                 cgpu_shader,                 cgpu_ishader,                 ishader_store)
CGPU_RESOLVE_HANDLE(              pipeline,               cgpu_pipeline,               cgpu_ipipeline,               ipipeline_store)
CGPU_RESOLVE_HANDLE(                 fence,                  cgpu_fence,                  cgpu_ifence,                  ifence_store)
CGPU_RESOLVE_HANDLE(        command_buffer,         cgpu_command_buffer,         cgpu_icommand_buffer,         icommand_buffer_store)
CGPU_RESOLVE_HANDLE(               sampler,                cgpu_sampler,                cgpu_isampler,                isampler_store)
CGPU_RESOLVE_HANDLE(acceleration_structure, cgpu_acceleration_structure, cgpu_iacceleration_structure, iacceleration_structure_store)

static cgpu_physical_device_features cgpu_translate_physical_device_features(const VkPhysicalDeviceFeatures* vk_features)
{
  cgpu_physical_device_features features = {0};
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

static cgpu_physical_device_properties cgpu_translate_physical_device_properties(const VkPhysicalDeviceLimits* vk_limits,
                                                                                 const VkPhysicalDeviceSubgroupProperties* vk_subgroup_props,
                                                                                 const VkPhysicalDeviceAccelerationStructurePropertiesKHR* vk_as_props)
{
  cgpu_physical_device_properties properties = {0};
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
  VkResult result = volkInitialize();

  if (result != VK_SUCCESS || volkGetInstanceVersion() < CGPU_MIN_VK_API_VERSION) {
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
  const char** validation_layers = NULL;
  uint32_t validation_layer_count = 0;
  const char** instance_extensions = NULL;
  uint32_t instance_extension_count = 0;
#endif

  VkApplicationInfo app_info;
  app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app_info.pNext = NULL;
  app_info.pApplicationName = p_app_name;
  app_info.applicationVersion = VK_MAKE_VERSION(version_major, version_minor, version_patch);
  app_info.pEngineName = p_app_name;
  app_info.engineVersion = VK_MAKE_VERSION(version_major, version_minor, version_patch);
  app_info.apiVersion = CGPU_MIN_VK_API_VERSION;

  VkInstanceCreateInfo create_info;
  create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  create_info.pNext = NULL;
  create_info.flags = 0;
  create_info.pApplicationInfo = &app_info;
  create_info.enabledLayerCount = validation_layer_count;
  create_info.ppEnabledLayerNames = validation_layers;
  create_info.enabledExtensionCount = instance_extension_count;
  create_info.ppEnabledExtensionNames = instance_extensions;

  result = vkCreateInstance(&create_info, NULL, &iinstance.instance);
  if (result != VK_SUCCESS) {
    CGPU_RETURN_ERROR("failed to create vulkan instance");
  }

  volkLoadInstanceOnly(iinstance.instance);

  resource_store_create(&idevice_store, sizeof(cgpu_idevice), 1);
  resource_store_create(&ibuffer_store, sizeof(cgpu_ibuffer), 16);
  resource_store_create(&iimage_store, sizeof(cgpu_iimage), 64);
  resource_store_create(&ishader_store, sizeof(cgpu_ishader), 16);
  resource_store_create(&ipipeline_store, sizeof(cgpu_ipipeline), 8);
  resource_store_create(&ifence_store, sizeof(cgpu_ifence), 8);
  resource_store_create(&icommand_buffer_store, sizeof(cgpu_icommand_buffer), 16);
  resource_store_create(&isampler_store, sizeof(cgpu_isampler), 8);
  resource_store_create(&iacceleration_structure_store, sizeof(cgpu_iacceleration_structure), 8);

  return true;
}

void cgpu_terminate(void)
{
  resource_store_destroy(&idevice_store);
  resource_store_destroy(&ibuffer_store);
  resource_store_destroy(&iimage_store);
  resource_store_destroy(&ishader_store);
  resource_store_destroy(&ipipeline_store);
  resource_store_destroy(&ifence_store);
  resource_store_destroy(&icommand_buffer_store);
  resource_store_destroy(&isampler_store);
  resource_store_destroy(&iacceleration_structure_store);

  vkDestroyInstance(iinstance.instance, NULL);
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

bool cgpu_create_device(cgpu_device* p_device)
{
  p_device->handle = resource_store_create_handle(&idevice_store);

  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(*p_device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  uint32_t phys_device_count;
  vkEnumeratePhysicalDevices(
    iinstance.instance,
    &phys_device_count,
    NULL
  );

  if (phys_device_count > CGPU_MAX_PHYSICAL_DEVICES)
  {
    resource_store_free_handle(&idevice_store, p_device->handle);
    CGPU_RETURN_ERROR_HARDCODED_LIMIT_REACHED;
  }

  if (phys_device_count == 0)
  {
    resource_store_free_handle(&idevice_store, p_device->handle);
    CGPU_RETURN_ERROR("no physical device found");
  }

  VkPhysicalDevice phys_devices[CGPU_MAX_PHYSICAL_DEVICES];

  vkEnumeratePhysicalDevices(
    iinstance.instance,
    &phys_device_count,
    phys_devices
  );

  idevice->physical_device = phys_devices[0];

  VkPhysicalDeviceFeatures features;
  vkGetPhysicalDeviceFeatures(idevice->physical_device, &features);
  idevice->features = cgpu_translate_physical_device_features(&features);

  VkPhysicalDeviceAccelerationStructurePropertiesKHR as_properties = {0};
  as_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR;
  as_properties.pNext = NULL;

  VkPhysicalDeviceRayTracingPipelinePropertiesKHR rt_pipeline_properties = {0};
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
    resource_store_free_handle(&idevice_store, p_device->handle);
    CGPU_RETURN_ERROR("unsupported vulkan version");
  }

  if ((subgroup_properties.supportedStages & VK_QUEUE_COMPUTE_BIT) != VK_QUEUE_COMPUTE_BIT ||
      (subgroup_properties.supportedOperations & VK_SUBGROUP_FEATURE_BASIC_BIT) != VK_SUBGROUP_FEATURE_BASIC_BIT ||
      (subgroup_properties.supportedOperations & VK_SUBGROUP_FEATURE_BALLOT_BIT) != VK_SUBGROUP_FEATURE_BALLOT_BIT)
  {
    resource_store_free_handle(&idevice_store, p_device->handle);
    CGPU_RETURN_ERROR("subgroup features not supported");
  }

  if (rt_pipeline_properties.maxRayRecursionDepth < CGPU_MAX_RAY_RECURSION_DEPTH)
  {
    fprintf(stderr, "error in %s:%d: max ray recursion depth of %d not supported\n", __FILE__, __LINE__, CGPU_MAX_RAY_RECURSION_DEPTH);
    return false;
  }

  const VkPhysicalDeviceLimits* limits = &device_properties.properties.limits;
  idevice->properties = cgpu_translate_physical_device_properties(limits, &subgroup_properties, &as_properties);

  uint32_t device_ext_count;
  vkEnumerateDeviceExtensionProperties(
    idevice->physical_device,
    NULL,
    &device_ext_count,
    NULL
  );

  if (device_ext_count > CGPU_MAX_DEVICE_EXTENSIONS)
  {
    resource_store_free_handle(&idevice_store, p_device->handle);
    CGPU_RETURN_ERROR_HARDCODED_LIMIT_REACHED;
  }

  VkExtensionProperties device_extensions[CGPU_MAX_DEVICE_EXTENSIONS];

  vkEnumerateDeviceExtensionProperties(
    idevice->physical_device,
    NULL,
    &device_ext_count,
    device_extensions
  );

  const char* required_extensions[] = {
    VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
    VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME, // required by VK_KHR_acceleration_structure
    VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME, // required by VK_KHR_acceleration_structure
    VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME, // required by VK_KHR_acceleration_structure
    VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
    VK_KHR_SPIRV_1_4_EXTENSION_NAME, // required by VK_KHR_ray_tracing_pipeline
    VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME // required by VK_KHR_spirv_1_4
  };
  uint32_t required_extension_count = sizeof(required_extensions) / sizeof(required_extensions[0]);

  uint32_t enabled_device_extension_count = 0;
  const char* enabled_device_extensions[32];

  for (uint32_t i = 0; i < required_extension_count; i++)
  {
    const char* extension = required_extensions[i];

    if (!cgpu_find_device_extension(extension, device_ext_count, device_extensions))
    {
      resource_store_free_handle(&idevice_store, p_device->handle);

      fprintf(stderr, "error in %s:%d: extension %s not supported\n", __FILE__, __LINE__, extension);
      return false;
    }

    enabled_device_extensions[enabled_device_extension_count] = extension;
    enabled_device_extension_count++;
  }

  const char* VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME = "VK_KHR_portability_subset";
  if (cgpu_find_device_extension(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME, device_ext_count, device_extensions))
  {
    enabled_device_extensions[enabled_device_extension_count] = VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME;
    enabled_device_extension_count++;
  }

#ifndef NDEBUG
  if (cgpu_find_device_extension(VK_KHR_SHADER_CLOCK_EXTENSION_NAME, device_ext_count, device_extensions) && features.shaderInt64)
  {
    idevice->features.shaderClock = true;
    enabled_device_extensions[enabled_device_extension_count] = VK_KHR_SHADER_CLOCK_EXTENSION_NAME;
    enabled_device_extension_count++;
  }

#ifndef __APPLE__
  if (cgpu_find_device_extension(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME, device_ext_count, device_extensions))
  {
    idevice->features.debugPrintf = true;
    enabled_device_extensions[enabled_device_extension_count] = VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME;
    enabled_device_extension_count++;
  }
#endif
#endif
  if (cgpu_find_device_extension(VK_EXT_MEMORY_PRIORITY_EXTENSION_NAME, device_ext_count, device_extensions) &&
      cgpu_find_device_extension(VK_EXT_PAGEABLE_DEVICE_LOCAL_MEMORY_EXTENSION_NAME, device_ext_count, device_extensions))
  {
    idevice->features.pageableDeviceLocalMemory = true;
    enabled_device_extensions[enabled_device_extension_count++] = VK_EXT_MEMORY_PRIORITY_EXTENSION_NAME;
    enabled_device_extensions[enabled_device_extension_count++] = VK_EXT_PAGEABLE_DEVICE_LOCAL_MEMORY_EXTENSION_NAME;
  }

  uint32_t queue_family_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(
    idevice->physical_device,
    &queue_family_count,
    NULL
  );

  if (queue_family_count > CGPU_MAX_QUEUE_FAMILIES)
  {
    resource_store_free_handle(&idevice_store, p_device->handle);
    CGPU_RETURN_ERROR_HARDCODED_LIMIT_REACHED;
  }

  VkQueueFamilyProperties queue_families[CGPU_MAX_QUEUE_FAMILIES];

  vkGetPhysicalDeviceQueueFamilyProperties(
    idevice->physical_device,
    &queue_family_count,
    queue_families
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
    resource_store_free_handle(&idevice_store, p_device->handle);
    CGPU_RETURN_ERROR("no suitable queue family");
  }

  VkDeviceQueueCreateInfo queue_create_info;
  queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queue_create_info.pNext = NULL;
  queue_create_info.flags = 0;
  queue_create_info.queueFamilyIndex = queue_family_index;
  queue_create_info.queueCount = 1;
  const float queue_priority = 1.0f;
  queue_create_info.pQueuePriorities = &queue_priority;

  void* pNext = NULL;

  VkPhysicalDevicePageableDeviceLocalMemoryFeaturesEXT pageable_memory_features = {0};
  pageable_memory_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PAGEABLE_DEVICE_LOCAL_MEMORY_FEATURES_EXT;
  pageable_memory_features.pNext = NULL;
  pageable_memory_features.pageableDeviceLocalMemory = VK_TRUE;

  if (idevice->features.pageableDeviceLocalMemory)
  {
    pNext = &pageable_memory_features;
  }

  VkPhysicalDeviceShaderClockFeaturesKHR shader_clock_features = {0};
  shader_clock_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR;
  shader_clock_features.pNext = pNext;
  shader_clock_features.shaderSubgroupClock = VK_TRUE;
  shader_clock_features.shaderDeviceClock = VK_FALSE;

  if (idevice->features.shaderClock)
  {
    pNext = &shader_clock_features;
  }

  VkPhysicalDeviceAccelerationStructureFeaturesKHR acceleration_structure_features = {0};
  acceleration_structure_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
  acceleration_structure_features.pNext = pNext;
  acceleration_structure_features.accelerationStructure = VK_TRUE;
  acceleration_structure_features.accelerationStructureCaptureReplay = VK_FALSE;
  acceleration_structure_features.accelerationStructureIndirectBuild = VK_FALSE;
  acceleration_structure_features.accelerationStructureHostCommands = VK_FALSE;
  acceleration_structure_features.descriptorBindingAccelerationStructureUpdateAfterBind = VK_FALSE;

  VkPhysicalDeviceRayTracingPipelineFeaturesKHR ray_tracing_pipeline_features = {0};
  ray_tracing_pipeline_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
  ray_tracing_pipeline_features.pNext = &acceleration_structure_features;
  ray_tracing_pipeline_features.rayTracingPipeline = VK_TRUE;
  ray_tracing_pipeline_features.rayTracingPipelineShaderGroupHandleCaptureReplay = VK_FALSE;
  ray_tracing_pipeline_features.rayTracingPipelineShaderGroupHandleCaptureReplayMixed = VK_FALSE;
  ray_tracing_pipeline_features.rayTracingPipelineTraceRaysIndirect = VK_FALSE;
  ray_tracing_pipeline_features.rayTraversalPrimitiveCulling = VK_FALSE;

  VkPhysicalDeviceBufferDeviceAddressFeaturesKHR buffer_device_address_features = {0};
  buffer_device_address_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
  buffer_device_address_features.pNext = &ray_tracing_pipeline_features;
  buffer_device_address_features.bufferDeviceAddress = VK_TRUE;
  buffer_device_address_features.bufferDeviceAddressCaptureReplay = VK_FALSE;
  buffer_device_address_features.bufferDeviceAddressMultiDevice = VK_FALSE;

  VkPhysicalDeviceDescriptorIndexingFeaturesEXT descriptor_indexing_features = {0};
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

  VkPhysicalDevice16BitStorageFeatures device_16bit_storage_featurs = {0};
  device_16bit_storage_featurs.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
  device_16bit_storage_featurs.pNext = &descriptor_indexing_features;
  device_16bit_storage_featurs.storageBuffer16BitAccess = VK_TRUE;
  device_16bit_storage_featurs.uniformAndStorageBuffer16BitAccess = VK_FALSE;
  device_16bit_storage_featurs.storagePushConstant16 = VK_FALSE;
  device_16bit_storage_featurs.storageInputOutput16 = VK_FALSE;

  VkPhysicalDeviceFeatures2 device_features2;
  device_features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  device_features2.pNext = &device_16bit_storage_featurs;
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
  device_create_info.ppEnabledLayerNames = NULL;
  device_create_info.enabledExtensionCount = enabled_device_extension_count;
  device_create_info.ppEnabledExtensionNames = enabled_device_extensions;
  device_create_info.pEnabledFeatures = NULL;

  VkResult result = vkCreateDevice(
    idevice->physical_device,
    &device_create_info,
    NULL,
    &idevice->logical_device
  );
  if (result != VK_SUCCESS) {
    resource_store_free_handle(&idevice_store, p_device->handle);
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
  pool_info.pNext = NULL;
  pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  pool_info.queueFamilyIndex = queue_family_index;

  result = idevice->table.vkCreateCommandPool(
    idevice->logical_device,
    &pool_info,
    NULL,
    &idevice->command_pool
  );

  if (result != VK_SUCCESS)
  {
    resource_store_free_handle(&idevice_store, p_device->handle);

    idevice->table.vkDestroyDevice(
      idevice->logical_device,
      NULL
    );

    CGPU_RETURN_ERROR("failed to create command pool");
  }

  VkQueryPoolCreateInfo timestamp_pool_info;
  timestamp_pool_info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
  timestamp_pool_info.pNext = NULL;
  timestamp_pool_info.flags = 0;
  timestamp_pool_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
  timestamp_pool_info.queryCount = CGPU_MAX_TIMESTAMP_QUERIES;
  timestamp_pool_info.pipelineStatistics = 0;

  result = idevice->table.vkCreateQueryPool(
    idevice->logical_device,
    &timestamp_pool_info,
    NULL,
    &idevice->timestamp_pool
  );

  if (result != VK_SUCCESS)
  {
    resource_store_free_handle(&idevice_store, p_device->handle);

    idevice->table.vkDestroyCommandPool(
      idevice->logical_device,
      idevice->command_pool,
      NULL
    );
    idevice->table.vkDestroyDevice(
      idevice->logical_device,
      NULL
    );

    CGPU_RETURN_ERROR("failed to create query pool");
  }

  VmaVulkanFunctions vulkan_functions = {0};
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

  VmaAllocatorCreateInfo alloc_create_info = {0};
  alloc_create_info.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
  alloc_create_info.vulkanApiVersion = CGPU_MIN_VK_API_VERSION;
  alloc_create_info.physicalDevice = idevice->physical_device;
  alloc_create_info.device = idevice->logical_device;
  alloc_create_info.instance = iinstance.instance;
  alloc_create_info.pVulkanFunctions = &vulkan_functions;

  result = vmaCreateAllocator(&alloc_create_info, &idevice->allocator);

  if (result != VK_SUCCESS)
  {
    resource_store_free_handle(&idevice_store, p_device->handle);

    idevice->table.vkDestroyQueryPool(
      idevice->logical_device,
      idevice->timestamp_pool,
      NULL
    );
    idevice->table.vkDestroyCommandPool(
      idevice->logical_device,
      idevice->command_pool,
      NULL
    );
    idevice->table.vkDestroyDevice(
      idevice->logical_device,
      NULL
    );
    CGPU_RETURN_ERROR("failed to create vma allocator");
  }

  return true;
}

bool cgpu_destroy_device(cgpu_device device)
{
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  vmaDestroyAllocator(idevice->allocator);

  idevice->table.vkDestroyQueryPool(
    idevice->logical_device,
    idevice->timestamp_pool,
    NULL
  );
  idevice->table.vkDestroyCommandPool(
    idevice->logical_device,
    idevice->command_pool,
    NULL
  );
  idevice->table.vkDestroyDevice(
    idevice->logical_device,
    NULL
  );

  resource_store_free_handle(&idevice_store, device.handle);
  return true;
}

bool cgpu_create_shader(cgpu_device device,
                        uint64_t size,
                        const uint8_t* p_source,
                        CgpuShaderStageFlags stage_flags,
                        cgpu_shader* p_shader)
{
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  p_shader->handle = resource_store_create_handle(&ishader_store);

  cgpu_ishader* ishader;
  if (!cgpu_resolve_shader(*p_shader, &ishader)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  VkShaderModuleCreateInfo shader_module_create_info;
  shader_module_create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  shader_module_create_info.pNext = NULL;
  shader_module_create_info.flags = 0;
  shader_module_create_info.codeSize = size;
  shader_module_create_info.pCode = (uint32_t*) p_source;

  VkResult result = idevice->table.vkCreateShaderModule(
    idevice->logical_device,
    &shader_module_create_info,
    NULL,
    &ishader->module
  );
  if (result != VK_SUCCESS) {
    resource_store_free_handle(&ishader_store, p_shader->handle);
    CGPU_RETURN_ERROR("failed to create shader module");
  }

  if (!cgpu_perform_shader_reflection(size, (uint32_t*) p_source, &ishader->reflection))
  {
    idevice->table.vkDestroyShaderModule(
      idevice->logical_device,
      ishader->module,
      NULL
    );
    resource_store_free_handle(&ishader_store, p_shader->handle);
    CGPU_RETURN_ERROR("failed to reflect shader");
  }

  ishader->stage_flags = (VkShaderStageFlags) stage_flags;

  return true;
}

bool cgpu_destroy_shader(cgpu_device device,
                         cgpu_shader shader)
{
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  cgpu_ishader* ishader;
  if (!cgpu_resolve_shader(shader, &ishader)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  cgpu_destroy_shader_reflection(&ishader->reflection);

  idevice->table.vkDestroyShaderModule(
    idevice->logical_device,
    ishader->module,
    NULL
  );

  resource_store_free_handle(&ishader_store, shader.handle);

  return true;
}

static bool cgpu_create_buffer_aligned(cgpu_device device,
                                       CgpuBufferUsageFlags usage,
                                       CgpuMemoryPropertyFlags memory_properties,
                                       uint64_t size,
                                       uint64_t alignment,
                                       cgpu_buffer* p_buffer)
{
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  p_buffer->handle = resource_store_create_handle(&ibuffer_store);

  cgpu_ibuffer* ibuffer;
  if (!cgpu_resolve_buffer(*p_buffer, &ibuffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  VkBufferCreateInfo buffer_info;
  buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_info.pNext = NULL;
  buffer_info.flags = 0;
  buffer_info.size = size;
  buffer_info.usage = (VkBufferUsageFlags) usage;
  buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  buffer_info.queueFamilyIndexCount = 0;
  buffer_info.pQueueFamilyIndices = NULL;

  VmaAllocationCreateInfo alloc_info = {0};
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
      NULL
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
      NULL
    );
  }

  if (result != VK_SUCCESS)
  {
    resource_store_free_handle(&ibuffer_store, p_buffer->handle);
    CGPU_RETURN_ERROR("failed to create buffer");
  }

  ibuffer->size = size;

  return true;
}

bool cgpu_create_buffer(cgpu_device device,
                        CgpuBufferUsageFlags usage,
                        CgpuMemoryPropertyFlags memory_properties,
                        uint64_t size,
                        cgpu_buffer* p_buffer)
{
  uint64_t alignment = 0;

  return cgpu_create_buffer_aligned(device, usage, memory_properties, size, alignment, p_buffer);
}

bool cgpu_destroy_buffer(cgpu_device device,
                         cgpu_buffer buffer)
{
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  cgpu_ibuffer* ibuffer;
  if (!cgpu_resolve_buffer(buffer, &ibuffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  vmaDestroyBuffer(idevice->allocator, ibuffer->buffer, ibuffer->allocation);

  resource_store_free_handle(&ibuffer_store, buffer.handle);

  return true;
}

bool cgpu_map_buffer(cgpu_device device,
                     cgpu_buffer buffer,
                     void** pp_mapped_mem)
{
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  cgpu_ibuffer* ibuffer;
  if (!cgpu_resolve_buffer(buffer, &ibuffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  if (vmaMapMemory(idevice->allocator, ibuffer->allocation, pp_mapped_mem) != VK_SUCCESS) {
    CGPU_RETURN_ERROR("failed to map buffer memory");
  }
  return true;
}

bool cgpu_unmap_buffer(cgpu_device device,
                       cgpu_buffer buffer)
{
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  cgpu_ibuffer* ibuffer;
  if (!cgpu_resolve_buffer(buffer, &ibuffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  vmaUnmapMemory(idevice->allocator, ibuffer->allocation);
  return true;
}

bool cgpu_create_image(cgpu_device device,
                       const cgpu_image_description* image_desc,
                       cgpu_image* p_image)
{
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  p_image->handle = resource_store_create_handle(&iimage_store);

  cgpu_iimage* iimage;
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
  image_info.pNext = NULL;
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
  image_info.pQueueFamilyIndices = NULL;
  image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

  VmaAllocationCreateInfo alloc_info = {0};
  alloc_info.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

  VkResult result = vmaCreateImage(
    idevice->allocator,
    &image_info,
    &alloc_info,
    &iimage->image,
    &iimage->allocation,
    NULL
  );

  if (result != VK_SUCCESS) {
    resource_store_free_handle(&iimage_store, p_image->handle);
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
  image_view_info.pNext = NULL;
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
    NULL,
    &iimage->image_view
  );
  if (result != VK_SUCCESS)
  {
    resource_store_free_handle(&iimage_store, p_image->handle);
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

bool cgpu_destroy_image(cgpu_device device,
                        cgpu_image image)
{
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  cgpu_iimage* iimage;
  if (!cgpu_resolve_image(image, &iimage)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  idevice->table.vkDestroyImageView(
    idevice->logical_device,
    iimage->image_view,
    NULL
  );

  vmaDestroyImage(idevice->allocator, iimage->image, iimage->allocation);

  resource_store_free_handle(&iimage_store, image.handle);

  return true;
}

bool cgpu_map_image(cgpu_device device,
                    cgpu_image image,
                    void** pp_mapped_mem)
{
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  cgpu_iimage* iimage;
  if (!cgpu_resolve_image(image, &iimage)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  if (vmaMapMemory(idevice->allocator, iimage->allocation, pp_mapped_mem) != VK_SUCCESS) {
    CGPU_RETURN_ERROR("failed to map image memory");
  }
  return true;
}

bool cgpu_unmap_image(cgpu_device device,
                      cgpu_image image)
{
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  cgpu_iimage* iimage;
  if (!cgpu_resolve_image(image, &iimage)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  vmaUnmapMemory(idevice->allocator, iimage->allocation);
  return true;
}

bool cgpu_create_sampler(cgpu_device device,
                         CgpuSamplerAddressMode address_mode_u,
                         CgpuSamplerAddressMode address_mode_v,
                         CgpuSamplerAddressMode address_mode_w,
                         cgpu_sampler* p_sampler)
{
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  p_sampler->handle = resource_store_create_handle(&isampler_store);

  cgpu_isampler* isampler;
  if (!cgpu_resolve_sampler(*p_sampler, &isampler)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  // Emulate MDL's clip wrap mode if necessary; use optimal mode (according to ARM) if not.
  bool clampToBlack = (address_mode_u == CGPU_SAMPLER_ADDRESS_MODE_CLAMP_TO_BLACK) ||
                      (address_mode_v == CGPU_SAMPLER_ADDRESS_MODE_CLAMP_TO_BLACK) ||
                      (address_mode_w == CGPU_SAMPLER_ADDRESS_MODE_CLAMP_TO_BLACK);

  VkSamplerCreateInfo create_info;
  create_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  create_info.pNext = NULL;
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
    NULL,
    &isampler->sampler
  );

  if (result != VK_SUCCESS) {
    resource_store_free_handle(&isampler_store, p_sampler->handle);
    CGPU_RETURN_ERROR("failed to create sampler");
  }

  return true;
}

bool cgpu_destroy_sampler(cgpu_device device,
                          cgpu_sampler sampler)
{
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  cgpu_isampler* isampler;
  if (!cgpu_resolve_sampler(sampler, &isampler)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  idevice->table.vkDestroySampler(idevice->logical_device, isampler->sampler, NULL);

  resource_store_free_handle(&isampler_store, sampler.handle);

  return true;
}

static bool cgpu_create_pipeline_layout(cgpu_idevice* idevice, cgpu_ipipeline* ipipeline, cgpu_ishader* ishader)
{
  VkPushConstantRange push_const_range;
  push_const_range.stageFlags = ishader->stage_flags;
  push_const_range.offset = 0;
  push_const_range.size = ishader->reflection.push_constants_size;

  VkPipelineLayoutCreateInfo pipeline_layout_create_info;
  pipeline_layout_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipeline_layout_create_info.pNext = NULL;
  pipeline_layout_create_info.flags = 0;
  pipeline_layout_create_info.setLayoutCount = 1;
  pipeline_layout_create_info.pSetLayouts = &ipipeline->descriptor_set_layout;
  pipeline_layout_create_info.pushConstantRangeCount = push_const_range.size ? 1 : 0;
  pipeline_layout_create_info.pPushConstantRanges = &push_const_range;

  return idevice->table.vkCreatePipelineLayout(idevice->logical_device,
                                               &pipeline_layout_create_info,
                                               NULL,
                                               &ipipeline->layout) == VK_SUCCESS;
}

static bool cgpu_create_pipeline_descriptors(cgpu_idevice* idevice, cgpu_ipipeline* ipipeline, cgpu_ishader* ishader)
{
  const cgpu_shader_reflection* shader_reflection = &ishader->reflection;

  if (shader_reflection->binding_count >= CGPU_MAX_DESCRIPTOR_SET_LAYOUT_BINDINGS)
  {
    CGPU_RETURN_ERROR_HARDCODED_LIMIT_REACHED;
  }

  ipipeline->descriptor_set_layout_binding_count = shader_reflection->binding_count;

  for (uint32_t i = 0; i < shader_reflection->binding_count; i++)
  {
    const cgpu_shader_reflection_binding* binding = &shader_reflection->bindings[i];

    VkDescriptorSetLayoutBinding* descriptor_set_layout_binding = &ipipeline->descriptor_set_layout_bindings[i];
    descriptor_set_layout_binding->binding = binding->binding;
    descriptor_set_layout_binding->descriptorType = binding->descriptor_type;
    descriptor_set_layout_binding->descriptorCount = binding->count;
    descriptor_set_layout_binding->stageFlags = ishader->stage_flags;
    descriptor_set_layout_binding->pImmutableSamplers = NULL;
  }

  VkDescriptorSetLayoutCreateInfo descriptor_set_layout_create_info;
  descriptor_set_layout_create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  descriptor_set_layout_create_info.pNext = NULL;
  descriptor_set_layout_create_info.flags = 0;
  descriptor_set_layout_create_info.bindingCount = shader_reflection->binding_count;
  descriptor_set_layout_create_info.pBindings = ipipeline->descriptor_set_layout_bindings;

  VkResult result = idevice->table.vkCreateDescriptorSetLayout(
    idevice->logical_device,
    &descriptor_set_layout_create_info,
    NULL,
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
    const cgpu_shader_reflection_binding* binding = &shader_reflection->bindings[i];

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
        NULL
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
  descriptor_pool_create_info.pNext = NULL;
  descriptor_pool_create_info.flags = 0;
  descriptor_pool_create_info.maxSets = 1;
  descriptor_pool_create_info.poolSizeCount = pool_size_count;
  descriptor_pool_create_info.pPoolSizes = pool_sizes;

  result = idevice->table.vkCreateDescriptorPool(
    idevice->logical_device,
    &descriptor_pool_create_info,
    NULL,
    &ipipeline->descriptor_pool
  );
  if (result != VK_SUCCESS) {
    idevice->table.vkDestroyDescriptorSetLayout(
      idevice->logical_device,
      ipipeline->descriptor_set_layout,
      NULL
    );
    CGPU_RETURN_ERROR("failed to create descriptor pool");
  }

  VkDescriptorSetAllocateInfo descriptor_set_allocate_info;
  descriptor_set_allocate_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  descriptor_set_allocate_info.pNext = NULL;
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
      NULL
    );
    idevice->table.vkDestroyDescriptorSetLayout(
      idevice->logical_device,
      ipipeline->descriptor_set_layout,
      NULL
    );
    CGPU_RETURN_ERROR("failed to allocate descriptor set");
  }

  return true;
}

bool cgpu_create_compute_pipeline(cgpu_device device,
                                  cgpu_shader shader,
                                  cgpu_pipeline* p_pipeline)
{
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  cgpu_ishader* ishader;
  if (!cgpu_resolve_shader(shader, &ishader)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  p_pipeline->handle = resource_store_create_handle(&ipipeline_store);

  cgpu_ipipeline* ipipeline;
  if (!cgpu_resolve_pipeline(*p_pipeline, &ipipeline)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  if (!cgpu_create_pipeline_descriptors(idevice, ipipeline, ishader))
  {
    resource_store_free_handle(&ipipeline_store, p_pipeline->handle);
    CGPU_RETURN_ERROR("failed to create descriptor set layout");
  }

  if (!cgpu_create_pipeline_layout(idevice, ipipeline, ishader))
  {
    resource_store_free_handle(&ipipeline_store, p_pipeline->handle);
    idevice->table.vkDestroyDescriptorSetLayout(
      idevice->logical_device,
      ipipeline->descriptor_set_layout,
      NULL
    );
    idevice->table.vkDestroyDescriptorPool(
      idevice->logical_device,
      ipipeline->descriptor_pool,
      NULL
    );
    CGPU_RETURN_ERROR("failed to create pipeline layout");
  }

  VkPipelineShaderStageCreateInfo pipeline_shader_stage_create_info;
  pipeline_shader_stage_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  pipeline_shader_stage_create_info.pNext = NULL;
  pipeline_shader_stage_create_info.flags = 0;
  pipeline_shader_stage_create_info.stage = ishader->stage_flags;
  pipeline_shader_stage_create_info.module = ishader->module;
  pipeline_shader_stage_create_info.pName = "main";
  pipeline_shader_stage_create_info.pSpecializationInfo = NULL;

  VkComputePipelineCreateInfo pipeline_create_info;
  pipeline_create_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipeline_create_info.pNext = NULL;
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
    NULL,
    &ipipeline->pipeline
  );

  if (result != VK_SUCCESS) {
    resource_store_free_handle(&ipipeline_store, p_pipeline->handle);
    idevice->table.vkDestroyPipelineLayout(
      idevice->logical_device,
      ipipeline->layout,
      NULL
    );
    idevice->table.vkDestroyDescriptorSetLayout(
      idevice->logical_device,
      ipipeline->descriptor_set_layout,
      NULL
    );
    idevice->table.vkDestroyDescriptorPool(
      idevice->logical_device,
      ipipeline->descriptor_pool,
      NULL
    );
    CGPU_RETURN_ERROR("failed to create compute pipeline");
  }

  ipipeline->shader = shader;
  ipipeline->bind_point = VK_PIPELINE_BIND_POINT_COMPUTE;

  return true;
}

bool cgpu_destroy_pipeline(cgpu_device device,
                           cgpu_pipeline pipeline)
{
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  cgpu_ipipeline* ipipeline;
  if (!cgpu_resolve_pipeline(pipeline, &ipipeline)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  idevice->table.vkDestroyDescriptorPool(
    idevice->logical_device,
    ipipeline->descriptor_pool,
    NULL
  );
  idevice->table.vkDestroyPipeline(
    idevice->logical_device,
    ipipeline->pipeline,
    NULL
  );
  idevice->table.vkDestroyPipelineLayout(
    idevice->logical_device,
    ipipeline->layout,
    NULL
  );
  idevice->table.vkDestroyDescriptorSetLayout(
    idevice->logical_device,
    ipipeline->descriptor_set_layout,
    NULL
  );

  resource_store_free_handle(&ipipeline_store, pipeline.handle);

  return true;
}

static uint64_t cgpu_get_acceleration_structure_buffer_address(cgpu_idevice* idevice, cgpu_ibuffer* ibuffer)
{
  VkBufferDeviceAddressInfoKHR address_info = {0};
  address_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
  address_info.pNext = NULL;
  address_info.buffer = ibuffer->buffer;
  return idevice->table.vkGetBufferDeviceAddressKHR(idevice->logical_device, &address_info);
}

static bool cgpu_create_top_or_bottom_as(cgpu_device device,
                                         VkAccelerationStructureTypeKHR as_type,
                                         VkAccelerationStructureGeometryKHR* as_geom,
                                         uint32_t primitive_count,
                                         cgpu_iacceleration_structure* iacceleration_structure)
{
  bool is_bottom_level = as_type == VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
  assert(is_bottom_level || as_type == VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR);

  cgpu_idevice* idevice;
  cgpu_resolve_device(device, &idevice);

  // Get AS size
  VkAccelerationStructureBuildGeometryInfoKHR as_build_geom_info = {0};
  as_build_geom_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
  as_build_geom_info.pNext = NULL;
  as_build_geom_info.type = as_type;
  as_build_geom_info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  as_build_geom_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
  as_build_geom_info.srcAccelerationStructure = VK_NULL_HANDLE;
  as_build_geom_info.dstAccelerationStructure = VK_NULL_HANDLE; // set in second round
  as_build_geom_info.geometryCount = 1;
  as_build_geom_info.pGeometries = as_geom;
  as_build_geom_info.ppGeometries = NULL;
  as_build_geom_info.scratchData.hostAddress = NULL;
  as_build_geom_info.scratchData.deviceAddress = 0; // set in second round

  VkAccelerationStructureBuildSizesInfoKHR as_build_sizes_info = {0};
  as_build_sizes_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
  as_build_sizes_info.pNext = NULL;
  as_build_sizes_info.accelerationStructureSize = 0; // output
  as_build_sizes_info.updateScratchSize = 0; // output
  as_build_sizes_info.buildScratchSize = 0; // output

  idevice->table.vkGetAccelerationStructureBuildSizesKHR(idevice->logical_device,
                                                         VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                                         &as_build_geom_info,
                                                         &primitive_count,
                                                         &as_build_sizes_info);

  // Create AS buffer & AS object
  cgpu_buffer* as_buffer = is_bottom_level ? &iacceleration_structure->blas_buffer : &iacceleration_structure->tlas_buffer;
  if (!cgpu_create_buffer(device,
                          CGPU_BUFFER_USAGE_FLAG_SHADER_DEVICE_ADDRESS | CGPU_BUFFER_USAGE_FLAG_ACCELERATION_STRUCTURE_STORAGE,
                          CGPU_MEMORY_PROPERTY_FLAG_DEVICE_LOCAL,
                          as_build_sizes_info.accelerationStructureSize,
                          as_buffer))
  {
    CGPU_RETURN_ERROR("failed to create AS buffer");
  }
  cgpu_ibuffer* ias_buffer;
  cgpu_resolve_buffer(*as_buffer, &ias_buffer);

  VkAccelerationStructureCreateInfoKHR as_create_info = {0};
  as_create_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
  as_create_info.pNext = NULL;
  as_create_info.createFlags = 0;
  as_create_info.buffer = ias_buffer->buffer;
  as_create_info.offset = 0;
  as_create_info.size = as_build_sizes_info.accelerationStructureSize;
  as_create_info.type = as_type;
  as_create_info.deviceAddress = 0; // used for capture-replay feature

  VkAccelerationStructureKHR* as = is_bottom_level ? &iacceleration_structure->blas : &iacceleration_structure->tlas;
  if (idevice->table.vkCreateAccelerationStructureKHR(idevice->logical_device, &as_create_info, NULL, as) != VK_SUCCESS) {
    cgpu_destroy_buffer(device, *as_buffer);
    CGPU_RETURN_ERROR("failed to create Vulkan AS object");
  }

  // Set up device-local scratch buffer
  cgpu_buffer scratch_buffer;
  if (!cgpu_create_buffer_aligned(device,
                                  CGPU_BUFFER_USAGE_FLAG_STORAGE_BUFFER | CGPU_BUFFER_USAGE_FLAG_SHADER_DEVICE_ADDRESS,
                                  CGPU_MEMORY_PROPERTY_FLAG_DEVICE_LOCAL,
                                  as_build_sizes_info.buildScratchSize,
                                  idevice->properties.minAccelerationStructureScratchOffsetAlignment,
                                  &scratch_buffer))
  {
    cgpu_destroy_buffer(device, *as_buffer);
    idevice->table.vkDestroyAccelerationStructureKHR(idevice->logical_device, *as, NULL);
    CGPU_RETURN_ERROR("failed to create AS scratch buffer");
  }

  cgpu_ibuffer* iscratch_buffer;
  cgpu_resolve_buffer(scratch_buffer, &iscratch_buffer);

  as_build_geom_info.dstAccelerationStructure = is_bottom_level ? iacceleration_structure->blas : iacceleration_structure->tlas;
  as_build_geom_info.scratchData.hostAddress = 0;
  as_build_geom_info.scratchData.deviceAddress = cgpu_get_acceleration_structure_buffer_address(idevice, iscratch_buffer);

  VkAccelerationStructureBuildRangeInfoKHR as_build_range_info;
  as_build_range_info.primitiveCount = primitive_count;
  as_build_range_info.primitiveOffset = 0;
  as_build_range_info.firstVertex = 0;
  as_build_range_info.transformOffset = 0;

  const VkAccelerationStructureBuildRangeInfoKHR* as_build_range_info_ptr = &as_build_range_info;

  cgpu_command_buffer command_buffer;
  if (!cgpu_create_command_buffer(device, &command_buffer)) {
    cgpu_destroy_buffer(device, scratch_buffer);
    cgpu_destroy_buffer(device, *as_buffer);
    idevice->table.vkDestroyAccelerationStructureKHR(idevice->logical_device, *as, NULL);
    CGPU_RETURN_ERROR("failed to create AS build command buffer");
  }

  cgpu_icommand_buffer* icommand_buffer;
  cgpu_resolve_command_buffer(command_buffer, &icommand_buffer);

  // Build AS on device
  cgpu_begin_command_buffer(command_buffer);
  idevice->table.vkCmdBuildAccelerationStructuresKHR(icommand_buffer->command_buffer, 1, &as_build_geom_info, &as_build_range_info_ptr);
  cgpu_end_command_buffer(command_buffer);

  cgpu_fence fence;
  if (!cgpu_create_fence(device, &fence)) {
    cgpu_destroy_buffer(device, scratch_buffer);
    cgpu_destroy_buffer(device, *as_buffer);
    idevice->table.vkDestroyAccelerationStructureKHR(idevice->logical_device, *as, NULL);
    CGPU_RETURN_ERROR("failed to create AS build fence");
  }
  cgpu_reset_fence(device, fence);
  cgpu_submit_command_buffer(device, command_buffer, fence);
  cgpu_wait_for_fence(device, fence);

  if (is_bottom_level)
  {
    VkAccelerationStructureDeviceAddressInfoKHR as_address_info = { 0 };
    as_address_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    as_address_info.pNext = NULL;
    as_address_info.accelerationStructure = *as;

    iacceleration_structure->blas_address = idevice->table.vkGetAccelerationStructureDeviceAddressKHR(idevice->logical_device, &as_address_info);
  }

  // Dispose resources
  cgpu_destroy_fence(device, fence);
  cgpu_destroy_command_buffer(device, command_buffer);
  cgpu_destroy_buffer(device, scratch_buffer);

  return true;
}

static bool cgpu_create_blas(cgpu_device device,
                             uint32_t vertex_count,
                             const cgpu_vertex* vertices,
                             uint32_t index_count,
                             const uint32_t* indices,
                             cgpu_iacceleration_structure* iacceleration_structure)
{
  if ((index_count % 3) != 0) {
    CGPU_RETURN_ERROR("BLAS indices do not represent triangles");
  }
  cgpu_idevice* idevice;
  cgpu_resolve_device(device, &idevice);

  // Create index buffer & copy data into it
  uint64_t index_buffer_size = index_count * sizeof(uint32_t);
  if (!cgpu_create_buffer(device,
                          CGPU_BUFFER_USAGE_FLAG_SHADER_DEVICE_ADDRESS | CGPU_BUFFER_USAGE_FLAG_ACCELERATION_STRUCTURE_BUILD_INPUT,
                          CGPU_MEMORY_PROPERTY_FLAG_HOST_VISIBLE | CGPU_MEMORY_PROPERTY_FLAG_HOST_COHERENT,
                          index_buffer_size,
                          &iacceleration_structure->index_buffer))
  {
    CGPU_RETURN_ERROR("failed to create BLAS index buffer");
  }

  cgpu_ibuffer* iindex_buffer;
  cgpu_resolve_buffer(iacceleration_structure->index_buffer, &iindex_buffer);

  {
    bool result;
    void* mapped_mem;
    result = cgpu_map_buffer(device, iacceleration_structure->index_buffer, &mapped_mem);
    if (!result) {
      cgpu_destroy_buffer(device, iacceleration_structure->index_buffer);
      CGPU_RETURN_ERROR("failed to map BLAS index buffer memory");
    }
    memcpy(mapped_mem, indices, index_buffer_size);
    result = cgpu_unmap_buffer(device, iacceleration_structure->index_buffer);
    assert(result);
  }

  // Create vertex buffer & copy data into it
  uint64_t vertex_buffer_size = vertex_count * sizeof(cgpu_vertex);
  if (!cgpu_create_buffer(device,
                          CGPU_BUFFER_USAGE_FLAG_SHADER_DEVICE_ADDRESS | CGPU_BUFFER_USAGE_FLAG_ACCELERATION_STRUCTURE_BUILD_INPUT,
                          CGPU_MEMORY_PROPERTY_FLAG_HOST_VISIBLE | CGPU_MEMORY_PROPERTY_FLAG_HOST_COHERENT,
                          vertex_buffer_size,
                          &iacceleration_structure->vertex_buffer))
  {
    cgpu_destroy_buffer(device, iacceleration_structure->index_buffer);
    CGPU_RETURN_ERROR("failed to create BLAS vertex buffer");
  }

  cgpu_ibuffer* ivertex_buffer;
  cgpu_resolve_buffer(iacceleration_structure->vertex_buffer, &ivertex_buffer);

  {
    void* mapped_mem;
    bool result = cgpu_map_buffer(device, iacceleration_structure->vertex_buffer, &mapped_mem);
    if (!result) {
      cgpu_destroy_buffer(device, iacceleration_structure->index_buffer);
      cgpu_destroy_buffer(device, iacceleration_structure->vertex_buffer);
      CGPU_RETURN_ERROR("failed to map BLAS vertex buffer memory");
    }
    memcpy(mapped_mem, vertices, vertex_buffer_size);
    result = cgpu_unmap_buffer(device, iacceleration_structure->vertex_buffer);
    assert(result);
  }

  // Create BLAS
  VkAccelerationStructureGeometryTrianglesDataKHR as_triangle_data = {0};
  as_triangle_data.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
  as_triangle_data.pNext = NULL;
  as_triangle_data.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
  as_triangle_data.vertexData.hostAddress = NULL;
  as_triangle_data.vertexData.deviceAddress = cgpu_get_acceleration_structure_buffer_address(idevice, ivertex_buffer);
  as_triangle_data.vertexStride = sizeof(cgpu_vertex);
  as_triangle_data.maxVertex = vertex_count;
  as_triangle_data.indexType = VK_INDEX_TYPE_UINT32;
  as_triangle_data.indexData.hostAddress = NULL;
  as_triangle_data.indexData.deviceAddress = cgpu_get_acceleration_structure_buffer_address(idevice, iindex_buffer);
  as_triangle_data.transformData.hostAddress = NULL;
  as_triangle_data.transformData.deviceAddress = 0; // optional

  VkAccelerationStructureGeometryDataKHR as_geom_data = {0};
  as_geom_data.triangles = as_triangle_data;

  VkAccelerationStructureGeometryKHR as_geom = {0};
  as_geom.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
  as_geom.pNext = NULL;
  as_geom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
  as_geom.geometry = as_geom_data;
  as_geom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;

  uint32_t triangle_count = index_count / 3;
  if (!cgpu_create_top_or_bottom_as(device, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, &as_geom, triangle_count, iacceleration_structure)) {
    cgpu_destroy_buffer(device, iacceleration_structure->index_buffer);
    cgpu_destroy_buffer(device, iacceleration_structure->vertex_buffer);
    CGPU_RETURN_ERROR("failed to build BLAS");
  }

  return true;
}

static bool cgpu_create_tlas(cgpu_device device, cgpu_iacceleration_structure* iacceleration_structure)
{
  cgpu_idevice* idevice;
  cgpu_resolve_device(device, &idevice);

  // Create instance buffer & copy matrix into it
  VkTransformMatrixKHR transform_matrix = {
    1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f
  };

  VkAccelerationStructureInstanceKHR as_instance = {0};
  as_instance.transform = transform_matrix;
  as_instance.instanceCustomIndex = 0;
  as_instance.mask = 0xFF;
  as_instance.instanceShaderBindingTableRecordOffset = 0;
  as_instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
  as_instance.accelerationStructureReference = iacceleration_structure->blas_address;

  if (!cgpu_create_buffer(device,
                          CGPU_BUFFER_USAGE_FLAG_SHADER_DEVICE_ADDRESS | CGPU_BUFFER_USAGE_FLAG_ACCELERATION_STRUCTURE_BUILD_INPUT,
                          CGPU_MEMORY_PROPERTY_FLAG_HOST_VISIBLE | CGPU_MEMORY_PROPERTY_FLAG_HOST_COHERENT,
                          sizeof(VkAccelerationStructureInstanceKHR),
                          &iacceleration_structure->instances_buffer))
  {
    CGPU_RETURN_ERROR("failed to create TLAS instances buffer");
  }

  cgpu_ibuffer* iinstances_buffer;
  cgpu_resolve_buffer(iacceleration_structure->instances_buffer, &iinstances_buffer);

  {
    void* mapped_mem;
    bool result = cgpu_map_buffer(device, iacceleration_structure->instances_buffer, &mapped_mem);
    if (!result) {
      cgpu_destroy_buffer(device, iacceleration_structure->instances_buffer);
      CGPU_RETURN_ERROR("failed to map TLAS instances buffer memory");
    }
    memcpy(mapped_mem, &as_instance, sizeof(VkAccelerationStructureInstanceKHR));
    result = cgpu_unmap_buffer(device, iacceleration_structure->instances_buffer);
    assert(result);
  }

  // Create TLAS
  VkAccelerationStructureGeometryKHR as_geom = {0};
  as_geom.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
  as_geom.pNext = NULL;
  as_geom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
	as_geom.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
	as_geom.geometry.instances.pNext = NULL;
	as_geom.geometry.instances.arrayOfPointers = VK_FALSE;
	as_geom.geometry.instances.data.hostAddress = NULL;
	as_geom.geometry.instances.data.deviceAddress = cgpu_get_acceleration_structure_buffer_address(idevice, iinstances_buffer);
  as_geom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;

  uint32_t primitive_count = 1;
  if (!cgpu_create_top_or_bottom_as(device, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, &as_geom, primitive_count, iacceleration_structure)) {
    cgpu_destroy_buffer(device, iacceleration_structure->instances_buffer);
    CGPU_RETURN_ERROR("failed to build TLAS");
  }

  return true;
}

bool cgpu_create_acceleration_structure(cgpu_device device,
                                        uint32_t vertex_count,
                                        const cgpu_vertex* vertices,
                                        uint32_t index_count,
                                        const uint32_t* indices,
                                        cgpu_acceleration_structure* p_acceleration_structure)
{
  p_acceleration_structure->handle = resource_store_create_handle(&iacceleration_structure_store);

  cgpu_iacceleration_structure* iacceleration_structure;
  if (!cgpu_resolve_acceleration_structure(*p_acceleration_structure, &iacceleration_structure))
  {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  if (!cgpu_create_blas(device, vertex_count, vertices, index_count, indices, iacceleration_structure))
  {
    resource_store_free_handle(&iacceleration_structure_store, p_acceleration_structure->handle);
    CGPU_RETURN_ERROR("failed to build AS");
  }

  if (!cgpu_create_tlas(device, iacceleration_structure))
  {
    resource_store_free_handle(&iacceleration_structure_store, p_acceleration_structure->handle);
    CGPU_RETURN_ERROR("failed to build AS");
  }

  return true;
}

bool cgpu_destroy_acceleration_structure(cgpu_device device, cgpu_acceleration_structure acceleration_structure)
{
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  cgpu_iacceleration_structure* iacceleration_structure;
  if (!cgpu_resolve_acceleration_structure(acceleration_structure, &iacceleration_structure)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  idevice->table.vkDestroyAccelerationStructureKHR(idevice->logical_device, iacceleration_structure->blas, NULL);
  idevice->table.vkDestroyAccelerationStructureKHR(idevice->logical_device, iacceleration_structure->tlas, NULL);

  cgpu_destroy_buffer(device, iacceleration_structure->instances_buffer);
  cgpu_destroy_buffer(device, iacceleration_structure->vertex_buffer);
  cgpu_destroy_buffer(device, iacceleration_structure->index_buffer);
  cgpu_destroy_buffer(device, iacceleration_structure->tlas_buffer);
  cgpu_destroy_buffer(device, iacceleration_structure->blas_buffer);

  resource_store_free_handle(&iacceleration_structure_store, acceleration_structure.handle);
  return true;
}

bool cgpu_create_command_buffer(cgpu_device device,
                                cgpu_command_buffer* p_command_buffer)
{
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  p_command_buffer->handle = resource_store_create_handle(&icommand_buffer_store);

  cgpu_icommand_buffer* icommand_buffer;
  if (!cgpu_resolve_command_buffer(*p_command_buffer, &icommand_buffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  icommand_buffer->device.handle = device.handle;

  VkCommandBufferAllocateInfo cmdbuf_alloc_info;
  cmdbuf_alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cmdbuf_alloc_info.pNext = NULL;
  cmdbuf_alloc_info.commandPool = idevice->command_pool;
  cmdbuf_alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cmdbuf_alloc_info.commandBufferCount = 1;

  VkResult result = idevice->table.vkAllocateCommandBuffers(
    idevice->logical_device,
    &cmdbuf_alloc_info,
    &icommand_buffer->command_buffer
  );
  if (result != VK_SUCCESS) {
    resource_store_free_handle(&icommand_buffer_store, p_command_buffer->handle);
    CGPU_RETURN_ERROR("failed to allocate command buffer");
  }

  return true;
}

bool cgpu_destroy_command_buffer(cgpu_device device,
                                 cgpu_command_buffer command_buffer)
{
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  cgpu_icommand_buffer* icommand_buffer;
  if (!cgpu_resolve_command_buffer(command_buffer, &icommand_buffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  idevice->table.vkFreeCommandBuffers(
    idevice->logical_device,
    idevice->command_pool,
    1,
    &icommand_buffer->command_buffer
  );

  resource_store_free_handle(&icommand_buffer_store, command_buffer.handle);
  return true;
}

bool cgpu_begin_command_buffer(cgpu_command_buffer command_buffer)
{
  cgpu_icommand_buffer* icommand_buffer;
  if (!cgpu_resolve_command_buffer(command_buffer, &icommand_buffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(icommand_buffer->device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  VkCommandBufferBeginInfo command_buffer_begin_info;
  command_buffer_begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  command_buffer_begin_info.pNext = NULL;
  command_buffer_begin_info.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
  command_buffer_begin_info.pInheritanceInfo = NULL;

  VkResult result = idevice->table.vkBeginCommandBuffer(
    icommand_buffer->command_buffer,
    &command_buffer_begin_info
  );

  if (result != VK_SUCCESS) {
    CGPU_RETURN_ERROR("failed to begin command buffer");
  }
  return true;
}

bool cgpu_cmd_bind_pipeline(cgpu_command_buffer command_buffer,
                            cgpu_pipeline pipeline)
{
  cgpu_icommand_buffer* icommand_buffer;
  if (!cgpu_resolve_command_buffer(command_buffer, &icommand_buffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(icommand_buffer->device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  cgpu_ipipeline* ipipeline;
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

bool cgpu_cmd_transition_shader_image_layouts(cgpu_command_buffer command_buffer,
                                              cgpu_shader shader,
                                              uint32_t image_count,
                                              const cgpu_image_binding* p_images)
{
  cgpu_ishader* ishader;
  if (!cgpu_resolve_shader(shader, &ishader)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  cgpu_icommand_buffer* icommand_buffer;
  if (!cgpu_resolve_command_buffer(command_buffer, &icommand_buffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(icommand_buffer->device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  VkImageMemoryBarrier barriers[CGPU_MAX_IMAGE_MEMORY_BARRIERS];
  uint32_t barrier_count = 0;

  /* FIXME: this has quadratic complexity */
  const cgpu_shader_reflection* reflection = &ishader->reflection;
  for (uint32_t i = 0; i < reflection->binding_count; i++)
  {
    const cgpu_shader_reflection_binding* binding = &reflection->bindings[i];

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
      const cgpu_image_binding* image_binding = NULL;
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

      cgpu_iimage* iimage;
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

      if (barrier_count >= CGPU_MAX_IMAGE_MEMORY_BARRIERS) {
        CGPU_RETURN_ERROR_HARDCODED_LIMIT_REACHED;
      }

      VkImageMemoryBarrier* barrier = &barriers[barrier_count++];
      barrier->sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
      barrier->pNext = NULL;
      barrier->srcAccessMask = iimage->access_mask;
      barrier->dstAccessMask = access_mask;
      barrier->oldLayout = old_layout;
      barrier->newLayout = new_layout;
      barrier->srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier->dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier->image = iimage->image;
      barrier->subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      barrier->subresourceRange.baseMipLevel = 0;
      barrier->subresourceRange.levelCount = 1;
      barrier->subresourceRange.baseArrayLayer = 0;
      barrier->subresourceRange.layerCount = 1;

      iimage->access_mask = access_mask;
      iimage->layout = new_layout;
    }
  }

  if (barrier_count > 0)
  {
    idevice->table.vkCmdPipelineBarrier(
      icommand_buffer->command_buffer,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      0,
      0,
      NULL,
      0,
      NULL,
      barrier_count,
      barriers
    );
  }

  return true;
}

bool cgpu_cmd_update_bindings(cgpu_command_buffer command_buffer,
                              cgpu_pipeline pipeline,
                              const cgpu_bindings* bindings
)
{
  cgpu_icommand_buffer* icommand_buffer;
  if (!cgpu_resolve_command_buffer(command_buffer, &icommand_buffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(icommand_buffer->device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  cgpu_ipipeline* ipipeline;
  if (!cgpu_resolve_pipeline(pipeline, &ipipeline)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  VkDescriptorBufferInfo buffer_infos[CGPU_MAX_DESCRIPTOR_BUFFER_INFOS];
  uint32_t buffer_info_count = 0;
  VkDescriptorImageInfo image_infos[CGPU_MAX_DESCRIPTOR_IMAGE_INFOS];
  uint32_t image_info_count = 0;
  VkWriteDescriptorSetAccelerationStructureKHR as_infos[CGPU_MAX_DESCRIPTOR_AS_INFOS];
  uint32_t as_info_count = 0;

  VkWriteDescriptorSet write_descriptor_sets[CGPU_MAX_WRITE_DESCRIPTOR_SETS];
  uint32_t write_descriptor_set_count = 0;

  /* FIXME: this has a rather high complexity */
  for (uint32_t i = 0; i < ipipeline->descriptor_set_layout_binding_count; i++)
  {
    const VkDescriptorSetLayoutBinding* layout_binding = &ipipeline->descriptor_set_layout_bindings[i];

    if (write_descriptor_set_count >= CGPU_MAX_WRITE_DESCRIPTOR_SETS) {
      CGPU_RETURN_ERROR_HARDCODED_LIMIT_REACHED;
    }

    VkWriteDescriptorSet* write_descriptor_set = &write_descriptor_sets[write_descriptor_set_count++];
    write_descriptor_set->sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write_descriptor_set->pNext = NULL;
    write_descriptor_set->dstSet = ipipeline->descriptor_set;
    write_descriptor_set->dstBinding = layout_binding->binding;
    write_descriptor_set->dstArrayElement = 0;
    write_descriptor_set->descriptorCount = layout_binding->descriptorCount;
    write_descriptor_set->descriptorType = layout_binding->descriptorType;
    write_descriptor_set->pTexelBufferView = NULL;
    write_descriptor_set->pBufferInfo = NULL;
    write_descriptor_set->pImageInfo = NULL;

    for (uint32_t j = 0; j < layout_binding->descriptorCount; j++)
    {
      bool slotHandled = false;

      if (layout_binding->descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
      {
        for (uint32_t k = 0; k < bindings->buffer_count; ++k)
        {
          const cgpu_buffer_binding* buffer_binding = &bindings->p_buffers[k];

          if (buffer_binding->binding != layout_binding->binding || buffer_binding->index != j)
          {
            continue;
          }

          cgpu_ibuffer* ibuffer;
          cgpu_buffer buffer = buffer_binding->buffer;
          if (!cgpu_resolve_buffer(buffer, &ibuffer)) {
            CGPU_RETURN_ERROR_INVALID_HANDLE;
          }

          if ((buffer_binding->offset % idevice->properties.minStorageBufferOffsetAlignment) != 0) {
            CGPU_RETURN_ERROR("buffer binding offset not aligned");
          }

          if (image_info_count >= CGPU_MAX_DESCRIPTOR_BUFFER_INFOS) {
            CGPU_RETURN_ERROR_HARDCODED_LIMIT_REACHED;
          }

          VkDescriptorBufferInfo* buffer_info = &buffer_infos[buffer_info_count++];
          buffer_info->buffer = ibuffer->buffer;
          buffer_info->offset = buffer_binding->offset;
          buffer_info->range = (buffer_binding->size == CGPU_WHOLE_SIZE) ? (ibuffer->size - buffer_binding->offset) : buffer_binding->size;

          if (j == 0)
          {
            write_descriptor_set->pBufferInfo = buffer_info;
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
          const cgpu_image_binding* image_binding = &bindings->p_images[k];

          if (image_binding->binding != layout_binding->binding || image_binding->index != j)
          {
            continue;
          }

          cgpu_iimage* iimage;
          cgpu_image image = image_binding->image;
          if (!cgpu_resolve_image(image, &iimage)) {
            CGPU_RETURN_ERROR_INVALID_HANDLE;
          }

          if (image_info_count >= CGPU_MAX_DESCRIPTOR_IMAGE_INFOS) {
            CGPU_RETURN_ERROR_HARDCODED_LIMIT_REACHED;
          }

          VkDescriptorImageInfo* image_info = &image_infos[image_info_count++];
          image_info->sampler = VK_NULL_HANDLE;
          image_info->imageView = iimage->image_view;
          image_info->imageLayout = iimage->layout;

          if (j == 0)
          {
            write_descriptor_set->pImageInfo = image_info;
          }

          slotHandled = true;
          break;
        }
      }
      else if (layout_binding->descriptorType == VK_DESCRIPTOR_TYPE_SAMPLER)
      {
        for (uint32_t k = 0; k < bindings->sampler_count; k++)
        {
          const cgpu_sampler_binding* sampler_binding = &bindings->p_samplers[k];

          if (sampler_binding->binding != layout_binding->binding || sampler_binding->index != j)
          {
            continue;
          }

          cgpu_isampler* isampler;
          cgpu_sampler sampler = sampler_binding->sampler;
          if (!cgpu_resolve_sampler(sampler, &isampler)) {
            CGPU_RETURN_ERROR_INVALID_HANDLE;
          }

          if (image_info_count >= CGPU_MAX_DESCRIPTOR_IMAGE_INFOS) {
            CGPU_RETURN_ERROR_HARDCODED_LIMIT_REACHED;
          }

          VkDescriptorImageInfo* image_info = &image_infos[image_info_count++];
          image_info->sampler = isampler->sampler;
          image_info->imageView = VK_NULL_HANDLE;
          image_info->imageLayout = VK_IMAGE_LAYOUT_UNDEFINED;

          if (j == 0)
          {
            write_descriptor_set->pImageInfo = image_info;
          }

          slotHandled = true;
          break;
        }
      }
      else if (layout_binding->descriptorType == VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR)
      {
        for (uint32_t k = 0; k < bindings->as_count; ++k)
        {
          const cgpu_acceleration_structure_binding* as_binding = &bindings->p_ases[k];

          if (as_binding->binding != layout_binding->binding || as_binding->index != j)
          {
            continue;
          }

          cgpu_iacceleration_structure* iacceleration_structure;
          cgpu_acceleration_structure acceleration_structure = as_binding->as;
          if (!cgpu_resolve_acceleration_structure(acceleration_structure, &iacceleration_structure)) {
            CGPU_RETURN_ERROR_INVALID_HANDLE;
          }

          if (as_info_count >= CGPU_MAX_DESCRIPTOR_AS_INFOS) {
            CGPU_RETURN_ERROR_HARDCODED_LIMIT_REACHED;
          }

          VkWriteDescriptorSetAccelerationStructureKHR* as_info = &as_infos[as_info_count++];
          as_info->sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
          as_info->pNext = NULL;
          as_info->accelerationStructureCount = 1;
          as_info->pAccelerationStructures = &iacceleration_structure->tlas;

          if (j == 0)
          {
            write_descriptor_set->pNext = as_info;
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
  }

  idevice->table.vkUpdateDescriptorSets(
    idevice->logical_device,
    write_descriptor_set_count,
    write_descriptor_sets,
    0,
    NULL
  );

  return true;
}

bool cgpu_cmd_copy_buffer(cgpu_command_buffer command_buffer,
                          cgpu_buffer source_buffer,
                          uint64_t source_offset,
                          cgpu_buffer destination_buffer,
                          uint64_t destination_offset,
                          uint64_t size)
{
  cgpu_icommand_buffer* icommand_buffer;
  if (!cgpu_resolve_command_buffer(command_buffer, &icommand_buffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(icommand_buffer->device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  cgpu_ibuffer* isource_buffer;
  if (!cgpu_resolve_buffer(source_buffer, &isource_buffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  cgpu_ibuffer* idestination_buffer;
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

bool cgpu_cmd_copy_buffer_to_image(cgpu_command_buffer command_buffer,
                                   cgpu_buffer buffer,
                                   uint64_t buffer_offset,
                                   cgpu_image image)
{
  cgpu_icommand_buffer* icommand_buffer;
  if (!cgpu_resolve_command_buffer(command_buffer, &icommand_buffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(icommand_buffer->device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  cgpu_ibuffer* ibuffer;
  if (!cgpu_resolve_buffer(buffer, &ibuffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  cgpu_iimage* iimage;
  if (!cgpu_resolve_image(image, &iimage)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  if (iimage->layout != VK_IMAGE_LAYOUT_GENERAL)
  {
    VkAccessFlags access_mask = iimage->access_mask | VK_ACCESS_MEMORY_WRITE_BIT;
    VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL;

    VkImageMemoryBarrier barrier;
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.pNext = NULL;
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
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      0,
      0,
      NULL,
      0,
      NULL,
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

bool cgpu_cmd_push_constants(cgpu_command_buffer command_buffer,
                             cgpu_pipeline pipeline,
                             CgpuShaderStageFlags stage_flags,
                             uint32_t size,
                             const void* p_data)
{
  cgpu_icommand_buffer* icommand_buffer;
  if (!cgpu_resolve_command_buffer(command_buffer, &icommand_buffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(icommand_buffer->device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  cgpu_ipipeline* ipipeline;
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

bool cgpu_cmd_dispatch(cgpu_command_buffer command_buffer,
                       uint32_t dim_x,
                       uint32_t dim_y,
                       uint32_t dim_z)
{
  cgpu_icommand_buffer* icommand_buffer;
  if (!cgpu_resolve_command_buffer(command_buffer, &icommand_buffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  cgpu_idevice* idevice;
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

bool cgpu_cmd_pipeline_barrier(cgpu_command_buffer command_buffer,
                               uint32_t barrier_count,
                               const cgpu_memory_barrier* p_barriers,
                               uint32_t buffer_barrier_count,
                               const cgpu_buffer_memory_barrier* p_buffer_barriers,
                               uint32_t image_barrier_count,
                               const cgpu_image_memory_barrier* p_image_barriers)
{
  cgpu_icommand_buffer* icommand_buffer;
  if (!cgpu_resolve_command_buffer(command_buffer, &icommand_buffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(icommand_buffer->device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  if (barrier_count >= CGPU_MAX_MEMORY_BARRIERS ||
      buffer_barrier_count >= CGPU_MAX_BUFFER_MEMORY_BARRIERS ||
      image_barrier_count >= CGPU_MAX_IMAGE_MEMORY_BARRIERS)
  {
    CGPU_RETURN_ERROR_HARDCODED_LIMIT_REACHED;
  }

  VkMemoryBarrier vk_memory_barriers[CGPU_MAX_MEMORY_BARRIERS];

  for (uint32_t i = 0; i < barrier_count; ++i)
  {
    const cgpu_memory_barrier* b_cgpu = &p_barriers[i];
    VkMemoryBarrier* b_vk = &vk_memory_barriers[i];
    b_vk->sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    b_vk->pNext = NULL;
    b_vk->srcAccessMask = (VkAccessFlags) b_cgpu->src_access_flags;
    b_vk->dstAccessMask = (VkAccessFlags) b_cgpu->dst_access_flags;
  }

  VkBufferMemoryBarrier vk_buffer_memory_barriers[CGPU_MAX_BUFFER_MEMORY_BARRIERS];
  VkImageMemoryBarrier vk_image_memory_barriers[CGPU_MAX_IMAGE_MEMORY_BARRIERS];

  for (uint32_t i = 0; i < buffer_barrier_count; ++i)
  {
    const cgpu_buffer_memory_barrier* b_cgpu = &p_buffer_barriers[i];

    cgpu_ibuffer* ibuffer;
    if (!cgpu_resolve_buffer(b_cgpu->buffer, &ibuffer)) {
      CGPU_RETURN_ERROR_INVALID_HANDLE;
    }

    VkBufferMemoryBarrier* b_vk = &vk_buffer_memory_barriers[i];
    b_vk->sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    b_vk->pNext = NULL;
    b_vk->srcAccessMask = (VkAccessFlags) b_cgpu->src_access_flags;
    b_vk->dstAccessMask = (VkAccessFlags) b_cgpu->dst_access_flags;
    b_vk->srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b_vk->dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b_vk->buffer = ibuffer->buffer;
    b_vk->offset = b_cgpu->offset;
    b_vk->size = (b_cgpu->size == CGPU_WHOLE_SIZE) ? VK_WHOLE_SIZE : b_cgpu->size;
  }

  for (uint32_t i = 0; i < image_barrier_count; ++i)
  {
    const cgpu_image_memory_barrier* b_cgpu = &p_image_barriers[i];

    cgpu_iimage* iimage;
    if (!cgpu_resolve_image(b_cgpu->image, &iimage)) {
      CGPU_RETURN_ERROR_INVALID_HANDLE;
    }

    VkAccessFlags access_mask = (VkAccessFlags) b_cgpu->access_mask;

    VkImageMemoryBarrier* b_vk = &vk_image_memory_barriers[i];
    b_vk->sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b_vk->pNext = NULL;
    b_vk->srcAccessMask = iimage->access_mask;
    b_vk->dstAccessMask = access_mask;
    b_vk->oldLayout = iimage->layout;
    b_vk->newLayout = iimage->layout;
    b_vk->srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b_vk->dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b_vk->image = iimage->image;
    b_vk->subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    b_vk->subresourceRange.baseMipLevel = 0;
    b_vk->subresourceRange.levelCount = 1;
    b_vk->subresourceRange.baseArrayLayer = 0;
    b_vk->subresourceRange.layerCount = 1;

    iimage->access_mask = access_mask;
  }

  idevice->table.vkCmdPipelineBarrier(
    icommand_buffer->command_buffer,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
    0,
    barrier_count,
    vk_memory_barriers,
    buffer_barrier_count,
    vk_buffer_memory_barriers,
    image_barrier_count,
    vk_image_memory_barriers
  );

  return true;
}

bool cgpu_cmd_reset_timestamps(cgpu_command_buffer command_buffer,
                               uint32_t offset,
                               uint32_t count)
{
  cgpu_icommand_buffer* icommand_buffer;
  if (!cgpu_resolve_command_buffer(command_buffer, &icommand_buffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  cgpu_idevice* idevice;
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

bool cgpu_cmd_write_timestamp(cgpu_command_buffer command_buffer,
                              uint32_t timestamp_index)
{
  cgpu_icommand_buffer* icommand_buffer;
  if (!cgpu_resolve_command_buffer(command_buffer, &icommand_buffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(icommand_buffer->device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  idevice->table.vkCmdWriteTimestamp(
    icommand_buffer->command_buffer,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    idevice->timestamp_pool,
    timestamp_index
  );

  return true;
}

bool cgpu_cmd_copy_timestamps(cgpu_command_buffer command_buffer,
                              cgpu_buffer buffer,
                              uint32_t offset,
                              uint32_t count,
                              bool wait_until_available)
{
  uint32_t last_index = offset + count;
  if (last_index >= CGPU_MAX_TIMESTAMP_QUERIES) {
    CGPU_RETURN_ERROR_HARDCODED_LIMIT_REACHED;
  }

  cgpu_icommand_buffer* icommand_buffer;
  if (!cgpu_resolve_command_buffer(command_buffer, &icommand_buffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(icommand_buffer->device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  cgpu_ibuffer* ibuffer;
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

bool cgpu_end_command_buffer(cgpu_command_buffer command_buffer)
{
  cgpu_icommand_buffer* icommand_buffer;
  if (!cgpu_resolve_command_buffer(command_buffer, &icommand_buffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(icommand_buffer->device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  idevice->table.vkEndCommandBuffer(icommand_buffer->command_buffer);
  return true;
}

bool cgpu_create_fence(cgpu_device device,
                       cgpu_fence* p_fence)
{
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  p_fence->handle = resource_store_create_handle(&ifence_store);

  cgpu_ifence* ifence;
  if (!cgpu_resolve_fence(*p_fence, &ifence)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  VkFenceCreateInfo fence_create_info;
  fence_create_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fence_create_info.pNext = NULL;
  fence_create_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

  VkResult result = idevice->table.vkCreateFence(
    idevice->logical_device,
    &fence_create_info,
    NULL,
    &ifence->fence
  );

  if (result != VK_SUCCESS) {
    resource_store_free_handle(&ifence_store, p_fence->handle);
    CGPU_RETURN_ERROR("failed to create fence");
  }
  return true;
}

bool cgpu_destroy_fence(cgpu_device device,
                        cgpu_fence fence)
{
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  cgpu_ifence* ifence;
  if (!cgpu_resolve_fence(fence, &ifence)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  idevice->table.vkDestroyFence(
    idevice->logical_device,
    ifence->fence,
    NULL
  );
  resource_store_free_handle(&ifence_store, fence.handle);
  return true;
}

bool cgpu_reset_fence(cgpu_device device,
                      cgpu_fence fence)
{
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  cgpu_ifence* ifence;
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

bool cgpu_wait_for_fence(cgpu_device device, cgpu_fence fence)
{
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  cgpu_ifence* ifence;
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

bool cgpu_submit_command_buffer(cgpu_device device,
                                cgpu_command_buffer command_buffer,
                                cgpu_fence fence)
{
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  cgpu_icommand_buffer* icommand_buffer;
  if (!cgpu_resolve_command_buffer(command_buffer, &icommand_buffer)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  cgpu_ifence* ifence;
  if (!cgpu_resolve_fence(fence, &ifence)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }

  VkSubmitInfo submit_info;
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.pNext = NULL;
  submit_info.waitSemaphoreCount = 0;
  submit_info.pWaitSemaphores = NULL;
  submit_info.pWaitDstStageMask = NULL;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &icommand_buffer->command_buffer;
  submit_info.signalSemaphoreCount = 0;
  submit_info.pSignalSemaphores = NULL;

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

bool cgpu_flush_mapped_memory(cgpu_device device,
                              cgpu_buffer buffer,
                              uint64_t offset,
                              uint64_t size)
{
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  cgpu_ibuffer* ibuffer;
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

bool cgpu_invalidate_mapped_memory(cgpu_device device,
                                   cgpu_buffer buffer,
                                   uint64_t offset,
                                   uint64_t size)
{
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  cgpu_ibuffer* ibuffer;
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

bool cgpu_get_physical_device_features(cgpu_device device,
                                       cgpu_physical_device_features* p_features)
{
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  memcpy(p_features, &idevice->features, sizeof(cgpu_physical_device_features));
  return true;
}

bool cgpu_get_physical_device_properties(cgpu_device device,
                                         cgpu_physical_device_properties* p_properties)
{
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    CGPU_RETURN_ERROR_INVALID_HANDLE;
  }
  memcpy(p_properties, &idevice->properties, sizeof(cgpu_physical_device_properties));
  return true;
}
