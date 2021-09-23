#include "cgpu.h"
#include "resource_store.h"

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <volk.h>

#define MIN_VK_API_VERSION VK_API_VERSION_1_1

/* Array and pool allocation limits. */

#define MAX_PHYSICAL_DEVICES 32
#define MAX_DEVICE_EXTENSIONS 1024
#define MAX_QUEUE_FAMILIES 64
#define MAX_TIMESTAMP_QUERIES 32
#define MAX_DESCRIPTOR_SET_BINDINGS 128
#define MAX_DESCRIPTOR_BUFFER_INFOS 64
#define MAX_DESCRIPTOR_IMAGE_INFOS 64
#define MAX_WRITE_DESCRIPTOR_SETS 128
#define MAX_BUFFER_MEMORY_BARRIERS 64
#define MAX_IMAGE_MEMORY_BARRIERS 64
#define MAX_MEMORY_BARRIERS 128

/* Internal structures. */

typedef struct cgpu_iinstance {
  VkInstance instance;
} cgpu_iinstance;

typedef struct cgpu_idevice {
  VkDevice                    logical_device;
  VkPhysicalDevice            physical_device;
  VkQueue                     compute_queue;
  VkCommandPool               command_pool;
  VkQueryPool                 timestamp_pool;
  VkSampler                   sampler;
  struct VolkDeviceTable      table;
  cgpu_physical_device_limits limits;
} cgpu_idevice;

typedef struct cgpu_ibuffer {
  VkBuffer       buffer;
  VkDeviceMemory memory;
  uint64_t       size;
} cgpu_ibuffer;

typedef struct cgpu_iimage {
  VkImage        image;
  VkImageView    image_view;
  VkDeviceMemory memory;
  uint64_t       size;
} cgpu_iimage;

typedef struct cgpu_ipipeline {
  VkPipeline            pipeline;
  VkPipelineLayout      layout;
  VkDescriptorSetLayout descriptor_set_layout;
  VkDescriptorSet       descriptor_set;
  VkDescriptorPool      descriptor_pool;
} cgpu_ipipeline;

typedef struct cgpu_ishader {
  VkShaderModule module;
} cgpu_ishader;

typedef struct cgpu_ifence {
  VkFence fence;
} cgpu_ifence;

typedef struct cgpu_icommand_buffer {
  VkCommandBuffer command_buffer;
  cgpu_device     device;
} cgpu_icommand_buffer;

/* Handle and structure storage. */

static resource_store idevice_store;
static resource_store ishader_store;
static resource_store ibuffer_store;
static resource_store iimage_store;
static resource_store ipipeline_store;
static resource_store icommand_buffer_store;
static resource_store ifence_store;
static cgpu_iinstance iinstance;

/* Helper functions. */

#define CGPU_RESOLVE_HANDLE(RESOURCE_NAME, HANDLE_TYPE, IRESOURCE_TYPE, RESOURCE_STORE)   \
  CGPU_INLINE static bool cgpu_resolve_##RESOURCE_NAME(                                   \
    HANDLE_TYPE handle,                                                                   \
    IRESOURCE_TYPE** idata)                                                               \
  {                                                                                       \
    return resource_store_get(&RESOURCE_STORE, handle.handle, (void**) idata);            \
  }

CGPU_RESOLVE_HANDLE(        device,         cgpu_device,         cgpu_idevice,         idevice_store)
CGPU_RESOLVE_HANDLE(        buffer,         cgpu_buffer,         cgpu_ibuffer,         ibuffer_store)
CGPU_RESOLVE_HANDLE(        shader,         cgpu_shader,         cgpu_ishader,         ishader_store)
CGPU_RESOLVE_HANDLE(         image,          cgpu_image,          cgpu_iimage,          iimage_store)
CGPU_RESOLVE_HANDLE(      pipeline,       cgpu_pipeline,       cgpu_ipipeline,       ipipeline_store)
CGPU_RESOLVE_HANDLE(         fence,          cgpu_fence,          cgpu_ifence,          ifence_store)
CGPU_RESOLVE_HANDLE(command_buffer, cgpu_command_buffer, cgpu_icommand_buffer, icommand_buffer_store)

static VkMemoryPropertyFlags cgpu_translate_memory_properties(
  CgpuMemoryPropertyFlags memory_properties)
{
  VkMemoryPropertyFlags mem_flags = 0;
  if ((memory_properties & CGPU_MEMORY_PROPERTY_FLAG_DEVICE_LOCAL)
        == CGPU_MEMORY_PROPERTY_FLAG_DEVICE_LOCAL) {
    mem_flags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  }
  if ((memory_properties & CGPU_MEMORY_PROPERTY_FLAG_HOST_VISIBLE)
        == CGPU_MEMORY_PROPERTY_FLAG_HOST_VISIBLE) {
    mem_flags |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
  }
  if ((memory_properties & CGPU_MEMORY_PROPERTY_FLAG_HOST_COHERENT)
        == CGPU_MEMORY_PROPERTY_FLAG_HOST_COHERENT) {
    mem_flags |= VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  }
  if ((memory_properties & CGPU_MEMORY_PROPERTY_FLAG_HOST_CACHED)
        == CGPU_MEMORY_PROPERTY_FLAG_HOST_CACHED) {
    mem_flags |= VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
  }
  if ((memory_properties & CGPU_MEMORY_PROPERTY_FLAG_LAZILY_ALLOCATED)
        == CGPU_MEMORY_PROPERTY_FLAG_LAZILY_ALLOCATED) {
    mem_flags |= VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT;
  }
  if ((memory_properties & CGPU_MEMORY_PROPERTY_FLAG_PROTECTED)
        == CGPU_MEMORY_PROPERTY_FLAG_PROTECTED) {
    mem_flags |= VK_MEMORY_PROPERTY_PROTECTED_BIT;
  }
  return mem_flags;
}

static VkAccessFlags cgpu_translate_access_flags(
  CgpuMemoryAccessFlags flags)
{
  VkAccessFlags vk_flags = 0;
  if ((flags & CGPU_MEMORY_ACCESS_FLAG_UNIFORM_READ)
              == CGPU_MEMORY_ACCESS_FLAG_UNIFORM_READ) {
    vk_flags |= VK_ACCESS_UNIFORM_READ_BIT;
  }
  if ((flags & CGPU_MEMORY_ACCESS_FLAG_SHADER_READ)
              == CGPU_MEMORY_ACCESS_FLAG_SHADER_READ) {
    vk_flags |= VK_ACCESS_SHADER_READ_BIT;
  }
  if ((flags & CGPU_MEMORY_ACCESS_FLAG_SHADER_WRITE)
              == CGPU_MEMORY_ACCESS_FLAG_SHADER_WRITE) {
    vk_flags |= VK_ACCESS_SHADER_WRITE_BIT;
  }
  if ((flags & CGPU_MEMORY_ACCESS_FLAG_TRANSFER_READ)
              == CGPU_MEMORY_ACCESS_FLAG_TRANSFER_READ) {
    vk_flags |= VK_ACCESS_TRANSFER_READ_BIT;
  }
  if ((flags & CGPU_MEMORY_ACCESS_FLAG_TRANSFER_WRITE)
              == CGPU_MEMORY_ACCESS_FLAG_TRANSFER_WRITE) {
    vk_flags |= VK_ACCESS_TRANSFER_WRITE_BIT;
  }
  if ((flags & CGPU_MEMORY_ACCESS_FLAG_HOST_READ)
              == CGPU_MEMORY_ACCESS_FLAG_HOST_READ) {
    vk_flags |= VK_ACCESS_HOST_READ_BIT;
  }
  if ((flags & CGPU_MEMORY_ACCESS_FLAG_HOST_WRITE)
              == CGPU_MEMORY_ACCESS_FLAG_HOST_WRITE) {
    vk_flags |= VK_ACCESS_HOST_WRITE_BIT;
  }
  if ((flags & CGPU_MEMORY_ACCESS_FLAG_MEMORY_READ)
              == CGPU_MEMORY_ACCESS_FLAG_MEMORY_READ) {
    vk_flags |= VK_ACCESS_MEMORY_READ_BIT;
  }
  if ((flags & CGPU_MEMORY_ACCESS_FLAG_MEMORY_WRITE)
              == CGPU_MEMORY_ACCESS_FLAG_MEMORY_WRITE) {
    vk_flags |= VK_ACCESS_MEMORY_WRITE_BIT;
  }
  return vk_flags;
}

static CgpuSampleCountFlags cgpu_translate_sample_count_flags(
  VkSampleCountFlags vk_flags)
{
  CgpuSampleCountFlags flags = 0;
  if ((vk_flags & VK_SAMPLE_COUNT_1_BIT) == VK_SAMPLE_COUNT_1_BIT) {
    flags |= CGPU_SAMPLE_COUNT_FLAG_1;
  }
  if ((vk_flags & VK_SAMPLE_COUNT_2_BIT) == VK_SAMPLE_COUNT_2_BIT) {
    flags |= CGPU_SAMPLE_COUNT_FLAG_2;
  }
  if ((vk_flags & VK_SAMPLE_COUNT_4_BIT) == VK_SAMPLE_COUNT_4_BIT) {
    flags |= CGPU_SAMPLE_COUNT_FLAG_4;
  }
  if ((vk_flags & VK_SAMPLE_COUNT_8_BIT) == VK_SAMPLE_COUNT_8_BIT) {
    flags |= CGPU_SAMPLE_COUNT_FLAG_8;
  }
  if ((vk_flags & VK_SAMPLE_COUNT_16_BIT) == VK_SAMPLE_COUNT_16_BIT) {
    flags |= CGPU_SAMPLE_COUNT_FLAG_16;
  }
  if ((vk_flags & VK_SAMPLE_COUNT_32_BIT) == VK_SAMPLE_COUNT_32_BIT) {
    flags |= CGPU_SAMPLE_COUNT_FLAG_32;
  }
  if ((vk_flags & VK_SAMPLE_COUNT_64_BIT) == VK_SAMPLE_COUNT_64_BIT) {
    flags |= CGPU_SAMPLE_COUNT_FLAG_64;
  }
  return flags;
}

static cgpu_physical_device_limits cgpu_translate_physical_device_limits(
  VkPhysicalDeviceLimits vk_limits,
  VkPhysicalDeviceSubgroupProperties vk_subgroup_props)
{
  cgpu_physical_device_limits limits;
  limits.maxImageDimension1D = vk_limits.maxImageDimension1D;
  limits.maxImageDimension2D = vk_limits.maxImageDimension2D;
  limits.maxImageDimension3D = vk_limits.maxImageDimension3D;
  limits.maxImageDimensionCube = vk_limits.maxImageDimensionCube;
  limits.maxImageArrayLayers = vk_limits.maxImageArrayLayers;
  limits.maxTexelBufferElements = vk_limits.maxTexelBufferElements;
  limits.maxUniformBufferRange = vk_limits.maxUniformBufferRange;
  limits.maxStorageBufferRange = vk_limits.maxStorageBufferRange;
  limits.maxPushConstantsSize = vk_limits.maxPushConstantsSize;
  limits.maxMemoryAllocationCount = vk_limits.maxMemoryAllocationCount;
  limits.maxSamplerAllocationCount = vk_limits.maxSamplerAllocationCount;
  limits.bufferImageGranularity = vk_limits.bufferImageGranularity;
  limits.sparseAddressSpaceSize = vk_limits.sparseAddressSpaceSize;
  limits.maxBoundDescriptorSets = vk_limits.maxBoundDescriptorSets;
  limits.maxPerStageDescriptorSamplers = vk_limits.maxPerStageDescriptorSamplers;
  limits.maxPerStageDescriptorUniformBuffers = vk_limits.maxPerStageDescriptorUniformBuffers;
  limits.maxPerStageDescriptorStorageBuffers = vk_limits.maxPerStageDescriptorStorageBuffers;
  limits.maxPerStageDescriptorSampledImages = vk_limits.maxPerStageDescriptorSampledImages;
  limits.maxPerStageDescriptorStorageImages = vk_limits.maxPerStageDescriptorStorageImages;
  limits.maxPerStageDescriptorInputAttachments = vk_limits.maxPerStageDescriptorInputAttachments;
  limits.maxPerStageResources = vk_limits.maxPerStageResources;
  limits.maxDescriptorSetSamplers = vk_limits.maxDescriptorSetSamplers;
  limits.maxDescriptorSetUniformBuffers = vk_limits.maxDescriptorSetUniformBuffers;
  limits.maxDescriptorSetUniformBuffersDynamic = vk_limits.maxDescriptorSetUniformBuffersDynamic;
  limits.maxDescriptorSetStorageBuffers = vk_limits.maxDescriptorSetStorageBuffers;
  limits.maxDescriptorSetStorageBuffersDynamic = vk_limits.maxDescriptorSetStorageBuffersDynamic;
  limits.maxDescriptorSetSampledImages = vk_limits.maxDescriptorSetSampledImages;
  limits.maxDescriptorSetStorageImages = vk_limits.maxDescriptorSetStorageImages;
  limits.maxDescriptorSetInputAttachments = vk_limits.maxDescriptorSetInputAttachments;
  limits.maxVertexInputAttributes = vk_limits.maxVertexInputAttributes;
  limits.maxVertexInputBindings = vk_limits.maxVertexInputBindings;
  limits.maxVertexInputAttributeOffset = vk_limits.maxVertexInputAttributeOffset;
  limits.maxVertexInputBindingStride = vk_limits.maxVertexInputBindingStride;
  limits.maxVertexOutputComponents = vk_limits.maxVertexOutputComponents;
  limits.maxTessellationGenerationLevel = vk_limits.maxTessellationGenerationLevel;
  limits.maxTessellationPatchSize = vk_limits.maxTessellationPatchSize;
  limits.maxTessellationControlPerVertexInputComponents = vk_limits.maxTessellationControlPerVertexInputComponents;
  limits.maxTessellationControlPerVertexOutputComponents = vk_limits.maxTessellationControlPerVertexOutputComponents;
  limits.maxTessellationControlPerPatchOutputComponents = vk_limits.maxTessellationControlPerPatchOutputComponents;
  limits.maxTessellationControlTotalOutputComponents = vk_limits.maxTessellationControlTotalOutputComponents;
  limits.maxTessellationEvaluationInputComponents = vk_limits.maxTessellationEvaluationInputComponents;
  limits.maxTessellationEvaluationOutputComponents = vk_limits.maxTessellationEvaluationOutputComponents;
  limits.maxGeometryShaderInvocations = vk_limits.maxGeometryShaderInvocations;
  limits.maxGeometryInputComponents = vk_limits.maxGeometryInputComponents;
  limits.maxGeometryOutputComponents = vk_limits.maxGeometryOutputComponents;
  limits.maxGeometryOutputVertices = vk_limits.maxGeometryOutputVertices;
  limits.maxGeometryTotalOutputComponents = vk_limits.maxGeometryTotalOutputComponents;
  limits.maxFragmentInputComponents = vk_limits.maxFragmentInputComponents;
  limits.maxFragmentOutputAttachments = vk_limits.maxFragmentOutputAttachments;
  limits.maxFragmentDualSrcAttachments = vk_limits.maxFragmentDualSrcAttachments;
  limits.maxFragmentCombinedOutputResources = vk_limits.maxFragmentCombinedOutputResources;
  limits.maxComputeSharedMemorySize = vk_limits.maxComputeSharedMemorySize;
  limits.maxComputeWorkGroupCount[0] = vk_limits.maxComputeWorkGroupCount[0];
  limits.maxComputeWorkGroupCount[1] = vk_limits.maxComputeWorkGroupCount[1];
  limits.maxComputeWorkGroupCount[2] = vk_limits.maxComputeWorkGroupCount[2];
  limits.maxComputeWorkGroupInvocations = vk_limits.maxComputeWorkGroupInvocations;
  limits.maxComputeWorkGroupSize[0] = vk_limits.maxComputeWorkGroupSize[0];
  limits.maxComputeWorkGroupSize[1] = vk_limits.maxComputeWorkGroupSize[1];
  limits.maxComputeWorkGroupSize[2] = vk_limits.maxComputeWorkGroupSize[2];
  limits.subPixelPrecisionBits = vk_limits.subPixelPrecisionBits;
  limits.subTexelPrecisionBits = vk_limits.subTexelPrecisionBits;
  limits.mipmapPrecisionBits = vk_limits.mipmapPrecisionBits;
  limits.maxDrawIndexedIndexValue = vk_limits.maxDrawIndexedIndexValue;
  limits.maxDrawIndirectCount = vk_limits.maxDrawIndirectCount;
  limits.maxSamplerLodBias = vk_limits.maxSamplerLodBias;
  limits.maxSamplerAnisotropy = vk_limits.maxSamplerAnisotropy;
  limits.maxViewports = vk_limits.maxViewports;
  limits.maxViewportDimensions[0] = vk_limits.maxViewportDimensions[0];
  limits.maxViewportDimensions[1] = vk_limits.maxViewportDimensions[1];
  limits.viewportBoundsRange[0] = vk_limits.viewportBoundsRange[0];
  limits.viewportBoundsRange[1] = vk_limits.viewportBoundsRange[1];
  limits.viewportSubPixelBits = vk_limits.viewportSubPixelBits;
  limits.minMemoryMapAlignment = vk_limits.minMemoryMapAlignment;
  limits.minTexelBufferOffsetAlignment = vk_limits.minTexelBufferOffsetAlignment;
  limits.minUniformBufferOffsetAlignment = vk_limits.minUniformBufferOffsetAlignment;
  limits.minStorageBufferOffsetAlignment = vk_limits.minStorageBufferOffsetAlignment;
  limits.minTexelOffset = vk_limits.minTexelOffset;
  limits.maxTexelOffset = vk_limits.maxTexelOffset;
  limits.minTexelGatherOffset = vk_limits.minTexelGatherOffset;
  limits.maxTexelGatherOffset = vk_limits.maxTexelGatherOffset;
  limits.minInterpolationOffset = vk_limits.minInterpolationOffset;
  limits.maxInterpolationOffset = vk_limits.maxInterpolationOffset;
  limits.subPixelInterpolationOffsetBits = vk_limits.subPixelInterpolationOffsetBits;
  limits.maxFramebufferWidth = vk_limits.maxFramebufferWidth;
  limits.maxFramebufferHeight = vk_limits.maxFramebufferHeight;
  limits.maxFramebufferLayers = vk_limits.maxFramebufferLayers;
  limits.framebufferColorSampleCounts =
      cgpu_translate_sample_count_flags(vk_limits.framebufferColorSampleCounts);
  limits.framebufferDepthSampleCounts =
      cgpu_translate_sample_count_flags(vk_limits.framebufferDepthSampleCounts);
  limits.framebufferStencilSampleCounts = cgpu_translate_sample_count_flags(
      vk_limits.framebufferStencilSampleCounts);
  limits.framebufferNoAttachmentsSampleCounts =
      cgpu_translate_sample_count_flags(
          vk_limits.framebufferNoAttachmentsSampleCounts);
  limits.maxColorAttachments = vk_limits.maxColorAttachments;
  limits.sampledImageColorSampleCounts = cgpu_translate_sample_count_flags(
      vk_limits.sampledImageColorSampleCounts);
  limits.sampledImageIntegerSampleCounts = cgpu_translate_sample_count_flags(
      vk_limits.sampledImageIntegerSampleCounts);
  limits.sampledImageDepthSampleCounts = cgpu_translate_sample_count_flags(
      vk_limits.sampledImageDepthSampleCounts);
  limits.sampledImageStencilSampleCounts = cgpu_translate_sample_count_flags(
      vk_limits.sampledImageStencilSampleCounts);
  limits.storageImageSampleCounts =
      cgpu_translate_sample_count_flags(vk_limits.storageImageSampleCounts);
  limits.maxSampleMaskWords = vk_limits.maxSampleMaskWords;
  limits.timestampComputeAndGraphics = vk_limits.timestampComputeAndGraphics;
  limits.timestampPeriod = vk_limits.timestampPeriod;
  limits.maxClipDistances = vk_limits.maxClipDistances;
  limits.maxCullDistances = vk_limits.maxCullDistances;
  limits.maxCombinedClipAndCullDistances = vk_limits.maxCombinedClipAndCullDistances;
  limits.discreteQueuePriorities = vk_limits.discreteQueuePriorities;
  limits.pointSizeGranularity = vk_limits.pointSizeGranularity;
  limits.lineWidthGranularity = vk_limits.lineWidthGranularity;
  limits.strictLines = vk_limits.strictLines;
  limits.standardSampleLocations = vk_limits.standardSampleLocations;
  limits.optimalBufferCopyOffsetAlignment = vk_limits.optimalBufferCopyOffsetAlignment;
  limits.optimalBufferCopyRowPitchAlignment = vk_limits.optimalBufferCopyRowPitchAlignment;
  limits.nonCoherentAtomSize = vk_limits.nonCoherentAtomSize;
  limits.subgroupSize = vk_subgroup_props.subgroupSize;
  return limits;
}

static VkFormat cgpu_translate_image_format(
  CgpuImageFormat image_format)
{
  if ((image_format & CGPU_IMAGE_FORMAT_UNDEFINED)
        == CGPU_IMAGE_FORMAT_UNDEFINED) { return VK_FORMAT_UNDEFINED; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R4G4_UNORM_PACK8)
        == CGPU_IMAGE_FORMAT_R4G4_UNORM_PACK8) { return VK_FORMAT_R4G4_UNORM_PACK8; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R4G4B4A4_UNORM_PACK16)
        == CGPU_IMAGE_FORMAT_R4G4B4A4_UNORM_PACK16) { return VK_FORMAT_R4G4B4A4_UNORM_PACK16; }
  else if ((image_format & CGPU_IMAGE_FORMAT_B4G4R4A4_UNORM_PACK16)
        == CGPU_IMAGE_FORMAT_B4G4R4A4_UNORM_PACK16) { return VK_FORMAT_B4G4R4A4_UNORM_PACK16; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R5G6B5_UNORM_PACK16)
        == CGPU_IMAGE_FORMAT_R5G6B5_UNORM_PACK16) { return VK_FORMAT_R5G6B5_UNORM_PACK16; }
  else if ((image_format & CGPU_IMAGE_FORMAT_B5G6R5_UNORM_PACK16)
        == CGPU_IMAGE_FORMAT_B5G6R5_UNORM_PACK16) { return VK_FORMAT_B5G6R5_UNORM_PACK16; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R5G5B5A1_UNORM_PACK16)
        == CGPU_IMAGE_FORMAT_R5G5B5A1_UNORM_PACK16) { return VK_FORMAT_R5G5B5A1_UNORM_PACK16; }
  else if ((image_format & CGPU_IMAGE_FORMAT_B5G5R5A1_UNORM_PACK16)
        == CGPU_IMAGE_FORMAT_B5G5R5A1_UNORM_PACK16) { return VK_FORMAT_B5G5R5A1_UNORM_PACK16; }
  else if ((image_format & CGPU_IMAGE_FORMAT_A1R5G5B5_UNORM_PACK16)
        == CGPU_IMAGE_FORMAT_A1R5G5B5_UNORM_PACK16) { return VK_FORMAT_A1R5G5B5_UNORM_PACK16; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R8_UNORM)
        == CGPU_IMAGE_FORMAT_R8_UNORM) { return VK_FORMAT_R8_UNORM; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R8_SNORM)
        == CGPU_IMAGE_FORMAT_R8_SNORM) { return VK_FORMAT_R8_SNORM; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R8_USCALED)
        == CGPU_IMAGE_FORMAT_R8_USCALED) { return VK_FORMAT_R8_USCALED; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R8_SSCALED)
        == CGPU_IMAGE_FORMAT_R8_SSCALED) { return VK_FORMAT_R8_SSCALED; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R8_UINT)
        == CGPU_IMAGE_FORMAT_R8_UINT) { return VK_FORMAT_R8_UINT; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R8_SINT)
        == CGPU_IMAGE_FORMAT_R8_SINT) { return VK_FORMAT_R8_SINT; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R8_SRGB)
        == CGPU_IMAGE_FORMAT_R8_SRGB) { return VK_FORMAT_R8_SRGB; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R8G8_UNORM)
        == CGPU_IMAGE_FORMAT_R8G8_UNORM) { return VK_FORMAT_R8G8_UNORM; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R8G8_SNORM)
        == CGPU_IMAGE_FORMAT_R8G8_SNORM) { return VK_FORMAT_R8G8_SNORM; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R8G8_USCALED)
        == CGPU_IMAGE_FORMAT_R8G8_USCALED) { return VK_FORMAT_R8G8_USCALED; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R8G8_SSCALED)
        == CGPU_IMAGE_FORMAT_R8G8_SSCALED) { return VK_FORMAT_R8G8_SSCALED; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R8G8_UINT)
        == CGPU_IMAGE_FORMAT_R8G8_UINT) { return VK_FORMAT_R8G8_UINT; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R8G8_SINT)
        == CGPU_IMAGE_FORMAT_R8G8_SINT) { return VK_FORMAT_R8G8_SINT; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R8G8_SRGB)
        == CGPU_IMAGE_FORMAT_R8G8_SRGB) { return VK_FORMAT_R8G8_SRGB; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R8G8B8_UNORM)
        == CGPU_IMAGE_FORMAT_R8G8B8_UNORM) { return VK_FORMAT_R8G8B8_UNORM; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R8G8B8_SNORM)
        == CGPU_IMAGE_FORMAT_R8G8B8_SNORM) { return VK_FORMAT_R8G8B8_SNORM; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R8G8B8_USCALED)
        == CGPU_IMAGE_FORMAT_R8G8B8_USCALED) { return VK_FORMAT_R8G8B8_USCALED; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R8G8B8_SSCALED)
        == CGPU_IMAGE_FORMAT_R8G8B8_SSCALED) { return VK_FORMAT_R8G8B8_SSCALED; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R8G8B8_UINT)
        == CGPU_IMAGE_FORMAT_R8G8B8_UINT) { return VK_FORMAT_R8G8B8_UINT; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R8G8B8_SINT)
        == CGPU_IMAGE_FORMAT_R8G8B8_SINT) { return VK_FORMAT_R8G8B8_SINT; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R8G8B8_SRGB)
        == CGPU_IMAGE_FORMAT_R8G8B8_SRGB) { return VK_FORMAT_R8G8B8_SRGB; }
  else if ((image_format & CGPU_IMAGE_FORMAT_B8G8R8_UNORM)
        == CGPU_IMAGE_FORMAT_B8G8R8_UNORM) { return VK_FORMAT_B8G8R8_UNORM; }
  else if ((image_format & CGPU_IMAGE_FORMAT_B8G8R8_SNORM)
        == CGPU_IMAGE_FORMAT_B8G8R8_SNORM) { return VK_FORMAT_B8G8R8_SNORM; }
  else if ((image_format & CGPU_IMAGE_FORMAT_B8G8R8_USCALED)
        == CGPU_IMAGE_FORMAT_B8G8R8_USCALED) { return VK_FORMAT_B8G8R8_USCALED; }
  else if ((image_format & CGPU_IMAGE_FORMAT_B8G8R8_SSCALED)
        == CGPU_IMAGE_FORMAT_B8G8R8_SSCALED) { return VK_FORMAT_B8G8R8_SSCALED; }
  else if ((image_format & CGPU_IMAGE_FORMAT_B8G8R8_UINT)
        == CGPU_IMAGE_FORMAT_B8G8R8_UINT) { return VK_FORMAT_B8G8R8_UINT; }
  else if ((image_format & CGPU_IMAGE_FORMAT_B8G8R8_SINT)
        == CGPU_IMAGE_FORMAT_B8G8R8_SINT) { return VK_FORMAT_B8G8R8_SINT; }
  else if ((image_format & CGPU_IMAGE_FORMAT_B8G8R8_SRGB)
        == CGPU_IMAGE_FORMAT_B8G8R8_SRGB) { return VK_FORMAT_B8G8R8_SRGB; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R8G8B8A8_UNORM)
        == CGPU_IMAGE_FORMAT_R8G8B8A8_UNORM) { return VK_FORMAT_R8G8B8A8_UNORM; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R8G8B8A8_SNORM)
        == CGPU_IMAGE_FORMAT_R8G8B8A8_SNORM) { return VK_FORMAT_R8G8B8A8_SNORM; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R8G8B8A8_USCALED)
        == CGPU_IMAGE_FORMAT_R8G8B8A8_USCALED) { return VK_FORMAT_R8G8B8A8_USCALED; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R8G8B8A8_SSCALED)
        == CGPU_IMAGE_FORMAT_R8G8B8A8_SSCALED) { return VK_FORMAT_R8G8B8A8_SSCALED; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R8G8B8A8_UINT)
        == CGPU_IMAGE_FORMAT_R8G8B8A8_UINT) { return VK_FORMAT_R8G8B8A8_UINT; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R8G8B8A8_SINT)
        == CGPU_IMAGE_FORMAT_R8G8B8A8_SINT) { return VK_FORMAT_R8G8B8A8_SINT; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R8G8B8A8_SRGB)
        == CGPU_IMAGE_FORMAT_R8G8B8A8_SRGB) { return VK_FORMAT_R8G8B8A8_SRGB; }
  else if ((image_format & CGPU_IMAGE_FORMAT_B8G8R8A8_UNORM)
        == CGPU_IMAGE_FORMAT_B8G8R8A8_UNORM) { return VK_FORMAT_B8G8R8A8_UNORM; }
  else if ((image_format & CGPU_IMAGE_FORMAT_B8G8R8A8_SNORM)
        == CGPU_IMAGE_FORMAT_B8G8R8A8_SNORM) { return VK_FORMAT_B8G8R8A8_SNORM; }
  else if ((image_format & CGPU_IMAGE_FORMAT_B8G8R8A8_USCALED)
        == CGPU_IMAGE_FORMAT_B8G8R8A8_USCALED) { return VK_FORMAT_B8G8R8A8_USCALED; }
  else if ((image_format & CGPU_IMAGE_FORMAT_B8G8R8A8_SSCALED)
        == CGPU_IMAGE_FORMAT_B8G8R8A8_SSCALED) { return VK_FORMAT_B8G8R8A8_SSCALED; }
  else if ((image_format & CGPU_IMAGE_FORMAT_B8G8R8A8_UINT)
        == CGPU_IMAGE_FORMAT_B8G8R8A8_UINT) { return VK_FORMAT_B8G8R8A8_UINT; }
  else if ((image_format & CGPU_IMAGE_FORMAT_B8G8R8A8_SINT)
        == CGPU_IMAGE_FORMAT_B8G8R8A8_SINT) { return VK_FORMAT_B8G8R8A8_SINT; }
  else if ((image_format & CGPU_IMAGE_FORMAT_B8G8R8A8_SRGB)
        == CGPU_IMAGE_FORMAT_B8G8R8A8_SRGB) { return VK_FORMAT_B8G8R8A8_SRGB; }
  else if ((image_format & CGPU_IMAGE_FORMAT_A8B8G8R8_UNORM_PACK32)
        == CGPU_IMAGE_FORMAT_A8B8G8R8_UNORM_PACK32) { return VK_FORMAT_A8B8G8R8_UNORM_PACK32; }
  else if ((image_format & CGPU_IMAGE_FORMAT_A8B8G8R8_SNORM_PACK32)
        == CGPU_IMAGE_FORMAT_A8B8G8R8_SNORM_PACK32) { return VK_FORMAT_A8B8G8R8_SNORM_PACK32; }
  else if ((image_format & CGPU_IMAGE_FORMAT_A8B8G8R8_USCALED_PACK32)
        == CGPU_IMAGE_FORMAT_A8B8G8R8_USCALED_PACK32) { return VK_FORMAT_A8B8G8R8_USCALED_PACK32; }
  else if ((image_format & CGPU_IMAGE_FORMAT_A8B8G8R8_SSCALED_PACK32)
        == CGPU_IMAGE_FORMAT_A8B8G8R8_SSCALED_PACK32) { return VK_FORMAT_A8B8G8R8_SSCALED_PACK32; }
  else if ((image_format & CGPU_IMAGE_FORMAT_A8B8G8R8_UINT_PACK32)
        == CGPU_IMAGE_FORMAT_A8B8G8R8_UINT_PACK32) { return VK_FORMAT_A8B8G8R8_UINT_PACK32; }
  else if ((image_format & CGPU_IMAGE_FORMAT_A8B8G8R8_SINT_PACK32)
        == CGPU_IMAGE_FORMAT_A8B8G8R8_SINT_PACK32) { return VK_FORMAT_A8B8G8R8_SINT_PACK32; }
  else if ((image_format & CGPU_IMAGE_FORMAT_A8B8G8R8_SRGB_PACK32)
        == CGPU_IMAGE_FORMAT_A8B8G8R8_SRGB_PACK32) { return VK_FORMAT_A8B8G8R8_SRGB_PACK32; }
  else if ((image_format & CGPU_IMAGE_FORMAT_A2R10G10B10_UNORM_PACK32)
        == CGPU_IMAGE_FORMAT_A2R10G10B10_UNORM_PACK32) { return VK_FORMAT_A2R10G10B10_UNORM_PACK32; }
  else if ((image_format & CGPU_IMAGE_FORMAT_A2R10G10B10_SNORM_PACK32)
        == CGPU_IMAGE_FORMAT_A2R10G10B10_SNORM_PACK32) { return VK_FORMAT_A2R10G10B10_SNORM_PACK32; }
  else if ((image_format & CGPU_IMAGE_FORMAT_A2R10G10B10_USCALED_PACK32)
        == CGPU_IMAGE_FORMAT_A2R10G10B10_USCALED_PACK32) { return VK_FORMAT_A2R10G10B10_USCALED_PACK32; }
  else if ((image_format & CGPU_IMAGE_FORMAT_A2R10G10B10_SSCALED_PACK32)
        == CGPU_IMAGE_FORMAT_A2R10G10B10_SSCALED_PACK32) { return VK_FORMAT_A2R10G10B10_SSCALED_PACK32; }
  else if ((image_format & CGPU_IMAGE_FORMAT_A2R10G10B10_UINT_PACK32)
        == CGPU_IMAGE_FORMAT_A2R10G10B10_UINT_PACK32) { return VK_FORMAT_A2R10G10B10_UINT_PACK32; }
  else if ((image_format & CGPU_IMAGE_FORMAT_A2R10G10B10_SINT_PACK32)
        == CGPU_IMAGE_FORMAT_A2R10G10B10_SINT_PACK32) { return VK_FORMAT_A2R10G10B10_SINT_PACK32; }
  else if ((image_format & CGPU_IMAGE_FORMAT_A2B10G10R10_UNORM_PACK32)
        == CGPU_IMAGE_FORMAT_A2B10G10R10_UNORM_PACK32) { return VK_FORMAT_A2B10G10R10_UNORM_PACK32; }
  else if ((image_format & CGPU_IMAGE_FORMAT_A2B10G10R10_SNORM_PACK32)
        == CGPU_IMAGE_FORMAT_A2B10G10R10_SNORM_PACK32) { return VK_FORMAT_A2B10G10R10_SNORM_PACK32; }
  else if ((image_format & CGPU_IMAGE_FORMAT_A2B10G10R10_USCALED_PACK32)
        == CGPU_IMAGE_FORMAT_A2B10G10R10_USCALED_PACK32) { return VK_FORMAT_A2B10G10R10_USCALED_PACK32; }
  else if ((image_format & CGPU_IMAGE_FORMAT_A2B10G10R10_SSCALED_PACK32)
        == CGPU_IMAGE_FORMAT_A2B10G10R10_SSCALED_PACK32) { return VK_FORMAT_A2B10G10R10_SSCALED_PACK32; }
  else if ((image_format & CGPU_IMAGE_FORMAT_A2B10G10R10_UINT_PACK32)
        == CGPU_IMAGE_FORMAT_A2B10G10R10_UINT_PACK32) { return VK_FORMAT_A2B10G10R10_UINT_PACK32; }
  else if ((image_format & CGPU_IMAGE_FORMAT_A2B10G10R10_SINT_PACK32)
        == CGPU_IMAGE_FORMAT_A2B10G10R10_SINT_PACK32) { return VK_FORMAT_A2B10G10R10_SINT_PACK32; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R16_UNORM)
        == CGPU_IMAGE_FORMAT_R16_UNORM) { return VK_FORMAT_R16_UNORM; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R16_SNORM)
        == CGPU_IMAGE_FORMAT_R16_SNORM) { return VK_FORMAT_R16_SNORM; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R16_USCALED)
        == CGPU_IMAGE_FORMAT_R16_USCALED) { return VK_FORMAT_R16_USCALED; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R16_SSCALED)
        == CGPU_IMAGE_FORMAT_R16_SSCALED) { return VK_FORMAT_R16_SSCALED; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R16_UINT)
        == CGPU_IMAGE_FORMAT_R16_UINT) { return VK_FORMAT_R16_UINT; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R16_SINT)
        == CGPU_IMAGE_FORMAT_R16_SINT) { return VK_FORMAT_R16_SINT; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R16_SFLOAT)
        == CGPU_IMAGE_FORMAT_R16_SFLOAT) { return VK_FORMAT_R16_SFLOAT; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R16G16_UNORM)
        == CGPU_IMAGE_FORMAT_R16G16_UNORM) { return VK_FORMAT_R16G16_UNORM; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R16G16_SNORM)
        == CGPU_IMAGE_FORMAT_R16G16_SNORM) { return VK_FORMAT_R16G16_SNORM; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R16G16_USCALED)
        == CGPU_IMAGE_FORMAT_R16G16_USCALED) { return VK_FORMAT_R16G16_USCALED; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R16G16_SSCALED)
        == CGPU_IMAGE_FORMAT_R16G16_SSCALED) { return VK_FORMAT_R16G16_SSCALED; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R16G16_UINT)
        == CGPU_IMAGE_FORMAT_R16G16_UINT) { return VK_FORMAT_R16G16_UINT; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R16G16_SINT)
        == CGPU_IMAGE_FORMAT_R16G16_SINT) { return VK_FORMAT_R16G16_SINT; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R16G16_SFLOAT)
        == CGPU_IMAGE_FORMAT_R16G16_SFLOAT) { return VK_FORMAT_R16G16_SFLOAT; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R16G16B16_UNORM)
        == CGPU_IMAGE_FORMAT_R16G16B16_UNORM) { return VK_FORMAT_R16G16B16_UNORM; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R16G16B16_SNORM)
        == CGPU_IMAGE_FORMAT_R16G16B16_SNORM) { return VK_FORMAT_R16G16B16_SNORM; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R16G16B16_USCALED)
        == CGPU_IMAGE_FORMAT_R16G16B16_USCALED) { return VK_FORMAT_R16G16B16_USCALED; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R16G16B16_SSCALED)
        == CGPU_IMAGE_FORMAT_R16G16B16_SSCALED) { return VK_FORMAT_R16G16B16_SSCALED; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R16G16B16_UINT)
        == CGPU_IMAGE_FORMAT_R16G16B16_UINT) { return VK_FORMAT_R16G16B16_UINT; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R16G16B16_SINT)
        == CGPU_IMAGE_FORMAT_R16G16B16_SINT) { return VK_FORMAT_R16G16B16_SINT; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R16G16B16_SFLOAT)
        == CGPU_IMAGE_FORMAT_R16G16B16_SFLOAT) { return VK_FORMAT_R16G16B16_SFLOAT; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R16G16B16A16_UNORM)
        == CGPU_IMAGE_FORMAT_R16G16B16A16_UNORM) { return VK_FORMAT_R16G16B16A16_UNORM; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R16G16B16A16_SNORM)
        == CGPU_IMAGE_FORMAT_R16G16B16A16_SNORM) { return VK_FORMAT_R16G16B16A16_SNORM; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R16G16B16A16_USCALED)
        == CGPU_IMAGE_FORMAT_R16G16B16A16_USCALED) { return VK_FORMAT_R16G16B16A16_USCALED; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R16G16B16A16_SSCALED)
        == CGPU_IMAGE_FORMAT_R16G16B16A16_SSCALED) { return VK_FORMAT_R16G16B16A16_SSCALED; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R16G16B16A16_UINT)
        == CGPU_IMAGE_FORMAT_R16G16B16A16_UINT) { return VK_FORMAT_R16G16B16A16_UINT; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R16G16B16A16_SINT)
        == CGPU_IMAGE_FORMAT_R16G16B16A16_SINT) { return VK_FORMAT_R16G16B16A16_SINT; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R16G16B16A16_SFLOAT)
        == CGPU_IMAGE_FORMAT_R16G16B16A16_SFLOAT) { return VK_FORMAT_R16G16B16A16_SFLOAT; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R32_UINT)
        == CGPU_IMAGE_FORMAT_R32_UINT) { return VK_FORMAT_R32_UINT; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R32_SINT)
        == CGPU_IMAGE_FORMAT_R32_SINT) { return VK_FORMAT_R32_SINT; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R32_SFLOAT)
        == CGPU_IMAGE_FORMAT_R32_SFLOAT) { return VK_FORMAT_R32_SFLOAT; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R32G32_UINT)
        == CGPU_IMAGE_FORMAT_R32G32_UINT) { return VK_FORMAT_R32G32_UINT; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R32G32_SINT)
        == CGPU_IMAGE_FORMAT_R32G32_SINT) { return VK_FORMAT_R32G32_SINT; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R32G32_SFLOAT)
        == CGPU_IMAGE_FORMAT_R32G32_SFLOAT) { return VK_FORMAT_R32G32_SFLOAT; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R32G32B32_UINT)
        == CGPU_IMAGE_FORMAT_R32G32B32_UINT) { return VK_FORMAT_R32G32B32_UINT; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R32G32B32_SINT)
        == CGPU_IMAGE_FORMAT_R32G32B32_SINT) { return VK_FORMAT_R32G32B32_SINT; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R32G32B32_SFLOAT)
        == CGPU_IMAGE_FORMAT_R32G32B32_SFLOAT) { return VK_FORMAT_R32G32B32_SFLOAT; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R32G32B32A32_UINT)
        == CGPU_IMAGE_FORMAT_R32G32B32A32_UINT) { return VK_FORMAT_R32G32B32A32_UINT; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R32G32B32A32_SINT)
        == CGPU_IMAGE_FORMAT_R32G32B32A32_SINT) { return VK_FORMAT_R32G32B32A32_SINT; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R32G32B32A32_SFLOAT)
        == CGPU_IMAGE_FORMAT_R32G32B32A32_SFLOAT) { return VK_FORMAT_R32G32B32A32_SFLOAT; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R64_UINT)
        == CGPU_IMAGE_FORMAT_R64_UINT) { return VK_FORMAT_R64_UINT; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R64_SINT)
        == CGPU_IMAGE_FORMAT_R64_SINT) { return VK_FORMAT_R64_SINT; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R64_SFLOAT)
        == CGPU_IMAGE_FORMAT_R64_SFLOAT) { return VK_FORMAT_R64_SFLOAT; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R64G64_UINT)
        == CGPU_IMAGE_FORMAT_R64G64_UINT) { return VK_FORMAT_R64G64_UINT; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R64G64_SINT)
        == CGPU_IMAGE_FORMAT_R64G64_SINT) { return VK_FORMAT_R64G64_SINT; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R64G64_SFLOAT)
        == CGPU_IMAGE_FORMAT_R64G64_SFLOAT) { return VK_FORMAT_R64G64_SFLOAT; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R64G64B64_UINT)
        == CGPU_IMAGE_FORMAT_R64G64B64_UINT) { return VK_FORMAT_R64G64B64_UINT; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R64G64B64_SINT)
        == CGPU_IMAGE_FORMAT_R64G64B64_SINT) { return VK_FORMAT_R64G64B64_SINT; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R64G64B64_SFLOAT)
        == CGPU_IMAGE_FORMAT_R64G64B64_SFLOAT) { return VK_FORMAT_R64G64B64_SFLOAT; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R64G64B64A64_UINT)
        == CGPU_IMAGE_FORMAT_R64G64B64A64_UINT) { return VK_FORMAT_R64G64B64A64_UINT; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R64G64B64A64_SINT)
        == CGPU_IMAGE_FORMAT_R64G64B64A64_SINT) { return VK_FORMAT_R64G64B64A64_SINT; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R64G64B64A64_SFLOAT)
        == CGPU_IMAGE_FORMAT_R64G64B64A64_SFLOAT) { return VK_FORMAT_R64G64B64A64_SFLOAT; }
  else if ((image_format & CGPU_IMAGE_FORMAT_B10G11R11_UFLOAT_PACK32)
        == CGPU_IMAGE_FORMAT_B10G11R11_UFLOAT_PACK32) { return VK_FORMAT_B10G11R11_UFLOAT_PACK32; }
  else if ((image_format & CGPU_IMAGE_FORMAT_E5B9G9R9_UFLOAT_PACK32)
        == CGPU_IMAGE_FORMAT_E5B9G9R9_UFLOAT_PACK32) { return VK_FORMAT_E5B9G9R9_UFLOAT_PACK32; }
  else if ((image_format & CGPU_IMAGE_FORMAT_D16_UNORM)
        == CGPU_IMAGE_FORMAT_D16_UNORM) { return VK_FORMAT_D16_UNORM; }
  else if ((image_format & CGPU_IMAGE_FORMAT_X8_D24_UNORM_PACK32)
        == CGPU_IMAGE_FORMAT_X8_D24_UNORM_PACK32) { return VK_FORMAT_X8_D24_UNORM_PACK32; }
  else if ((image_format & CGPU_IMAGE_FORMAT_D32_SFLOAT)
        == CGPU_IMAGE_FORMAT_D32_SFLOAT) { return VK_FORMAT_D32_SFLOAT; }
  else if ((image_format & CGPU_IMAGE_FORMAT_S8_UINT)
        == CGPU_IMAGE_FORMAT_S8_UINT) { return VK_FORMAT_S8_UINT; }
  else if ((image_format & CGPU_IMAGE_FORMAT_D16_UNORM_S8_UINT)
        == CGPU_IMAGE_FORMAT_D16_UNORM_S8_UINT) { return VK_FORMAT_D16_UNORM_S8_UINT; }
  else if ((image_format & CGPU_IMAGE_FORMAT_D24_UNORM_S8_UINT)
        == CGPU_IMAGE_FORMAT_D24_UNORM_S8_UINT) { return VK_FORMAT_D24_UNORM_S8_UINT; }
  else if ((image_format & CGPU_IMAGE_FORMAT_D32_SFLOAT_S8_UINT)
        == CGPU_IMAGE_FORMAT_D32_SFLOAT_S8_UINT) { return VK_FORMAT_D32_SFLOAT_S8_UINT; }
  else if ((image_format & CGPU_IMAGE_FORMAT_BC1_RGB_UNORM_BLOCK)
        == CGPU_IMAGE_FORMAT_BC1_RGB_UNORM_BLOCK) { return VK_FORMAT_BC1_RGB_UNORM_BLOCK; }
  else if ((image_format & CGPU_IMAGE_FORMAT_BC1_RGB_SRGB_BLOCK)
        == CGPU_IMAGE_FORMAT_BC1_RGB_SRGB_BLOCK) { return VK_FORMAT_BC1_RGB_SRGB_BLOCK; }
  else if ((image_format & CGPU_IMAGE_FORMAT_BC1_RGBA_UNORM_BLOCK)
        == CGPU_IMAGE_FORMAT_BC1_RGBA_UNORM_BLOCK) { return VK_FORMAT_BC1_RGBA_UNORM_BLOCK; }
  else if ((image_format & CGPU_IMAGE_FORMAT_BC1_RGBA_SRGB_BLOCK)
        == CGPU_IMAGE_FORMAT_BC1_RGBA_SRGB_BLOCK) { return VK_FORMAT_BC1_RGBA_SRGB_BLOCK; }
  else if ((image_format & CGPU_IMAGE_FORMAT_BC2_UNORM_BLOCK)
        == CGPU_IMAGE_FORMAT_BC2_UNORM_BLOCK) { return VK_FORMAT_BC2_UNORM_BLOCK; }
  else if ((image_format & CGPU_IMAGE_FORMAT_BC2_SRGB_BLOCK)
        == CGPU_IMAGE_FORMAT_BC2_SRGB_BLOCK) { return VK_FORMAT_BC2_SRGB_BLOCK; }
  else if ((image_format & CGPU_IMAGE_FORMAT_BC3_UNORM_BLOCK)
        == CGPU_IMAGE_FORMAT_BC3_UNORM_BLOCK) { return VK_FORMAT_BC3_UNORM_BLOCK; }
  else if ((image_format & CGPU_IMAGE_FORMAT_BC3_SRGB_BLOCK)
        == CGPU_IMAGE_FORMAT_BC3_SRGB_BLOCK) { return VK_FORMAT_BC3_SRGB_BLOCK; }
  else if ((image_format & CGPU_IMAGE_FORMAT_BC4_UNORM_BLOCK)
        == CGPU_IMAGE_FORMAT_BC4_UNORM_BLOCK) { return VK_FORMAT_BC4_UNORM_BLOCK; }
  else if ((image_format & CGPU_IMAGE_FORMAT_BC4_SNORM_BLOCK)
        == CGPU_IMAGE_FORMAT_BC4_SNORM_BLOCK) { return VK_FORMAT_BC4_SNORM_BLOCK; }
  else if ((image_format & CGPU_IMAGE_FORMAT_BC5_UNORM_BLOCK)
        == CGPU_IMAGE_FORMAT_BC5_UNORM_BLOCK) { return VK_FORMAT_BC5_UNORM_BLOCK; }
  else if ((image_format & CGPU_IMAGE_FORMAT_BC5_SNORM_BLOCK)
        == CGPU_IMAGE_FORMAT_BC5_SNORM_BLOCK) { return VK_FORMAT_BC5_SNORM_BLOCK; }
  else if ((image_format & CGPU_IMAGE_FORMAT_BC6H_UFLOAT_BLOCK)
        == CGPU_IMAGE_FORMAT_BC6H_UFLOAT_BLOCK) { return VK_FORMAT_BC6H_UFLOAT_BLOCK; }
  else if ((image_format & CGPU_IMAGE_FORMAT_BC6H_SFLOAT_BLOCK)
        == CGPU_IMAGE_FORMAT_BC6H_SFLOAT_BLOCK) { return VK_FORMAT_BC6H_SFLOAT_BLOCK; }
  else if ((image_format & CGPU_IMAGE_FORMAT_BC7_UNORM_BLOCK)
        == CGPU_IMAGE_FORMAT_BC7_UNORM_BLOCK) { return VK_FORMAT_BC7_UNORM_BLOCK; }
  else if ((image_format & CGPU_IMAGE_FORMAT_BC7_SRGB_BLOCK)
        == CGPU_IMAGE_FORMAT_BC7_SRGB_BLOCK) { return VK_FORMAT_BC7_SRGB_BLOCK; }
  else if ((image_format & CGPU_IMAGE_FORMAT_ETC2_R8G8B8_UNORM_BLOCK)
        == CGPU_IMAGE_FORMAT_ETC2_R8G8B8_UNORM_BLOCK) { return VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK; }
  else if ((image_format & CGPU_IMAGE_FORMAT_ETC2_R8G8B8_SRGB_BLOCK)
        == CGPU_IMAGE_FORMAT_ETC2_R8G8B8_SRGB_BLOCK) { return VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK; }
  else if ((image_format & CGPU_IMAGE_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK)
        == CGPU_IMAGE_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK) { return VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK; }
  else if ((image_format & CGPU_IMAGE_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK)
        == CGPU_IMAGE_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK) { return VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK; }
  else if ((image_format & CGPU_IMAGE_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK)
        == CGPU_IMAGE_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK) { return VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK; }
  else if ((image_format & CGPU_IMAGE_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK)
        == CGPU_IMAGE_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK) { return VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK; }
  else if ((image_format & CGPU_IMAGE_FORMAT_EAC_R11_UNORM_BLOCK)
        == CGPU_IMAGE_FORMAT_EAC_R11_UNORM_BLOCK) { return VK_FORMAT_EAC_R11_UNORM_BLOCK; }
  else if ((image_format & CGPU_IMAGE_FORMAT_EAC_R11_SNORM_BLOCK)
        == CGPU_IMAGE_FORMAT_EAC_R11_SNORM_BLOCK) { return VK_FORMAT_EAC_R11_SNORM_BLOCK; }
  else if ((image_format & CGPU_IMAGE_FORMAT_EAC_R11G11_UNORM_BLOCK)
        == CGPU_IMAGE_FORMAT_EAC_R11G11_UNORM_BLOCK) { return VK_FORMAT_EAC_R11G11_UNORM_BLOCK; }
  else if ((image_format & CGPU_IMAGE_FORMAT_EAC_R11G11_SNORM_BLOCK)
        == CGPU_IMAGE_FORMAT_EAC_R11G11_SNORM_BLOCK) { return VK_FORMAT_EAC_R11G11_SNORM_BLOCK; }
  else if ((image_format & CGPU_IMAGE_FORMAT_ASTC_4x4_UNORM_BLOCK)
        == CGPU_IMAGE_FORMAT_ASTC_4x4_UNORM_BLOCK) { return VK_FORMAT_ASTC_4x4_UNORM_BLOCK; }
  else if ((image_format & CGPU_IMAGE_FORMAT_ASTC_4x4_SRGB_BLOCK)
        == CGPU_IMAGE_FORMAT_ASTC_4x4_SRGB_BLOCK) { return VK_FORMAT_ASTC_4x4_SRGB_BLOCK; }
  else if ((image_format & CGPU_IMAGE_FORMAT_ASTC_5x4_UNORM_BLOCK)
        == CGPU_IMAGE_FORMAT_ASTC_5x4_UNORM_BLOCK) { return VK_FORMAT_ASTC_5x4_UNORM_BLOCK; }
  else if ((image_format & CGPU_IMAGE_FORMAT_ASTC_5x4_SRGB_BLOCK)
        == CGPU_IMAGE_FORMAT_ASTC_5x4_SRGB_BLOCK) { return VK_FORMAT_ASTC_5x4_SRGB_BLOCK; }
  else if ((image_format & CGPU_IMAGE_FORMAT_ASTC_5x5_UNORM_BLOCK)
        == CGPU_IMAGE_FORMAT_ASTC_5x5_UNORM_BLOCK) { return VK_FORMAT_ASTC_5x5_UNORM_BLOCK; }
  else if ((image_format & CGPU_IMAGE_FORMAT_ASTC_5x5_SRGB_BLOCK)
        == CGPU_IMAGE_FORMAT_ASTC_5x5_SRGB_BLOCK) { return VK_FORMAT_ASTC_5x5_SRGB_BLOCK; }
  else if ((image_format & CGPU_IMAGE_FORMAT_ASTC_6x5_UNORM_BLOCK)
        == CGPU_IMAGE_FORMAT_ASTC_6x5_UNORM_BLOCK) { return VK_FORMAT_ASTC_6x5_UNORM_BLOCK; }
  else if ((image_format & CGPU_IMAGE_FORMAT_ASTC_6x5_SRGB_BLOCK)
        == CGPU_IMAGE_FORMAT_ASTC_6x5_SRGB_BLOCK) { return VK_FORMAT_ASTC_6x5_SRGB_BLOCK; }
  else if ((image_format & CGPU_IMAGE_FORMAT_ASTC_6x6_UNORM_BLOCK)
        == CGPU_IMAGE_FORMAT_ASTC_6x6_UNORM_BLOCK) { return VK_FORMAT_ASTC_6x6_UNORM_BLOCK; }
  else if ((image_format & CGPU_IMAGE_FORMAT_ASTC_6x6_SRGB_BLOCK)
        == CGPU_IMAGE_FORMAT_ASTC_6x6_SRGB_BLOCK) { return VK_FORMAT_ASTC_6x6_SRGB_BLOCK; }
  else if ((image_format & CGPU_IMAGE_FORMAT_ASTC_8x5_UNORM_BLOCK)
        == CGPU_IMAGE_FORMAT_ASTC_8x5_UNORM_BLOCK) { return VK_FORMAT_ASTC_8x5_UNORM_BLOCK; }
  else if ((image_format & CGPU_IMAGE_FORMAT_ASTC_8x5_SRGB_BLOCK)
        == CGPU_IMAGE_FORMAT_ASTC_8x5_SRGB_BLOCK) { return VK_FORMAT_ASTC_8x5_SRGB_BLOCK; }
  else if ((image_format & CGPU_IMAGE_FORMAT_ASTC_8x6_UNORM_BLOCK)
        == CGPU_IMAGE_FORMAT_ASTC_8x6_UNORM_BLOCK) { return VK_FORMAT_ASTC_8x6_UNORM_BLOCK; }
  else if ((image_format & CGPU_IMAGE_FORMAT_ASTC_8x6_SRGB_BLOCK)
        == CGPU_IMAGE_FORMAT_ASTC_8x6_SRGB_BLOCK) { return VK_FORMAT_ASTC_8x6_SRGB_BLOCK; }
  else if ((image_format & CGPU_IMAGE_FORMAT_ASTC_8x8_UNORM_BLOCK)
        == CGPU_IMAGE_FORMAT_ASTC_8x8_UNORM_BLOCK) { return VK_FORMAT_ASTC_8x8_UNORM_BLOCK; }
  else if ((image_format & CGPU_IMAGE_FORMAT_ASTC_8x8_SRGB_BLOCK)
        == CGPU_IMAGE_FORMAT_ASTC_8x8_SRGB_BLOCK) { return VK_FORMAT_ASTC_8x8_SRGB_BLOCK; }
  else if ((image_format & CGPU_IMAGE_FORMAT_ASTC_10x5_UNORM_BLOCK)
        == CGPU_IMAGE_FORMAT_ASTC_10x5_UNORM_BLOCK) { return VK_FORMAT_ASTC_10x5_UNORM_BLOCK; }
  else if ((image_format & CGPU_IMAGE_FORMAT_ASTC_10x5_SRGB_BLOCK)
        == CGPU_IMAGE_FORMAT_ASTC_10x5_SRGB_BLOCK) { return VK_FORMAT_ASTC_10x5_SRGB_BLOCK; }
  else if ((image_format & CGPU_IMAGE_FORMAT_ASTC_10x6_UNORM_BLOCK)
        == CGPU_IMAGE_FORMAT_ASTC_10x6_UNORM_BLOCK) { return VK_FORMAT_ASTC_10x6_UNORM_BLOCK; }
  else if ((image_format & CGPU_IMAGE_FORMAT_ASTC_10x6_SRGB_BLOCK)
        == CGPU_IMAGE_FORMAT_ASTC_10x6_SRGB_BLOCK) { return VK_FORMAT_ASTC_10x6_SRGB_BLOCK; }
  else if ((image_format & CGPU_IMAGE_FORMAT_ASTC_10x8_UNORM_BLOCK)
        == CGPU_IMAGE_FORMAT_ASTC_10x8_UNORM_BLOCK) { return VK_FORMAT_ASTC_10x8_UNORM_BLOCK; }
  else if ((image_format & CGPU_IMAGE_FORMAT_ASTC_10x8_SRGB_BLOCK)
        == CGPU_IMAGE_FORMAT_ASTC_10x8_SRGB_BLOCK) { return VK_FORMAT_ASTC_10x8_SRGB_BLOCK; }
  else if ((image_format & CGPU_IMAGE_FORMAT_ASTC_10x10_UNORM_BLOCK)
        == CGPU_IMAGE_FORMAT_ASTC_10x10_UNORM_BLOCK) { return VK_FORMAT_ASTC_10x10_UNORM_BLOCK; }
  else if ((image_format & CGPU_IMAGE_FORMAT_ASTC_10x10_SRGB_BLOCK)
        == CGPU_IMAGE_FORMAT_ASTC_10x10_SRGB_BLOCK) { return VK_FORMAT_ASTC_10x10_SRGB_BLOCK; }
  else if ((image_format & CGPU_IMAGE_FORMAT_ASTC_12x10_UNORM_BLOCK)
        == CGPU_IMAGE_FORMAT_ASTC_12x10_UNORM_BLOCK) { return VK_FORMAT_ASTC_12x10_UNORM_BLOCK; }
  else if ((image_format & CGPU_IMAGE_FORMAT_ASTC_12x10_SRGB_BLOCK)
        == CGPU_IMAGE_FORMAT_ASTC_12x10_SRGB_BLOCK) { return VK_FORMAT_ASTC_12x10_SRGB_BLOCK; }
  else if ((image_format & CGPU_IMAGE_FORMAT_ASTC_12x12_UNORM_BLOCK)
        == CGPU_IMAGE_FORMAT_ASTC_12x12_UNORM_BLOCK) { return VK_FORMAT_ASTC_12x12_UNORM_BLOCK; }
  else if ((image_format & CGPU_IMAGE_FORMAT_ASTC_12x12_SRGB_BLOCK)
        == CGPU_IMAGE_FORMAT_ASTC_12x12_SRGB_BLOCK) { return VK_FORMAT_ASTC_12x12_SRGB_BLOCK; }
  else if ((image_format & CGPU_IMAGE_FORMAT_G8B8G8R8_422_UNORM)
        == CGPU_IMAGE_FORMAT_G8B8G8R8_422_UNORM) { return VK_FORMAT_G8B8G8R8_422_UNORM; }
  else if ((image_format & CGPU_IMAGE_FORMAT_B8G8R8G8_422_UNORM)
        == CGPU_IMAGE_FORMAT_B8G8R8G8_422_UNORM) { return VK_FORMAT_B8G8R8G8_422_UNORM; }
  else if ((image_format & CGPU_IMAGE_FORMAT_G8_B8_R8_3PLANE_420_UNORM)
        == CGPU_IMAGE_FORMAT_G8_B8_R8_3PLANE_420_UNORM) { return VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM; }
  else if ((image_format & CGPU_IMAGE_FORMAT_G8_B8R8_2PLANE_420_UNORM)
        == CGPU_IMAGE_FORMAT_G8_B8R8_2PLANE_420_UNORM) { return VK_FORMAT_G8_B8R8_2PLANE_420_UNORM; }
  else if ((image_format & CGPU_IMAGE_FORMAT_G8_B8_R8_3PLANE_422_UNORM)
        == CGPU_IMAGE_FORMAT_G8_B8_R8_3PLANE_422_UNORM) { return VK_FORMAT_G8_B8_R8_3PLANE_422_UNORM; }
  else if ((image_format & CGPU_IMAGE_FORMAT_G8_B8R8_2PLANE_422_UNORM)
        == CGPU_IMAGE_FORMAT_G8_B8R8_2PLANE_422_UNORM) { return VK_FORMAT_G8_B8R8_2PLANE_422_UNORM; }
  else if ((image_format & CGPU_IMAGE_FORMAT_G8_B8_R8_3PLANE_444_UNORM)
        == CGPU_IMAGE_FORMAT_G8_B8_R8_3PLANE_444_UNORM) { return VK_FORMAT_G8_B8_R8_3PLANE_444_UNORM; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R10X6_UNORM_PACK16)
        == CGPU_IMAGE_FORMAT_R10X6_UNORM_PACK16) { return VK_FORMAT_R10X6_UNORM_PACK16; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R10X6G10X6_UNORM_2PACK16)
        == CGPU_IMAGE_FORMAT_R10X6G10X6_UNORM_2PACK16) { return VK_FORMAT_R10X6G10X6_UNORM_2PACK16; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R10X6G10X6B10X6A10X6_UNORM_4PACK16)
        == CGPU_IMAGE_FORMAT_R10X6G10X6B10X6A10X6_UNORM_4PACK16) { return VK_FORMAT_R10X6G10X6B10X6A10X6_UNORM_4PACK16; }
  else if ((image_format & CGPU_IMAGE_FORMAT_G10X6B10X6G10X6R10X6_422_UNORM_4PACK16)
        == CGPU_IMAGE_FORMAT_G10X6B10X6G10X6R10X6_422_UNORM_4PACK16) { return VK_FORMAT_G10X6B10X6G10X6R10X6_422_UNORM_4PACK16; }
  else if ((image_format & CGPU_IMAGE_FORMAT_B10X6G10X6R10X6G10X6_422_UNORM_4PACK16)
        == CGPU_IMAGE_FORMAT_B10X6G10X6R10X6G10X6_422_UNORM_4PACK16) { return VK_FORMAT_B10X6G10X6R10X6G10X6_422_UNORM_4PACK16; }
  else if ((image_format & CGPU_IMAGE_FORMAT_G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16)
        == CGPU_IMAGE_FORMAT_G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16) { return VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16; }
  else if ((image_format & CGPU_IMAGE_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16)
        == CGPU_IMAGE_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16) { return VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16; }
  else if ((image_format & CGPU_IMAGE_FORMAT_G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16)
        == CGPU_IMAGE_FORMAT_G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16) { return VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16; }
  else if ((image_format & CGPU_IMAGE_FORMAT_G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16)
        == CGPU_IMAGE_FORMAT_G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16) { return VK_FORMAT_G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16; }
  else if ((image_format & CGPU_IMAGE_FORMAT_G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16)
        == CGPU_IMAGE_FORMAT_G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16) { return VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R12X4_UNORM_PACK16)
        == CGPU_IMAGE_FORMAT_R12X4_UNORM_PACK16) { return VK_FORMAT_R12X4_UNORM_PACK16; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R12X4G12X4_UNORM_2PACK16)
        == CGPU_IMAGE_FORMAT_R12X4G12X4_UNORM_2PACK16) { return VK_FORMAT_R12X4G12X4_UNORM_2PACK16; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R12X4G12X4B12X4A12X4_UNORM_4PACK16)
        == CGPU_IMAGE_FORMAT_R12X4G12X4B12X4A12X4_UNORM_4PACK16) { return VK_FORMAT_R12X4G12X4B12X4A12X4_UNORM_4PACK16; }
  else if ((image_format & CGPU_IMAGE_FORMAT_G12X4B12X4G12X4R12X4_422_UNORM_4PACK16)
        == CGPU_IMAGE_FORMAT_G12X4B12X4G12X4R12X4_422_UNORM_4PACK16) { return VK_FORMAT_G12X4B12X4G12X4R12X4_422_UNORM_4PACK16; }
  else if ((image_format & CGPU_IMAGE_FORMAT_B12X4G12X4R12X4G12X4_422_UNORM_4PACK16)
        == CGPU_IMAGE_FORMAT_B12X4G12X4R12X4G12X4_422_UNORM_4PACK16) { return VK_FORMAT_B12X4G12X4R12X4G12X4_422_UNORM_4PACK16; }
  else if ((image_format & CGPU_IMAGE_FORMAT_G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16)
        == CGPU_IMAGE_FORMAT_G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16) { return VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16; }
  else if ((image_format & CGPU_IMAGE_FORMAT_G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16)
        == CGPU_IMAGE_FORMAT_G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16) { return VK_FORMAT_G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16; }
  else if ((image_format & CGPU_IMAGE_FORMAT_G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16)
        == CGPU_IMAGE_FORMAT_G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16) { return VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16; }
  else if ((image_format & CGPU_IMAGE_FORMAT_G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16)
        == CGPU_IMAGE_FORMAT_G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16) { return VK_FORMAT_G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16; }
  else if ((image_format & CGPU_IMAGE_FORMAT_G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16)
        == CGPU_IMAGE_FORMAT_G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16) { return VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16; }
  else if ((image_format & CGPU_IMAGE_FORMAT_G16B16G16R16_422_UNORM)
        == CGPU_IMAGE_FORMAT_G16B16G16R16_422_UNORM) { return VK_FORMAT_G16B16G16R16_422_UNORM; }
  else if ((image_format & CGPU_IMAGE_FORMAT_B16G16R16G16_422_UNORM)
        == CGPU_IMAGE_FORMAT_B16G16R16G16_422_UNORM) { return VK_FORMAT_B16G16R16G16_422_UNORM; }
  else if ((image_format & CGPU_IMAGE_FORMAT_G16_B16_R16_3PLANE_420_UNORM)
        == CGPU_IMAGE_FORMAT_G16_B16_R16_3PLANE_420_UNORM) { return VK_FORMAT_G16_B16_R16_3PLANE_420_UNORM; }
  else if ((image_format & CGPU_IMAGE_FORMAT_G16_B16R16_2PLANE_420_UNORM)
        == CGPU_IMAGE_FORMAT_G16_B16R16_2PLANE_420_UNORM) { return VK_FORMAT_G16_B16R16_2PLANE_420_UNORM; }
  else if ((image_format & CGPU_IMAGE_FORMAT_G16_B16_R16_3PLANE_422_UNORM)
        == CGPU_IMAGE_FORMAT_G16_B16_R16_3PLANE_422_UNORM) { return VK_FORMAT_G16_B16_R16_3PLANE_422_UNORM; }
  else if ((image_format & CGPU_IMAGE_FORMAT_G16_B16R16_2PLANE_422_UNORM)
        == CGPU_IMAGE_FORMAT_G16_B16R16_2PLANE_422_UNORM) { return VK_FORMAT_G16_B16R16_2PLANE_422_UNORM; }
  else if ((image_format & CGPU_IMAGE_FORMAT_G16_B16_R16_3PLANE_444_UNORM)
        == CGPU_IMAGE_FORMAT_G16_B16_R16_3PLANE_444_UNORM) { return VK_FORMAT_G16_B16_R16_3PLANE_444_UNORM; }
  else if ((image_format & CGPU_IMAGE_FORMAT_PVRTC1_2BPP_UNORM_BLOCK_IMG)
        == CGPU_IMAGE_FORMAT_PVRTC1_2BPP_UNORM_BLOCK_IMG) { return VK_FORMAT_PVRTC1_2BPP_UNORM_BLOCK_IMG; }
  else if ((image_format & CGPU_IMAGE_FORMAT_PVRTC1_4BPP_UNORM_BLOCK_IMG)
        == CGPU_IMAGE_FORMAT_PVRTC1_4BPP_UNORM_BLOCK_IMG) { return VK_FORMAT_PVRTC1_4BPP_UNORM_BLOCK_IMG; }
  else if ((image_format & CGPU_IMAGE_FORMAT_PVRTC2_2BPP_UNORM_BLOCK_IMG)
        == CGPU_IMAGE_FORMAT_PVRTC2_2BPP_UNORM_BLOCK_IMG) { return VK_FORMAT_PVRTC2_2BPP_UNORM_BLOCK_IMG; }
  else if ((image_format & CGPU_IMAGE_FORMAT_PVRTC2_4BPP_UNORM_BLOCK_IMG)
        == CGPU_IMAGE_FORMAT_PVRTC2_4BPP_UNORM_BLOCK_IMG) { return VK_FORMAT_PVRTC2_4BPP_UNORM_BLOCK_IMG; }
  else if ((image_format & CGPU_IMAGE_FORMAT_PVRTC1_2BPP_SRGB_BLOCK_IMG)
        == CGPU_IMAGE_FORMAT_PVRTC1_2BPP_SRGB_BLOCK_IMG) { return VK_FORMAT_PVRTC1_2BPP_SRGB_BLOCK_IMG; }
  else if ((image_format & CGPU_IMAGE_FORMAT_PVRTC1_4BPP_SRGB_BLOCK_IMG)
        == CGPU_IMAGE_FORMAT_PVRTC1_4BPP_SRGB_BLOCK_IMG) { return VK_FORMAT_PVRTC1_4BPP_SRGB_BLOCK_IMG; }
  else if ((image_format & CGPU_IMAGE_FORMAT_PVRTC2_2BPP_SRGB_BLOCK_IMG)
        == CGPU_IMAGE_FORMAT_PVRTC2_2BPP_SRGB_BLOCK_IMG) { return VK_FORMAT_PVRTC2_2BPP_SRGB_BLOCK_IMG; }
  else if ((image_format & CGPU_IMAGE_FORMAT_PVRTC2_4BPP_SRGB_BLOCK_IMG)
        == CGPU_IMAGE_FORMAT_PVRTC2_4BPP_SRGB_BLOCK_IMG) { return VK_FORMAT_PVRTC2_4BPP_SRGB_BLOCK_IMG; }
  else if ((image_format & CGPU_IMAGE_FORMAT_ASTC_4x4_SFLOAT_BLOCK_EXT)
        == CGPU_IMAGE_FORMAT_ASTC_4x4_SFLOAT_BLOCK_EXT) { return VK_FORMAT_ASTC_4x4_SFLOAT_BLOCK_EXT; }
  else if ((image_format & CGPU_IMAGE_FORMAT_ASTC_5x4_SFLOAT_BLOCK_EXT)
        == CGPU_IMAGE_FORMAT_ASTC_5x4_SFLOAT_BLOCK_EXT) { return VK_FORMAT_ASTC_5x4_SFLOAT_BLOCK_EXT; }
  else if ((image_format & CGPU_IMAGE_FORMAT_ASTC_5x5_SFLOAT_BLOCK_EXT)
        == CGPU_IMAGE_FORMAT_ASTC_5x5_SFLOAT_BLOCK_EXT) { return VK_FORMAT_ASTC_5x5_SFLOAT_BLOCK_EXT; }
  else if ((image_format & CGPU_IMAGE_FORMAT_ASTC_6x5_SFLOAT_BLOCK_EXT)
        == CGPU_IMAGE_FORMAT_ASTC_6x5_SFLOAT_BLOCK_EXT) { return VK_FORMAT_ASTC_6x5_SFLOAT_BLOCK_EXT; }
  else if ((image_format & CGPU_IMAGE_FORMAT_ASTC_6x6_SFLOAT_BLOCK_EXT)
        == CGPU_IMAGE_FORMAT_ASTC_6x6_SFLOAT_BLOCK_EXT) { return VK_FORMAT_ASTC_6x6_SFLOAT_BLOCK_EXT; }
  else if ((image_format & CGPU_IMAGE_FORMAT_ASTC_8x5_SFLOAT_BLOCK_EXT)
        == CGPU_IMAGE_FORMAT_ASTC_8x5_SFLOAT_BLOCK_EXT) { return VK_FORMAT_ASTC_8x5_SFLOAT_BLOCK_EXT; }
  else if ((image_format & CGPU_IMAGE_FORMAT_ASTC_8x6_SFLOAT_BLOCK_EXT)
        == CGPU_IMAGE_FORMAT_ASTC_8x6_SFLOAT_BLOCK_EXT) { return VK_FORMAT_ASTC_8x6_SFLOAT_BLOCK_EXT; }
  else if ((image_format & CGPU_IMAGE_FORMAT_ASTC_8x8_SFLOAT_BLOCK_EXT)
        == CGPU_IMAGE_FORMAT_ASTC_8x8_SFLOAT_BLOCK_EXT) { return VK_FORMAT_ASTC_8x8_SFLOAT_BLOCK_EXT; }
  else if ((image_format & CGPU_IMAGE_FORMAT_ASTC_10x5_SFLOAT_BLOCK_EXT)
        == CGPU_IMAGE_FORMAT_ASTC_10x5_SFLOAT_BLOCK_EXT) { return VK_FORMAT_ASTC_10x5_SFLOAT_BLOCK_EXT; }
  else if ((image_format & CGPU_IMAGE_FORMAT_ASTC_10x6_SFLOAT_BLOCK_EXT)
        == CGPU_IMAGE_FORMAT_ASTC_10x6_SFLOAT_BLOCK_EXT) { return VK_FORMAT_ASTC_10x6_SFLOAT_BLOCK_EXT; }
  else if ((image_format & CGPU_IMAGE_FORMAT_ASTC_10x8_SFLOAT_BLOCK_EXT)
        == CGPU_IMAGE_FORMAT_ASTC_10x8_SFLOAT_BLOCK_EXT) { return VK_FORMAT_ASTC_10x8_SFLOAT_BLOCK_EXT; }
  else if ((image_format & CGPU_IMAGE_FORMAT_ASTC_10x10_SFLOAT_BLOCK_EXT)
        == CGPU_IMAGE_FORMAT_ASTC_10x10_SFLOAT_BLOCK_EXT) { return VK_FORMAT_ASTC_10x10_SFLOAT_BLOCK_EXT; }
  else if ((image_format & CGPU_IMAGE_FORMAT_ASTC_12x10_SFLOAT_BLOCK_EXT)
        == CGPU_IMAGE_FORMAT_ASTC_12x10_SFLOAT_BLOCK_EXT) { return VK_FORMAT_ASTC_12x10_SFLOAT_BLOCK_EXT; }
  else if ((image_format & CGPU_IMAGE_FORMAT_ASTC_12x12_SFLOAT_BLOCK_EXT)
        == CGPU_IMAGE_FORMAT_ASTC_12x12_SFLOAT_BLOCK_EXT) { return VK_FORMAT_ASTC_12x12_SFLOAT_BLOCK_EXT; }
  else if ((image_format & CGPU_IMAGE_FORMAT_G8B8G8R8_422_UNORM_KHR)
        == CGPU_IMAGE_FORMAT_G8B8G8R8_422_UNORM_KHR) { return VK_FORMAT_G8B8G8R8_422_UNORM_KHR; }
  else if ((image_format & CGPU_IMAGE_FORMAT_B8G8R8G8_422_UNORM_KHR)
        == CGPU_IMAGE_FORMAT_B8G8R8G8_422_UNORM_KHR) { return VK_FORMAT_B8G8R8G8_422_UNORM_KHR; }
  else if ((image_format & CGPU_IMAGE_FORMAT_G8_B8_R8_3PLANE_420_UNORM_KHR)
        == CGPU_IMAGE_FORMAT_G8_B8_R8_3PLANE_420_UNORM_KHR) { return VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM_KHR; }
  else if ((image_format & CGPU_IMAGE_FORMAT_G8_B8R8_2PLANE_420_UNORM_KHR)
        == CGPU_IMAGE_FORMAT_G8_B8R8_2PLANE_420_UNORM_KHR) { return VK_FORMAT_G8_B8R8_2PLANE_420_UNORM_KHR; }
  else if ((image_format & CGPU_IMAGE_FORMAT_G8_B8_R8_3PLANE_422_UNORM_KHR)
        == CGPU_IMAGE_FORMAT_G8_B8_R8_3PLANE_422_UNORM_KHR) { return VK_FORMAT_G8_B8_R8_3PLANE_422_UNORM_KHR; }
  else if ((image_format & CGPU_IMAGE_FORMAT_G8_B8R8_2PLANE_422_UNORM_KHR)
        == CGPU_IMAGE_FORMAT_G8_B8R8_2PLANE_422_UNORM_KHR) { return VK_FORMAT_G8_B8R8_2PLANE_422_UNORM_KHR; }
  else if ((image_format & CGPU_IMAGE_FORMAT_G8_B8_R8_3PLANE_444_UNORM_KHR)
        == CGPU_IMAGE_FORMAT_G8_B8_R8_3PLANE_444_UNORM_KHR) { return VK_FORMAT_G8_B8_R8_3PLANE_444_UNORM_KHR; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R10X6_UNORM_PACK16_KHR)
        == CGPU_IMAGE_FORMAT_R10X6_UNORM_PACK16_KHR) { return VK_FORMAT_R10X6_UNORM_PACK16_KHR; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R10X6G10X6_UNORM_2PACK16_KHR)
        == CGPU_IMAGE_FORMAT_R10X6G10X6_UNORM_2PACK16_KHR) { return VK_FORMAT_R10X6G10X6_UNORM_2PACK16_KHR; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R10X6G10X6B10X6A10X6_UNORM_4PACK16_KHR)
        == CGPU_IMAGE_FORMAT_R10X6G10X6B10X6A10X6_UNORM_4PACK16_KHR) { return VK_FORMAT_R10X6G10X6B10X6A10X6_UNORM_4PACK16_KHR; }
  else if ((image_format & CGPU_IMAGE_FORMAT_G10X6B10X6G10X6R10X6_422_UNORM_4PACK16_KHR)
        == CGPU_IMAGE_FORMAT_G10X6B10X6G10X6R10X6_422_UNORM_4PACK16_KHR) { return VK_FORMAT_G10X6B10X6G10X6R10X6_422_UNORM_4PACK16_KHR; }
  else if ((image_format & CGPU_IMAGE_FORMAT_B10X6G10X6R10X6G10X6_422_UNORM_4PACK16_KHR)
        == CGPU_IMAGE_FORMAT_B10X6G10X6R10X6G10X6_422_UNORM_4PACK16_KHR) { return VK_FORMAT_B10X6G10X6R10X6G10X6_422_UNORM_4PACK16_KHR; }
  else if ((image_format & CGPU_IMAGE_FORMAT_G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16_KHR)
      == CGPU_IMAGE_FORMAT_G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16_KHR) { return VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16_KHR; }
  else if ((image_format & CGPU_IMAGE_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16_KHR)
        == CGPU_IMAGE_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16_KHR) { return VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16_KHR; }
  else if ((image_format & CGPU_IMAGE_FORMAT_G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16_KHR)
        == CGPU_IMAGE_FORMAT_G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16_KHR) { return VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16_KHR; }
  else if ((image_format & CGPU_IMAGE_FORMAT_G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16_KHR)
        == CGPU_IMAGE_FORMAT_G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16_KHR) { return VK_FORMAT_G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16_KHR; }
  else if ((image_format & CGPU_IMAGE_FORMAT_G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16_KHR)
        == CGPU_IMAGE_FORMAT_G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16_KHR) { return VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16_KHR; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R12X4_UNORM_PACK16_KHR)
        == CGPU_IMAGE_FORMAT_R12X4_UNORM_PACK16_KHR) { return VK_FORMAT_R12X4_UNORM_PACK16_KHR; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R12X4G12X4_UNORM_2PACK16_KHR)
        == CGPU_IMAGE_FORMAT_R12X4G12X4_UNORM_2PACK16_KHR) { return VK_FORMAT_R12X4G12X4_UNORM_2PACK16_KHR; }
  else if ((image_format & CGPU_IMAGE_FORMAT_R12X4G12X4B12X4A12X4_UNORM_4PACK16_KHR)
        == CGPU_IMAGE_FORMAT_R12X4G12X4B12X4A12X4_UNORM_4PACK16_KHR) { return VK_FORMAT_R12X4G12X4B12X4A12X4_UNORM_4PACK16_KHR; }
  else if ((image_format & CGPU_IMAGE_FORMAT_G12X4B12X4G12X4R12X4_422_UNORM_4PACK16_KHR)
        == CGPU_IMAGE_FORMAT_G12X4B12X4G12X4R12X4_422_UNORM_4PACK16_KHR) { return VK_FORMAT_G12X4B12X4G12X4R12X4_422_UNORM_4PACK16_KHR; }
  else if ((image_format & CGPU_IMAGE_FORMAT_B12X4G12X4R12X4G12X4_422_UNORM_4PACK16_KHR)
        == CGPU_IMAGE_FORMAT_B12X4G12X4R12X4G12X4_422_UNORM_4PACK16_KHR) { return VK_FORMAT_B12X4G12X4R12X4G12X4_422_UNORM_4PACK16_KHR; }
  else if ((image_format & CGPU_IMAGE_FORMAT_G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16_KHR)
        == CGPU_IMAGE_FORMAT_G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16_KHR) { return VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16_KHR; }
  else if ((image_format & CGPU_IMAGE_FORMAT_G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16_KHR)
        == CGPU_IMAGE_FORMAT_G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16_KHR) { return VK_FORMAT_G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16_KHR; }
  else if ((image_format & CGPU_IMAGE_FORMAT_G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16_KHR)
        == CGPU_IMAGE_FORMAT_G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16_KHR) { return VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16_KHR; }
  else if ((image_format & CGPU_IMAGE_FORMAT_G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16_KHR)
        == CGPU_IMAGE_FORMAT_G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16_KHR) { return VK_FORMAT_G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16_KHR; }
  else if ((image_format & CGPU_IMAGE_FORMAT_G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16_KHR)
        == CGPU_IMAGE_FORMAT_G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16_KHR) { return VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16_KHR; }
  else if ((image_format & CGPU_IMAGE_FORMAT_G16B16G16R16_422_UNORM_KHR)
        == CGPU_IMAGE_FORMAT_G16B16G16R16_422_UNORM_KHR) { return VK_FORMAT_G16B16G16R16_422_UNORM_KHR; }
  else if ((image_format & CGPU_IMAGE_FORMAT_B16G16R16G16_422_UNORM_KHR)
        == CGPU_IMAGE_FORMAT_B16G16R16G16_422_UNORM_KHR) { return VK_FORMAT_B16G16R16G16_422_UNORM_KHR; }
  else if ((image_format & CGPU_IMAGE_FORMAT_G16_B16_R16_3PLANE_420_UNORM_KHR)
        == CGPU_IMAGE_FORMAT_G16_B16_R16_3PLANE_420_UNORM_KHR) { return VK_FORMAT_G16_B16_R16_3PLANE_420_UNORM_KHR; }
  else if ((image_format & CGPU_IMAGE_FORMAT_G16_B16R16_2PLANE_420_UNORM_KHR)
        == CGPU_IMAGE_FORMAT_G16_B16R16_2PLANE_420_UNORM_KHR) { return VK_FORMAT_G16_B16R16_2PLANE_420_UNORM_KHR; }
  else if ((image_format & CGPU_IMAGE_FORMAT_G16_B16_R16_3PLANE_422_UNORM_KHR)
        == CGPU_IMAGE_FORMAT_G16_B16_R16_3PLANE_422_UNORM_KHR) { return VK_FORMAT_G16_B16_R16_3PLANE_422_UNORM_KHR; }
  else if ((image_format & CGPU_IMAGE_FORMAT_G16_B16R16_2PLANE_422_UNORM_KHR)
        == CGPU_IMAGE_FORMAT_G16_B16R16_2PLANE_422_UNORM_KHR) { return VK_FORMAT_G16_B16R16_2PLANE_422_UNORM_KHR; }
  else if ((image_format & CGPU_IMAGE_FORMAT_G16_B16_R16_3PLANE_444_UNORM_KHR)
        == CGPU_IMAGE_FORMAT_G16_B16_R16_3PLANE_444_UNORM_KHR) { return VK_FORMAT_G16_B16_R16_3PLANE_444_UNORM_KHR; }
  return VK_FORMAT_UNDEFINED;
}

/* API method implementation. */

CgpuResult cgpu_initialize(
  const char* p_app_name,
  uint32_t version_major,
  uint32_t version_minor,
  uint32_t version_patch)
{
  VkResult result = volkInitialize();

  if (result != VK_SUCCESS) {
    return CGPU_FAIL_UNABLE_TO_INITIALIZE_VOLK;
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
  app_info.applicationVersion = VK_MAKE_VERSION(
    version_major,
    version_minor,
    version_patch
  );
  app_info.pEngineName = p_app_name;
  app_info.engineVersion = VK_MAKE_VERSION(
    version_major,
    version_minor,
    version_patch);
  app_info.apiVersion = MIN_VK_API_VERSION;

  VkInstanceCreateInfo create_info;
  create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  create_info.pNext = NULL;
  create_info.flags = 0;
  create_info.pApplicationInfo = &app_info;
  create_info.enabledLayerCount = validation_layer_count;
  create_info.ppEnabledLayerNames = validation_layers;
  create_info.enabledExtensionCount = instance_extension_count;
  create_info.ppEnabledExtensionNames = instance_extensions;

  result = vkCreateInstance(
    &create_info,
    NULL,
    &iinstance.instance
  );
  if (result != VK_SUCCESS) {
    return CGPU_FAIL_UNABLE_TO_INITIALIZE_VULKAN;
  }

  volkLoadInstance(iinstance.instance);

  resource_store_create(&idevice_store, sizeof(cgpu_idevice), 1);
  resource_store_create(&ishader_store, sizeof(cgpu_ishader), 16);
  resource_store_create(&ibuffer_store, sizeof(cgpu_ibuffer), 16);
  resource_store_create(&iimage_store, sizeof(cgpu_iimage), 64);
  resource_store_create(&ipipeline_store, sizeof(cgpu_ipipeline), 8);
  resource_store_create(&icommand_buffer_store, sizeof(cgpu_icommand_buffer), 16);
  resource_store_create(&ifence_store, sizeof(cgpu_ifence), 8);

  return CGPU_OK;
}

CgpuResult cgpu_terminate(void)
{
  resource_store_destroy(&idevice_store);
  resource_store_destroy(&ishader_store);
  resource_store_destroy(&ibuffer_store);
  resource_store_destroy(&iimage_store);
  resource_store_destroy(&ipipeline_store);
  resource_store_destroy(&icommand_buffer_store);
  resource_store_destroy(&ifence_store);

  vkDestroyInstance(iinstance.instance, NULL);

  return CGPU_OK;
}

CgpuResult cgpu_get_device_count(uint32_t* p_device_count)
{
  vkEnumeratePhysicalDevices(
    iinstance.instance,
    p_device_count,
    NULL
  );
  return CGPU_OK;
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

CgpuResult cgpu_create_device(
  uint32_t index,
  cgpu_device* p_device)
{
  p_device->handle = resource_store_create_handle(&idevice_store);

  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(*p_device, &idevice)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }

  uint32_t phys_device_count;
  vkEnumeratePhysicalDevices(
    iinstance.instance,
    &phys_device_count,
    NULL
  );

  if (phys_device_count > MAX_PHYSICAL_DEVICES)
  {
    resource_store_free_handle(&idevice_store, p_device->handle);
    return CGPU_FAIL_MAX_PHYSICAL_DEVICES_REACHED;
  }

  if (phys_device_count == 0 ||
      index >= phys_device_count)
  {
    resource_store_free_handle(&idevice_store, p_device->handle);
    return CGPU_FAIL_NO_DEVICE_AT_INDEX;
  }

  VkPhysicalDevice phys_devices[MAX_PHYSICAL_DEVICES];

  vkEnumeratePhysicalDevices(
    iinstance.instance,
    &phys_device_count,
    phys_devices
  );

  idevice->physical_device = phys_devices[index];

  VkPhysicalDeviceSubgroupProperties subgroup_properties;
  subgroup_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
  subgroup_properties.pNext = NULL;

  VkPhysicalDeviceProperties2 device_properties;
  device_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  device_properties.pNext = &subgroup_properties;

  vkGetPhysicalDeviceProperties2(
    idevice->physical_device,
    &device_properties
  );

  if (device_properties.properties.apiVersion < MIN_VK_API_VERSION)
  {
    resource_store_free_handle(&idevice_store, p_device->handle);
    return CGPU_FAIL_VK_VERSION_NOT_SUPPORTED;
  }

  if ((subgroup_properties.supportedStages & VK_QUEUE_COMPUTE_BIT) != VK_QUEUE_COMPUTE_BIT ||
      (subgroup_properties.supportedOperations & VK_SUBGROUP_FEATURE_BASIC_BIT) != VK_SUBGROUP_FEATURE_BASIC_BIT ||
      (subgroup_properties.supportedOperations & VK_SUBGROUP_FEATURE_BALLOT_BIT) != VK_SUBGROUP_FEATURE_BALLOT_BIT)
  {
    resource_store_free_handle(&idevice_store, p_device->handle);
    return CGPU_FAIL_FEATURE_REQUIREMENTS_NOT_MET;
  }

  idevice->limits =
    cgpu_translate_physical_device_limits(device_properties.properties.limits, subgroup_properties);

  uint32_t device_ext_count;
  vkEnumerateDeviceExtensionProperties(
    idevice->physical_device,
    NULL,
    &device_ext_count,
    NULL
  );

  if (device_ext_count > MAX_DEVICE_EXTENSIONS)
  {
    resource_store_free_handle(&idevice_store, p_device->handle);
    return CGPU_FAIL_MAX_DEVICE_EXTENSIONS_REACHED;
  }

  VkExtensionProperties device_extensions[MAX_DEVICE_EXTENSIONS];

  vkEnumerateDeviceExtensionProperties(
    idevice->physical_device,
    NULL,
    &device_ext_count,
    device_extensions
  );

  const char* VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME = "VK_KHR_portability_subset";

  bool has_portability_subset = cgpu_find_device_extension(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME,
                                                           device_ext_count,
                                                           device_extensions);

  uint32_t queue_family_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(
    idevice->physical_device,
    &queue_family_count,
    NULL
  );

  if (queue_family_count > MAX_QUEUE_FAMILIES)
  {
    resource_store_free_handle(&idevice_store, p_device->handle);
    return CGPU_FAIL_MAX_QUEUE_FAMILIES_REACHED;
  }

  VkQueueFamilyProperties queue_families[MAX_QUEUE_FAMILIES];

  vkGetPhysicalDeviceQueueFamilyProperties(
    idevice->physical_device,
    &queue_family_count,
    queue_families
  );

  /* Since ray tracing is a continuous, compute-heavy task, we don't need
     to schedule work or translate command buffers very often. Therefore,
     we also don't need async execution and can operate on a single queue. */
  int32_t queue_family_index = -1;
  for (uint32_t i = 0; i < queue_family_count; ++i) {
    const VkQueueFamilyProperties* queue_family = &queue_families[i];
    if ((queue_family->queueFlags & VK_QUEUE_COMPUTE_BIT) &&
        (queue_family->queueFlags & VK_QUEUE_TRANSFER_BIT)) {
      queue_family_index = i;
    }
  }
  if (queue_family_index == -1) {
    resource_store_free_handle(&idevice_store, p_device->handle);
    return CGPU_FAIL_DEVICE_HAS_NO_COMPUTE_QUEUE_FAMILY;
  }

  VkDeviceQueueCreateInfo queue_create_info;
  queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queue_create_info.pNext = NULL;
  queue_create_info.flags = 0;
  queue_create_info.queueFamilyIndex = queue_family_index;
  queue_create_info.queueCount = 1;
  const float queue_priority = 1.0f;
  queue_create_info.pQueuePriorities = &queue_priority;

  VkPhysicalDeviceFeatures2 device_features2;
  device_features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  device_features2.pNext = NULL;
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
  device_features2.features.shaderImageGatherExtended = VK_FALSE;
  device_features2.features.shaderStorageImageExtendedFormats = VK_FALSE;
  device_features2.features.shaderStorageImageMultisample = VK_FALSE;
  device_features2.features.shaderStorageImageReadWithoutFormat = VK_FALSE;
  device_features2.features.shaderStorageImageWriteWithoutFormat = VK_FALSE;
  device_features2.features.shaderUniformBufferArrayDynamicIndexing = VK_FALSE;
  device_features2.features.shaderSampledImageArrayDynamicIndexing = VK_FALSE;
  device_features2.features.shaderStorageBufferArrayDynamicIndexing = VK_FALSE;
  device_features2.features.shaderStorageImageArrayDynamicIndexing = VK_FALSE;
  device_features2.features.shaderClipDistance = VK_FALSE;
  device_features2.features.shaderCullDistance = VK_FALSE;
  device_features2.features.shaderFloat64 = VK_FALSE;
  device_features2.features.shaderInt64 = VK_FALSE;
  device_features2.features.shaderInt16 = VK_FALSE;
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
   nowadays, there is no difference to instance validation layers. */
  device_create_info.enabledLayerCount = 0;
  device_create_info.ppEnabledLayerNames = NULL;
  device_create_info.enabledExtensionCount = has_portability_subset ? 1 : 0;
  device_create_info.ppEnabledExtensionNames = &VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME;
  device_create_info.pEnabledFeatures = NULL;

  VkResult result = vkCreateDevice(
    idevice->physical_device,
    &device_create_info,
    NULL,
    &idevice->logical_device
  );
  if (result != VK_SUCCESS) {
    resource_store_free_handle(&idevice_store, p_device->handle);
    return CGPU_FAIL_CAN_NOT_CREATE_LOGICAL_DEVICE;
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
  pool_info.flags = 0;
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
    return CGPU_FAIL_CAN_NOT_CREATE_COMMAND_POOL;
  }

  VkSamplerCreateInfo sampler_info;
  sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  sampler_info.pNext = NULL;
  sampler_info.flags = 0;
  sampler_info.magFilter = VK_FILTER_LINEAR;
  sampler_info.minFilter = VK_FILTER_LINEAR;
  sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  sampler_info.mipLodBias = 0.0f;
  sampler_info.anisotropyEnable = VK_TRUE;
  sampler_info.maxAnisotropy = 16.0f;
  sampler_info.compareEnable = VK_FALSE;
  sampler_info.compareOp = VK_COMPARE_OP_ALWAYS;
  sampler_info.minLod = 0.0f;
  sampler_info.maxLod = 0.0f;
  sampler_info.borderColor = VK_BORDER_COLOR_INT_TRANSPARENT_BLACK;
  sampler_info.unnormalizedCoordinates = VK_FALSE;

  result = idevice->table.vkCreateSampler(
    idevice->logical_device,
    &sampler_info,
    NULL,
    &idevice->sampler
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
    return CGPU_FAIL_CAN_NOT_CREATE_COMMAND_POOL;
  }

  VkQueryPoolCreateInfo timestamp_pool_info;
  timestamp_pool_info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
  timestamp_pool_info.pNext = NULL;
  timestamp_pool_info.flags = 0;
  timestamp_pool_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
  timestamp_pool_info.queryCount = MAX_TIMESTAMP_QUERIES;
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

    idevice->table.vkDestroySampler(
      idevice->logical_device,
      idevice->sampler,
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
    return CGPU_FAIL_UNABLE_TO_CREATE_QUERY_POOL;
  }

  return CGPU_OK;
}

CgpuResult cgpu_destroy_device(
  cgpu_device device)
{
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }

  idevice->table.vkDestroyQueryPool(
    idevice->logical_device,
    idevice->timestamp_pool,
    NULL
  );
  idevice->table.vkDestroySampler(
    idevice->logical_device,
    idevice->sampler,
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
  return CGPU_OK;
}

CgpuResult cgpu_create_shader(
  cgpu_device device,
  uint64_t size,
  const uint32_t* p_source,
  cgpu_shader* p_shader)
{
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }

  p_shader->handle = resource_store_create_handle(&ishader_store);

  cgpu_ishader* ishader;
  if (!cgpu_resolve_shader(*p_shader, &ishader)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }

  VkShaderModuleCreateInfo shader_module_create_info;
  shader_module_create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  shader_module_create_info.pNext = NULL;
  shader_module_create_info.flags = 0;
  shader_module_create_info.codeSize = size;
  shader_module_create_info.pCode = p_source;

  const VkResult result = idevice->table.vkCreateShaderModule(
    idevice->logical_device,
    &shader_module_create_info,
    NULL,
    &ishader->module
  );
  if (result != VK_SUCCESS) {
    resource_store_free_handle(&ishader_store, p_shader->handle);
    return CGPU_FAIL_UNABLE_TO_CREATE_SHADER_MODULE;
  }

  return CGPU_OK;
}

CgpuResult cgpu_destroy_shader(
  cgpu_device device,
  cgpu_shader shader)
{
  cgpu_idevice* idevice;
  cgpu_ishader* ishader;
  if (!cgpu_resolve_device(device, &idevice)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  if (!cgpu_resolve_shader(shader, &ishader)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }

  idevice->table.vkDestroyShaderModule(
    idevice->logical_device,
    ishader->module,
    NULL
  );

  resource_store_free_handle(&ishader_store, shader.handle);

  return CGPU_OK;
}

CgpuResult cgpu_create_buffer(
  cgpu_device device,
  CgpuBufferUsageFlags usage,
  CgpuMemoryPropertyFlags memory_properties,
  uint64_t size,
  cgpu_buffer* p_buffer)
{
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }

  p_buffer->handle = resource_store_create_handle(&ibuffer_store);

  cgpu_ibuffer* ibuffer;
  if (!cgpu_resolve_buffer(*p_buffer, &ibuffer)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }

  VkBufferUsageFlags vk_buffer_usage = 0;
  if ((usage & CGPU_BUFFER_USAGE_FLAG_TRANSFER_SRC)
        == CGPU_BUFFER_USAGE_FLAG_TRANSFER_SRC) {
    vk_buffer_usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  }
  if ((usage & CGPU_BUFFER_USAGE_FLAG_TRANSFER_DST)
        == CGPU_BUFFER_USAGE_FLAG_TRANSFER_DST) {
    vk_buffer_usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  }
  if ((usage & CGPU_BUFFER_USAGE_FLAG_UNIFORM_BUFFER)
        == CGPU_BUFFER_USAGE_FLAG_UNIFORM_BUFFER) {
    vk_buffer_usage |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
  }
  if ((usage & CGPU_BUFFER_USAGE_FLAG_STORAGE_BUFFER)
        == CGPU_BUFFER_USAGE_FLAG_STORAGE_BUFFER) {
    vk_buffer_usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  }
  if ((usage & CGPU_BUFFER_USAGE_FLAG_UNIFORM_TEXEL_BUFFER)
        == CGPU_BUFFER_USAGE_FLAG_UNIFORM_TEXEL_BUFFER) {
    vk_buffer_usage |= VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT;
  }
  if ((usage & CGPU_BUFFER_USAGE_FLAG_STORAGE_TEXEL_BUFFER)
        == CGPU_BUFFER_USAGE_FLAG_STORAGE_TEXEL_BUFFER) {
    vk_buffer_usage |= VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT;
  }

  VkBufferCreateInfo buffer_info;
  buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_info.pNext = NULL;
  buffer_info.flags = 0;
  buffer_info.size = size;
  buffer_info.usage = vk_buffer_usage;
  buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  buffer_info.queueFamilyIndexCount = 0;
  buffer_info.pQueueFamilyIndices = NULL;

  VkResult result = idevice->table.vkCreateBuffer(
    idevice->logical_device,
    &buffer_info,
    NULL,
    &ibuffer->buffer
  );
  if (result != VK_SUCCESS) {
    resource_store_free_handle(&ibuffer_store, p_buffer->handle);
    return CGPU_FAIL_UNABLE_TO_CREATE_BUFFER;
  }

  VkPhysicalDeviceMemoryProperties physical_device_memory_properties;
  vkGetPhysicalDeviceMemoryProperties(
    idevice->physical_device,
    &physical_device_memory_properties
  );

  VkMemoryRequirements mem_requirements;
  idevice->table.vkGetBufferMemoryRequirements(
    idevice->logical_device,
    ibuffer->buffer,
    &mem_requirements
  );

  const VkMemoryPropertyFlags mem_flags =
      cgpu_translate_memory_properties(memory_properties);

  int32_t mem_index = -1;
  for (uint32_t i = 0; i < physical_device_memory_properties.memoryTypeCount; ++i) {
    if ((mem_requirements.memoryTypeBits & (1 << i)) &&
        (physical_device_memory_properties.memoryTypes[i].propertyFlags & mem_flags) == mem_flags) {
      mem_index = i;
      break;
    }
  }
  if (mem_index == -1) {
    resource_store_free_handle(&ibuffer_store, p_buffer->handle);
    idevice->table.vkDestroyBuffer(idevice->logical_device, ibuffer->buffer, NULL);
    return CGPU_FAIL_NO_SUITABLE_MEMORY_TYPE;
  }

  VkMemoryAllocateInfo mem_alloc_info;
  mem_alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  mem_alloc_info.pNext = NULL;
  mem_alloc_info.allocationSize = mem_requirements.size;
  mem_alloc_info.memoryTypeIndex = mem_index;

  result = idevice->table.vkAllocateMemory(
    idevice->logical_device,
    &mem_alloc_info,
    NULL,
    &ibuffer->memory
  );
  if (result != VK_SUCCESS) {
    resource_store_free_handle(&ibuffer_store, p_buffer->handle);
    idevice->table.vkDestroyBuffer(idevice->logical_device, ibuffer->buffer, NULL);
    return CGPU_FAIL_UNABLE_TO_ALLOCATE_MEMORY;
  }

  idevice->table.vkBindBufferMemory(
    idevice->logical_device,
    ibuffer->buffer,
    ibuffer->memory,
    0
  );

  ibuffer->size = size;

  return CGPU_OK;
}

CgpuResult cgpu_destroy_buffer(
  cgpu_device device,
  cgpu_buffer buffer)
{
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  cgpu_ibuffer* ibuffer;
  if (!cgpu_resolve_buffer(buffer, &ibuffer)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }

  idevice->table.vkDestroyBuffer(
    idevice->logical_device,
    ibuffer->buffer,
    NULL
  );
  idevice->table.vkFreeMemory(
    idevice->logical_device,
    ibuffer->memory,
    NULL
  );

  resource_store_free_handle(&ibuffer_store, buffer.handle);

  return CGPU_OK;
}

CgpuResult cgpu_map_buffer(
  cgpu_device device,
  cgpu_buffer buffer,
  uint64_t offset,
  uint64_t size,
  void** pp_mapped_mem)
{
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  cgpu_ibuffer* ibuffer;
  if (!cgpu_resolve_buffer(buffer, &ibuffer)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }

  const VkResult result = idevice->table.vkMapMemory(
    idevice->logical_device,
    ibuffer->memory,
    offset,
    (size == CGPU_WHOLE_SIZE) ? ibuffer->size : size,
    0,
    pp_mapped_mem
  );

  if (result != VK_SUCCESS) {
    return CGPU_FAIL_UNABLE_TO_MAP_MEMORY;
  }

  return CGPU_OK;
}

CgpuResult cgpu_unmap_buffer(
  cgpu_device device,
  cgpu_buffer buffer)
{
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  cgpu_ibuffer* ibuffer;
  if (!cgpu_resolve_buffer(buffer, &ibuffer)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  idevice->table.vkUnmapMemory(
    idevice->logical_device,
    ibuffer->memory
  );
  return CGPU_OK;
}

CgpuResult cgpu_create_image(
  cgpu_device device,
  uint32_t width,
  uint32_t height,
  CgpuImageFormat format,
  CgpuImageUsageFlags usage,
  CgpuMemoryPropertyFlags memory_properties,
  cgpu_image* p_image)
{
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }

  p_image->handle = resource_store_create_handle(&iimage_store);

  cgpu_iimage* iimage;
  if (!cgpu_resolve_image(*p_image, &iimage)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }

  VkImageTiling vk_image_tiling = VK_IMAGE_TILING_OPTIMAL;
  if ((usage & CGPU_IMAGE_USAGE_FLAG_TRANSFER_SRC)
        == CGPU_IMAGE_USAGE_FLAG_TRANSFER_SRC) {
    vk_image_tiling = VK_IMAGE_TILING_LINEAR;
  }
  else if ((usage & CGPU_IMAGE_USAGE_FLAG_TRANSFER_DST)
        == CGPU_IMAGE_USAGE_FLAG_TRANSFER_DST) {
    vk_image_tiling = VK_IMAGE_TILING_LINEAR;
  }

  VkImageUsageFlags vk_image_usage = 0;
  if ((usage & CGPU_IMAGE_USAGE_FLAG_TRANSFER_SRC)
        == CGPU_IMAGE_USAGE_FLAG_TRANSFER_SRC) {
    vk_image_usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  }
  if ((usage & CGPU_IMAGE_USAGE_FLAG_TRANSFER_DST)
        == CGPU_IMAGE_USAGE_FLAG_TRANSFER_DST) {
    vk_image_usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  }
  if ((usage & CGPU_IMAGE_USAGE_FLAG_SAMPLED)
        == CGPU_IMAGE_USAGE_FLAG_SAMPLED) {
    vk_image_usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
  }
  if ((usage & CGPU_IMAGE_USAGE_FLAG_STORAGE)
        == CGPU_IMAGE_USAGE_FLAG_STORAGE) {
    vk_image_usage |= VK_IMAGE_USAGE_STORAGE_BIT;
  }

  const VkFormat vk_format = cgpu_translate_image_format(format);

  VkImageCreateInfo image_info;
  image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  image_info.pNext = NULL;
  image_info.flags = 0;
  image_info.imageType = VK_IMAGE_TYPE_2D;
  image_info.format = vk_format;
  image_info.extent.width = width;
  image_info.extent.height = height;
  image_info.extent.depth = 1;
  image_info.mipLevels = 1;
  image_info.arrayLayers = 1;
  image_info.samples = VK_SAMPLE_COUNT_1_BIT;
  image_info.tiling = vk_image_tiling;
  image_info.usage = vk_image_usage;
  image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  image_info.queueFamilyIndexCount = 0;
  image_info.pQueueFamilyIndices = NULL;
  image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

  VkResult result = idevice->table.vkCreateImage(
    idevice->logical_device,
    &image_info,
    NULL,
    &iimage->image
  );
  if (result != VK_SUCCESS) {
    resource_store_free_handle(&iimage_store, p_image->handle);
    return CGPU_FAIL_UNABLE_TO_CREATE_IMAGE;
  }

  VkPhysicalDeviceMemoryProperties physical_device_memory_properties;
  vkGetPhysicalDeviceMemoryProperties(
    idevice->physical_device,
    &physical_device_memory_properties
  );

  VkMemoryRequirements mem_requirements;
  idevice->table.vkGetImageMemoryRequirements(
    idevice->logical_device,
    iimage->image,
    &mem_requirements
  );

  const VkMemoryPropertyFlags mem_flags =
      cgpu_translate_memory_properties(memory_properties);

  int32_t mem_index = -1;
  for (uint32_t i = 0; i < physical_device_memory_properties.memoryTypeCount; ++i) {
    if ((mem_requirements.memoryTypeBits & (1 << i)) &&
        (physical_device_memory_properties.memoryTypes[i].propertyFlags & mem_flags) == mem_flags) {
      mem_index = i;
      break;
    }
  }
  if (mem_index == -1) {
    resource_store_free_handle(&iimage_store, p_image->handle);
    idevice->table.vkDestroyImage(idevice->logical_device, iimage->image, NULL);
    return CGPU_FAIL_NO_SUITABLE_MEMORY_TYPE;
  }

  VkMemoryAllocateInfo mem_alloc_info;
  mem_alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  mem_alloc_info.pNext = NULL;
  mem_alloc_info.allocationSize = mem_requirements.size;
  mem_alloc_info.memoryTypeIndex = mem_index;

  result = idevice->table.vkAllocateMemory(
    idevice->logical_device,
    &mem_alloc_info,
    NULL,
    &iimage->memory
  );
  if (result != VK_SUCCESS) {
    resource_store_free_handle(&iimage_store, p_image->handle);
    idevice->table.vkDestroyImage(idevice->logical_device, iimage->image, NULL);
    return CGPU_FAIL_UNABLE_TO_ALLOCATE_MEMORY;
  }

  idevice->table.vkBindImageMemory(
    idevice->logical_device,
    iimage->image,
    iimage->memory,
    0
  );

  iimage->size = mem_requirements.size;

  VkImageViewCreateInfo image_view_info;
  image_view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  image_view_info.pNext = NULL;
  image_view_info.flags = 0;
  image_view_info.image = iimage->image;
  image_view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
  image_view_info.format = vk_format;
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
    idevice->table.vkDestroyImage(idevice->logical_device, iimage->image, NULL);
    idevice->table.vkFreeMemory(
      idevice->logical_device,
      iimage->memory,
      NULL
    );
  }

  return CGPU_OK;
}

CgpuResult cgpu_destroy_image(
  cgpu_device device,
  cgpu_image image)
{
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  cgpu_iimage* iimage;
  if (!cgpu_resolve_image(image, &iimage)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }

  idevice->table.vkDestroyImageView(
    idevice->logical_device,
    iimage->image_view,
    NULL
  );

  idevice->table.vkDestroyImage(
    idevice->logical_device,
    iimage->image,
    NULL
  );

  idevice->table.vkFreeMemory(
    idevice->logical_device,
    iimage->memory,
    NULL
  );

  resource_store_free_handle(&iimage_store, image.handle);

  return CGPU_OK;
}

CgpuResult cgpu_map_image(
  cgpu_device device,
  cgpu_image image,
  uint64_t offset,
  uint64_t size,
  void** pp_mapped_mem)
{
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  cgpu_iimage* iimage;
  if (!cgpu_resolve_image(image, &iimage)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }

  const VkResult result = idevice->table.vkMapMemory(
    idevice->logical_device,
    iimage->memory,
    offset,
    (size == CGPU_WHOLE_SIZE) ? iimage->size : size,
    0,
    pp_mapped_mem
  );

  if (result != VK_SUCCESS) {
    return CGPU_FAIL_UNABLE_TO_MAP_MEMORY;
  }

  return CGPU_OK;
}

CgpuResult cgpu_unmap_image(
  cgpu_device device,
  cgpu_image image)
{
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  cgpu_iimage* iimage;
  if (!cgpu_resolve_image(image, &iimage)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  idevice->table.vkUnmapMemory(
    idevice->logical_device,
    iimage->memory
  );
  return CGPU_OK;
}

CgpuResult cgpu_create_pipeline(
  cgpu_device device,
  uint32_t buffer_resource_count,
  const cgpu_shader_resource_buffer* p_buffer_resources,
  uint32_t image_resource_count,
  const cgpu_shader_resource_image* p_image_resources,
  cgpu_shader shader,
  const char* p_shader_entry_point,
  uint32_t push_constants_size,
  cgpu_pipeline* p_pipeline)
{
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  cgpu_ishader* ishader;
  if (!cgpu_resolve_shader(shader, &ishader)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }

  p_pipeline->handle = resource_store_create_handle(&ipipeline_store);

  cgpu_ipipeline* ipipeline;
  if (!cgpu_resolve_pipeline(*p_pipeline, &ipipeline)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }

  VkDescriptorSetLayoutBinding descriptor_set_bindings[MAX_DESCRIPTOR_SET_BINDINGS];

  for (uint32_t i = 0; i < buffer_resource_count; ++i)
  {
    const cgpu_shader_resource_buffer* shader_resource_buffer = &p_buffer_resources[i];
    VkDescriptorSetLayoutBinding* descriptor_set_layout_binding = &descriptor_set_bindings[i];
    descriptor_set_layout_binding->binding = shader_resource_buffer->binding;
    descriptor_set_layout_binding->descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_set_layout_binding->descriptorCount = 1;
    descriptor_set_layout_binding->stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    descriptor_set_layout_binding->pImmutableSamplers = NULL;
  }

  for (uint32_t i = 0; i < image_resource_count; ++i)
  {
    const cgpu_shader_resource_image* shader_resource_buffer = &p_image_resources[i];
    VkDescriptorSetLayoutBinding* descriptor_set_layout_binding =
        &descriptor_set_bindings[buffer_resource_count + i];
    descriptor_set_layout_binding->binding = shader_resource_buffer->binding;
    descriptor_set_layout_binding->descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    descriptor_set_layout_binding->descriptorCount = 1;
    descriptor_set_layout_binding->stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    descriptor_set_layout_binding->pImmutableSamplers = NULL;
  }

  const uint32_t desc_set_binding_count = buffer_resource_count + image_resource_count;

  VkDescriptorSetLayoutCreateInfo descriptor_set_layout_create_info;
  descriptor_set_layout_create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  descriptor_set_layout_create_info.pNext = NULL;
  descriptor_set_layout_create_info.flags = 0;
  descriptor_set_layout_create_info.bindingCount = desc_set_binding_count;
  descriptor_set_layout_create_info.pBindings = descriptor_set_bindings;

  VkResult result = idevice->table.vkCreateDescriptorSetLayout(
    idevice->logical_device,
    &descriptor_set_layout_create_info,
    NULL,
    &ipipeline->descriptor_set_layout
  );

  if (result != VK_SUCCESS) {
    resource_store_free_handle(&ipipeline_store, p_pipeline->handle);
    return CGPU_FAIL_UNABLE_TO_CREATE_DESCRIPTOR_LAYOUT;
  }

  VkPushConstantRange push_const_range;
  push_const_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  push_const_range.offset = 0;
  push_const_range.size = push_constants_size;

  VkPipelineLayoutCreateInfo pipeline_layout_create_info;
  pipeline_layout_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipeline_layout_create_info.pNext = NULL;
  pipeline_layout_create_info.flags = 0;
  pipeline_layout_create_info.setLayoutCount = 1;
  pipeline_layout_create_info.pSetLayouts = &ipipeline->descriptor_set_layout;
  pipeline_layout_create_info.pushConstantRangeCount = push_constants_size ? 1 : 0;
  pipeline_layout_create_info.pPushConstantRanges = &push_const_range;

  result = idevice->table.vkCreatePipelineLayout(
    idevice->logical_device,
    &pipeline_layout_create_info,
    NULL,
    &ipipeline->layout
  );
  if (result != VK_SUCCESS) {
    resource_store_free_handle(&ipipeline_store, p_pipeline->handle);
    idevice->table.vkDestroyDescriptorSetLayout(
      idevice->logical_device,
      ipipeline->descriptor_set_layout,
      NULL
    );
    return CGPU_FAIL_UNABLE_TO_CREATE_PIPELINE_LAYOUT;
  }

  VkPipelineShaderStageCreateInfo pipeline_shader_stage_create_info;
  pipeline_shader_stage_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  pipeline_shader_stage_create_info.pNext = NULL;
  pipeline_shader_stage_create_info.flags = 0;
  pipeline_shader_stage_create_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  pipeline_shader_stage_create_info.module = ishader->module;
  pipeline_shader_stage_create_info.pName = p_shader_entry_point;
  pipeline_shader_stage_create_info.pSpecializationInfo = NULL;

  VkComputePipelineCreateInfo pipeline_create_info;
  pipeline_create_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipeline_create_info.pNext = NULL;
  pipeline_create_info.flags = VK_PIPELINE_CREATE_DISPATCH_BASE;
  pipeline_create_info.stage = pipeline_shader_stage_create_info;
  pipeline_create_info.layout = ipipeline->layout;
  pipeline_create_info.basePipelineHandle = VK_NULL_HANDLE;
  pipeline_create_info.basePipelineIndex = 0;

  result = idevice->table.vkCreateComputePipelines(
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
    return CGPU_FAIL_UNABLE_TO_CREATE_COMPUTE_PIPELINE;
  }

  VkDescriptorPoolSize descriptor_pool_size;
  descriptor_pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  descriptor_pool_size.descriptorCount = desc_set_binding_count;

  VkDescriptorPoolCreateInfo descriptor_pool_create_info;
  descriptor_pool_create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  descriptor_pool_create_info.pNext = NULL;
  descriptor_pool_create_info.flags = 0;
  descriptor_pool_create_info.maxSets = 1;
  descriptor_pool_create_info.poolSizeCount = 1;
  descriptor_pool_create_info.pPoolSizes = &descriptor_pool_size;

  result = idevice->table.vkCreateDescriptorPool(
    idevice->logical_device,
    &descriptor_pool_create_info,
    NULL,
    &ipipeline->descriptor_pool
  );
  if (result != VK_SUCCESS) {
    resource_store_free_handle(&ipipeline_store, p_pipeline->handle);
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
    return CGPU_FAIL_UNABLE_TO_CREATE_DESCRIPTOR_POOL;
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
    resource_store_free_handle(&ipipeline_store, p_pipeline->handle);
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
    return CGPU_FAIL_UNABLE_TO_ALLOCATE_DESCRIPTOR_SET;
  }

  VkDescriptorBufferInfo descriptor_buffer_infos[MAX_DESCRIPTOR_BUFFER_INFOS];
  VkDescriptorImageInfo descriptor_image_infos[MAX_DESCRIPTOR_IMAGE_INFOS];
  VkWriteDescriptorSet write_descriptor_sets[MAX_WRITE_DESCRIPTOR_SETS];

  uint32_t write_desc_set_count = 0;

  for (uint32_t i = 0; i < buffer_resource_count; ++i)
  {
    const cgpu_shader_resource_buffer* shader_resource_buffer = &p_buffer_resources[i];

    cgpu_ibuffer* ibuffer;
    const cgpu_buffer buffer = shader_resource_buffer->buffer;
    if (!cgpu_resolve_buffer(buffer, &ibuffer)) {
      return CGPU_FAIL_INVALID_HANDLE;
    }

    if ((shader_resource_buffer->offset % idevice->limits.minStorageBufferOffsetAlignment) != 0) {
      return CGPU_FAIL_BUFFER_OFFSET_NOT_ALIGNED;
    }

    VkDescriptorBufferInfo* descriptor_buffer_info = &descriptor_buffer_infos[i];
    descriptor_buffer_info->buffer = ibuffer->buffer;
    descriptor_buffer_info->offset = shader_resource_buffer->offset;
    descriptor_buffer_info->range =
      (shader_resource_buffer->size == CGPU_WHOLE_SIZE) ?
        ibuffer->size - shader_resource_buffer->offset :
        shader_resource_buffer->size;

    VkWriteDescriptorSet* write_descriptor_set = &write_descriptor_sets[write_desc_set_count];
    write_descriptor_set->sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write_descriptor_set->pNext = NULL;
    write_descriptor_set->dstSet = ipipeline->descriptor_set;
    write_descriptor_set->dstBinding = shader_resource_buffer->binding;
    write_descriptor_set->dstArrayElement = 0;
    write_descriptor_set->descriptorCount = 1;
    write_descriptor_set->descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write_descriptor_set->pImageInfo = NULL;
    write_descriptor_set->pBufferInfo = descriptor_buffer_info;
    write_descriptor_set->pTexelBufferView = NULL;
    write_desc_set_count++;
  }

  for (uint32_t i = 0; i < image_resource_count; ++i)
  {
    const cgpu_shader_resource_image* shader_resource_image = &p_image_resources[i];

    cgpu_iimage* iimage;
    const cgpu_image image = shader_resource_image->image;
    if (!cgpu_resolve_image(image, &iimage)) {
      return CGPU_FAIL_INVALID_HANDLE;
    }

    VkDescriptorImageInfo* descriptor_image_info = &descriptor_image_infos[i];
    descriptor_image_info->sampler = idevice->sampler;
    descriptor_image_info->imageView = iimage->image_view;
    descriptor_image_info->imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkWriteDescriptorSet* write_descriptor_set =
      &write_descriptor_sets[write_desc_set_count];
    write_descriptor_set->sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write_descriptor_set->pNext = NULL;
    write_descriptor_set->dstSet = ipipeline->descriptor_set;
    write_descriptor_set->dstBinding = shader_resource_image->binding;
    write_descriptor_set->dstArrayElement = 0;
    write_descriptor_set->descriptorCount = 1;
    write_descriptor_set->descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    write_descriptor_set->pImageInfo = descriptor_image_info;
    write_descriptor_set->pBufferInfo = NULL;
    write_descriptor_set->pTexelBufferView = NULL;
    write_desc_set_count++;
  }

  idevice->table.vkUpdateDescriptorSets(
    idevice->logical_device,
    write_desc_set_count,
    write_descriptor_sets,
    0,
    NULL
  );

  return CGPU_OK;
}

CgpuResult cgpu_destroy_pipeline(
  cgpu_device device,
  cgpu_pipeline pipeline)
{
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  cgpu_ipipeline* ipipeline;
  if (!cgpu_resolve_pipeline(pipeline, &ipipeline)) {
    return CGPU_FAIL_INVALID_HANDLE;
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

  return CGPU_OK;
}

CgpuResult cgpu_create_command_buffer(
  cgpu_device device,
  cgpu_command_buffer* p_command_buffer)
{
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }

  p_command_buffer->handle = resource_store_create_handle(&icommand_buffer_store);

  cgpu_icommand_buffer* icommand_buffer;
  if (!cgpu_resolve_command_buffer(*p_command_buffer, &icommand_buffer)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  icommand_buffer->device.handle = device.handle;

  VkCommandBufferAllocateInfo cmdbuf_alloc_info;
  cmdbuf_alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cmdbuf_alloc_info.pNext = NULL;
  cmdbuf_alloc_info.commandPool = idevice->command_pool;
  cmdbuf_alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cmdbuf_alloc_info.commandBufferCount = 1;

  const VkResult result = idevice->table.vkAllocateCommandBuffers(
    idevice->logical_device,
    &cmdbuf_alloc_info,
    &icommand_buffer->command_buffer
  );
  if (result != VK_SUCCESS) {
    return CGPU_FAIL_UNABLE_TO_ALLOCATE_COMMAND_BUFFER;
  }

  return CGPU_OK;
}

CgpuResult cgpu_destroy_command_buffer(
  cgpu_device device,
  cgpu_command_buffer command_buffer)
{
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  cgpu_icommand_buffer* icommand_buffer;
  if (!cgpu_resolve_command_buffer(command_buffer, &icommand_buffer)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }

  idevice->table.vkFreeCommandBuffers(
    idevice->logical_device,
    idevice->command_pool,
    1,
    &icommand_buffer->command_buffer
  );

  resource_store_free_handle(&icommand_buffer_store, command_buffer.handle);
  return CGPU_OK;
}

CgpuResult cgpu_begin_command_buffer(
  cgpu_command_buffer command_buffer)
{
  cgpu_icommand_buffer* icommand_buffer;
  if (!cgpu_resolve_command_buffer(command_buffer, &icommand_buffer)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(icommand_buffer->device, &idevice)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }

  VkCommandBufferBeginInfo command_buffer_begin_info;
  command_buffer_begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  command_buffer_begin_info.pNext = NULL;
  command_buffer_begin_info.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
  command_buffer_begin_info.pInheritanceInfo = NULL;

  const VkResult result = idevice->table.vkBeginCommandBuffer(
    icommand_buffer->command_buffer,
    &command_buffer_begin_info
  );

  if (result != VK_SUCCESS) {
    return CGPU_FAIL_UNABLE_TO_BEGIN_COMMAND_BUFFER;
  }
  return CGPU_OK;
}

CgpuResult cgpu_cmd_bind_pipeline(
  cgpu_command_buffer command_buffer,
  cgpu_pipeline pipeline)
{
  cgpu_icommand_buffer* icommand_buffer;
  if (!cgpu_resolve_command_buffer(command_buffer, &icommand_buffer)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(icommand_buffer->device, &idevice)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  cgpu_ipipeline* ipipeline;
  if (!cgpu_resolve_pipeline(pipeline, &ipipeline)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  idevice->table.vkCmdBindPipeline(
    icommand_buffer->command_buffer,
    VK_PIPELINE_BIND_POINT_COMPUTE,
    ipipeline->pipeline
  );
  idevice->table.vkCmdBindDescriptorSets(
    icommand_buffer->command_buffer,
    VK_PIPELINE_BIND_POINT_COMPUTE,
    ipipeline->layout,
    0,
    1,
    &ipipeline->descriptor_set,
    0,
    0
  );
  return CGPU_OK;
}

CgpuResult cgpu_cmd_copy_buffer(
  cgpu_command_buffer command_buffer,
  cgpu_buffer source_buffer,
  uint64_t source_offset,
  cgpu_buffer destination_buffer,
  uint64_t destination_offset,
  uint64_t size)
{
  cgpu_icommand_buffer* icommand_buffer;
  if (!cgpu_resolve_command_buffer(command_buffer, &icommand_buffer)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(icommand_buffer->device, &idevice)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  cgpu_ibuffer* isource_buffer;
  if (!cgpu_resolve_buffer(source_buffer, &isource_buffer)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  cgpu_ibuffer* idestination_buffer;
  if (!cgpu_resolve_buffer(destination_buffer, &idestination_buffer)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }

  VkBufferCopy region;
  region.srcOffset = source_offset;
  region.dstOffset = destination_offset;
  region.size =
    (size == CGPU_WHOLE_SIZE) ? isource_buffer->size : size;

  idevice->table.vkCmdCopyBuffer(
    icommand_buffer->command_buffer,
    isource_buffer->buffer,
    idestination_buffer->buffer,
    1,
    &region
  );

  return CGPU_OK;
}

CgpuResult cgpu_cmd_push_constants(
  cgpu_command_buffer command_buffer,
  cgpu_pipeline pipeline,
  uint32_t size,
  const void* p_data)
{
  cgpu_icommand_buffer* icommand_buffer;
  if (!cgpu_resolve_command_buffer(command_buffer, &icommand_buffer)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(icommand_buffer->device, &idevice)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  cgpu_ipipeline* ipipeline;
  if (!cgpu_resolve_pipeline(pipeline, &ipipeline)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  idevice->table.vkCmdPushConstants(
    icommand_buffer->command_buffer,
    ipipeline->layout,
    VK_SHADER_STAGE_COMPUTE_BIT,
    0,
    size,
    p_data
  );
  return CGPU_OK;
}

CgpuResult cgpu_cmd_dispatch(
  cgpu_command_buffer command_buffer,
  uint32_t dim_x,
  uint32_t dim_y,
  uint32_t dim_z)
{
  cgpu_icommand_buffer* icommand_buffer;
  if (!cgpu_resolve_command_buffer(command_buffer, &icommand_buffer)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(icommand_buffer->device, &idevice)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  idevice->table.vkCmdDispatch(
    icommand_buffer->command_buffer,
    dim_x,
    dim_y,
    dim_z
  );
  return CGPU_OK;
}

CgpuResult cgpu_cmd_pipeline_barrier(
  cgpu_command_buffer command_buffer,
  uint32_t barrier_count,
  const cgpu_memory_barrier* p_barriers,
  uint32_t buffer_barrier_count,
  const cgpu_buffer_memory_barrier* p_buffer_barriers,
  uint32_t image_barrier_count,
  const cgpu_image_memory_barrier* p_image_barriers)
{
  cgpu_icommand_buffer* icommand_buffer;
  if (!cgpu_resolve_command_buffer(command_buffer, &icommand_buffer)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(icommand_buffer->device, &idevice)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }

  VkMemoryBarrier vk_memory_barriers[MAX_MEMORY_BARRIERS];

  for (uint32_t i = 0; i < barrier_count; ++i)
  {
    const cgpu_memory_barrier* b_cgpu = &p_barriers[i];
    VkMemoryBarrier* b_vk = &vk_memory_barriers[i];
    b_vk->sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    b_vk->pNext = NULL;
    b_vk->srcAccessMask = cgpu_translate_access_flags(b_cgpu->src_access_flags);
    b_vk->dstAccessMask = cgpu_translate_access_flags(b_cgpu->dst_access_flags);
  }

  VkBufferMemoryBarrier vk_buffer_memory_barriers[MAX_BUFFER_MEMORY_BARRIERS];
  VkImageMemoryBarrier vk_image_memory_barriers[MAX_IMAGE_MEMORY_BARRIERS];

  for (uint32_t i = 0; i < buffer_barrier_count; ++i)
  {
    const cgpu_buffer_memory_barrier* b_cgpu = &p_buffer_barriers[i];

    cgpu_ibuffer* ibuffer;
    if (!cgpu_resolve_buffer(b_cgpu->buffer, &ibuffer)) {
      return CGPU_FAIL_INVALID_HANDLE;
    }

    VkBufferMemoryBarrier* b_vk = &vk_buffer_memory_barriers[i];
    b_vk->sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    b_vk->pNext = NULL;
    b_vk->srcAccessMask = cgpu_translate_access_flags(b_cgpu->src_access_flags);
    b_vk->dstAccessMask = cgpu_translate_access_flags(b_cgpu->dst_access_flags);
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
      return CGPU_FAIL_INVALID_HANDLE;
    }

    VkImageMemoryBarrier* b_vk = &vk_image_memory_barriers[i];
    b_vk->sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b_vk->pNext = NULL;
    b_vk->srcAccessMask = cgpu_translate_access_flags(b_cgpu->src_access_flags);
    b_vk->dstAccessMask = cgpu_translate_access_flags(b_cgpu->dst_access_flags);
    b_vk->oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    b_vk->newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    b_vk->srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b_vk->dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b_vk->image = iimage->image;
    b_vk->subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    b_vk->subresourceRange.baseMipLevel = 0;
    b_vk->subresourceRange.levelCount = 1;
    b_vk->subresourceRange.baseArrayLayer = 0;
    b_vk->subresourceRange.layerCount = 1;
  }

  idevice->table.vkCmdPipelineBarrier(
    icommand_buffer->command_buffer,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
      VK_PIPELINE_STAGE_TRANSFER_BIT,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
      VK_PIPELINE_STAGE_TRANSFER_BIT,
    0,
    barrier_count,
    vk_memory_barriers,
    buffer_barrier_count,
    vk_buffer_memory_barriers,
    image_barrier_count,
    vk_image_memory_barriers
  );

  return CGPU_OK;
}

CGPU_API CgpuResult CGPU_CDECL cgpu_cmd_reset_timestamps(
  cgpu_command_buffer command_buffer,
  uint32_t offset,
  uint32_t count)
{
  cgpu_icommand_buffer* icommand_buffer;
  if (!cgpu_resolve_command_buffer(command_buffer, &icommand_buffer)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(icommand_buffer->device, &idevice)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }

  idevice->table.vkCmdResetQueryPool(
    icommand_buffer->command_buffer,
    idevice->timestamp_pool,
    offset,
    count
  );

  return CGPU_OK;
}

CGPU_API CgpuResult CGPU_CDECL cgpu_cmd_write_timestamp(
  cgpu_command_buffer command_buffer,
  uint32_t timestamp_index)
{
  cgpu_icommand_buffer* icommand_buffer;
  if (!cgpu_resolve_command_buffer(command_buffer, &icommand_buffer)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(icommand_buffer->device, &idevice)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }

  idevice->table.vkCmdWriteTimestamp(
    icommand_buffer->command_buffer,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    idevice->timestamp_pool,
    timestamp_index
  );

  return CGPU_OK;
}

CgpuResult cgpu_cmd_copy_timestamps(
  cgpu_command_buffer command_buffer,
  cgpu_buffer buffer,
  uint32_t offset,
  uint32_t count,
  bool wait_until_available)
{
  if ((offset + count) > MAX_TIMESTAMP_QUERIES) {
    return CGPU_FAIL_MAX_TIMESTAMP_QUERY_INDEX_REACHED;
  }

  cgpu_icommand_buffer* icommand_buffer;
  if (!cgpu_resolve_command_buffer(command_buffer, &icommand_buffer)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(icommand_buffer->device, &idevice)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  cgpu_ibuffer* ibuffer;
  if (!cgpu_resolve_buffer(buffer, &ibuffer)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }

  idevice->table.vkCmdCopyQueryPoolResults(
    icommand_buffer->command_buffer,
    idevice->timestamp_pool,
    offset,
    count,
    ibuffer->buffer,
    0,
    sizeof(uint64_t),
    VK_QUERY_RESULT_64_BIT | (wait_until_available ?
      VK_QUERY_RESULT_WAIT_BIT : VK_QUERY_RESULT_WITH_AVAILABILITY_BIT)
  );

  return CGPU_OK;
}

CgpuResult cgpu_end_command_buffer(
  cgpu_command_buffer command_buffer)
{
  cgpu_icommand_buffer* icommand_buffer;
  if (!cgpu_resolve_command_buffer(command_buffer, &icommand_buffer)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(icommand_buffer->device, &idevice)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  idevice->table.vkEndCommandBuffer(icommand_buffer->command_buffer);
  return CGPU_OK;
}

CgpuResult cgpu_create_fence(
  cgpu_device device,
  cgpu_fence* p_fence)
{
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }

  p_fence->handle = resource_store_create_handle(&ifence_store);

  cgpu_ifence* ifence;
  if (!cgpu_resolve_fence(*p_fence, &ifence)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }

  VkFenceCreateInfo fence_create_info;
  fence_create_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fence_create_info.pNext = NULL;
  fence_create_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

  const VkResult result = idevice->table.vkCreateFence(
    idevice->logical_device,
    &fence_create_info,
    NULL,
    &ifence->fence
  );
  if (result != VK_SUCCESS) {
    resource_store_free_handle(&ifence_store, p_fence->handle);
    return CGPU_FAIL_UNABLE_TO_CREATE_FENCE;
  }
  return CGPU_OK;
}

CgpuResult cgpu_destroy_fence(
  cgpu_device device,
  cgpu_fence fence)
{
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  cgpu_ifence* ifence;
  if (!cgpu_resolve_fence(fence, &ifence)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  idevice->table.vkDestroyFence(
    idevice->logical_device,
    ifence->fence,
    NULL
  );
  resource_store_free_handle(&ifence_store, fence.handle);
  return CGPU_OK;
}

CgpuResult cgpu_reset_fence(
  cgpu_device device,
  cgpu_fence fence)
{
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  cgpu_ifence* ifence;
  if (!cgpu_resolve_fence(fence, &ifence)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  const VkResult result = idevice->table.vkResetFences(
    idevice->logical_device,
    1,
    &ifence->fence
  );
  if (result != VK_SUCCESS) {
    return CGPU_FAIL_UNABLE_TO_RESET_FENCE;
  }
  return CGPU_OK;
}

CgpuResult cgpu_wait_for_fence(
  cgpu_device device,
  cgpu_fence fence)
{
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  cgpu_ifence* ifence;
  if (!cgpu_resolve_fence(fence, &ifence)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  const VkResult result = idevice->table.vkWaitForFences(
    idevice->logical_device,
    1,
    &ifence->fence,
    VK_TRUE,
    UINT64_MAX
  );
  if (result != VK_SUCCESS) {
    return CGPU_FAIL_UNABLE_TO_WAIT_FOR_FENCE;
  }
  return CGPU_OK;
}

CgpuResult cgpu_submit_command_buffer(
  cgpu_device device,
  cgpu_command_buffer command_buffer,
  cgpu_fence fence)
{
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  cgpu_icommand_buffer* icommand_buffer;
  if (!cgpu_resolve_command_buffer(command_buffer, &icommand_buffer)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  cgpu_ifence* ifence;
  if (!cgpu_resolve_fence(fence, &ifence)) {
    return CGPU_FAIL_INVALID_HANDLE;
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

  const VkResult result = idevice->table.vkQueueSubmit(
    idevice->compute_queue,
    1,
    &submit_info,
    ifence->fence
  );

  if (result != VK_SUCCESS) {
    return CGPU_FAIL_UNABLE_TO_SUBMIT_COMMAND_BUFFER;
  }
  return CGPU_OK;
}

CgpuResult cgpu_flush_mapped_memory(
  cgpu_device device,
  cgpu_buffer buffer,
  uint64_t offset,
  uint64_t size)
{
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  cgpu_ibuffer* ibuffer;
  if (!cgpu_resolve_buffer(buffer, &ibuffer)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }

  VkMappedMemoryRange memory_range;
  memory_range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
  memory_range.pNext = NULL;
  memory_range.memory = ibuffer->memory;
  memory_range.offset = offset;
  memory_range.size =
    (size == CGPU_WHOLE_SIZE) ? ibuffer->size : size;

  const VkResult result = idevice->table.vkFlushMappedMemoryRanges(
    idevice->logical_device,
    1,
    &memory_range
  );

  if (result != VK_SUCCESS) {
    return CGPU_FAIL_UNABLE_TO_INVALIDATE_MEMORY;
  }
  return CGPU_OK;
}

CgpuResult cgpu_invalidate_mapped_memory(
  cgpu_device device,
  cgpu_buffer buffer,
  uint64_t offset,
  uint64_t size)
{
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  cgpu_ibuffer* ibuffer;
  if (!cgpu_resolve_buffer(buffer, &ibuffer)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }

  VkMappedMemoryRange memory_range;
  memory_range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
  memory_range.pNext = NULL;
  memory_range.memory = ibuffer->memory;
  memory_range.offset = offset;
  memory_range.size =
    (size == CGPU_WHOLE_SIZE) ? ibuffer->size : size;

  const VkResult result = idevice->table.vkInvalidateMappedMemoryRanges(
    idevice->logical_device,
    1,
    &memory_range
  );

  if (result != VK_SUCCESS) {
    return CGPU_FAIL_UNABLE_TO_INVALIDATE_MEMORY;
  }
  return CGPU_OK;
}

CgpuResult cgpu_get_physical_device_limits(
  cgpu_device device,
  cgpu_physical_device_limits* p_limits)
{
  cgpu_idevice* idevice;
  if (!cgpu_resolve_device(device, &idevice)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  memcpy(
    p_limits,
    &idevice->limits,
    sizeof(cgpu_physical_device_limits)
  );
  return CGPU_OK;
}
