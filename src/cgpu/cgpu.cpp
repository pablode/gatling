#include "cgpu.hpp"
#include "handle_store.hpp"

#include <stdint.h>
#include <stddef.h>
#include <array>
#include <vector>
#include <string>
#include <cstdint>
#include <string_view>
#include <volk.h>

using namespace cgpu;

// Internal structures.

struct _gpu_instance
{
  VkInstance instance;
};

struct _gpu_device
{
  VkDevice                    logical_device;
  VkPhysicalDevice            physical_device;
  VkQueue                     compute_queue;
  VkCommandPool               command_pool;
  VolkDeviceTable             table;
  cgpu_physical_device_limits limits;
};

struct _gpu_buffer
{
  VkBuffer       buffer;
  VkDeviceMemory memory;
  uint64_t       size_in_bytes;
};

struct _gpu_image
{
  VkImage        image;
  VkDeviceMemory memory;
  uint64_t       size_in_bytes;
};

struct _gpu_pipeline
{
  VkPipeline            pipeline;
  VkPipelineLayout      layout;
  VkDescriptorSetLayout descriptor_set_layout;
  VkDescriptorSet       descriptor_set;
  VkDescriptorPool      descriptor_pool;
};

struct _gpu_shader
{
  VkShaderModule module;
};

struct _gpu_fence
{
  VkFence fence;
};

struct _gpu_command_buffer
{
  VkCommandBuffer command_buffer;
};

// Handle and structure storage.

handle_store gpu_device_store;
handle_store gpu_shader_store;
handle_store gpu_buffer_store;
handle_store gpu_image_store;
handle_store gpu_pipeline_store;
handle_store gpu_command_buffer_store;
handle_store gpu_fence_store;

std::vector<_gpu_device>           gpu_devices;
std::vector<_gpu_shader>           gpu_shaders;
std::vector<_gpu_buffer>           gpu_buffers;
std::vector<_gpu_image>            gpu_images;
std::vector<_gpu_pipeline>         gpu_pipelines;
std::vector<_gpu_command_buffer>   gpu_command_buffers;
std::vector<_gpu_fence>            gpu_fences;

_gpu_instance gpu_instance;

// Helper functions.

#ifndef NDEBUG

#define _cgpu_gen_resolve_handle_func(                      \
  HANDLE_TYPE, IDATA_TYPE, STORAGE_HANDLES, STORAGE_VECTOR) \
inline bool _cgpu_resolve_handle(                           \
  const uint64_t& handle,                                   \
  IDATA_TYPE** idata)                                       \
{                                                           \
  if (!STORAGE_HANDLES.is_valid(handle)) {                  \
    return false;                                           \
  }                                                         \
  const uint32_t index =                                    \
    STORAGE_HANDLES.extract_index(handle);                  \
  if (index >= STORAGE_VECTOR.size()) {                     \
    STORAGE_VECTOR.resize(index + 1);                       \
  }                                                         \
  *idata = &(STORAGE_VECTOR[index]);                         \
  return true;                                              \
}

#elif

#define _cgpu_gen_resolve_handle_func(                      \
  HANDLE_TYPE, IDATA_TYPE, STORAGE_HANDLES, STORAGE_VECTOR) \
inline bool _cgpu_resolve_handle(                           \
  const uint64_t& handle,                                   \
  IDATA_TYPE** idata)                                       \
{                                                           \
  const uint32_t index =                                    \
    STORAGE_HANDLES.extract_index(handle);                  \
  if (index >= STORAGE_VECTOR.size()) {                     \
    STORAGE_VECTOR.resize(index + 1);                       \
  }                                                         \
  idata = &STORAGE_VECTOR[index];                           \
  return true;                                              \
}

#endif

_cgpu_gen_resolve_handle_func(
  cgpu_device, _gpu_device, gpu_device_store, gpu_devices)
_cgpu_gen_resolve_handle_func(
  cgpu_buffer, _gpu_buffer, gpu_buffer_store, gpu_buffers)
_cgpu_gen_resolve_handle_func(
  cgpu_shader, _gpu_shader, gpu_shader_store, gpu_shaders)
_cgpu_gen_resolve_handle_func(
  cgpu_image, _gpu_image, gpu_image_store, gpu_images)
_cgpu_gen_resolve_handle_func(
  cgpu_pipeline, _gpu_pipeline, gpu_pipeline_store, gpu_pipelines)
_cgpu_gen_resolve_handle_func(
  cgpu_fence, _gpu_fence, gpu_fence_store, gpu_fences)
_cgpu_gen_resolve_handle_func(
  cgpu_command_buffer, _gpu_command_buffer,
  gpu_command_buffer_store, gpu_command_buffers)

VkMemoryPropertyFlags _cgpu_translate_memory_properties(
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

VkAccessFlags _cgpu_translate_access_flags(
  CgpuMemoryAccessFlags flags)
{
  VkAccessFlags vk_flags = {};
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
};

CgpuSampleCountFlags _cgpu_translate_sample_count_flags(
  const VkSampleCountFlags& vk_flags)
{
  CgpuSampleCountFlags flags = {};
  if ((vk_flags & VK_SAMPLE_COUNT_1_BIT)
        == VK_SAMPLE_COUNT_1_BIT) {
    flags |= CGPU_SAMPLE_COUNT_FLAG_1;
  }
  if ((vk_flags & VK_SAMPLE_COUNT_2_BIT)
        == VK_SAMPLE_COUNT_2_BIT) {
    flags |= CGPU_SAMPLE_COUNT_FLAG_2;
  }
  if ((vk_flags & VK_SAMPLE_COUNT_4_BIT)
        == VK_SAMPLE_COUNT_4_BIT) {
    flags |= CGPU_SAMPLE_COUNT_FLAG_4;
  }
  if ((vk_flags & VK_SAMPLE_COUNT_8_BIT)
        == VK_SAMPLE_COUNT_8_BIT) {
    flags |= CGPU_SAMPLE_COUNT_FLAG_8;
  }
  if ((vk_flags & VK_SAMPLE_COUNT_16_BIT)
        == VK_SAMPLE_COUNT_16_BIT) {
    flags |= CGPU_SAMPLE_COUNT_FLAG_16;
  }
  if ((vk_flags & VK_SAMPLE_COUNT_32_BIT)
        == VK_SAMPLE_COUNT_32_BIT) {
    flags |= CGPU_SAMPLE_COUNT_FLAG_32;
  }
  if ((vk_flags & VK_SAMPLE_COUNT_64_BIT)
        == VK_SAMPLE_COUNT_64_BIT) {
    flags |= CGPU_SAMPLE_COUNT_FLAG_64;
  }
  return flags;
}

cgpu_physical_device_limits _cgpu_translate_physical_device_limits(
  const VkPhysicalDeviceLimits& vk_limits)
{
  cgpu_physical_device_limits limits = {};
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
    _cgpu_translate_sample_count_flags(vk_limits.framebufferColorSampleCounts);
  limits.framebufferDepthSampleCounts =
    _cgpu_translate_sample_count_flags(vk_limits.framebufferDepthSampleCounts);
  limits.framebufferStencilSampleCounts =
    _cgpu_translate_sample_count_flags(vk_limits.framebufferStencilSampleCounts);
  limits.framebufferNoAttachmentsSampleCounts =
    _cgpu_translate_sample_count_flags(vk_limits.framebufferNoAttachmentsSampleCounts);
  limits.maxColorAttachments = vk_limits.maxColorAttachments;
  limits.sampledImageColorSampleCounts =
    _cgpu_translate_sample_count_flags(vk_limits.sampledImageColorSampleCounts);
  limits.sampledImageIntegerSampleCounts =
    _cgpu_translate_sample_count_flags(vk_limits.sampledImageIntegerSampleCounts);
  limits.sampledImageDepthSampleCounts =
    _cgpu_translate_sample_count_flags(vk_limits.sampledImageDepthSampleCounts);
  limits.sampledImageStencilSampleCounts =
    _cgpu_translate_sample_count_flags(vk_limits.sampledImageStencilSampleCounts);
  limits.storageImageSampleCounts =
    _cgpu_translate_sample_count_flags(vk_limits.storageImageSampleCounts);
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
  return limits;
}

VkFormat _cgpu_translate_image_format(
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

// API method implementation.

CgpuResult cgpu_initialize(
  const char* p_app_name,
  const uint32_t& version_major,
  const uint32_t& version_minor,
  const uint32_t& version_patch)
{
  VkResult result = volkInitialize();

  if (result != VK_SUCCESS) {
    return CGPU_FAIL_UNABLE_TO_INITIALIZE_VOLK;
  }

  std::vector<const char*> enabled_layers;
#ifndef NDEBUG
  {
    constexpr static std::array<const char*, 1> DEBUG_VALIDATION_LAYERS {{
        "VK_LAYER_LUNARG_standard_validation"
    }};

    uint32_t available_layer_count;
    vkEnumerateInstanceLayerProperties(&available_layer_count, nullptr);
    std::vector<VkLayerProperties> available_layers{available_layer_count};
    vkEnumerateInstanceLayerProperties(&available_layer_count, available_layers.data());

    const auto is_validation_layer_supported =
      [&available_layers](const char* layer_name)
    {
      for (const auto &layer_properties : available_layers) {
        if (std::strcmp(layer_name, layer_properties.layerName) == 0) {
          return true;
        }
      }
      return false;
    };

    for (const auto& layer_name : DEBUG_VALIDATION_LAYERS)
    {
      if (!is_validation_layer_supported(layer_name)) {
        std::printf("Validation layer %s not supported\n", layer_name);
        continue;
      }
      std::printf("Enabled validation layer %s\n", layer_name);
      enabled_layers.push_back(layer_name);
    }
  }
  std::printf("%zu validation layer(s) enabled\n", enabled_layers.size());
#endif

  std::vector<const char*> enabled_instance_extensions;
#ifndef NDEBUG
  {
    struct debug_instance_extension
    {
      const char* layerName;
      const char* extensionName;
    };

    constexpr static std::array<debug_instance_extension, 1> DEBUG_INSTANCE_EXTENSIONS {
      {nullptr, VK_EXT_DEBUG_UTILS_EXTENSION_NAME}
    };

    for (const debug_instance_extension& ext_desc : DEBUG_INSTANCE_EXTENSIONS)
    {
      const char* layer_name = ext_desc.layerName;
      const char* ext_name = ext_desc.extensionName;

      if (layer_name) {
        if (std::find(enabled_layers.begin(),
                      enabled_layers.end(),
                      layer_name) == enabled_layers.end())
        {
          std::printf("Layer %s for instance extension %s not available\n", layer_name, ext_name);
          continue;
        }
      }

      uint32_t available_extension_count;
      vkEnumerateInstanceExtensionProperties(
        layer_name,
        &available_extension_count,
        nullptr
      );

      std::vector<VkExtensionProperties> available_extensions{available_extension_count};
      vkEnumerateInstanceExtensionProperties(
        layer_name,
        &available_extension_count,
        available_extensions.data()
      );

      const auto is_extension_supported =
        [&available_extensions](const char* ext_name)
      {
        for (const auto &extension_properties : available_extensions) {
          if (std::strcmp(ext_name, extension_properties.extensionName) == 0) {
            return true;
          }
        }
        return false;
      };

      const bool ext_supported = is_extension_supported(ext_name);
      if (!ext_supported) {
        if (layer_name) {
          std::printf("Instance extension %s not supported by layer %s\n", ext_name, layer_name);
        } else {
          std::printf("Instance extension %s not supported\n", ext_name);
        }
        continue;
      }

      if (layer_name) {
        std::printf("Enabled instance extension %s for layer %s\n", ext_name, layer_name);
      } else {
        std::printf("Enabled instance extension %s\n", ext_name);
      }
      enabled_instance_extensions.push_back(ext_name);
    }
  }
  std::printf("%zu instance extension(s) enabled\n", enabled_instance_extensions.size());
#endif

  VkApplicationInfo app_info = {};
  app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app_info.pNext = nullptr;
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
  app_info.apiVersion = VK_API_VERSION_1_1;

  VkInstanceCreateInfo create_info = {};
  create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  create_info.pNext = nullptr;
  create_info.pApplicationInfo = &app_info;
  create_info.enabledExtensionCount = enabled_instance_extensions.size();
  create_info.ppEnabledExtensionNames = enabled_instance_extensions.data();
  create_info.enabledLayerCount = enabled_layers.size();
  create_info.ppEnabledLayerNames = enabled_layers.data();

  result = vkCreateInstance(
    &create_info,
    nullptr,
    &gpu_instance.instance
  );
  if (result != VK_SUCCESS) {
    return CGPU_FAIL_UNABLE_TO_INITIALIZE_VULKAN;
  }

  volkLoadInstance(gpu_instance.instance);

  return CGPU_OK;
}

CgpuResult cgpu_destroy()
{
  vkDestroyInstance(gpu_instance.instance, nullptr);
  return CGPU_OK;
}

CgpuResult cgpu_get_device_count(uint32_t* p_device_count)
{
  vkEnumeratePhysicalDevices(
    gpu_instance.instance,
    p_device_count,
    nullptr
  );
  return CGPU_OK;
}

CgpuResult cgpu_create_device(
  const uint32_t& index,
  const uint32_t& required_extension_count,
  const char** pp_required_extensions,
  cgpu_device& device)
{
  device.handle = gpu_device_store.create();

  _gpu_device* idevice;
  if (!_cgpu_resolve_handle(device.handle, &idevice)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }

  uint32_t num_phys_devices;
  vkEnumeratePhysicalDevices(
    gpu_instance.instance,
    &num_phys_devices,
    nullptr
  );

  if (num_phys_devices == 0 ||
      index > (num_phys_devices - 1))
  {
    gpu_device_store.free(device.handle);
    return CGPU_FAIL_NO_DEVICE_AT_INDEX;
  }

  std::vector<VkPhysicalDevice> phys_devices{num_phys_devices};
  vkEnumeratePhysicalDevices(
    gpu_instance.instance,
    &num_phys_devices,
    phys_devices.data()
  );

  idevice->physical_device = phys_devices[index];

  uint32_t num_device_extensions;
  vkEnumerateDeviceExtensionProperties(
    idevice->physical_device,
    nullptr,
    &num_device_extensions,
    nullptr
  );
  std::vector<VkExtensionProperties> device_extensions;
  device_extensions.resize(num_device_extensions);
  vkEnumerateDeviceExtensionProperties(
    idevice->physical_device,
    nullptr,
    &num_device_extensions,
    device_extensions.data()
  );

  const auto hash_device_extension =
    [&device_extensions](const char* extension_name)
  {
    for (const auto &extension : device_extensions) {
      if (std::strcmp(extension.extensionName, extension_name) == 0) {
        return true;
      }
    }
    return false;
  };
  for (uint32_t i = 0; i < required_extension_count; ++i)
  {
    const char* required_extension = *(pp_required_extensions + i);
    if (!hash_device_extension(required_extension)) {
      gpu_device_store.free(device.handle);
      return CGPU_FAIL_DEVICE_EXTENSION_NOT_SUPPORTED;
    }
  }

  uint32_t queue_family_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(
    idevice->physical_device,
    &queue_family_count,
    nullptr
  );
  std::vector<VkQueueFamilyProperties> queue_families{queue_family_count};
  vkGetPhysicalDeviceQueueFamilyProperties(
    idevice->physical_device,
    &queue_family_count,
    queue_families.data()
  );

  // Since raytracing is a continous, compute-heavy task, we don't need
  // to schedule work or translate command buffers very often. Therefore,
  // we also don't need async execution and can operate on a single queue.
  uint32_t queue_family_index = -1;
  uint32_t i = 0;
  for (const VkQueueFamilyProperties& queue_family : queue_families) {
    if (queue_family.queueFlags & VK_QUEUE_COMPUTE_BIT) {
      queue_family_index = i;
    }
    i++;
  }
  if (queue_family_index == -1) {
    gpu_device_store.free(device.handle);
    return CGPU_FAIL_DEVICE_HAS_NO_COMPUTE_QUEUE_FAMILY;
  }

  VkDeviceQueueCreateInfo queue_create_info = {};
  queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queue_create_info.pNext = nullptr;
  queue_create_info.queueFamilyIndex = queue_family_index;
  queue_create_info.queueCount = 1;
  const float queue_priority = 1.0f;
  queue_create_info.pQueuePriorities = &queue_priority;

  VkPhysicalDeviceFeatures device_features = {};

  VkDeviceCreateInfo device_create_info = {};
  device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  device_create_info.pNext = nullptr;
  device_create_info.queueCreateInfoCount = 1;
  device_create_info.pQueueCreateInfos = &queue_create_info;
  device_create_info.pEnabledFeatures = &device_features;
  device_create_info.enabledExtensionCount = required_extension_count;
  device_create_info.ppEnabledExtensionNames = pp_required_extensions;
  // These two fields are ignores by up-to-date implementations since
  // nowadays, there is no difference to instance validation layers.
  device_create_info.enabledLayerCount = 0;
  device_create_info.ppEnabledLayerNames = nullptr;

  VkResult result = vkCreateDevice(
    idevice->physical_device,
    &device_create_info,
    nullptr,
    &idevice->logical_device
  );
  if (result != VK_SUCCESS) {
    gpu_device_store.free(device.handle);
    return CGPU_FAIL_CAN_NOT_CREATE_LOGICAL_DEVICE;
  }

  vkGetDeviceQueue(
    idevice->logical_device,
    queue_family_index,
    0,
    &idevice->compute_queue
  );

  VkCommandPoolCreateInfo pool_info = {};
  pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  pool_info.pNext = nullptr;
  pool_info.queueFamilyIndex = queue_family_index;
  pool_info.flags = 0;

  result = vkCreateCommandPool(
    idevice->logical_device,
    &pool_info,
    nullptr,
    &idevice->command_pool
  );

  if (result != VK_SUCCESS)
  {
    gpu_device_store.free(device.handle);

    vkDestroyDevice(
      idevice->logical_device,
      nullptr
    );
    return CGPU_FAIL_CAN_NOT_CREATE_COMMAND_POOL;
  }

  volkLoadDeviceTable(
    &idevice->table,
    idevice->logical_device
  );

  VkPhysicalDeviceProperties device_properties;
  vkGetPhysicalDeviceProperties(
    idevice->physical_device,
    &device_properties
  );

  idevice->limits =
    _cgpu_translate_physical_device_limits(device_properties.limits);

  return CGPU_OK;
}

CgpuResult cgpu_destroy_device(
  const cgpu_device& device)
{
  _gpu_device* idevice;
  if (!_cgpu_resolve_handle(device.handle, &idevice)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }

  vkDestroyCommandPool(
    idevice->logical_device,
    idevice->command_pool,
    nullptr
  );
  vkDestroyDevice(
    idevice->logical_device,
    nullptr
  );

  gpu_device_store.free(device.handle);
  return CGPU_OK;
}

CgpuResult cgpu_create_shader(
  const cgpu_device& device,
  const uint32_t& source_size_in_bytes,
  const uint8_t* p_source,
  cgpu_shader& shader)
{
  _gpu_device* idevice;
  if (!_cgpu_resolve_handle(device.handle, &idevice)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }

  shader.handle = gpu_shader_store.create();

  _gpu_shader* ishader;
  if (!_cgpu_resolve_handle(shader.handle, &ishader)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }

  VkShaderModuleCreateInfo shader_module_create_info = {};
  shader_module_create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  shader_module_create_info.pNext = nullptr;
  shader_module_create_info.codeSize = source_size_in_bytes;
  shader_module_create_info.pCode = reinterpret_cast<const uint32_t*>(p_source);

  const VkResult result = vkCreateShaderModule(
    idevice->logical_device,
    &shader_module_create_info,
    nullptr,
    &ishader->module
  );
  if (result != VK_SUCCESS) {
    gpu_shader_store.free(shader.handle);
    return CGPU_FAIL_UNABLE_TO_CREATE_SHADER_MODULE;
  }

  return CGPU_OK;
}

CgpuResult cgpu_destroy_shader(
  const cgpu_device& device,
  const cgpu_shader& shader)
{
  _gpu_device* idevice;
  _gpu_shader* ishader;
  if (!_cgpu_resolve_handle(device.handle, &idevice)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  if (!_cgpu_resolve_handle(shader.handle, &ishader)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }

  vkDestroyShaderModule(
    idevice->logical_device,
    ishader->module,
    nullptr
  );

  gpu_shader_store.free(shader.handle);

  return CGPU_OK;
}

CgpuResult cgpu_create_buffer(
  const cgpu_device& device,
  const CgpuBufferUsageFlags usage,
  const CgpuMemoryPropertyFlags memory_properties,
  const uint32_t& size_in_bytes,
  cgpu_buffer& buffer)
{
  _gpu_device* idevice;
  if (!_cgpu_resolve_handle(device.handle, &idevice)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }

  buffer.handle = gpu_buffer_store.create();

  _gpu_buffer* ibuffer;
  if (!_cgpu_resolve_handle(buffer.handle, &ibuffer)) {
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

  VkBufferCreateInfo buffer_info = {};
  buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_info.pNext = nullptr;
  buffer_info.size = size_in_bytes;
  buffer_info.usage = vk_buffer_usage;
  buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VkResult result = vkCreateBuffer(
    idevice->logical_device,
    &buffer_info,
    nullptr,
    &ibuffer->buffer
  );
  if (result != VK_SUCCESS) {
    gpu_buffer_store.free(buffer.handle);
    return CGPU_FAIL_UNABLE_TO_CREATE_BUFFER;
  }

  VkMemoryRequirements mem_requirements;
  vkGetBufferMemoryRequirements(
    idevice->logical_device,
    ibuffer->buffer,
    &mem_requirements
  );

  VkPhysicalDeviceMemoryProperties physical_device_memory_properties;
  vkGetPhysicalDeviceMemoryProperties(
    idevice->physical_device,
    &physical_device_memory_properties
  );

  VkMemoryAllocateInfo mem_alloc_info = {};
  mem_alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  mem_alloc_info.pNext = nullptr;
  mem_alloc_info.allocationSize = mem_requirements.size;

  const VkMemoryPropertyFlags mem_flags =
    _cgpu_translate_memory_properties(memory_properties);

  uint32_t mem_index = -1;
  for (uint32_t i = 0; i < physical_device_memory_properties.memoryTypeCount; ++i) {
    if ((mem_requirements.memoryTypeBits & (1 << i)) &&
        (physical_device_memory_properties.memoryTypes[i].propertyFlags & mem_flags) == mem_flags) {
      mem_index = i;
      break;
    }
  }
  if (mem_index == -1) {
    gpu_buffer_store.free(buffer.handle);
    return CGPU_FAIL_NO_SUITABLE_MEMORY_TYPE;
  }
  mem_alloc_info.memoryTypeIndex = mem_index;

  result = vkAllocateMemory(
    idevice->logical_device,
    &mem_alloc_info,
    nullptr,
    &ibuffer->memory
  );
  if (result != VK_SUCCESS) {
    gpu_buffer_store.free(buffer.handle);
    return CGPU_FAIL_UNABLE_TO_ALLOCATE_MEMORY;
  }

  vkBindBufferMemory(
    idevice->logical_device,
    ibuffer->buffer,
    ibuffer->memory,
    0
  );

  ibuffer->size_in_bytes = mem_requirements.size;

  return CGPU_OK;
}

CgpuResult cgpu_destroy_buffer(
  const cgpu_device& device,
  const cgpu_buffer& buffer)
{
  _gpu_device* idevice;
  _gpu_buffer* ibuffer;
  if (!_cgpu_resolve_handle(device.handle, &idevice)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  if (!_cgpu_resolve_handle(buffer.handle, &ibuffer)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }

  vkDestroyBuffer(
    idevice->logical_device,
    ibuffer->buffer,
    nullptr
  );
  vkFreeMemory(
    idevice->logical_device,
    ibuffer->memory,
    nullptr
  );

  gpu_buffer_store.free(buffer.handle);

  return CGPU_OK;
}

CgpuResult cgpu_map_buffer(
  const cgpu_device& device,
  const cgpu_buffer& buffer,
  void** pp_mapped_mem)
{
  _gpu_device* idevice;
  _gpu_buffer* ibuffer;
  if (!_cgpu_resolve_handle(device.handle, &idevice)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  if (!_cgpu_resolve_handle(buffer.handle, &ibuffer)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }

  const VkResult result = vkMapMemory(
    idevice->logical_device,
    ibuffer->memory,
    0,
    ibuffer->size_in_bytes,
    0,
    pp_mapped_mem
  );

  if (result != VK_SUCCESS) {
    return CGPU_FAIL_UNABLE_TO_MAP_MEMORY;
  }

  return CGPU_OK;
}

CgpuResult cgpu_map_buffer(
  const cgpu_device& device,
  const cgpu_buffer& buffer,
  const uint32_t& source_byte_offset,
  const uint32_t& byte_count,
  void** pp_mapped_mem)
{
  _gpu_device* idevice;
  _gpu_buffer* ibuffer;
  if (!_cgpu_resolve_handle(device.handle, &idevice)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  if (!_cgpu_resolve_handle(buffer.handle, &ibuffer)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }

  const VkResult result = vkMapMemory(
    idevice->logical_device,
    ibuffer->memory,
    source_byte_offset,
    byte_count,
    0,
    pp_mapped_mem
  );

  if (result != VK_SUCCESS) {
    return CGPU_FAIL_UNABLE_TO_MAP_MEMORY;
  }

  return CGPU_OK;
}

CgpuResult cgpu_unmap_buffer(
  const cgpu_device& device,
  const cgpu_buffer& buffer)
{
  _gpu_device* idevice;
  _gpu_buffer* ibuffer;
  if (!_cgpu_resolve_handle(device.handle, &idevice)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  if (!_cgpu_resolve_handle(buffer.handle, &ibuffer)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  vkUnmapMemory(
    idevice->logical_device,
    ibuffer->memory
  );
  return CGPU_OK;
}

CgpuResult cgpu_create_image(
  const cgpu_device& device,
  const uint32_t& width,
  const uint32_t& height,
  const CgpuImageFormat format,
  const CgpuImageUsageFlags usage,
  const CgpuMemoryPropertyFlags memory_properties,
  cgpu_image& image)
{
  _gpu_device* idevice;
  if (!_cgpu_resolve_handle(device.handle, &idevice)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }

  image.handle = gpu_image_store.create();

  _gpu_image* iimage;
  if (!_cgpu_resolve_handle(image.handle, &iimage)) {
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

  const VkFormat vk_format =
    _cgpu_translate_image_format(format);

  VkImageCreateInfo image_info = {};
  image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  image_info.pNext = nullptr;
  image_info.imageType = VK_IMAGE_TYPE_2D;
  image_info.extent.width = width;
  image_info.extent.height = height;
  image_info.extent.depth = 1;
  image_info.mipLevels = 1;
  image_info.arrayLayers = 1;
  image_info.format = vk_format;
  image_info.tiling = vk_image_tiling;
  image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  image_info.usage = vk_image_usage;
  image_info.samples = VK_SAMPLE_COUNT_1_BIT;
  image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VkResult result = vkCreateImage(
    idevice->logical_device,
    &image_info,
    nullptr,
    &iimage->image
  );
  if (result != VK_SUCCESS) {
    gpu_image_store.free(image.handle);
    return CGPU_FAIL_UNABLE_TO_CREATE_IMAGE;
  }

  VkMemoryRequirements mem_requirements;
  vkGetImageMemoryRequirements(
    idevice->logical_device,
    iimage->image,
    &mem_requirements
  );

  VkPhysicalDeviceMemoryProperties physical_device_memory_properties;
  vkGetPhysicalDeviceMemoryProperties(
    idevice->physical_device,
    &physical_device_memory_properties
  );

  VkMemoryAllocateInfo mem_alloc_info = {};
  mem_alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  mem_alloc_info.pNext = nullptr;
  mem_alloc_info.allocationSize = mem_requirements.size;

  const VkMemoryPropertyFlags mem_flags =
    _cgpu_translate_memory_properties(memory_properties);

  uint32_t mem_index = -1;
  for (uint32_t i = 0; i < physical_device_memory_properties.memoryTypeCount; ++i) {
    if ((mem_requirements.memoryTypeBits & (1 << i)) &&
        (physical_device_memory_properties.memoryTypes[i].propertyFlags & mem_flags) == mem_flags) {
      mem_index = i;
      break;
    }
  }
  if (mem_index == -1) {
    gpu_image_store.free(image.handle);
    return CGPU_FAIL_NO_SUITABLE_MEMORY_TYPE;
  }
  mem_alloc_info.memoryTypeIndex = mem_index;

  result = vkAllocateMemory(
    idevice->logical_device,
    &mem_alloc_info,
    nullptr,
    &iimage->memory
  );
  if (result != VK_SUCCESS) {
    gpu_image_store.free(image.handle);
    return CGPU_FAIL_UNABLE_TO_ALLOCATE_MEMORY;
  }

  vkBindImageMemory(
    idevice->logical_device,
    iimage->image,
    iimage->memory,
    0
  );

  iimage->size_in_bytes = mem_requirements.size;

  return CGPU_OK;
}

CgpuResult cgpu_destroy_image(
  const cgpu_device& device,
  const cgpu_image& image)
{
  _gpu_device* idevice;
  _gpu_image* iimage;
  if (!_cgpu_resolve_handle(device.handle, &idevice)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  if (!_cgpu_resolve_handle(image.handle, &iimage)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }

  vkDestroyImage(
    idevice->logical_device,
    iimage->image,
    nullptr
  );

  gpu_image_store.free(image.handle);

  return CGPU_OK;
}

CgpuResult cgpu_map_image(
  const cgpu_device& device,
  const cgpu_image& image,
  void** pp_mapped_mem)
{
  _gpu_device* idevice;
  _gpu_image* iimage;
  if (!_cgpu_resolve_handle(device.handle, &idevice)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  if (!_cgpu_resolve_handle(image.handle, &iimage)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }

  const VkResult result = vkMapMemory(
    idevice->logical_device,
    iimage->memory,
    0,
    iimage->size_in_bytes,
    0,
    pp_mapped_mem
  );

  if (result != VK_SUCCESS) {
    return CGPU_FAIL_UNABLE_TO_MAP_MEMORY;
  }

  return CGPU_OK;
}

CgpuResult cgpu_map_image(
  const cgpu_device& device,
  const cgpu_image& image,
  const uint32_t& source_byte_offset,
  const uint32_t& byte_count,
  void** pp_mapped_mem)
{
  _gpu_device* idevice;
  _gpu_image* iimage;
  if (!_cgpu_resolve_handle(device.handle, &idevice)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  if (!_cgpu_resolve_handle(image.handle, &iimage)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }

  const VkResult result = vkMapMemory(
    idevice->logical_device,
    iimage->memory,
    source_byte_offset,
    byte_count,
    0,
    pp_mapped_mem
  );

  if (result != VK_SUCCESS) {
    return CGPU_FAIL_UNABLE_TO_MAP_MEMORY;
  }

  return CGPU_OK;
}

CgpuResult cgpu_unmap_image(
  const cgpu_device& device,
  const cgpu_image& image)
{
  _gpu_device* idevice;
  _gpu_image* iimage;
  if (!_cgpu_resolve_handle(device.handle, &idevice)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  if (!_cgpu_resolve_handle(image.handle, &iimage)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  vkUnmapMemory(
    idevice->logical_device,
    iimage->memory
  );
  return CGPU_OK;
}

CgpuResult cgpu_create_pipeline(
  const cgpu_device& device,
  const uint32_t& shader_resources_buffer_count,
  const cgpu_shader_resource_buffer* p_shader_resources_buffers,
  const uint32_t& shader_resources_image_count,
  const cgpu_shader_resource_image* p_shader_resources_images,
  const cgpu_shader& shader,
  const char* p_shader_entry_point,
  cgpu_pipeline& pipeline)
{
  _gpu_device* idevice;
  if (!_cgpu_resolve_handle(device.handle, &idevice)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  _gpu_shader* ishader;
  if (!_cgpu_resolve_handle(shader.handle, &ishader)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }

  pipeline.handle = gpu_pipeline_store.create();

  _gpu_pipeline* ipipeline;
  if (!_cgpu_resolve_handle(pipeline.handle, &ipipeline)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }

  std::vector<VkDescriptorSetLayoutBinding> descriptor_set_bindings;

  for (uint32_t i = 0; i < shader_resources_buffer_count; ++i)
  {
    const cgpu_shader_resource_buffer& shader_resource_buffer = p_shader_resources_buffers[i];
    VkDescriptorSetLayoutBinding descriptor_set_layout_binding = {};
    descriptor_set_layout_binding.binding = shader_resource_buffer.binding;
    descriptor_set_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_set_layout_binding.descriptorCount = 1;
    descriptor_set_layout_binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    descriptor_set_layout_binding.pImmutableSamplers = nullptr;
    descriptor_set_bindings.push_back(descriptor_set_layout_binding);
  }

  for (uint32_t i = 0; i < shader_resources_image_count; ++i)
  {
    const cgpu_shader_resource_image& shader_resource_buffer = p_shader_resources_images[i];
    VkDescriptorSetLayoutBinding descriptor_set_layout_binding = {};
    descriptor_set_layout_binding.binding = shader_resource_buffer.binding;
    descriptor_set_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    descriptor_set_layout_binding.descriptorCount = 1;
    descriptor_set_layout_binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    descriptor_set_layout_binding.pImmutableSamplers = nullptr;
    descriptor_set_bindings.push_back(descriptor_set_layout_binding);
  }

  VkDescriptorSetLayoutCreateInfo descriptor_set_layout_create_info = {};
  descriptor_set_layout_create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  descriptor_set_layout_create_info.pNext = nullptr;
  descriptor_set_layout_create_info.flags = 0;
  descriptor_set_layout_create_info.bindingCount = descriptor_set_bindings.size();
  descriptor_set_layout_create_info.pBindings = descriptor_set_bindings.data();

  VkResult result = vkCreateDescriptorSetLayout(
    idevice->logical_device,
    &descriptor_set_layout_create_info,
    nullptr,
    &ipipeline->descriptor_set_layout
  );
  if (result != VK_SUCCESS) {
    gpu_pipeline_store.free(pipeline.handle);
    return CGPU_FAIL_UNABLE_TO_CREATE_DESCRIPTOR_LAYOUT;
  }

  VkPipelineLayoutCreateInfo pipeline_layout_create_info = {};
  pipeline_layout_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipeline_layout_create_info.pNext = nullptr;
  pipeline_layout_create_info.flags = 0;
  pipeline_layout_create_info.setLayoutCount = 1;
  pipeline_layout_create_info.pSetLayouts = &ipipeline->descriptor_set_layout;
  pipeline_layout_create_info.pushConstantRangeCount = 0;
  pipeline_layout_create_info.pPushConstantRanges = nullptr;

  result = vkCreatePipelineLayout(
    idevice->logical_device,
    &pipeline_layout_create_info,
    nullptr,
    &ipipeline->layout
  );
  if (result != VK_SUCCESS) {
    gpu_pipeline_store.free(pipeline.handle);
    vkDestroyDescriptorSetLayout(
      idevice->logical_device,
      ipipeline->descriptor_set_layout,
      nullptr
    );
    return CGPU_FAIL_UNABLE_TO_CREATE_PIPELINE_LAYOUT;
  }

  VkPipelineShaderStageCreateInfo pipeline_shader_stage_create_info = {};
  pipeline_shader_stage_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  pipeline_shader_stage_create_info.pNext = nullptr;
  pipeline_shader_stage_create_info.flags = 0;
  pipeline_shader_stage_create_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  pipeline_shader_stage_create_info.module = ishader->module;
  pipeline_shader_stage_create_info.pName = p_shader_entry_point;
  pipeline_shader_stage_create_info.pSpecializationInfo = nullptr;

  VkComputePipelineCreateInfo pipeline_create_info = {};
  pipeline_create_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipeline_create_info.pNext = nullptr;
  pipeline_create_info.flags = VK_PIPELINE_CREATE_DISPATCH_BASE;
  pipeline_create_info.stage = pipeline_shader_stage_create_info;
  pipeline_create_info.layout = ipipeline->layout;
  pipeline_create_info.basePipelineHandle = VK_NULL_HANDLE;
  pipeline_create_info.basePipelineIndex = 0;

  result = vkCreateComputePipelines(
    idevice->logical_device,
    nullptr,
    1,
    &pipeline_create_info,
    nullptr,
    &ipipeline->pipeline
  );
  if (result != VK_SUCCESS) {
    gpu_pipeline_store.free(pipeline.handle);
    vkDestroyPipelineLayout(
      idevice->logical_device,
      ipipeline->layout,
      nullptr
    );
    vkDestroyDescriptorSetLayout(
      idevice->logical_device,
      ipipeline->descriptor_set_layout,
      nullptr
    );
    return CGPU_FAIL_UNABLE_TO_CREATE_COMPUTE_PIPELINE;
  }

  VkDescriptorPoolSize descriptor_pool_size = {};
  descriptor_pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  descriptor_pool_size.descriptorCount = descriptor_set_bindings.size();

  VkDescriptorPoolCreateInfo descriptor_pool_create_info = {};
  descriptor_pool_create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  descriptor_pool_create_info.pNext = nullptr;
  descriptor_pool_create_info.poolSizeCount = 1;
  descriptor_pool_create_info.pPoolSizes = &descriptor_pool_size;
  descriptor_pool_create_info.maxSets = 1;

  result = vkCreateDescriptorPool(
    idevice->logical_device,
    &descriptor_pool_create_info,
    nullptr,
    &ipipeline->descriptor_pool
  );
  if (result != VK_SUCCESS) {
    gpu_pipeline_store.free(pipeline.handle);
    vkDestroyPipeline(
      idevice->logical_device,
      ipipeline->pipeline,
      nullptr
    );
    vkDestroyPipelineLayout(
      idevice->logical_device,
      ipipeline->layout,
      nullptr
    );
    vkDestroyDescriptorSetLayout(
      idevice->logical_device,
      ipipeline->descriptor_set_layout,
      nullptr
    );
    return CGPU_FAIL_UNABLE_TO_CREATE_DESCRIPTOR_POOL;
  }

  VkDescriptorSetAllocateInfo descriptor_set_allocate_info = {};
  descriptor_set_allocate_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  descriptor_set_allocate_info.pNext = nullptr;
  descriptor_set_allocate_info.descriptorPool = ipipeline->descriptor_pool;
  descriptor_set_allocate_info.descriptorSetCount = 1;
  descriptor_set_allocate_info.pSetLayouts = &ipipeline->descriptor_set_layout;

  result = vkAllocateDescriptorSets(
    idevice->logical_device,
    &descriptor_set_allocate_info,
    &ipipeline->descriptor_set
  );
  if (result != VK_SUCCESS) {
    gpu_pipeline_store.free(pipeline.handle);
    vkDestroyDescriptorPool(
      idevice->logical_device,
      ipipeline->descriptor_pool,
      nullptr
    );
    vkDestroyPipeline(
      idevice->logical_device,
      ipipeline->pipeline,
      nullptr
    );
    vkDestroyPipelineLayout(
      idevice->logical_device,
      ipipeline->layout,
      nullptr
    );
    vkDestroyDescriptorSetLayout(
      idevice->logical_device,
      ipipeline->descriptor_set_layout,
      nullptr
    );
    return CGPU_FAIL_UNABLE_TO_ALLOCATE_DESCRIPTOR_SET;
  }

  std::vector<VkDescriptorBufferInfo> descriptor_buffer_infos;
  std::vector<VkDescriptorImageInfo> descriptor_image_infos;
  std::vector<VkWriteDescriptorSet> write_descriptor_sets;

  descriptor_buffer_infos.resize(shader_resources_buffer_count);
  descriptor_image_infos.resize(shader_resources_image_count);

  for (uint32_t i = 0; i < shader_resources_buffer_count; ++i)
  {
    const cgpu_shader_resource_buffer& shader_resource_buffer = p_shader_resources_buffers[i];
    const cgpu_buffer& buffer = shader_resource_buffer.buffer;

    const CgpuShaderResourceUsageFlags& usage = shader_resource_buffer.usage;
    if (usage != CGPU_SHADER_RESOURCE_USAGE_FLAG_WRITE) {
      continue;
    }

    _gpu_buffer* ibuffer;
    if (!_cgpu_resolve_handle(buffer.handle, &ibuffer)) {
      return CGPU_FAIL_INVALID_HANDLE;
    }

    VkDescriptorBufferInfo& descriptor_buffer_info = descriptor_buffer_infos[i];
    descriptor_buffer_info.buffer = ibuffer->buffer;
    descriptor_buffer_info.offset = 0;
    descriptor_buffer_info.range = ibuffer->size_in_bytes;

    VkWriteDescriptorSet write_descriptor_set = {};
    write_descriptor_set.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write_descriptor_set.pNext = nullptr;
    write_descriptor_set.dstSet = ipipeline->descriptor_set;
    write_descriptor_set.dstBinding = shader_resource_buffer.binding;
    write_descriptor_set.dstArrayElement = 0;
    write_descriptor_set.descriptorCount = 1;
    write_descriptor_set.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write_descriptor_set.pImageInfo = nullptr;
    write_descriptor_set.pBufferInfo = &descriptor_buffer_info;
    write_descriptor_set.pTexelBufferView = nullptr;
    write_descriptor_sets.push_back(write_descriptor_set);
  }

  for (uint32_t i = 0; i < shader_resources_image_count; ++i)
  {
    const cgpu_shader_resource_image& shader_resource_image = p_shader_resources_images[i];
    const cgpu_image& image = shader_resource_image.image;

    const CgpuShaderResourceUsageFlags& usage = shader_resource_image.usage;
    if (usage != CGPU_SHADER_RESOURCE_USAGE_FLAG_WRITE) {
      continue;
    }

    _gpu_image* iimage;
    if (!_cgpu_resolve_handle(image.handle, &iimage)) {
      return CGPU_FAIL_INVALID_HANDLE;
    }

    VkDescriptorImageInfo& descriptor_image_info = descriptor_image_infos[i];
    // TODO:
    //descriptor_image_info.sampler = 0;
    //descriptor_image_info.imageView = 0;
    //descriptor_image_info.imageLayout = 0;

    VkWriteDescriptorSet write_descriptor_set = {};
    write_descriptor_set.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write_descriptor_set.pNext = nullptr;
    write_descriptor_set.dstSet = ipipeline->descriptor_set;
    write_descriptor_set.dstBinding = shader_resource_image.binding;
    write_descriptor_set.dstArrayElement = 0;
    write_descriptor_set.descriptorCount = 1;
    write_descriptor_set.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    write_descriptor_set.pImageInfo = &descriptor_image_info;
    write_descriptor_set.pBufferInfo = nullptr;
    write_descriptor_set.pTexelBufferView = nullptr;
    write_descriptor_sets.push_back(write_descriptor_set);
  }

  vkUpdateDescriptorSets(
    idevice->logical_device,
    write_descriptor_sets.size(),
    write_descriptor_sets.data(),
    0,
    nullptr
  );

  return CGPU_OK;
}

CgpuResult cgpu_destroy_pipeline(
  const cgpu_device& device,
  const cgpu_pipeline& pipeline)
{
  _gpu_device* idevice;
  _gpu_pipeline* ipipeline;
  if (!_cgpu_resolve_handle(device.handle, &idevice)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  if (!_cgpu_resolve_handle(pipeline.handle, &ipipeline)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  vkDestroyDescriptorPool(
    idevice->logical_device,
    ipipeline->descriptor_pool,
    nullptr
  );
  vkDestroyPipeline(
    idevice->logical_device,
    ipipeline->pipeline,
    nullptr
  );
  vkDestroyPipelineLayout(
    idevice->logical_device,
    ipipeline->layout,
    nullptr
  );
  vkDestroyDescriptorSetLayout(
    idevice->logical_device,
    ipipeline->descriptor_set_layout,
    nullptr
  );
  gpu_pipeline_store.free(pipeline.handle);
  return CGPU_OK;
}

CgpuResult cgpu_create_command_buffer(
  const cgpu_device& device,
  cgpu_command_buffer& command_buffer)
{
  _gpu_device* idevice;
  if (!_cgpu_resolve_handle(device.handle, &idevice)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }

  command_buffer.handle = gpu_command_buffer_store.create();

  _gpu_command_buffer* icommand_buffer;
  if (!_cgpu_resolve_handle(command_buffer.handle, &icommand_buffer)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }

  VkCommandBufferAllocateInfo cmdbuf_alloc_info = {};
  cmdbuf_alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cmdbuf_alloc_info.commandPool = idevice->command_pool;
  cmdbuf_alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cmdbuf_alloc_info.commandBufferCount = 1;

  const VkResult result = vkAllocateCommandBuffers(
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
  const cgpu_device& device,
  const cgpu_command_buffer& command_buffer)
{
  _gpu_device* idevice;
  _gpu_command_buffer* icommand_buffer;
  if (!_cgpu_resolve_handle(device.handle, &idevice)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  if (!_cgpu_resolve_handle(command_buffer.handle, &icommand_buffer)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  vkFreeCommandBuffers(
    idevice->logical_device,
    idevice->command_pool,
    1,
    &icommand_buffer->command_buffer
  );
  gpu_command_buffer_store.free(command_buffer.handle);
  return CGPU_OK;
}

CgpuResult cgpu_begin_command_buffer(
  const cgpu_command_buffer& command_buffer)
{
  _gpu_command_buffer* icommand_buffer;
  if (!_cgpu_resolve_handle(command_buffer.handle, &icommand_buffer)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  VkCommandBufferBeginInfo command_buffer_begin_info = {};
  command_buffer_begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  command_buffer_begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT; // TODO
  command_buffer_begin_info.pInheritanceInfo = nullptr;
  const VkResult result = vkBeginCommandBuffer(
    icommand_buffer->command_buffer,
    &command_buffer_begin_info
  );
  if (result != VK_SUCCESS) {
    return CGPU_FAIL_UNABLE_TO_BEGIN_COMMAND_BUFFER;
  }
  return CGPU_OK;
}

CgpuResult cgpu_cmd_bind_pipeline(
  const cgpu_command_buffer& command_buffer,
  const cgpu_pipeline& pipeline)
{
  _gpu_command_buffer* icommand_buffer;
  _gpu_pipeline* ipipeline;
  if (!_cgpu_resolve_handle(command_buffer.handle, &icommand_buffer)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  if (!_cgpu_resolve_handle(pipeline.handle, &ipipeline)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  vkCmdBindPipeline(
    icommand_buffer->command_buffer,
    VK_PIPELINE_BIND_POINT_COMPUTE,
    ipipeline->pipeline
  );
  vkCmdBindDescriptorSets(
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
  const cgpu_command_buffer& command_buffer,
  const cgpu_buffer& source_buffer,
  const uint32_t& source_byte_offset,
  const cgpu_buffer& destination_buffer,
  const uint32_t& destination_byte_offset,
  const uint32_t& byte_count)
{
  _gpu_command_buffer* icommand_buffer;
  _gpu_buffer* isource_buffer;
  _gpu_buffer* idestination_buffer;
  if (!_cgpu_resolve_handle(command_buffer.handle, &icommand_buffer)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  if (!_cgpu_resolve_handle(source_buffer.handle, &isource_buffer)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  if (!_cgpu_resolve_handle(destination_buffer.handle, &idestination_buffer)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  VkBufferCopy region = {};
  region.srcOffset = source_byte_offset;
  region.dstOffset = destination_byte_offset;
  region.size = byte_count;
  vkCmdCopyBuffer(
    icommand_buffer->command_buffer,
    isource_buffer->buffer,
    idestination_buffer->buffer,
    1,
    &region
  );
  return CGPU_OK;
}

CgpuResult cgpu_cmd_copy_buffer(
  const cgpu_command_buffer& command_buffer,
  const cgpu_buffer& source_buffer,
  const cgpu_buffer& destination_buffer)
{
  _gpu_command_buffer* icommand_buffer;
  _gpu_buffer* isource_buffer;
  _gpu_buffer* idestination_buffer;
  if (!_cgpu_resolve_handle(command_buffer.handle, &icommand_buffer)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  if (!_cgpu_resolve_handle(source_buffer.handle, &isource_buffer)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  if (!_cgpu_resolve_handle(destination_buffer.handle, &idestination_buffer)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  VkBufferCopy region = {};
  region.srcOffset = 0;
  region.dstOffset = 0;
  region.size = isource_buffer->size_in_bytes;
  vkCmdCopyBuffer(
    icommand_buffer->command_buffer,
    isource_buffer->buffer,
    idestination_buffer->buffer,
    1,
    &region
  );
  return CGPU_OK;
}

CgpuResult cgpu_cmd_dispatch(
  const cgpu_command_buffer& command_buffer,
  const uint32_t& dim_x,
  const uint32_t& dim_y,
  const uint32_t& dim_z)
{
  _gpu_command_buffer* icommand_buffer;
  if (!_cgpu_resolve_handle(command_buffer.handle, &icommand_buffer)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  vkCmdDispatch(
    icommand_buffer->command_buffer,
    dim_x,
    dim_y,
    dim_z
  );
  return CGPU_OK;
}

CgpuResult cgpu_cmd_pipeline_barrier(
  const cgpu_command_buffer& command_buffer,
  uint32_t num_memory_barriers,
  cgpu_memory_barrier* p_memory_barriers,
  uint32_t num_buffer_memory_barriers,
  cgpu_buffer_memory_barrier* p_buffer_memory_barriers,
  uint32_t num_image_memory_barriers,
  cgpu_image_memory_barrier* p_image_memory_barriers)
{
  _gpu_command_buffer* icommand_buffer;
  if (!_cgpu_resolve_handle(command_buffer.handle, &icommand_buffer)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }

  VkMemoryBarrier* vk_memory_barriers = new VkMemoryBarrier[num_memory_barriers];

  for (uint32_t i = 0; i < num_memory_barriers; ++i)
  {
    const cgpu_memory_barrier& b_cgpu = p_memory_barriers[i];
    VkMemoryBarrier& b_vk = vk_memory_barriers[i];
    b_vk = {};
    b_vk.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    b_vk.srcAccessMask = _cgpu_translate_access_flags(b_cgpu.src_access_flags);
    b_vk.dstAccessMask = _cgpu_translate_access_flags(b_cgpu.dst_access_flags);
  }

  VkBufferMemoryBarrier* vk_buffer_memory_barriers
    = new VkBufferMemoryBarrier[num_buffer_memory_barriers];

  for (uint32_t i = 0; i < num_buffer_memory_barriers; ++i)
  {
    const cgpu_buffer_memory_barrier& b_cgpu = p_buffer_memory_barriers[i];
    VkBufferMemoryBarrier& b_vk = vk_buffer_memory_barriers[i];
    b_vk = {};
    b_vk.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    b_vk.srcAccessMask = _cgpu_translate_access_flags(b_cgpu.src_access_flags);
    b_vk.dstAccessMask = _cgpu_translate_access_flags(b_cgpu.dst_access_flags);
    b_vk.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b_vk.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    _gpu_buffer* ibuffer;
    if (!_cgpu_resolve_handle(b_cgpu.buffer.handle, &ibuffer)) {
      return CGPU_FAIL_INVALID_HANDLE;
    }
    b_vk.buffer = ibuffer->buffer;
    b_vk.offset = b_cgpu.byte_offset;
    b_vk.size = b_cgpu.num_bytes;
  }

  // TODO: translate image barrier

  vkCmdPipelineBarrier(
    icommand_buffer->command_buffer,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
      VK_PIPELINE_STAGE_TRANSFER_BIT,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
      VK_PIPELINE_STAGE_TRANSFER_BIT,
    0,
    num_memory_barriers,
    vk_memory_barriers,
    num_buffer_memory_barriers,
    vk_buffer_memory_barriers,
    0,//TODOnum_image_memory_barriers,
    nullptr//TODOconst VkImageMemoryBarrier* p_image_memory_barriers
  );

  delete[] vk_memory_barriers;
  delete[] vk_buffer_memory_barriers;

  return CGPU_OK;
}

CgpuResult cgpu_end_command_buffer(
  const cgpu_command_buffer& command_buffer)
{
  _gpu_command_buffer* icommand_buffer;
  if (!_cgpu_resolve_handle(command_buffer.handle, &icommand_buffer)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  vkEndCommandBuffer(icommand_buffer->command_buffer);
  return CGPU_OK;
}

CgpuResult cgpu_create_fence(
  const cgpu_device& device,
  cgpu_fence& fence)
{
  _gpu_device* idevice;
  if (!_cgpu_resolve_handle(device.handle, &idevice)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }

  fence.handle = gpu_fence_store.create();

  _gpu_fence* ifence;
  if (!_cgpu_resolve_handle(fence.handle, &ifence)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }

  VkFenceCreateInfo fence_create_info = {};
  fence_create_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fence_create_info.pNext = nullptr;
  fence_create_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

  const VkResult result = vkCreateFence(
    idevice->logical_device,
    &fence_create_info,
    nullptr,
    &ifence->fence
  );
  if (result != VK_SUCCESS) {
    gpu_fence_store.free(fence.handle);
    return CGPU_FAIL_UNABLE_TO_CREATE_FENCE;
  }
  return CGPU_OK;
}

CgpuResult cgpu_destroy_fence(
  const cgpu_device& device,
  const cgpu_fence& fence)
{
  _gpu_device* idevice;
  _gpu_fence* ifence;
  if (!_cgpu_resolve_handle(device.handle, &idevice)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  if (!_cgpu_resolve_handle(fence.handle, &ifence)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  vkDestroyFence(
    idevice->logical_device,
    ifence->fence,
    nullptr
  );
  gpu_fence_store.free(fence.handle);
  return CGPU_OK;
}

CgpuResult cgpu_reset_fence(
  const cgpu_device& device,
  const cgpu_fence& fence)
{
  _gpu_device* idevice;
  _gpu_fence* ifence;
  if (!_cgpu_resolve_handle(device.handle, &idevice)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  if (!_cgpu_resolve_handle(fence.handle, &ifence)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  const VkResult result = vkResetFences(
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
  const cgpu_device& device,
  const cgpu_fence& fence)
{
  _gpu_device* idevice;
  _gpu_fence* ifence;
  if (!_cgpu_resolve_handle(device.handle, &idevice)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  if (!_cgpu_resolve_handle(fence.handle, &ifence)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  const VkResult result = vkWaitForFences(
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
  const cgpu_device& device,
  const cgpu_command_buffer& command_buffer,
  const cgpu_fence& fence)
{
  _gpu_device* idevice;
  _gpu_command_buffer* icommand_buffer;
  _gpu_fence* ifence;
  if (!_cgpu_resolve_handle(device.handle, &idevice)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  if (!_cgpu_resolve_handle(command_buffer.handle, &icommand_buffer)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  if (!_cgpu_resolve_handle(fence.handle, &ifence)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }

  VkSubmitInfo submit_info = {};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.pNext = nullptr;
  submit_info.waitSemaphoreCount = 0;
  submit_info.pWaitSemaphores = nullptr;
  submit_info.pWaitDstStageMask = nullptr;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &icommand_buffer->command_buffer;
  submit_info.signalSemaphoreCount = 0;
  submit_info.pSignalSemaphores = nullptr;

  const VkResult result = vkQueueSubmit(
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
  const cgpu_device& device,
  const cgpu_buffer& buffer,
  const uint64_t& byte_offset,
  const uint64_t& byte_count)
{
  _gpu_device* idevice;
  _gpu_buffer* ibuffer;
  if (!_cgpu_resolve_handle(device.handle, &idevice)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  if (!_cgpu_resolve_handle(buffer.handle, &ibuffer)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }

  VkMappedMemoryRange memory_range = {};
  memory_range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
  memory_range.pNext = nullptr;
  memory_range.memory = ibuffer->memory;
  memory_range.offset = byte_offset;
  memory_range.size =
    (byte_count == CGPU_WHOLE_SIZE) ? VK_WHOLE_SIZE : byte_count;

  const VkResult result = vkFlushMappedMemoryRanges(
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
  const cgpu_device& device,
  const cgpu_buffer& buffer,
  const uint64_t& byte_offset,
  const uint64_t& byte_count)
{
  _gpu_device* idevice;
  _gpu_buffer* ibuffer;
  if (!_cgpu_resolve_handle(device.handle, &idevice)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  if (!_cgpu_resolve_handle(buffer.handle, &ibuffer)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }

  VkMappedMemoryRange memory_range = {};
  memory_range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
  memory_range.pNext = nullptr;
  memory_range.memory = ibuffer->memory;
  memory_range.offset = byte_offset;
  memory_range.size =
    (byte_count == CGPU_WHOLE_SIZE) ? VK_WHOLE_SIZE : byte_count; // TODO: do this everywhere

  const VkResult result = vkInvalidateMappedMemoryRanges(
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
  const cgpu_device& device,
  cgpu_physical_device_limits& limits)
{
  _gpu_device* idevice;
  if (!_cgpu_resolve_handle(device.handle, &idevice)) {
    return CGPU_FAIL_INVALID_HANDLE;
  }
  std::memcpy(
    &limits,
    &idevice->limits,
    sizeof(cgpu_physical_device_limits)
  );
  return CGPU_OK;
}
