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

#pragma once

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

namespace gtl
{
  constexpr static const uint64_t CGPU_WHOLE_SIZE = ~0ULL;
  constexpr static const uint32_t CGPU_MAX_TIMESTAMP_QUERIES = 32;
  constexpr static const uint32_t CGPU_MAX_DESCRIPTOR_SET_COUNT = 4;

  typedef uint32_t CgpuBufferUsageFlags;

  enum CgpuBufferUsageFlagBits
  {
    CGPU_BUFFER_USAGE_FLAG_TRANSFER_SRC = 0x00000001,
    CGPU_BUFFER_USAGE_FLAG_TRANSFER_DST = 0x00000002,
    CGPU_BUFFER_USAGE_FLAG_UNIFORM_BUFFER = 0x00000010,
    CGPU_BUFFER_USAGE_FLAG_STORAGE_BUFFER = 0x00000020,
    CGPU_BUFFER_USAGE_FLAG_SHADER_DEVICE_ADDRESS = 0x00020000,
    CGPU_BUFFER_USAGE_FLAG_ACCELERATION_STRUCTURE_BUILD_INPUT = 0x00080000,
    CGPU_BUFFER_USAGE_FLAG_ACCELERATION_STRUCTURE_STORAGE = 0x00100000,
    CGPU_BUFFER_USAGE_FLAG_SHADER_BINDING_TABLE_BIT_KHR = 0x00000400
  };

  typedef uint32_t CgpuMemoryPropertyFlags;

  enum CgpuMemoryPropertyFlagBits
  {
    CGPU_MEMORY_PROPERTY_FLAG_DEVICE_LOCAL = 0x00000001,
    CGPU_MEMORY_PROPERTY_FLAG_HOST_VISIBLE = 0x00000002,
    CGPU_MEMORY_PROPERTY_FLAG_HOST_COHERENT = 0x00000004,
    CGPU_MEMORY_PROPERTY_FLAG_HOST_CACHED = 0x00000008
  };

  typedef uint32_t CgpuImageUsageFlags;

  enum CgpuImageUsageFlagBits
  {
    CGPU_IMAGE_USAGE_FLAG_TRANSFER_SRC = 0x00000001,
    CGPU_IMAGE_USAGE_FLAG_TRANSFER_DST = 0x00000002,
    CGPU_IMAGE_USAGE_FLAG_SAMPLED = 0x00000004,
    CGPU_IMAGE_USAGE_FLAG_STORAGE = 0x00000008
  };

  enum CgpuImageFormat
  {
    CGPU_IMAGE_FORMAT_UNDEFINED = 0,
    CGPU_IMAGE_FORMAT_R8G8B8A8_UNORM = 37,
    CGPU_IMAGE_FORMAT_R32_SFLOAT = 100
  };

  typedef uint32_t CgpuMemoryAccessFlags;

  enum CgpuMemoryAccessFlagBits
  {
    CGPU_MEMORY_ACCESS_FLAG_UNIFORM_READ = 0x00000008,
    CGPU_MEMORY_ACCESS_FLAG_SHADER_READ = 0x00000020,
    CGPU_MEMORY_ACCESS_FLAG_SHADER_WRITE = 0x00000040,
    CGPU_MEMORY_ACCESS_FLAG_TRANSFER_READ = 0x00000800,
    CGPU_MEMORY_ACCESS_FLAG_TRANSFER_WRITE = 0x00001000,
    CGPU_MEMORY_ACCESS_FLAG_HOST_READ = 0x00002000,
    CGPU_MEMORY_ACCESS_FLAG_HOST_WRITE = 0x00004000,
    CGPU_MEMORY_ACCESS_FLAG_MEMORY_READ = 0x00008000,
    CGPU_MEMORY_ACCESS_FLAG_MEMORY_WRITE = 0x00010000,
    CGPU_MEMORY_ACCESS_FLAG_ACCELERATION_STRUCTURE_READ = 0x00200000,
    CGPU_MEMORY_ACCESS_FLAG_ACCELERATION_STRUCTURE_WRITE = 0x00400000,
  };

  enum CgpuSamplerAddressMode
  {
    CGPU_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE = 0,
    CGPU_SAMPLER_ADDRESS_MODE_REPEAT = 1,
    CGPU_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT = 2,
    CGPU_SAMPLER_ADDRESS_MODE_CLAMP_TO_BLACK = 3
  };

  typedef uint32_t CgpuShaderStageFlags;

  enum CgpuShaderStageFlagBits
  {
    CGPU_SHADER_STAGE_FLAG_COMPUTE = 0x00000020,
    CGPU_SHADER_STAGE_FLAG_RAYGEN = 0x00000100,
    CGPU_SHADER_STAGE_FLAG_ANY_HIT = 0x00000200,
    CGPU_SHADER_STAGE_FLAG_CLOSEST_HIT = 0x00000400,
    CGPU_SHADER_STAGE_FLAG_MISS = 0x00000800
  };

  typedef uint32_t CgpuPipelineStageFlags;

  enum CgpuPipelineStageFlagBits
  {
    CGPU_PIPELINE_STAGE_FLAG_COMPUTE_SHADER = 0x00000800,
    CGPU_PIPELINE_STAGE_FLAG_TRANSFER = 0x00001000,
    CGPU_PIPELINE_STAGE_FLAG_HOST = 0x00004000,
    CGPU_PIPELINE_STAGE_FLAG_RAY_TRACING_SHADER = 0x00200000,
    CGPU_PIPELINE_STAGE_FLAG_ACCELERATION_STRUCTURE_BUILD = 0x02000000,
  };

  struct CgpuInstance      { uint64_t handle = 0; };
  struct CgpuDevice        { uint64_t handle = 0; };
  struct CgpuBuffer        { uint64_t handle = 0; };
  struct CgpuImage         { uint64_t handle = 0; };
  struct CgpuShader        { uint64_t handle = 0; };
  struct CgpuPipeline      { uint64_t handle = 0; };
  struct CgpuSemaphore     { uint64_t handle = 0; };
  struct CgpuCommandBuffer { uint64_t handle = 0; };
  struct CgpuSampler       { uint64_t handle = 0; };
  struct CgpuBlas          { uint64_t handle = 0; };
  struct CgpuTlas          { uint64_t handle = 0; };

  struct CgpuImageCreateInfo
  {
    uint32_t width;
    uint32_t height;
    bool is3d = false;
    uint32_t depth = 1;
    CgpuImageFormat format = CGPU_IMAGE_FORMAT_R8G8B8A8_UNORM;
    CgpuImageUsageFlags usage = CGPU_IMAGE_USAGE_FLAG_TRANSFER_DST | CGPU_IMAGE_USAGE_FLAG_SAMPLED;
    const char* debugName = nullptr;
  };

  struct CgpuBufferCreateInfo
  {
    CgpuBufferUsageFlags usage;
    CgpuMemoryPropertyFlags memoryProperties;
    uint64_t size;
    const char* debugName = nullptr;
    uint32_t alignment = 0; // no explicit alignment
  };

  struct CgpuShaderCreateInfo
  {
    uint64_t size;
    const uint8_t* source;
    CgpuShaderStageFlags stageFlags;
    const char* debugName = nullptr;
    uint32_t maxRayPayloadSize = 0; // for RT shaders
    uint32_t maxRayHitAttributeSize = 0; // for RT shaders
  };

  struct CgpuSamplerCreateInfo
  {
    CgpuSamplerAddressMode addressModeU;
    CgpuSamplerAddressMode addressModeV;
    CgpuSamplerAddressMode addressModeW;
  };

  struct CgpuComputePipelineCreateInfo
  {
    CgpuShader shader;
    const char* debugName = nullptr;
  };

  // TODO: replace with buffer offset & stride
  struct CgpuVertex
  {
    float x;
    float y;
    float z;
  };

  struct CgpuBlasInstance
  {
    CgpuBlas as;
    uint32_t hitGroupIndex;
    uint32_t instanceCustomIndex;
    float transform[3][4];
  };

  struct CgpuRtHitGroup
  {
    CgpuShader closestHitShader; // optional
    CgpuShader anyHitShader;     // optional
  };

  struct CgpuRtPipelineCreateInfo
  {
    CgpuShader rgenShader;
    uint32_t missShaderCount = 0;
    CgpuShader* missShaders = nullptr;
    uint32_t hitGroupCount = 0;
    const CgpuRtHitGroup* hitGroups = nullptr;
    const char* debugName = nullptr;
    uint32_t maxRayPayloadSize = 0;
    uint32_t maxRayHitAttributeSize = 0;
  };

  struct CgpuBlasCreateInfo
  {
    CgpuBuffer vertexBuffer;
    CgpuBuffer indexBuffer;
    uint32_t maxVertex;
    uint32_t triangleCount;
    bool isOpaque;
    const char* debugName = nullptr;
  };

  struct CgpuTlasCreateInfo
  {
    uint32_t instanceCount;
    const CgpuBlasInstance* instances;
    const char* debugName = nullptr;
  };

  struct CgpuBufferBinding
  {
    uint32_t binding;
    CgpuBuffer buffer;
    uint32_t index = 0;
    uint64_t offset = 0;
    uint64_t size = CGPU_WHOLE_SIZE;
  };

  struct CgpuImageBinding
  {
    uint32_t binding;
    CgpuImage image;
    uint32_t index = 0;
  };

  struct CgpuSamplerBinding
  {
    uint32_t binding;
    CgpuSampler sampler;
    uint32_t index = 0;
  };

  struct CgpuTlasBinding
  {
    uint32_t binding;
    CgpuTlas as;
    uint32_t index = 0;
  };

  struct CgpuBindings
  {
    uint32_t bufferCount = 0;
    const CgpuBufferBinding* buffers = nullptr;
    uint32_t imageCount = 0;
    const CgpuImageBinding* images = nullptr;
    uint32_t samplerCount = 0;
    const CgpuSamplerBinding* samplers = nullptr;
    uint32_t tlasCount = 0;
    const CgpuTlasBinding* tlases = nullptr;
  };

  struct CgpuMemoryBarrier
  {
    CgpuPipelineStageFlags srcStageMask;
    CgpuMemoryAccessFlags srcAccessMask;
    CgpuPipelineStageFlags dstStageMask;
    CgpuMemoryAccessFlags dstAccessMask;
  };

  struct CgpuBufferMemoryBarrier
  {
    CgpuBuffer buffer;
    CgpuPipelineStageFlags srcStageMask;
    CgpuMemoryAccessFlags srcAccessMask;
    CgpuPipelineStageFlags dstStageMask;
    CgpuMemoryAccessFlags dstAccessMask;
    uint64_t offset = 0;
    uint64_t size = CGPU_WHOLE_SIZE;
  };

  struct CgpuImageMemoryBarrier
  {
    CgpuImage image;
    CgpuPipelineStageFlags srcStageMask;
    CgpuPipelineStageFlags dstStageMask;
    CgpuMemoryAccessFlags accessMask;
  };

  struct CgpuPipelineBarrier
  {
    uint32_t memoryBarrierCount = 0;
    const CgpuMemoryBarrier* memoryBarriers = nullptr;
    uint32_t bufferBarrierCount = 0;
    const CgpuBufferMemoryBarrier* bufferBarriers = nullptr;
    uint32_t imageBarrierCount = 0;
    const CgpuImageMemoryBarrier* imageBarriers = nullptr;
  };

  struct CgpuDeviceFeatures
  {
    bool debugPrintf;
    bool rayTracingInvocationReorder;
    bool shaderClock;
    bool shaderFloat64;
    bool shaderInt16;
    bool textureCompressionBC;
  };

  struct CgpuDeviceProperties
  {
    uint32_t maxComputeSharedMemorySize;
    uint32_t maxPushConstantsSize;
    uint32_t maxRayHitAttributeSize;
    uint32_t subgroupSize;
  };

  struct CgpuWaitSemaphoreInfo
  {
    CgpuSemaphore semaphore;
    uint64_t value = 0;
  };

  struct CgpuSignalSemaphoreInfo
  {
    CgpuSemaphore semaphore;
    uint64_t value = 0;
  };

  struct CgpuBufferImageCopyDesc
  {
    uint64_t bufferOffset = 0;
    int32_t texelOffsetX = 0;
    int32_t texelOffsetY = 0;
    int32_t texelOffsetZ = 0;
    uint32_t texelExtentX;
    uint32_t texelExtentY;
    uint32_t texelExtentZ;
  };

  bool cgpuInitialize(
    const char* appName,
    uint32_t versionMajor,
    uint32_t versionMinor,
    uint32_t versionPatch
  );

  void cgpuTerminate();

  bool cgpuCreateDevice(
    CgpuDevice* device
  );

  void cgpuDestroyDevice(
    CgpuDevice device
  );

  bool cgpuCreateShader(
    CgpuDevice device,
    CgpuShaderCreateInfo createInfo,
    CgpuShader* shader
  );

  bool cgpuCreateShaders(
    CgpuDevice device,
    uint32_t shaderCount,
    CgpuShaderCreateInfo* createInfos,
    CgpuShader* shaders
  );

  void cgpuDestroyShader(
    CgpuDevice device,
    CgpuShader shader
  );

  bool cgpuCreateBuffer(
    CgpuDevice device,
    CgpuBufferCreateInfo createInfo,
    CgpuBuffer* buffer
  );

  void cgpuDestroyBuffer(
    CgpuDevice device,
    CgpuBuffer buffer
  );

  void cgpuMapBuffer(
    CgpuDevice device,
    CgpuBuffer buffer,
    void** mappedMem
  );

  void cgpuUnmapBuffer(
    CgpuDevice device,
    CgpuBuffer buffer
  );

  uint64_t cgpuGetBufferAddress(
    CgpuDevice device,
    CgpuBuffer buffer
  );

  bool cgpuCreateImage(
    CgpuDevice device,
    CgpuImageCreateInfo createInfo,
    CgpuImage* image
  );

  void cgpuDestroyImage(
    CgpuDevice device,
    CgpuImage image
  );

  bool cgpuCreateSampler(
    CgpuDevice device,
    CgpuSamplerCreateInfo createInfo,
    CgpuSampler* sampler
  );

  void cgpuDestroySampler(
    CgpuDevice device,
    CgpuSampler sampler
  );

  void cgpuCreateComputePipeline(
    CgpuDevice device,
    CgpuComputePipelineCreateInfo createInfo,
    CgpuPipeline* pipeline
  );

  void cgpuCreateRtPipeline(
    CgpuDevice device,
    CgpuRtPipelineCreateInfo createInfo,
    CgpuPipeline* pipeline
  );

  void cgpuDestroyPipeline(
    CgpuDevice device,
    CgpuPipeline pipeline
  );

  bool cgpuCreateBlas(
    CgpuDevice device,
    CgpuBlasCreateInfo createInfo,
    CgpuBlas* blas
  );

  bool cgpuCreateTlas(
    CgpuDevice device,
    CgpuTlasCreateInfo createInfo,
    CgpuTlas* tlas
  );

  void cgpuDestroyBlas(
    CgpuDevice device,
    CgpuBlas blas
  );

  void cgpuDestroyTlas(
    CgpuDevice device,
    CgpuTlas tlas
  );

  bool cgpuCreateCommandBuffer(
    CgpuDevice device,
    CgpuCommandBuffer* commandBuffer
  );

  bool cgpuBeginCommandBuffer(
    CgpuCommandBuffer commandBuffer
  );

  void cgpuCmdBindPipeline(
    CgpuCommandBuffer commandBuffer,
    CgpuPipeline pipeline
  );

  void cgpuCmdTransitionShaderImageLayouts(
    CgpuCommandBuffer commandBuffer,
    CgpuShader shader,
    uint32_t descriptorSetIndex,
    uint32_t imageCount,
    const CgpuImageBinding* images
  );

  void cgpuCmdUpdateBindings(
    CgpuCommandBuffer commandBuffer,
    CgpuPipeline pipeline,
    uint32_t descriptorSetIndex,
    const CgpuBindings* bindings
  );

  void cgpuCmdUpdateBuffer(
    CgpuCommandBuffer commandBuffer,
    const uint8_t* data,
    uint64_t size,
    CgpuBuffer dstBuffer,
    uint64_t dstOffset
  );

  void cgpuCmdCopyBuffer(
    CgpuCommandBuffer commandBuffer,
    CgpuBuffer srcBuffer,
    uint64_t srcOffset,
    CgpuBuffer dstBuffer,
    uint64_t dstOffset = 0,
    uint64_t size = CGPU_WHOLE_SIZE
  );

  void cgpuCmdCopyBufferToImage(
    CgpuCommandBuffer commandBuffer,
    CgpuBuffer buffer,
    CgpuImage image,
    const CgpuBufferImageCopyDesc* desc
  );

  void cgpuCmdPushConstants(
    CgpuCommandBuffer commandBuffer,
    CgpuPipeline pipeline,
    CgpuShaderStageFlags stageFlags,
    uint32_t size,
    const void* data
  );

  void cgpuCmdDispatch(
    CgpuCommandBuffer commandBuffer,
    uint32_t dimX,
    uint32_t dimY,
    uint32_t dimZ
  );

  void cgpuCmdPipelineBarrier(
    CgpuCommandBuffer commandBuffer,
    const CgpuPipelineBarrier* barrier
  );

  void cgpuCmdResetTimestamps(
    CgpuCommandBuffer commandBuffer,
    uint32_t offset,
    uint32_t count
  );

  void cgpuCmdWriteTimestamp(
    CgpuCommandBuffer commandBuffer,
    uint32_t timestampIndex
  );

  void cgpuCmdCopyTimestamps(
    CgpuCommandBuffer commandBuffer,
    CgpuBuffer buffer,
    uint32_t offset,
    uint32_t count,
    bool waitUntilAvailable
  );

  void cgpuCmdTraceRays(
    CgpuCommandBuffer commandBuffer,
    CgpuPipeline rtPipeline,
    uint32_t width,
    uint32_t height
  );

  void cgpuCmdFillBuffer(
    CgpuCommandBuffer commandBuffer,
    CgpuBuffer buffer,
    uint64_t dstOffset = 0,
    uint64_t size = CGPU_WHOLE_SIZE,
    uint8_t data = 0
  );

  void cgpuEndCommandBuffer(
    CgpuCommandBuffer commandBuffer
  );

  void cgpuDestroyCommandBuffer(
    CgpuDevice device,
    CgpuCommandBuffer commandBuffer
  );

  bool cgpuCreateSemaphore(
    CgpuDevice device,
    CgpuSemaphore* semaphore,
    uint64_t initialValue = 0
  );

  void cgpuDestroySemaphore(
    CgpuDevice device,
    CgpuSemaphore semaphore
  );

  bool cgpuWaitSemaphores(
    CgpuDevice device,
    uint32_t semaphoreInfoCount,
    CgpuWaitSemaphoreInfo* semaphoreInfos,
    uint64_t timeoutNs = UINT64_MAX
  );

  void cgpuSubmitCommandBuffer(
    CgpuDevice device,
    CgpuCommandBuffer commandBuffer,
    uint32_t signalSemaphoreInfoCount = 0,
    CgpuSignalSemaphoreInfo* signalSemaphoreInfos = nullptr,
    uint32_t waitSemaphoreInfoCount = 0,
    CgpuWaitSemaphoreInfo* waitSemaphoreInfos = nullptr
  );

  void cgpuFlushMappedMemory(
    CgpuDevice device,
    CgpuBuffer buffer,
    uint64_t offset,
    uint64_t size
  );

  void cgpuInvalidateMappedMemory(
    CgpuDevice device,
    CgpuBuffer buffer,
    uint64_t offset,
    uint64_t size
  );

  void cgpuGetDeviceFeatures(
    CgpuDevice device,
    CgpuDeviceFeatures& features
  );

  void cgpuGetDeviceProperties(
    CgpuDevice device,
    CgpuDeviceProperties& limits
  );
}
