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

#include <gtl/gb/Enum.h>

namespace gtl
{
  constexpr static const uint64_t CGPU_WHOLE_SIZE = ~0ULL;
  constexpr static const uint32_t CGPU_MAX_TIMESTAMP_QUERIES = 32;
  constexpr static const uint32_t CGPU_MAX_DESCRIPTOR_SET_COUNT = 4;
  constexpr static const uint32_t CGPU_MAX_BUFFER_UPDATE_SIZE = 65535;

  enum class CgpuBufferUsage
  {
    TransferSrc = 0x00000001,
    TransferDst = 0x00000002,
    Storage = 0x00000020,
    ShaderDeviceAddress = 0x00020000,
    AccelerationStructureBuild = 0x00080000,
    AccelerationStructureStorage = 0x00100000,
    ShaderBindingTable = 0x00000400
  };
  GB_DECLARE_ENUM_BITOPS(CgpuBufferUsage);

  enum class CgpuMemoryProperties
  {
    DeviceLocal = 0x00000001,
    HostVisible = 0x00000002,
    HostCoherent = 0x00000004,
    HostCached = 0x00000008
  };
  GB_DECLARE_ENUM_BITOPS(CgpuMemoryProperties);

  enum class CgpuImageUsage
  {
    TransferSrc = 0x00000001,
    TransferDst = 0x00000002,
    Sampled = 0x00000004,
    Storage = 0x00000008
  };
  GB_DECLARE_ENUM_BITOPS(CgpuImageUsage);

  enum class CgpuImageFormat
  {
    Undefined = 0,
    R8G8B8A8Unorm = 37,
    R16G16B16Sfloat = 90,
    R16G16B16A16Sfloat = 97,
    R32Sfloat = 100
  };

  enum class CgpuMemoryAccess
  {
    ShaderRead = 0x00000020,
    ShaderWrite = 0x00000040,
    TransferRead = 0x00000800,
    TransferWrite = 0x00001000,
    HostRead = 0x00002000,
    HostWrite = 0x00004000,
    MemoryRead = 0x00008000,
    MemoryWrite = 0x00010000,
    AccelerationStructureRead = 0x00200000,
    AccelerationStructureWrite = 0x00400000
  };
  GB_DECLARE_ENUM_BITOPS(CgpuMemoryAccess);

  enum class CgpuSamplerAddressMode
  {
    ClampToEdge = 0,
    Repeat = 1,
    MirrorRepeat = 2,
    ClampToBlack = 3
  };

  enum class CgpuShaderStage
  {
    Compute = 0x00000020,
    RayGen = 0x00000100,
    AnyHit = 0x00000200,
    ClosestHit = 0x00000400,
    Miss = 0x00000800
  };
  GB_DECLARE_ENUM_BITOPS(CgpuShaderStage);

  enum class CgpuPipelineStage
  {
    ComputeShader = 0x00000800,
    Transfer = 0x00001000,
    Host = 0x00004000,
    RayTracingShader = 0x00200000,
    AccelerationStructureBuild = 0x02000000
  };
  GB_DECLARE_ENUM_BITOPS(CgpuPipelineStage);

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
  struct CgpuBindSet       { uint64_t handle = 0; };

  struct CgpuImageCreateInfo
  {
    uint32_t width;
    uint32_t height;
    bool is3d = false;
    uint32_t depth = 1;
    CgpuImageFormat format = CgpuImageFormat::R8G8B8A8Unorm;
    CgpuImageUsage usage = CgpuImageUsage::TransferDst | CgpuImageUsage::Sampled;
    const char* debugName = nullptr;
  };

  struct CgpuBufferCreateInfo
  {
    CgpuBufferUsage usage;
    CgpuMemoryProperties memoryProperties;
    uint64_t size;
    const char* debugName = nullptr;
    uint32_t alignment = 0; // no explicit alignment
  };

  struct CgpuShaderCreateInfo
  {
    uint64_t size;
    const uint8_t* source;
    CgpuShaderStage stageFlags;
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
    CgpuBuffer vertexPosBuffer;
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
    CgpuPipelineStage srcStageMask;
    CgpuMemoryAccess srcAccessMask;
    CgpuPipelineStage dstStageMask;
    CgpuMemoryAccess dstAccessMask;
  };

  struct CgpuBufferMemoryBarrier
  {
    CgpuBuffer buffer;
    CgpuPipelineStage srcStageMask;
    CgpuMemoryAccess srcAccessMask;
    CgpuPipelineStage dstStageMask;
    CgpuMemoryAccess dstAccessMask;
    uint64_t offset = 0;
    uint64_t size = CGPU_WHOLE_SIZE;
  };

  struct CgpuImageMemoryBarrier
  {
    CgpuImage image;
    CgpuPipelineStage srcStageMask;
    CgpuPipelineStage dstStageMask;
    CgpuMemoryAccess accessMask;
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
    bool rebar;
    bool shaderClock;
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

  bool cgpuCreateShadersParallel(
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

  void cgpuCreateBindSets(
    CgpuDevice device,
    CgpuPipeline pipeline,
    CgpuBindSet* bindSets,
    uint32_t bindSetCount
  );

  void cgpuDestroyBindSets(
    CgpuDevice device,
    CgpuBindSet* bindSets,
    uint32_t bindSetCount
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
    CgpuPipeline pipeline,
    const CgpuBindSet* bindSets,
    uint32_t bindSetCount
  );

  void cgpuCmdTransitionShaderImageLayouts(
    CgpuCommandBuffer commandBuffer,
    CgpuShader shader,
    uint32_t descriptorSetIndex,
    uint32_t imageCount,
    const CgpuImageBinding* images
  );

  void cgpuCmdUpdateBindSet(
    CgpuCommandBuffer commandBuffer,
    CgpuBindSet bindSet,
    const CgpuBindings* bindings
  );

  void cgpuCmdUpdateBuffer(
    CgpuCommandBuffer commandBuffer,
    const uint8_t* data,
    uint64_t size,
    CgpuBuffer dstBuffer,
    uint64_t dstOffset = 0
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

  const CgpuDeviceFeatures& cgpuGetDeviceFeatures(CgpuDevice device);

  const CgpuDeviceProperties& cgpuGetDeviceProperties(CgpuDevice device);
}
