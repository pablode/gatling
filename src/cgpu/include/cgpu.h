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

#define CGPU_WHOLE_SIZE (~0ULL)

#define CGPU_MAX_TIMESTAMP_QUERIES 32

typedef uint32_t CgpuBufferUsageFlags;

enum CgpuBufferUsageFlagBits
{
  CGPU_BUFFER_USAGE_FLAG_TRANSFER_SRC                       = 0x00000001,
  CGPU_BUFFER_USAGE_FLAG_TRANSFER_DST                       = 0x00000002,
  CGPU_BUFFER_USAGE_FLAG_UNIFORM_BUFFER                     = 0x00000010,
  CGPU_BUFFER_USAGE_FLAG_STORAGE_BUFFER                     = 0x00000020,
  CGPU_BUFFER_USAGE_FLAG_SHADER_DEVICE_ADDRESS              = 0x00020000,
  CGPU_BUFFER_USAGE_FLAG_ACCELERATION_STRUCTURE_BUILD_INPUT = 0x00080000,
  CGPU_BUFFER_USAGE_FLAG_ACCELERATION_STRUCTURE_STORAGE     = 0x00100000,
  CGPU_BUFFER_USAGE_FLAG_SHADER_BINDING_TABLE_BIT_KHR       = 0x00000400
};

typedef uint32_t CgpuMemoryPropertyFlags;

enum CgpuMemoryPropertyFlagBits
{
  CGPU_MEMORY_PROPERTY_FLAG_DEVICE_LOCAL  = 0x00000001,
  CGPU_MEMORY_PROPERTY_FLAG_HOST_VISIBLE  = 0x00000002,
  CGPU_MEMORY_PROPERTY_FLAG_HOST_COHERENT = 0x00000004,
  CGPU_MEMORY_PROPERTY_FLAG_HOST_CACHED   = 0x00000008
};

typedef uint32_t CgpuImageUsageFlags;

enum CgpuImageUsageFlagBits
{
  CGPU_IMAGE_USAGE_FLAG_TRANSFER_SRC = 0x00000001,
  CGPU_IMAGE_USAGE_FLAG_TRANSFER_DST = 0x00000002,
  CGPU_IMAGE_USAGE_FLAG_SAMPLED      = 0x00000004,
  CGPU_IMAGE_USAGE_FLAG_STORAGE      = 0x00000008
};

enum CgpuImageFormat
{
  CGPU_IMAGE_FORMAT_UNDEFINED = 0,
  CGPU_IMAGE_FORMAT_R4G4_UNORM_PACK8 = 1,
  CGPU_IMAGE_FORMAT_R4G4B4A4_UNORM_PACK16 = 2,
  CGPU_IMAGE_FORMAT_B4G4R4A4_UNORM_PACK16 = 3,
  CGPU_IMAGE_FORMAT_R5G6B5_UNORM_PACK16 = 4,
  CGPU_IMAGE_FORMAT_B5G6R5_UNORM_PACK16 = 5,
  CGPU_IMAGE_FORMAT_R5G5B5A1_UNORM_PACK16 = 6,
  CGPU_IMAGE_FORMAT_B5G5R5A1_UNORM_PACK16 = 7,
  CGPU_IMAGE_FORMAT_A1R5G5B5_UNORM_PACK16 = 8,
  CGPU_IMAGE_FORMAT_R8_UNORM = 9,
  CGPU_IMAGE_FORMAT_R8_SNORM = 10,
  CGPU_IMAGE_FORMAT_R8_USCALED = 11,
  CGPU_IMAGE_FORMAT_R8_SSCALED = 12,
  CGPU_IMAGE_FORMAT_R8_UINT = 13,
  CGPU_IMAGE_FORMAT_R8_SINT = 14,
  CGPU_IMAGE_FORMAT_R8_SRGB = 15,
  CGPU_IMAGE_FORMAT_R8G8_UNORM = 16,
  CGPU_IMAGE_FORMAT_R8G8_SNORM = 17,
  CGPU_IMAGE_FORMAT_R8G8_USCALED = 18,
  CGPU_IMAGE_FORMAT_R8G8_SSCALED = 19,
  CGPU_IMAGE_FORMAT_R8G8_UINT = 20,
  CGPU_IMAGE_FORMAT_R8G8_SINT = 21,
  CGPU_IMAGE_FORMAT_R8G8_SRGB = 22,
  CGPU_IMAGE_FORMAT_R8G8B8_UNORM = 23,
  CGPU_IMAGE_FORMAT_R8G8B8_SNORM = 24,
  CGPU_IMAGE_FORMAT_R8G8B8_USCALED = 25,
  CGPU_IMAGE_FORMAT_R8G8B8_SSCALED = 26,
  CGPU_IMAGE_FORMAT_R8G8B8_UINT = 27,
  CGPU_IMAGE_FORMAT_R8G8B8_SINT = 28,
  CGPU_IMAGE_FORMAT_R8G8B8_SRGB = 29,
  CGPU_IMAGE_FORMAT_B8G8R8_UNORM = 30,
  CGPU_IMAGE_FORMAT_B8G8R8_SNORM = 31,
  CGPU_IMAGE_FORMAT_B8G8R8_USCALED = 32,
  CGPU_IMAGE_FORMAT_B8G8R8_SSCALED = 33,
  CGPU_IMAGE_FORMAT_B8G8R8_UINT = 34,
  CGPU_IMAGE_FORMAT_B8G8R8_SINT = 35,
  CGPU_IMAGE_FORMAT_B8G8R8_SRGB = 36,
  CGPU_IMAGE_FORMAT_R8G8B8A8_UNORM = 37,
  CGPU_IMAGE_FORMAT_R8G8B8A8_SNORM = 38,
  CGPU_IMAGE_FORMAT_R8G8B8A8_USCALED = 39,
  CGPU_IMAGE_FORMAT_R8G8B8A8_SSCALED = 40,
  CGPU_IMAGE_FORMAT_R8G8B8A8_UINT = 41,
  CGPU_IMAGE_FORMAT_R8G8B8A8_SINT = 42,
  CGPU_IMAGE_FORMAT_R8G8B8A8_SRGB = 43,
  CGPU_IMAGE_FORMAT_B8G8R8A8_UNORM = 44,
  CGPU_IMAGE_FORMAT_B8G8R8A8_SNORM = 45,
  CGPU_IMAGE_FORMAT_B8G8R8A8_USCALED = 46,
  CGPU_IMAGE_FORMAT_B8G8R8A8_SSCALED = 47,
  CGPU_IMAGE_FORMAT_B8G8R8A8_UINT = 48,
  CGPU_IMAGE_FORMAT_B8G8R8A8_SINT = 49,
  CGPU_IMAGE_FORMAT_B8G8R8A8_SRGB = 50,
  CGPU_IMAGE_FORMAT_A8B8G8R8_UNORM_PACK32 = 51,
  CGPU_IMAGE_FORMAT_A8B8G8R8_SNORM_PACK32 = 52,
  CGPU_IMAGE_FORMAT_A8B8G8R8_USCALED_PACK32 = 53,
  CGPU_IMAGE_FORMAT_A8B8G8R8_SSCALED_PACK32 = 54,
  CGPU_IMAGE_FORMAT_A8B8G8R8_UINT_PACK32 = 55,
  CGPU_IMAGE_FORMAT_A8B8G8R8_SINT_PACK32 = 56,
  CGPU_IMAGE_FORMAT_A8B8G8R8_SRGB_PACK32 = 57,
  CGPU_IMAGE_FORMAT_A2R10G10B10_UNORM_PACK32 = 58,
  CGPU_IMAGE_FORMAT_A2R10G10B10_SNORM_PACK32 = 59,
  CGPU_IMAGE_FORMAT_A2R10G10B10_USCALED_PACK32 = 60,
  CGPU_IMAGE_FORMAT_A2R10G10B10_SSCALED_PACK32 = 61,
  CGPU_IMAGE_FORMAT_A2R10G10B10_UINT_PACK32 = 62,
  CGPU_IMAGE_FORMAT_A2R10G10B10_SINT_PACK32 = 63,
  CGPU_IMAGE_FORMAT_A2B10G10R10_UNORM_PACK32 = 64,
  CGPU_IMAGE_FORMAT_A2B10G10R10_SNORM_PACK32 = 65,
  CGPU_IMAGE_FORMAT_A2B10G10R10_USCALED_PACK32 = 66,
  CGPU_IMAGE_FORMAT_A2B10G10R10_SSCALED_PACK32 = 67,
  CGPU_IMAGE_FORMAT_A2B10G10R10_UINT_PACK32 = 68,
  CGPU_IMAGE_FORMAT_A2B10G10R10_SINT_PACK32 = 69,
  CGPU_IMAGE_FORMAT_R16_UNORM = 70,
  CGPU_IMAGE_FORMAT_R16_SNORM = 71,
  CGPU_IMAGE_FORMAT_R16_USCALED = 72,
  CGPU_IMAGE_FORMAT_R16_SSCALED = 73,
  CGPU_IMAGE_FORMAT_R16_UINT = 74,
  CGPU_IMAGE_FORMAT_R16_SINT = 75,
  CGPU_IMAGE_FORMAT_R16_SFLOAT = 76,
  CGPU_IMAGE_FORMAT_R16G16_UNORM = 77,
  CGPU_IMAGE_FORMAT_R16G16_SNORM = 78,
  CGPU_IMAGE_FORMAT_R16G16_USCALED = 79,
  CGPU_IMAGE_FORMAT_R16G16_SSCALED = 80,
  CGPU_IMAGE_FORMAT_R16G16_UINT = 81,
  CGPU_IMAGE_FORMAT_R16G16_SINT = 82,
  CGPU_IMAGE_FORMAT_R16G16_SFLOAT = 83,
  CGPU_IMAGE_FORMAT_R16G16B16_UNORM = 84,
  CGPU_IMAGE_FORMAT_R16G16B16_SNORM = 85,
  CGPU_IMAGE_FORMAT_R16G16B16_USCALED = 86,
  CGPU_IMAGE_FORMAT_R16G16B16_SSCALED = 87,
  CGPU_IMAGE_FORMAT_R16G16B16_UINT = 88,
  CGPU_IMAGE_FORMAT_R16G16B16_SINT = 89,
  CGPU_IMAGE_FORMAT_R16G16B16_SFLOAT = 90,
  CGPU_IMAGE_FORMAT_R16G16B16A16_UNORM = 91,
  CGPU_IMAGE_FORMAT_R16G16B16A16_SNORM = 92,
  CGPU_IMAGE_FORMAT_R16G16B16A16_USCALED = 93,
  CGPU_IMAGE_FORMAT_R16G16B16A16_SSCALED = 94,
  CGPU_IMAGE_FORMAT_R16G16B16A16_UINT = 95,
  CGPU_IMAGE_FORMAT_R16G16B16A16_SINT = 96,
  CGPU_IMAGE_FORMAT_R16G16B16A16_SFLOAT = 97,
  CGPU_IMAGE_FORMAT_R32_UINT = 98,
  CGPU_IMAGE_FORMAT_R32_SINT = 99,
  CGPU_IMAGE_FORMAT_R32_SFLOAT = 100,
  CGPU_IMAGE_FORMAT_R32G32_UINT = 101,
  CGPU_IMAGE_FORMAT_R32G32_SINT = 102,
  CGPU_IMAGE_FORMAT_R32G32_SFLOAT = 103,
  CGPU_IMAGE_FORMAT_R32G32B32_UINT = 104,
  CGPU_IMAGE_FORMAT_R32G32B32_SINT = 105,
  CGPU_IMAGE_FORMAT_R32G32B32_SFLOAT = 106,
  CGPU_IMAGE_FORMAT_R32G32B32A32_UINT = 107,
  CGPU_IMAGE_FORMAT_R32G32B32A32_SINT = 108,
  CGPU_IMAGE_FORMAT_R32G32B32A32_SFLOAT = 109,
  CGPU_IMAGE_FORMAT_R64_UINT = 110,
  CGPU_IMAGE_FORMAT_R64_SINT = 111,
  CGPU_IMAGE_FORMAT_R64_SFLOAT = 112,
  CGPU_IMAGE_FORMAT_R64G64_UINT = 113,
  CGPU_IMAGE_FORMAT_R64G64_SINT = 114,
  CGPU_IMAGE_FORMAT_R64G64_SFLOAT = 115,
  CGPU_IMAGE_FORMAT_R64G64B64_UINT = 116,
  CGPU_IMAGE_FORMAT_R64G64B64_SINT = 117,
  CGPU_IMAGE_FORMAT_R64G64B64_SFLOAT = 118,
  CGPU_IMAGE_FORMAT_R64G64B64A64_UINT = 119,
  CGPU_IMAGE_FORMAT_R64G64B64A64_SINT = 120,
  CGPU_IMAGE_FORMAT_R64G64B64A64_SFLOAT = 121,
  CGPU_IMAGE_FORMAT_B10G11R11_UFLOAT_PACK32 = 122,
  CGPU_IMAGE_FORMAT_E5B9G9R9_UFLOAT_PACK32 = 123,
  CGPU_IMAGE_FORMAT_D16_UNORM = 124,
  CGPU_IMAGE_FORMAT_X8_D24_UNORM_PACK32 = 125,
  CGPU_IMAGE_FORMAT_D32_SFLOAT = 126,
  CGPU_IMAGE_FORMAT_S8_UINT = 127,
  CGPU_IMAGE_FORMAT_D16_UNORM_S8_UINT = 128,
  CGPU_IMAGE_FORMAT_D24_UNORM_S8_UINT = 129,
  CGPU_IMAGE_FORMAT_D32_SFLOAT_S8_UINT = 130,
  CGPU_IMAGE_FORMAT_BC7_UNORM_BLOCK = 145,
  CGPU_IMAGE_FORMAT_BC7_SRGB_BLOCK = 146,
  CGPU_IMAGE_FORMAT_G8B8G8R8_422_UNORM = 1000156000,
  CGPU_IMAGE_FORMAT_B8G8R8G8_422_UNORM = 1000156001,
  CGPU_IMAGE_FORMAT_G8_B8_R8_3PLANE_420_UNORM = 1000156002,
  CGPU_IMAGE_FORMAT_G8_B8R8_2PLANE_420_UNORM = 1000156003,
  CGPU_IMAGE_FORMAT_G8_B8_R8_3PLANE_422_UNORM = 1000156004,
  CGPU_IMAGE_FORMAT_G8_B8R8_2PLANE_422_UNORM = 1000156005,
  CGPU_IMAGE_FORMAT_G8_B8_R8_3PLANE_444_UNORM = 1000156006,
  CGPU_IMAGE_FORMAT_R10X6_UNORM_PACK16 = 1000156007,
  CGPU_IMAGE_FORMAT_R10X6G10X6_UNORM_2PACK16 = 1000156008,
  CGPU_IMAGE_FORMAT_R10X6G10X6B10X6A10X6_UNORM_4PACK16 = 1000156009,
  CGPU_IMAGE_FORMAT_G10X6B10X6G10X6R10X6_422_UNORM_4PACK16 = 1000156010,
  CGPU_IMAGE_FORMAT_B10X6G10X6R10X6G10X6_422_UNORM_4PACK16 = 1000156011,
  CGPU_IMAGE_FORMAT_G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16 = 1000156012,
  CGPU_IMAGE_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16 = 1000156013,
  CGPU_IMAGE_FORMAT_G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16 = 1000156014,
  CGPU_IMAGE_FORMAT_G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16 = 1000156015,
  CGPU_IMAGE_FORMAT_G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16 = 1000156016,
  CGPU_IMAGE_FORMAT_R12X4_UNORM_PACK16 = 1000156017,
  CGPU_IMAGE_FORMAT_R12X4G12X4_UNORM_2PACK16 = 1000156018,
  CGPU_IMAGE_FORMAT_R12X4G12X4B12X4A12X4_UNORM_4PACK16 = 1000156019,
  CGPU_IMAGE_FORMAT_G12X4B12X4G12X4R12X4_422_UNORM_4PACK16 = 1000156020,
  CGPU_IMAGE_FORMAT_B12X4G12X4R12X4G12X4_422_UNORM_4PACK16 = 1000156021,
  CGPU_IMAGE_FORMAT_G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16 = 1000156022,
  CGPU_IMAGE_FORMAT_G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16 = 1000156023,
  CGPU_IMAGE_FORMAT_G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16 = 1000156024,
  CGPU_IMAGE_FORMAT_G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16 = 1000156025,
  CGPU_IMAGE_FORMAT_G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16 = 1000156026,
  CGPU_IMAGE_FORMAT_G16B16G16R16_422_UNORM = 1000156027,
  CGPU_IMAGE_FORMAT_B16G16R16G16_422_UNORM = 1000156028,
  CGPU_IMAGE_FORMAT_G16_B16_R16_3PLANE_420_UNORM = 1000156029,
  CGPU_IMAGE_FORMAT_G16_B16R16_2PLANE_420_UNORM = 1000156030,
  CGPU_IMAGE_FORMAT_G16_B16_R16_3PLANE_422_UNORM = 1000156031,
  CGPU_IMAGE_FORMAT_G16_B16R16_2PLANE_422_UNORM = 1000156032,
  CGPU_IMAGE_FORMAT_G16_B16_R16_3PLANE_444_UNORM = 1000156033
};

typedef uint32_t CgpuMemoryAccessFlags;

enum CgpuMemoryAccessFlagBits
{
  CGPU_MEMORY_ACCESS_FLAG_UNIFORM_READ   = 0x00000008,
  CGPU_MEMORY_ACCESS_FLAG_SHADER_READ    = 0x00000020,
  CGPU_MEMORY_ACCESS_FLAG_SHADER_WRITE   = 0x00000040,
  CGPU_MEMORY_ACCESS_FLAG_TRANSFER_READ  = 0x00000800,
  CGPU_MEMORY_ACCESS_FLAG_TRANSFER_WRITE = 0x00001000,
  CGPU_MEMORY_ACCESS_FLAG_HOST_READ      = 0x00002000,
  CGPU_MEMORY_ACCESS_FLAG_HOST_WRITE     = 0x00004000,
  CGPU_MEMORY_ACCESS_FLAG_MEMORY_READ    = 0x00008000,
  CGPU_MEMORY_ACCESS_FLAG_MEMORY_WRITE   = 0x00010000
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
  CGPU_SHADER_STAGE_COMPUTE     = 0x00000020,
  CGPU_SHADER_STAGE_RAYGEN      = 0x00000100,
  CGPU_SHADER_STAGE_ANY_HIT     = 0x00000200,
  CGPU_SHADER_STAGE_CLOSEST_HIT = 0x00000400,
  CGPU_SHADER_STAGE_MISS        = 0x00000800
};

struct CgpuInstance      { uint64_t handle = 0; };
struct CgpuDevice        { uint64_t handle = 0; };
struct CgpuBuffer        { uint64_t handle = 0; };
struct CgpuImage         { uint64_t handle = 0; };
struct CgpuShader        { uint64_t handle = 0; };
struct CgpuPipeline      { uint64_t handle = 0; };
struct CgpuFence         { uint64_t handle = 0; };
struct CgpuCommandBuffer { uint64_t handle = 0; };
struct CgpuSampler       { uint64_t handle = 0; };
struct CgpuBlas          { uint64_t handle = 0; };
struct CgpuTlas          { uint64_t handle = 0; };

struct CgpuImageDesc
{
  uint32_t width;
  uint32_t height;
  bool is3d = false;
  uint32_t depth = 1;
  CgpuImageFormat format = CGPU_IMAGE_FORMAT_R8G8B8A8_UNORM;
  CgpuImageUsageFlags usage = CGPU_IMAGE_USAGE_FLAG_TRANSFER_DST |
                              CGPU_IMAGE_USAGE_FLAG_SAMPLED;
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
  CgpuMemoryAccessFlags srcAccessFlags;
  CgpuMemoryAccessFlags dstAccessFlags;
};

struct CgpuBufferMemoryBarrier
{
  CgpuBuffer buffer;
  CgpuMemoryAccessFlags srcAccessFlags;
  CgpuMemoryAccessFlags dstAccessFlags;
  uint64_t offset = 0;
  uint64_t size = CGPU_WHOLE_SIZE;
};

struct CgpuImageMemoryBarrier
{
  CgpuImage image;
  CgpuMemoryAccessFlags accessMask;
};

struct CgpuPhysicalDeviceFeatures
{
  bool pageableDeviceLocalMemory;
  bool debugPrintf;
  bool textureCompressionBC;
  bool pipelineStatisticsQuery;
  bool shaderImageGatherExtended;
  bool shaderStorageImageExtendedFormats;
  bool shaderStorageImageReadWithoutFormat;
  bool shaderStorageImageWriteWithoutFormat;
  bool shaderUniformBufferArrayDynamicIndexing;
  bool shaderSampledImageArrayDynamicIndexing;
  bool shaderStorageBufferArrayDynamicIndexing;
  bool shaderStorageImageArrayDynamicIndexing;
  bool shaderClock;
  bool shaderFloat64;
  bool shaderInt64;
  bool shaderInt16;
  bool sparseBinding;
  bool sparseResidencyBuffer;
  bool sparseResidencyImage2D;
  bool sparseResidencyImage3D;
  bool sparseResidencyAliased;
  bool rayTracingInvocationReorder;
};

struct CgpuPhysicalDeviceProperties
{
  uint32_t maxImageDimension1D;
  uint32_t maxImageDimension2D;
  uint32_t maxImageDimension3D;
  uint32_t maxImageDimensionCube;
  uint32_t maxImageArrayLayers;
  uint32_t maxUniformBufferRange;
  uint32_t maxStorageBufferRange;
  uint32_t maxPushConstantsSize;
  uint32_t maxMemoryAllocationCount;
  uint32_t maxSamplerAllocationCount;
  uint64_t bufferImageGranularity;
  uint64_t sparseAddressSpaceSize;
  uint32_t maxBoundDescriptorSets;
  uint32_t maxPerStageDescriptorSamplers;
  uint32_t maxPerStageDescriptorUniformBuffers;
  uint32_t maxPerStageDescriptorStorageBuffers;
  uint32_t maxPerStageDescriptorSampledImages;
  uint32_t maxPerStageDescriptorStorageImages;
  uint32_t maxPerStageDescriptorInputAttachments;
  uint32_t maxPerStageResources;
  uint32_t maxDescriptorSetSamplers;
  uint32_t maxDescriptorSetUniformBuffers;
  uint32_t maxDescriptorSetUniformBuffersDynamic;
  uint32_t maxDescriptorSetStorageBuffers;
  uint32_t maxDescriptorSetStorageBuffersDynamic;
  uint32_t maxDescriptorSetSampledImages;
  uint32_t maxDescriptorSetStorageImages;
  uint32_t maxDescriptorSetInputAttachments;
  uint32_t maxComputeSharedMemorySize;
  uint32_t maxComputeWorkGroupCount[3];
  uint32_t maxComputeWorkGroupInvocations;
  uint32_t maxComputeWorkGroupSize[3];
  uint32_t mipmapPrecisionBits;
  float    maxSamplerLodBias;
  float    maxSamplerAnisotropy;
  size_t   minMemoryMapAlignment;
  uint64_t minUniformBufferOffsetAlignment;
  uint64_t minStorageBufferOffsetAlignment;
  int32_t  minTexelOffset;
  uint32_t maxTexelOffset;
  int32_t  minTexelGatherOffset;
  uint32_t maxTexelGatherOffset;
  float    minInterpolationOffset;
  float    maxInterpolationOffset;
  uint32_t subPixelInterpolationOffsetBits;
  uint32_t maxSampleMaskWords;
  bool     timestampComputeAndGraphics;
  float    timestampPeriod;
  uint32_t discreteQueuePriorities;
  uint64_t optimalBufferCopyOffsetAlignment;
  uint64_t optimalBufferCopyRowPitchAlignment;
  uint64_t nonCoherentAtomSize;
  uint32_t subgroupSize;
  uint64_t minAccelerationStructureScratchOffsetAlignment;
  uint32_t shaderGroupHandleSize;
  uint32_t maxShaderGroupStride;
  uint32_t shaderGroupBaseAlignment;
  uint32_t shaderGroupHandleCaptureReplaySize;
  uint32_t maxRayDispatchInvocationCount;
  uint32_t shaderGroupHandleAlignment;
  uint32_t maxRayHitAttributeSize;
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
  uint32_t faceIndexOffset;
  uint32_t hitGroupIndex;
  float transform[3][4];
};

struct CgpuRtHitGroup
{
  CgpuShader closestHitShader; // optional
  CgpuShader anyHitShader;     // optional
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

bool cgpuDestroyDevice(
  CgpuDevice device
);

bool cgpuCreateShader(
  CgpuDevice device,
  uint64_t size,
  const uint8_t* source,
  CgpuShaderStageFlags stageFlags,
  CgpuShader* shader
);

bool cgpuDestroyShader(
  CgpuDevice device,
  CgpuShader shader
);

bool cgpuCreateBuffer(
  CgpuDevice device,
  CgpuBufferUsageFlags usage,
  CgpuMemoryPropertyFlags memoryProperties,
  uint64_t size,
  CgpuBuffer* buffer
);

bool cgpuDestroyBuffer(
  CgpuDevice device,
  CgpuBuffer buffer
);

bool cgpuMapBuffer(
  CgpuDevice device,
  CgpuBuffer buffer,
  void** mappedMem
);

bool cgpuUnmapBuffer(
  CgpuDevice device,
  CgpuBuffer buffer
);

bool cgpuCreateImage(
  CgpuDevice device,
  const CgpuImageDesc* imageDesc,
  CgpuImage* image
);

bool cgpuDestroyImage(
  CgpuDevice device,
  CgpuImage image
);

bool cgpuMapImage(
  CgpuDevice device,
  CgpuImage image,
  void** mappedMem
);

bool cgpuUnmapImage(
  CgpuDevice device,
  CgpuImage image
);

bool cgpuCreateSampler(
  CgpuDevice device,
  CgpuSamplerAddressMode addressModeU,
  CgpuSamplerAddressMode addressModeV,
  CgpuSamplerAddressMode addressModeW,
  CgpuSampler* sampler
);

bool cgpuDestroySampler(
  CgpuDevice device,
  CgpuSampler sampler
);

bool cgpuCreateComputePipeline(
  CgpuDevice device,
  CgpuShader shader,
  CgpuPipeline* pipeline
);

struct CgpuRtPipelineDesc
{
  CgpuShader rgenShader;
  uint32_t missShaderCount = 0;
  CgpuShader* missShaders = nullptr;
  uint32_t hitGroupCount = 0;
  const CgpuRtHitGroup* hitGroups = nullptr;
};

bool cgpuCreateRtPipeline(
  CgpuDevice device,
  const CgpuRtPipelineDesc* desc,
  CgpuPipeline* pipeline
);

bool cgpuDestroyPipeline(
  CgpuDevice device,
  CgpuPipeline pipeline
);

bool cgpuCreateBlas(
  CgpuDevice device,
  uint32_t vertexCount,
  const CgpuVertex* vertices,
  uint32_t indexCount,
  const uint32_t* indices,
  bool isOpaque,
  CgpuBlas* blas
);

bool cgpuCreateTlas(
  CgpuDevice device,
  uint32_t instanceCount,
  const CgpuBlasInstance* instances,
  CgpuTlas* tlas
);

bool cgpuDestroyBlas(
  CgpuDevice device,
  CgpuBlas blas
);

bool cgpuDestroyTlas(
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

bool cgpuCmdBindPipeline(
  CgpuCommandBuffer commandBuffer,
  CgpuPipeline pipeline
);

bool cgpuCmdTransitionShaderImageLayouts(
  CgpuCommandBuffer commandBuffer,
  CgpuShader shader,
  uint32_t imageCount,
  const CgpuImageBinding* images
);

bool cgpuCmdUpdateBindings(
  CgpuCommandBuffer commandBuffer,
  CgpuPipeline pipeline,
  const CgpuBindings* bindings
);

bool cgpuCmdUpdateBuffer(
  CgpuCommandBuffer commandBuffer,
  const uint8_t* data,
  uint64_t size,
  CgpuBuffer dstBuffer,
  uint64_t dstOffset
);

bool cgpuCmdCopyBuffer(
  CgpuCommandBuffer commandBuffer,
  CgpuBuffer srcBuffer,
  uint64_t srcOffset,
  CgpuBuffer dstBuffer,
  uint64_t dstOffset,
  uint64_t size
);

struct CgpuBufferImageCopyDesc
{
  uint64_t bufferOffset = 0;
  uint32_t texelOffsetX = 0;
  uint32_t texelOffsetY = 0;
  uint32_t texelOffsetZ = 0;
  uint32_t texelExtentX;
  uint32_t texelExtentY;
  uint32_t texelExtentZ;
};

bool cgpuCmdCopyBufferToImage(
  CgpuCommandBuffer commandBuffer,
  CgpuBuffer buffer,
  CgpuImage image,
  const CgpuBufferImageCopyDesc* desc
);

bool cgpuCmdPushConstants(
  CgpuCommandBuffer commandBuffer,
  CgpuPipeline pipeline,
  CgpuShaderStageFlags stageFlags,
  uint32_t size,
  const void* data
);

bool cgpuCmdDispatch(
  CgpuCommandBuffer commandBuffer,
  uint32_t dimX,
  uint32_t dimY,
  uint32_t dimZ
);

bool cgpuCmdPipelineBarrier(
  CgpuCommandBuffer commandBuffer,
  uint32_t barrierCount = 0,
  const CgpuMemoryBarrier* barriers = nullptr,
  uint32_t bufferBarrierCount = 0,
  const CgpuBufferMemoryBarrier* bufferBarriers = nullptr,
  uint32_t imageBarrierCount = 0,
  const CgpuImageMemoryBarrier* imageBarriers = nullptr
);

bool cgpuCmdResetTimestamps(
  CgpuCommandBuffer commandBuffer,
  uint32_t offset,
  uint32_t count
);

bool cgpuCmdWriteTimestamp(
  CgpuCommandBuffer commandBuffer,
  uint32_t timestampIndex
);

bool cgpuCmdCopyTimestamps(
  CgpuCommandBuffer commandBuffer,
  CgpuBuffer buffer,
  uint32_t offset,
  uint32_t count,
  bool waitUntilAvailable
);

bool cgpuCmdTraceRays(
  CgpuCommandBuffer commandBuffer,
  CgpuPipeline rtPipeline,
  uint32_t width,
  uint32_t height
);

bool cgpuEndCommandBuffer(
  CgpuCommandBuffer commandBuffer
);

bool cgpuDestroyCommandBuffer(
  CgpuDevice device,
  CgpuCommandBuffer commandBuffer
);

bool cgpuCreateFence(
  CgpuDevice device,
  CgpuFence* fence
);

bool cgpuResetFence(
  CgpuDevice device,
  CgpuFence fence
);

bool cgpuWaitForFence(
  CgpuDevice device,
  CgpuFence fence
);

bool cgpuDestroyFence(
  CgpuDevice device,
  CgpuFence fence
);

bool cgpuSubmitCommandBuffer(
  CgpuDevice device,
  CgpuCommandBuffer commandBuffer,
  CgpuFence fence
);

bool cgpuFlushMappedMemory(
  CgpuDevice device,
  CgpuBuffer buffer,
  uint64_t offset,
  uint64_t size
);

bool cgpuInvalidateMappedMemory(
  CgpuDevice device,
  CgpuBuffer buffer,
  uint64_t offset,
  uint64_t size
);

bool cgpuGetPhysicalDeviceFeatures(
  CgpuDevice device,
  CgpuPhysicalDeviceFeatures* features
);

bool cgpuGetPhysicalDeviceProperties(
  CgpuDevice device,
  CgpuPhysicalDeviceProperties* limits
);
