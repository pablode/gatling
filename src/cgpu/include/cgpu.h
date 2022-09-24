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

#ifndef CGPU_H
#define CGPU_H

#ifdef __cplusplus
extern "C" {
#endif

#if defined(CGPU_EXPORT_SYMBOLS) && defined(CGPU_IMPORT_SYMBOLS)
  #error "Symbols can not be exported and imported at the same time."
#endif

#if defined(BUILD_SHARED_LIBS)
  #if defined(_WIN32) && (defined(_MSC_VER) || defined(__MING32__))
    #if defined(CGPU_EXPORT_SYMBOLS)
      #define CGPU_API __declspec(dllexport)
    #elif defined(CGPU_IMPORT_SYMBOLS)
      #define CGPU_API __declspec(dllimport)
    #endif
  #elif defined(_WIN32) && defined(__GNUC__)
    #if defined(CGPU_EXPORT_SYMBOLS)
      #define CGPU_API __attribute__((dllexport))
    #elif defined(CGPU_IMPORT_SYMBOLS)
      #define CGPU_API __attribute__((dllimport))
    #endif
  #elif defined(__GNUC__)
    #define CGPU_API __attribute__((__visibility__("default")))
  #endif
#endif

#ifndef CGPU_API
  #define CGPU_API
#endif

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

#if defined(_MSC_VER)
  #define CGPU_CDECL __cdecl
#elif defined(__GNUC__) && defined(__i386__) && !defined(__x86_64__)
  #define CGPU_CDECL __attribute__((__cdecl__))
#else
  #define CGPU_CDECL
#endif

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#define CGPU_WHOLE_SIZE (~0ULL)
#define CGPU_INVALID_HANDLE 0xFFFFFFFFFFFFFFFF

#define CGPU_MAX_TIMESTAMP_QUERIES 32

typedef uint32_t CgpuBufferUsageFlags;

typedef enum CgpuBufferUsageFlagBits {
  CGPU_BUFFER_USAGE_FLAG_TRANSFER_SRC = 1,
  CGPU_BUFFER_USAGE_FLAG_TRANSFER_DST = 2,
  CGPU_BUFFER_USAGE_FLAG_UNIFORM_BUFFER = 4,
  CGPU_BUFFER_USAGE_FLAG_STORAGE_BUFFER = 8
} CgpuBufferUsageFlagBits;

typedef uint32_t CgpuMemoryPropertyFlags;

typedef enum CgpuMemoryPropertyFlagBits {
  CGPU_MEMORY_PROPERTY_FLAG_DEVICE_LOCAL = 1,
  CGPU_MEMORY_PROPERTY_FLAG_HOST_VISIBLE = 2,
  CGPU_MEMORY_PROPERTY_FLAG_HOST_COHERENT = 4,
  CGPU_MEMORY_PROPERTY_FLAG_HOST_CACHED = 8,
} CgpuMemoryPropertyFlagBits;

typedef uint32_t CgpuImageUsageFlags;

typedef enum CgpuImageUsageFlagBits {
  CGPU_IMAGE_USAGE_FLAG_TRANSFER_SRC = 1,
  CGPU_IMAGE_USAGE_FLAG_TRANSFER_DST = 2,
  CGPU_IMAGE_USAGE_FLAG_SAMPLED = 4,
  CGPU_IMAGE_USAGE_FLAG_STORAGE = 8
} CgpuImageUsageFlagBits;

typedef enum CgpuImageFormat {
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
} CgpuImageFormat;

typedef uint32_t CgpuMemoryAccessFlags;

typedef enum CgpuMemoryAccessFlagBits {
  CGPU_MEMORY_ACCESS_FLAG_UNDEFINED = 0,
  CGPU_MEMORY_ACCESS_FLAG_UNIFORM_READ = 1,
  CGPU_MEMORY_ACCESS_FLAG_SHADER_READ = 2,
  CGPU_MEMORY_ACCESS_FLAG_SHADER_WRITE = 4,
  CGPU_MEMORY_ACCESS_FLAG_TRANSFER_READ = 8,
  CGPU_MEMORY_ACCESS_FLAG_TRANSFER_WRITE = 16,
  CGPU_MEMORY_ACCESS_FLAG_HOST_READ = 32,
  CGPU_MEMORY_ACCESS_FLAG_HOST_WRITE = 64,
  CGPU_MEMORY_ACCESS_FLAG_MEMORY_READ = 128,
  CGPU_MEMORY_ACCESS_FLAG_MEMORY_WRITE = 256
} CgpuMemoryAccessFlagBits;

typedef enum CgpuSamplerAddressMode {
  CGPU_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE = 0,
  CGPU_SAMPLER_ADDRESS_MODE_REPEAT = 1,
  CGPU_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT = 2,
  CGPU_SAMPLER_ADDRESS_MODE_CLAMP_TO_BLACK = 3
} CgpuSamplerAddressMode;

typedef enum CgpuShaderStage {
  CGPU_SHADER_STAGE_COMPUTE,
  CGPU_SHADER_STAGE_RAYGEN,
  CGPU_SHADER_STAGE_ANY_HIT,
  CGPU_SHADER_STAGE_CLOSEST_HIT,
  CGPU_SHADER_STAGE_MISS
} CgpuShaderStage;

typedef struct cgpu_instance       { uint64_t handle; } cgpu_instance;
typedef struct cgpu_device         { uint64_t handle; } cgpu_device;
typedef struct cgpu_buffer         { uint64_t handle; } cgpu_buffer;
typedef struct cgpu_image          { uint64_t handle; } cgpu_image;
typedef struct cgpu_shader         { uint64_t handle; } cgpu_shader;
typedef struct cgpu_pipeline       { uint64_t handle; } cgpu_pipeline;
typedef struct cgpu_fence          { uint64_t handle; } cgpu_fence;
typedef struct cgpu_command_buffer { uint64_t handle; } cgpu_command_buffer;
typedef struct cgpu_sampler        { uint64_t handle; } cgpu_sampler;

typedef struct cgpu_image_description {
  bool is3d;
  uint32_t width;
  uint32_t height;
  uint32_t depth;
  CgpuImageFormat format;
  CgpuImageUsageFlags usage;
} cgpu_image_description;

typedef struct cgpu_buffer_binding {
  uint32_t binding;
  uint32_t index;
  cgpu_buffer buffer;
  uint64_t offset;
  uint64_t size;
} cgpu_buffer_binding;

typedef struct cgpu_image_binding {
  uint32_t binding;
  uint32_t index;
  cgpu_image image;
} cgpu_image_binding;

typedef struct cgpu_sampler_binding {
  uint32_t binding;
  uint32_t index;
  cgpu_sampler sampler;
} cgpu_sampler_binding;

typedef struct cgpu_bindings {
  uint32_t buffer_count;
  const cgpu_buffer_binding* p_buffers;
  uint32_t image_count;
  const cgpu_image_binding* p_images;
  uint32_t sampler_count;
  const cgpu_sampler_binding* p_samplers;
} cgpu_bindings;

typedef struct cgpu_memory_barrier {
  CgpuMemoryAccessFlags src_access_flags;
  CgpuMemoryAccessFlags dst_access_flags;
} cgpu_memory_barrier;

typedef struct cgpu_buffer_memory_barrier {
  cgpu_buffer buffer;
  CgpuMemoryAccessFlags src_access_flags;
  CgpuMemoryAccessFlags dst_access_flags;
  uint64_t offset;
  uint64_t size;
} cgpu_buffer_memory_barrier;

typedef struct cgpu_image_memory_barrier {
  cgpu_image image;
  CgpuMemoryAccessFlags access_mask;
} cgpu_image_memory_barrier;

typedef struct cgpu_physical_device_features {
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
} cgpu_physical_device_features;

typedef struct cgpu_physical_device_limits {
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
} cgpu_physical_device_limits;

CGPU_API bool CGPU_CDECL cgpu_initialize(
  const char* p_app_name,
  uint32_t version_major,
  uint32_t version_minor,
  uint32_t version_patch
);

CGPU_API void CGPU_CDECL cgpu_terminate(void);

CGPU_API bool CGPU_CDECL cgpu_create_device(
  cgpu_device* p_device
);

CGPU_API bool CGPU_CDECL cgpu_destroy_device(
  cgpu_device device
);

CGPU_API bool CGPU_CDECL cgpu_create_shader(
  cgpu_device device,
  uint64_t size,
  const uint8_t* p_source,
  CgpuShaderStage stage,
  cgpu_shader* p_shader
);

CGPU_API bool CGPU_CDECL cgpu_destroy_shader(
  cgpu_device device,
  cgpu_shader shader
);

CGPU_API bool CGPU_CDECL cgpu_create_buffer(
  cgpu_device device,
  CgpuBufferUsageFlags usage,
  CgpuMemoryPropertyFlags memory_properties,
  uint64_t size,
  cgpu_buffer* p_buffer
);

CGPU_API bool CGPU_CDECL cgpu_destroy_buffer(
  cgpu_device device,
  cgpu_buffer buffer
);

CGPU_API bool CGPU_CDECL cgpu_map_buffer(
  cgpu_device device,
  cgpu_buffer buffer,
  void** pp_mapped_mem
);

CGPU_API bool CGPU_CDECL cgpu_unmap_buffer(
  cgpu_device device,
  cgpu_buffer buffer
);

CGPU_API bool CGPU_CDECL cgpu_create_image(
  cgpu_device device,
  const cgpu_image_description* image_desc,
  cgpu_image* p_image
);

CGPU_API bool CGPU_CDECL cgpu_destroy_image(
  cgpu_device device,
  cgpu_image image
);

CGPU_API bool CGPU_CDECL cgpu_map_image(
  cgpu_device device,
  cgpu_image image,
  void** pp_mapped_mem
);

CGPU_API bool CGPU_CDECL cgpu_unmap_image(
  cgpu_device device,
  cgpu_image image
);

CGPU_API bool CGPU_CDECL cgpu_create_sampler(
  cgpu_device device,
  CgpuSamplerAddressMode address_mode_u,
  CgpuSamplerAddressMode address_mode_v,
  CgpuSamplerAddressMode address_mode_w,
  cgpu_sampler* p_sampler
);

CGPU_API bool CGPU_CDECL cgpu_destroy_sampler(
  cgpu_device device,
  cgpu_sampler sampler
);

CGPU_API bool CGPU_CDECL cgpu_create_pipeline(
  cgpu_device device,
  cgpu_shader shader,
  cgpu_pipeline* p_pipeline
);

CGPU_API bool CGPU_CDECL cgpu_destroy_pipeline(
  cgpu_device device,
  cgpu_pipeline pipeline
);

CGPU_API bool CGPU_CDECL cgpu_create_command_buffer(
  cgpu_device device,
  cgpu_command_buffer* p_command_buffer
);

CGPU_API bool CGPU_CDECL cgpu_begin_command_buffer(
  cgpu_command_buffer command_buffer
);

CGPU_API bool CGPU_CDECL cgpu_cmd_bind_pipeline(
  cgpu_command_buffer command_buffer,
  cgpu_pipeline pipeline
);

CGPU_API bool CGPU_CDECL cgpu_cmd_update_bindings(
  cgpu_command_buffer command_buffer,
  cgpu_pipeline pipeline,
  const cgpu_bindings* bindings
);

CGPU_API bool CGPU_CDECL cgpu_cmd_copy_buffer(
  cgpu_command_buffer command_buffer,
  cgpu_buffer source,
  uint64_t source_offset,
  cgpu_buffer destination,
  uint64_t destination_offset,
  uint64_t size
);

CGPU_API bool CGPU_CDECL cgpu_cmd_copy_buffer_to_image(
  cgpu_command_buffer command_buffer,
  cgpu_buffer buffer,
  uint64_t buffer_offset,
  cgpu_image image
);

CGPU_API bool CGPU_CDECL cgpu_cmd_push_constants(
  cgpu_command_buffer command_buffer,
  cgpu_pipeline pipeline,
  const void* p_data
);

CGPU_API bool CGPU_CDECL cgpu_cmd_dispatch(
  cgpu_command_buffer command_buffer,
  uint32_t dim_x,
  uint32_t dim_y,
  uint32_t dim_z
);

CGPU_API bool CGPU_CDECL cgpu_cmd_pipeline_barrier(
  cgpu_command_buffer command_buffer,
  uint32_t barrier_count,
  const cgpu_memory_barrier* p_barriers,
  uint32_t buffer_barrier_count,
  const cgpu_buffer_memory_barrier* p_buffer_barriers,
  uint32_t image_barrier_count,
  const cgpu_image_memory_barrier* p_image_barriers
);

CGPU_API bool CGPU_CDECL cgpu_cmd_reset_timestamps(
  cgpu_command_buffer command_buffer,
  uint32_t offset,
  uint32_t count
);

CGPU_API bool CGPU_CDECL cgpu_cmd_write_timestamp(
  cgpu_command_buffer command_buffer,
  uint32_t timestamp_index
);

CGPU_API bool CGPU_CDECL cgpu_cmd_copy_timestamps(
  cgpu_command_buffer command_buffer,
  cgpu_buffer buffer,
  uint32_t offset,
  uint32_t count,
  bool wait_until_available
);

CGPU_API bool CGPU_CDECL cgpu_end_command_buffer(
  cgpu_command_buffer command_buffer
);

CGPU_API bool CGPU_CDECL cgpu_destroy_command_buffer(
  cgpu_device device,
  cgpu_command_buffer command_buffer
);

CGPU_API bool CGPU_CDECL cgpu_create_fence(
  cgpu_device device,
  cgpu_fence* p_fence
);

CGPU_API bool CGPU_CDECL cgpu_reset_fence(
  cgpu_device device,
  cgpu_fence fence
);

CGPU_API bool CGPU_CDECL cgpu_wait_for_fence(
  cgpu_device device,
  cgpu_fence fence
);

CGPU_API bool CGPU_CDECL cgpu_destroy_fence(
  cgpu_device device,
  cgpu_fence fence
);

CGPU_API bool CGPU_CDECL cgpu_submit_command_buffer(
  cgpu_device device,
  cgpu_command_buffer command_buffer,
  cgpu_fence fence
);

CGPU_API bool CGPU_CDECL cgpu_flush_mapped_memory(
  cgpu_device device,
  cgpu_buffer buffer,
  uint64_t offset,
  uint64_t size
);

CGPU_API bool CGPU_CDECL cgpu_invalidate_mapped_memory(
  cgpu_device device,
  cgpu_buffer buffer,
  uint64_t offset,
  uint64_t size
);

CGPU_API bool CGPU_CDECL cgpu_get_physical_device_features(
  cgpu_device device,
  cgpu_physical_device_features* p_features
);

CGPU_API bool CGPU_CDECL cgpu_get_physical_device_limits(
  cgpu_device device,
  cgpu_physical_device_limits* p_limits
);

#ifdef __cplusplus
}
#endif

#endif
