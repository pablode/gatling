//
// Copyright (C) 2025 Pablo Delgado Krämer
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

#include <gtl/gb/Fmt.h>
#include <gtl/gb/Log.h>
#include <gtl/gb/LinearDataStore.h>
#include <gtl/gb/SmallVector.h>

#include <spirv_cross_c.h>

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Metal.hpp>

namespace gtl
{
  /* Constants. */

  // TODO
  // constexpr static const uint32_t CGPU_SOMETHING = 123;

  /* Internal structures. */

  struct CgpuIDevice
  {
    MTL::Device* device;
  };

  struct CgpuIBuffer
  {
    MTL::Buffer* buffer;
    uint64_t size;
  };

  struct CgpuIImage
  {
    MTL::Texture* texture;
    uint32_t width;
    uint32_t height;
    uint32_t depth;
  };

  struct CgpuIPipeline
  {
    // TODO
  };

  struct CgpuIShader
  {
    MTL::Library* library;
    CgpuShaderReflection reflection;
  };

  struct CgpuISemaphore
  {
    // TODO
  };

  struct CgpuICommandBuffer
  {
    // TODO
    CgpuDevice device;
  };

  struct CgpuIBlas
  {
    // TODO
  };

  struct CgpuITlas
  {
    // TODO
  };

  struct CgpuISampler
  {
    MTL::SamplerState* sampler;
  };

  struct CgpuIInstance
  {
    spvc_context spvcContext;
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

#define CGPU_FATAL(msg)                             \
  do {                                              \
    GB_ERROR("{}:{}: {}", __FILE__, __LINE__, msg); \
    exit(EXIT_FAILURE);                             \
  } while (false)

#define CGPU_RETURN_ERROR_INVALID_HANDLE                              \
  CGPU_RETURN_ERROR("invalid handle")

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

#define CGPU_RESOLVE_OR_EXIT(HANDLE, VAR_NAME, ITYPE, RESOLVE_FUNC)      \
  ITYPE* VAR_NAME;                                                       \
  if (!RESOLVE_FUNC(HANDLE, &VAR_NAME)) [[unlikely]] {                   \
    CGPU_FATAL("invalid handle!");                                       \
  }

#define CGPU_RESOLVE_DEVICE(HANDLE, VAR_NAME)         CGPU_RESOLVE_OR_EXIT(HANDLE, VAR_NAME, CgpuIDevice, cgpuResolveDevice)
#define CGPU_RESOLVE_BUFFER(HANDLE, VAR_NAME)         CGPU_RESOLVE_OR_EXIT(HANDLE, VAR_NAME, CgpuIBuffer, cgpuResolveBuffer)
#define CGPU_RESOLVE_IMAGE(HANDLE, VAR_NAME)          CGPU_RESOLVE_OR_EXIT(HANDLE, VAR_NAME, CgpuIImage, cgpuResolveImage)
#define CGPU_RESOLVE_SHADER(HANDLE, VAR_NAME)         CGPU_RESOLVE_OR_EXIT(HANDLE, VAR_NAME, CgpuIShader, cgpuResolveShader)
#define CGPU_RESOLVE_PIPELINE(HANDLE, VAR_NAME)       CGPU_RESOLVE_OR_EXIT(HANDLE, VAR_NAME, CgpuIPipeline, cgpuResolvePipeline)
#define CGPU_RESOLVE_SEMAPHORE(HANDLE, VAR_NAME)      CGPU_RESOLVE_OR_EXIT(HANDLE, VAR_NAME, CgpuISemaphore, cgpuResolveSemaphore)
#define CGPU_RESOLVE_COMMAND_BUFFER(HANDLE, VAR_NAME) CGPU_RESOLVE_OR_EXIT(HANDLE, VAR_NAME, CgpuICommandBuffer, cgpuResolveCommandBuffer)
#define CGPU_RESOLVE_SAMPLER(HANDLE, VAR_NAME)        CGPU_RESOLVE_OR_EXIT(HANDLE, VAR_NAME, CgpuISampler, cgpuResolveSampler)
#define CGPU_RESOLVE_BLAS(HANDLE, VAR_NAME)           CGPU_RESOLVE_OR_EXIT(HANDLE, VAR_NAME, CgpuIBlas, cgpuResolveBlas)
#define CGPU_RESOLVE_TLAS(HANDLE, VAR_NAME)           CGPU_RESOLVE_OR_EXIT(HANDLE, VAR_NAME, CgpuITlas, cgpuResolveTlas)

#define CHK_MTL(X, E)    \
  if (!X) {              \
    GB_ERROR("{}:{}: {} (code {})", __FILE__, __LINE__, error->localizedDescription()->utf8String(), error->code()); \
    exit(EXIT_FAILURE);  \
  }

#define CHK_MTL_NP(X)    \
  if (!X) {              \
    GB_ERROR("{}:{}: metal returned nullptr", __FILE__, __LINE__); \
    exit(EXIT_FAILURE);  \
  }

  /* Helper methods. */

  // TODO
  /*
  static CgpuPhysicalDeviceFeatures cgpuTranslatePhysicalDeviceFeatures(const ...& inFeatures)
  {
    CgpuPhysicalDeviceFeatures features = {};
    // ...
    return features;
  }

  static CgpuPhysicalDeviceProperties cgpuTranslatePhysicalDeviceProperties(const ...& inLimits)
  {
    CgpuPhysicalDeviceProperties properties = {};
    // ...
    return properties;
  }
  */

  template<typename T>
  static T cgpuPadToAlignment(T value, T alignment)
  {
      return (value + (alignment - 1)) & ~(alignment - 1);
  }

  /* API method implementation. */

  bool cgpuInitialize(const char* appName, uint32_t versionMajor, uint32_t versionMinor, uint32_t versionPatch)
  {
    iinstance = std::make_unique<CgpuIInstance>();

    if (spvc_result r = spvc_context_create(&iinstance->spvcContext); r != SPVC_SUCCESS)
    {
      CGPU_FATAL("failed to init SPIRV-Cross");
    }

    spvc_context_set_error_callback(iinstance->spvcContext, [](void *userdata, const char *error) {
      GB_ERROR("[SPVC] {}", error);
    }, nullptr);

    return true;
  }

  void cgpuTerminate()
  {
    spvc_context_destroy(iinstance->spvcContext);

    iinstance.reset();
  }

  bool cgpuCreateDevice(CgpuDevice* device)
  {
    uint64_t handle = iinstance->ideviceStore.allocate();

    CGPU_RESOLVE_DEVICE({ handle }, idevice);

    // TODO: select best device suitable
    MTL::Device* mtlDevice = MTL::CreateSystemDefaultDevice();

    if (!mtlDevice->supportsFamily(MTL::GPUFamilyApple9))
    {
      CGPU_FATAL("feature set not supported");
    }
    if (!mtlDevice->supportsRaytracing())
    {
      CGPU_FATAL("ray tracing not supported");
    }

    // TODO: check device capabilities

    idevice->device = mtlDevice;
    return false;
  }

  bool cgpuDestroyDevice(CgpuDevice device)
  {
    CGPU_RESOLVE_DEVICE(device, idevice);

    idevice->device->release();

    iinstance->ideviceStore.free(device.handle);
    return true;
  }

  static bool cgpuCreateShader(CgpuIDevice* idevice,
                               const CgpuShaderCreateInfo& createInfo,
                               CgpuIShader* ishader)
  {
    if (!cgpuReflectShader((uint32_t*) createInfo.source, createInfo.size, &ishader->reflection))
    {
      CGPU_FATAL("failed to reflect shader");
    }

#define CHK_SPVC(X) \
  if (spvc_result r = X; r != SPVC_SUCCESS) {                            \
    GB_ERROR("{}:{}: SPIRV-Cross error {}", __FILE__, __LINE__, int(r)); \
    exit(EXIT_FAILURE);                                                  \
  }

    spvc_compiler spvcCompiler;
    {
      spvc_parsed_ir ir;
      CHK_SPVC(spvc_context_parse_spirv(iinstance->spvcContext, (const SpvId*) createInfo.source, createInfo.size, &ir));
      CHK_SPVC(spvc_context_create_compiler(iinstance->spvcContext, SPVC_BACKEND_MSL, ir, SPVC_CAPTURE_MODE_TAKE_OWNERSHIP, &spvcCompiler));

      spvc_compiler_options spvcCompilerOptions;
      CHK_SPVC(spvc_compiler_create_compiler_options(spvcCompiler, &spvcCompilerOptions));
      CHK_SPVC(spvc_compiler_options_set_uint(spvcCompilerOptions, SPVC_COMPILER_OPTION_MSL_PLATFORM, SPVC_MSL_PLATFORM_MACOS));
      CHK_SPVC(spvc_compiler_install_compiler_options(spvcCompiler, spvcCompilerOptions));
    }

    const char* mslSrc; // owned by context
    CHK_SPVC(spvc_compiler_compile(spvcCompiler, &mslSrc));

#undef CHK_SPVC

    MTL::CompileOptions* compileOptions = MTL::CompileOptions::alloc();
#ifndef NDEBUG
    compileOptions->setEnableLogging(true);
#endif

    NS::Error* error;
    NS::String* mslStr = NS::String::string(mslSrc, NS::UTF8StringEncoding);
    MTL::Library* library = idevice->device->newLibrary(mslStr, compileOptions, &error);
    CHK_MTL(library, error);

    compileOptions->release();

    ishader->library = library;
    return true; // TODO: for shader hotloading, errors shouldn't be fatal
  }

  bool cgpuCreateShader(CgpuDevice device,
                        CgpuShaderCreateInfo createInfo,
                        CgpuShader* shader)
  {
    CGPU_RESOLVE_DEVICE(device, idevice);

    shader->handle = iinstance->ishaderStore.allocate();

    CGPU_RESOLVE_SHADER(*shader, ishader);

    cgpuCreateShader(idevice, createInfo, ishader);

    return true;
  }

  bool cgpuCreateShaders(CgpuDevice device,
                         uint32_t shaderCount,
                         CgpuShaderCreateInfo* createInfos,
                         CgpuShader* shaders)
  {
    CGPU_RESOLVE_DEVICE(device, idevice);

    for (uint32_t i = 0; i < shaderCount; i++)
    {
      shaders[i].handle = iinstance->ishaderStore.allocate();
    }

    std::vector<CgpuIShader*> ishaders;
    ishaders.resize(shaderCount, nullptr);

    for (uint32_t i = 0; i < shaderCount; i++)
    {
      CGPU_RESOLVE_SHADER(shaders[i], ishader);
      ishaders[i] = ishader;
    }

#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < int(shaderCount); i++)
    {
      cgpuCreateShader(idevice, createInfos[i], ishaders[i]);
    }

    return true;
  }

  bool cgpuDestroyShader(CgpuDevice device, CgpuShader shader)
  {
    CGPU_RESOLVE_DEVICE(device, idevice);
    CGPU_RESOLVE_SHADER(shader, ishader);

    ishader->library->release();

    iinstance->ishaderStore.free(shader.handle);

    return true;
  }

  static MTL::ResourceOptions cgpuMakeResourceOptions(CgpuMemoryPropertyFlags memoryProperties)
  {
    if (memoryProperties == CGPU_MEMORY_PROPERTY_FLAG_DEVICE_LOCAL)
    {
      return MTL::ResourceStorageModePrivate;
    }

    return MTL::ResourceStorageModeShared;
  }

  bool cgpuCreateBuffer(CgpuDevice device,
                        CgpuBufferCreateInfo createInfo,
                        CgpuBuffer* buffer)
  {
    CGPU_RESOLVE_DEVICE(device, idevice);

    uint64_t handle = iinstance->ibufferStore.allocate();

    CGPU_RESOLVE_BUFFER({ handle }, ibuffer);

    constexpr static uint64_t BASE_ALIGNMENT = 4;
    uint64_t size = cgpuPadToAlignment(createInfo.size, BASE_ALIGNMENT); // for performance
    assert(size > 0);

    MTL::ResourceOptions options = cgpuMakeResourceOptions(createInfo.memoryProperties);

    MTL::Buffer* mtlBuffer = idevice->device->newBuffer(nullptr, NS::UInteger(size), options);
    if (!mtlBuffer)
    {
      iinstance->ibufferStore.free(handle);
      CGPU_RETURN_ERROR("failed to create buffer");
    }

    if (createInfo.debugName)
    {
      mtlBuffer->setLabel(NS::String::string(createInfo.debugName, NS::StringEncoding::UTF8StringEncoding));
    }

    ibuffer->size = size;
    ibuffer->buffer = mtlBuffer;

    buffer->handle = handle;
    return true;
  }

  bool cgpuDestroyBuffer(CgpuDevice device, CgpuBuffer buffer)
  {
    CGPU_RESOLVE_DEVICE(device, idevice);
    CGPU_RESOLVE_BUFFER(buffer, ibuffer);

    ibuffer->buffer->release();

    iinstance->ibufferStore.free(buffer.handle);
    return true;
  }

  bool cgpuMapBuffer(CgpuDevice device, CgpuBuffer buffer, void** mappedMem)
  {
    CGPU_RESOLVE_DEVICE(device, idevice);
    CGPU_RESOLVE_BUFFER(buffer, ibuffer);

    *mappedMem = ibuffer->buffer->contents();
    return true;
  }

  bool cgpuUnmapBuffer(CgpuDevice device, CgpuBuffer buffer)
  {
    CGPU_RESOLVE_DEVICE(device, idevice);
    CGPU_RESOLVE_BUFFER(buffer, ibuffer);

    // Nothing to do.
    return true;
  }

  uint64_t cgpuGetBufferAddress(CgpuDevice device, CgpuBuffer buffer)
  {
    CGPU_RESOLVE_BUFFER(buffer, ibuffer);

    return ibuffer->buffer->gpuAddress();
  }

  static MTL::TextureUsage cgpuTranslateImageUsage(CgpuImageUsageFlags usage)
  {
    MTL::TextureUsage mtlUsage = MTL::TextureUsageUnknown;

    if (bool(usage & CGPU_IMAGE_USAGE_FLAG_SAMPLED))
    {
      mtlUsage |= MTL::TextureUsageShaderRead;
    }
    if (bool(usage & CGPU_IMAGE_USAGE_FLAG_STORAGE))
    {
      // TODO: increase granularity of cgpu flags
      mtlUsage |= MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite;
    }

    return mtlUsage;
  }

  static MTL::PixelFormat cgpuTranslateImageFormat(CgpuImageFormat format)
  {
    // TODO: cover missing formats using LUT
    if (format == CGPU_IMAGE_FORMAT_R32_SFLOAT)
    {
      return MTL::PixelFormatR32Float;
    }
    return MTL::PixelFormatRGBA8Unorm;
  }

  bool cgpuCreateImage(CgpuDevice device,
                       CgpuImageCreateInfo createInfo,
                       CgpuImage* image)
  {
    CGPU_RESOLVE_DEVICE(device, idevice);

    uint64_t handle = iinstance->iimageStore.allocate();

    CGPU_RESOLVE_IMAGE({ handle }, iimage);

    MTL::TextureUsage usage = cgpuTranslateImageUsage(createInfo.usage);
    MTL::PixelFormat pixelFormat = cgpuTranslateImageFormat(createInfo.format);
    bool mipmapped = false;

    MTL::TextureDescriptor* descriptor = MTL::TextureDescriptor::alloc()->init();
    CHK_MTL_NP(descriptor);

    descriptor->setTextureType(createInfo.is3d ? MTL::TextureType3D : MTL::TextureType2D);
    descriptor->setPixelFormat(pixelFormat);
    descriptor->setWidth(createInfo.width);
    descriptor->setHeight(createInfo.height);
    descriptor->setDepth(createInfo.depth);
    descriptor->setUsage(usage);
    descriptor->setStorageMode(MTL::StorageModePrivate);

    MTL::Texture* texture = idevice->device->newTexture(descriptor);

    descriptor->release();

    if (!texture)
    {
      iinstance->iimageStore.free(handle);
      CGPU_RETURN_ERROR("failed to create image");
    }

    if (createInfo.debugName)
    {
      texture->setLabel(NS::String::string(createInfo.debugName, NS::StringEncoding::UTF8StringEncoding));
    }

    iimage->texture = texture;
    iimage->width = createInfo.width;
    iimage->height = createInfo.height;
    iimage->depth = createInfo.is3d ? createInfo.depth : 1;

    image->handle = handle;
    return true;
  }

  bool cgpuDestroyImage(CgpuDevice device, CgpuImage image)
  {
    CGPU_RESOLVE_DEVICE(device, idevice);
    CGPU_RESOLVE_IMAGE(image, iimage);

    iimage->texture->release();

    iinstance->iimageStore.free(image.handle);

    return false;
  }

  bool cgpuCreateSampler(CgpuDevice device,
                         CgpuSamplerCreateInfo createInfo,
                         CgpuSampler* sampler)
  {
    CGPU_RESOLVE_DEVICE(device, idevice);

    uint64_t handle = iinstance->isamplerStore.allocate();

    CGPU_RESOLVE_SAMPLER({ handle }, isampler);

    const auto translateAddressMode = [](CgpuSamplerAddressMode m)
    {
      switch (m)
      {
      case CGPU_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE:
        return MTL::SamplerAddressModeClampToEdge;
      case CGPU_SAMPLER_ADDRESS_MODE_REPEAT:
        return MTL::SamplerAddressModeRepeat;
      case CGPU_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT:
        return MTL::SamplerAddressModeMirrorRepeat;
      case CGPU_SAMPLER_ADDRESS_MODE_CLAMP_TO_BLACK:
        return MTL::SamplerAddressModeClampToBorderColor;
      default:
        CGPU_FATAL("sampler address mode not handled");
      }
    };

    MTL::SamplerDescriptor* descriptor = MTL::SamplerDescriptor::alloc()->init();

    descriptor->setSAddressMode(translateAddressMode(createInfo.addressModeU));
    descriptor->setTAddressMode(translateAddressMode(createInfo.addressModeV));
    descriptor->setRAddressMode(translateAddressMode(createInfo.addressModeW));
    descriptor->setMinFilter(MTL::SamplerMinMagFilterLinear);
    descriptor->setMagFilter(MTL::SamplerMinMagFilterLinear);
    descriptor->setNormalizedCoordinates(false);
    descriptor->setBorderColor(MTL::SamplerBorderColorOpaqueBlack);

    MTL::SamplerState* mtlSampler = idevice->device->newSamplerState(descriptor);

    descriptor->release();

    if (!mtlSampler)
    {
      iinstance->isamplerStore.free(handle);
      CGPU_RETURN_ERROR("failed to create sampler");
    }

    isampler->sampler = mtlSampler;

    sampler->handle = handle;
    return true;
  }

  bool cgpuDestroySampler(CgpuDevice device, CgpuSampler sampler)
  {
    CGPU_RESOLVE_DEVICE(device, idevice);
    CGPU_RESOLVE_SAMPLER(sampler, isampler);

    isampler->sampler->release();

    iinstance->isamplerStore.free(sampler.handle);
    return false;
  }

  bool cgpuCreateComputePipeline(CgpuDevice device,
                                 CgpuComputePipelineCreateInfo createInfo,
                                 CgpuPipeline* pipeline)
  {
    CGPU_RESOLVE_DEVICE(device, idevice);
    CGPU_RESOLVE_SHADER(createInfo.shader, ishader);

    uint64_t handle = iinstance->ipipelineStore.allocate();

    CGPU_RESOLVE_PIPELINE({ handle }, ipipeline);

    // TODO

    pipeline->handle = handle;
    return false;
  }

  bool cgpuCreateRtPipeline(CgpuDevice device,
                            CgpuRtPipelineCreateInfo createInfo,
                            CgpuPipeline* pipeline)
  {
    CGPU_RESOLVE_DEVICE(device, idevice);

    uint64_t handle = iinstance->ipipelineStore.allocate();

    CGPU_RESOLVE_PIPELINE({ handle }, ipipeline);

    // TODO
    return false;
  }

  bool cgpuDestroyPipeline(CgpuDevice device, CgpuPipeline pipeline)
  {
    CGPU_RESOLVE_DEVICE(device, idevice);
    CGPU_RESOLVE_PIPELINE(pipeline, ipipeline);

    // TODO

    iinstance->ipipelineStore.free(pipeline.handle);

    return false;
  }

  bool cgpuCreateBlas(CgpuDevice device,
                      CgpuBlasCreateInfo createInfo,
                      CgpuBlas* blas)
  {
    CGPU_RESOLVE_DEVICE(device, idevice);
    CGPU_RESOLVE_BUFFER(createInfo.vertexBuffer, ivertexBuffer);
    CGPU_RESOLVE_BUFFER(createInfo.indexBuffer, iindexBuffer);

    uint64_t handle = iinstance->iblasStore.allocate();

    CGPU_RESOLVE_BLAS({ handle }, iblas);

    // TODO

    blas->handle = handle;
    return false;
  }

  bool cgpuCreateTlas(CgpuDevice device,
                      CgpuTlasCreateInfo createInfo,
                      CgpuTlas* tlas)
  {
    CGPU_RESOLVE_DEVICE(device, idevice);

    uint64_t handle = iinstance->itlasStore.allocate();

    CGPU_RESOLVE_TLAS({ handle }, itlas);

    // TODO

    tlas->handle = handle;
    return false;
  }

  bool cgpuDestroyBlas(CgpuDevice device, CgpuBlas blas)
  {
    CGPU_RESOLVE_DEVICE(device, idevice);
    CGPU_RESOLVE_BLAS(blas, iblas);

    // TODO

    iinstance->iblasStore.free(blas.handle);
    return true;
  }

  bool cgpuDestroyTlas(CgpuDevice device, CgpuTlas tlas)
  {
    CGPU_RESOLVE_DEVICE(device, idevice);
    CGPU_RESOLVE_TLAS(tlas, itlas);

    // TODO

    iinstance->itlasStore.free(tlas.handle);
    return true;
  }

  bool cgpuCreateCommandBuffer(CgpuDevice device, CgpuCommandBuffer* commandBuffer)
  {
    CGPU_RESOLVE_DEVICE(device, idevice);

    uint64_t handle = iinstance->icommandBufferStore.allocate();

    CGPU_RESOLVE_COMMAND_BUFFER({ handle }, icommandBuffer);

    // TODO

    commandBuffer->handle = handle;
    return false;
  }

  bool cgpuDestroyCommandBuffer(CgpuDevice device, CgpuCommandBuffer commandBuffer)
  {
    CGPU_RESOLVE_DEVICE(device, idevice);
    CGPU_RESOLVE_COMMAND_BUFFER(commandBuffer, icommandBuffer);

    // TODO

    iinstance->icommandBufferStore.free(commandBuffer.handle);
    return true;
  }

  bool cgpuBeginCommandBuffer(CgpuCommandBuffer commandBuffer)
  {
    CGPU_RESOLVE_COMMAND_BUFFER(commandBuffer, icommandBuffer);
    CGPU_RESOLVE_DEVICE(icommandBuffer->device, idevice);

    // TODO: is this a no-op?
    return true;
  }

  void cgpuCmdBindPipeline(CgpuCommandBuffer commandBuffer, CgpuPipeline pipeline)
  {
    CGPU_RESOLVE_COMMAND_BUFFER(commandBuffer, icommandBuffer);
    CGPU_RESOLVE_DEVICE(icommandBuffer->device, idevice);
    CGPU_RESOLVE_PIPELINE(pipeline, ipipeline);

    // TODO
  }

  void cgpuCmdTransitionShaderImageLayouts(CgpuCommandBuffer commandBuffer,
                                           CgpuShader shader,
                                           uint32_t descriptorSetIndex,
                                           uint32_t imageCount,
                                           const CgpuImageBinding* images)
  {
    // TODO
  }

  void cgpuCmdUpdateBindings(CgpuCommandBuffer commandBuffer,
                             CgpuPipeline pipeline,
                             uint32_t descriptorSetIndex,
                             const CgpuBindings* bindings)
  {
    CGPU_RESOLVE_COMMAND_BUFFER(commandBuffer, icommandBuffer);
    CGPU_RESOLVE_DEVICE(icommandBuffer->device, idevice);
    CGPU_RESOLVE_PIPELINE(pipeline, ipipeline);

    // TODO
  }

  void cgpuCmdUpdateBuffer(CgpuCommandBuffer commandBuffer,
                           const uint8_t* data,
                           uint64_t size,
                           CgpuBuffer dstBuffer,
                           uint64_t dstOffset)
  {
    CGPU_RESOLVE_COMMAND_BUFFER(commandBuffer, icommandBuffer);
    CGPU_RESOLVE_DEVICE(icommandBuffer->device, idevice);
    CGPU_RESOLVE_BUFFER(dstBuffer, idstBuffer);

    // TODO
  }

  void cgpuCmdCopyBuffer(CgpuCommandBuffer commandBuffer,
                         CgpuBuffer srcBuffer,
                         uint64_t srcOffset,
                         CgpuBuffer dstBuffer,
                         uint64_t dstOffset,
                         uint64_t size)
  {
    CGPU_RESOLVE_COMMAND_BUFFER(commandBuffer, icommandBuffer);
    CGPU_RESOLVE_DEVICE(icommandBuffer->device, idevice);
    CGPU_RESOLVE_BUFFER(srcBuffer, isrcBuffer);
    CGPU_RESOLVE_BUFFER(dstBuffer, idstBuffer);

    // TODO
  }

  void cgpuCmdCopyBufferToImage(CgpuCommandBuffer commandBuffer,
                                CgpuBuffer buffer,
                                CgpuImage image,
                                const CgpuBufferImageCopyDesc* desc)
  {
    CGPU_RESOLVE_COMMAND_BUFFER(commandBuffer, icommandBuffer);
    CGPU_RESOLVE_DEVICE(icommandBuffer->device, idevice);
    CGPU_RESOLVE_BUFFER(buffer, ibuffer);
    CGPU_RESOLVE_IMAGE(image, iimage);

    // TODO
  }

  void cgpuCmdPushConstants(CgpuCommandBuffer commandBuffer,
                            CgpuPipeline pipeline,
                            CgpuShaderStageFlags stageFlags,
                            uint32_t size,
                            const void* data)
  {
    CGPU_RESOLVE_COMMAND_BUFFER(commandBuffer, icommandBuffer);
    CGPU_RESOLVE_DEVICE(icommandBuffer->device, idevice);
    CGPU_RESOLVE_PIPELINE(pipeline, ipipeline);

    // TODO
  }

  void cgpuCmdDispatch(CgpuCommandBuffer commandBuffer,
                       uint32_t dim_x,
                       uint32_t dim_y,
                       uint32_t dim_z)
  {
    CGPU_RESOLVE_COMMAND_BUFFER(commandBuffer, icommandBuffer);
    CGPU_RESOLVE_DEVICE(icommandBuffer->device, idevice);

    // TODO
  }

  void cgpuCmdPipelineBarrier(CgpuCommandBuffer commandBuffer,
                              const CgpuPipelineBarrier* barrier)
  {
    CGPU_RESOLVE_COMMAND_BUFFER(commandBuffer, icommandBuffer);
    CGPU_RESOLVE_DEVICE(icommandBuffer->device, idevice);

    // TODO: no-op?
  }

  void cgpuCmdResetTimestamps(CgpuCommandBuffer commandBuffer,
                              uint32_t offset,
                              uint32_t count)
  {
    CGPU_RESOLVE_COMMAND_BUFFER(commandBuffer, icommandBuffer);
    CGPU_RESOLVE_DEVICE(icommandBuffer->device, idevice);

    // TODO
  }

  void cgpuCmdWriteTimestamp(CgpuCommandBuffer commandBuffer,
                             uint32_t timestampIndex)
  {
    CGPU_RESOLVE_COMMAND_BUFFER(commandBuffer, icommandBuffer);
    CGPU_RESOLVE_DEVICE(icommandBuffer->device, idevice);

    // TODO
  }

  void cgpuCmdCopyTimestamps(CgpuCommandBuffer commandBuffer,
                             CgpuBuffer buffer,
                             uint32_t offset,
                             uint32_t count,
                             bool waitUntilAvailable)
  {
    CGPU_RESOLVE_COMMAND_BUFFER(commandBuffer, icommandBuffer);
    CGPU_RESOLVE_DEVICE(icommandBuffer->device, idevice);
    CGPU_RESOLVE_BUFFER(buffer, ibuffer);

    uint32_t lastIndex = offset + count;
    if (lastIndex >= CGPU_MAX_TIMESTAMP_QUERIES) {
      CGPU_FATAL("max timestamp query count exceeded!");
    }

    // TODO
  }

  void cgpuCmdTraceRays(CgpuCommandBuffer commandBuffer, CgpuPipeline rtPipeline, uint32_t width, uint32_t height)
  {
    CGPU_RESOLVE_COMMAND_BUFFER(commandBuffer, icommandBuffer);
    CGPU_RESOLVE_DEVICE(icommandBuffer->device, idevice);
    CGPU_RESOLVE_PIPELINE(rtPipeline, ipipeline);

    // TODO
  }

  void cgpuEndCommandBuffer(CgpuCommandBuffer commandBuffer)
  {
    CGPU_RESOLVE_COMMAND_BUFFER(commandBuffer, icommandBuffer);
    CGPU_RESOLVE_DEVICE(icommandBuffer->device, idevice);

    // TODO: no-op?
  }

  bool cgpuCreateSemaphore(CgpuDevice device, CgpuSemaphore* semaphore, uint64_t initialValue)
  {
    CGPU_RESOLVE_DEVICE(device, idevice);

    uint64_t handle = iinstance->isemaphoreStore.allocate();

    CGPU_RESOLVE_SEMAPHORE({ handle }, isemaphore);

    // TODO

    semaphore->handle = handle;
    return true;
  }

  bool cgpuDestroySemaphore(CgpuDevice device, CgpuSemaphore semaphore)
  {
    CGPU_RESOLVE_DEVICE(device, idevice);
    CGPU_RESOLVE_SEMAPHORE(semaphore, isemaphore);

    // TODO

    iinstance->isemaphoreStore.free(semaphore.handle);
    return true;
  }

  bool cgpuWaitSemaphores(CgpuDevice device,
                          uint32_t semaphoreInfoCount,
                          CgpuWaitSemaphoreInfo* semaphoreInfos,
                          uint64_t timeoutNs)
  {
    CGPU_RESOLVE_DEVICE(device, idevice);

    // TODO
    return true;
  }

  bool cgpuSubmitCommandBuffer(CgpuDevice device,
                               CgpuCommandBuffer commandBuffer,
                               uint32_t signalSemaphoreInfoCount,
                               CgpuSignalSemaphoreInfo* signalSemaphoreInfos,
                               uint32_t waitSemaphoreInfoCount,
                               CgpuWaitSemaphoreInfo* waitSemaphoreInfos)
  {
    CGPU_RESOLVE_DEVICE(device, idevice);
    CGPU_RESOLVE_COMMAND_BUFFER(commandBuffer, icommandBuffer);

    // TODO
    return false;
  }

  bool cgpuFlushMappedMemory(CgpuDevice device,
                             CgpuBuffer buffer,
                             uint64_t offset,
                             uint64_t size)
  {
    CGPU_RESOLVE_BUFFER(buffer, ibuffer);

    NS::Range range(offset, size);
    ibuffer->buffer->didModifyRange(range);
    return true;
  }

  bool cgpuInvalidateMappedMemory(CgpuDevice device,
                                  CgpuBuffer buffer,
                                  uint64_t offset,
                                  uint64_t size)
  {
    // No equivalent.
    return true;
  }

  bool cgpuGetPhysicalDeviceFeatures(CgpuDevice device,
                                     CgpuPhysicalDeviceFeatures& features)
  {
    CGPU_RESOLVE_DEVICE(device, idevice);
    return false; // TODO
  }

  bool cgpuGetPhysicalDeviceProperties(CgpuDevice device,
                                       CgpuPhysicalDeviceProperties& properties)
  {
    CGPU_RESOLVE_DEVICE(device, idevice);
    return false; // TODO
  }
}
