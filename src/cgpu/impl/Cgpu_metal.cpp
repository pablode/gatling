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
#include <variant>

#include <gtl/gb/Fmt.h>
#include <gtl/gb/Log.h>
#include <gtl/gb/LinearDataStore.h>
#include <gtl/gb/SmallVector.h>

#include <spirv_cross_c.h>

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
// NOTE: there's a bug in beta2 where MTL4AccelerationStructure.hpp is not included in Metal.hpp
#include <Metal.hpp>

// TODO: MTLSTR macro?

namespace gtl
{
  /* Constants. */

  constexpr static const uint32_t SPVC_MSL_VERSION = SPVC_MAKE_MSL_VERSION(3, 1, 0);

  // for shader reflection
  typedef enum VkDescriptorType {
    VK_DESCRIPTOR_TYPE_SAMPLER = 0,
    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER = 1,
    VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE = 2,
    VK_DESCRIPTOR_TYPE_STORAGE_IMAGE = 3,
    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER = 6,
    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER = 7,
    VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR = 1000150000
  } VkDescriptorType;

  //typedef enum VkImageType {
  //  VK_IMAGE_TYPE_1D = 0,
  //  VK_IMAGE_TYPE_2D = 1,
  //  VK_IMAGE_TYPE_3D = 2,
  //} VkImageType;

  /* Internal structures. */

  struct CgpuIDevice
  {
    MTL4::Compiler*      compiler;
    MTL::Device*         device;
    MTL4::CommandQueue*  commandQueue;
    MTL4::CounterHeap*   counterHeap;
#ifndef NDEBUG
    MTL::LogState*       logState; // nullable
    MTL4::CommitOptions* commitOptions;
#endif
    uint32_t             uniqueShaderEntryPointCounter;
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
    MTL::ComputePipelineState* state;

    MTL4::ArgumentTable* argumentTable;
    std::vector<MTL::ArgumentEncoder*> argumentEncoders; // last entry is for aux descriptor set
    std::vector<MTL::Buffer*> argumentBuffers; // last entry is reference to aux descriptor set buffer
    // NOTE: the aux descriptor set contains at 0: buffer for PCs, 1: intersection function argument buffer
    //       the PC buffer resides in the command buffer, the IFBA here (below).

    // only for RT pipeline
    MTL::IntersectionFunctionTable* intersectionFunctionTable = nullptr;
    MTL::Buffer* intersectionFunctionBuffer = nullptr;
    // TODO: rename to ifba. table to ift, etc.
    MTL::Buffer* intersectionFunctionBufferArgs = nullptr;

    // TODO: in the dev/master branch, this should be a member of BindGroup
    MTL::ResidencySet* residencySet = nullptr;
  };

  struct CgpuIShader
  {
    std::string entryPoint;
    MTL::Library* library;
    CgpuShaderReflection reflection;
  };

  struct CgpuISemaphore
  {
    MTL::Event* event;
  };

  struct CgpuICommandBuffer
  {
    MTL4::CommandBuffer* commandBuffer;
    MTL4::CommandAllocator* commandAllocator;
    MTL4::ComputeCommandEncoder* encoder;

    // NOTE: maybe rethink if this should live here or in IPipeline
    MTL::Buffer* pcBuffer;
    void* pcMem;

    MTL4::CounterHeap* counterHeap; // owned by device
#ifndef NDEBUG
    MTL::LogState* logState; // nullable, owned by device
    MTL4::CommitOptions* commitOptions; // owned by device
#endif
  };

  struct CgpuIBlas
  {
    MTL::AccelerationStructure* as;
    MTL::Buffer* buffer;
  };

  struct CgpuITlas
  {
    MTL::AccelerationStructure* as;
    MTL::Buffer* buffer;
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

#define LOG_MTL_ERR(E) \
  if (E) { GB_ERROR("{}:{}: {} (code {})", __FILE__, __LINE__, E->localizedDescription()->utf8String(), E->code()); }

#define CHK_MTL(X, E)    \
  if (!X) { LOG_MTL_ERR(E); fflush(stdout); assert(false); exit(EXIT_FAILURE); }

// TODO: replace with proper error handling in some cases
#define CHK_MTL_NP(X)    \
  if (!X) {              \
    GB_ERROR("{}:{}: metal returned nullptr", __FILE__, __LINE__); \
    assert(false); \
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

  static MTL::ResourceOptions cgpuTranslateMemoryProperties(CgpuMemoryPropertyFlags memoryProperties)
  {
    if (memoryProperties == CGPU_MEMORY_PROPERTY_FLAG_DEVICE_LOCAL)
    {
      return MTL::ResourceStorageModePrivate;
    }

    return MTL::ResourceStorageModeShared;
  }

  static MTL::Stages cgpuTranslatePipelineStages(CgpuPipelineStageFlags stages)
  {
    MTL::Stages newStages = 0;

    if (bool(stages & CGPU_PIPELINE_STAGE_FLAG_COMPUTE_SHADER) ||
        bool(stages & CGPU_PIPELINE_STAGE_FLAG_RAY_TRACING_SHADER))
    {
      newStages |= MTL::StageDispatch;
    }
    if (bool(stages & CGPU_PIPELINE_STAGE_FLAG_TRANSFER) ||
        bool(stages & CGPU_PIPELINE_STAGE_FLAG_HOST))
    {
      newStages |= MTL::StageBlit;
    }
    if (bool(stages & CGPU_PIPELINE_STAGE_FLAG_ACCELERATION_STRUCTURE_BUILD))
    {
      newStages |= MTL::StageAccelerationStructure;
    }

    return newStages;
  }

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
    CHK_MTL_NP(mtlDevice);

    if (!mtlDevice->supportsFamily(MTL::GPUFamilyApple9))
    {
      CGPU_FATAL("feature set not supported");
    }
    if (!mtlDevice->supportsRaytracing())
    {
      CGPU_FATAL("ray tracing not supported");
    }
    if (mtlDevice->argumentBuffersSupport() != MTL::ArgumentBuffersTier2)
    {
      CGPU_FATAL("tier 2 argument buffers not supported");
    }

    MTL4::CommandQueue* commandQueue = mtlDevice->newMTL4CommandQueue();
    CHK_MTL_NP(commandQueue);

    MTL4::CounterHeap* counterHeap;
    {
      auto* desc = MTL4::CounterHeapDescriptor::alloc()->init();
      desc->setCount(CGPU_MAX_TIMESTAMP_QUERIES);
      desc->setType(MTL4::CounterHeapTypeTimestamp);

      NS::Error* error = nullptr;
      counterHeap = mtlDevice->newCounterHeap(desc, &error);
      CHK_MTL(counterHeap, error);
      desc->release();
    }

#ifndef NDEBUG
    MTL::LogState* logState = nullptr;
    {
      auto* desc = MTL::LogStateDescriptor::alloc()->init();
      //desc->setBufferSize(4 * 1024 * 1024);
      desc->setLevel(MTL::LogLevelDebug);

      NS::Error* error = nullptr;
      logState = idevice->device->newLogState(desc, &error);
      desc->release();

      auto logHandler = [](NS::String* subsystem, NS::String* category, MTL::LogLevel logLevel, NS::String* message) {
        std::string msg;
        if (logLevel == MTL::LogLevelError || logLevel == MTL::LogLevelFault)
        {
          GB_ERROR("[MTL] ({}/{}) {}", subsystem->utf8String(), category->utf8String(), message->utf8String());
        }
        else
        {
          GB_LOG("[MTL] ({}/{}) {}", subsystem->utf8String(), category->utf8String(), message->utf8String());
        }

        if (logLevel == MTL::LogLevelFault)
        {
          exit(EXIT_FAILURE);
        }
      };

      if (logState)
      {
        logState->addLogHandler(logHandler);
      }
      else
      {
        LOG_MTL_ERR(error);
      }
    }

    MTL4::CommitOptions* commitOptions = MTL4::CommitOptions::alloc()->init();
    commitOptions->addFeedbackHandler([](MTL4::CommitFeedback* f) {
      NS::Error* error = f->error();
      LOG_MTL_ERR(error);
    });
#endif

    MTL4::Compiler* compiler;
    {
      auto* descriptor = MTL4::CompilerDescriptor::alloc()->init();

      NS::Error* error = nullptr;
      compiler = mtlDevice->newCompiler(descriptor, &error);
      CHK_MTL(compiler, error);

      descriptor->release();
    }

    idevice->compiler = compiler;
    idevice->device = mtlDevice;
    idevice->commandQueue = commandQueue;
    idevice->counterHeap = counterHeap;
#ifndef NDEBUG
    idevice->logState = logState;
    idevice->commitOptions = commitOptions;
#endif
    idevice->uniqueShaderEntryPointCounter = 0;

    device->handle = handle;
    return true;
  }

  bool cgpuDestroyDevice(CgpuDevice device)
  {
    CGPU_RESOLVE_DEVICE(device, idevice);

    idevice->compiler->release();
    idevice->counterHeap->release();
    idevice->commandQueue->release();
    idevice->device->release();
#ifndef NDEBUG
    idevice->logState->release();
    idevice->commitOptions->release();
#endif

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
    assert(false);                                                       \
    exit(EXIT_FAILURE);                                                  \
  }

    spvc_compiler spvcCompiler; // TODO: rename to compiler
    {
      spvc_parsed_ir ir;
      CHK_SPVC(spvc_context_parse_spirv(iinstance->spvcContext, (const SpvId*) createInfo.source, createInfo.size / sizeof(SpvId), &ir));
      CHK_SPVC(spvc_context_create_compiler(iinstance->spvcContext, SPVC_BACKEND_MSL, ir, SPVC_CAPTURE_MODE_TAKE_OWNERSHIP, &spvcCompiler));

      SpvExecutionModel execModel = spvc_compiler_get_execution_model(spvcCompiler);
      CHK_SPVC(spvc_compiler_rename_entry_point(spvcCompiler, CGPU_SHADER_ENTRY_POINT, ishader->entryPoint.c_str(), execModel));

      spvc_compiler_options spvcCompilerOptions; // TODO: rename to compilerOptions
      CHK_SPVC(spvc_compiler_create_compiler_options(spvcCompiler, &spvcCompilerOptions));
      CHK_SPVC(spvc_compiler_options_set_uint(spvcCompilerOptions, SPVC_COMPILER_OPTION_MSL_PLATFORM, SPVC_MSL_PLATFORM_MACOS));
      CHK_SPVC(spvc_compiler_options_set_uint(spvcCompilerOptions, SPVC_COMPILER_OPTION_MSL_VERSION, SPVC_MSL_VERSION));
      CHK_SPVC(spvc_compiler_install_compiler_options(spvcCompiler, spvcCompilerOptions));
    }

    const char* mslSrc; // owned by context
    CHK_SPVC(spvc_compiler_compile(spvcCompiler, &mslSrc));

// NOTE: enable to print SPIRV-Cross output (MSL code)
//GB_LOG("{}", mslSrc);

#undef CHK_SPVC

    MTL::CompileOptions* compileOptions = MTL::CompileOptions::alloc()->init();
#ifndef NDEBUG
    compileOptions->setEnableLogging(true);
#endif
    // TODO: consider other compileOptions fields

    NS::Error* error = nullptr;
    NS::String* mslStr = NS::String::string(mslSrc, NS::UTF8StringEncoding);
    MTL::Library* library = idevice->device->newLibrary(mslStr, compileOptions, &error);
    CHK_MTL(library, error);

    compileOptions->release();

    ishader->library = library;
    return true; // TODO: for shader hotloading, errors shouldn't be fatal
  }

  static std::string cgpuMakeShaderEntryPoint(CgpuIDevice* idevice, CgpuShaderStageFlags stageFlag)
  {
    return GB_FMT("main{}", idevice->uniqueShaderEntryPointCounter++);
  }

  bool cgpuCreateShader(CgpuDevice device,
                        CgpuShaderCreateInfo createInfo,
                        CgpuShader* shader)
  {
    CGPU_RESOLVE_DEVICE(device, idevice);

    shader->handle = iinstance->ishaderStore.allocate();

    CGPU_RESOLVE_SHADER(*shader, ishader);

    ishader->entryPoint = cgpuMakeShaderEntryPoint(idevice, createInfo.stageFlags);

    return cgpuCreateShader(idevice, createInfo, ishader);
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
      ishader->entryPoint = cgpuMakeShaderEntryPoint(idevice, createInfos[i].stageFlags);
      ishaders[i] = ishader;
    }

    std::atomic<bool> success = true;

#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < int(shaderCount); i++)
    {
      if (!cgpuCreateShader(idevice, createInfos[i], ishaders[i]))
      {
        success = false;
      }
    }

    return success;
  }

  bool cgpuDestroyShader(CgpuDevice device, CgpuShader shader)
  {
    CGPU_RESOLVE_SHADER(shader, ishader);

    ishader->library->release();

    iinstance->ishaderStore.free(shader.handle);
    return true;
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

    MTL::ResourceOptions options = cgpuTranslateMemoryProperties(createInfo.memoryProperties);

    MTL::Buffer* mtlBuffer = idevice->device->newBuffer(size, options);
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

    auto* descriptor = MTL::TextureDescriptor::alloc()->init();
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
    CGPU_RESOLVE_IMAGE(image, iimage);

    iimage->texture->release();

    iinstance->iimageStore.free(image.handle);
    return true;
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

    auto* descriptor = MTL::SamplerDescriptor::alloc()->init();

    descriptor->setSAddressMode(translateAddressMode(createInfo.addressModeU));
    descriptor->setTAddressMode(translateAddressMode(createInfo.addressModeV));
    descriptor->setRAddressMode(translateAddressMode(createInfo.addressModeW));
    descriptor->setMinFilter(MTL::SamplerMinMagFilterLinear);
    descriptor->setMagFilter(MTL::SamplerMinMagFilterLinear);
    descriptor->setNormalizedCoordinates(true);
    descriptor->setBorderColor(MTL::SamplerBorderColorOpaqueBlack);
    descriptor->setSupportArgumentBuffers(true);

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
    CGPU_RESOLVE_SAMPLER(sampler, isampler);

    isampler->sampler->release();

    iinstance->isamplerStore.free(sampler.handle);
    return true;
  }

  static bool cgpuCreateComputePipeline(CgpuIDevice* idevice,
                                        CgpuIShader* ishader,
                                        const char* debugName,
                                        CgpuPipeline* pipeline,
                                        const MTL4::StaticLinkingDescriptor* linkingDescriptor = nullptr)
  {
    uint64_t handle = iinstance->ipipelineStore.allocate();

    CGPU_RESOLVE_PIPELINE({ handle }, ipipeline);

    auto* descriptor = MTL4::ComputePipelineDescriptor::alloc()->init();
    CHK_MTL_NP(descriptor);

    NS::String* entryFuncName = NS::String::string(ishader->entryPoint.c_str(), NS::UTF8StringEncoding);

    auto* entryFunDesc = MTL4::LibraryFunctionDescriptor::alloc()->init();
    entryFunDesc->setLibrary(ishader->library);
    entryFunDesc->setName(entryFuncName);

    descriptor->setComputeFunctionDescriptor(entryFunDesc);

    if (linkingDescriptor)
    {
      // TODO: this hangs infinitely :( maybe because of duplicate functions? but they should be static..
      //descriptor->setStaticLinkingDescriptor(linkingDescriptor);
    }

    MTL4::Compiler* compiler = idevice->compiler;

    GB_LOG("before!");
    NS::Error* error = nullptr;
    MTL::ComputePipelineState* state = compiler->newComputePipelineState(descriptor, nullptr, &error);
    LOG_MTL_ERR(error);
    CHK_MTL_NP(state);
    GB_LOG("after!");

    descriptor->release();
    entryFunDesc->release();

    MTL4::ArgumentTable* argumentTable;
    {
      auto* desc = MTL4::ArgumentTableDescriptor::alloc()->init();
      CHK_MTL_NP(desc);
      desc->setMaxBufferBindCount(CGPU_MAX_DESCRIPTOR_SET_COUNT + 1/* PC emulation */);
      desc->setMaxSamplerStateBindCount(0);
      desc->setMaxTextureBindCount(0);

      NS::Error* error;
      argumentTable = idevice->device->newArgumentTable(desc, &error);
      CHK_MTL(argumentTable, error);
      desc->release();
    }

    const CgpuShaderReflection& reflection = ishader->reflection;
    uint32_t descriptorSetCount = reflection.descriptorSets.size();

    std::vector<MTL::ArgumentEncoder*> argumentEncoders;
    std::vector<MTL::Buffer*> argumentBuffers;
    argumentEncoders.reserve(descriptorSetCount);
    argumentBuffers.reserve(descriptorSetCount);

    auto pushDescriptorSet = [&](uint32_t descriptorSetIndex, std::vector<MTL::ArgumentDescriptor*> descriptors)
    {
      NS::Array* descriptorArray = NS::Array::array((const NS::Object* const*) descriptors.data(), (uint32_t) descriptors.size());
      CHK_MTL_NP(descriptorArray);

      MTL::ArgumentEncoder* argumentEncoder = idevice->device->newArgumentEncoder(descriptorArray);
      CHK_MTL_NP(argumentEncoder);

      uint64_t argumentBufferSize = argumentEncoder->encodedLength();
      MTL::Buffer* argumentBuffer = idevice->device->newBuffer(argumentBufferSize, MTL::ResourceStorageModeShared); // TODO: or Managed

      argumentEncoders.push_back(argumentEncoder);
      argumentBuffers.push_back(argumentBuffer);

      argumentTable->setAddress(argumentBuffer->gpuAddress(), descriptorSetIndex);
    };

    for (uint32_t i = 0; i < descriptorSetCount; i++)
    {
      const CgpuShaderReflectionDescriptorSet& descriptorSet = reflection.descriptorSets[i];
      const std::vector<CgpuShaderReflectionBinding>& bindings = descriptorSet.bindings;

      std::vector<MTL::ArgumentDescriptor*> argumentDescriptors;
      argumentDescriptors.reserve(bindings.size());

      for (const CgpuShaderReflectionBinding& binding : bindings)
      {
        VkDescriptorType descriptorType = (VkDescriptorType) binding.descriptorType;

        MTL::DataType dataType;
        switch (descriptorType)
        {
        case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
        case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
          dataType = MTL::DataTypeTexture;
          break;
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
          dataType = MTL::DataTypePointer;
          break;
        case VK_DESCRIPTOR_TYPE_SAMPLER:
          dataType = MTL::DataTypeSampler;
          break;
        case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:
          dataType = MTL::DataTypeInstanceAccelerationStructure;
          break;
        default:
          CGPU_FATAL("unhandled data type");
        }

        MTL::BindingAccess access;
        if (binding.readAccess && binding.writeAccess)
        {
          access = MTL::BindingAccessReadWrite;
        }
        else if (binding.writeAccess)
        {
          access = MTL::BindingAccessWriteOnly;
        }
        else
	      {
          access = MTL::BindingAccessReadOnly;
	      }

        //MTL::TextureType textureType;
        //if (descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ||
        //    descriptorType == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE ||
        //    descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
        //{
        //  switch (binding.imageType)
        //  {
        //  case VK_IMAGE_TYPE_1D:
        //    textureType = MTL::TextureType1D;
        //    break;
        //  case VK_IMAGE_TYPE_2D:
        //    textureType = MTL::TextureType2D;
        //    break;
        //  case VK_IMAGE_TYPE_3D:
        //    textureType = MTL::TextureType3D;
        //    break;
        //  default:
        //    CGPU_FATAL("unhandled image type");
        //  }
        //}

        auto* desc = MTL::ArgumentDescriptor::alloc()->init();
        CHK_MTL_NP(desc);
        desc->setDataType(dataType);
        desc->setIndex(binding.binding);
        desc->setAccess(access);
        desc->setArrayLength(binding.count);
        desc->setTextureType(MTL::TextureType2D); // NOTE: can't derive 3d/2d from SPIR-V

        argumentDescriptors.push_back(desc);
      }

      pushDescriptorSet(i, argumentDescriptors);

      for (MTL::ArgumentDescriptor* desc : argumentDescriptors)
      {
        desc->release();
      }
    }

    // Aux descriptor set
    {
      static_assert(SPVC_MSL_PUSH_CONSTANT_BINDING == 0, "assumption invalidated");

      auto* pcDesc = MTL::ArgumentDescriptor::alloc()->init();
      CHK_MTL_NP(pcDesc);
      pcDesc->setDataType(MTL::DataTypePointer);
      pcDesc->setIndex(0);
      pcDesc->setAccess(MTL::BindingAccessReadOnly);

      auto* ifbaDesc = MTL::ArgumentDescriptor::alloc()->init();
      CHK_MTL_NP(ifbaDesc);
      ifbaDesc->setDataType(MTL::DataTypePointer); // TODO: DataTypeIntersectionFunctionTable?
      ifbaDesc->setIndex(1);
      ifbaDesc->setAccess(MTL::BindingAccessReadOnly);

      pushDescriptorSet(/*SPVC_MSL_PUSH_CONSTANT_DESC_SET*/descriptorSetCount, { pcDesc, ifbaDesc }); // TODO: rename SPIRV-Cross constant

      pcDesc->release();
      ifbaDesc->release();
    }

    ipipeline->state = state;
    ipipeline->argumentTable = argumentTable;
    ipipeline->argumentEncoders = argumentEncoders;
    ipipeline->argumentBuffers = argumentBuffers;

    pipeline->handle = handle;
    return true;
  }


  bool cgpuCreateComputePipeline(CgpuDevice device,
                                 CgpuComputePipelineCreateInfo createInfo,
                                 CgpuPipeline* pipeline)
  {
    CGPU_RESOLVE_DEVICE(device, idevice);
    CGPU_RESOLVE_SHADER(createInfo.shader, ishader);

    return cgpuCreateComputePipeline(idevice, ishader, createInfo.debugName, pipeline);
  }

  bool cgpuCreateRtPipeline(CgpuDevice device,
                            CgpuRtPipelineCreateInfo createInfo,
                            CgpuPipeline* pipeline)
  {

    CGPU_RESOLVE_DEVICE(device, idevice);
    CGPU_RESOLVE_SHADER(createInfo.rgenShader, irgenShader);

    // Collect all shaders and create pipeline
    uint32_t functionCount = createInfo.hitGroupCount; // TODO: * 2 for anyhit (probably in user space)

    std::vector<MTL4::FunctionDescriptor*> hitFunctions(functionCount); // TODO: rename
    MTL4::StaticLinkingDescriptor* linkedFunctions = nullptr;

    if (functionCount > 0)
    {
      for (uint32_t i = 0; i < functionCount; i++)
      {
        CgpuShader shader = createInfo.hitGroups[i].closestHitShader; // TODO: ignores any hit
        CGPU_RESOLVE_SHADER(shader, ishader);

        NS::String* entryFuncName = NS::String::string(ishader->entryPoint.c_str(), NS::UTF8StringEncoding);

        auto* funDesc = MTL4::LibraryFunctionDescriptor::alloc()->init();
        funDesc->setLibrary(ishader->library);
        funDesc->setName(entryFuncName);

        hitFunctions[i] = funDesc;
      }

      NS::Array* ar = NS::Array::array((const NS::Object* const*) hitFunctions.data(), (uint32_t) hitFunctions.size());
      CHK_MTL_NP(ar);

      linkedFunctions = MTL4::StaticLinkingDescriptor::alloc()->init();
      linkedFunctions->setFunctionDescriptors(ar);
    }

    if (!cgpuCreateComputePipeline(idevice, irgenShader, createInfo.debugName, pipeline, linkedFunctions))
    {
      return false;
    }

    linkedFunctions->release();

    CGPU_RESOLVE_PIPELINE(*pipeline, ipipeline);

    MTL::IntersectionFunctionTable* intersectionFunctionTable;
    MTL::Buffer* intersectionFunctionBuffer;
    MTL::Buffer* intersectionFunctionBufferArgs;
    {
      // Create IFT
      auto* descriptor = MTL::IntersectionFunctionTableDescriptor::alloc()->init();
      CHK_MTL_NP(descriptor);

      descriptor->setFunctionCount(functionCount);

      intersectionFunctionTable = ipipeline->state->newIntersectionFunctionTable(descriptor);
      CHK_MTL_NP(intersectionFunctionTable);

      descriptor->release();

      // Create intersection function buffer
      constexpr static uint64_t FUNCTION_STRIDE = sizeof(MTL::ResourceID);
      static_assert(FUNCTION_STRIDE == 8, "intersection function buffer stride must be 0 or 8");

      uint64_t intersectionFunctionBufferSize = FUNCTION_STRIDE * functionCount;

      intersectionFunctionBuffer = idevice->device->newBuffer(intersectionFunctionBufferSize, MTL::ResourceStorageModeShared);
      CHK_MTL_NP(intersectionFunctionBuffer);

      auto* ifBufferMem = (uint8_t*) intersectionFunctionBuffer->contents();

      // Fill intersection function table & buffer
      intersectionFunctionTable->setBuffer(intersectionFunctionBuffer, /*offset*/0, /*index*/0);

      MTL::IntersectionFunctionSignature functionSignature = MTL::IntersectionFunctionSignatureInstancing |
                                                             MTL::IntersectionFunctionSignatureTriangleData |
                                                             MTL::IntersectionFunctionSignatureWorldSpaceData |
                                                             MTL::IntersectionFunctionSignatureIntersectionFunctionBuffer |
                                                             MTL::IntersectionFunctionSignatureUserData;

      for (uint32_t i = 0; i < functionCount; i++)
      {
        MTL4::FunctionDescriptor* hitFunc = hitFunctions[i];

        // TODO: no idea if this is correct
        //MTL::ResourceID resourceId = funcHandle->gpuResourceID();
        //memcpy((void*) &ifBufferMem[i * FUNCTION_STRIDE], &resourceId, FUNCTION_STRIDE);

        intersectionFunctionTable->setOpaqueTriangleIntersectionFunction(functionSignature, i);
      }

      // TODO: it seems that we need to have an explicit ref to this buffer in MSL.
      //
      // TODO: one idea: maybe we can rename PC descriptor set to 'aux' descriptor set.
      // TODO: and as binding 2, we can expect a pointer to this function (if RGEN shader).

      // Upload argument buffer
      MTL::IntersectionFunctionBufferArguments args = {
        .intersectionFunctionBuffer = intersectionFunctionBuffer->gpuAddress(),
        .intersectionFunctionBufferSize = intersectionFunctionBufferSize,
        .intersectionFunctionStride = FUNCTION_STRIDE
      };

      uint64_t ifBufferArgsSize = sizeof(MTL::IntersectionFunctionBufferArguments);
      intersectionFunctionBufferArgs = idevice->device->newBuffer(ifBufferArgsSize, MTL::ResourceStorageModeShared);
      CHK_MTL_NP(intersectionFunctionBufferArgs);

      void* ifBufferArgsMem = intersectionFunctionBufferArgs->contents();
      memcpy(ifBufferArgsMem, &args, ifBufferArgsSize);
    }

    for (MTL4::FunctionDescriptor* fun : hitFunctions)
    {
      fun->release();
    }

    ipipeline->intersectionFunctionTable = intersectionFunctionTable;
    ipipeline->intersectionFunctionBufferArgs = intersectionFunctionBufferArgs;
    ipipeline->intersectionFunctionBuffer = intersectionFunctionBuffer;

    return true;
  }

  bool cgpuDestroyPipeline(CgpuDevice device, CgpuPipeline pipeline)
  {
    CGPU_RESOLVE_PIPELINE(pipeline, ipipeline);

    if (ipipeline->intersectionFunctionTable)
    {
      ipipeline->intersectionFunctionTable->release();
    }
    if (ipipeline->intersectionFunctionBuffer)
    {
      ipipeline->intersectionFunctionBuffer->release();
    }
    if (ipipeline->intersectionFunctionBufferArgs)
    {
      ipipeline->intersectionFunctionBufferArgs->release();
    }

    for (MTL::ArgumentEncoder* encoder : ipipeline->argumentEncoders)
    {
      encoder->release();
    }
    for (MTL::Buffer* buffer : ipipeline->argumentBuffers)
    {
      buffer->release();
    }
    ipipeline->argumentTable->release();

    ipipeline->state->release();

    iinstance->ipipelineStore.free(pipeline.handle);
    return true;
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

    auto vertexBufferRange = MTL4::BufferRange::Make(ivertexBuffer->buffer->gpuAddress(), ivertexBuffer->size);
    auto indexBufferRange = MTL4::BufferRange::Make(iindexBuffer->buffer->gpuAddress(), iindexBuffer->size);

    auto* triDesc = MTL4::AccelerationStructureTriangleGeometryDescriptor::alloc()->init();
    triDesc->setVertexBuffer(vertexBufferRange);
    triDesc->setVertexStride(sizeof(float) * 3);
    triDesc->setIndexBuffer(indexBufferRange);
    triDesc->setIndexType(MTL::IndexTypeUInt32);
    triDesc->setTriangleCount(createInfo.triangleCount);
    triDesc->setOpaque(createInfo.isOpaque);

    auto* blasDesc = MTL4::PrimitiveAccelerationStructureDescriptor::alloc()->init();
    CHK_MTL_NP(blasDesc);

    NS::Array* geoDescs = NS::Array::array(triDesc);
    CHK_MTL_NP(geoDescs);
    blasDesc->setGeometryDescriptors(geoDescs);

    MTL::AccelerationStructureSizes sizes = idevice->device->accelerationStructureSizes(blasDesc);

    MTL::Buffer* blasBuffer = idevice->device->newBuffer(sizes.accelerationStructureSize, MTL::ResourceStorageModePrivate);
    CHK_MTL_NP(blasBuffer);

    MTL::Buffer* scratchBuffer = idevice->device->newBuffer(sizes.buildScratchBufferSize, MTL::ResourceStorageModePrivate);
    CHK_MTL_NP(scratchBuffer);

    MTL::AccelerationStructure* as = idevice->device->newAccelerationStructure(sizes.accelerationStructureSize);
    CHK_MTL_NP(as);

    MTL::Event* event = idevice->device->newEvent();
    CHK_MTL_NP(event);
    MTL4::CommandBuffer* commandBuffer = idevice->device->newCommandBuffer();

    auto* commandAllocator = idevice->device->newCommandAllocator();
    commandBuffer->beginCommandBuffer(commandAllocator); // TODO: pass options with LogState

    CHK_MTL_NP(commandBuffer);
    MTL4::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();
    CHK_MTL_NP(encoder);

    auto scratchBufferRange = MTL4::BufferRange::Make(scratchBuffer->gpuAddress(), sizes.buildScratchBufferSize);
    encoder->buildAccelerationStructure(as, blasDesc, scratchBufferRange);
    encoder->endEncoding();

    commandBuffer->endCommandBuffer();

    MTL4::CommandQueue* commandQueue = idevice->commandQueue;
    commandQueue->commit(&commandBuffer, 1, idevice->commitOptions);
    commandQueue->signalEvent(event, 42);
    commandQueue->wait(event, 42);

    event->release();
    commandBuffer->release();
    commandAllocator->release();

    scratchBuffer->release();
    blasDesc->release();
    triDesc->release();

    if (createInfo.debugName)
    {
      as->setLabel(NS::String::string(createInfo.debugName, NS::StringEncoding::UTF8StringEncoding));
    }

    iblas->as = as;
    iblas->buffer = blasBuffer;

    blas->handle = handle;
    return true;
  }

  bool cgpuCreateTlas(CgpuDevice device,
                      CgpuTlasCreateInfo createInfo,
                      CgpuTlas* tlas)
  {
    CGPU_RESOLVE_DEVICE(device, idevice);

    uint64_t handle = iinstance->itlasStore.allocate();

    CGPU_RESOLVE_TLAS({ handle }, itlas);

    // Upload instance buffer.
    uint64_t instanceBufferSize;
    MTL::Buffer* instanceBuffer;
    {
      std::vector<MTL::AccelerationStructureUserIDInstanceDescriptor> instances(createInfo.instanceCount);

      for (uint32_t i = 0; i < createInfo.instanceCount; i++)
      {
        const CgpuBlasInstance& instance = createInfo.instances[i];

// TODO: it could be that we need to use a Metal 4 type here
        MTL::AccelerationStructureUserIDInstanceDescriptor& d = instances[i];
        //d.allowDuplicateIntersectionFunctionInvocation = false;
        d.options = MTL::AccelerationStructureInstanceOptionNone; // TODO: propagate opaque flag
        d.mask = 0xFFFFFFFF;
        d.intersectionFunctionTableOffset = instance.hitGroupIndex;
        d.accelerationStructureIndex = i;
        d.userID = instance.instanceCustomIndex;;
        // TODO: if transposed, set MTL::AccelerationStructureTriangleGeometryDescriptor::transformationMatrixLayout (MTL::MatrixLayout)
        memcpy(&d.transformationMatrix, instance.transform, sizeof(instance.transform)); // TODO: might be transposed
      }

      instanceBufferSize = 72/*sizeof(MTL::AccelerationStructureUserIDInstanceDescriptor)*/ * instances.size();

      instanceBuffer = idevice->device->newBuffer(instanceBufferSize, MTL::ResourceStorageModeShared);
      CHK_MTL_NP(instanceBuffer);

      memcpy(instanceBuffer->contents(), instances.data(), instanceBufferSize);
      // TODO: only allowed for StorageModeManaged
      //instanceBuffer->didModifyRange(NS::Range::Make(0, instanceBufferSize));
    }

    auto instanceBufferRange = MTL4::BufferRange::Make(instanceBuffer->gpuAddress(), instanceBufferSize);

    auto* descriptor = MTL4::InstanceAccelerationStructureDescriptor::alloc()->init();
    CHK_MTL_NP(descriptor);
    descriptor->setInstanceDescriptorBuffer(instanceBufferRange);
    descriptor->setInstanceDescriptorStride(/*sizeof(MTL::AccelerationStructureUserIDInstanceDescriptor)*/72);
    descriptor->setInstanceCount(createInfo.instanceCount);

    // Build TLAS.
    MTL::Buffer* tlasBuffer;
    MTL::AccelerationStructure* as;
    {
      MTL::AccelerationStructureSizes sizes = idevice->device->accelerationStructureSizes(descriptor);

      tlasBuffer = idevice->device->newBuffer(sizes.accelerationStructureSize, MTL::ResourceStorageModePrivate);
      CHK_MTL_NP(tlasBuffer);

      as = idevice->device->newAccelerationStructure(sizes.accelerationStructureSize);
      CHK_MTL_NP(as);

      MTL::Buffer* scratchBuffer = idevice->device->newBuffer(sizes.buildScratchBufferSize, MTL::ResourceStorageModePrivate);
      CHK_MTL_NP(scratchBuffer);

      MTL::Event* event = idevice->device->newEvent();
      CHK_MTL_NP(event);
      MTL4::CommandBuffer* commandBuffer = idevice->device->newCommandBuffer();
      CHK_MTL_NP(commandBuffer);

      auto* commandAllocator = idevice->device->newCommandAllocator();
      commandBuffer->beginCommandBuffer(commandAllocator); // TODO: pass options with LogState

      MTL4::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();
      CHK_MTL_NP(encoder);

      auto scratchBufferRange = MTL4::BufferRange::Make(scratchBuffer->gpuAddress(), sizes.buildScratchBufferSize);
      encoder->buildAccelerationStructure(as, descriptor, scratchBufferRange);
      encoder->endEncoding();

      commandBuffer->endCommandBuffer();

      MTL4::CommandQueue* commandQueue = idevice->commandQueue;
      commandQueue->commit(&commandBuffer, 1, idevice->commitOptions);
      commandQueue->signalEvent(event, 42);
      commandQueue->wait(event, 42);

      event->release();
      commandBuffer->release();
      commandAllocator->release();
      scratchBuffer->release();
    }

    descriptor->release();
    instanceBuffer->release();

    itlas->as = as;
    itlas->buffer = tlasBuffer;

    tlas->handle = handle;
    return true;
  }

  bool cgpuDestroyBlas(CgpuDevice device, CgpuBlas blas)
  {
    CGPU_RESOLVE_BLAS(blas, iblas);

    iblas->as->release();
    iblas->buffer->release();

    iinstance->iblasStore.free(blas.handle);
    return true;
  }

  bool cgpuDestroyTlas(CgpuDevice device, CgpuTlas tlas)
  {
    CGPU_RESOLVE_TLAS(tlas, itlas);

    itlas->as->release();
    itlas->buffer->release();

    iinstance->itlasStore.free(tlas.handle);
    return true;
  }

  bool cgpuCreateCommandBuffer(CgpuDevice device, CgpuCommandBuffer* commandBuffer)
  {
    CGPU_RESOLVE_DEVICE(device, idevice);

    uint64_t handle = iinstance->icommandBufferStore.allocate();

    CGPU_RESOLVE_COMMAND_BUFFER({ handle }, icommandBuffer);

    MTL::ResourceOptions options = MTL::ResourceOptionCPUCacheModeWriteCombined;
    MTL::Buffer* pcBuffer = idevice->device->newBuffer(CGPU_MAX_PUSH_CONSTANTS_SIZE, options);
    CHK_MTL_NP(pcBuffer);

    icommandBuffer->commandAllocator = idevice->device->newCommandAllocator();
    icommandBuffer->commandBuffer = idevice->device->newCommandBuffer();
    icommandBuffer->encoder = nullptr;
    icommandBuffer->pcBuffer = pcBuffer;
    icommandBuffer->pcMem = pcBuffer->contents();
    icommandBuffer->counterHeap = idevice->counterHeap;
#ifndef NDEBUG
    icommandBuffer->logState = idevice->logState;
    icommandBuffer->commitOptions = idevice->commitOptions;
#endif

    commandBuffer->handle = handle;
    return true;
  }

  bool cgpuDestroyCommandBuffer(CgpuDevice device, CgpuCommandBuffer commandBuffer)
  {
    CGPU_RESOLVE_COMMAND_BUFFER(commandBuffer, icommandBuffer);

    icommandBuffer->commandBuffer->release();
    icommandBuffer->commandAllocator->release();
    icommandBuffer->pcBuffer->release();

    iinstance->icommandBufferStore.free(commandBuffer.handle);
    return true;
  }

  bool cgpuBeginCommandBuffer(CgpuCommandBuffer commandBuffer)
  {
    CGPU_RESOLVE_COMMAND_BUFFER(commandBuffer, icommandBuffer);

    auto* options = MTL4::CommandBufferOptions::alloc()->init();
#ifndef NDEBUG
    options->setLogState(icommandBuffer->logState);
#endif

    icommandBuffer->commandBuffer->beginCommandBuffer(icommandBuffer->commandAllocator, options);
    icommandBuffer->encoder = icommandBuffer->commandBuffer->computeCommandEncoder();

    options->release();
    return true;
  }

  void cgpuCmdBindPipeline(CgpuCommandBuffer commandBuffer, CgpuPipeline pipeline)
  {
    CGPU_RESOLVE_COMMAND_BUFFER(commandBuffer, icommandBuffer);
    CGPU_RESOLVE_PIPELINE(pipeline, ipipeline);

    MTL4::ComputeCommandEncoder* encoder = icommandBuffer->encoder;

    encoder->setComputePipelineState(ipipeline->state);

    icommandBuffer->commandBuffer->useResidencySet(ipipeline->residencySet);

// TODO: bind SBT here?
    //encoder->setIntersectionFunctionTable(ipipeline->ift, bufferIndex);
  }

  void cgpuCmdTransitionShaderImageLayouts(CgpuCommandBuffer commandBuffer,
                                           CgpuShader shader,
                                           uint32_t descriptorSetIndex,
                                           uint32_t imageCount,
                                           const CgpuImageBinding* images)
  {
    CGPU_RESOLVE_COMMAND_BUFFER(commandBuffer, icommandBuffer);

    MTL4::ComputeCommandEncoder* encoder = icommandBuffer->encoder;

    for (uint32_t i = 0; i < imageCount; i++)
    {
      const CgpuImageBinding& b = images[i];
      CGPU_RESOLVE_IMAGE(b.image, iimage);

      encoder->optimizeContentsForGPUAccess(iimage->texture);
    }
  }

  void cgpuCmdUpdateBindings(CgpuDevice device,
                             CgpuCommandBuffer commandBuffer,
                             CgpuPipeline pipeline,
                             uint32_t descriptorSetIndex,
                             const CgpuBindings* bindings)
  {
    CGPU_RESOLVE_DEVICE(device, idevice);
    CGPU_RESOLVE_COMMAND_BUFFER(commandBuffer, icommandBuffer);
    CGPU_RESOLVE_PIPELINE(pipeline, ipipeline);

    if (ipipeline->residencySet)
    {
      ipipeline->residencySet->release();
    }

    // TODO: in the dev/master branch, this should be a member of BindGroup
    MTL::ResidencySet* residencySet = nullptr;
    {
      auto* desc = MTL::ResidencySetDescriptor::alloc()->init();
      desc->setInitialCapacity(bindings->bufferCount + bindings->imageCount + bindings->samplerCount + bindings->tlasCount);

      NS::Error* error = nullptr;
      residencySet = idevice->device->newResidencySet(desc, &error);
      CHK_MTL(residencySet, error);

      desc->release();
    }

    MTL::ArgumentEncoder* argumentEncoder = ipipeline->argumentEncoders[descriptorSetIndex];
    MTL::Buffer* argumentBuffer = ipipeline->argumentBuffers[descriptorSetIndex];

    uint32_t offset = 0;
    argumentEncoder->setArgumentBuffer(argumentBuffer, offset);

    for (uint32_t i = 0; i < bindings->bufferCount; i++)
    {
      const CgpuBufferBinding& b = bindings->buffers[i];

      CGPU_RESOLVE_BUFFER(b.buffer, ibuffer);
      argumentEncoder->setBuffer(ibuffer->buffer, b.offset, b.binding);
      residencySet->addAllocation(ibuffer->buffer);
    }

    for (uint32_t i = 0; i < bindings->imageCount; i++)
    {
      const CgpuImageBinding& b = bindings->images[i];

      // TODO: there's no option to set a texture within an array!
      //       -> we might have to rewrite tex2d array assignment
      CGPU_RESOLVE_IMAGE(b.image, iimage);
      argumentEncoder->setTexture(iimage->texture, b.binding);
      residencySet->addAllocation(iimage->texture);
    }

    for (uint32_t i = 0; i < bindings->samplerCount; i++)
    {
      const CgpuSamplerBinding& b = bindings->samplers[i];

      CGPU_RESOLVE_SAMPLER(b.sampler, isampler);
      argumentEncoder->setSamplerState(isampler->sampler, b.binding);
      //residencySet->addAllocation(isampler->sampler);
    }

    for (uint32_t i = 0; i < bindings->tlasCount; i++)
    {
      const CgpuTlasBinding& b = bindings->tlases[i];

      CGPU_RESOLVE_TLAS(b.as, itlas);
      argumentEncoder->setAccelerationStructure(itlas->as, b.binding);
      residencySet->addAllocation(itlas->as);
    }

    ipipeline->residencySet = residencySet;

// TODO: we need to do something similar for push constants
    icommandBuffer->encoder->setArgumentTable(ipipeline->argumentTable);
  }

  void cgpuCmdUpdateBuffer(CgpuCommandBuffer commandBuffer,
                           const uint8_t* data,
                           uint64_t size,
                           CgpuBuffer dstBuffer,
                           uint64_t dstOffset)
  {
    CGPU_FATAL("command not supported"); // because maxBufferUpdateSize is 0.
  }

  void cgpuCmdCopyBuffer(CgpuCommandBuffer commandBuffer,
                         CgpuBuffer srcBuffer,
                         uint64_t srcOffset,
                         CgpuBuffer dstBuffer,
                         uint64_t dstOffset,
                         uint64_t size)
  {
    CGPU_RESOLVE_COMMAND_BUFFER(commandBuffer, icommandBuffer);
    CGPU_RESOLVE_BUFFER(srcBuffer, isrcBuffer);
    CGPU_RESOLVE_BUFFER(dstBuffer, idstBuffer);

    MTL4::ComputeCommandEncoder* encoder = icommandBuffer->encoder;

    uint64_t rangeSize = (size == CGPU_WHOLE_SIZE) ? std::min(isrcBuffer->size, idstBuffer->size) : size;
    encoder->copyFromBuffer(isrcBuffer->buffer, srcOffset, idstBuffer->buffer, dstOffset, rangeSize);
  }

  void cgpuCmdCopyBufferToImage(CgpuCommandBuffer commandBuffer,
                                CgpuBuffer buffer,
                                CgpuImage image,
                                const CgpuBufferImageCopyDesc* desc)
  {
    CGPU_RESOLVE_COMMAND_BUFFER(commandBuffer, icommandBuffer);
    CGPU_RESOLVE_BUFFER(buffer, ibuffer);
    CGPU_RESOLVE_IMAGE(image, iimage);

    MTL4::ComputeCommandEncoder* encoder = icommandBuffer->encoder;

    uint32_t bytesPerPixel = 4; // TODO: from helper function
    uint32_t srcBytesPerRow = iimage->width * bytesPerPixel;
    uint32_t srcBytesPerImage = iimage->width * iimage->height * iimage->depth * bytesPerPixel;
    MTL::Size srcSize(desc->texelExtentX, desc->texelExtentY, desc->texelExtentZ);

    uint32_t dstSlice = 0;
    uint32_t dstMipmapLevel = 0;
    MTL::Origin dstOrigin(desc->texelOffsetX, desc->texelOffsetY, desc->texelOffsetZ);

    assert((desc->bufferOffset % bytesPerPixel) == 0); // TODO: need to expose as property. Metal docs make this requirement.
    assert((desc->bufferOffset + srcBytesPerImage) <= ibuffer->size); // don't read OOB
    assert(!iimage->texture->isFramebufferOnly());

    encoder->copyFromBuffer(
      ibuffer->buffer,
      desc->bufferOffset,
      srcBytesPerRow,
      srcBytesPerImage,
      srcSize,
      iimage->texture,
      dstSlice,
      dstMipmapLevel,
      dstOrigin
    );
  }

  void cgpuCmdPushConstants(CgpuCommandBuffer commandBuffer,
                            CgpuPipeline pipeline,
                            CgpuShaderStageFlags stageFlags,
                            uint32_t size,
                            const void* data)
  {
    CGPU_RESOLVE_COMMAND_BUFFER(commandBuffer, icommandBuffer);
    CGPU_RESOLVE_PIPELINE(pipeline, ipipeline);

    MTL4::ComputeCommandEncoder* encoder = icommandBuffer->encoder;

    memcpy(icommandBuffer->pcMem, data, size);

    MTL::ArgumentEncoder* argumentEncoder = ipipeline->argumentEncoders.back(); // PC encoder
    MTL::Buffer* argumentBuffer = ipipeline->argumentBuffers.back(); // PC buffer

    uint32_t offset = 0;
    argumentEncoder->setArgumentBuffer(argumentBuffer, offset);

    // TODO: do we need to batch them? when and where else to set IFBA?
    static_assert(SPVC_MSL_PUSH_CONSTANT_BINDING == 0, "assumption invalidated");
    argumentEncoder->setBuffer(icommandBuffer->pcBuffer, offset, 0);
    // TODO: bind IntersectionFunctionTable instead?
    argumentEncoder->setBuffer(ipipeline->intersectionFunctionBufferArgs, offset, 1);
  }

  static void cgpuCmdDispatch(CgpuICommandBuffer* icommandBuffer,
                              uint32_t dim_x,
                              uint32_t dim_y,
                              uint32_t dim_z)
  {
    MTL4::ComputeCommandEncoder* encoder = icommandBuffer->encoder;

    MTL::Size groupsPerGrid(dim_x, dim_y, dim_x);
    MTL::Size threadsPerGroup(32, 32, 1); // TODO: retrieve this from the bound pipeline
   // encoder->dispatchThreadgroups(groupsPerGrid, threadsPerGroup);
  }

  void cgpuCmdDispatch(CgpuCommandBuffer commandBuffer,
                       uint32_t dim_x,
                       uint32_t dim_y,
                       uint32_t dim_z)
  {
    CGPU_RESOLVE_COMMAND_BUFFER(commandBuffer, icommandBuffer);

    cgpuCmdDispatch(icommandBuffer, dim_x, dim_y, dim_z);
  }

  void cgpuCmdPipelineBarrier(CgpuCommandBuffer commandBuffer,
                              const CgpuPipelineBarrier* barrier)
  {
    CGPU_RESOLVE_COMMAND_BUFFER(commandBuffer, icommandBuffer);

    MTL4::ComputeCommandEncoder* encoder = icommandBuffer->encoder;

    for (uint32_t i = 0; i < barrier->memoryBarrierCount; i++)
    {
      const CgpuMemoryBarrier& b = barrier->memoryBarriers[i];

      MTL::Stages beforeStages = cgpuTranslatePipelineStages(b.srcStageMask);
      MTL::Stages afterStages = cgpuTranslatePipelineStages(b.dstStageMask);

      encoder->barrierAfterEncoderStages(afterStages, beforeStages, MTL4::VisibilityOptionDevice);
    }

    for (uint32_t i = 0; i < barrier->bufferBarrierCount; i++)
    {
      const CgpuBufferMemoryBarrier& b = barrier->bufferBarriers[i];

      MTL::Stages beforeStages = cgpuTranslatePipelineStages(b.srcStageMask);
      MTL::Stages afterStages = cgpuTranslatePipelineStages(b.dstStageMask);

      encoder->barrierAfterEncoderStages(afterStages, beforeStages, MTL4::VisibilityOptionDevice);
    }

    for (uint32_t i = 0; i < barrier->imageBarrierCount; i++)
    {
      const CgpuImageMemoryBarrier& b = barrier->imageBarriers[i];

      MTL::Stages beforeStages = cgpuTranslatePipelineStages(b.srcStageMask);
      MTL::Stages afterStages = cgpuTranslatePipelineStages(b.dstStageMask);

      encoder->barrierAfterEncoderStages(afterStages, beforeStages, MTL4::VisibilityOptionDevice);
    }
  }

  void cgpuCmdResetTimestamps(CgpuCommandBuffer commandBuffer,
                              uint32_t offset,
                              uint32_t count)
  {
    CGPU_RESOLVE_COMMAND_BUFFER(commandBuffer, icommandBuffer);

    NS::Range range(offset, count);
    icommandBuffer->counterHeap->invalidateCounterRange(range); // clears to 0
  }

  void cgpuCmdWriteTimestamp(CgpuCommandBuffer commandBuffer,
                             uint32_t timestampIndex)
  {
    CGPU_RESOLVE_COMMAND_BUFFER(commandBuffer, icommandBuffer);

    // For debug configuration, we want precise profiling measurements.
    MTL4::TimestampGranularity granularity;
#ifndef NDEBUG
    granularity = MTL4::TimestampGranularityPrecise;
#else
    granularity = MTL4::TimestampGranularityRelaxed;
#endif

    MTL4::ComputeCommandEncoder* encoder = icommandBuffer->encoder;
    encoder->writeTimestamp(granularity, icommandBuffer->counterHeap, timestampIndex);
  }

  void cgpuCmdCopyTimestamps(CgpuCommandBuffer commandBuffer,
                             CgpuBuffer buffer,
                             uint32_t offset,
                             uint32_t count,
                             bool waitUntilAvailable)
  {
    CGPU_RESOLVE_COMMAND_BUFFER(commandBuffer, icommandBuffer);
    CGPU_RESOLVE_BUFFER(buffer, ibuffer);

    uint32_t lastIndex = offset + count;
    if (lastIndex >= CGPU_MAX_TIMESTAMP_QUERIES)
    {
      CGPU_FATAL("max timestamp query count exceeded!");
    }

    for (uint32_t i = offset; i < count; i++)
    {
      icommandBuffer->commandBuffer->writeTimestampIntoHeap(icommandBuffer->counterHeap, i);
    }

    if (!waitUntilAvailable)
    {
      return;
    }

    NS::Range range(offset, count);
    auto bufferRange = MTL4::BufferRange::Make(ibuffer->buffer->gpuAddress(), ibuffer->size);

    icommandBuffer->commandBuffer->resolveCounterHeap(
      icommandBuffer->counterHeap,
      range,
      bufferRange,
      nullptr,
      nullptr
    );
  }

  void cgpuCmdTraceRays(CgpuCommandBuffer commandBuffer, uint32_t width, uint32_t height)
  {
    CGPU_RESOLVE_COMMAND_BUFFER(commandBuffer, icommandBuffer);
    cgpuCmdDispatch(icommandBuffer, width, height, 1);
  }

  void cgpuEndCommandBuffer(CgpuCommandBuffer commandBuffer)
  {
    CGPU_RESOLVE_COMMAND_BUFFER(commandBuffer, icommandBuffer);

    icommandBuffer->encoder->endEncoding();

    icommandBuffer->commandBuffer->endCommandBuffer();
  }

  bool cgpuCreateSemaphore(CgpuDevice device, CgpuSemaphore* semaphore, uint64_t initialValue)
  {
    CGPU_RESOLVE_DEVICE(device, idevice);

    uint64_t handle = iinstance->isemaphoreStore.allocate();

    CGPU_RESOLVE_SEMAPHORE({ handle }, isemaphore);

    isemaphore->event = idevice->device->newEvent();

    semaphore->handle = handle;
    return true;
  }

  bool cgpuDestroySemaphore(CgpuDevice device, CgpuSemaphore semaphore)
  {
    CGPU_RESOLVE_SEMAPHORE(semaphore, isemaphore);

    isemaphore->event->release();

    iinstance->isemaphoreStore.free(semaphore.handle);
    return true;
  }

  bool cgpuWaitSemaphores(CgpuDevice device,
                          uint32_t semaphoreInfoCount,
                          CgpuWaitSemaphoreInfo* semaphoreInfos)
  {
    CGPU_RESOLVE_DEVICE(device, idevice);

    MTL4::CommandQueue* commandQueue = idevice->commandQueue;
    for (uint32_t i = 0; i < semaphoreInfoCount; i++)
    {
      CGPU_RESOLVE_SEMAPHORE(semaphoreInfos[i].semaphore, isemaphore);
      commandQueue->wait(isemaphore->event, semaphoreInfos[i].value);
    }

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

    MTL4::CommandQueue* commandQueue = idevice->commandQueue;
    commandQueue->commit(&icommandBuffer->commandBuffer, 1, idevice->commitOptions);

    cgpuWaitSemaphores(device, waitSemaphoreInfoCount, waitSemaphoreInfos);

    for (uint32_t i = 0; i < signalSemaphoreInfoCount; i++)
    {
      CGPU_RESOLVE_SEMAPHORE(signalSemaphoreInfos[i].semaphore, isemaphore);
      commandQueue->signalEvent(isemaphore->event, signalSemaphoreInfos[i].value);
    }

    return true;
  }

  bool cgpuFlushMappedMemory(CgpuDevice device,
                             CgpuBuffer buffer,
                             uint64_t offset,
                             uint64_t size)
  {
    CGPU_RESOLVE_BUFFER(buffer, ibuffer);

    NS::Range range(offset, size);
    // TODO: only allowed for StorageModeManaged
    //ibuffer->buffer->didModifyRange(range);
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

    features = CgpuPhysicalDeviceFeatures {
      .debugPrintf = false,
      .pageableDeviceLocalMemory = false,
      .pipelineLibraries = false,
      .pipelineStatisticsQuery = false,
      .rayTracingInvocationReorder = false,
      .rayTracingValidation = false,
      .shaderClock = true,
      .shaderFloat64 = false,
      .shaderImageGatherExtended = false,
      .shaderInt16 = true,
      .shaderInt64 = false,
      .shaderSampledImageArrayDynamicIndexing = true,
      .shaderStorageBufferArrayDynamicIndexing = true,
      .shaderStorageImageArrayDynamicIndexing = true,
      .shaderStorageImageExtendedFormats = false,
      .shaderStorageImageReadWithoutFormat = false,
      .shaderStorageImageWriteWithoutFormat = false,
      .shaderUniformBufferArrayDynamicIndexing = false,
      .sparseBinding = false,
      .sparseResidencyAliased = false,
      .sparseResidencyBuffer = false,
      .sparseResidencyImage2D = false,
      .sparseResidencyImage3D = false,
      .textureCompressionBC = idevice->device->supportsBCTextureCompression()
    };

    return true;
  }

  // TODO: init device member instead
  bool cgpuGetPhysicalDeviceProperties(CgpuDevice device,
                                       CgpuPhysicalDeviceProperties& properties)
  {
    CGPU_RESOLVE_DEVICE(device, idevice);

    properties = CgpuPhysicalDeviceProperties {
      .maxBufferUpdateSize = 0, // not supported by Metal backend
      .maxComputeSharedMemorySize = uint32_t(idevice->device->maxThreadgroupMemoryLength())
    };

    return true;
  }
}
