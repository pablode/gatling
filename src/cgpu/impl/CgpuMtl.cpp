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
#include <atomic>

#include <gtl/gb/Fmt.h>
#include <gtl/gb/Log.h>
#include <gtl/gb/LinearDataStore.h>
#include <gtl/gb/SmallVector.h>

#include <spirv_cross_c.h>

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Metal.hpp>

//#define CGPU_MTL_CAPTURE_ENABLED

namespace gtl
{
  /* Constants */

  constexpr static const uint32_t SPVC_MSL_VERSION = SPVC_MAKE_MSL_VERSION(4, 0, 0);

  constexpr static const uint32_t CGPU_MAX_ARGUMENT_BUFFER_COUNT = 30;

  typedef enum VkDescriptorType {
    VK_DESCRIPTOR_TYPE_SAMPLER = 0,
    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER = 1,
    VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE = 2,
    VK_DESCRIPTOR_TYPE_STORAGE_IMAGE = 3,
    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER = 6,
    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER = 7,
    VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR = 1000150000
  } VkDescriptorType;

  const static CgpuDeviceFeatures CGPU_DEVICE_FEATURES =
  {
    .debugPrintf = true,
    .rayTracingInvocationReorder = false,
    .shaderClock = false,
    .sharedMemory = true
  };

  // See Apple Feature Set table PDF
  const static CgpuDeviceProperties CGPU_DEVICE_PROPERTIES =
  {
    .minStorageBufferOffsetAlignment = 4,
    .minUniformBufferOffsetAlignment = 4,
    .maxComputeSharedMemorySize = 32 * 1024 * 1024,
    .maxRayHitAttributeSize = UINT32_MAX,
    .subgroupSize = 32
  };

  /* Internal structures */

  struct CgpuIDevice
  {
    MTL::Device*        device;

    MTL4::CommandQueue* commandQueue;
    MTL4::Compiler*     compiler;
    MTL::ResidencySet*  residencySet; // for shader device address buffers
    uint32_t            uniqueShaderEntryPointCounter;

    MTL4::CommandBufferOptions* commandBufferOptions;

#ifndef NDEBUG
    MTL::LogState*       logState; // nullable
    MTL4::CommitOptions* commitOptions;
#endif
  };

  struct CgpuIBuffer
  {
    MTL::Buffer* buffer;
    uint64_t size;
    bool isDynamic;
  };

  struct CgpuIImage
  {
    MTL::Texture* texture;
    uint32_t width;
    uint32_t height;
    uint32_t depth;
    CgpuImageFormat format;
  };

  struct CgpuIBindSet
  {
    MTL::ArgumentEncoder* argumentEncoder;
    MTL::Buffer* argumentBuffer;
    MTL::ResidencySet* residencySet;
    std::vector<CgpuBufferBinding> dynamicBuffers;
  };

  struct CgpuRtFunctionTables
  {
    MTL::IntersectionFunctionTable* ift;
    MTL::VisibleFunctionTable* missVft;
    MTL::VisibleFunctionTable* chitVft;
  };

  struct CgpuIPipeline
  {
    MTL::ComputePipelineState* state;
    MTL::Size threadsPerGroup;
    MTL4::ArgumentTable* argumentTable;
    CgpuShaderReflection computeReflection;

    std::vector<CgpuRtFunctionTables> fts; // RT only
  };

  struct CgpuIShader
  {
    MTL::Library* library;
    CgpuShaderReflection reflection;
    NS::String* entryPointName;
  };

  struct CgpuISemaphore
  {
    MTL::SharedEvent* event;
  };

  struct CgpuICommandBuffer
  {
    MTL4::CommandBuffer* commandBuffer;
    MTL4::CommandAllocator* commandAllocator;
    MTL::ResidencySet* auxResidencySet;

    // following member's memory is not owned:
    MTL::ResidencySet* deviceResidencySet;
    std::vector<MTL::ResidencySet*> residencySets;
    CgpuIPipeline* pipeline = nullptr;
#ifndef NDEBUG
    MTL4::CommitOptions* commitOptions;
#endif
    MTL4::CommandBufferOptions* commandBufferOptions;
  };

  struct CgpuIBlas
  {
    MTL::AccelerationStructure* as;
    bool isOpaque;
  };

  struct CgpuITlas
  {
    MTL::AccelerationStructure* as;
    std::unordered_set<const MTL::AccelerationStructure*> blases; // weak refs
  };

  struct CgpuISampler
  {
    MTL::SamplerState* sampler;
  };

  /* Context */

  struct CgpuContext
  {
#ifdef CGPU_MTL_CAPTURE_ENABLED
    MTL::CaptureManager* captureManager;
#endif
    CgpuIDevice idevice;
    spvc_context spvc;

    GbLinearDataStore<CgpuIBuffer, 16> ibufferStore;
    GbLinearDataStore<CgpuIImage, 128> iimageStore;
    GbLinearDataStore<CgpuIShader, 32> ishaderStore;
    GbLinearDataStore<CgpuIPipeline, 8> ipipelineStore;
    GbLinearDataStore<CgpuISemaphore, 16> isemaphoreStore;
    GbLinearDataStore<CgpuICommandBuffer, 16> icommandBufferStore;
    GbLinearDataStore<CgpuISampler, 8> isamplerStore;
    GbLinearDataStore<CgpuIBlas, 1024> iblasStore;
    GbLinearDataStore<CgpuITlas, 1> itlasStore;
    GbLinearDataStore<CgpuIBindSet, 32> ibindSetStore;
  };

  /* Helper macros */

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

#define CGPU_LOG_ERROR(msg)                         \
  do {                                              \
    GB_ERROR("{}:{}: {}", __FILE__, __LINE__, msg); \
    gbLogFlush();                                   \
  } while (false)

#define CGPU_RETURN_ERROR(msg) \
  do {                         \
    CGPU_LOG_ERROR(msg);       \
    return false;              \
  } while (false)

#define CGPU_FATAL(msg)  \
  do {                   \
    CGPU_LOG_ERROR(msg); \
    exit(EXIT_FAILURE);  \
  } while (false)

#define CGPU_LOG_MTL_ERR(E) \
  do { if (E) {             \
    GB_ERROR("{}:{}: {} (code {})", __FILE__, __LINE__, E->localizedDescription()->utf8String(), E->code()); \
    gbLogFlush();           \
  } } while (false)

#define CGPU_CHK(X, E)    \
  do { if (!X) { CGPU_LOG_MTL_ERR(E); assert(false); exit(EXIT_FAILURE); } } while (false)

#define CGPU_CHK_NP(X)                 \
  do { if (!X) {                       \
    CGPU_FATAL("encountered nullptr"); \
  } } while(false)

#define CGPU_RESOLVE_HANDLE(RESOURCE_NAME, HANDLE_TYPE, IRESOURCE_TYPE, RESOURCE_STORE)                            \
  CGPU_INLINE static bool cgpuResolve##RESOURCE_NAME(CgpuContext* ctx, HANDLE_TYPE handle, IRESOURCE_TYPE** idata) \
  {                                                                                                                \
    return ctx->RESOURCE_STORE.get(handle.handle, idata);                                                          \
  }

  CGPU_RESOLVE_HANDLE(       Buffer,        CgpuBuffer,        CgpuIBuffer,        ibufferStore)
  CGPU_RESOLVE_HANDLE(        Image,         CgpuImage,         CgpuIImage,         iimageStore)
  CGPU_RESOLVE_HANDLE(       Shader,        CgpuShader,        CgpuIShader,        ishaderStore)
  CGPU_RESOLVE_HANDLE(     Pipeline,      CgpuPipeline,      CgpuIPipeline,      ipipelineStore)
  CGPU_RESOLVE_HANDLE(    Semaphore,     CgpuSemaphore,     CgpuISemaphore,     isemaphoreStore)
  CGPU_RESOLVE_HANDLE(CommandBuffer, CgpuCommandBuffer, CgpuICommandBuffer, icommandBufferStore)
  CGPU_RESOLVE_HANDLE(      Sampler,       CgpuSampler,       CgpuISampler,       isamplerStore)
  CGPU_RESOLVE_HANDLE(         Blas,          CgpuBlas,          CgpuIBlas,          iblasStore)
  CGPU_RESOLVE_HANDLE(         Tlas,          CgpuTlas,          CgpuITlas,          itlasStore)
  CGPU_RESOLVE_HANDLE(      BindSet,       CgpuBindSet,       CgpuIBindSet,       ibindSetStore)

#define CGPU_RESOLVE_OR_EXIT(CTX, HANDLE, VAR_NAME, ITYPE, RESOLVE_FUNC) \
  ITYPE* VAR_NAME;                                                       \
  if (!RESOLVE_FUNC(CTX, HANDLE, &VAR_NAME)) [[unlikely]] {                   \
    CGPU_FATAL("invalid handle!");                                       \
  }

#define CGPU_RESOLVE_BUFFER(CTX, HANDLE, VAR_NAME)         CGPU_RESOLVE_OR_EXIT(CTX, HANDLE, VAR_NAME, CgpuIBuffer, cgpuResolveBuffer)
#define CGPU_RESOLVE_IMAGE(CTX, HANDLE, VAR_NAME)          CGPU_RESOLVE_OR_EXIT(CTX, HANDLE, VAR_NAME, CgpuIImage, cgpuResolveImage)
#define CGPU_RESOLVE_SHADER(CTX, HANDLE, VAR_NAME)         CGPU_RESOLVE_OR_EXIT(CTX, HANDLE, VAR_NAME, CgpuIShader, cgpuResolveShader)
#define CGPU_RESOLVE_PIPELINE(CTX, HANDLE, VAR_NAME)       CGPU_RESOLVE_OR_EXIT(CTX, HANDLE, VAR_NAME, CgpuIPipeline, cgpuResolvePipeline)
#define CGPU_RESOLVE_SEMAPHORE(CTX, HANDLE, VAR_NAME)      CGPU_RESOLVE_OR_EXIT(CTX, HANDLE, VAR_NAME, CgpuISemaphore, cgpuResolveSemaphore)
#define CGPU_RESOLVE_COMMAND_BUFFER(CTX, HANDLE, VAR_NAME) CGPU_RESOLVE_OR_EXIT(CTX, HANDLE, VAR_NAME, CgpuICommandBuffer, cgpuResolveCommandBuffer)
#define CGPU_RESOLVE_SAMPLER(CTX, HANDLE, VAR_NAME)        CGPU_RESOLVE_OR_EXIT(CTX, HANDLE, VAR_NAME, CgpuISampler, cgpuResolveSampler)
#define CGPU_RESOLVE_BLAS(CTX, HANDLE, VAR_NAME)           CGPU_RESOLVE_OR_EXIT(CTX, HANDLE, VAR_NAME, CgpuIBlas, cgpuResolveBlas)
#define CGPU_RESOLVE_TLAS(CTX, HANDLE, VAR_NAME)           CGPU_RESOLVE_OR_EXIT(CTX, HANDLE, VAR_NAME, CgpuITlas, cgpuResolveTlas)
#define CGPU_RESOLVE_BIND_SET(CTX, HANDLE, VAR_NAME)       CGPU_RESOLVE_OR_EXIT(CTX, HANDLE, VAR_NAME, CgpuIBindSet, cgpuResolveBindSet)

  /* Helper methods */

  const static MTL::ResourceOptions CGPU_DEFAULT_RESOURCE_OPTIONS =
    MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeUntracked;

  static MTL::Stages cgpuTranslatePipelineStages(CgpuPipelineStage stages)
  {
    MTL::Stages mtlStages = 0;

    if (bool(stages & CgpuPipelineStage::ComputeShader) ||
        bool(stages & CgpuPipelineStage::RayTracingShader))
    {
      mtlStages |= MTL::StageDispatch;
    }
    if (bool(stages & CgpuPipelineStage::Transfer) ||
        bool(stages & CgpuPipelineStage::Host))
    {
      mtlStages |= MTL::StageBlit;
    }
    if (bool(stages & CgpuPipelineStage::AccelerationStructureBuild))
    {
      mtlStages |= MTL::StageAccelerationStructure;
    }

    return mtlStages;
  }

  static MTL::TextureUsage cgpuTranslateImageUsage(CgpuImageUsage usage)
  {
    MTL::TextureUsage mtlUsage = MTL::TextureUsageUnknown;

    if (bool(usage & CgpuImageUsage::Sampled))
    {
      mtlUsage |= MTL::TextureUsageShaderRead;
    }
    if (bool(usage & CgpuImageUsage::Storage))
    {
      mtlUsage |= MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite;
    }

    return mtlUsage;
  }

  static MTL::PixelFormat cgpuTranslateImageFormat(CgpuImageFormat format)
  {
    switch (format)
    {
    case CgpuImageFormat::R8G8B8A8Unorm:
      return MTL::PixelFormatRGBA8Unorm;
    case CgpuImageFormat::R16G16B16A16Sfloat:
      return MTL::PixelFormatRGBA16Float;
    case CgpuImageFormat::R32Sfloat:
      return MTL::PixelFormatR32Float;
    default:
      CGPU_FATAL("unhandled image format");
    }
  }

  static uint32_t cgpuGetImageFormatBpp(CgpuImageFormat format)
  {
    switch (format)
    {
    case CgpuImageFormat::R8G8B8A8Unorm:
    case CgpuImageFormat::R32Sfloat:
      return 4;
    case CgpuImageFormat::R16G16B16A16Sfloat:
      return 8;
    default:
      CGPU_FATAL("unhandled image format");
    }
  }

  static MTL::SamplerAddressMode cgpuTranslateAddressMode(CgpuSamplerAddressMode mode)
  {
    switch (mode)
    {
    case CgpuSamplerAddressMode::ClampToEdge:
      return MTL::SamplerAddressModeClampToEdge;
    case CgpuSamplerAddressMode::Repeat:
      return MTL::SamplerAddressModeRepeat;
    case CgpuSamplerAddressMode::MirrorRepeat:
      return MTL::SamplerAddressModeMirrorRepeat;
    case CgpuSamplerAddressMode::ClampToBlack:
      return MTL::SamplerAddressModeClampToBorderColor;
    default:
      CGPU_FATAL("sampler address mode not handled");
    }
  };

  template<typename T>
  static T cgpuAlign(T value, T alignment)
  {
    return (value + (alignment - 1)) & ~(alignment - 1);
  }

  static NS::String* cgpuMakeShaderEntryPointName(CgpuIDevice* idevice)
  {
    std::string name = GB_FMT("main{}", idevice->uniqueShaderEntryPointCounter++);
    return NS::String::alloc()->init(name.c_str(), NS::UTF8StringEncoding);
  }

  static MTL::ResidencySet* cgpuCreateResidencySet(MTL::Device* device, uint32_t initialCapacity)
  {
    auto* desc = MTL::ResidencySetDescriptor::alloc()->init();
    desc->setInitialCapacity(initialCapacity);

    NS::Error* error = nullptr;
    MTL::ResidencySet* set = device->newResidencySet(desc, &error);
    CGPU_CHK(set, error);

    desc->release();
    return set;
  }

  /* Implementation */

  static bool cgpuCreateIDevice(CgpuIDevice* idevice)
  {
    MTL::Device* mtlDevice = MTL::CreateSystemDefaultDevice();
    if (!mtlDevice)
    {
      CGPU_RETURN_ERROR("failed to create system default device");
    }

    if (!mtlDevice->supportsFamily(MTL::GPUFamilyApple9)) // needed for buffer-based AS builds
    {
      CGPU_RETURN_ERROR("GPU not supported (too old)");
    }
    if (!mtlDevice->supportsRaytracing())
    {
      CGPU_RETURN_ERROR("ray tracing not supported");
    }
    if (!mtlDevice->supportsShaderBarycentricCoordinates())
    {
      CGPU_RETURN_ERROR("barycentric coordinates not supported");
    }
    if (mtlDevice->argumentBuffersSupport() != MTL::ArgumentBuffersTier2)
    {
      CGPU_RETURN_ERROR("tier 2 argument buffers not supported");
    }
    if (!mtlDevice->hasUnifiedMemory())
    {
      CGPU_RETURN_ERROR("UMA not supported");
    }
    if (!mtlDevice->supportsFunctionPointers())
    {
      CGPU_RETURN_ERROR("function pointers not supported");
    }
    if (!mtlDevice->readWriteTextureSupport())
    {
      CGPU_RETURN_ERROR("R/W textures not supported");
    }

    GB_LOG("GPU properties:");
    GB_LOG("> name: {}", mtlDevice->name()->utf8String());
    GB_LOG("> architecure: {}", mtlDevice->architecture()->name()->utf8String());
    GB_LOG("> registryID: {}", mtlDevice->registryID());

    if (mtlDevice->isLowPower())
    {
      GB_WARN("GPU is in low power mode");
    }

    mtlDevice->setShouldMaximizeConcurrentCompilation(true); // many hit shaders

    MTL4::CommandQueue* commandQueue = mtlDevice->newMTL4CommandQueue();
    CGPU_CHK_NP(commandQueue);

#ifndef NDEBUG
    MTL::LogState* logState = nullptr;
    {
      auto* desc = MTL::LogStateDescriptor::alloc()->init();
      desc->setLevel(MTL::LogLevelDebug);

      NS::Error* error = nullptr;
      logState = mtlDevice->newLogState(desc, &error);
      desc->release();

      auto logHandler = [](NS::String* subsystem, NS::String* category, MTL::LogLevel logLevel, NS::String* message) {
        if (logLevel == MTL::LogLevelError || logLevel == MTL::LogLevelFault)
        {
          GB_ERROR("[MTL] {}", message->utf8String());
        }
        else
        {
          GB_LOG("[MTL] {}", message->utf8String());
        }

        if (logLevel == MTL::LogLevelFault)
        {
          gbLogFlush();
          exit(EXIT_FAILURE);
        }
      };

      if (logState)
      {
        logState->addLogHandler(logHandler);
      }
      else
      {
        CGPU_LOG_MTL_ERR(error);
      }
    }

    MTL4::CommitOptions* commitOptions = MTL4::CommitOptions::alloc()->init();

    commitOptions->addFeedbackHandler([](MTL4::CommitFeedback* feedback) {
      NS::Error* error = feedback->error();
      CGPU_LOG_MTL_ERR(error);
    });
#endif

    MTL4::Compiler* compiler;
    {
      auto* desc = MTL4::CompilerDescriptor::alloc()->init();

      NS::Error* error = nullptr;
      compiler = mtlDevice->newCompiler(desc, &error);
      CGPU_CHK(compiler, error);

      desc->release();
    }

    uint32_t initialCapacity = 1024;
    MTL::ResidencySet* residencySet = cgpuCreateResidencySet(mtlDevice, initialCapacity);

    auto* commandBufferOptions = MTL4::CommandBufferOptions::alloc()->init();
#ifndef NDEBUG
    commandBufferOptions->setLogState(logState);
#endif

    idevice->compiler = compiler;
    idevice->device = mtlDevice;
    idevice->commandQueue = commandQueue;
#ifndef NDEBUG
    idevice->logState = logState;
    idevice->commitOptions = commitOptions;
#endif
    idevice->uniqueShaderEntryPointCounter = 0;
    idevice->residencySet = residencySet;
    idevice->commandBufferOptions = commandBufferOptions;

    return true;
  }

  static void cgpuDestroyIDevice(CgpuIDevice* idevice)
  {
    idevice->commandBufferOptions->release();
    idevice->residencySet->release();
#ifndef NDEBUG
    idevice->commitOptions->release();
    idevice->logState->release();
#endif
    idevice->commandQueue->release();
    idevice->device->release();
    idevice->compiler->release();
  }

  CgpuContext* cgpuCreateContext(const char* appName, uint32_t versionMajor, uint32_t versionMinor, uint32_t versionPatch)
  {
    CgpuIDevice idevice;
    if (!cgpuCreateIDevice(&idevice))
    {
      return nullptr;
    }

    spvc_context spvc;
    if (spvc_result r = spvc_context_create(&spvc); r != SPVC_SUCCESS)
    {
      CGPU_FATAL("failed to init SPIRV-Cross");
    }

    spvc_context_set_error_callback(spvc, [](void *userData, const char *error) {
      GB_ERROR("[SPVC] {}", error);
    }, nullptr);

#ifdef CGPU_MTL_CAPTURE_ENABLED
    MTL::CaptureManager* captureManager = MTL::CaptureManager::sharedCaptureManager();
    {
      auto* desc = MTL::CaptureDescriptor::alloc()->init();
      desc->setCaptureObject(idevice.device);

      NS::Error* error = nullptr;
      captureManager->startCapture(desc, &error);
      if (error)
      {
        CGPU_LOG_MTL_ERR(error);
      }

      desc->release();
    }
#endif

    CgpuContext* ctx = new CgpuContext {
#ifdef CGPU_MTL_CAPTURE_ENABLED
      .captureManager = captureManager,
#endif
      .idevice = idevice,
      .spvc = spvc
    };

    return ctx;
  }

  void cgpuDestroyContext(CgpuContext* ctx)
  {
#ifdef CGPU_MTL_CAPTURE_ENABLED
    ctx->captureManager->stopCapture();
#endif

    cgpuDestroyIDevice(&ctx->idevice);

    spvc_context_destroy(ctx->spvc);

    delete ctx;
  }

  static bool cgpuCreateShader(CgpuContext* ctx,
                               const CgpuShaderCreateInfo& createInfo,
                               CgpuIShader* ishader)
  {
    CgpuIDevice* idevice = &ctx->idevice;

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
      CHK_SPVC(spvc_context_parse_spirv(ctx->spvc, (const SpvId*) createInfo.source, createInfo.size / sizeof(SpvId), &ir));
      CHK_SPVC(spvc_context_create_compiler(ctx->spvc, SPVC_BACKEND_MSL, ir, SPVC_CAPTURE_MODE_TAKE_OWNERSHIP, &spvcCompiler));

      SpvExecutionModel execModel = spvc_compiler_get_execution_model(spvcCompiler);
      CHK_SPVC(spvc_compiler_rename_entry_point(spvcCompiler, CGPU_SHADER_ENTRY_POINT, ishader->entryPointName->utf8String(), execModel));

      spvc_compiler_options spvcCompilerOptions;
      CHK_SPVC(spvc_compiler_create_compiler_options(spvcCompiler, &spvcCompilerOptions));
      CHK_SPVC(spvc_compiler_options_set_uint(spvcCompilerOptions, SPVC_COMPILER_OPTION_FLIP_VERTEX_Y, 1));
      CHK_SPVC(spvc_compiler_options_set_uint(spvcCompilerOptions, SPVC_COMPILER_OPTION_MSL_PLATFORM, SPVC_MSL_PLATFORM_MACOS));
      CHK_SPVC(spvc_compiler_options_set_uint(spvcCompilerOptions, SPVC_COMPILER_OPTION_MSL_VERSION, SPVC_MSL_VERSION));
      CHK_SPVC(spvc_compiler_options_set_uint(spvcCompilerOptions, SPVC_COMPILER_OPTION_MSL_ARGUMENT_BUFFERS, 1));
      CHK_SPVC(spvc_compiler_options_set_uint(spvcCompilerOptions, SPVC_COMPILER_OPTION_MSL_ARGUMENT_BUFFERS_TIER, 2));
      CHK_SPVC(spvc_compiler_options_set_uint(spvcCompilerOptions, SPVC_COMPILER_OPTION_MSL_FORCE_ACTIVE_ARGUMENT_BUFFER_RESOURCES, 1)); // preserve descriptor ABI
      CHK_SPVC(spvc_compiler_options_set_uint(spvcCompilerOptions, SPVC_COMPILER_OPTION_MSL_ENABLE_DECORATION_BINDING, 1)); // keep original resource binding indices
      CHK_SPVC(spvc_compiler_options_set_uint(spvcCompilerOptions, SPVC_COMPILER_OPTION_MSL_FORCE_NATIVE_ARRAYS, 1)); // otherwise can't access array in ray payload
      CHK_SPVC(spvc_compiler_install_compiler_options(spvcCompiler, spvcCompilerOptions));
    }

    const char* mslSrc; // owned by context
    CHK_SPVC(spvc_compiler_compile(spvcCompiler, &mslSrc));
#undef CHK_SPVC

    if (getenv("GTL_DUMP_MSL"))
    {
      GB_LOG("{}", mslSrc);
    }

    MTL::CompileOptions* compileOptions = MTL::CompileOptions::alloc()->init();
    compileOptions->setLanguageVersion(MTL::LanguageVersion4_0);
#ifndef NDEBUG
    compileOptions->setEnableLogging(true);
#endif

    auto* mslStr = NS::String::alloc()->init(mslSrc, NS::UTF8StringEncoding);

    auto libDesc = MTL4::LibraryDescriptor::alloc()->init();
    if (createInfo.debugName)
    {
      auto* debugName = NS::String::alloc()->init(createInfo.debugName, NS::StringEncoding::UTF8StringEncoding);
      libDesc->setName(debugName);
      debugName->release();
    }
    libDesc->setOptions(compileOptions);
    libDesc->setSource(mslStr);

    // We use the async code path as the synchronous one returns a corrupt error object
    std::mutex mutex;
    std::condition_variable cv;
    bool done = false;

    MTL4::CompilerTask* task = idevice->compiler->newLibrary(libDesc, [&](MTL::Library* result, NS::Error* error) {
      if (result)
      {
        std::unique_lock<std::mutex> lock(mutex);
        ishader->library = result;
        result->retain();
      }
      else
      {
        GB_LOG("{}", mslSrc);
        gbLogFlush();
        CGPU_LOG_MTL_ERR(error);
      }
      done = true;
      cv.notify_one();
    });
    CGPU_CHK_NP(task);

    {
      std::unique_lock<std::mutex> lock(mutex);
      cv.wait(lock, [&]() { return done; });
    }

    mslStr->release();
    task->release();
    libDesc->release();
    compileOptions->release();

    return bool(ishader->library);
  }

  bool cgpuCreateShader(CgpuContext* ctx,
                        CgpuShaderCreateInfo createInfo,
                        CgpuShader* shader)
  {
    CgpuIDevice* idevice = &ctx->idevice;

    shader->handle = ctx->ishaderStore.allocate();

    CGPU_RESOLVE_SHADER(ctx, *shader, ishader);

    ishader->entryPointName = cgpuMakeShaderEntryPointName(idevice);

    if (cgpuCreateShader(ctx, createInfo, ishader))
    {
      return true;
    }

    ishader->entryPointName->release();
    ctx->ishaderStore.free(shader->handle);
    return false;
  }

  bool cgpuCreateShadersParallel(CgpuContext* ctx,
                         uint32_t shaderCount,
                         CgpuShaderCreateInfo* createInfos,
                         CgpuShader* shaders)
  {
    CgpuIDevice* idevice = &ctx->idevice;
    std::atomic<bool> success = true;

    for (uint32_t i = 0; i < shaderCount; i++)
    {
      shaders[i].handle = ctx->ishaderStore.allocate();
    }

    std::vector<CgpuIShader*> ishaders;
    ishaders.resize(shaderCount, nullptr);

    for (uint32_t i = 0; i < shaderCount; i++)
    {
      CGPU_RESOLVE_SHADER(ctx, shaders[i], ishader);
      ishader->library = nullptr;
      ishader->entryPointName = cgpuMakeShaderEntryPointName(idevice);
      ishaders[i] = ishader;
    }

#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < int(shaderCount); i++)
    {
      if (!cgpuCreateShader(ctx, createInfos[i], ishaders[i]))
      {
        success = false;
      }
    }

    if (success)
    {
      return true;
    }

    for (uint32_t i = 0; i < shaderCount; i++)
    {
      if (ishaders[i]->library)
      {
        ishaders[i]->library->release();
      }
      ishaders[i]->entryPointName->release();
      ctx->ishaderStore.free(shaders[i].handle);
    }

    return false;
  }

  void cgpuDestroyShader(CgpuContext* ctx, CgpuShader shader)
  {
    CGPU_RESOLVE_SHADER(ctx, shader, ishader);

    ishader->library->release();
    ishader->entryPointName->release();

    ctx->ishaderStore.free(shader.handle);
  }

  bool cgpuCreateBuffer(CgpuContext* ctx,
                        CgpuBufferCreateInfo createInfo,
                        CgpuBuffer* buffer)
  {
    CgpuIDevice* idevice = &ctx->idevice;

    uint64_t handle = ctx->ibufferStore.allocate();

    CGPU_RESOLVE_BUFFER(ctx, { handle }, ibuffer);

    constexpr static uint64_t BASE_ALIGNMENT = 32; // size of largest math primitive (vec4); ensure that
                                                   // compiler can emit wide loads.
    uint64_t size = cgpuAlign(createInfo.size, BASE_ALIGNMENT);
    assert(size > 0);

    MTL::Buffer* mtlBuffer = idevice->device->newBuffer(size, CGPU_DEFAULT_RESOURCE_OPTIONS);
    if (!mtlBuffer)
    {
      ctx->ibufferStore.free(handle);
      CGPU_RETURN_ERROR("failed to create buffer");
    }

    if (createInfo.debugName)
    {
      auto* debugName = NS::String::alloc()->init(createInfo.debugName, NS::StringEncoding::UTF8StringEncoding);
      mtlBuffer->setLabel(debugName);
      debugName->release();
    }

    if (bool(createInfo.usage & CgpuBufferUsage::ShaderDeviceAddress))
    {
      idevice->residencySet->addAllocation(mtlBuffer);
    }

    ibuffer->size = size;
    ibuffer->buffer = mtlBuffer;
    ibuffer->isDynamic = bool(createInfo.usage & CgpuBufferUsage::Uniform);

    buffer->handle = handle;
    return true;
  }

  void cgpuDestroyBuffer(CgpuContext* ctx, CgpuBuffer buffer)
  {
    CgpuIDevice* idevice = &ctx->idevice;
    CGPU_RESOLVE_BUFFER(ctx, buffer, ibuffer);

    if (idevice->residencySet->containsAllocation(ibuffer->buffer))
    {
      idevice->residencySet->removeAllocation(ibuffer->buffer);
    }
    ibuffer->buffer->release();

    ctx->ibufferStore.free(buffer.handle);
  }

  void* cgpuGetBufferCpuPtr(CgpuContext* ctx, CgpuBuffer buffer)
  {
    CGPU_RESOLVE_BUFFER(ctx, buffer, ibuffer);

    return ibuffer->buffer->contents();
  }

  uint64_t cgpuGetBufferGpuAddress(CgpuContext* ctx, CgpuBuffer buffer)
  {
    CGPU_RESOLVE_BUFFER(ctx, buffer, ibuffer);

    return ibuffer->buffer->gpuAddress();
  }

  bool cgpuCreateImage(CgpuContext* ctx,
                       CgpuImageCreateInfo createInfo,
                       CgpuImage* image)
  {
    CgpuIDevice* idevice = &ctx->idevice;

    uint64_t handle = ctx->iimageStore.allocate();

    CGPU_RESOLVE_IMAGE(ctx, { handle }, iimage);

    MTL::TextureUsage usage = cgpuTranslateImageUsage(createInfo.usage);
    MTL::PixelFormat pixelFormat = cgpuTranslateImageFormat(createInfo.format);

    MTL::TextureDescriptor* descriptor = MTL::TextureDescriptor::alloc()->init();
    descriptor->setTextureType(createInfo.is3d ? MTL::TextureType3D : MTL::TextureType2D);
    descriptor->setPixelFormat(pixelFormat);
    descriptor->setWidth(createInfo.width);
    descriptor->setHeight(createInfo.height);
    descriptor->setDepth(createInfo.depth);
    descriptor->setUsage(usage);
    descriptor->setStorageMode(MTL::StorageModeShared);
    descriptor->setAllowGPUOptimizedContents(true);

    MTL::Texture* texture = idevice->device->newTexture(descriptor);

    descriptor->release();

    if (!texture)
    {
      ctx->iimageStore.free(handle);
      CGPU_RETURN_ERROR("failed to create image");
    }

    if (createInfo.debugName)
    {
      auto* debugName = NS::String::alloc()->init(createInfo.debugName, NS::StringEncoding::UTF8StringEncoding);
      texture->setLabel(debugName);
      debugName->release();
    }

    iimage->texture = texture;
    iimage->width = createInfo.width;
    iimage->height = createInfo.height;
    iimage->depth = createInfo.is3d ? createInfo.depth : 1;
    iimage->format = createInfo.format;

    image->handle = handle;
    return true;
  }

  void cgpuDestroyImage(CgpuContext* ctx, CgpuImage image)
  {
    CGPU_RESOLVE_IMAGE(ctx, image, iimage);

    iimage->texture->release();

    ctx->iimageStore.free(image.handle);
  }

  bool cgpuCreateSampler(CgpuContext* ctx,
                         CgpuSamplerCreateInfo createInfo,
                         CgpuSampler* sampler)
  {
    CgpuIDevice* idevice = &ctx->idevice;

    uint64_t handle = ctx->isamplerStore.allocate();

    CGPU_RESOLVE_SAMPLER(ctx, { handle }, isampler);

    auto* descriptor = MTL::SamplerDescriptor::alloc()->init();

    descriptor->setSAddressMode(cgpuTranslateAddressMode(createInfo.addressModeU));
    descriptor->setTAddressMode(cgpuTranslateAddressMode(createInfo.addressModeV));
    descriptor->setRAddressMode(cgpuTranslateAddressMode(createInfo.addressModeW));
    descriptor->setMinFilter(MTL::SamplerMinMagFilterLinear);
    descriptor->setMagFilter(MTL::SamplerMinMagFilterLinear);
    descriptor->setBorderColor(MTL::SamplerBorderColorOpaqueBlack);
    descriptor->setNormalizedCoordinates(true);
    descriptor->setSupportArgumentBuffers(true);

    MTL::SamplerState* mtlSampler = idevice->device->newSamplerState(descriptor);

    descriptor->release();

    if (!mtlSampler)
    {
      ctx->isamplerStore.free(handle);
      CGPU_RETURN_ERROR("failed to create sampler");
    }

    isampler->sampler = mtlSampler;

    sampler->handle = handle;
    return true;
  }

  void cgpuDestroySampler(CgpuContext* ctx, CgpuSampler sampler)
  {
    CGPU_RESOLVE_SAMPLER(ctx, sampler, isampler);

    isampler->sampler->release();

    ctx->isamplerStore.free(sampler.handle);
  }

  static void cgpuCreateComputePipeline(CgpuContext* ctx,
                                        CgpuIShader* ishader,
                                        const char* debugName,
                                        CgpuPipeline* pipeline,
                                        MTL::Size threadsPerGroup,
                                        const MTL::LinkedFunctions* linkedFunctions = nullptr)
  {
    CgpuIDevice* idevice = &ctx->idevice;

    uint64_t handle = ctx->ipipelineStore.allocate();

    CGPU_RESOLVE_PIPELINE(ctx, { handle }, ipipeline);

    NS::Error* error = nullptr;

    auto* funDesc = MTL::FunctionDescriptor::alloc()->init();
    funDesc->setName(ishader->entryPointName);
    funDesc->setOptions(MTL::FunctionOptionPipelineIndependent | MTL::FunctionOptionCompileToBinary);

    MTL::Function* entryFun = ishader->library->newFunction(funDesc, &error);
    CGPU_CHK(entryFun, error);
    funDesc->release();

    auto* descriptor = MTL::ComputePipelineDescriptor::alloc()->init();
#ifndef NDEBUG
    descriptor->setShaderValidation(MTL::ShaderValidationEnabled);
#endif
    if (linkedFunctions)
    {
      descriptor->setLinkedFunctions(linkedFunctions);
    }
    descriptor->setComputeFunction(entryFun);

    MTL::PipelineOption pipelineOptions = MTL::PipelineOptionNone;
    MTL::ComputePipelineState* state = idevice->device->newComputePipelineState(descriptor, pipelineOptions, nullptr, &error);
    CGPU_CHK(state, error);

    descriptor->release();
    entryFun->release();

    const CgpuShaderReflection& reflection = ishader->reflection;

    MTL4::ArgumentTable* argumentTable;
    {
      uint32_t argumentBufferCount = SPVC_MSL_RT_FUNCTION_TABLES_BUFFER_INDEX +
                                     reflection.payloadCount * 3/*ift, missVft, chitVft*/;
      assert(argumentBufferCount < CGPU_MAX_ARGUMENT_BUFFER_COUNT);

      auto* desc = MTL4::ArgumentTableDescriptor::alloc()->init();
      desc->setMaxBufferBindCount(argumentBufferCount);
      desc->setMaxSamplerStateBindCount(0);
      desc->setMaxTextureBindCount(0);

      NS::Error* error = nullptr;
      argumentTable = idevice->device->newArgumentTable(desc, &error);
      CGPU_CHK(argumentTable, error);
      desc->release();
    }

    ipipeline->state = state;
    ipipeline->threadsPerGroup = threadsPerGroup;
    ipipeline->argumentTable = argumentTable;
    ipipeline->computeReflection = reflection;

    pipeline->handle = handle;
  }

  void cgpuCreateComputePipeline(CgpuContext* ctx,
                                 CgpuComputePipelineCreateInfo createInfo,
                                 CgpuPipeline* pipeline)
  {
    CGPU_RESOLVE_SHADER(ctx, createInfo.shader, ishader);

    const CgpuShaderReflection& reflection = ishader->reflection;

    MTL::Size threadsPerGroup(reflection.workgroupSize[0], reflection.workgroupSize[1], reflection.workgroupSize[2]);

    cgpuCreateComputePipeline(ctx, ishader, createInfo.debugName, pipeline, threadsPerGroup);
  }

  static void cgpuCreatRtFunctionTables(CgpuRtFunctionTables& tables,
                                        MTL::ComputePipelineState* pipeline,
                                        const std::vector<MTL::Function*>& missFunctions,
                                        const std::vector<MTL::Function*>& chitFunctions,
                                        const std::vector<MTL::Function*>& ahitFunctions,
                                        uint32_t payloadStride,
                                        uint32_t payloadOffset)
  {
    auto missFunctionCount = uint32_t(missFunctions.size() / payloadStride);
    auto hitFunctionCount = uint32_t(ahitFunctions.size() / payloadStride);
    assert(ahitFunctions.size() == chitFunctions.size());

    MTL::IntersectionFunctionTable* ift;
    {
      auto* desc = MTL::IntersectionFunctionTableDescriptor::alloc()->init();
      desc->setFunctionCount(hitFunctionCount);
      ift = pipeline->newIntersectionFunctionTable(desc);
      desc->release();
    }

    MTL::VisibleFunctionTable* missVft;
    {
      auto* desc = MTL::VisibleFunctionTableDescriptor::alloc()->init();
      desc->setFunctionCount(missFunctionCount);
      missVft = pipeline->newVisibleFunctionTable(desc);
      desc->release();
    }

    MTL::VisibleFunctionTable* chitVft;
    {
      auto* desc = MTL::VisibleFunctionTableDescriptor::alloc()->init();
      desc->setFunctionCount(hitFunctionCount);
      chitVft = pipeline->newVisibleFunctionTable(desc);
      desc->release();
    }

    for (uint32_t i = 0; i < uint32_t(missFunctions.size()); i++)
    {
      if ((i % payloadStride) != payloadOffset)
      {
        continue;
      }

      MTL::Function* fun = missFunctions[i];
      MTL::FunctionHandle* funHandle = pipeline->functionHandle(fun);
      CGPU_CHK_NP(funHandle);

      missVft->setFunction(funHandle, i / payloadStride);
    }

    for (uint32_t i = 0; i < uint32_t(chitFunctions.size()); i++)
    {
      if ((i % payloadStride) != payloadOffset)
      {
        continue;
      }

      MTL::Function* cfun = chitFunctions[i];

      if (cfun)
      {
        MTL::FunctionHandle* funHandle = pipeline->functionHandle(cfun);
        CGPU_CHK_NP(funHandle);

        chitVft->setFunction(funHandle, i / payloadStride);
      }

      MTL::Function* afun = ahitFunctions[i];

      if (afun)
      {
        MTL::FunctionHandle* funHandle = pipeline->functionHandle(afun);
        CGPU_CHK_NP(funHandle);

        ift->setFunction(funHandle, i / payloadStride);
      }
    }

    tables.ift = ift;
    tables.missVft = missVft;
    tables.chitVft = chitVft;
  }

  static void cgpuDestroyRtFunctionTables(CgpuRtFunctionTables& tables)
  {
    tables.chitVft->release();
    tables.missVft->release();
    tables.ift->release();
  }

  void cgpuCreateRtPipeline(CgpuContext* ctx,
                            CgpuRtPipelineCreateInfo createInfo,
                            CgpuPipeline* pipeline)
  {
    CGPU_RESOLVE_SHADER(ctx, createInfo.rgenShader, irgenShader);

    std::vector<MTL::Function*> missFunctions;
    std::vector<MTL::Function*> chitFunctions;
    std::vector<MTL::Function*> ahitFunctions;
    std::vector<MTL::Function*> linkedFunctions;

    chitFunctions.resize(createInfo.hitGroupCount, nullptr);
    ahitFunctions.resize(createInfo.hitGroupCount, nullptr);

    for (uint32_t i = 0; i < createInfo.hitGroupCount; i++)
    {
      NS::Error* error = nullptr;
      CgpuShader chitShader = createInfo.hitGroups[i].closestHitShader;
      CgpuShader ahitShader = createInfo.hitGroups[i].anyHitShader;

      if (chitShader.handle)
      {
        CGPU_RESOLVE_SHADER(ctx, chitShader, ishader);

        MTL::FunctionDescriptor* funDesc = MTL::FunctionDescriptor::alloc()->init();
        funDesc->setName(ishader->entryPointName);
        funDesc->setOptions(MTL::FunctionOptionPipelineIndependent | MTL::FunctionOptionCompileToBinary);

        MTL::Function* fun = ishader->library->newFunction(funDesc, &error);
        CGPU_CHK(fun, error);
        funDesc->release();

        chitFunctions[i] = fun;
        linkedFunctions.push_back(fun);
      }

      if (ahitShader.handle)
      {
        CGPU_RESOLVE_SHADER(ctx, ahitShader, ishader);

        MTL::IntersectionFunctionDescriptor* funDesc = MTL::IntersectionFunctionDescriptor::alloc()->init();
        funDesc->setName(ishader->entryPointName);
        funDesc->setOptions(MTL::FunctionOptionPipelineIndependent | MTL::FunctionOptionCompileToBinary);

        MTL::Function* fun = ishader->library->newIntersectionFunction(funDesc, &error);
        CGPU_CHK(fun, error);
        funDesc->release();

        ahitFunctions[i] = fun;
        linkedFunctions.push_back(fun);
      }
    }

    for (uint32_t i = 0; i < createInfo.missShaderCount; i++)
    {
      CgpuShader shader = createInfo.missShaders[i];
      CGPU_RESOLVE_SHADER(ctx, shader, ishader);

      auto* funDesc = MTL::FunctionDescriptor::alloc()->init();
      funDesc->setName(ishader->entryPointName);
      funDesc->setOptions(MTL::FunctionOptionPipelineIndependent | MTL::FunctionOptionCompileToBinary);

      NS::Error* error = nullptr;
      MTL::Function* fun = ishader->library->newFunction(funDesc, &error);
      CGPU_CHK(fun, error);
      funDesc->release();

      missFunctions.push_back(fun);
      linkedFunctions.push_back(fun);
    }

    auto* linkingDescriptor = MTL::LinkedFunctions::alloc()->init();

    NS::Array* lfs = NS::Array::alloc()->init((const NS::Object* const*) linkedFunctions.data(), (uint32_t) linkedFunctions.size());
    linkingDescriptor->setFunctions(lfs);

    MTL::Size threadsPerGroup(8, 4, 1); // assuming that 32 threads is best

    cgpuCreateComputePipeline(ctx, irgenShader, createInfo.debugName, pipeline, threadsPerGroup, linkingDescriptor);

    linkingDescriptor->release();
    lfs->release();

    CGPU_RESOLVE_PIPELINE(ctx, *pipeline, ipipeline);

    const CgpuShaderReflection& reflection = ipipeline->computeReflection;

    std::vector<CgpuRtFunctionTables> fts(reflection.payloadCount);
    for (uint32_t i = 0; i < reflection.payloadCount; i++)
    {
      cgpuCreatRtFunctionTables(fts[i], ipipeline->state, missFunctions, chitFunctions, ahitFunctions, createInfo.payloadStride, i);
    }

    for (MTL::Function* f : linkedFunctions)
    {
      if (f) f->release();
    }

    uint32_t argBufIdx = SPVC_MSL_RT_FUNCTION_TABLES_BUFFER_INDEX;
    for (const CgpuRtFunctionTables& ft : fts)
    {
      ipipeline->argumentTable->setResource(ft.ift->gpuResourceID(), argBufIdx++);
      ipipeline->argumentTable->setResource(ft.missVft->gpuResourceID(), argBufIdx++);
      ipipeline->argumentTable->setResource(ft.chitVft->gpuResourceID(), argBufIdx++);
    }

    ipipeline->fts = fts;
  }

  void cgpuDestroyPipeline(CgpuContext* ctx, CgpuPipeline pipeline)
  {
    CGPU_RESOLVE_PIPELINE(ctx, pipeline, ipipeline);

    for (CgpuRtFunctionTables& fts : ipipeline->fts)
    {
      cgpuDestroyRtFunctionTables(fts);
    }
    ipipeline->argumentTable->release();
    ipipeline->state->release();

    ctx->ipipelineStore.free(pipeline.handle);
  }

  bool cgpuCreateBlas(CgpuContext* ctx,
                      CgpuBlasCreateInfo createInfo,
                      CgpuBlas* blas)
  {
    CgpuIDevice* idevice = &ctx->idevice;

    CGPU_RESOLVE_BUFFER(ctx, createInfo.vertexPosBuffer, ivertexBuffer);
    CGPU_RESOLVE_BUFFER(ctx, createInfo.indexBuffer, iindexBuffer);

    uint64_t handle = ctx->iblasStore.allocate();

    CGPU_RESOLVE_BLAS(ctx, { handle }, iblas);

    auto vertexBufferRange = MTL4::BufferRange::Make(ivertexBuffer->buffer->gpuAddress(), ivertexBuffer->size);
    auto indexBufferRange = MTL4::BufferRange::Make(iindexBuffer->buffer->gpuAddress(), iindexBuffer->size);

    auto* triDesc = MTL4::AccelerationStructureTriangleGeometryDescriptor::alloc()->init();
    triDesc->setVertexBuffer(vertexBufferRange);
    triDesc->setVertexFormat(MTL::AttributeFormatFloat3);
    triDesc->setIndexBuffer(indexBufferRange);
    triDesc->setIndexType(MTL::IndexTypeUInt32);
    triDesc->setTriangleCount(createInfo.triangleCount);
    triDesc->setOpaque(createInfo.isOpaque);
    triDesc->setAllowDuplicateIntersectionFunctionInvocation(false); // on primitive

    auto* blasDesc = MTL4::PrimitiveAccelerationStructureDescriptor::alloc()->init();
    CGPU_CHK_NP(blasDesc);
    blasDesc->setUsage(MTL::AccelerationStructureUsagePreferFastIntersection);

    NS::Array* geoDescs = NS::Array::alloc()->init((const NS::Object* const*) &triDesc, 1);
    blasDesc->setGeometryDescriptors(geoDescs);

    MTL::AccelerationStructureSizes sizes = idevice->device->accelerationStructureSizes(blasDesc);

    MTL::Buffer* scratchBuffer = idevice->device->newBuffer(sizes.buildScratchBufferSize, CGPU_DEFAULT_RESOURCE_OPTIONS);
    if (!scratchBuffer)
    {
      blasDesc->release();
      geoDescs->release();
      triDesc->release();
      CGPU_RETURN_ERROR("failed to allocate BLAS scratch buffer");
    }
    scratchBuffer->setLabel(MTLSTR("[AS scratch buffer]"));

    MTL::AccelerationStructure* as = idevice->device->newAccelerationStructure(sizes.accelerationStructureSize);
    if (!as)
    {
      blasDesc->release();
      geoDescs->release();
      triDesc->release();
      scratchBuffer->release();
      CGPU_RETURN_ERROR("failed to allocate BLAS");
    }

    MTL::SharedEvent* event = idevice->device->newSharedEvent();
    CGPU_CHK_NP(event);
    MTL4::CommandBuffer* commandBuffer = idevice->device->newCommandBuffer();
    CGPU_CHK_NP(commandBuffer);

    MTL::ResidencySet* residencySet = cgpuCreateResidencySet(idevice->device, 4);
    residencySet->addAllocation(ivertexBuffer->buffer);
    residencySet->addAllocation(iindexBuffer->buffer);
    residencySet->addAllocation(scratchBuffer);
    residencySet->addAllocation(as);
    residencySet->commit();

    auto* commandAllocator = idevice->device->newCommandAllocator();
    CGPU_CHK_NP(commandAllocator);
    commandBuffer->beginCommandBuffer(commandAllocator, idevice->commandBufferOptions);

    commandBuffer->useResidencySet(residencySet);

    MTL4::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();

    auto scratchBufferRange = MTL4::BufferRange::Make(scratchBuffer->gpuAddress(), sizes.buildScratchBufferSize);
    encoder->buildAccelerationStructure(as, blasDesc, scratchBufferRange);

    encoder->endEncoding();

    commandBuffer->endCommandBuffer();

    MTL4::CommandQueue* commandQueue = idevice->commandQueue;
    commandQueue->commit(&commandBuffer, 1, idevice->commitOptions);

    constexpr static uint32_t SIGNAL_VALUE = 42;
    commandQueue->signalEvent(event, SIGNAL_VALUE);
    event->waitUntilSignaledValue(SIGNAL_VALUE,  UINT64_MAX);

    event->release();
    commandBuffer->release();
    commandAllocator->release();

    scratchBuffer->release();
    blasDesc->release();
    geoDescs->release();
    triDesc->release();
    residencySet->release();

    if (createInfo.debugName)
    {
      auto* debugName = NS::String::alloc()->init(createInfo.debugName, NS::StringEncoding::UTF8StringEncoding);
      as->setLabel(debugName);
      debugName->release();
    }

    iblas->as = as;
    iblas->isOpaque = createInfo.isOpaque;

    blas->handle = handle;
    return true;
  }

  bool cgpuCreateTlas(CgpuContext* ctx,
                      CgpuTlasCreateInfo createInfo,
                      CgpuTlas* tlas)
  {
    CgpuIDevice* idevice = &ctx->idevice;

    uint64_t handle = ctx->itlasStore.allocate();

    CGPU_RESOLVE_TLAS(ctx, { handle }, itlas);

    std::unordered_set<const MTL::AccelerationStructure*> blases;
    blases.reserve(createInfo.instanceCount);

    // Upload instance buffer.
    uint64_t instanceBufferSize;
    MTL::Buffer* instanceBuffer;
    {
      std::vector<MTL::IndirectAccelerationStructureInstanceDescriptor> instances;

      for (uint32_t i = 0; i < createInfo.instanceCount; i++)
      {
        const CgpuBlasInstance& instance = createInfo.instances[i];

        CGPU_RESOLVE_BLAS(ctx, instance.as, iblas);
        blases.insert(iblas->as);

        MTL::AccelerationStructureInstanceOptions options = MTL::AccelerationStructureInstanceOptionDisableTriangleCulling;
        if (iblas->isOpaque)
        {
          options |= MTL::AccelerationStructureInstanceOptionOpaque;
        }
        else
        {
          options |= MTL::AccelerationStructureInstanceOptionNonOpaque;
        }

        uint32_t functionIndex = instance.hitGroupIndex / 2; // each hit group has 2 functions (chit and ahit)
        uint32_t userID = (functionIndex << 22) | instance.instanceCustomIndex;

        assert(functionIndex < (1 << 10)); // max 1024 materials
        assert(instance.instanceCustomIndex < (1 << 22));

        MTL::IndirectAccelerationStructureInstanceDescriptor d;
        d.options = options;
        d.mask = 0xFF;
        d.intersectionFunctionTableOffset = functionIndex;
        d.accelerationStructureID = iblas->as->gpuResourceID();
        d.userID = userID;
        memcpy(&d.transformationMatrix, instance.transform, sizeof(MTL::PackedFloat4x3));

        instances.push_back(d);
      }

      auto instanceCount = instances.empty() ? 1u : uint32_t(instances.size()); // prevent zero-alloc
      instanceBufferSize = sizeof(MTL::IndirectAccelerationStructureInstanceDescriptor) * instanceCount;

      instanceBuffer = idevice->device->newBuffer(instanceBufferSize, CGPU_DEFAULT_RESOURCE_OPTIONS);
      if (!instanceBuffer)
      {
        CGPU_RETURN_ERROR("failed to create TLAS instance buffer");
      }
      instanceBuffer->setLabel(MTLSTR("[TLAS instance buffer]"));

      if (!instances.empty())
      {
        memcpy(instanceBuffer->contents(), instances.data(), instanceBufferSize);
      }
    }

    auto instanceBufferRange = MTL4::BufferRange::Make(instanceBuffer->gpuAddress(), instanceBufferSize);

    auto* descriptor = MTL4::InstanceAccelerationStructureDescriptor::alloc()->init();
    descriptor->setUsage(MTL::AccelerationStructureUsagePreferFastIntersection);
    descriptor->setInstanceTransformationMatrixLayout(MTL::MatrixLayoutRowMajor);
    descriptor->setInstanceCount(createInfo.instanceCount);
    descriptor->setInstanceDescriptorBuffer(instanceBufferRange);
    descriptor->setInstanceDescriptorType(MTL::AccelerationStructureInstanceDescriptorTypeIndirect);

    // Build TLAS.
    MTL::AccelerationStructure* as;
    {
      MTL::AccelerationStructureSizes sizes = idevice->device->accelerationStructureSizes(descriptor);

      as = idevice->device->newAccelerationStructure(sizes.accelerationStructureSize);
      if (!as)
      {
        instanceBuffer->release();
        descriptor->release();
        CGPU_RETURN_ERROR("failed to create TLAS");
      }

      MTL::Buffer* scratchBuffer = idevice->device->newBuffer(sizes.buildScratchBufferSize, CGPU_DEFAULT_RESOURCE_OPTIONS);
      if (!scratchBuffer)
      {
        as->release();
        instanceBuffer->release();
        descriptor->release();
        CGPU_RETURN_ERROR("failed to create TLAS scratch buffer");
      }
      scratchBuffer->setLabel(MTLSTR("[TLAS scratch buffer]"));

      MTL::SharedEvent* event = idevice->device->newSharedEvent();
      CGPU_CHK_NP(event);
      MTL4::CommandBuffer* commandBuffer = idevice->device->newCommandBuffer();
      CGPU_CHK_NP(commandBuffer);

      MTL::ResidencySet* residencySet = cgpuCreateResidencySet(idevice->device, 3 + blases.size());
      residencySet->addAllocation(instanceBuffer);
      residencySet->addAllocation(scratchBuffer);
      residencySet->addAllocation(as);
      for (const MTL::AccelerationStructure* blas : blases)
      {
        residencySet->addAllocation(blas);
      }
      residencySet->commit();

      auto* commandAllocator = idevice->device->newCommandAllocator();
      CGPU_CHK_NP(commandAllocator);
      commandBuffer->beginCommandBuffer(commandAllocator, idevice->commandBufferOptions);

      commandBuffer->useResidencySet(residencySet);

      MTL4::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();

      auto scratchBufferRange = MTL4::BufferRange::Make(scratchBuffer->gpuAddress(), sizes.buildScratchBufferSize);
      encoder->buildAccelerationStructure(as, descriptor, scratchBufferRange);

      encoder->endEncoding();

      commandBuffer->endCommandBuffer();

      MTL4::CommandQueue* commandQueue = idevice->commandQueue;
      commandQueue->commit(&commandBuffer, 1, idevice->commitOptions);

      constexpr static uint32_t SIGNAL_VALUE = 42;
      commandQueue->signalEvent(event, SIGNAL_VALUE);
      event->waitUntilSignaledValue(SIGNAL_VALUE, UINT64_MAX);

      event->release();
      commandBuffer->release();
      commandAllocator->release();
      scratchBuffer->release();
      residencySet->release();
    }

    descriptor->release();
    instanceBuffer->release();

    if (createInfo.debugName)
    {
      auto* debugName = NS::String::alloc()->init(createInfo.debugName, NS::StringEncoding::UTF8StringEncoding);
      as->setLabel(debugName);
      debugName->release();
    }

    itlas->as = as;
    itlas->blases = blases;

    tlas->handle = handle;
    return true;
  }

  void cgpuDestroyBlas(CgpuContext* ctx, CgpuBlas blas)
  {
    CGPU_RESOLVE_BLAS(ctx, blas, iblas);

    iblas->as->release();

    ctx->iblasStore.free(blas.handle);
  }

  void cgpuDestroyTlas(CgpuContext* ctx, CgpuTlas tlas)
  {
    CGPU_RESOLVE_TLAS(ctx, tlas, itlas);

    itlas->as->release();

    ctx->itlasStore.free(tlas.handle);
  }

  void cgpuCreateBindSets(CgpuContext* ctx, CgpuPipeline pipeline, CgpuBindSet* bindSets, uint32_t bindSetCount)
  {
    CgpuIDevice* idevice = &ctx->idevice;
    CGPU_RESOLVE_PIPELINE(ctx, pipeline, ipipeline);

    const CgpuShaderReflection& reflection = ipipeline->computeReflection;
    assert(bindSetCount == reflection.descriptorSets.size());

    const auto collectArgumentDescriptors = [&](const CgpuShaderReflectionDescriptorSet& set)
    {
      std::vector<MTL::ArgumentDescriptor*> argumentDescriptors;
      argumentDescriptors.reserve(set.bindings.size());

      for (const CgpuShaderReflectionBinding& binding : set.bindings)
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
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
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

        MTL::TextureType textureType;
        if (descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ||
            descriptorType == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE ||
            descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
        {
          switch (binding.dim)
          {
          case 1:
            textureType = MTL::TextureType1D;
            break;
          case 2:
            textureType = MTL::TextureType2D;
            break;
          case 3:
            textureType = MTL::TextureType3D;
            break;
          default:
            CGPU_FATAL("unsupported image dimensions");
          }
        }

        auto* desc = MTL::ArgumentDescriptor::alloc()->init();
        desc->setDataType(dataType);
        desc->setIndex(binding.binding);
        desc->setAccess(access);
        desc->setArrayLength(binding.count);
        desc->setTextureType(textureType);

        argumentDescriptors.push_back(desc);
      }

      return argumentDescriptors;
    };

    for (uint32_t i = 0; i < bindSetCount; i++)
    {
      bindSets[i].handle = ctx->ibindSetStore.allocate();

      CGPU_RESOLVE_BIND_SET(ctx, bindSets[i], ibindSet);

      const CgpuShaderReflectionDescriptorSet& descriptorSet = reflection.descriptorSets[i];
      std::vector<MTL::ArgumentDescriptor*> argumentDescriptors = collectArgumentDescriptors(descriptorSet);

      NS::Array* descriptorArray = NS::Array::alloc()->init(
        (const NS::Object* const*) argumentDescriptors.data(),
        (uint32_t) argumentDescriptors.size()
      );

      MTL::ArgumentEncoder* argumentEncoder = idevice->device->newArgumentEncoder(descriptorArray);
      CGPU_CHK_NP(argumentEncoder);
      {
        auto* debugName = NS::String::alloc()->init(GB_FMT("[argument encoder {}]", i).c_str(), NS::StringEncoding::UTF8StringEncoding);
        argumentEncoder->setLabel(debugName);
        debugName->release();
      }

      descriptorArray->release();

      uint64_t argumentBufferSize = argumentEncoder->encodedLength();
      MTL::Buffer* argumentBuffer = idevice->device->newBuffer(argumentBufferSize, CGPU_DEFAULT_RESOURCE_OPTIONS);
      CGPU_CHK_NP(argumentBuffer);
      {
        auto* debugName = NS::String::alloc()->init(GB_FMT("[argument buffer {}]", i).c_str(), NS::StringEncoding::UTF8StringEncoding);
        argumentBuffer->setLabel(debugName);
        debugName->release();
      }

      uint32_t offset = 0;
      argumentEncoder->setArgumentBuffer(argumentBuffer, offset);

      for (MTL::ArgumentDescriptor* desc : argumentDescriptors)
      {
        desc->release();
      }

      auto initialCapacity = uint32_t(argumentDescriptors.size());
      MTL::ResidencySet* residencySet = cgpuCreateResidencySet(idevice->device, initialCapacity);

      ibindSet->argumentBuffer = argumentBuffer;
      ibindSet->argumentEncoder = argumentEncoder;
      ibindSet->residencySet = residencySet;
    }
  }

  void cgpuDestroyBindSets(CgpuContext* ctx, CgpuBindSet* bindSets, uint32_t bindSetCount)
  {
    for (uint32_t i = 0; i < bindSetCount; i++)
    {
      CGPU_RESOLVE_BIND_SET(ctx, bindSets[i], ibindSet);

      ibindSet->argumentBuffer->release();
      ibindSet->argumentEncoder->release();
      ibindSet->residencySet->release();

      ctx->ibindSetStore.free(bindSets[i].handle);
    }
  }

  void cgpuUpdateBindSet(CgpuContext* ctx,
                         CgpuBindSet bindSet,
                         const CgpuBindings* bindings)
  {
    CGPU_RESOLVE_BIND_SET(ctx, bindSet, ibindSet);

    MTL::ResidencySet* residencySet = ibindSet->residencySet;
    residencySet->removeAllAllocations();

    ibindSet->dynamicBuffers.clear();

    MTL::ArgumentEncoder* argumentEncoder = ibindSet->argumentEncoder;

    for (uint32_t i = 0; i < bindings->bufferCount; i++)
    {
      const CgpuBufferBinding& b = bindings->buffers[i];

      CGPU_RESOLVE_BUFFER(ctx, b.buffer, ibuffer);

      if (ibuffer->isDynamic)
      {
        ibindSet->dynamicBuffers.push_back(b); // set later with offset
      }
      else
      {
        argumentEncoder->setBuffer(ibuffer->buffer, b.offset, b.binding);
      }

      residencySet->addAllocation(ibuffer->buffer);
    }

    for (uint32_t i = 0; i < bindings->imageCount; i++)
    {
      const CgpuImageBinding& b = bindings->images[i];

      CGPU_RESOLVE_IMAGE(ctx, b.image, iimage);

      // We only support two kinds of descriptor sets - texture heaps (basically as single array
      // of textures) and 'no texture' descriptor sets. This makes it possible to bind single images
      // to array slots below. Putting the arrays into separate sets also fixes a crash on AMD.
      assert(bindings->bufferCount == 0);
      assert(bindings->samplerCount == 0);
      assert(bindings->tlasCount == 0);

      argumentEncoder->setTexture(iimage->texture, b.index);

      residencySet->addAllocation(iimage->texture);
    }

    for (uint32_t i = 0; i < bindings->samplerCount; i++)
    {
      const CgpuSamplerBinding& b = bindings->samplers[i];

      CGPU_RESOLVE_SAMPLER(ctx, b.sampler, isampler);
      argumentEncoder->setSamplerState(isampler->sampler, b.binding);
    }

    for (uint32_t i = 0; i < bindings->tlasCount; i++)
    {
      const CgpuTlasBinding& b = bindings->tlases[i];

      CGPU_RESOLVE_TLAS(ctx, b.as, itlas);
      argumentEncoder->setAccelerationStructure(itlas->as, b.binding);

      for (const MTL::AccelerationStructure* as : itlas->blases)
      {
        residencySet->addAllocation(as);
      }
      residencySet->addAllocation(itlas->as);
    }

    MTL::Buffer* argumentBuffer = ibindSet->argumentBuffer;
    residencySet->addAllocation(argumentBuffer);

    residencySet->commit();
    residencySet->requestResidency();
  }

  bool cgpuCreateCommandBuffer(CgpuContext* ctx, CgpuCommandBuffer* commandBuffer)
  {
    CgpuIDevice* idevice = &ctx->idevice;

    uint64_t handle = ctx->icommandBufferStore.allocate();

    CGPU_RESOLVE_COMMAND_BUFFER(ctx, { handle }, icommandBuffer);

    uint32_t initialCapacity = 32;
    MTL::ResidencySet* auxResidencySet = cgpuCreateResidencySet(idevice->device, initialCapacity);

    MTL4::CommandAllocator* commandAllocator = idevice->device->newCommandAllocator();
    CGPU_CHK_NP(commandAllocator);
    MTL4::CommandBuffer* mtlCommandBuffer = idevice->device->newCommandBuffer();
    CGPU_CHK_NP(mtlCommandBuffer);

    icommandBuffer->commandAllocator = commandAllocator;
    icommandBuffer->commandBuffer = mtlCommandBuffer;
#ifndef NDEBUG
    icommandBuffer->commitOptions = idevice->commitOptions;
#endif
    icommandBuffer->commandBufferOptions = idevice->commandBufferOptions;
    icommandBuffer->auxResidencySet = auxResidencySet;
    icommandBuffer->deviceResidencySet = idevice->residencySet;

    commandBuffer->handle = handle;
    return true;
  }

  void cgpuDestroyCommandBuffer(CgpuContext* ctx, CgpuCommandBuffer commandBuffer)
  {
    CGPU_RESOLVE_COMMAND_BUFFER(ctx, commandBuffer, icommandBuffer);

    icommandBuffer->commandBuffer->release();
    icommandBuffer->commandAllocator->release();
    icommandBuffer->auxResidencySet->release();

    ctx->icommandBufferStore.free(commandBuffer.handle);
  }

  bool cgpuBeginCommandBuffer(CgpuContext* ctx, CgpuCommandBuffer commandBuffer)
  {
    CGPU_RESOLVE_COMMAND_BUFFER(ctx, commandBuffer, icommandBuffer);

    icommandBuffer->auxResidencySet->removeAllAllocations();
    icommandBuffer->residencySets.clear();

    icommandBuffer->commandBuffer->beginCommandBuffer(
      icommandBuffer->commandAllocator,
      icommandBuffer->commandBufferOptions
    );

    return true;
  }

  void cgpuCmdBindPipeline(CgpuContext* ctx,
                           CgpuCommandBuffer commandBuffer,
                           CgpuPipeline pipeline,
                           const CgpuBindSet* bindSets,
                           uint32_t bindSetCount,
                           uint32_t dynamicOffsetCount,
                           const uint32_t* dynamicOffsets)
  {
    CGPU_RESOLVE_COMMAND_BUFFER(ctx, commandBuffer, icommandBuffer);
    CGPU_RESOLVE_PIPELINE(ctx, pipeline, ipipeline);

    icommandBuffer->pipeline = ipipeline;

    uint32_t dynamicBufferIndex = 0;

    for (uint32_t i = 0; i < bindSetCount; i++)
    {
      CGPU_RESOLVE_BIND_SET(ctx, bindSets[i], ibindSet);

      for (const CgpuBufferBinding& b : ibindSet->dynamicBuffers)
      {
        CGPU_RESOLVE_BUFFER(ctx, b.buffer, ibuffer);

        uint32_t offset = b.offset + dynamicOffsets[dynamicBufferIndex];
        ibindSet->argumentEncoder->setBuffer(ibuffer->buffer, offset, b.binding);

        dynamicBufferIndex++;
      }

      MTL::Buffer* argumentBuffer = ibindSet->argumentBuffer;

      for (const CgpuRtFunctionTables& fts : ipipeline->fts)
      {
        uint32_t bufferOffset = 0;
        fts.ift->setBuffer(argumentBuffer, bufferOffset, i);
      }

      ipipeline->argumentTable->setAddress(argumentBuffer->gpuAddress(), i);

      icommandBuffer->residencySets.push_back(ibindSet->residencySet);
    }

    assert(dynamicBufferIndex == dynamicOffsetCount);
  }

  void cgpuCmdTransitionShaderImageLayouts(CgpuContext* ctx,
                                           CgpuCommandBuffer commandBuffer,
                                           CgpuShader shader,
                                           uint32_t descriptorSetIndex,
                                           uint32_t imageCount,
                                           const CgpuImageBinding* images)
  {
    CGPU_RESOLVE_SHADER(ctx, shader, ishader);
    CGPU_RESOLVE_COMMAND_BUFFER(ctx, commandBuffer, icommandBuffer);

    MTL4::ComputeCommandEncoder* encoder = icommandBuffer->commandBuffer->computeCommandEncoder();

    for (uint32_t i = 0; i < imageCount; i++)
    {
      const CgpuImageBinding& b = images[i];
      CGPU_RESOLVE_IMAGE(ctx, b.image, iimage);

      encoder->optimizeContentsForGPUAccess(iimage->texture);
    }

    encoder->endEncoding();
  }

  void cgpuCmdCopyBuffer(CgpuContext* ctx,
                         CgpuCommandBuffer commandBuffer,
                         CgpuBuffer srcBuffer,
                         uint64_t srcOffset,
                         CgpuBuffer dstBuffer,
                         uint64_t dstOffset,
                         uint64_t size)
  {
    CGPU_RESOLVE_COMMAND_BUFFER(ctx, commandBuffer, icommandBuffer);
    CGPU_RESOLVE_BUFFER(ctx, srcBuffer, isrcBuffer);
    CGPU_RESOLVE_BUFFER(ctx, dstBuffer, idstBuffer);

    MTL4::ComputeCommandEncoder* encoder = icommandBuffer->commandBuffer->computeCommandEncoder();

    uint64_t rangeSize = (size == CGPU_WHOLE_SIZE) ? std::min(isrcBuffer->size, idstBuffer->size) : size;
    encoder->copyFromBuffer(isrcBuffer->buffer, srcOffset, idstBuffer->buffer, dstOffset, rangeSize);

    icommandBuffer->auxResidencySet->addAllocation(isrcBuffer->buffer);
    icommandBuffer->auxResidencySet->addAllocation(idstBuffer->buffer);

    encoder->endEncoding();
  }

  void cgpuCmdCopyBufferToImage(CgpuContext* ctx,
                                CgpuCommandBuffer commandBuffer,
                                CgpuBuffer buffer,
                                CgpuImage image,
                                const CgpuBufferImageCopyDesc* desc)
  {
    CGPU_RESOLVE_COMMAND_BUFFER(ctx, commandBuffer, icommandBuffer);
    CGPU_RESOLVE_BUFFER(ctx, buffer, ibuffer);
    CGPU_RESOLVE_IMAGE(ctx, image, iimage);

    uint32_t bytesPerPixel = cgpuGetImageFormatBpp(iimage->format);
    uint32_t srcBytesPerRow = iimage->width * bytesPerPixel;
    uint32_t srcBytesPerImage = (iimage->depth == 1) ? 0 : (iimage->height * srcBytesPerRow);
    MTL::Size srcSize(desc->texelExtentX, desc->texelExtentY, desc->texelExtentZ);
    uint32_t dstSlice = 0;
    uint32_t dstMipmapLevel = 0;
    MTL::Origin dstOrigin(desc->texelOffsetX, desc->texelOffsetY, desc->texelOffsetZ);

    MTL4::ComputeCommandEncoder* encoder = icommandBuffer->commandBuffer->computeCommandEncoder();

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

    icommandBuffer->auxResidencySet->addAllocation(ibuffer->buffer);
    icommandBuffer->auxResidencySet->addAllocation(iimage->texture);

    encoder->endEncoding();
  }

  void cgpuCmdDispatch(CgpuContext* ctx,
                       CgpuCommandBuffer commandBuffer,
                       uint32_t dimX,
                       uint32_t dimY,
                       uint32_t dimZ)
  {
    CGPU_RESOLVE_COMMAND_BUFFER(ctx, commandBuffer, icommandBuffer);
    CgpuIPipeline* ipipeline = icommandBuffer->pipeline;

    MTL4::ComputeCommandEncoder* encoder = icommandBuffer->commandBuffer->computeCommandEncoder();

    for (const CgpuRtFunctionTables& fts : ipipeline->fts)
    {
      icommandBuffer->auxResidencySet->addAllocation(fts.ift);
      icommandBuffer->auxResidencySet->addAllocation(fts.missVft);
      icommandBuffer->auxResidencySet->addAllocation(fts.chitVft);
    }

    encoder->setComputePipelineState(ipipeline->state);
    encoder->setArgumentTable(ipipeline->argumentTable);

    auto threadsPerGrid = MTL::Size(dimX, dimY, dimZ);
    encoder->dispatchThreads(threadsPerGrid, ipipeline->threadsPerGroup);

    encoder->endEncoding();
  }

  void cgpuCmdPipelineBarrier(CgpuContext* ctx,
                              CgpuCommandBuffer commandBuffer,
                              const CgpuPipelineBarrier* barrier)
  {
    CGPU_RESOLVE_COMMAND_BUFFER(ctx, commandBuffer, icommandBuffer);

    MTL::Stages beforeStages = 0;
    MTL::Stages afterStages = 0;

    for (uint32_t i = 0; i < barrier->memoryBarrierCount; i++)
    {
      const CgpuMemoryBarrier& b = barrier->memoryBarriers[i];

      afterStages |= cgpuTranslatePipelineStages(b.srcStageMask);
      beforeStages |= cgpuTranslatePipelineStages(b.dstStageMask);
    }

    for (uint32_t i = 0; i < barrier->bufferBarrierCount; i++)
    {
      const CgpuBufferMemoryBarrier& b = barrier->bufferBarriers[i];

      afterStages |= cgpuTranslatePipelineStages(b.srcStageMask);
      beforeStages |= cgpuTranslatePipelineStages(b.dstStageMask);
    }

    for (uint32_t i = 0; i < barrier->imageBarrierCount; i++)
    {
      const CgpuImageMemoryBarrier& b = barrier->imageBarriers[i];

      afterStages |= cgpuTranslatePipelineStages(b.srcStageMask);
      beforeStages |= cgpuTranslatePipelineStages(b.dstStageMask);
    }

    assert(beforeStages || afterStages);

    MTL4::ComputeCommandEncoder* encoder = icommandBuffer->commandBuffer->computeCommandEncoder();

    encoder->barrierAfterQueueStages(afterStages, beforeStages, MTL4::VisibilityOptionDevice);

    encoder->endEncoding();
  }

  void cgpuCmdTraceRays(CgpuContext* ctx,
                        CgpuCommandBuffer commandBuffer,
                        uint32_t width,
                        uint32_t height)
  {
    CGPU_RESOLVE_COMMAND_BUFFER(ctx, commandBuffer, icommandBuffer);

    cgpuCmdDispatch(ctx, commandBuffer, width, height, 1);
  }

  void cgpuCmdFillBuffer(CgpuContext* ctx,
                         CgpuCommandBuffer commandBuffer,
                         CgpuBuffer buffer,
                         uint64_t dstOffset,
                         uint64_t size,
                         uint8_t data)
  {
    CGPU_RESOLVE_COMMAND_BUFFER(ctx, commandBuffer, icommandBuffer);
    CGPU_RESOLVE_BUFFER(ctx, buffer, ibuffer);

    MTL4::ComputeCommandEncoder* encoder = icommandBuffer->commandBuffer->computeCommandEncoder();

    NS::Range range(dstOffset, size);
    encoder->fillBuffer(ibuffer->buffer, range, data);

    encoder->endEncoding();
  }

  void cgpuEndCommandBuffer(CgpuContext* ctx, CgpuCommandBuffer commandBuffer)
  {
    CGPU_RESOLVE_COMMAND_BUFFER(ctx, commandBuffer, icommandBuffer);

    icommandBuffer->deviceResidencySet->commit();
    icommandBuffer->residencySets.push_back(icommandBuffer->deviceResidencySet);

    icommandBuffer->auxResidencySet->commit();
    icommandBuffer->residencySets.push_back(icommandBuffer->auxResidencySet);

    icommandBuffer->commandBuffer->useResidencySets(icommandBuffer->residencySets.data(),
                                                    icommandBuffer->residencySets.size());

    icommandBuffer->commandBuffer->endCommandBuffer();
  }

  bool cgpuCreateSemaphore(CgpuContext* ctx, CgpuSemaphore* semaphore, uint64_t initialValue)
  {
    CgpuIDevice* idevice = &ctx->idevice;

    uint64_t handle = ctx->isemaphoreStore.allocate();

    CGPU_RESOLVE_SEMAPHORE(ctx, { handle }, isemaphore);

    MTL::SharedEvent* event = idevice->device->newSharedEvent();
    if (!event)
    {
      CGPU_RETURN_ERROR("failed to create event");
    }

    isemaphore->event = event;

    semaphore->handle = handle;
    return true;
  }

  void cgpuDestroySemaphore(CgpuContext* ctx, CgpuSemaphore semaphore)
  {
    CGPU_RESOLVE_SEMAPHORE(ctx, semaphore, isemaphore);

    isemaphore->event->release();

    ctx->isemaphoreStore.free(semaphore.handle);
  }

  bool cgpuWaitSemaphores(CgpuContext* ctx,
                          uint32_t semaphoreInfoCount,
                          CgpuWaitSemaphoreInfo* semaphoreInfos,
                          uint64_t timeoutNs)
  {
    uint64_t timeoutMs = timeoutNs / 1000000;

    bool success = true;
    for (uint32_t i = 0; i < semaphoreInfoCount; i++)
    {
      CGPU_RESOLVE_SEMAPHORE(ctx, semaphoreInfos[i].semaphore, isemaphore);

      success &= isemaphore->event->waitUntilSignaledValue(semaphoreInfos[i].value, timeoutMs);
    }

    return success;
  }

  void cgpuSubmitCommandBuffer(CgpuContext* ctx,
                               CgpuCommandBuffer commandBuffer,
                               uint32_t signalSemaphoreInfoCount,
                               CgpuSignalSemaphoreInfo* signalSemaphoreInfos,
                               uint32_t waitSemaphoreInfoCount,
                               CgpuWaitSemaphoreInfo* waitSemaphoreInfos)
  {
    CgpuIDevice* idevice = &ctx->idevice;
    CGPU_RESOLVE_COMMAND_BUFFER(ctx, commandBuffer, icommandBuffer);

    MTL4::CommandQueue* commandQueue = idevice->commandQueue;

    for (uint32_t i = 0; i < waitSemaphoreInfoCount; i++)
    {
      CGPU_RESOLVE_SEMAPHORE(ctx, waitSemaphoreInfos[i].semaphore, isemaphore);
      commandQueue->wait(isemaphore->event, waitSemaphoreInfos[i].value);
    }

    commandQueue->commit(&icommandBuffer->commandBuffer, 1, idevice->commitOptions);

    for (uint32_t i = 0; i < signalSemaphoreInfoCount; i++)
    {
      CGPU_RESOLVE_SEMAPHORE(ctx, signalSemaphoreInfos[i].semaphore, isemaphore);
      commandQueue->signalEvent(isemaphore->event, signalSemaphoreInfos[i].value);
    }
  }

  const CgpuDeviceFeatures& cgpuGetDeviceFeatures(CgpuContext* ctx)
  {
    return CGPU_DEVICE_FEATURES;
  }

  const CgpuDeviceProperties& cgpuGetDeviceProperties(CgpuContext* ctx)
  {
    return CGPU_DEVICE_PROPERTIES;
  }
}
