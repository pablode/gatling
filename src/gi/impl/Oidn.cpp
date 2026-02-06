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

#include "Oidn.h"
#include "GlslShaderGen.h"
#include "interface/rp_oidn.h"
#include "interface/rp_max_luminance.h"

#include <gtl/ggpu/DelayedResourceDestroyer.h>
#include <gtl/ggpu/Stager.h>
#include <gtl/gb/Enum.h>
#include <gtl/gb/Log.h>
#include <gtl/gb/Fmt.h>

#include <functional>

namespace gtl
{
  namespace rp = shader_interface::rp_oidn;
  namespace rp_ml = shader_interface::rp_max_luminance;

  using GiOidnPostOp = GiGlslShaderGen::OidnPostOp;

  class GiOidnNet
  {
  private:
    enum class Buffer
    {
      Invalid, Pool0, Pool1, Pool2, Pool3, Output, Scratch, COUNT
    };

    struct PipelineStep
    {
      CgpuPipeline pipeline;
      Buffer input1;
      Buffer input2;
      Buffer output;
      uint32_t weightOffset;
      uint32_t biasOffset;
      GiOidnPostOp postOp;
      uint32_t outDims;
    };

  private:
    CgpuBuffer m_pool0;
    CgpuBuffer m_pool1;
    CgpuBuffer m_pool2;
    CgpuBuffer m_pool3;
    CgpuBuffer m_scratchMem[2]; // TODO: replace with single buffer + suballocation
    CgpuBuffer m_outputPool;

    CgpuBuffer m_tensorBuffer;

    std::vector<PipelineStep> m_steps;

    uint32_t m_imageWidth = 0; // 16px multiple
    uint32_t m_imageHeight = 0; // 16px multiple
    uint32_t m_wgSizeX;
    uint32_t m_wgSizeY;

    GgpuDelayedResourceDestroyer& m_resourceDestroyer;

    std::array<uint32_t, int(Buffer::COUNT)> m_bufferSizeMuls;
    std::array<uint32_t, int(Buffer::COUNT)> m_bufferLastDims;

  private:
    struct TensorUpload
    {
      uint32_t offset;
      GiTzaTensorLayout layout;
      std::vector<int> dimensions; // OIHW
    };

    struct BuildContext
    {
      CgpuDevice device;
      GiGlslShaderGen& shaderGen;
      GgpuStager& stager;
      const GiTzaTensorDescriptions& tensorDescriptions;
      const uint8_t* tensorData;
      uint32_t vendorId;

      int depth = 0; // track up- and downsampling
      std::unordered_map<std::string, TensorUpload> tensorUploads;
    };

    void uploadTensors(BuildContext& c)
    {
      // create buffer
      uint32_t bufferSize = 0;
      for (const auto& ntPair : c.tensorDescriptions)
      {
        const auto& tensorDesc = ntPair.second;

        bufferSize += (tensorDesc.dataSize + 4 - 1) / 4 * 4; // Vk requires 4 byte size
      }

      CgpuBufferUsageFlags bufferUsage = CGPU_BUFFER_USAGE_FLAG_STORAGE_BUFFER | CGPU_BUFFER_USAGE_FLAG_TRANSFER_DST;
      if (!cgpuCreateBuffer(c.device, { .usage = bufferUsage, .size = bufferSize }, &m_tensorBuffer))
      {
        GB_FATAL("failed to alloc tensor buffer");
      }

      // upload data
      uint32_t dataOffset = 0;
      for (const auto& ntPair : c.tensorDescriptions)
      {
        const auto& tensorDesc = ntPair.second;

        uint32_t dataSize = (tensorDesc.dataSize + 4 - 1) / 4 * 4; // Vk requires 4 byte size

        c.tensorUploads[ntPair.first] = TensorUpload{ dataOffset, tensorDesc.layout, tensorDesc.dimensions };

        if (!c.stager.stageToBuffer(&c.tensorData[tensorDesc.dataOffset], dataSize, m_tensorBuffer, dataOffset))
        {
          GB_FATAL("failed to upload tensor");
        }

        dataOffset += dataSize;
      }
    }

    std::string makeConvolutionPipelineDebugName(int inDims, int outDims, GiOidnPostOp postOp)
    {
      std::string opStr;
      if (bool(postOp & GiOidnPostOp::MaxPool)) { opStr += "_MaxPool"; }
      else if (bool(postOp & GiOidnPostOp::Upsample)) { opStr += "_Upsample"; }
      else if (bool(postOp & GiOidnPostOp::Concat)) { opStr += "_Concat"; }
      else if (bool(postOp & GiOidnPostOp::WriteBackRgba32)) { opStr += "_WriteBackRgba32"; }
      else if (bool(postOp & GiOidnPostOp::ScaleInputInv)) { opStr += "_ScaleInputInv"; }
      else if (bool(postOp & GiOidnPostOp::ScaleOutput)) { opStr += "_ScaleOutput"; }
      else if (postOp != GiOidnPostOp::None) { GB_FATAL("Unhandled OIDN post op"); }

      return GB_FMT("Oidn_{}->{}{}", inDims, outDims, opStr);
    }

    void addConvReLU(BuildContext& c, Buffer input1, Buffer output, std::string_view tensor, GiOidnPostOp postOp = GiOidnPostOp::None, Buffer input2 = Buffer::Invalid)
    {
      std::string weightName = GB_FMT("{}.weight", tensor);
      std::string biasName = GB_FMT("{}.bias", tensor);

      auto ntIt = c.tensorDescriptions.find(weightName);
      if (ntIt == c.tensorDescriptions.end())
      {
        GB_FATAL("tensor not loaded");
      }

      if ((bool(postOp & GiOidnPostOp::Upsample) && bool(postOp & GiOidnPostOp::MaxPool)) ||
          (bool(postOp & GiOidnPostOp::Concat) && bool(postOp & GiOidnPostOp::WriteBackRgba32)))
      {
        GB_FATAL("unsupported post op combination");
      }

      const auto& tensorDesc = ntIt->second;

      int in1Dims = tensorDesc.dimensions[1]; // OIHW
      int in2Dims = m_bufferLastDims[int(input2)];
      int convDims = tensorDesc.dimensions[0];
      int outDims = bool(postOp & GiOidnPostOp::WriteBackRgba32) ? 4 : convDims;

      if (uint32_t lastDims = m_bufferLastDims[int(input1)]; lastDims != 0 && in1Dims != lastDims)
      {
        GB_FATAL("layer dimension mismatch");
      }

      if (bool(postOp & GiOidnPostOp::Concat))
      {
        if (in2Dims == 0) { GB_FATAL("join not preceeded by convolution"); };
        if (input2 == Buffer::Invalid) { GB_FATAL("join with invalid second input buffer"); };
        outDims += in2Dims;
      }

      std::string debugName = makeConvolutionPipelineDebugName(in1Dims, outDims, postOp);

      int convImpl = rp::CONV_IMPL_SHMEM;
      if (c.vendorId == CGPU_VENDOR_ID_MESA)
      {
        // Shared memory implementation is orders of magnitude slower on SW Vulkan runtime
        // Mesa lavapipe (used in CI graphical tests). Prefer sequential path.
        convImpl = rp::CONV_IMPL_SEQ;
      }

      GiGlslShaderGen::OidnParams sgParams{
        .wgSizeX = int(m_wgSizeX),
        .wgSizeY = int(m_wgSizeY),
        .in1ChannelCount = in1Dims,
        .outChannelCount = outDims,
        .convChannelCount = convDims,
        .convolutionImpl = convImpl,
        .postOp = postOp
      };
      if (bool(postOp & GiOidnPostOp::Concat))
      {
        sgParams.in2ChannelCount = in2Dims;
      }

      std::vector<uint8_t> spv;
      if (!c.shaderGen.generateDenoisingSpirv(sgParams, spv))
      {
        GB_FATAL("failed to compile OIDN shader");
      }

      CgpuShader shader;
      if (!cgpuCreateShader(c.device, { .size = spv.size(), .source = spv.data(), .stageFlags = CGPU_SHADER_STAGE_FLAG_COMPUTE }, &shader))
      {
        GB_FATAL("failed to create OIDN shader");
      }

      CgpuPipeline pipeline;
      if (!cgpuCreateComputePipeline(c.device, { .shader = shader , .debugName = debugName.c_str() }, &pipeline))
      {
        GB_FATAL("failed to create OIDN pipeline");
      }
      cgpuDestroyShader(c.device, shader);

      uint32_t inSizeMul = in1Dims / std::pow(2, c.depth) * sizeof(float)/2;

      if (bool(postOp & GiOidnPostOp::MaxPool))
      {
        c.depth++;
      }
      else if (bool(postOp & GiOidnPostOp::Upsample))
      {
        c.depth--;
      }

      uint32_t outTypeSize = bool(postOp & GiOidnPostOp::WriteBackRgba32) ? sizeof(float) : (sizeof(float) / 2);
      uint32_t outSizeMul = outDims / std::pow(2, c.depth) * outTypeSize;

      m_bufferSizeMuls[int(input1)] = std::max(m_bufferSizeMuls[int(input1)], inSizeMul);
      m_bufferSizeMuls[int(output)] = std::max(m_bufferSizeMuls[int(output)], outSizeMul);

      m_bufferLastDims[int(input1)] = in1Dims;
      m_bufferLastDims[int(output)] = outDims;

      const TensorUpload& weights = c.tensorUploads[weightName];
      const TensorUpload& bias = c.tensorUploads[biasName];

      if (weights.layout != GiTzaTensorLayout::oihw) { GB_FATAL("unexpected tensor layout"); }
      if (bias.layout != GiTzaTensorLayout::x) { GB_FATAL("unexpected tensor layout"); }
      if (weights.dimensions.size() != 4 || weights.dimensions[2] != 3 || weights.dimensions[3] != 3) { GB_FATAL("unsupported kernel dimensions"); }

      m_steps.push_back(PipelineStep{
        .pipeline = pipeline,
        .input1 = input1,
        .input2 = input2,
        .output = output,
        .weightOffset = weights.offset / 2,
        .biasOffset = bias.offset / 2,
        .postOp = postOp,
        .outDims = uint32_t(outDims)
      });
    }

    void reallocBuffers(CgpuDevice device, uint32_t width, uint32_t height)
    {
      m_resourceDestroyer.enqueueDestruction(m_pool0, m_pool1, m_pool2, m_pool3, m_scratchMem[0], m_scratchMem[1], m_outputPool);

      uint64_t pixelCount = width * height;
      uint64_t pool0Size = pixelCount * m_bufferSizeMuls[int(Buffer::Pool0)];

      CgpuBufferUsageFlags pool0Usage = CGPU_BUFFER_USAGE_FLAG_STORAGE_BUFFER | CGPU_BUFFER_USAGE_FLAG_TRANSFER_SRC | CGPU_BUFFER_USAGE_FLAG_TRANSFER_DST;
      if (!cgpuCreateBuffer(device, { .usage = pool0Usage, .size = pool0Size }, &m_pool0))
      {
        GB_FATAL("failed to allocate OIDN buffer");
      }

      uint64_t pool1Size = pixelCount * m_bufferSizeMuls[int(Buffer::Pool1)];
      uint64_t pool2Size = pixelCount * m_bufferSizeMuls[int(Buffer::Pool2)];
      uint64_t pool3Size = pixelCount * m_bufferSizeMuls[int(Buffer::Pool3)];
      CgpuBufferUsageFlags poolNBufferUsage = CGPU_BUFFER_USAGE_FLAG_STORAGE_BUFFER | CGPU_BUFFER_USAGE_FLAG_TRANSFER_SRC;
      if (!cgpuCreateBuffer(device, { .usage = poolNBufferUsage, .size = pool1Size }, &m_pool1) ||
          !cgpuCreateBuffer(device, { .usage = poolNBufferUsage, .size = pool2Size }, &m_pool2) ||
          !cgpuCreateBuffer(device, { .usage = poolNBufferUsage, .size = pool3Size }, &m_pool3))
      {
        GB_FATAL("failed to allocate OIDN buffer");
      }

      uint64_t scratchSize = pixelCount * m_bufferSizeMuls[int(Buffer::Scratch)];
      CgpuBufferUsageFlags scratchUsage = CGPU_BUFFER_USAGE_FLAG_STORAGE_BUFFER | CGPU_BUFFER_USAGE_FLAG_TRANSFER_DST/*concat*/;
      if (!cgpuCreateBuffer(device, { .usage = scratchUsage, .size = scratchSize }, &m_scratchMem[0]) ||
          !cgpuCreateBuffer(device, { .usage = scratchUsage, .size = scratchSize }, &m_scratchMem[1]))
      {
        GB_FATAL("failed to allocate OIDN buffer");
      }

      uint64_t outputSize = pixelCount * m_bufferSizeMuls[int(Buffer::Output)];
      if (!cgpuCreateBuffer(device, { .usage = CGPU_BUFFER_USAGE_FLAG_STORAGE_BUFFER | CGPU_BUFFER_USAGE_FLAG_TRANSFER_SRC, .size = outputSize }, &m_outputPool))
      {
        GB_FATAL("failed to allocate OIDN buffer");
      }
    }

  public:
    GiOidnNet(CgpuDevice device,
           GiGlslShaderGen& shaderGen,
           GgpuStager& stager,
           GgpuDelayedResourceDestroyer& resourceDestroyer,
           const GiTzaTensorDescriptions& tensorDescs,
           const uint8_t* tensorData)
      : m_resourceDestroyer(resourceDestroyer)
    {
      CgpuPhysicalDeviceProperties deviceProperties;
      cgpuGetPhysicalDeviceProperties(device, deviceProperties);

      uint32_t shmemSize = deviceProperties.maxComputeSharedMemorySize;

      if (shmemSize >= 65536) { m_wgSizeX = 24; m_wgSizeY = 16; }
      else if (shmemSize >= 49152) { m_wgSizeX = 16; m_wgSizeY = 16; }
      else { m_wgSizeX = 16; m_wgSizeY = 8; }

      for (uint32_t& i : m_bufferLastDims) { i = 0; }
      for (uint32_t& i : m_bufferSizeMuls) { i = 0; }

      BuildContext c{ device, shaderGen, stager, tensorDescs, tensorData, deviceProperties.vendorId };

      uploadTensors(c);

      addConvReLU(c, Buffer::Pool0, Buffer::Scratch, "enc_conv0", GiOidnPostOp::ScaleInputInv);
      addConvReLU(c, Buffer::Scratch, Buffer::Pool1, "enc_conv1", GiOidnPostOp::MaxPool);
      addConvReLU(c, Buffer::Pool1, Buffer::Pool2, "enc_conv2", GiOidnPostOp::MaxPool);
      addConvReLU(c, Buffer::Pool2, Buffer::Pool3, "enc_conv3", GiOidnPostOp::MaxPool);
      addConvReLU(c, Buffer::Pool3, Buffer::Scratch, "enc_conv4", GiOidnPostOp::MaxPool);
      addConvReLU(c, Buffer::Scratch, Buffer::Scratch, "enc_conv5a");
      addConvReLU(c, Buffer::Scratch, Buffer::Scratch, "enc_conv5b", GiOidnPostOp::Upsample | GiOidnPostOp::Concat, Buffer::Pool3);
      addConvReLU(c, Buffer::Scratch, Buffer::Scratch, "dec_conv4a");
      addConvReLU(c, Buffer::Scratch, Buffer::Scratch, "dec_conv4b", GiOidnPostOp::Upsample | GiOidnPostOp::Concat, Buffer::Pool2);
      addConvReLU(c, Buffer::Scratch, Buffer::Scratch, "dec_conv3a");
      addConvReLU(c, Buffer::Scratch, Buffer::Scratch, "dec_conv3b", GiOidnPostOp::Upsample | GiOidnPostOp::Concat, Buffer::Pool1);
      addConvReLU(c, Buffer::Scratch, Buffer::Scratch, "dec_conv2a");
      addConvReLU(c, Buffer::Scratch, Buffer::Scratch, "dec_conv2b", GiOidnPostOp::Upsample | GiOidnPostOp::Concat, Buffer::Pool0);
      addConvReLU(c, Buffer::Scratch, Buffer::Scratch, "dec_conv1a");
      addConvReLU(c, Buffer::Scratch, Buffer::Scratch, "dec_conv1b");
      addConvReLU(c, Buffer::Scratch, Buffer::Output, "dec_conv0", GiOidnPostOp::ScaleOutput | GiOidnPostOp::WriteBackRgba32);

      if (c.depth != 0) { GB_FATAL("invalid network architecture"); }
    }

    ~GiOidnNet()
    {
      m_resourceDestroyer.enqueueDestruction(m_pool0, m_pool1, m_pool2, m_pool3, m_scratchMem[0], m_scratchMem[1], m_outputPool, m_tensorBuffer);

      for (const PipelineStep& step : m_steps)
      {
        m_resourceDestroyer.enqueueDestruction(step.pipeline);
      }
    }

    // TODO: need to call this before getting pool0 outside (e.g. in giRender at the start of frame)
    void updateViewport(CgpuDevice device, uint32_t width, uint32_t height)
    {
      if (width == m_imageWidth && height == m_imageHeight)
      {
        return;
      }

      reallocBuffers(device, width, height);

      m_imageWidth = width;
      m_imageHeight = height;
    }

    CgpuBuffer getInputBuffer() const
    {
      return m_pool0;
    }

    CgpuBuffer getOutputBuffer() const
    {
      return m_outputPool;
    }

    void runInference(CgpuCommandBuffer commandBuffer, CgpuBuffer valueScaleBuffer)
    {
      uint32_t scratchIdx = 0; // ping pong

      auto resolveBuffer = [&](Buffer b) {
        if (b == Buffer::Pool0) { return m_pool0; }
        else if (b == Buffer::Pool1) { return m_pool1; }
        else if (b == Buffer::Pool2) { return m_pool2; }
        else if (b == Buffer::Pool3) { return m_pool3; }
        else if (b == Buffer::Output) { return m_outputPool; }
        else if (b == Buffer::Scratch) { return m_scratchMem[scratchIdx]; }
        else { GB_FATAL("unable to resolve OIDN buffer"); }
      };

      auto resolveInputBuffer = [&](Buffer b) { return resolveBuffer(b); };
      auto resolveOututBuffer = [&](Buffer b)
      {
        if (b == Buffer::Scratch)
        {
          scratchIdx = (scratchIdx + 1) % 2;
        }
        return resolveBuffer(b);
      };

      uint32_t imageWidth = m_imageWidth;
      uint32_t imageHeight = m_imageHeight;

      for (const PipelineStep& step : m_steps)
      {
        std::vector<CgpuBufferMemoryBarrier> bufferBarriers;

        CgpuBuffer inBuffer1 = resolveInputBuffer(step.input1);
        CgpuBuffer inBuffer2;
        if (bool(step.postOp & GiOidnPostOp::Concat))
        {
          inBuffer2 = resolveInputBuffer(step.input2);
        }
        CgpuBuffer outBuffer = resolveOututBuffer(step.output);

        // TODO: minimize (src) barriers depending on op
        bufferBarriers.push_back(CgpuBufferMemoryBarrier{
          .buffer = inBuffer1,
          .srcStageMask = CGPU_PIPELINE_STAGE_FLAG_RAY_TRACING_SHADER | CGPU_PIPELINE_STAGE_FLAG_COMPUTE_SHADER | CGPU_PIPELINE_STAGE_FLAG_TRANSFER,
          .srcAccessMask = CGPU_MEMORY_ACCESS_FLAG_SHADER_WRITE | CGPU_MEMORY_ACCESS_FLAG_TRANSFER_WRITE,
          .dstStageMask = CGPU_PIPELINE_STAGE_FLAG_COMPUTE_SHADER,
          .dstAccessMask = CGPU_MEMORY_ACCESS_FLAG_SHADER_WRITE
        });
        if (bool(step.postOp & GiOidnPostOp::Concat))
        {
          bufferBarriers.push_back(CgpuBufferMemoryBarrier{
            .buffer = inBuffer2,
            .srcStageMask = CGPU_PIPELINE_STAGE_FLAG_RAY_TRACING_SHADER | CGPU_PIPELINE_STAGE_FLAG_COMPUTE_SHADER | CGPU_PIPELINE_STAGE_FLAG_TRANSFER,
            .srcAccessMask = CGPU_MEMORY_ACCESS_FLAG_SHADER_WRITE | CGPU_MEMORY_ACCESS_FLAG_TRANSFER_WRITE,
            .dstStageMask = CGPU_PIPELINE_STAGE_FLAG_COMPUTE_SHADER,
            .dstAccessMask = CGPU_MEMORY_ACCESS_FLAG_SHADER_WRITE
          });
        }
        bufferBarriers.push_back(CgpuBufferMemoryBarrier{
          .buffer = outBuffer,
          .srcStageMask = CGPU_PIPELINE_STAGE_FLAG_RAY_TRACING_SHADER | CGPU_PIPELINE_STAGE_FLAG_COMPUTE_SHADER | CGPU_PIPELINE_STAGE_FLAG_TRANSFER,
          .srcAccessMask = CGPU_MEMORY_ACCESS_FLAG_SHADER_WRITE | CGPU_MEMORY_ACCESS_FLAG_TRANSFER_WRITE,
          .dstStageMask = CGPU_PIPELINE_STAGE_FLAG_COMPUTE_SHADER,
          .dstAccessMask = CGPU_MEMORY_ACCESS_FLAG_SHADER_WRITE
        });

        CgpuPipelineBarrier barrier = {
          .bufferBarrierCount = (uint32_t) bufferBarriers.size(),
          .bufferBarriers = bufferBarriers.data()
        };
        cgpuCmdPipelineBarrier(commandBuffer, &barrier);

        rp::PushConstants pushData = {
          .imageWidth = imageWidth,
          .imageHeight = imageHeight,
          .weightOffset = step.weightOffset,
          .biasOffset = step.biasOffset
        };

        std::vector<CgpuBufferBinding> bufferBindings = {
          CgpuBufferBinding{.binding = rp::BINDING_INDEX_INPUT_BUF1, .buffer = inBuffer1 },
          CgpuBufferBinding{.binding = rp::BINDING_INDEX_OUTPUT_BUF, .buffer = outBuffer },
          CgpuBufferBinding{.binding = rp::BINDING_INDEX_TENSOR_BUF, .buffer = m_tensorBuffer }
        };
        if (bool(step.postOp & GiOidnPostOp::Concat))
        {
          bufferBindings.push_back(CgpuBufferBinding{ .binding = rp::BINDING_INDEX_INPUT_BUF2, .buffer = inBuffer2 });
        }
        if (bool(step.postOp & GiOidnPostOp::ScaleInputInv) || bool(step.postOp & GiOidnPostOp::ScaleOutput))
        {
          bufferBindings.push_back(CgpuBufferBinding{ .binding = rp::BINDING_INDEX_VALUE_SCALE_BUF, .buffer = valueScaleBuffer });
        }

        CgpuBindings bindings0 = { .bufferCount = (uint32_t) bufferBindings.size(), .buffers = bufferBindings.data() };
        cgpuCmdUpdateBindings(commandBuffer, step.pipeline, 0/*descriptorSetIndex*/, &bindings0);

        cgpuCmdBindPipeline(commandBuffer, step.pipeline);

        cgpuCmdPushConstants(commandBuffer, step.pipeline, CGPU_SHADER_STAGE_FLAG_COMPUTE, sizeof(pushData), &pushData);

        uint32_t wgCountX = (imageWidth + m_wgSizeX - 1) / m_wgSizeX;
        uint32_t wgCountY = (imageHeight + m_wgSizeY - 1) / m_wgSizeY;
        cgpuCmdDispatch(commandBuffer, wgCountX, wgCountY, 1);

        if (bool(step.postOp & GiOidnPostOp::MaxPool))
        {
          imageWidth /= 2;
          imageHeight /= 2;
        }
        else if (bool(step.postOp & GiOidnPostOp::Upsample))
        {
          imageWidth *= 2;
          imageHeight *= 2;
        }
      }

      // TODO: more appropriate in Gi.cpp ?

      // output buffer barrier
      CgpuBufferMemoryBarrier bufferBarrier{
        .buffer = m_outputPool,
        .srcStageMask = CGPU_PIPELINE_STAGE_FLAG_COMPUTE_SHADER,
        .srcAccessMask = CGPU_MEMORY_ACCESS_FLAG_SHADER_WRITE,
        .dstStageMask = CGPU_PIPELINE_STAGE_FLAG_TRANSFER,
        .dstAccessMask = CGPU_MEMORY_ACCESS_FLAG_MEMORY_READ
      };

      CgpuPipelineBarrier barrier = {
        .bufferBarrierCount = 1,
        .bufferBarriers = &bufferBarrier
      };
      cgpuCmdPipelineBarrier(commandBuffer, &barrier);
    }
  };

  // TODO: remove state proxy by exposing GiOidnNet in header
  struct GiOidnState
  {
    GiOidnNet net;
// TODO: it is not clear if luminance reduction belongs here
    CgpuPipeline maxLuminanceReduction;
    CgpuBuffer maxLuminanceBuffer;
uint32_t imageWidth=0;
uint32_t imageHeight=0;
  };

  GiOidnState* giOidnCreateState(CgpuDevice device,
                                 GiGlslShaderGen& shaderGen,
                                 GgpuStager& stager,
                                 GgpuDelayedResourceDestroyer& resourceDestroyer,
                                 const GiTzaTensorDescriptions& tensorDescriptions,
                                 const uint8_t* tensorData)
  {
    std::vector<uint8_t> spv;
    if (!shaderGen.generateMaxLuminanceReductionSpirv(spv))
    {
      GB_FATAL("failed to compile shader");
    }

    CgpuShader shader;
    if (!cgpuCreateShader(device, { .size = spv.size(), .source = spv.data(), .stageFlags = CGPU_SHADER_STAGE_FLAG_COMPUTE }, &shader))
    {
      GB_FATAL("failed to create shader");
    }

    CgpuPipeline pipeline;
    if (!cgpuCreateComputePipeline(device, { .shader = shader , .debugName = "Max luminance reduction" }, &pipeline))
    {
      GB_FATAL("failed to create pipeline");
    }
    cgpuDestroyShader(device, shader);

    CgpuBuffer buffer;
    CgpuBufferUsageFlags usage = CGPU_BUFFER_USAGE_FLAG_STORAGE_BUFFER | CGPU_BUFFER_USAGE_FLAG_TRANSFER_DST;
    if (!cgpuCreateBuffer(device, { .usage = usage, .size = sizeof(float) }, &buffer))
    {
      GB_FATAL("failed to allocate OIDN buffer");
    }

    return new GiOidnState {
      GiOidnNet { device, shaderGen, stager, resourceDestroyer, tensorDescriptions, tensorData },
      pipeline,
      buffer
    };
  }

  void giOidnDestroyState(GiOidnState* state)
  {
    // TODO: destroy maxL buffer, pipeline
    delete state;
  }

  bool giOidnUpdateState(GiOidnState* state, CgpuDevice device, uint32_t imageWidth, uint32_t imageHeight)
  {
state->imageWidth = imageWidth;
state->imageHeight = imageHeight;
    state->net.updateViewport(device, imageWidth, imageHeight);
    return true;
  }

  CgpuBuffer giOidnGetInputBuffer(GiOidnState* state)
  {
    return state->net.getInputBuffer();
  }

  CgpuBuffer giOidnGetOutputBuffer(GiOidnState* state)
  {
    return state->net.getOutputBuffer();
  }

  void giOidnRender(GiOidnState* state, CgpuCommandBuffer commandBuffer)
  {
    cgpuCmdFillBuffer(commandBuffer, state->maxLuminanceBuffer);

    CgpuBufferMemoryBarrier bufferBarriers[2] = {
      {
        .buffer = state->net.getInputBuffer(),
        .srcStageMask = CGPU_PIPELINE_STAGE_FLAG_RAY_TRACING_SHADER | CGPU_PIPELINE_STAGE_FLAG_COMPUTE_SHADER,
        .srcAccessMask = CGPU_MEMORY_ACCESS_FLAG_SHADER_WRITE,
        .dstStageMask = CGPU_PIPELINE_STAGE_FLAG_COMPUTE_SHADER,
        .dstAccessMask = CGPU_MEMORY_ACCESS_FLAG_SHADER_READ
      },
      {
        .buffer = state->maxLuminanceBuffer,
        .srcStageMask = CGPU_PIPELINE_STAGE_FLAG_TRANSFER,
        .srcAccessMask = CGPU_MEMORY_ACCESS_FLAG_MEMORY_WRITE,
        .dstStageMask = CGPU_PIPELINE_STAGE_FLAG_COMPUTE_SHADER,
        .dstAccessMask = CGPU_MEMORY_ACCESS_FLAG_SHADER_WRITE
      }
    };

    CgpuPipelineBarrier barrier = {
      .bufferBarrierCount = 2,
      .bufferBarriers = bufferBarriers
    };
    cgpuCmdPipelineBarrier(commandBuffer, &barrier);

    rp::PushConstants pushData = {
      .imageWidth = state->imageWidth,
      .imageHeight = state->imageHeight
    };

    std::vector<CgpuBufferBinding> bufferBindings = {
      CgpuBufferBinding{.binding = rp_ml::BINDING_INDEX_INPUT_BUF, .buffer = state->net.getInputBuffer() },
      CgpuBufferBinding{.binding = rp_ml::BINDING_INDEX_OUTPUT_BUF, .buffer = state->maxLuminanceBuffer }
    };

    CgpuBindings bindings0 = { .bufferCount = (uint32_t) bufferBindings.size(), .buffers = bufferBindings.data() };
    cgpuCmdUpdateBindings(commandBuffer, state->maxLuminanceReduction, 0/*descriptorSetIndex*/, &bindings0);

    cgpuCmdBindPipeline(commandBuffer, state->maxLuminanceReduction);

    cgpuCmdPushConstants(commandBuffer, state->maxLuminanceReduction, CGPU_SHADER_STAGE_FLAG_COMPUTE, sizeof(pushData), &pushData);

    uint32_t wgCountX = (state->imageWidth + rp_ml::WG_SIZE_X - 1) / rp_ml::WG_SIZE_X;
    uint32_t wgCountY = (state->imageHeight + rp_ml::WG_SIZE_Y - 1) / rp_ml::WG_SIZE_Y;
    cgpuCmdDispatch(commandBuffer, wgCountX, wgCountY, 1);

    state->net.runInference(commandBuffer, state->maxLuminanceBuffer);
  }
}
