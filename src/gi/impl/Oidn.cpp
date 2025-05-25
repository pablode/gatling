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

#include <gtl/ggpu/DelayedResourceDestroyer.h>
#include <gtl/ggpu/Stager.h>
#include <gtl/gb/Log.h>
#include <gtl/gb/Fmt.h>

#include <functional>

namespace gtl
{
  namespace rp = shader_interface::rp_oidn;

#if 1
  class GiOidnNet
  {
  private:
    enum class Buffer
    {
      Pool0, Pool1, Pool2, Pool3, Output, Scratch, COUNT
    };

    enum class PostOp
    {
      None, MaxPool, Upsample
    };

    struct PipelineStep
    {
      CgpuPipeline pipeline;
      Buffer input1;
      Buffer input2;
      Buffer output;
      uint32_t weightOffset;
      uint32_t biasOffset;
      GiGlslShaderGen::OidnOp op;
    };

  private:
    CgpuBuffer m_pool0;
    CgpuBuffer m_pool1;
    CgpuBuffer m_pool2;
    CgpuBuffer m_pool3;
    CgpuBuffer m_scratchMem;
    CgpuBuffer m_outputPool;

    CgpuBuffer m_tensorBuffer;

    std::vector<PipelineStep> m_steps;

    uint32_t m_imageWidth; // 16 byte aligned
    uint32_t m_imageHeight; // 16 byte aligned

    GgpuDelayedResourceDestroyer& m_resourceDestroyer;

    uint32_t m_bufferSizeMuls[int(Buffer::COUNT)];

  private:
    struct BuildContext
    {
      CgpuDevice device;
      GiGlslShaderGen& shaderGen;
      GgpuStager& stager;
      const GiTzaTensorDescriptions& tensorDescriptions;
      const uint8_t* tensorData;

      int depth = 0; // track up- and downsampling
      std::unordered_map<std::string, uint32_t> tensorOffsets;
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

        c.tensorOffsets[ntPair.first] = dataOffset;

        if (!c.stager.stageToBuffer(&c.tensorData[tensorDesc.dataOffset], dataSize, m_tensorBuffer, dataOffset))
        {
          GB_FATAL("failed to upload tensor");
        }

        dataOffset += dataSize;
      }
    }

    std::string makeConvPipelineDebugName(int inDims, int outDims, GiGlslShaderGen::OidnOp op)
    {
      const char* opStr = "";
      if (op == GiGlslShaderGen::OidnOp::Convolve) { opStr = "Convolve"; }
      if (op == GiGlslShaderGen::OidnOp::MaxPool) { opStr = "MaxPool"; }
      if (op == GiGlslShaderGen::OidnOp::Upsample) { opStr = "Upsample"; }
      if (op == GiGlslShaderGen::OidnOp::CopyChannels) { opStr = "CopyChannels"; }
      if (op == GiGlslShaderGen::OidnOp::Join) { opStr = "Join"; }

      return GB_FMT("Oidn_{}_->{}", opStr, inDims, outDims);
    }

    void addConvReLU(BuildContext& c, Buffer input, Buffer output, std::string_view tensor, PostOp postOp = PostOp::None)
    {
      auto pushPipeline = [&](GiGlslShaderGen::OidnOp op)
      {
        std::string weightName = GB_FMT("{}.weight", tensor);
        std::string biasName = GB_FMT("{}.bias", tensor);

        auto ntIt = c.tensorDescriptions.find(weightName);
        if (ntIt == c.tensorDescriptions.end())
        {
          GB_FATAL("tensor not loaded");
        }

        const auto& tensorDesc = ntIt->second;
        int inDims = tensorDesc.dimensions[1]; // OIHW
        int outDims = tensorDesc.dimensions[0]; // TODO: consider join

        std::string debugName = makeConvPipelineDebugName(inDims, outDims, op);

        GiGlslShaderGen::OidnParams sgParams{
          .inChannelCount = inDims,
          .outChannelCount = outDims,
          .op = op
        };

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

        uint32_t inSizeMul = inDims * std::pow(2, c.depth) * sizeof(float)/2;

        if (op == GiGlslShaderGen::OidnOp::MaxPool)
        {
          c.depth++;
        }
        else if (op == GiGlslShaderGen::OidnOp::Upsample)
        {
          c.depth--;
        }

        uint32_t outTypeSize = (op == GiGlslShaderGen::OidnOp::CopyChannels) ? 4/*f32*/ : 2/*f16*/;
        uint32_t outSizeMul = outDims * std::pow(2, c.depth) * outTypeSize;

        m_bufferSizeMuls[int(input)] = std::max(m_bufferSizeMuls[int(input)], inSizeMul);
        m_bufferSizeMuls[int(output)] = std::max(m_bufferSizeMuls[int(output)], outSizeMul);

        m_steps.push_back(PipelineStep{
          .pipeline = pipeline,
          .input1 = input,
          .output = output,
          .weightOffset = c.tensorOffsets[weightName],
          .biasOffset = c.tensorOffsets[biasName],
          .op = op
        });
      };

      // TODO: currently the post ops are separate steps
      pushPipeline(GiGlslShaderGen::OidnOp::Convolve);

      if (postOp == PostOp::MaxPool)
      {
        pushPipeline(GiGlslShaderGen::OidnOp::MaxPool);
      }
      else if (postOp == PostOp::Upsample)
      {
        pushPipeline(GiGlslShaderGen::OidnOp::Upsample);
      }
      else if (postOp != PostOp::None)
      {
        GB_FATAL("post op not handled");
      }
    }

    void addJoin(BuildContext& c, Buffer input1, Buffer input2, Buffer output)
    {
      // TODO: remove this function? add PostOp => handling above?
    }

    void reallocBuffers(CgpuDevice device, uint32_t width, uint32_t height)
    {
      // TODO: destroy old buffers

      uint64_t pixelCount = width * height;
      uint64_t pool0Size = pixelCount * m_bufferSizeMuls[int(Buffer::Pool0)];

      CgpuBufferUsageFlags pool0Usage = CGPU_BUFFER_USAGE_FLAG_STORAGE_BUFFER | CGPU_BUFFER_USAGE_FLAG_TRANSFER_SRC |
                                              CGPU_BUFFER_USAGE_FLAG_TRANSFER_DST;
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
      CgpuBufferUsageFlags scratchUsage = CGPU_BUFFER_USAGE_FLAG_STORAGE_BUFFER | CGPU_BUFFER_USAGE_FLAG_TRANSFER_DST/*join*/;
      if (!cgpuCreateBuffer(device, { .usage = scratchUsage, .size = scratchSize }, &m_scratchMem))
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
      BuildContext c{ device, shaderGen, stager, tensorDescs, tensorData};

      uploadTensors(c);

      for (uint32_t& sizeMul :  m_bufferSizeMuls)
      {
        sizeMul = 0;
      }

      addConvReLU(c, Buffer::Pool0, Buffer::Scratch, "enc_conv0");
      addConvReLU(c, Buffer::Scratch, Buffer::Pool1, "enc_conv1", PostOp::MaxPool);
      addConvReLU(c, Buffer::Pool1, Buffer::Pool2, "enc_conv2", PostOp::MaxPool);
      addConvReLU(c, Buffer::Pool2, Buffer::Pool3, "enc_conv3", PostOp::MaxPool);
      addConvReLU(c, Buffer::Pool3, Buffer::Scratch, "enc_conv4", PostOp::MaxPool);
      addConvReLU(c, Buffer::Scratch, Buffer::Scratch, "enc_conv5a");
      addConvReLU(c, Buffer::Scratch, Buffer::Scratch, "enc_conv5b", PostOp::Upsample);
      addJoin(c, Buffer::Scratch, Buffer::Pool3, Buffer::Scratch);
      addConvReLU(c, Buffer::Scratch, Buffer::Scratch, "dec_conv4a");
      addConvReLU(c, Buffer::Scratch, Buffer::Scratch, "dec_conv4b", PostOp::Upsample);
      addJoin(c, Buffer::Scratch, Buffer::Pool2, Buffer::Scratch);
      addConvReLU(c, Buffer::Scratch, Buffer::Scratch, "dec_conv3a");
      addConvReLU(c, Buffer::Scratch, Buffer::Scratch, "dec_conv3b", PostOp::Upsample);
      addJoin(c, Buffer::Scratch, Buffer::Pool1, Buffer::Scratch);
      addConvReLU(c, Buffer::Scratch, Buffer::Scratch, "dec_conv2a");
      addConvReLU(c, Buffer::Scratch, Buffer::Scratch, "dec_conv2b", PostOp::Upsample);
      addJoin(c, Buffer::Scratch, Buffer::Pool0, Buffer::Scratch);
      addConvReLU(c, Buffer::Scratch, Buffer::Scratch, "dec_conv1a");
      addConvReLU(c, Buffer::Scratch, Buffer::Scratch, "dec_conv1b");
      addConvReLU(c, Buffer::Scratch, Buffer::Output, "dec_conv0");

      if (c.depth != 0) { GB_FATAL("invalid network architecture"); }
    }

    // TODO: need to call this before getting pool0 outside
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

    void runInference(uint32_t width, uint32_t height)
    {
      // TODO
    }
  };
#endif












  struct GiOidnPipelines
  {
    CgpuPipeline debug;
    CgpuPipeline debug2;
    CgpuPipeline conv9_32;
    CgpuPipeline conv32_32;
    CgpuPipeline maxPool32;
    CgpuPipeline maxPool32_debug;
    CgpuPipeline conv32_48;
    CgpuPipeline maxPool48;
    CgpuPipeline conv48_64;
    CgpuPipeline maxPool64;
    CgpuPipeline conv64_80;
    CgpuPipeline maxPool80;
    CgpuPipeline conv80_96;
    CgpuPipeline upsample96a;
    CgpuPipeline upsample96b;
    CgpuPipeline conv160_112;
    CgpuPipeline conv112_112;
    CgpuPipeline upsample112;
    CgpuPipeline conv160_96;
    // avoid descriptor set binding woes...
    CgpuPipeline conv96_96a;
    CgpuPipeline conv96_96b;
    CgpuPipeline conv128_64;
    CgpuPipeline conv64_64;
    CgpuPipeline upsample64;
    CgpuPipeline conv73_64;
    CgpuPipeline conv64_32;
    CgpuPipeline conv32_3;
    CgpuPipeline copyToOutput;
    CgpuPipeline join96_64;
    CgpuPipeline join112_48;
    CgpuPipeline join96_32;
    CgpuPipeline join64_9;
  };

  struct GiOidnBufferOffsets
  {
    uint32_t encConv0_weight;
    uint32_t encConv0_bias;
    uint32_t encConv1_weight;
    uint32_t encConv1_bias;
    uint32_t encConv2_weight;
    uint32_t encConv2_bias;
    uint32_t encConv3_weight;
    uint32_t encConv3_bias;
    uint32_t encConv4_weight;
    uint32_t encConv4_bias;
    uint32_t encConv5a_weight;
    uint32_t encConv5a_bias;
    uint32_t encConv5b_weight;
    uint32_t encConv5b_bias;
    uint32_t decConv4a_weight;
    uint32_t decConv4a_bias;
    uint32_t decConv4b_weight;
    uint32_t decConv4b_bias;
    uint32_t decConv3a_weight;
    uint32_t decConv3a_bias;
    uint32_t decConv3b_weight;
    uint32_t decConv3b_bias;
    uint32_t decConv2a_weight;
    uint32_t decConv2a_bias;
    uint32_t decConv2b_weight;
    uint32_t decConv2b_bias;
    uint32_t decConv1a_weight;
    uint32_t decConv1a_bias;
    uint32_t decConv1b_weight;
    uint32_t decConv1b_bias;
    uint32_t decConv0_weight;
    uint32_t decConv0_bias;
  };

  struct GiOidnState
  {
    GiOidnPipelines pipelines;
    GiOidnBufferOffsets offsets;

    CgpuBuffer weightBuffer;
    CgpuBuffer biasBuffer;

    uint32_t imageWidth = 0;
    uint32_t imageHeight = 0;

    CgpuBuffer pool0;
    CgpuBuffer pool1;
    CgpuBuffer pool2;
    CgpuBuffer pool3;

    CgpuBuffer pingPongData[2];

    CgpuBuffer outputImage;

    GgpuDelayedResourceDestroyer resourceDestroyer;
  };

  static GiOidnPipelines giOidnCreatePipelines(CgpuDevice device, GiGlslShaderGen& shaderGen)
  {
    auto createPipeline = [&](const GiGlslShaderGen::OidnParams params)
    {
      std::vector<uint8_t> spv;
      if (!shaderGen.generateDenoisingSpirv(params, spv))
      {
        GB_FATAL("failed to compile OIDN shader");
      }

      CgpuShader shader;
      if (!cgpuCreateShader(device, {
                              .size = spv.size(),
                              .source = spv.data(),
                              .stageFlags = CGPU_SHADER_STAGE_FLAG_COMPUTE
                            }, &shader))
      {
        GB_FATAL("failed to create OIDN shader");
      }

      const char* opName = nullptr;
      if (params.op == GiGlslShaderGen::OidnOp::Convolve) opName = "Convolve";
      if (params.op == GiGlslShaderGen::OidnOp::MaxPool) opName = "MaxPool";
      if (params.op == GiGlslShaderGen::OidnOp::Upsample) opName = "Upsample";
      if (params.op == GiGlslShaderGen::OidnOp::CopyChannels) opName = "CopyChannels";
      if (params.op == GiGlslShaderGen::OidnOp::Join) opName = "Join";
      std::string debugName = GB_FMT("Oidn_{}_{}->{}", opName, params.inChannelCount, params.outChannelCount);
      GB_LOG(" {}", debugName);

      CgpuPipeline pipeline;
      if (!cgpuCreateComputePipeline(device, { .shader = shader , .debugName = debugName.c_str() }, &pipeline))
      {
        GB_FATAL("failed to create OIDN pipeline");
      }

      cgpuDestroyShader(device, shader);
      return pipeline;
    };

    GB_LOG("creating OIDN pipelines:");
    return GiOidnPipelines{
      .debug = createPipeline(GiGlslShaderGen::OidnParams{ .inChannelCount = 32, .outChannelCount = 32, .op = GiGlslShaderGen::OidnOp::Upsample }),
      .debug2 = createPipeline(GiGlslShaderGen::OidnParams{ .inChannelCount = 32, .outChannelCount = 32, .op = GiGlslShaderGen::OidnOp::Upsample }),
      .conv9_32 = createPipeline(GiGlslShaderGen::OidnParams{ .inChannelCount = 9, .outChannelCount = 32 }),
      .conv32_32 = createPipeline(GiGlslShaderGen::OidnParams{ .inChannelCount = 32, .outChannelCount = 32 }),
      .maxPool32 = createPipeline(GiGlslShaderGen::OidnParams{ .inChannelCount = 32, .outChannelCount = 32, .op = GiGlslShaderGen::OidnOp::MaxPool }),
      .maxPool32_debug = createPipeline(GiGlslShaderGen::OidnParams{ .inChannelCount = 32, .outChannelCount = 32, .op = GiGlslShaderGen::OidnOp::MaxPool }),
      .conv32_48 = createPipeline(GiGlslShaderGen::OidnParams{ .inChannelCount = 32, .outChannelCount = 48 }),
      .maxPool48 = createPipeline(GiGlslShaderGen::OidnParams{ .inChannelCount = 48, .outChannelCount = 48, .op = GiGlslShaderGen::OidnOp::MaxPool }),
      .conv48_64 = createPipeline(GiGlslShaderGen::OidnParams{ .inChannelCount = 48, .outChannelCount = 64 }),
      .maxPool64 = createPipeline(GiGlslShaderGen::OidnParams{ .inChannelCount = 64, .outChannelCount = 64, .op = GiGlslShaderGen::OidnOp::MaxPool }),
      .conv64_80 = createPipeline(GiGlslShaderGen::OidnParams{ .inChannelCount = 64, .outChannelCount = 80 }),
      .maxPool80 = createPipeline(GiGlslShaderGen::OidnParams{ .inChannelCount = 80, .outChannelCount = 80, .op = GiGlslShaderGen::OidnOp::MaxPool }),
      .conv80_96 = createPipeline(GiGlslShaderGen::OidnParams{ .inChannelCount = 80, .outChannelCount = 96 }),
      .upsample96a = createPipeline(GiGlslShaderGen::OidnParams{ .inChannelCount = 96, .outChannelCount = 96, .op = GiGlslShaderGen::OidnOp::Upsample }),
      .upsample96b = createPipeline(GiGlslShaderGen::OidnParams{ .inChannelCount = 96, .outChannelCount = 96, .op = GiGlslShaderGen::OidnOp::Upsample }),
      .conv160_112 = createPipeline(GiGlslShaderGen::OidnParams{ .inChannelCount = 160, .outChannelCount = 112 }),
      .conv112_112 = createPipeline(GiGlslShaderGen::OidnParams{ .inChannelCount = 112, .outChannelCount = 112 }),
      .upsample112 = createPipeline(GiGlslShaderGen::OidnParams{ .inChannelCount = 112, .outChannelCount = 112, .op = GiGlslShaderGen::OidnOp::Upsample }),
      .conv160_96 = createPipeline(GiGlslShaderGen::OidnParams{ .inChannelCount = 160, .outChannelCount = 96 }),
      .conv96_96a = createPipeline(GiGlslShaderGen::OidnParams{ .inChannelCount = 96, .outChannelCount = 96 }),
      .conv96_96b = createPipeline(GiGlslShaderGen::OidnParams{ .inChannelCount = 96, .outChannelCount = 96 }),
      .conv128_64 = createPipeline(GiGlslShaderGen::OidnParams{ .inChannelCount = 128, .outChannelCount = 64 }),
      .conv64_64 = createPipeline(GiGlslShaderGen::OidnParams{ .inChannelCount = 64, .outChannelCount = 64 }),
      .upsample64 = createPipeline(GiGlslShaderGen::OidnParams{ .inChannelCount = 64, .outChannelCount = 64, .op = GiGlslShaderGen::OidnOp::Upsample }),
      .conv73_64 = createPipeline(GiGlslShaderGen::OidnParams{ .inChannelCount = 73, .outChannelCount = 64 }),
      .conv64_32 = createPipeline(GiGlslShaderGen::OidnParams{ .inChannelCount = 64, .outChannelCount = 32 }),
      .conv32_3 = createPipeline(GiGlslShaderGen::OidnParams{ .inChannelCount = 32, .outChannelCount = 3 }),
      .copyToOutput = createPipeline(GiGlslShaderGen::OidnParams{ .inChannelCount = 3, .outChannelCount = 4, .op = GiGlslShaderGen::OidnOp::CopyChannels }),
      .join96_64 = createPipeline(GiGlslShaderGen::OidnParams{ .inChannelCount = 96, .outChannelCount = 64, .op = GiGlslShaderGen::OidnOp::Join }),
      .join112_48 = createPipeline(GiGlslShaderGen::OidnParams{ .inChannelCount = 112, .outChannelCount = 48, .op = GiGlslShaderGen::OidnOp::Join }),
      .join96_32 = createPipeline(GiGlslShaderGen::OidnParams{ .inChannelCount = 96, .outChannelCount = 32, .op = GiGlslShaderGen::OidnOp::Join }),
      .join64_9 = createPipeline(GiGlslShaderGen::OidnParams{ .inChannelCount = 64, .outChannelCount = 9, .op = GiGlslShaderGen::OidnOp::Join })
    };
  }

  GiOidnState* giOidnCreateState(CgpuDevice device,
                                 GiGlslShaderGen& shaderGen,
                                 GgpuStager& stager,
                                 GgpuDelayedResourceDestroyer& resourceDestroyer,
                                 const GiTzaTensorDescriptions& tensorDescriptions,
                                 const uint8_t* tensorData)
  {
    const std::vector<const char*> TENSOR_NAMES = { "enc_conv0", "enc_conv1", "enc_conv2", "enc_conv3",
      "enc_conv4", "enc_conv5a", "enc_conv5b", "dec_conv4a", "dec_conv4b", "dec_conv3a", "dec_conv3b",
      "dec_conv2a", "dec_conv2b", "dec_conv1a", "dec_conv1b", "dec_conv0" };

    size_t weightBufferSize = 0;
    for (const char* name : TENSOR_NAMES)
    {
      auto newName = std::string(name) + ".weight";
      assert(tensorDescriptions.count(newName) > 0);
      const GiTzaTensorDescription& desc = tensorDescriptions.at(newName);
      weightBufferSize += (desc.dataSize + 4 - 1) / 4  * 4; // required align
    }
weightBufferSize*=2; // just to be sure. TODO: bc it does not incorporate padding introduced below
    size_t biasBufferSize = 0;
    for (const char* name : TENSOR_NAMES)
    {
      auto newName = std::string(name) + ".bias";
      assert(tensorDescriptions.count(newName) > 0);
      const GiTzaTensorDescription& desc = tensorDescriptions.at(newName);
      biasBufferSize += (desc.dataSize + 4 - 1) / 4  * 4; // required align
    }
biasBufferSize*=2; //just to be sure

    CgpuBufferUsageFlags tensorBufferUsage = CGPU_BUFFER_USAGE_FLAG_STORAGE_BUFFER | CGPU_BUFFER_USAGE_FLAG_TRANSFER_DST;

    CgpuBuffer weightBuffer, biasBuffer;
    if (!cgpuCreateBuffer(device, { .usage = tensorBufferUsage, .size = weightBufferSize }, &weightBuffer) ||
        !cgpuCreateBuffer(device, { .usage = tensorBufferUsage, .size = biasBufferSize }, &biasBuffer))
    {
      GB_FATAL("failed to alloc tensor buffer");
    }

    uint32_t nextWeightOffset = 0;
    const auto uploadWeights = [&](const char* name) {
      assert(tensorDescriptions.count(name) > 0);
      const GiTzaTensorDescription& desc = tensorDescriptions.at(name);
      uint32_t dataOffset = (nextWeightOffset + 4 - 1) / 4 * 4; // required align
      uint32_t dataSize = (desc.dataSize + 4 - 1) / 4 * 4; // TODO: this can read OOB...

      bool s = stager.stageToBuffer(&tensorData[desc.dataOffset], dataSize, weightBuffer, dataOffset);
      if (!s) GB_FATAL("failed to stage data");

      nextWeightOffset += dataSize;
      return dataOffset/4;
    };

    uint32_t nextBiasOffset = 0;
    const auto uploadBiases = [&](const char* name) {
      assert(tensorDescriptions.count(name) > 0);
      const GiTzaTensorDescription& desc = tensorDescriptions.at(name);
      uint32_t dataOffset = (nextBiasOffset + 4 - 1) / 4 * 4; // required align
      uint32_t dataSize = (desc.dataSize + 4 - 1) / 4 * 4; // TODO: see above

      bool s = stager.stageToBuffer(&tensorData[desc.dataOffset], dataSize, biasBuffer, dataOffset);
      if (!s) GB_FATAL("failed to stage data");

      nextBiasOffset += dataSize;
      return dataOffset/4;
    };

    GiOidnBufferOffsets offsets = {
      .encConv0_weight = uploadWeights("enc_conv0.weight"),
      .encConv0_bias = uploadBiases("enc_conv0.bias"),
      .encConv1_weight = uploadWeights("enc_conv1.weight"),
      .encConv1_bias = uploadBiases("enc_conv1.bias"),
      .encConv2_weight = uploadWeights("enc_conv2.weight"),
      .encConv2_bias = uploadBiases("enc_conv2.bias"),
      .encConv3_weight = uploadWeights("enc_conv3.weight"),
      .encConv3_bias = uploadBiases("enc_conv3.bias"),
      .encConv4_weight = uploadWeights("enc_conv4.weight"),
      .encConv4_bias = uploadBiases("enc_conv4.bias"),
      .encConv5a_weight = uploadWeights("enc_conv5a.weight"),
      .encConv5a_bias = uploadBiases("enc_conv5a.bias"),
      .encConv5b_weight = uploadWeights("enc_conv5b.weight"),
      .encConv5b_bias = uploadBiases("enc_conv5b.bias"),
      .decConv4a_weight = uploadWeights("dec_conv4a.weight"),
      .decConv4a_bias = uploadBiases("dec_conv4a.bias"),
      .decConv4b_weight = uploadWeights("dec_conv4b.weight"),
      .decConv4b_bias = uploadBiases("dec_conv4b.bias"),
      .decConv3a_weight = uploadWeights("dec_conv3a.weight"),
      .decConv3a_bias = uploadBiases("dec_conv3a.bias"),
      .decConv3b_weight = uploadWeights("dec_conv3b.weight"),
      .decConv3b_bias = uploadBiases("dec_conv3b.bias"),
      .decConv2a_weight = uploadWeights("dec_conv2a.weight"),
      .decConv2a_bias = uploadBiases("dec_conv2a.bias"),
      .decConv2b_weight = uploadWeights("dec_conv2b.weight"),
      .decConv2b_bias = uploadBiases("dec_conv2b.bias"),
      .decConv1a_weight = uploadWeights("dec_conv1a.weight"),
      .decConv1a_bias = uploadBiases("dec_conv1a.bias"),
      .decConv1b_weight = uploadWeights("dec_conv1b.weight"),
      .decConv1b_bias = uploadBiases("dec_conv1b.bias"),
      .decConv0_weight = uploadWeights("dec_conv0.weight"),
      .decConv0_bias = uploadBiases("dec_conv0.bias")
    };

// TODO: temp
GB_LOG("encConv0_weight: {}", offsets.encConv0_weight);
GB_LOG("encConv0_bias: {}", offsets.encConv0_bias);
GB_LOG("encConv1_weight: {}", offsets.encConv1_weight);
GB_LOG("encConv1_bias: {}", offsets.encConv1_bias);
GB_LOG("encConv2_weight: {}", offsets.encConv2_weight);
GB_LOG("encConv2_bias: {}", offsets.encConv2_bias);

    GiOidnPipelines pipelines = giOidnCreatePipelines(device, shaderGen);

    stager.flush(); // optional

    auto* state = new GiOidnState {
      .pipelines = pipelines,
      .offsets = offsets,
      .weightBuffer = weightBuffer,
      .biasBuffer = biasBuffer,
      .resourceDestroyer = resourceDestroyer
    };

    return state;
  }

  void giOidnDestroyState(GiOidnState* state)
  {
    // TODO: destroy pipelines
    // TODO: destroy buffers
    delete state;
  }

  bool giOidnUpdateState(GiOidnState* state, CgpuDevice device, uint32_t imageWidth, uint32_t imageHeight)
  {
    if (state->imageWidth == imageWidth && state->imageHeight == imageHeight)
    {
      return true;
    }

// TODO: need to align to 16 bytes
//imageWidth = (imageWidth+15)/16*16;
//imageHeight = (imageHeight+15)/16*16;

    // TODO: destroy buffers with delayedResourceDestroyer

    uint64_t pool0Size = (imageWidth/1) * (imageHeight/1) * sizeof(float)/2 * 9;
    uint64_t pool1Size = (imageWidth/2) * (imageHeight/2) * sizeof(float)/2 * 32;
    uint64_t pool2Size = (imageWidth/4) * (imageHeight/4) * sizeof(float)/2 * 48;
    uint64_t pool3Size = (imageWidth/8) * (imageHeight/8) * sizeof(float)/2 * 64;

    CgpuBufferUsageFlags pool0BufferUsage = CGPU_BUFFER_USAGE_FLAG_STORAGE_BUFFER |
                                            CGPU_BUFFER_USAGE_FLAG_TRANSFER_SRC |
                                            CGPU_BUFFER_USAGE_FLAG_TRANSFER_DST;
    if (!cgpuCreateBuffer(device, { .usage = pool0BufferUsage, .size = pool0Size }, &state->pool0))
    {
      GB_FATAL("failed to allocate OIDN buffer");
    }

    // NOTE: at Full HD, a single of the buffers is ~1.2GB
    CgpuBufferUsageFlags poolNBufferUsage = CGPU_BUFFER_USAGE_FLAG_STORAGE_BUFFER | CGPU_BUFFER_USAGE_FLAG_TRANSFER_SRC;
    if (!cgpuCreateBuffer(device, { .usage = poolNBufferUsage, .size = pool1Size }, &state->pool1) ||
        !cgpuCreateBuffer(device, { .usage = poolNBufferUsage, .size = pool2Size }, &state->pool2) ||
        !cgpuCreateBuffer(device, { .usage = poolNBufferUsage, .size = pool3Size }, &state->pool3))
    {
      GB_FATAL("failed to allocate OIDN buffer");
    }

    uint64_t pingPongSliceSize = imageWidth * imageHeight * sizeof(float)/2 * 73/*max usage*/;
    CgpuBufferUsageFlags pingPongBufferUsage = CGPU_BUFFER_USAGE_FLAG_STORAGE_BUFFER |
                                               CGPU_BUFFER_USAGE_FLAG_TRANSFER_DST | // for join
                                               CGPU_BUFFER_USAGE_FLAG_TRANSFER_SRC; // TODO: only needed for debug viz
    if (!cgpuCreateBuffer(device, { .usage = pingPongBufferUsage, .size = pingPongSliceSize }, &state->pingPongData[0]) ||
        !cgpuCreateBuffer(device, { .usage = pingPongBufferUsage, .size = pingPongSliceSize }, &state->pingPongData[1]))
    {
      GB_FATAL("failed to allocate OIDN buffer");
    }

    uint32_t outputImageSize = (imageWidth * imageHeight) * sizeof(float) * 4;
    if (!cgpuCreateBuffer(device, { .usage = CGPU_BUFFER_USAGE_FLAG_STORAGE_BUFFER | CGPU_BUFFER_USAGE_FLAG_TRANSFER_SRC, .size = outputImageSize }, &state->outputImage))
    {
      GB_FATAL("failed to allocate OIDN buffer");
    }

    state->imageWidth = imageWidth;
    state->imageHeight = imageHeight;
    return true;
  }

  CgpuBuffer giOidnGetInputBuffer(GiOidnState* state)
  {
    return state->pool0;
  }

  CgpuBuffer giOidnGetOutputBuffer(GiOidnState* state)
  {
    return state->outputImage;
  }

  // TODO: for starters, copy 3 channels of input AOV to color AOV (viz aux normal & albedo)
  void giOidnRender(GiOidnState* state, CgpuCommandBuffer commandBuffer)
  {
    // TODO: apparently the input W, H need to be aligned to 16 pixels !
    const auto& pipelines = state->pipelines;
    const auto& offsets = state->offsets;

    uint32_t imageWidth = state->imageWidth;
    uint32_t imageHeight = state->imageHeight;

    auto dispatchPipeline = [&](CgpuPipeline pipeline, uint32_t weightOffset, uint32_t biasOffset,
                                CgpuBuffer inBuffer, CgpuBuffer outBuffer)
    {
      CgpuBufferMemoryBarrier bufferBarrier {
        .buffer = inBuffer,
        .srcStageMask = CGPU_PIPELINE_STAGE_FLAG_RAY_TRACING_SHADER | CGPU_PIPELINE_STAGE_FLAG_COMPUTE_SHADER |
                        CGPU_PIPELINE_STAGE_FLAG_TRANSFER, // TODO: minimize
        .srcAccessMask = CGPU_MEMORY_ACCESS_FLAG_SHADER_WRITE | CGPU_MEMORY_ACCESS_FLAG_TRANSFER_WRITE, // TODO: minimize
        .dstStageMask = CGPU_PIPELINE_STAGE_FLAG_COMPUTE_SHADER,
        .dstAccessMask = CGPU_MEMORY_ACCESS_FLAG_SHADER_WRITE
      };

      CgpuPipelineBarrier barrier = {
        .bufferBarrierCount = 1,
        .bufferBarriers = &bufferBarrier
      };
      cgpuCmdPipelineBarrier(commandBuffer, &barrier);

      rp::PushConstants pushData = {
        .imageWidth = imageWidth,
        .imageHeight = imageHeight,
        .weightOffset = weightOffset,
        .biasOffset = biasOffset
      };

      std::array<CgpuBufferBinding, 4> bufferBindings = {
        CgpuBufferBinding{ .binding = 0, .buffer = inBuffer },
        CgpuBufferBinding{ .binding = 1, .buffer = outBuffer },
        CgpuBufferBinding{ .binding = 2, .buffer = state->weightBuffer },
        CgpuBufferBinding{ .binding = 3, .buffer = state->biasBuffer }
      };

      CgpuBindings bindings0 = { .bufferCount = (uint32_t) bufferBindings.size(), .buffers = bufferBindings.data() };
      cgpuCmdUpdateBindings(commandBuffer, pipeline, 0/*descriptorSetIndex*/, &bindings0);

      cgpuCmdBindPipeline(commandBuffer, pipeline);

      cgpuCmdPushConstants(commandBuffer, pipeline, CGPU_SHADER_STAGE_FLAG_COMPUTE, sizeof(pushData), &pushData);

      uint32_t wgCountX = (imageWidth + rp::WG_SIZE_X - 1) / rp::WG_SIZE_X;
      uint32_t wgCountY = (imageHeight + rp::WG_SIZE_Y - 1) / rp::WG_SIZE_Y;
      cgpuCmdDispatch(commandBuffer, wgCountX, wgCountY, 1);
    };

    auto joinChannels = [&](CgpuPipeline pipeline, CgpuBuffer inBuffer, CgpuBuffer inBuffer2, CgpuBuffer outBuffer)
    {
      CgpuBufferMemoryBarrier bufferBarriers[2] = {
        CgpuBufferMemoryBarrier{
          .buffer = inBuffer,
          .srcStageMask = CGPU_PIPELINE_STAGE_FLAG_RAY_TRACING_SHADER | CGPU_PIPELINE_STAGE_FLAG_COMPUTE_SHADER |
                          CGPU_PIPELINE_STAGE_FLAG_TRANSFER, // TODO: minimize
          .srcAccessMask = CGPU_MEMORY_ACCESS_FLAG_SHADER_WRITE | CGPU_MEMORY_ACCESS_FLAG_TRANSFER_WRITE, // TODO: minimize
          .dstStageMask = CGPU_PIPELINE_STAGE_FLAG_COMPUTE_SHADER,
          .dstAccessMask = CGPU_MEMORY_ACCESS_FLAG_SHADER_WRITE
        },
        CgpuBufferMemoryBarrier{
          .buffer = inBuffer2,
          .srcStageMask = CGPU_PIPELINE_STAGE_FLAG_RAY_TRACING_SHADER | CGPU_PIPELINE_STAGE_FLAG_COMPUTE_SHADER |
                          CGPU_PIPELINE_STAGE_FLAG_TRANSFER, // TODO: minimize
          .srcAccessMask = CGPU_MEMORY_ACCESS_FLAG_SHADER_WRITE | CGPU_MEMORY_ACCESS_FLAG_TRANSFER_WRITE, // TODO: minimize
          .dstStageMask = CGPU_PIPELINE_STAGE_FLAG_COMPUTE_SHADER,
          .dstAccessMask = CGPU_MEMORY_ACCESS_FLAG_SHADER_WRITE
        }
      };

      CgpuPipelineBarrier barrier = {
        .bufferBarrierCount = 2,
        .bufferBarriers = &bufferBarriers[0]
      };
      cgpuCmdPipelineBarrier(commandBuffer, &barrier);

      rp::PushConstants pushData = {
        .imageWidth = imageWidth,
        .imageHeight = imageHeight
      };

      std::array<CgpuBufferBinding, 5> bufferBindings = {
        CgpuBufferBinding{ .binding = 0, .buffer = inBuffer },
        CgpuBufferBinding{ .binding = 1, .buffer = outBuffer },
        CgpuBufferBinding{ .binding = 2, .buffer = state->weightBuffer },
        CgpuBufferBinding{ .binding = 3, .buffer = state->biasBuffer },
        CgpuBufferBinding{ .binding = 4, .buffer = inBuffer2 }
      };

      CgpuBindings bindings0 = { .bufferCount = (uint32_t) bufferBindings.size(), .buffers = bufferBindings.data() };
      cgpuCmdUpdateBindings(commandBuffer, pipeline, 0/*descriptorSetIndex*/, &bindings0);

      cgpuCmdBindPipeline(commandBuffer, pipeline);

      cgpuCmdPushConstants(commandBuffer, pipeline, CGPU_SHADER_STAGE_FLAG_COMPUTE, sizeof(pushData), &pushData);

      uint32_t wgCountX = (imageWidth + rp::WG_SIZE_X - 1) / rp::WG_SIZE_X;
      uint32_t wgCountY = (imageHeight + rp::WG_SIZE_Y - 1) / rp::WG_SIZE_Y;
      cgpuCmdDispatch(commandBuffer, wgCountX, wgCountY, 1);
    };


    uint32_t i = 0;
    const auto pingPong = state->pingPongData;

#if 0
    imageWidth /= 2;
    imageHeight /= 2;
    imageWidth /= 2;
    imageHeight /= 2;
    imageWidth /= 2;
    imageHeight /= 2;
    imageWidth /= 2;
    imageHeight /= 2;
    imageWidth *= 2;
    imageHeight *= 2;
    imageWidth *= 2;
    imageHeight *= 2;
    imageWidth *= 2;
    imageHeight *= 2;
    imageWidth *= 2;
    imageHeight *= 2;
GB_LOG("iw_new: {}, iw_old: {}", imageWidth, state->imageWidth);
GB_LOG("ih_new: {}, ih_old: {}", imageHeight, state->imageHeight);
#endif
#if 0
    dispatchPipeline(pipelines.conv9_32, offsets.encConv0_weight, offsets.encConv0_bias, state->pool0, pingPong[0]); // 3->32
    imageWidth /= 2;
    imageHeight /= 2;
    dispatchPipeline(pipelines.maxPool32, 0, 0, pingPong[0], pingPong[1]); // downsample
    imageWidth /= 2;
    imageHeight /= 2;
    dispatchPipeline(pipelines.maxPool32_debug, 0, 0, pingPong[1], pingPong[0]); // downsample
    dispatchPipeline(pipelines.debug, 0, 0, pingPong[0], pingPong[1]); // upsample
    imageWidth *= 2;
    imageHeight *= 2;
    dispatchPipeline(pipelines.debug2, 0, 0, pingPong[1], pingPong[0]); // upsample
    imageWidth *= 2;
    imageHeight *= 2;
    dispatchPipeline(pipelines.conv32_3, offsets.decConv0_weight, offsets.decConv0_bias, pingPong[0], pingPong[1]); // 32->3
    dispatchPipeline(pipelines.copyToOutput, 0, 0, pingPong[1], rgbResult); // debug viz to color AOV
return;
#endif

    // l0
    dispatchPipeline(pipelines.conv9_32, offsets.encConv0_weight, offsets.encConv0_bias, state->pool0, pingPong[0]);
    dispatchPipeline(pipelines.conv32_32, offsets.encConv1_weight, offsets.encConv1_bias, pingPong[0], pingPong[1]);
    imageWidth /= 2;
    imageHeight /= 2;
    dispatchPipeline(pipelines.maxPool32, 0, 0, pingPong[1], state->pool1);

    // l1
    dispatchPipeline(pipelines.conv32_48, offsets.encConv2_weight, offsets.encConv2_bias, state->pool1, pingPong[0]);
    imageWidth /= 2;
    imageHeight /= 2;
    dispatchPipeline(pipelines.maxPool48, 0, 0, pingPong[0], state->pool2);

    // l2
    dispatchPipeline(pipelines.conv48_64, offsets.encConv3_weight, offsets.encConv3_bias, state->pool2, pingPong[0]);
    imageWidth /= 2;
    imageHeight /= 2;
    dispatchPipeline(pipelines.maxPool64, 0, 0, pingPong[0], state->pool3);

    // l3
    dispatchPipeline(pipelines.conv64_80, offsets.encConv4_weight, offsets.encConv4_bias, state->pool3, pingPong[0]);
    imageWidth /= 2;
    imageHeight /= 2;
    dispatchPipeline(pipelines.maxPool80, 0, 0, pingPong[0], pingPong[1]);

    // l4
    dispatchPipeline(pipelines.conv80_96, offsets.encConv5a_weight, offsets.encConv5a_bias, pingPong[1], pingPong[0]);
    dispatchPipeline(pipelines.conv96_96a, offsets.encConv5b_weight, offsets.encConv5b_bias, pingPong[0], pingPong[1]);
    dispatchPipeline(pipelines.upsample96a, 0, 0, pingPong[1], pingPong[0]);
    imageWidth *= 2;
    imageHeight *= 2;

    // l3
    joinChannels(pipelines.join96_64, pingPong[0], state->pool3, pingPong[1]);
    dispatchPipeline(pipelines.conv160_112, offsets.decConv4a_weight, offsets.decConv4a_bias, pingPong[1], pingPong[0]);
    dispatchPipeline(pipelines.conv112_112, offsets.decConv4b_weight, offsets.decConv4b_bias, pingPong[0], pingPong[1]);
    dispatchPipeline(pipelines.upsample112, 0, 0, pingPong[1], pingPong[0]);
    imageWidth *= 2;
    imageHeight *= 2;

    // l2
    joinChannels(pipelines.join112_48, pingPong[0], state->pool2, pingPong[1]);
    dispatchPipeline(pipelines.conv160_96, offsets.decConv3a_weight, offsets.decConv3a_bias, pingPong[1], pingPong[0]);
    dispatchPipeline(pipelines.conv96_96b, offsets.decConv3b_weight, offsets.decConv3b_bias, pingPong[0], pingPong[1]);
    dispatchPipeline(pipelines.upsample96b, 0, 0, pingPong[1], pingPong[0]);
    imageWidth *= 2;
    imageHeight *= 2;

    // l1
    joinChannels(pipelines.join96_32, pingPong[0], state->pool1, pingPong[1]);
    dispatchPipeline(pipelines.conv128_64, offsets.decConv2a_weight, offsets.decConv2a_bias, pingPong[1], pingPong[0]);
    dispatchPipeline(pipelines.conv64_64, offsets.decConv2b_weight, offsets.decConv2b_bias, pingPong[0], pingPong[1]);
    dispatchPipeline(pipelines.upsample64, 0, 0, pingPong[1], pingPong[0]);
    imageWidth *= 2;
    imageHeight *= 2;

    // l0
    joinChannels(pipelines.join64_9, pingPong[0], state->pool0, pingPong[1]);
    dispatchPipeline(pipelines.conv73_64, offsets.decConv1a_weight, offsets.decConv1a_bias, pingPong[1], pingPong[0]);
    dispatchPipeline(pipelines.conv64_32, offsets.decConv1b_weight, offsets.decConv1b_bias, pingPong[0], pingPong[1]);
    dispatchPipeline(pipelines.conv32_3, offsets.decConv0_weight, offsets.decConv0_bias, pingPong[1], pingPong[0]);

    dispatchPipeline(pipelines.copyToOutput, 0, 0, pingPong[0], state->outputImage); // debug viz to color AOV

    CgpuBufferMemoryBarrier bufferBarrier {
      .buffer = state->outputImage,
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
}
