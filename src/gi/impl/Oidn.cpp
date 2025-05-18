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

namespace gtl
{
  namespace rp = shader_interface::rp_oidn;

  struct GiOidnPipelines
  {
    CgpuPipeline debug;

    CgpuPipeline conv3_32;
    CgpuPipeline conv32_32;
    CgpuPipeline maxPool32;
    CgpuPipeline conv32_48;
    CgpuPipeline maxPool48;
    CgpuPipeline conv48_64;
    CgpuPipeline maxPool64;
    CgpuPipeline conv64_80;
    CgpuPipeline maxPool80;
    CgpuPipeline conv80_96;
    CgpuPipeline upsample96;
    CgpuPipeline conv160_112;
    CgpuPipeline conv112_112;
    CgpuPipeline upsample112;
    CgpuPipeline conv160_96;
    CgpuPipeline conv96_96;
    CgpuPipeline conv128_64;
    CgpuPipeline conv64_64;
    CgpuPipeline upsample64;
    CgpuPipeline conv73_64;
    CgpuPipeline conv64_32;
    CgpuPipeline conv32_3;
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
      .debug = createPipeline(GiGlslShaderGen::OidnParams{ .inChannelCount = 3, .outChannelCount = 4 }),
      .conv3_32 = createPipeline(GiGlslShaderGen::OidnParams{ .inChannelCount = 3, .outChannelCount = 32 }),
      .conv32_32 = createPipeline(GiGlslShaderGen::OidnParams{ .inChannelCount = 32, .outChannelCount = 32 }),
      .maxPool32 = createPipeline(GiGlslShaderGen::OidnParams{ .inChannelCount = 32, .outChannelCount = 32, .op = GiGlslShaderGen::OidnOp::MaxPool }),
      .conv32_48 = createPipeline(GiGlslShaderGen::OidnParams{ .inChannelCount = 32, .outChannelCount = 48 }),
      .maxPool48 = createPipeline(GiGlslShaderGen::OidnParams{ .inChannelCount = 48, .outChannelCount = 48, .op = GiGlslShaderGen::OidnOp::MaxPool }),
      .conv48_64 = createPipeline(GiGlslShaderGen::OidnParams{ .inChannelCount = 48, .outChannelCount = 64 }),
      .maxPool64 = createPipeline(GiGlslShaderGen::OidnParams{ .inChannelCount = 64, .outChannelCount = 64, .op = GiGlslShaderGen::OidnOp::MaxPool }),
      .conv64_80 = createPipeline(GiGlslShaderGen::OidnParams{ .inChannelCount = 64, .outChannelCount = 80 }),
      .maxPool80 = createPipeline(GiGlslShaderGen::OidnParams{ .inChannelCount = 80, .outChannelCount = 80, .op = GiGlslShaderGen::OidnOp::MaxPool }),
      .conv80_96 = createPipeline(GiGlslShaderGen::OidnParams{ .inChannelCount = 80, .outChannelCount = 96 }),
      .upsample96 = createPipeline(GiGlslShaderGen::OidnParams{ .inChannelCount = 96, .outChannelCount = 96, .op = GiGlslShaderGen::OidnOp::Upsample }),
      .conv160_112 = createPipeline(GiGlslShaderGen::OidnParams{ .inChannelCount = 160, .outChannelCount = 112 }),
      .conv112_112 = createPipeline(GiGlslShaderGen::OidnParams{ .inChannelCount = 112, .outChannelCount = 112 }),
      .upsample112 = createPipeline(GiGlslShaderGen::OidnParams{ .inChannelCount = 112, .outChannelCount = 112, .op = GiGlslShaderGen::OidnOp::Upsample }),
      .conv160_96 = createPipeline(GiGlslShaderGen::OidnParams{ .inChannelCount = 160, .outChannelCount = 96 }),
      .conv96_96 = createPipeline(GiGlslShaderGen::OidnParams{ .inChannelCount = 96, .outChannelCount = 96 }),
      .conv128_64 = createPipeline(GiGlslShaderGen::OidnParams{ .inChannelCount = 128, .outChannelCount = 64 }),
      .conv64_64 = createPipeline(GiGlslShaderGen::OidnParams{ .inChannelCount = 64, .outChannelCount = 64 }),
      .upsample64 =createPipeline(GiGlslShaderGen::OidnParams{ .inChannelCount = 64, .outChannelCount = 64, .op = GiGlslShaderGen::OidnOp::Upsample }),
      .conv73_64 = createPipeline(GiGlslShaderGen::OidnParams{ .inChannelCount = 73, .outChannelCount = 64 }),
      .conv64_32 = createPipeline(GiGlslShaderGen::OidnParams{ .inChannelCount = 64, .outChannelCount = 32 }),
      .conv32_3 = createPipeline(GiGlslShaderGen::OidnParams{ .inChannelCount = 32, .outChannelCount = 3 })
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
      weightBufferSize += (desc.dataSize + 4 - 1) / 4  * 4;
    }
    size_t biasBufferSize = 0;
    for (const char* name : TENSOR_NAMES)
    {
      auto newName = std::string(name) + ".bias";
      assert(tensorDescriptions.count(newName) > 0);
      const GiTzaTensorDescription& desc = tensorDescriptions.at(newName);
      biasBufferSize += (desc.dataSize + 4 - 1) / 4  * 4;
    }

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
      uint32_t dataOffset = (nextWeightOffset + 4 - 1) / 4 * 4;
      uint32_t dataSize = (desc.dataSize + 4 - 1) / 4 * 4;

      bool s = stager.stageToBuffer(&tensorData[desc.dataOffset], dataSize, weightBuffer, dataOffset);
      if (!s) GB_FATAL("failed to stage data");

      nextWeightOffset += desc.dataSize;
      return dataOffset;
    };

    uint32_t nextBiasOffset = 0;
    const auto uploadBiases = [&](const char* name) {
      assert(tensorDescriptions.count(name) > 0);
      const GiTzaTensorDescription& desc = tensorDescriptions.at(name);
      uint32_t dataOffset = (nextBiasOffset + 4 - 1) / 4 * 4;
      uint32_t dataSize = (desc.dataSize + 4 - 1) / 4 * 4;

      bool s = stager.stageToBuffer(&tensorData[desc.dataOffset], dataSize, biasBuffer, dataOffset);
      if (!s) GB_FATAL("failed to stage data");

      nextBiasOffset += desc.dataSize;
      return dataOffset;
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

    // TODO: destroy buffers with delayedResourceDestroyer

    size_t channelSize = imageWidth * imageHeight * sizeof(float);

    CgpuBufferUsageFlags pool0BufferUsage = CGPU_BUFFER_USAGE_FLAG_STORAGE_BUFFER |
                                            CGPU_BUFFER_USAGE_FLAG_TRANSFER_SRC |
                                            CGPU_BUFFER_USAGE_FLAG_TRANSFER_DST;
    if (!cgpuCreateBuffer(device, { .usage = pool0BufferUsage, .size = channelSize * 3 }, &state->pool0))
    {
      GB_FATAL("failed to allocate OIDN buffer");
    }

    // NOTE: at Full HD, a single of the buffers is ~1.2GB
    CgpuBufferUsageFlags poolNBufferUsage = CGPU_BUFFER_USAGE_FLAG_STORAGE_BUFFER | CGPU_BUFFER_USAGE_FLAG_TRANSFER_SRC;
    if (!cgpuCreateBuffer(device, { .usage = poolNBufferUsage, .size = channelSize * 32 }, &state->pool1) ||
        !cgpuCreateBuffer(device, { .usage = poolNBufferUsage, .size = channelSize * 48 }, &state->pool2) ||
        !cgpuCreateBuffer(device, { .usage = poolNBufferUsage, .size = channelSize * 64 }, &state->pool3))
    {
      GB_FATAL("failed to allocate OIDN buffer");
    }

    CgpuBufferUsageFlags pingPongBufferUsage = CGPU_BUFFER_USAGE_FLAG_STORAGE_BUFFER;
    if (!cgpuCreateBuffer(device, { .usage = pingPongBufferUsage, .size = channelSize * 160 }, &state->pingPongData[0]) ||
        !cgpuCreateBuffer(device, { .usage = pingPongBufferUsage, .size = channelSize * 160 }, &state->pingPongData[1]))
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

  // TODO: for starters, copy 3 channels of input AOV to color AOV (viz aux normal & albedo)
  void giOidnRender(GiOidnState* state, CgpuCommandBuffer commandBuffer, CgpuBuffer rgbResult)
  {
    // TODO: apparently the input W, H need to be aligned to 16 pixels !
   const auto& pipelines = state->pipelines;
   const auto& offsets = state->offsets;

   uint32_t dispatchSizeX = state->imageWidth;
   uint32_t dispatchSizeY = state->imageHeight;

   auto dispatchConvolution = [&](CgpuPipeline pipeline, uint32_t weightOffset, uint32_t biasOffset,
                                  CgpuBuffer inBuffer, CgpuBuffer outBuffer)
   {
     rp::PushConstants pushData = {
       .imageWidth = state->imageWidth,
       .imageHeight = state->imageHeight,
       .weightOffset = 0, // TODO: per kernel
       .biasOffset = 0 // TODO: per kernel
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

     uint32_t wgCountX = (dispatchSizeX + rp::WG_SIZE_X - 1) / rp::WG_SIZE_X;
     uint32_t wgCountY = (dispatchSizeY + rp::WG_SIZE_Y - 1) / rp::WG_SIZE_Y;
     cgpuCmdDispatch(commandBuffer, wgCountX, wgCountY, 1);
    };

    dispatchConvolution(pipelines.debug, 0, 0, state->pool0, rgbResult);
  }
}
