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

#include <offsetAllocator.hpp>

namespace gtl
{
  namespace rp = shader_interface::rp_oidn;

  struct GiOidnState
  {
    std::unique_ptr<OffsetAllocator::Allocator> offsetAllocator;

    CgpuPipeline basicPipeline; // tmp
    CgpuBuffer tensorBuffer;

    CgpuBuffer dataBuffer; // image slices
    size_t bufferSize = 0;
    uint32_t imageWidth = 0;
    uint32_t imageHeight = 0;

    uint64_t tensorOffseEncConv0;
    uint64_t tensorOffseEncConv1;
    uint64_t tensorOffseEncConv2;
    uint64_t tensorOffseEncConv3;
    uint64_t tensorOffseEncConv4;
    uint64_t tensorOffseEncConv5a;
    uint64_t tensorOffseEncConv5b;
    uint64_t tensorOffseDecConv4a;
    uint64_t tensorOffseDecConv4b;
    uint64_t tensorOffseDecConv3a;
    uint64_t tensorOffseDecConv3b;
    uint64_t tensorOffseDecConv2a;
    uint64_t tensorOffseDecConv2b;
    uint64_t tensorOffseDecConv1a;
    uint64_t tensorOffseDecConv1b;
    uint64_t tensorOffseDecConv0;

    GgpuDelayedResourceDestroyer resourceDestroyer;
  };

  static CgpuPipeline giOidnCreatePipelines(CgpuDevice device, GiGlslShaderGen& shaderGen)
  {
    GiGlslShaderGen::OidnParams params = {
      .inChannelCount = 3,
      .outChannelCount = 4
    };

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

    CgpuPipeline pipeline;
    if (!cgpuCreateComputePipeline(device, { .shader = shader , .debugName = "OIDN_Basic" }, &pipeline))
    {
      GB_FATAL("failed to create OIDN pipeline");
    }

    cgpuDestroyShader(device, shader);
    return pipeline;
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

    size_t tensorBufferSize = 0;
    for (const char* name : TENSOR_NAMES)
    {
auto newName = std::string(name) + ".weight";
assert(tensorDescriptions.count(newName) > 0);
      const GiTzaTensorDescription& desc = tensorDescriptions.at(newName);
      tensorBufferSize += (desc.dataSize + 4 - 1) / 4  * 4;
    }

    CgpuBuffer tensorBuffer;
    CgpuBufferUsageFlags tensorBufferUsage = CGPU_BUFFER_USAGE_FLAG_STORAGE_BUFFER | CGPU_BUFFER_USAGE_FLAG_TRANSFER_DST;
    if (!cgpuCreateBuffer(device, { .usage = tensorBufferUsage, .size = tensorBufferSize }, &tensorBuffer))
    {
GB_LOG("tensor buffer size: {}", tensorBufferSize);
      GB_FATAL("failed to alloc tensor buffer");
    }

    uint64_t nextTensorOffset = 0;
    const auto uploadTensor = [&](const char* name) {
      const GiTzaTensorDescription& desc = tensorDescriptions.at(std::string(name) + ".weight");
      uint64_t tensorOffset = (nextTensorOffset + 4 - 1) / 4 * 4;
      uint64_t dataSize = (desc.dataSize + 4 - 1) / 4 * 4;

      bool s = stager.stageToBuffer(&tensorData[desc.dataOffset], dataSize, tensorBuffer, tensorOffset);
      if (!s) GB_FATAL("failed to stage OIDN tensor");

      nextTensorOffset += desc.dataSize;
      return tensorOffset;
    };

    CgpuPipeline basicPipeline = giOidnCreatePipelines(device, shaderGen);

    auto* state = new GiOidnState {
      .offsetAllocator = std::make_unique<OffsetAllocator::Allocator>(0),
      .basicPipeline = basicPipeline,
      .tensorBuffer = tensorBuffer,
      .resourceDestroyer = resourceDestroyer
    };

    state->tensorOffseEncConv0 = uploadTensor("enc_conv0");
    state->tensorOffseEncConv1 = uploadTensor("enc_conv1");
    state->tensorOffseEncConv2 = uploadTensor("enc_conv2");
    state->tensorOffseEncConv3 = uploadTensor("enc_conv3");
    state->tensorOffseEncConv4 = uploadTensor("enc_conv4");
    state->tensorOffseEncConv5a = uploadTensor("enc_conv5a");
    state->tensorOffseEncConv5b = uploadTensor("enc_conv5b");
    state->tensorOffseDecConv4a = uploadTensor("dec_conv4a");
    state->tensorOffseDecConv4b = uploadTensor("dec_conv4b");
    state->tensorOffseDecConv3a = uploadTensor("dec_conv3a");
    state->tensorOffseDecConv3b = uploadTensor("dec_conv3b");
    state->tensorOffseDecConv2a = uploadTensor("dec_conv2a");
    state->tensorOffseDecConv2b = uploadTensor("dec_conv2b");
    state->tensorOffseDecConv1a = uploadTensor("dec_conv1a");
    state->tensorOffseDecConv1b = uploadTensor("dec_conv1b");
    state->tensorOffseDecConv0 = uploadTensor("dec_conv0");

// TODO: temp
GB_LOG("tensorOffseEncConv0: {}", state->tensorOffseEncConv0);
GB_LOG("tensorOffseEncConv1: {}", state->tensorOffseEncConv1);
GB_LOG("tensorOffseEncConv2: {}", state->tensorOffseEncConv2);

    stager.flush(); // optional

    return state;
  }

  void giOidnDestroyState(GiOidnState* state)
  {
    if (state->dataBuffer.handle)
    {
      state->resourceDestroyer.enqueueDestruction(state->dataBuffer);
    }
    if (state->basicPipeline.handle)
    {
      state->resourceDestroyer.enqueueDestruction(state->basicPipeline);
    }
    if (state->tensorBuffer.handle)
    {
      state->resourceDestroyer.enqueueDestruction(state->tensorBuffer);
    }
    delete state;
  }

  bool giOidnUpdateState(GiOidnState* state, CgpuDevice device, uint32_t imageWidth, uint32_t imageHeight)
  {
    size_t requiredMemory = imageWidth * imageHeight * 3 * sizeof(float); // TODO: only covers input AOV

    if (state->bufferSize >= requiredMemory && state->imageWidth == imageWidth && state->imageHeight == imageHeight)
    {
      return true; // nothing to do
    }

    if (state->dataBuffer.handle)
    {
      state->resourceDestroyer.enqueueDestruction(state->dataBuffer);
    }

    if (state->dataBuffer.handle)
    {
      cgpuDestroyBuffer(device, state->dataBuffer);
    }

    CgpuBufferUsageFlags bufferUsage = CGPU_BUFFER_USAGE_FLAG_STORAGE_BUFFER |
                                       CGPU_BUFFER_USAGE_FLAG_TRANSFER_SRC |
                                       CGPU_BUFFER_USAGE_FLAG_TRANSFER_DST;
    if (!cgpuCreateBuffer(device, { .usage = bufferUsage, .size = requiredMemory }, &state->dataBuffer))
    {
      GB_FATAL("failed to allocate OIDN buffer");
      fprintf(stderr,"%zu MB\n", requiredMemory / (1024*1024));
      return false;
    }

    state->offsetAllocator = std::make_unique<OffsetAllocator::Allocator>(requiredMemory);
    state->bufferSize = requiredMemory;
    state->imageWidth = imageWidth;
    state->imageHeight = imageHeight;
    return true;
  }

  CgpuBuffer giOidnGetInputBuffer(GiOidnState* state)
  {
    return state->dataBuffer;
  }

  // TODO: for starters, copy 3 channels of input AOV to color AOV (viz aux normal & albedo)
  void giOidnRender(GiOidnState* state, CgpuCommandBuffer commandBuffer, CgpuBuffer rgbResult)
  {
    // TODO: apparently the input W, H need to be aligned to 16 pixels !

    rp::PushConstants pushData = {
      .imageWidth = state->imageWidth,
      .imageHeight = state->imageHeight,
      .weightsOffset = 0 // TODO: per kernel
    };

    std::array<CgpuBufferBinding, 3> bufferBindings = {
      CgpuBufferBinding{ .binding = 0, .buffer = state->dataBuffer },
      CgpuBufferBinding{ .binding = 1, .buffer = rgbResult },
      CgpuBufferBinding{ .binding = 2, .buffer = state->tensorBuffer }
    };

    CgpuBindings bindings0 = { .bufferCount = (uint32_t) bufferBindings.size(), .buffers = bufferBindings.data() };
    cgpuCmdUpdateBindings(commandBuffer, state->basicPipeline, 0/*descriptorSetIndex*/, &bindings0);

    cgpuCmdBindPipeline(commandBuffer, state->basicPipeline);

    cgpuCmdPushConstants(commandBuffer, state->basicPipeline, CGPU_SHADER_STAGE_FLAG_COMPUTE, sizeof(pushData), &pushData);

    uint32_t wgCountX = (state->imageWidth + rp::WG_SIZE_X - 1) / rp::WG_SIZE_X;
    uint32_t wgCountY = (state->imageHeight + rp::WG_SIZE_Y - 1) / rp::WG_SIZE_Y;
    cgpuCmdDispatch(commandBuffer, wgCountX, wgCountY, 1);
  }
}
