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

#include <offsetAllocator.hpp>

#define EXIT_FATAL(msg) \
  fprintf(stderr, "%s:%i: %s\n", __FILE__, int(__LINE__), msg); \
  exit(-1);

namespace gtl
{
  namespace rp = shader_interface::rp_oidn;

  struct GiOidnState
  {
    std::unique_ptr<OffsetAllocator::Allocator> offsetAllocator;

    CgpuPipeline basicPipeline; // tmp

    CgpuBuffer dataBuffer;
    size_t bufferSize = 0;
    uint32_t imageWidth = 0;
    uint32_t imageHeight = 0;

    uint32_t bufferOffsetInput = 0;

    GgpuDelayedResourceDestroyer& resourceDestroyer;
  };

  static CgpuPipeline giOidnCreatePipelines(CgpuDevice device, GiGlslShaderGen& shaderGen)
  {
    std::vector<uint8_t> spv;
    if (!shaderGen.generateDenoisingSpirv(spv))
    {
      EXIT_FATAL("failed to compile OIDN shader");
    }

    CgpuShader shader;
    if (!cgpuCreateShader(device, {
                            .size = spv.size(),
                            .source = spv.data(),
                            .stageFlags = CGPU_SHADER_STAGE_FLAG_COMPUTE
                          }, &shader))
    {
      EXIT_FATAL("failed to create OIDN shader");
    }

    CgpuPipeline pipeline;
    if (!cgpuCreateComputePipeline(device, { .shader = shader , .debugName = "OIDN_Basic" }, &pipeline))
    {
      EXIT_FATAL("failed to create OIDN pipeline");
    }

    cgpuDestroyShader(device, shader);
    return pipeline;
  }

  GiOidnState* giOidnCreateState(CgpuDevice device,
                                 GiGlslShaderGen& shaderGen,
                                 GgpuDelayedResourceDestroyer& resourceDestroyer)
  {
    CgpuPipeline basicPipeline = giOidnCreatePipelines(device, shaderGen);

    return new GiOidnState {
      .offsetAllocator = std::make_unique<OffsetAllocator::Allocator>(0),
      .basicPipeline = basicPipeline,
      .resourceDestroyer = resourceDestroyer
    };
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
    delete state;
  }

  bool giOidnUpdateState(GiOidnState* state, CgpuDevice device, uint32_t imageWidth, uint32_t imageHeight)
  {
    size_t requiredMemory = imageWidth * imageHeight * 9 * sizeof(float); // TODO: only covers input AOV

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

    if (!cgpuCreateBuffer(device, { .usage = CGPU_BUFFER_USAGE_FLAG_STORAGE_BUFFER |
                                             CGPU_BUFFER_USAGE_FLAG_TRANSFER_SRC |
                                             CGPU_BUFFER_USAGE_FLAG_TRANSFER_DST,
                                    .size = requiredMemory }, &state->dataBuffer))
    {
      EXIT_FATAL("failed to allocate OIDN buffer");
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
    rp::PushConstants pushData = {
      .imageWidth = state->imageWidth,
      .imageHeight = state->imageHeight
    };

    std::array<CgpuBufferBinding, 2> bufferBindings = {
      CgpuBufferBinding{ .binding = 0, .buffer = state->dataBuffer },
      CgpuBufferBinding{ .binding = 1, .buffer = rgbResult }
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
